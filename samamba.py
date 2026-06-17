import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model implementing Eqs.(7-12) from the paper.

    Continuous-time SSM:
        dh/dt = A h(t) + B x(t)
        y(t)  = C h(t) + D x(t)

    ZOH discretization (simplified Mamba approximation):
        A_bar = exp(Delta * A)
        B_bar = Delta * B          (standard Mamba low-rank approx)
        h_n   = A_bar h_{n-1} + B_bar x_n
        y_n   = C h_n

    Selective mechanism (Eq.12):
        B_n = Linear_B(x_n),  C_n = Linear_C(x_n),  Delta_n = Softplus(Linear_Delta(x_n))
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        dt_rank = max(1, math.ceil(d_model / 16))
        self.dt_rank = dt_rank

        self.linear_delta_raw = nn.Linear(d_model, dt_rank, bias=False)
        self.linear_B = nn.Linear(d_model, d_state, bias=False)
        self.linear_C = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(d_model))
        self.D._no_weight_decay = True

        nn.init.uniform_(self.dt_proj.bias, -4, -1)

    def forward(self, x):
        B_n = self.linear_B(x)
        C_n = self.linear_C(x)
        delta = F.softplus(self.dt_proj(self.linear_delta_raw(x)))
        A = -torch.exp(self.A_log.float())
        y = self._selective_scan(x.float(), delta.float(), A, B_n.float(), C_n.float(), self.D.float())
        return y.to(x.dtype)

    def _selective_scan(self, u, delta, A, B, C, D):
        B_batch, L, d_model = u.shape
        N = A.shape[-1]
        dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        dBu = torch.einsum('bld,bln->bldn', delta, B) * u.unsqueeze(-1)
        h = torch.zeros(B_batch, d_model, N, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dBu[:, i]
            y_i = torch.einsum('bdn,bn->bd', h, C[:, i])
            ys.append(y_i)
        y = torch.stack(ys, dim=1)
        y = y + u * D.unsqueeze(0).unsqueeze(0)
        return y


class TwoDSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        y_fwd = self.ssm_fwd(x)
        y_bwd = self.ssm_bwd(x.flip(1)).flip(1)
        y_cat = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.out_proj(y_cat)
        gate = torch.sigmoid(self.gate_proj(y))
        return gate * x + (1 - gate) * y


class SAMambaBlock(nn.Module):
    """
    SAMamba block implementing Eqs.(11-14):
        F_conv = DWConv( LN(F_seq) )                              Eq.(11)
        F_ssm  = SSM(F_conv)
        F_attn = sigma( Linear(F_ssm) ) ⊙ F_conv                 Eq.(12)
        F_out  = F_seq + FFN( LN(F_attn) )                       Eq.(13)
        FFN(x) = Linear_2( GELU( Linear_1(x) ) )                 Eq.(14)
    """
    def __init__(self, d_model, d_state=16, d_ffn=None, dw_kernel_size=3):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_model * 4
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2, groups=d_model
        )
        self.ssm_2d = TwoDSSM(d_model, d_state)
        self.linear_gate = nn.Linear(d_model, d_model)
        self.ffn_linear1 = nn.Linear(d_model, d_ffn)
        self.ffn_linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        residual = x
        x_ln = self.ln1(x)
        F_conv = self.dw_conv(x_ln.permute(0, 2, 1)).permute(0, 2, 1)
        F_ssm = self.ssm_2d(F_conv)
        gate = torch.sigmoid(self.linear_gate(F_ssm))
        F_attn = gate * F_conv
        F_ffn = self.ffn_linear2(F.gelu(self.ffn_linear1(self.ln2(F_attn))))
        return residual + F_ffn
