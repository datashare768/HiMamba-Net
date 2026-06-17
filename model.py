import torch
import torch.nn as nn
import torch.nn.functional as F
from patch_encoder import InPatchEncoder
from graph_context import MultiScaleGraphContext
from samamba import SAMambaBlock
from hilbert import hilbert_sort_indices_torch, statistical_outlier_removal


class HiMambaNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        d_model=256,
        d_state=16,
        n_blocks=4,
        k=32,
        hilbert_order=5,
        n_levels=3,
        scale_ratios=(1.0, 0.5, 0.25),
        emb_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.hilbert_order = hilbert_order
        self.d_model = d_model

        feat_channels = max(0, in_channels - 3)
        self.patch_encoder = InPatchEncoder(
            feat_channels=feat_channels,
            embed_dim=d_model // 4,
            out_dim=d_model,
            k=k,
        )

        self.graph_context = MultiScaleGraphContext(
            in_dim=d_model,
            out_dim=d_model,
            k=k,
            n_levels=n_levels,
            scale_ratios=scale_ratios,
        )

        self.samamba_blocks = nn.ModuleList([
            SAMambaBlock(d_model=d_model, d_state=d_state, d_ffn=d_model * 4)
            for _ in range(n_blocks)
        ])

        self.dropout = nn.Dropout(dropout)

        self.sem_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.inst_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, emb_dim),
        )

        self.off_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 3),
        )

    @staticmethod
    def _run_head(head, flat, B, N):
        x = flat
        for layer in head:
            x = layer(x)
        return x.view(B, N, -1)

    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape

        local_feat = self.patch_encoder(xyz, features)
        context_feat = self.graph_context(xyz, local_feat)

        perm_list, inv_perm_list = [], []
        for b in range(B):
            perm = hilbert_sort_indices_torch(xyz[b], order=self.hilbert_order)
            perm_list.append(perm)
            inv_perm_list.append(torch.argsort(perm))

        feat_seq = torch.stack([context_feat[b][perm_list[b]] for b in range(B)], dim=0)

        for block in self.samamba_blocks:
            feat_seq = block(feat_seq)

        feat_out = torch.stack([feat_seq[b][inv_perm_list[b]] for b in range(B)], dim=0)
        feat_out = self.dropout(feat_out)

        flat = feat_out.view(B * N, self.d_model)

        sem_logits = self._run_head(self.sem_head, flat, B, N)
        inst_emb = F.normalize(self._run_head(self.inst_head, flat, B, N), p=2, dim=-1)
        offset = self._run_head(self.off_head, flat, B, N)

        return sem_logits, inst_emb, offset


def build_himamba_net(cfg):
    return HiMambaNet(
        in_channels=cfg.get('in_channels', 3),
        num_classes=cfg.get('num_classes', 4),
        d_model=cfg.get('d_model', 256),
        d_state=cfg.get('d_state', 16),
        n_blocks=cfg.get('n_blocks', 4),
        k=cfg.get('k', 32),
        hilbert_order=cfg.get('hilbert_order', 5),
        n_levels=cfg.get('n_levels', 3),
        scale_ratios=cfg.get('scale_ratios', (1.0, 0.5, 0.25)),
        emb_dim=cfg.get('emb_dim', 256),
        dropout=cfg.get('dropout', 0.1),
    )
