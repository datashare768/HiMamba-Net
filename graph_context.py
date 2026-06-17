import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_utils import (
    farthest_point_sample, knn_query,
    index_points, interpolate_features
)


class GraphConvLayer(nn.Module):
    """
    Graph convolution: f_i^(l) = sigma( W^(l) * mean_{j in N(i)} f_j^(l-1) )
    Eq.(6) in the paper.
    """
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, xyz, features):
        B, N, C = features.shape
        k = self.k
        knn_idx, _ = knn_query(xyz, xyz, k)
        grouped = index_points(features, knn_idx)
        agg = grouped.mean(dim=2)
        flat = agg.view(B * N, C)
        out = F.relu(self.bn(self.linear(flat)))
        return out.view(B, N, -1)


class MultiScaleGraphContext(nn.Module):
    """
    Hierarchical multi-scale graph context:
      P^(0) = original point cloud
      P^(l) = FPS( P^(l-1), N_l )   [hierarchical, not from original each time]
    Features at each scale are graph-convolved, then interpolated back to N
    and concatenated for final projection.
    """
    def __init__(self, in_dim, out_dim, k=32, n_levels=3, scale_ratios=(1.0, 0.5, 0.25)):
        super().__init__()
        self.n_levels = n_levels
        self.scale_ratios = scale_ratios
        mid_dim = out_dim // n_levels

        self.level_convs = nn.ModuleList()
        for i in range(n_levels):
            d_in = in_dim if i == 0 else mid_dim
            self.level_convs.append(GraphConvLayer(d_in, mid_dim, k))

        self.proj = nn.Sequential(
            nn.Linear(mid_dim * n_levels, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, features):
        B, N, _ = xyz.shape

        level_xyz = [xyz]
        level_feat_out = []

        current_feat = features
        for i, conv in enumerate(self.level_convs):
            cur_xyz = level_xyz[i]
            feat_out = conv(cur_xyz, current_feat)
            level_feat_out.append((cur_xyz, feat_out))

            if i < self.n_levels - 1:
                n_sub = max(int(level_xyz[0].shape[1] * self.scale_ratios[i + 1]), 32)
                fps_idx = farthest_point_sample(cur_xyz, n_sub)
                sub_xyz = index_points(cur_xyz, fps_idx)
                sub_feat = index_points(feat_out, fps_idx)
                level_xyz.append(sub_xyz)
                current_feat = sub_feat
            else:
                current_feat = feat_out

        interp_feats = []
        for i, (lxyz, lfeat) in enumerate(level_feat_out):
            if lxyz.shape[1] == N:
                interp_feats.append(lfeat)
            else:
                interp_feats.append(interpolate_features(lxyz, xyz, lfeat, k=3))

        multi_feat = torch.cat(interp_feats, dim=-1)
        B2, N2, C2 = multi_feat.shape
        flat = multi_feat.view(B2 * N2, C2)
        out = self.proj[0](flat)
        out = self.proj[1](out)
        out = self.proj[2](out)
        return out.view(B2, N2, -1)
