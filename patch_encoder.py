import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_utils import knn_query, index_points


class SharedMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        B, N, K, C = x.shape
        x = x.view(B * N * K, C)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x.view(B, N, K, -1)


class InPatchEncoder(nn.Module):
    def __init__(self, feat_channels=0, embed_dim=64, out_dim=256, k=32):
        super().__init__()
        self.k = k
        self.feat_channels = feat_channels
        input_dim = 3 + feat_channels
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.mlp = SharedMLP(embed_dim, embed_dim * 2, out_dim)

    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape
        k = self.k

        knn_idx, _ = knn_query(xyz, xyz, k)
        grouped_xyz = index_points(xyz, knn_idx)
        center_xyz = xyz.unsqueeze(2).expand_as(grouped_xyz)
        rel_xyz = grouped_xyz - center_xyz

        if features is not None and self.feat_channels > 0:
            grouped_feat = index_points(features, knn_idx)
            patch_input = torch.cat([rel_xyz, grouped_feat], dim=-1)
        else:
            patch_input = rel_xyz

        B2, N2, K2, C2 = patch_input.shape
        flat = patch_input.view(B2 * N2 * K2, C2)
        embedded = self.embed[0](flat)
        embedded = self.embed[1](embedded)
        embedded = self.embed[2](embedded)
        embedded = embedded.view(B2, N2, K2, -1)

        processed = self.mlp(embedded)
        local_feat = processed.max(dim=2)[0]
        return local_feat
