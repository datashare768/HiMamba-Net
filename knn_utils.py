import torch
import torch.nn.functional as F


def index_points(points, idx):
    B = points.shape[0]
    device = points.device
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, n_samples):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_query(query_xyz, key_xyz, k):
    dists = torch.cdist(query_xyz, key_xyz)
    knn_dists, knn_idx = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
    return knn_idx, knn_dists


def group_points_by_idx(features, idx):
    B, N, C = features.shape
    B, M, K = idx.shape
    features_flat = features.view(B * N, C)
    idx_flat = idx.view(B, M * K)
    batch_offset = torch.arange(B, device=features.device).unsqueeze(1) * N
    idx_flat = idx_flat + batch_offset
    grouped = features_flat[idx_flat.view(-1)].view(B, M, K, C)
    return grouped


def interpolate_features(xyz_src, xyz_dst, features_src, k=3):
    B, N_src, _ = xyz_src.shape
    B, N_dst, _ = xyz_dst.shape
    dists = torch.cdist(xyz_dst, xyz_src)
    dists_k, idx_k = torch.topk(dists, k, dim=-1, largest=False)
    dists_k = torch.clamp(dists_k, min=1e-10)
    weights = 1.0 / dists_k
    weights = weights / weights.sum(dim=-1, keepdim=True)
    grouped = index_points(features_src, idx_k)
    interpolated = (grouped * weights.unsqueeze(-1)).sum(dim=2)
    return interpolated
