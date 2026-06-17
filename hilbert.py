import numpy as np
import torch


def _hilbert_encode_3d(X, p):
    n = 3
    X = list(X)
    M = 1 << (p - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            if X[i] & Q:
                X[0] ^= P
            else:
                t = (X[0] ^ X[i]) & P
                X[0] ^= t
                X[i] ^= t
        Q >>= 1
    for i in range(1, n):
        X[i] ^= X[i - 1]
    t = 0
    Q = M
    while Q > 1:
        if X[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        X[i] ^= t
    h = 0
    for i in range(p):
        for j in range(n):
            h = (h << 1) | ((X[n - 1 - j] >> (p - 1 - i)) & 1)
    return h


def _hilbert_encode_3d_batch(ix, iy, iz, order):
    N = len(ix)
    n = 3
    X = np.stack([ix.copy(), iy.copy(), iz.copy()], axis=1).astype(np.int64)
    M = 1 << (order - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            mask = (X[:, i] & Q) != 0
            X[mask, 0] ^= P
            nm = ~mask
            t = (X[nm, 0] ^ X[nm, i]) & P
            X[nm, 0] ^= t
            X[nm, i] ^= t
        Q >>= 1
    for i in range(1, n):
        X[:, i] ^= X[:, i - 1]
    t = np.zeros(N, dtype=np.int64)
    Q = M
    while Q > 1:
        mask = (X[:, n - 1] & Q) != 0
        t[mask] ^= Q - 1
        Q >>= 1
    for i in range(n):
        X[:, i] ^= t
    h = np.zeros(N, dtype=np.int64)
    for i in range(order):
        for j in range(n):
            bit = (X[:, n - 1 - j] >> (order - 1 - i)) & 1
            h |= bit.astype(np.int64) << (i * n + j)
    return h


def hilbert_sort_indices(coords, order=5):
    lo = np.percentile(coords, 1, axis=0)
    hi = np.percentile(coords, 99, axis=0)
    rng = np.maximum(hi - lo, 1e-8)
    norm = np.clip((coords - lo) / rng, 0.0, 1.0)
    max_coord = (1 << order) - 1
    ix = np.clip((norm[:, 0] * max_coord).astype(np.int64), 0, max_coord)
    iy = np.clip((norm[:, 1] * max_coord).astype(np.int64), 0, max_coord)
    iz = np.clip((norm[:, 2] * max_coord).astype(np.int64), 0, max_coord)
    h = _hilbert_encode_3d_batch(ix, iy, iz, order)
    return np.argsort(h, kind='stable')


def hilbert_sort_indices_torch(coords, order=5):
    coords_np = coords.detach().cpu().numpy().astype(np.float64)
    perm = hilbert_sort_indices(coords_np, order=order)
    return torch.from_numpy(perm).long().to(coords.device)


def statistical_outlier_removal(coords, k=20, std_ratio=2.0):
    N = coords.shape[0]
    if N <= k + 1:
        return torch.ones(N, dtype=torch.bool, device=coords.device)
    chunk = min(N, 4096)
    mean_dists = []
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sub = coords[start:end]
        d = torch.cdist(sub, coords)
        topk, _ = torch.topk(d, k + 1, dim=1, largest=False)
        mean_dists.append(topk[:, 1:].mean(dim=1))
    mean_dists = torch.cat(mean_dists, dim=0)
    mu = mean_dists.mean()
    sigma = mean_dists.std()
    return mean_dists <= (mu + std_ratio * sigma)
