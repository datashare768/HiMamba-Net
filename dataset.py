import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


CROP_CLASSES = {
    'background': 0,
    'maize': 1,
    'potato': 2,
    'rapeseed': 3,
}

NUM_CLASSES = len(CROP_CLASSES)


def normalize_point_cloud(xyz):
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    scale = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if scale > 0:
        xyz = xyz / scale
    return xyz


def random_scale_jitter(xyz, scale_range=(0.8, 1.2), jitter_sigma=0.01):
    scale = np.random.uniform(*scale_range)
    xyz = xyz * scale
    jitter = np.clip(np.random.randn(*xyz.shape) * jitter_sigma, -0.05, 0.05)
    xyz = xyz + jitter
    return xyz


def random_rotate_z(xyz):
    angle = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return xyz @ R.T


def random_flip(xyz):
    if np.random.rand() > 0.5:
        xyz[:, 0] = -xyz[:, 0]
    if np.random.rand() > 0.5:
        xyz[:, 1] = -xyz[:, 1]
    return xyz


def load_point_cloud(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.npy':
        data = np.load(path)
    elif ext == '.npz':
        loaded = np.load(path)
        data = loaded[loaded.files[0]]
    elif ext in ('.txt', '.csv'):
        data = np.loadtxt(path)
    elif ext == '.ply':
        try:
            from plyfile import PlyData
            plydata = PlyData.read(path)
            v = plydata['vertex']
            cols = [v['x'], v['y'], v['z']]
            if 'red' in v._property_lookup:
                cols += [v['red'], v['green'], v['blue']]
            if 'label' in v._property_lookup:
                cols.append(v['label'])
            if 'instance' in v._property_lookup:
                cols.append(v['instance'])
            data = np.stack(cols, axis=1).astype(np.float32)
        except ImportError:
            raise ImportError("plyfile package required to read .ply files")
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return data.astype(np.float32)


def compute_instance_offsets(xyz, instance_labels, semantic_labels):
    offsets = np.zeros_like(xyz)
    for iid in np.unique(instance_labels):
        if iid < 0:
            continue
        mask = instance_labels == iid
        sem = semantic_labels[mask]
        if (sem == 0).all():
            continue
        center = xyz[mask].mean(axis=0)
        offsets[mask] = center - xyz[mask]
    return offsets


def random_subsample(xyz, features, sem_labels, inst_labels, n_points):
    N = xyz.shape[0]
    if N >= n_points:
        idx = np.random.choice(N, n_points, replace=False)
    else:
        idx = np.random.choice(N, n_points, replace=True)
    return xyz[idx], features[idx] if features is not None else None, sem_labels[idx], inst_labels[idx]


class Crops3DDataset(Dataset):
    def __init__(
        self,
        root,
        split='train',
        n_points=20480,
        use_rgb=True,
        augment=True,
        hilbert_order=5,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.n_points = n_points
        self.use_rgb = use_rgb
        self.augment = augment and (split == 'train')
        self.hilbert_order = hilbert_order

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            split_dir = root

        patterns = ['*.npy', '*.npz', '*.txt', '*.ply']
        self.files = []
        for pat in patterns:
            self.files += sorted(glob.glob(os.path.join(split_dir, '**', pat), recursive=True))
            self.files += sorted(glob.glob(os.path.join(split_dir, pat)))
        self.files = sorted(list(set(self.files)))

        if len(self.files) == 0:
            raise RuntimeError(f"No data files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = load_point_cloud(path)

        xyz = data[:, :3]
        has_rgb = data.shape[1] >= 6
        has_sem = data.shape[1] >= 7
        has_inst = data.shape[1] >= 8

        if has_rgb and self.use_rgb:
            rgb = data[:, 3:6]
            rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
            features = rgb.astype(np.float32)
        else:
            features = None

        sem_labels = data[:, 6].astype(np.int64) if has_sem else np.zeros(len(xyz), dtype=np.int64)
        inst_labels = data[:, 7].astype(np.int64) if has_inst else np.full(len(xyz), -1, dtype=np.int64)

        xyz, features, sem_labels, inst_labels = random_subsample(
            xyz, features, sem_labels, inst_labels, self.n_points
        )

        xyz = normalize_point_cloud(xyz)

        if self.augment:
            xyz = random_rotate_z(xyz)
            xyz = random_flip(xyz)
            xyz = random_scale_jitter(xyz).astype(np.float32)

        gt_offsets = compute_instance_offsets(xyz, inst_labels, sem_labels)

        xyz_t = torch.from_numpy(xyz.astype(np.float32))
        sem_t = torch.from_numpy(sem_labels)
        inst_t = torch.from_numpy(inst_labels)
        off_t = torch.from_numpy(gt_offsets.astype(np.float32))

        if features is not None:
            feat_t = torch.from_numpy(features.astype(np.float32))
        else:
            feat_t = None

        return xyz_t, feat_t, sem_t, inst_t, off_t


def collate_fn(batch):
    xyz_list, feat_list, sem_list, inst_list, off_list = zip(*batch)
    xyz = torch.stack(xyz_list, dim=0)
    sem = torch.stack(sem_list, dim=0)
    inst = torch.stack(inst_list, dim=0)
    off = torch.stack(off_list, dim=0)
    if feat_list[0] is not None:
        feat = torch.stack(feat_list, dim=0)
    else:
        feat = None
    return xyz, feat, sem, inst, off
