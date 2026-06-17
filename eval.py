import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import build_himamba_net
from dataset import Crops3DDataset, collate_fn

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    from sklearn.cluster import DBSCAN


def compute_iou_per_class(pred, gt, num_classes, ignore_index=-1):
    ious = {}
    for c in range(num_classes):
        tp = ((pred == c) & (gt == c)).sum().item()
        fp = ((pred == c) & (gt != c) & (gt != ignore_index)).sum().item()
        fn = ((pred != c) & (gt == c)).sum().item()
        denom = tp + fp + fn
        if denom == 0:
            continue
        ious[c] = tp / (denom + 1e-8)
    return ious


def instance_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum().item()
    union = (pred_mask | gt_mask).sum().item()
    if union == 0:
        return 0.0
    return intersection / union


def compute_ap_at_iou(pred_instances, gt_instances, iou_threshold=0.5):
    if len(gt_instances) == 0:
        return 0.0
    if len(pred_instances) == 0:
        return 0.0

    matched_gt = set()
    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for pred_mask in pred_instances:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt_mask in enumerate(gt_instances):
            if gi in matched_gt:
                continue
            iou = instance_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

        precisions.append(tp / (tp + fp + 1e-8))
        recalls.append(tp / (len(gt_instances) + 1e-8))

    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def cluster_embeddings(embeddings, xyz, offsets, min_cluster_size=50, min_samples=10, eps=0.5):
    shifted_xyz = xyz + offsets
    combined = np.concatenate([shifted_xyz, embeddings], axis=1)

    if HAS_HDBSCAN:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=eps,
            metric='euclidean',
        )
        labels = clusterer.fit_predict(combined)
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clusterer.fit_predict(combined)

    return labels


def evaluate_sample(sem_logits, inst_emb, pred_offsets, xyz, gt_sem, gt_inst, num_classes, args):
    pred_sem = sem_logits.argmax(dim=-1).cpu().numpy()
    gt_sem_np = gt_sem.cpu().numpy()
    gt_inst_np = gt_inst.cpu().numpy()

    ious = compute_iou_per_class(
        torch.from_numpy(pred_sem),
        gt_sem,
        num_classes
    )

    fg_mask = pred_sem > 0
    n_fg = fg_mask.sum()

    if n_fg < args.min_cluster_size:
        pred_instances = []
    else:
        emb_np = inst_emb.cpu().numpy()[fg_mask]
        off_np = pred_offsets.cpu().numpy()[fg_mask]
        xyz_np = xyz.cpu().numpy()[fg_mask]
        cluster_labels = cluster_embeddings(
            emb_np, xyz_np, off_np,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            eps=args.eps,
        )
        N_full = sem_logits.shape[0]
        full_cluster_labels = np.full(N_full, -1, dtype=np.int64)
        full_cluster_labels[fg_mask] = cluster_labels

        unique_ids = np.unique(cluster_labels)
        unique_ids = unique_ids[unique_ids >= 0]
        pred_instances = []
        for uid in unique_ids:
            mask = torch.zeros(N_full, dtype=torch.bool)
            mask[full_cluster_labels == uid] = True
            pred_instances.append(mask)

    unique_gt_ids = np.unique(gt_inst_np)
    unique_gt_ids = unique_gt_ids[unique_gt_ids >= 0]
    gt_instances = []
    for uid in unique_gt_ids:
        gt_sem_in_inst = gt_sem_np[gt_inst_np == uid]
        if (gt_sem_in_inst == 0).all():
            continue
        mask = torch.zeros(len(gt_inst_np), dtype=torch.bool)
        mask[gt_inst_np == uid] = True
        gt_instances.append(mask)

    ap_25 = compute_ap_at_iou(pred_instances, gt_instances, iou_threshold=0.25)
    ap_50 = compute_ap_at_iou(pred_instances, gt_instances, iou_threshold=0.50)
    ap_75 = compute_ap_at_iou(pred_instances, gt_instances, iou_threshold=0.75)

    return ious, ap_25, ap_50, ap_75


def evaluate(model, loader, device, num_classes, args):
    model.eval()
    all_ious = {c: [] for c in range(num_classes)}
    all_ap_25, all_ap_50, all_ap_75 = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            xyz, feat, sem_labels, inst_labels, gt_offsets = batch
            B = xyz.shape[0]
            xyz = xyz.to(device)
            sem_labels = sem_labels.to(device)
            inst_labels = inst_labels.to(device)
            if feat is not None:
                feat = feat.to(device)

            sem_logits, inst_emb, pred_offsets = model(xyz, feat)

            for b in range(B):
                ious, ap25, ap50, ap75 = evaluate_sample(
                    sem_logits[b], inst_emb[b], pred_offsets[b],
                    xyz[b], sem_labels[b], inst_labels[b],
                    num_classes, args
                )
                for c, v in ious.items():
                    all_ious[c].append(v)
                all_ap_25.append(ap25)
                all_ap_50.append(ap50)
                all_ap_75.append(ap75)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Evaluated {batch_idx + 1}/{len(loader)} batches")

    class_ious = {}
    for c in range(num_classes):
        if all_ious[c]:
            class_ious[c] = np.mean(all_ious[c])
    miou = np.mean(list(class_ious.values())) if class_ious else 0.0
    map_25 = np.mean(all_ap_25) if all_ap_25 else 0.0
    map_50 = np.mean(all_ap_50) if all_ap_50 else 0.0
    map_75 = np.mean(all_ap_75) if all_ap_75 else 0.0
    mean_ap = (map_25 + map_50 + map_75) / 3.0

    return miou, class_ious, map_25, map_50, map_75, mean_ap


CLASS_NAMES = {0: 'background', 1: 'maize', 2: 'potato', 3: 'rapeseed'}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get('cfg', {})
    cfg.setdefault('num_classes', args.num_classes)

    model = build_himamba_net(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    test_dataset = Crops3DDataset(
        root=args.data_root,
        split=args.split,
        n_points=args.n_points,
        use_rgb=cfg.get('in_channels', 3) > 3,
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Evaluating on {len(test_dataset)} samples from '{args.split}' split...")
    miou, class_ious, ap25, ap50, ap75, mean_ap = evaluate(
        model, test_loader, device, cfg['num_classes'], args
    )

    print("\n===== Evaluation Results =====")
    print(f"Semantic Segmentation:")
    print(f"  mIoU: {miou:.4f}")
    for c, iou in sorted(class_ious.items()):
        name = CLASS_NAMES.get(c, f'class_{c}')
        print(f"  IoU ({name}): {iou:.4f}")
    print(f"Instance Segmentation:")
    print(f"  AP@0.25: {ap25:.4f}")
    print(f"  AP@0.50: {ap50:.4f}")
    print(f"  AP@0.75: {ap75:.4f}")
    print(f"  mAP:     {mean_ap:.4f}")
    print("===============================")

    if args.output:
        import json
        results = {
            'semantic': {'mIoU': miou, 'class_iou': {str(k): v for k, v in class_ious.items()}},
            'instance': {'AP@0.25': ap25, 'AP@0.50': ap50, 'AP@0.75': ap75, 'mAP': mean_ap},
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description='HiMamba-Net Evaluation')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n_points', type=int, default=20480)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--min_cluster_size', type=int, default=50)
    parser.add_argument('--min_samples', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
