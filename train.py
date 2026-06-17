import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import build_himamba_net
from loss import MultiTaskLoss
from dataset import Crops3DDataset, collate_fn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_miou(pred_labels, gt_labels, num_classes, ignore_index=-1):
    iou_list = []
    for c in range(num_classes):
        tp = ((pred_labels == c) & (gt_labels == c)).sum().item()
        fp = ((pred_labels == c) & (gt_labels != c) & (gt_labels != ignore_index)).sum().item()
        fn = ((pred_labels != c) & (gt_labels == c) & (gt_labels != ignore_index)).sum().item()
        if tp + fp + fn == 0:
            continue
        iou_list.append(tp / (tp + fp + fn + 1e-8))
    return sum(iou_list) / len(iou_list) if iou_list else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    loss_dict_accum = {'sem': 0.0, 'inst': 0.0, 'off': 0.0}
    n_batches = 0

    for batch in loader:
        xyz, feat, sem_labels, inst_labels, gt_offsets = batch
        xyz = xyz.to(device)
        sem_labels = sem_labels.to(device)
        inst_labels = inst_labels.to(device)
        gt_offsets = gt_offsets.to(device)
        if feat is not None:
            feat = feat.to(device)

        optimizer.zero_grad()
        sem_logits, inst_emb, pred_offsets = model(xyz, feat)
        loss, loss_dict = criterion(sem_logits, inst_emb, pred_offsets, sem_labels, inst_labels, gt_offsets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in loss_dict:
            loss_dict_accum[k] += loss_dict[k]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_dict = {k: v / max(n_batches, 1) for k, v in loss_dict_accum.items()}
    return avg_loss, avg_dict


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            xyz, feat, sem_labels, inst_labels, gt_offsets = batch
            xyz = xyz.to(device)
            sem_labels = sem_labels.to(device)
            inst_labels = inst_labels.to(device)
            gt_offsets = gt_offsets.to(device)
            if feat is not None:
                feat = feat.to(device)

            sem_logits, inst_emb, pred_offsets = model(xyz, feat)
            loss, _ = criterion(sem_logits, inst_emb, pred_offsets, sem_labels, inst_labels, gt_offsets)
            total_loss += loss.item()

            pred = sem_logits.argmax(dim=-1)
            all_pred.append(pred.cpu())
            all_gt.append(sem_labels.cpu())
            n_batches += 1

    all_pred = torch.cat([p.view(-1) for p in all_pred])
    all_gt = torch.cat([g.view(-1) for g in all_gt])
    miou = compute_miou(all_pred, all_gt, num_classes)
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, miou


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    in_channels = 3 if not args.use_rgb else 6
    cfg = {
        'in_channels': in_channels,
        'num_classes': args.num_classes,
        'd_model': args.d_model,
        'd_state': args.d_state,
        'n_blocks': args.n_blocks,
        'k': args.k,
        'hilbert_order': args.hilbert_order,
        'n_levels': args.n_levels,
        'emb_dim': args.emb_dim,
        'dropout': args.dropout,
    }
    model = build_himamba_net(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    train_dataset = Crops3DDataset(
        root=args.data_root,
        split='train',
        n_points=args.n_points,
        use_rgb=args.use_rgb,
        augment=True,
    )
    val_dataset = Crops3DDataset(
        root=args.data_root,
        split='val',
        n_points=args.n_points,
        use_rgb=args.use_rgb,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    class_weights = [1.0] * args.num_classes
    class_weights[0] = 0.5
    criterion = MultiTaskLoss(
        num_classes=args.num_classes,
        class_weights=class_weights,
        lambda_sem=args.lambda_sem,
        lambda_inst=args.lambda_inst,
        lambda_off=args.lambda_off,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_miou = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dict = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_miou = validate(model, val_loader, criterion, device, args.num_classes)
        scheduler.step(epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} "
            f"(sem={train_dict['sem']:.4f}, inst={train_dict['inst']:.4f}, off={train_dict['off']:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | "
            f"Time: {elapsed:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'cfg': cfg,
            }, save_path)
            print(f"  -> Best model saved (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if epoch % 50 == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch:03d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_miou': val_miou,
                'cfg': cfg,
            }, ckpt_path)

    print(f"Training complete. Best validation mIoU: {best_miou:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='HiMamba-Net Training')
    parser.add_argument('--data_root', type=str, required=True, help='Path to Crops3D dataset root')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_points', type=int, default=20480)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--hilbert_order', type=int, default=5)
    parser.add_argument('--n_levels', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_rgb', action='store_true', default=True)
    parser.add_argument('--lambda_sem', type=float, default=1.0)
    parser.add_argument('--lambda_inst', type=float, default=1.0)
    parser.add_argument('--lambda_off', type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
