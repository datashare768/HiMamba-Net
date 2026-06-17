import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer('weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weights = None

    def forward(self, pred, target):
        B, N, C = pred.shape
        pred_flat = pred.view(B * N, C)
        target_flat = target.view(B * N)
        return F.cross_entropy(pred_flat, target_flat, weight=self.weights, ignore_index=-1)


class DiscriminativeLoss(nn.Module):
    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, embeddings, instance_labels, sem_labels=None):
        B = embeddings.shape[0]
        total_loss = embeddings.new_zeros(1)
        for b in range(B):
            emb = embeddings[b]
            inst = instance_labels[b]
            if sem_labels is not None:
                fg_mask = sem_labels[b] > 0
                emb = emb[fg_mask]
                inst = inst[fg_mask]
            if emb.shape[0] == 0:
                continue
            unique_ids = torch.unique(inst)
            unique_ids = unique_ids[unique_ids >= 0]
            K = len(unique_ids)
            if K == 0:
                continue
            means = []
            pull_loss = emb.new_zeros(1)
            for uid in unique_ids:
                mask = inst == uid
                pts = emb[mask]
                mu = pts.mean(dim=0)
                means.append(mu)
                dist = torch.norm(pts - mu.unsqueeze(0), dim=1)
                hinge = F.relu(dist - self.delta_v)
                pull_loss = pull_loss + (hinge ** 2).mean()
            pull_loss = pull_loss / K
            means = torch.stack(means, dim=0)
            push_loss = emb.new_zeros(1)
            if K > 1:
                for i in range(K):
                    for j in range(K):
                        if i == j:
                            continue
                        dist = torch.norm(means[i] - means[j])
                        hinge = F.relu(self.delta_d - dist)
                        push_loss = push_loss + hinge ** 2
                push_loss = push_loss / (K * (K - 1))
            reg_loss = (means ** 2).sum(dim=1).mean()
            loss_b = self.alpha * pull_loss + self.beta * push_loss + self.gamma * reg_loss
            total_loss = total_loss + loss_b
        return total_loss / B


class OffsetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_offsets, gt_offsets, sem_labels):
        fg_mask = sem_labels > 0
        n_fg = fg_mask.sum()
        if n_fg == 0:
            return pred_offsets.new_zeros(1)
        pred = pred_offsets[fg_mask]
        gt = gt_offsets[fg_mask]
        return F.smooth_l1_loss(pred, gt, reduction='mean')


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        num_classes=4,
        class_weights=None,
        lambda_sem=1.0,
        lambda_inst=1.0,
        lambda_off=0.5,
        delta_v=0.5,
        delta_d=1.5,
    ):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.lambda_inst = lambda_inst
        self.lambda_off = lambda_off
        self.sem_loss = SemanticLoss(num_classes, class_weights)
        self.inst_loss = DiscriminativeLoss(delta_v=delta_v, delta_d=delta_d)
        self.off_loss = OffsetLoss()

    def forward(self, sem_logits, inst_emb, pred_offsets, sem_labels, inst_labels, gt_offsets):
        l_sem = self.sem_loss(sem_logits, sem_labels)
        l_inst = self.inst_loss(inst_emb, inst_labels, sem_labels)
        B, N, _ = pred_offsets.shape
        l_off_list = []
        for b in range(B):
            l_off_list.append(self.off_loss(pred_offsets[b], gt_offsets[b], sem_labels[b]))
        l_off = torch.stack(l_off_list).mean()
        total = self.lambda_sem * l_sem + self.lambda_inst * l_inst + self.lambda_off * l_off
        return total, {'sem': l_sem.item(), 'inst': l_inst.item(), 'off': l_off.item()}
