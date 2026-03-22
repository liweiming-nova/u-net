import os
import random
from typing import Dict

import cv2
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def binarize_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    return float(((2 * inter + eps) / (denom + eps)).mean().item())


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return float(((inter + eps) / (union + eps)).mean().item())


def accuracy_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred == target).float().mean().item())


def sensitivity_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    tp = (pred * target).sum()
    fn = ((1.0 - pred) * target).sum()
    return float(((tp + eps) / (tp + fn + eps)).item())


def specificity_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    tn = ((1.0 - pred) * (1.0 - target)).sum()
    fp = (pred * (1.0 - target)).sum()
    return float(((tn + eps) / (tn + fp + eps)).item())


def auc_score_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    probs = torch.sigmoid(logits).detach().reshape(-1).cpu().numpy()
    labels = target.detach().reshape(-1).cpu().numpy().astype(np.int64)

    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1, dtype=np.float64)
    pos_rank_sum = ranks[labels == 1].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred = binarize_logits(logits, threshold=threshold)
    return {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "acc": accuracy_score(pred, target),
        "sen": sensitivity_score(pred, target),
        "spe": specificity_score(pred, target),
        "auc": auc_score_from_logits(logits, target),
    }


def save_mask(mask: torch.Tensor, path: str) -> None:
    m = mask.squeeze().detach().cpu().numpy()
    m = (m * 255).astype(np.uint8)
    cv2.imwrite(path, m)
