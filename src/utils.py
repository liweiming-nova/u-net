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


def save_mask(mask: torch.Tensor, path: str) -> None:
    m = mask.squeeze().detach().cpu().numpy()
    m = (m * 255).astype(np.uint8)
    cv2.imwrite(path, m)
