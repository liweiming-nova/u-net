import argparse
import json
import os
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import RetinaVesselDataset, build_train_augment
from src.losses import CombinedLoss
from src.models import build_model
from src.utils import ensure_dir, segmentation_metrics, set_seed


cfg_train_grad_clip = None


def _build_device(cfg: Dict) -> torch.device:
    want = cfg.get("device", "cuda")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_one_epoch(model, loader, criterion, optimizer, device, amp, scaler, threshold=0.5):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    totals = {"dice": 0.0, "iou": 0.0, "acc": 0.0, "sen": 0.0, "spe": 0.0, "auc": 0.0}

    iterator = tqdm(loader, leave=False)
    for images, masks, _ in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                scaler.scale(loss).backward()
                if cfg_train_grad_clip is not None and cfg_train_grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train_grad_clip)
                scaler.step(optimizer)
                scaler.update()

        metrics = segmentation_metrics(logits, masks, threshold=threshold)
        total_loss += float(loss.item())
        for key, value in metrics.items():
            totals[key] += value

    n = len(loader)
    result = {"loss": total_loss / max(n, 1)}
    for key, value in totals.items():
        result[key] = value / max(n, 1)
    return result


def train(cfg: Dict) -> None:
    global cfg_train_grad_clip

    set_seed(cfg["seed"])
    device = _build_device(cfg)
    ensure_dir(cfg["train"]["save_dir"])

    data_cfg = cfg["data"]
    preprocess_cfg = data_cfg.get("preprocess", {})
    train_aug = build_train_augment(data_cfg.get("augment"))

    model = build_model(
        cfg["model"]["name"],
        cfg["model"]["in_channels"],
        cfg["model"]["out_channels"],
        cfg["model"]["base_channels"],
    ).to(device)

    dataset_kwargs = {
        "image_size": data_cfg["image_size"],
        "use_grayscale": bool(preprocess_cfg.get("use_grayscale", False)),
        "apply_clahe": bool(preprocess_cfg.get("apply_clahe", False)),
        "clahe_clip_limit": float(preprocess_cfg.get("clahe_clip_limit", 2.0)),
        "clahe_tile_grid_size": int(preprocess_cfg.get("clahe_tile_grid_size", 8)),
    }

    train_ds = RetinaVesselDataset(
        data_cfg["train_image_dir"],
        data_cfg["train_mask_dir"],
        augment=train_aug,
        **dataset_kwargs,
    )
    val_ds = RetinaVesselDataset(
        data_cfg["val_image_dir"],
        data_cfg["val_mask_dir"],
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    criterion = CombinedLoss(
        dice_weight=cfg["loss"]["dice_weight"],
        focal_weight=cfg["loss"]["focal_weight"],
        focal_gamma=cfg["loss"]["focal_gamma"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"]) if cfg["train"]["scheduler"] == "cosine" else None

    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp)
    cfg_train_grad_clip = cfg["train"].get("grad_clip", None)
    threshold = float(cfg.get("infer", {}).get("threshold", 0.5))

    best_dice = -1.0
    logs = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr = _run_one_epoch(model, train_loader, criterion, optimizer, device, amp, scaler, threshold=threshold)
        va = _run_one_epoch(model, val_loader, criterion, None, device, amp, scaler, threshold=threshold)

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_dice": tr["dice"],
            "train_iou": tr["iou"],
            "train_acc": tr["acc"],
            "train_sen": tr["sen"],
            "train_spe": tr["spe"],
            "train_auc": tr["auc"],
            "val_loss": va["loss"],
            "val_dice": va["dice"],
            "val_iou": va["iou"],
            "val_acc": va["acc"],
            "val_sen": va["sen"],
            "val_spe": va["spe"],
            "val_auc": va["auc"],
        }
        logs.append(row)
        print(row)

        last_path = os.path.join(cfg["train"]["save_dir"], "last.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, last_path)

        if va["dice"] > best_dice:
            best_dice = va["dice"]
            best_path = os.path.join(cfg["train"]["save_dir"], "best.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, best_path)

    with open(os.path.join(cfg["train"]["save_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()
