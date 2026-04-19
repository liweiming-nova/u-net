import argparse
import ast
import json
import os
import time
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import RetinaVesselDataset, build_train_augment
from src.losses import CombinedLoss
from src.models import build_model
from src.plot_metrics import plot_training_curves
from src.utils import ensure_dir, segmentation_metrics, set_seed


cfg_train_grad_clip = None


def _save_checkpoint(payload: Dict, path: str, label: str) -> str:
    directory = os.path.dirname(path)
    ensure_dir(directory)

    last_error = None
    for attempt in range(3):
        try:
            torch.save(payload, path)
            return path
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))

    root, ext = os.path.splitext(path)
    fallback_path = f"{root}_{label}{ext}"
    try:
        torch.save(payload, fallback_path)
        print({"warning": f"Failed to save {path}, fallback saved to {fallback_path}", "error": str(last_error)})
        return fallback_path
    except RuntimeError:
        raise last_error


def _build_device(cfg: Dict) -> torch.device:
    want = cfg.get("device", "cuda")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    cfg: Dict,
    epoch: int,
    best_dice: float,
    logs,
) -> Dict:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "epoch": epoch,
        "best_dice": best_dice,
        "logs": logs,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def _load_metrics_history(metrics_path: str):
    if not os.path.exists(metrics_path):
        return _load_metrics_history_from_log(os.path.join(os.path.dirname(metrics_path), "train.log"))
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
        if isinstance(logs, list):
            return logs
    except (OSError, json.JSONDecodeError):
        return _load_metrics_history_from_log(os.path.join(os.path.dirname(metrics_path), "train.log"))
    return []


def _load_metrics_history_from_log(log_path: str):
    if not os.path.exists(log_path):
        return []
    try:
        text = open(log_path, "r", encoding="utf-8", errors="ignore").read().replace("\x00", "")
    except OSError:
        return []

    rows_by_epoch = {}
    for line in text.splitlines():
        line = line.strip()
        if "'epoch':" not in line:
            continue
        start = line.find("{")
        end = line.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = line[start:end + 1]
        try:
            row = ast.literal_eval(snippet)
        except (ValueError, SyntaxError):
            continue
        if isinstance(row, dict) and "epoch" in row:
            rows_by_epoch[int(row["epoch"])] = row
    return [rows_by_epoch[k] for k in sorted(rows_by_epoch)]


def _resume_training_if_possible(
    model,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    cfg: Dict,
    resume_path: Optional[str],
) -> Dict:
    save_dir = cfg["train"]["save_dir"]
    candidate = resume_path or os.path.join(save_dir, "last.pt")
    if not candidate or not os.path.exists(candidate):
        return {"start_epoch": 1, "best_dice": -1.0, "logs": []}

    checkpoint = torch.load(candidate, map_location=device)
    model.load_state_dict(checkpoint["model"])

    optimizer_loaded = False
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_loaded = True

    scheduler_loaded = False
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
        scheduler_loaded = True

    scaler_loaded = False
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
        scaler_loaded = True

    logs = checkpoint.get("logs")
    if not isinstance(logs, list):
        logs = _load_metrics_history(os.path.join(save_dir, "metrics.json"))

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_dice = float(
        checkpoint.get(
            "best_dice",
            max((float(row.get("val_dice", -1.0)) for row in logs), default=-1.0),
        )
    )

    if scheduler is not None and not scheduler_loaded:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        scheduler.last_epoch = start_epoch - 2
        scheduler._step_count = max(start_epoch - 1, 0)

    print(
        {
            "resume": True,
            "resume_path": candidate,
            "start_epoch": start_epoch,
            "optimizer_restored": optimizer_loaded,
            "scheduler_restored": scheduler_loaded,
            "scaler_restored": scaler_loaded,
            "history_rows": len(logs),
        }
    )
    return {"start_epoch": start_epoch, "best_dice": best_dice, "logs": logs}


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


def train(cfg: Dict, resume_path: Optional[str] = None, resume: bool = True) -> None:
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
    patch_cfg = data_cfg.get("patch", {})
    train_patch_cfg = patch_cfg if patch_cfg.get("enabled", False) else None
    val_patch_cfg = patch_cfg if patch_cfg.get("enabled", False) and patch_cfg.get("apply_to_val", False) else None

    train_ds = RetinaVesselDataset(
        data_cfg["train_image_dir"],
        data_cfg["train_mask_dir"],
        augment=train_aug,
        patch_cfg=train_patch_cfg,
        **dataset_kwargs,
    )
    val_ds = RetinaVesselDataset(
        data_cfg["val_image_dir"],
        data_cfg["val_mask_dir"],
        patch_cfg=val_patch_cfg,
        **dataset_kwargs,
    )

    num_workers = int(data_cfg.get("num_workers", 0))
    if os.name == "nt" and num_workers > 0:
        print({"info": f"Windows detected, fallback num_workers from {num_workers} to 0 for stable training."})
        num_workers = 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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

    start_epoch = 1
    best_dice = -1.0
    logs = []
    if resume:
        state = _resume_training_if_possible(model, optimizer, scheduler, scaler, device, cfg, resume_path)
        start_epoch = state["start_epoch"]
        best_dice = state["best_dice"]
        logs = state["logs"]

    if start_epoch > cfg["train"]["epochs"]:
        print(
            {
                "resume": True,
                "message": f"Checkpoint epoch already reached configured epochs={cfg['train']['epochs']}.",
            }
        )
        metrics_path = os.path.join(cfg["train"]["save_dir"], "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        curve_paths = plot_training_curves(logs, cfg["train"]["save_dir"])
        print({"saved_metrics": metrics_path, **curve_paths})
        return

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
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
        checkpoint_payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
            best_dice=best_dice,
            logs=logs,
        )
        _save_checkpoint(checkpoint_payload, last_path, f"epoch_{epoch}")

        if va["dice"] > best_dice:
            best_dice = va["dice"]
            best_path = os.path.join(cfg["train"]["save_dir"], "best.pt")
            checkpoint_payload["best_dice"] = best_dice
            _save_checkpoint(checkpoint_payload, best_path, f"best_epoch_{epoch}")

    with open(os.path.join(cfg["train"]["save_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    curve_paths = plot_training_curves(logs, cfg["train"]["save_dir"])
    print({"saved_metrics": os.path.join(cfg["train"]["save_dir"], "metrics.json"), **curve_paths})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()
