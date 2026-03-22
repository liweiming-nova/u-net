import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import RetinaVesselDataset
from src.models import build_model
from src.utils import ensure_dir, load_yaml, save_mask


def predict(config_path: str) -> None:
    cfg = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu")

    ckpt = torch.load(cfg["infer"]["ckpt_path"], map_location=device)

    model = build_model(
        cfg["model"]["name"],
        cfg["model"]["in_channels"],
        cfg["model"]["out_channels"],
        cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preprocess_cfg = cfg["data"].get("preprocess", {})
    ds = RetinaVesselDataset(
        cfg["data"]["test_image_dir"],
        mask_dir=None,
        image_size=cfg["data"]["image_size"],
        use_grayscale=bool(preprocess_cfg.get("use_grayscale", False)),
        apply_clahe=bool(preprocess_cfg.get("apply_clahe", False)),
        clahe_clip_limit=float(preprocess_cfg.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=int(preprocess_cfg.get("clahe_tile_grid_size", 8)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    out_dir = cfg["infer"]["output_dir"]
    ensure_dir(out_dir)
    threshold = cfg["infer"].get("threshold", 0.5)

    with torch.no_grad():
        for images, _, names in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            pred = (probs > threshold).float()
            out_path = os.path.join(out_dir, names[0])
            save_mask(pred[0], out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    predict(args.config)
