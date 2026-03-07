import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.models import build_model
from src.utils import ensure_dir, load_yaml


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "base.yaml"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["RESULT_DIR"] = BASE_DIR / "static" / "results"
ensure_dir(str(app.config["RESULT_DIR"]))

_model = None
_cfg = None
_device = None


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_infer_runtime():
    global _model, _cfg, _device
    if _model is not None:
        return _model, _cfg, _device

    _cfg = load_yaml(str(CONFIG_PATH))
    _device = torch.device("cuda" if torch.cuda.is_available() and _cfg.get("device", "cuda") == "cuda" else "cpu")

    ckpt_path = BASE_DIR / _cfg["infer"]["ckpt_path"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=_device)
    _model = build_model(
        _cfg["model"]["name"],
        _cfg["model"]["in_channels"],
        _cfg["model"]["out_channels"],
        _cfg["model"]["base_channels"],
    ).to(_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()

    return _model, _cfg, _device


def _run_segmentation(image_bgr: np.ndarray):
    model, cfg, device = _load_infer_runtime()

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    size = int(cfg["data"]["image_size"])
    threshold = float(cfg["infer"].get("threshold", 0.5))

    x = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float().squeeze().cpu().numpy().astype(np.uint8)

    mask = cv2.resize(pred * 255, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = image_bgr.copy()
    vessel_idx = mask > 127
    overlay[vessel_idx] = (0, 0, 255)
    overlay = cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)

    return mask, overlay


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="请先选择一张眼底图像。")

    if not _allowed_file(file.filename):
        return render_template("index.html", error="仅支持 png/jpg/jpeg/bmp/tif/tiff 格式。")

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return render_template("index.html", error="图像读取失败，请更换文件后重试。")

    try:
        mask, overlay = _run_segmentation(image_bgr)
    except FileNotFoundError as e:
        return render_template("index.html", error=str(e))
    except Exception as e:
        return render_template("index.html", error=f"推理失败: {e}")

    stem = secure_filename(Path(file.filename).stem) or "retina"
    tag = uuid.uuid4().hex[:8]
    raw_name = f"{stem}_{tag}_raw.png"
    mask_name = f"{stem}_{tag}_mask.png"
    overlay_name = f"{stem}_{tag}_overlay.png"

    raw_path = app.config["RESULT_DIR"] / raw_name
    mask_path = app.config["RESULT_DIR"] / mask_name
    overlay_path = app.config["RESULT_DIR"] / overlay_name

    cv2.imwrite(str(raw_path), image_bgr)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)

    return render_template(
        "index.html",
        raw_url=f"/static/results/{raw_name}",
        mask_url=f"/static/results/{mask_name}",
        overlay_url=f"/static/results/{overlay_name}",
        success="推理完成。",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
