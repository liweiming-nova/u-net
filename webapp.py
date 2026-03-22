import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.datasets import preprocess_image
from src.history_manager import HistoryManager, build_history_record
from src.models import build_model
from src.utils import ensure_dir, load_yaml


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "base.yaml"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
HISTORY_ROOT = BASE_DIR / "local_history"
HISTORY_FILE = HISTORY_ROOT / "history.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
app.config["RESULT_DIR"] = BASE_DIR / "static" / "results"
ensure_dir(str(app.config["RESULT_DIR"]))
ensure_dir(str(HISTORY_ROOT))

history_manager = HistoryManager(HISTORY_FILE)

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


def _result_names(source_name: str):
    stem = secure_filename(Path(source_name).stem) or "retina"
    tag = uuid.uuid4().hex[:8]
    return (
        f"{stem}_{tag}_raw.png",
        f"{stem}_{tag}_mask.png",
        f"{stem}_{tag}_overlay.png",
    )


def _save_result(image_bgr: np.ndarray, mask: np.ndarray, overlay: np.ndarray, source_name: str):
    raw_name, mask_name, overlay_name = _result_names(source_name)
    raw_path = app.config["RESULT_DIR"] / raw_name
    mask_path = app.config["RESULT_DIR"] / mask_name
    overlay_path = app.config["RESULT_DIR"] / overlay_name

    cv2.imwrite(str(raw_path), image_bgr)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)

    record = build_history_record(
        item_id=uuid.uuid4().hex,
        source_name=source_name,
        raw_url=f"/static/results/{raw_name}",
        mask_url=f"/static/results/{mask_name}",
        overlay_url=f"/static/results/{overlay_name}",
    )
    history_manager.append_record(record)
    return record


def _run_segmentation(image_bgr: np.ndarray):
    model, cfg, device = _load_infer_runtime()

    h, w = image_bgr.shape[:2]
    size = int(cfg["data"]["image_size"])
    threshold = float(cfg["infer"].get("threshold", 0.5))
    preprocess_cfg = cfg["data"].get("preprocess", {})

    x = preprocess_image(
        image_bgr,
        image_size=size,
        use_grayscale=bool(preprocess_cfg.get("use_grayscale", False)),
        apply_clahe=bool(preprocess_cfg.get("apply_clahe", False)),
        clahe_clip_limit=float(preprocess_cfg.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=int(preprocess_cfg.get("clahe_tile_grid_size", 8)),
    )
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
    return render_template("index.html", history=history_manager.list_records())


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose one image.", history=history_manager.list_records())

    if not _allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Only png/jpg/jpeg/bmp/tif/tiff files are supported.",
            history=history_manager.list_records(),
        )

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return render_template("index.html", error="Failed to read the image.", history=history_manager.list_records())

    try:
        mask, overlay = _run_segmentation(image_bgr)
        record = _save_result(image_bgr, mask, overlay, file.filename)
    except FileNotFoundError as e:
        return render_template("index.html", error=str(e), history=history_manager.list_records())
    except Exception as e:
        return render_template("index.html", error=f"Inference failed: {e}", history=history_manager.list_records())

    return render_template(
        "index.html",
        raw_url=record["raw_url"],
        mask_url=record["mask_url"],
        overlay_url=record["overlay_url"],
        success="Inference finished.",
        history=history_manager.list_records(),
    )


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        return render_template("index.html", error="Please choose at least one image.", history=history_manager.list_records())

    batch_results = []
    try:
        for file in files:
            if not _allowed_file(file.filename):
                raise ValueError(f"Unsupported file format: {file.filename}")
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Failed to read image: {file.filename}")
            mask, overlay = _run_segmentation(image_bgr)
            batch_results.append(_save_result(image_bgr, mask, overlay, file.filename))
    except FileNotFoundError as e:
        return render_template("index.html", error=str(e), history=history_manager.list_records())
    except Exception as e:
        return render_template("index.html", error=f"Batch inference failed: {e}", history=history_manager.list_records())

    return render_template(
        "index.html",
        success=f"Batch inference finished for {len(batch_results)} images.",
        batch_results=batch_results,
        history=history_manager.list_records(),
    )


@app.route("/load-json", methods=["POST"])
def load_json():
    file = request.files.get("history_json")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose one history JSON file.", history=history_manager.list_records())

    if not file.filename.lower().endswith(".json"):
        return render_template("index.html", error="Only .json files are supported.", history=history_manager.list_records())

    temp_name = f"import_{uuid.uuid4().hex}.json"
    temp_path = HISTORY_ROOT / temp_name
    file.save(str(temp_path))

    try:
        added = history_manager.load_json(temp_path)
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        return render_template("index.html", error=f"JSON import failed: {e}", history=history_manager.list_records())

    temp_path.unlink(missing_ok=True)
    return render_template(
        "index.html",
        success=f"JSON import succeeded. Added {added} records.",
        history=history_manager.list_records(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
