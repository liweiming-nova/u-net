import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils import ensure_dir


Color = Tuple[int, int, int]
Point = Tuple[float, float]

FONT_SERIF = "C:\\Windows\\Fonts\\times.ttf"
FONT_SERIF_BOLD = "C:\\Windows\\Fonts\\timesbd.ttf"
FONT_FALLBACK = "C:\\Windows\\Fonts\\arial.ttf"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [FONT_SERIF_BOLD if bold else FONT_SERIF, FONT_FALLBACK]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: Tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Color,
    anchor: str = "la",
) -> None:
    draw.text(position, text, font=font, fill=fill, anchor=anchor)


def _format_tick(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _to_points(values: Sequence[float], left: int, right: int, top: int, bottom: int, ymin: float, ymax: float) -> List[Point]:
    if not values:
        return []

    if len(values) == 1:
        xs = [left]
    else:
        xs = np.linspace(left, right, num=len(values))

    points: List[Point] = []
    for x, value in zip(xs, values):
        ratio = (value - ymin) / max(ymax - ymin, 1e-12)
        y = bottom - ratio * (bottom - top)
        points.append((float(x), float(y)))
    return points


def _nice_axis_range(values: Sequence[float], prefer_unit: bool = False) -> Tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    if prefer_unit:
        margin = max((vmax - vmin) * 0.08, 0.01)
        return max(0.0, vmin - margin), min(1.0, vmax + margin)

    margin = max((vmax - vmin) * 0.12, 0.005)
    ymin = max(0.0, vmin - margin)
    ymax = vmax + margin
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 0.1
    return ymin, ymax


def _draw_curve_panel(
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Tuple[str, Sequence[float], Color]],
    output_path: str,
    prefer_unit_axis: bool = False,
) -> None:
    width, height = 960, 760
    left, top, right, bottom = 115, 70, 860, 610

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = _load_font(28, bold=False)
    label_font = _load_font(20, bold=False)
    tick_font = _load_font(16, bold=False)
    legend_font = _load_font(16, bold=False)
    caption_font = _load_font(18, bold=True)

    all_values = [float(v) for _, values, _ in series for v in values]
    if not all_values:
        raise ValueError("No metric values available to plot.")

    ymin, ymax = _nice_axis_range(all_values, prefer_unit=prefer_unit_axis)

    draw.rectangle((left, top, right, bottom), outline=(110, 110, 110), width=1)

    y_ticks = 6
    for i in range(y_ticks):
        y = bottom - i * (bottom - top) / (y_ticks - 1)
        tick_value = ymin + i * (ymax - ymin) / (y_ticks - 1)
        _draw_text(draw, (left - 12, y), _format_tick(tick_value, 3 if ymax < 2 else 2), tick_font, (40, 40, 40), anchor="rm")

    epochs = max(len(values) for _, values, _ in series)
    x_ticks = [1]
    if epochs > 1:
        step = max(10, epochs // 5)
        x_ticks = list(range(0, epochs + 1, step))
        if x_ticks[0] != 0:
            x_ticks.insert(0, 0)
        if x_ticks[-1] != epochs:
            x_ticks.append(epochs)
    for tick in x_ticks:
        epoch_value = max(1, tick) if tick == 0 and epochs > 1 else tick
        if epochs == 1:
            x = left
            label = "1"
        else:
            x = left + (tick / epochs) * (right - left)
            label = str(tick)
        _draw_text(draw, (x, bottom + 16), label, tick_font, (40, 40, 40), anchor="ma")

    for name, values, color in series:
        points = _to_points(values, left, right, top, bottom, ymin, ymax)
        if len(points) >= 2:
            draw.line([(int(round(x)), int(round(y))) for x, y in points], fill=color, width=3)

    _draw_text(draw, ((left + right) / 2, 24), title, title_font, (20, 20, 20), anchor="ma")
    _draw_text(draw, ((left + right) / 2, bottom + 54), x_label, label_font, (20, 20, 20), anchor="ma")
    _draw_text(draw, (34, (top + bottom) / 2), y_label, label_font, (20, 20, 20), anchor="la")

    legend_width = 182
    legend_height = 26 + 24 * len(series)
    legend_left = right - legend_width - 10
    legend_top = top + 10
    draw.rectangle((legend_left, legend_top, legend_left + legend_width, legend_top + legend_height), outline=(180, 180, 180), width=1)
    for idx, (name, _, color) in enumerate(series):
        y = legend_top + 18 + idx * 24
        draw.line((legend_left + 10, y, legend_left + 42, y), fill=color, width=3)
        _draw_text(draw, (legend_left + 52, y), name, legend_font, (20, 20, 20), anchor="lm")

    image.save(output_path, format="PNG")


def _align_series(a: Sequence[float], b: Sequence[float]) -> Tuple[List[float], List[float]]:
    n = min(len(a), len(b))
    return list(a[:n]), list(b[:n])


def _draw_caption_strip(title: str, caption: str, output_path: str) -> None:
    width, height = 960, 70
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_bold = _load_font(18, bold=True)
    font_regular = _load_font(18, bold=False)
    _draw_text(draw, (18, 24), title, font_bold, (0, 0, 0), anchor="la")
    title_w, _ = _measure_text(draw, title, font_bold)
    _draw_text(draw, (24 + title_w, 24), caption, font_regular, (0, 0, 0), anchor="la")
    image.save(output_path, format="PNG")


def _best_row(logs: Sequence[Dict]) -> Dict:
    return max(logs, key=lambda row: float(row.get("val_dice", -1.0)))


def _draw_metric_comparison_panel(
    baseline_logs: Sequence[Dict],
    improved_logs: Sequence[Dict],
    baseline_label: str,
    improved_label: str,
    output_path: str,
) -> None:
    baseline = _best_row(baseline_logs)
    improved = _best_row(improved_logs)

    metrics = [
        ("Accuracy", "val_acc"),
        ("Dice", "val_dice"),
        ("Sensitivity", "val_sen"),
        ("Specificity", "val_spe"),
        ("AUC", "val_auc"),
    ]

    width, height = 1120, 760
    left, top, right, bottom = 110, 90, 1030, 610
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = _load_font(28, bold=False)
    label_font = _load_font(20, bold=False)
    tick_font = _load_font(16, bold=False)
    legend_font = _load_font(16, bold=False)
    value_font = _load_font(14, bold=False)

    draw.rectangle((left, top, right, bottom), outline=(110, 110, 110), width=1)

    ymin, ymax = 0.0, 1.0
    y_ticks = 6
    for i in range(y_ticks):
        y = bottom - i * (bottom - top) / (y_ticks - 1)
        tick_value = ymin + i * (ymax - ymin) / (y_ticks - 1)
        _draw_text(draw, (left - 12, y), _format_tick(tick_value, 2), tick_font, (40, 40, 40), anchor="rm")

    group_width = (right - left) / len(metrics)
    bar_width = 34
    baseline_color = (61, 79, 201)
    improved_color = (200, 77, 77)

    for idx, (label, key) in enumerate(metrics):
        center_x = left + group_width * (idx + 0.5)
        baseline_val = float(baseline[key])
        improved_val = float(improved[key])

        b_left = center_x - 42
        i_left = center_x + 8
        b_top = bottom - (baseline_val - ymin) / (ymax - ymin) * (bottom - top)
        i_top = bottom - (improved_val - ymin) / (ymax - ymin) * (bottom - top)

        draw.rectangle((b_left, b_top, b_left + bar_width, bottom), fill=baseline_color, outline=baseline_color)
        draw.rectangle((i_left, i_top, i_left + bar_width, bottom), fill=improved_color, outline=improved_color)

        _draw_text(draw, (b_left + bar_width / 2, b_top - 8), f"{baseline_val:.3f}", value_font, (30, 30, 30), anchor="ms")
        _draw_text(draw, (i_left + bar_width / 2, i_top - 8), f"{improved_val:.3f}", value_font, (30, 30, 30), anchor="ms")
        _draw_text(draw, (center_x, bottom + 22), label, tick_font, (40, 40, 40), anchor="ma")

    _draw_text(draw, ((left + right) / 2, 34), "Performance comparison of baseline and improved models", title_font, (20, 20, 20), anchor="ma")
    _draw_text(draw, ((left + right) / 2, bottom + 60), "evaluation metrics", label_font, (20, 20, 20), anchor="ma")
    _draw_text(draw, (36, (top + bottom) / 2), "score", label_font, (20, 20, 20), anchor="la")

    legend_left = right - 240
    legend_top = 30
    draw.rectangle((legend_left, legend_top, legend_left + 220, legend_top + 56), outline=(180, 180, 180), width=1)
    draw.rectangle((legend_left + 12, legend_top + 12, legend_left + 34, legend_top + 28), fill=baseline_color, outline=baseline_color)
    draw.rectangle((legend_left + 12, legend_top + 34, legend_left + 34, legend_top + 50), fill=improved_color, outline=improved_color)
    _draw_text(draw, (legend_left + 46, legend_top + 20), baseline_label, legend_font, (20, 20, 20), anchor="lm")
    _draw_text(draw, (legend_left + 46, legend_top + 42), improved_label, legend_font, (20, 20, 20), anchor="lm")

    note = (
        f"Baseline best Dice: epoch {int(baseline['epoch'])}, {float(baseline['val_dice']):.3f}    "
        f"Improved best Dice: epoch {int(improved['epoch'])}, {float(improved['val_dice']):.3f}"
    )
    _draw_text(draw, (left, 675), note, legend_font, (40, 40, 40), anchor="la")

    image.save(output_path, format="PNG")


def plot_training_curves(logs: Sequence[Dict], output_dir: str) -> Dict[str, str]:
    if not logs:
        raise ValueError("metrics logs are empty")

    ensure_dir(output_dir)

    train_loss = [float(item["train_loss"]) for item in logs]
    val_loss = [float(item["val_loss"]) for item in logs]
    train_dice = [float(item["train_dice"]) for item in logs]
    val_dice = [float(item["val_dice"]) for item in logs]

    loss_path = os.path.join(output_dir, "loss_curve.png")
    dice_path = os.path.join(output_dir, "dice_curve.png")
    loss_caption_path = os.path.join(output_dir, "loss_caption.png")
    dice_caption_path = os.path.join(output_dir, "dice_caption.png")

    _draw_curve_panel(
        title="Training and Validation loss",
        x_label="epoch",
        y_label="loss",
        series=[
            ("Training loss", train_loss, (61, 79, 201)),
            ("Validation val_loss", val_loss, (200, 77, 77)),
        ],
        output_path=loss_path,
        prefer_unit_axis=False,
    )
    _draw_curve_panel(
        title="Training and Validation Dice",
        x_label="epoch",
        y_label="Dice",
        series=[
            ("Training dice", train_dice, (61, 79, 201)),
            ("Validation val_dice", val_dice, (200, 77, 77)),
        ],
        output_path=dice_path,
        prefer_unit_axis=True,
    )
    _draw_caption_strip(
        "Fig. 8",
        "The change of loss during the training process.",
        loss_caption_path,
    )
    _draw_caption_strip(
        "Fig. 9",
        "The change of Dice during the training process.",
        dice_caption_path,
    )
    return {
        "loss_curve": loss_path,
        "dice_curve": dice_path,
        "loss_caption": loss_caption_path,
        "dice_caption": dice_caption_path,
    }


def plot_training_curves_from_json(metrics_path: str, output_dir: str = None) -> Dict[str, str]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    target_dir = output_dir or os.path.dirname(metrics_path)
    return plot_training_curves(logs, target_dir)


def plot_comparison_curves(
    baseline_logs: Sequence[Dict],
    improved_logs: Sequence[Dict],
    output_dir: str,
    baseline_label: str = "Baseline U-Net",
    improved_label: str = "Improved U-Net",
) -> Dict[str, str]:
    if not baseline_logs or not improved_logs:
        raise ValueError("Both baseline logs and improved logs are required.")

    ensure_dir(output_dir)

    baseline_train_loss = [float(item["train_loss"]) for item in baseline_logs]
    baseline_val_loss = [float(item["val_loss"]) for item in baseline_logs]
    baseline_train_dice = [float(item["train_dice"]) for item in baseline_logs]
    baseline_val_dice = [float(item["val_dice"]) for item in baseline_logs]

    improved_train_loss = [float(item["train_loss"]) for item in improved_logs]
    improved_val_loss = [float(item["val_loss"]) for item in improved_logs]
    improved_train_dice = [float(item["train_dice"]) for item in improved_logs]
    improved_val_dice = [float(item["val_dice"]) for item in improved_logs]

    compare_loss_path = os.path.join(output_dir, "compare_loss_curve.png")
    compare_dice_path = os.path.join(output_dir, "compare_dice_curve.png")
    compare_metrics_path = os.path.join(output_dir, "compare_metrics_bar.png")
    compare_loss_caption_path = os.path.join(output_dir, "compare_loss_caption.png")
    compare_dice_caption_path = os.path.join(output_dir, "compare_dice_caption.png")
    compare_metrics_caption_path = os.path.join(output_dir, "compare_metrics_caption.png")

    _draw_curve_panel(
        title="Baseline and Improved Model loss comparison",
        x_label="epoch",
        y_label="loss",
        series=[
            (f"{baseline_label} train", baseline_train_loss, (61, 79, 201)),
            (f"{baseline_label} val", baseline_val_loss, (200, 77, 77)),
            (f"{improved_label} train", improved_train_loss, (40, 150, 107)),
            (f"{improved_label} val", improved_val_loss, (219, 140, 62)),
        ],
        output_path=compare_loss_path,
        prefer_unit_axis=False,
    )
    _draw_curve_panel(
        title="Baseline and Improved Model Dice comparison",
        x_label="epoch",
        y_label="Dice",
        series=[
            (f"{baseline_label} train", baseline_train_dice, (61, 79, 201)),
            (f"{baseline_label} val", baseline_val_dice, (200, 77, 77)),
            (f"{improved_label} train", improved_train_dice, (40, 150, 107)),
            (f"{improved_label} val", improved_val_dice, (219, 140, 62)),
        ],
        output_path=compare_dice_path,
        prefer_unit_axis=True,
    )
    _draw_caption_strip(
        "Fig. 10",
        "Comparison of loss curves between baseline U-Net and improved U-Net.",
        compare_loss_caption_path,
    )
    _draw_caption_strip(
        "Fig. 11",
        "Comparison of Dice curves between baseline U-Net and improved U-Net.",
        compare_dice_caption_path,
    )
    _draw_metric_comparison_panel(
        baseline_logs=baseline_logs,
        improved_logs=improved_logs,
        baseline_label=baseline_label,
        improved_label=improved_label,
        output_path=compare_metrics_path,
    )
    _draw_caption_strip(
        "Fig. 12",
        "Comparison of key segmentation metrics between baseline U-Net and improved U-Net.",
        compare_metrics_caption_path,
    )
    return {
        "compare_loss_curve": compare_loss_path,
        "compare_dice_curve": compare_dice_path,
        "compare_metrics_bar": compare_metrics_path,
        "compare_loss_caption": compare_loss_caption_path,
        "compare_dice_caption": compare_dice_caption_path,
        "compare_metrics_caption": compare_metrics_caption_path,
    }


def plot_comparison_curves_from_json(
    baseline_metrics_path: str,
    improved_metrics_path: str,
    output_dir: str,
    baseline_label: str = "Baseline U-Net",
    improved_label: str = "Improved U-Net",
) -> Dict[str, str]:
    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        baseline_logs = json.load(f)
    with open(improved_metrics_path, "r", encoding="utf-8") as f:
        improved_logs = json.load(f)

    return plot_comparison_curves(
        baseline_logs=baseline_logs,
        improved_logs=improved_logs,
        output_dir=output_dir,
        baseline_label=baseline_label,
        improved_label=improved_label,
    )
