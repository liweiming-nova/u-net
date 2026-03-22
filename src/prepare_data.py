import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils import ensure_dir, load_yaml, set_seed


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def _stem_tokens(path: Path) -> List[str]:
    return path.stem.lower().replace("-", "_").split("_")


def _drive_id(path: Path) -> Optional[str]:
    for token in _stem_tokens(path):
        if token.isdigit():
            return token
    return None


def _copy_pairs(pairs: List[Tuple[Path, Path]], split_dir: Path) -> None:
    image_out = split_dir / "images"
    mask_out = split_dir / "masks"
    ensure_dir(str(image_out))
    ensure_dir(str(mask_out))
    for image_path, mask_path in pairs:
        shutil.copy2(image_path, image_out / image_path.name)
        shutil.copy2(mask_path, mask_out / image_path.name)


def _copy_images(images: List[Path], split_dir: Path) -> None:
    image_out = split_dir / "images"
    ensure_dir(str(image_out))
    for image_path in images:
        shutil.copy2(image_path, image_out / image_path.name)


def _clean_output_root(output_root: Path) -> None:
    for split in ["train", "val", "test"]:
        split_dir = output_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def _paired_files_generic(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_map: Dict[str, Path] = {}
    for mask_path in sorted(mask_dir.iterdir()):
        if mask_path.suffix.lower() not in IMG_EXTS:
            continue
        mask_map[mask_path.stem.lower()] = mask_path

    items = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        mask_path = mask_map.get(image_path.stem.lower())
        if mask_path is not None:
            items.append((image_path, mask_path))

    if not items:
        raise ValueError(f"No paired files found in {image_dir} and {mask_dir}")
    return items


def _build_drive_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_map: Dict[str, Path] = {}
    for mask_path in sorted(mask_dir.iterdir()):
        if mask_path.suffix.lower() not in IMG_EXTS:
            continue
        item_id = _drive_id(mask_path)
        if item_id is not None:
            mask_map[item_id] = mask_path

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        item_id = _drive_id(image_path)
        if item_id is None:
            continue
        mask_path = mask_map.get(item_id)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    if not pairs:
        raise ValueError(f"No DRIVE pairs found in {image_dir} and {mask_dir}")
    return pairs


def _list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    return images


def _prepare_generic(cfg: Dict, output_root: Path, seed: int) -> None:
    source_image_dir = Path(cfg["source_image_dir"])
    source_mask_dir = Path(cfg["source_mask_dir"])
    train_ratio = float(cfg.get("train_ratio", 0.8))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.1))

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    pairs = _paired_files_generic(source_image_dir, source_mask_dir)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    _copy_pairs(train_pairs, output_root / "train")
    _copy_pairs(val_pairs, output_root / "val")
    _copy_pairs(test_pairs, output_root / "test")

    print({"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs), "output_root": str(output_root)})


def _prepare_drive(cfg: Dict, output_root: Path, seed: int) -> None:
    drive_root = Path(cfg["drive_root"])
    train_images_dir = drive_root / "training" / "images"
    train_masks_dir = drive_root / "training" / "1st_manual"
    test_images_dir = drive_root / "test" / "images"
    optional_test_manual_dir = drive_root / "test" / "1st_manual"
    val_ratio = float(cfg.get("val_ratio_within_training", 0.2))

    if not drive_root.exists():
        raise FileNotFoundError(f"DRIVE root directory not found: {drive_root}")

    train_pairs = _build_drive_pairs(train_images_dir, train_masks_dir)
    test_images = _list_images(test_images_dir)

    rng = random.Random(seed)
    rng.shuffle(train_pairs)

    n_val = max(1, int(len(train_pairs) * val_ratio))
    val_pairs = train_pairs[:n_val]
    actual_train_pairs = train_pairs[n_val:]

    _copy_pairs(actual_train_pairs, output_root / "train")
    _copy_pairs(val_pairs, output_root / "val")
    _copy_images(test_images, output_root / "test")

    if optional_test_manual_dir.exists():
        test_pairs = _build_drive_pairs(test_images_dir, optional_test_manual_dir)
        _copy_pairs(test_pairs, output_root / "test")

    print(
        {
            "train": len(actual_train_pairs),
            "val": len(val_pairs),
            "test_images": len(test_images),
            "test_manual_available": optional_test_manual_dir.exists(),
            "drive_root": str(drive_root),
            "output_root": str(output_root),
        }
    )


def prepare_data(cfg_path: str) -> None:
    cfg = load_yaml(cfg_path)
    data_cfg = cfg.get("prepare_data", {})
    output_root = Path(data_cfg.get("output_root", "data"))
    mode = str(data_cfg.get("source_format", "generic")).lower()
    seed = int(cfg.get("seed", 42))

    set_seed(seed)
    _clean_output_root(output_root)

    if mode == "drive":
        _prepare_drive(data_cfg, output_root, seed)
        return
    if mode == "generic":
        _prepare_generic(data_cfg, output_root, seed)
        return
    raise ValueError(f"Unsupported prepare_data.source_format: {mode}")
