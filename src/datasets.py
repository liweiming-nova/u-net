import os
from typing import Callable, Dict, List, Optional, Tuple

try:
    import albumentations as A
except ModuleNotFoundError:
    A = None

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _list_images(image_dir: str) -> List[str]:
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files


def preprocess_image(
    image_bgr: np.ndarray,
    image_size: int,
    use_grayscale: bool = False,
    apply_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Input image is None")

    if use_grayscale:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if apply_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
            )
            image = clahe.apply(image)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        return image[..., None]

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if apply_clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size),
        )
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def build_train_augment(cfg: Optional[Dict]) -> Optional[Callable]:
    cfg = cfg or {}
    if not cfg.get("enabled", False) or A is None:
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=float(cfg.get("hflip_p", 0.5))),
            A.VerticalFlip(p=float(cfg.get("vflip_p", 0.5))),
            A.Rotate(limit=int(cfg.get("rotate_limit", 30)), p=float(cfg.get("rotate_p", 0.5))),
            A.RandomScale(scale_limit=float(cfg.get("scale_limit", 0.2)), p=float(cfg.get("scale_p", 0.5))),
            A.ElasticTransform(
                alpha=float(cfg.get("elastic_alpha", 20.0)),
                sigma=float(cfg.get("elastic_sigma", 6.0)),
                p=float(cfg.get("elastic_p", 0.2)),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=float(cfg.get("brightness_limit", 0.2)),
                contrast_limit=float(cfg.get("contrast_limit", 0.2)),
                p=float(cfg.get("brightness_contrast_p", 0.3)),
            ),
            A.GaussNoise(
                var_limit=tuple(cfg.get("gauss_var_limit", [10.0, 50.0])),
                p=float(cfg.get("gauss_p", 0.2)),
            ),
        ]
    )


class RetinaVesselDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        image_size: int = 512,
        augment: Optional[Callable] = None,
        use_grayscale: bool = False,
        apply_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.use_grayscale = use_grayscale
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        self.image_files = _list_images(image_dir)
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _read_image(self, path: str) -> np.ndarray:
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        return preprocess_image(
            image_bgr=image_bgr,
            image_size=self.image_size,
            use_grayscale=self.use_grayscale,
            apply_clahe=self.apply_clahe,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid_size=self.clahe_tile_grid_size,
        )

    def _read_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, name)
        image = self._read_image(image_path)

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, name)
            mask = self._read_mask(mask_path)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        if self.augment is not None:
            out = self.augment(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask, name
