"""
Augmentation pipelines for the fracture detection dataset.

STRICT RULE (audit Section 27): augmentation is applied ONLY to the
training split. Validation/test pipelines are deterministic.

⚠️ ARCHITECTURAL NOTE — READ BEFORE USING:
If the training loop uses Ultralytics' high-level API
(`YOLO(...).train(data=...)`), augmentation is handled INTERNALLY by
Ultralytics via hyperparameters in configs/model_config.yaml
(hsv_h, hsv_s, hsv_v, degrees, translate, scale, fliplr, mosaic, mixup...).
In that case Ultralytics ALREADY guarantees val/test receive no
augmentation, and this module is not on the training critical path.

This module becomes necessary only if:
    (a) a custom PyTorch training loop is implemented instead of the
        Ultralytics high-level API, or
    (b) a consistent, explicit preprocessing boundary is needed for the
        inference/serving layer (src/inference/), independent of training.

Do not wire this into the training pipeline until that decision (custom
loop vs Ultralytics API) is made explicit in src/training/trainer.py.
"""
from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

_DETERMINISTIC_TRANSFORMS = (
    A.Resize, A.LongestMaxSize, A.PadIfNeeded, A.Normalize, A.CenterCrop, ToTensorV2,
)


def get_train_transforms(image_size: int = 640) -> A.Compose:
    """Stochastic augmentation — TRAIN SPLIT ONLY."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=(0, 0, 0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.Affine(rotate=(-8, 8), translate_percent=(0.0, 0.05), scale=(0.95, 1.05), p=0.4),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    )


def get_eval_transforms(image_size: int = 640) -> A.Compose:
    """Deterministic pipeline — VALIDATION / TEST ONLY. No randomness."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=(0, 0, 0)),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def assert_no_random_ops(transform: A.Compose) -> None:
    """Guard: raises if a supposedly-deterministic pipeline contains randomness."""
    for t in transform.transforms:
        if not isinstance(t, _DETERMINISTIC_TRANSFORMS):
            raise RuntimeError(
                f"Non-deterministic transform '{type(t).__name__}' found in an eval "
                f"pipeline — this would leak augmentation into validation/test data."
            )


def get_transforms(split: str, image_size: int = 640) -> A.Compose:
    if split == "train":
        return get_train_transforms(image_size)
    elif split in ("val", "test"):
        t = get_eval_transforms(image_size)
        assert_no_random_ops(t)
        return t
    raise ValueError(f"Unknown split '{split}'")