"""
Image loading and inspection utilities.
Supports JPG, PNG, and DICOM formats.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_image(path: Path) -> Optional[np.ndarray]:
    """
    Load an image from disk. Supports JPG, PNG, and DICOM.

    Args:
        path: Path to image file.

    Returns:
        numpy array in BGR format (H, W, 3) or None if loading fails.
    """
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        return _load_dicom(path)

    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"cv2 could not open image: {path}")
        return None

    return img


def _load_dicom(path: Path) -> Optional[np.ndarray]:
    """
    Load a DICOM file and convert to 3-channel uint8.

    Args:
        path: Path to .dcm file.

    Returns:
        numpy array (H, W, 3) or None.
    """
    try:
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)

        # Normalise to 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        arr = arr.astype(np.uint8)

        # Convert grayscale to 3-channel BGR
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = cv2.cvtColor(arr.squeeze(-1), cv2.COLOR_GRAY2BGR)

        return arr

    except Exception as e:
        logger.error(f"Failed to load DICOM {path}: {e}")
        return None


def get_image_dimensions(path: Path) -> Optional[Tuple[int, int, int]]:
    """
    Return (height, width, channels) without loading the full image.
    Falls back to full load if fast read fails.

    Args:
        path: Image file path.

    Returns:
        Tuple (height, width, channels) or None.
    """
    if path.suffix.lower() == ".dcm":
        img = load_image(path)
        if img is None:
            return None
        return img.shape[0], img.shape[1], img.shape[2]

    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    return h, w, c


def is_image_valid(path: Path) -> bool:
    """
    Return True if the image at path can be successfully opened.

    Args:
        path: Image file path.

    Returns:
        Boolean validity flag.
    """
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    img = load_image(path)
    return img is not None


def draw_bboxes(
    image: np.ndarray,
    boxes: list,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw YOLO-format bounding boxes on an image copy.

    Args:
        image:     BGR image array.
        boxes:     List of [class_id, x_center, y_center, width, height] (normalised).
        color:     BGR colour tuple.
        thickness: Line thickness in pixels.

    Returns:
        Annotated image copy.
    """
    h, w = image.shape[:2]
    output = image.copy()

    for box in boxes:
        if len(box) < 5:
            continue
        _, xc, yc, bw, bh = box[:5]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

    return output