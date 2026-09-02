"""
Image loading and inspection utilities.
Supports JPG, PNG, and DICOM formats.

IMPORTANT — Integrity vs. Speed tradeoff:
    - get_image_dimensions(): FAST, header-only read. Does NOT guarantee
      the pixel data is intact. A truncated JPEG can have a perfectly
      valid header.
    - check_image_integrity(): SLOW, forces a full pixel decode. This is
      the ONLY function that reliably detects "Premature end of JPEG
      file" style truncation, which cv2.imread() silently tolerates
      (returning a non-None but partially garbled array while printing
      a libjpeg warning directly to stderr — NOT a Python exception,
      NOT reflected in the return value).

Use check_image_integrity() during dataset preparation / auditing
(one-time, authoritative). Use get_image_dimensions() for routine,
repeated metadata reads once a file's integrity is already established.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PIL_MODE_TO_CHANNELS = {"1": 1, "L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: Path) -> Optional[np.ndarray]:
    """
    Load an image from disk. Supports JPG, PNG, and DICOM.

    NOTE: For truncated/corrupted JPEGs, cv2.imread() may return a
    non-None array containing garbled pixel data while printing a
    libjpeg warning to stderr. This function does NOT catch that case.
    Use check_image_integrity() first if correctness matters more than
    speed (e.g. during dataset preparation).

    Args:
        path: Path to image file.

    Returns:
        numpy array in BGR format (H, W, 3) or None if loading fails outright.
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

        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = cv2.cvtColor(arr.squeeze(-1), cv2.COLOR_GRAY2BGR)

        return arr

    except Exception as e:
        logger.error(f"Failed to load DICOM {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Fast metadata (header-only, does NOT guarantee pixel data integrity)
# ---------------------------------------------------------------------------

def get_image_dimensions(path: Path) -> Optional[Tuple[int, int, int]]:
    """
    Fast dimension read. Reads only the file header where possible
    (PIL lazy-loads headers without decoding pixel data for most formats).

    This is intentionally fast and is NOT a substitute for
    check_image_integrity(). A file can report valid dimensions here
    while still being truncated in its pixel data — these are two
    independent failure modes.

    Args:
        path: Image file path.

    Returns:
        Tuple (height, width, channels) or None if the header cannot be read.
    """
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        img = load_image(path)
        if img is None:
            return None
        return img.shape[0], img.shape[1], img.shape[2]

    try:
        with Image.open(path) as im:
            w, h = im.size
            c = _PIL_MODE_TO_CHANNELS.get(im.mode, 3)
        return h, w, c
    except Exception as e:
        logger.warning(f"Could not read header/dimensions for {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Deep integrity check (SLOW, authoritative)
# ---------------------------------------------------------------------------

def check_image_integrity(path: Path) -> Tuple[bool, str]:
    """
    Deep integrity check — forces a full pixel decode to reliably detect
    truncated/corrupted files.

    This is the ONLY function in this module that will catch a truncated
    JPEG ("Premature end of JPEG file"). cv2.imread() does NOT reliably
    catch this — it can return a valid-looking but incomplete array.

    Args:
        path: Path to image file (JPG/PNG/DICOM).

    Returns:
        Tuple (is_fully_readable, status), where status is one of:
            "ok"
            "missing"
            "zero_byte"
            "truncated_or_corrupted: <detail>"
    """
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "zero_byte"

    suffix = path.suffix.lower()

    if suffix == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))
            _ = ds.pixel_array  # forces full pixel data decode
            return True, "ok"
        except Exception as e:
            return False, f"truncated_or_corrupted: {e}"

    original_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False  # do not silently tolerate truncation
    try:
        with Image.open(path) as img:
            img.verify()  # fast structural check, invalidates the handle
        with Image.open(path) as img:
            img.load()   # forces full pixel decode; raises on truncation
        return True, "ok"
    except Exception as e:
        return False, f"truncated_or_corrupted: {e}"
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = original_setting


def is_image_valid(path: Path, deep: bool = False) -> bool:
    """
    Return True if the image at path can be successfully opened.

    Args:
        path: Image file path.
        deep: If True, uses check_image_integrity() (slow, catches
              truncation). If False (default, backward-compatible),
              uses the cheaper cv2-based check that does NOT catch
              truncated files.

    Returns:
        Boolean validity flag.
    """
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    if deep:
        ok, _ = check_image_integrity(path)
        return ok
    img = load_image(path)
    return img is not None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

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