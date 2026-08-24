"""
File system utilities used across the project.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm"}


def find_images(directory: Path, extensions: Optional[set] = None) -> List[Path]:
    """
    Recursively find all image files under a directory.

    Args:
        directory:  Root path to search.
        extensions: Set of lowercase extensions to include.
                    Defaults to IMAGE_EXTENSIONS.

    Returns:
        Sorted list of image paths.
    """
    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    found = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(found)


def find_files(directory: Path, extension: str) -> List[Path]:
    """
    Recursively find all files with a given extension.

    Args:
        directory: Root path to search.
        extension: Extension including dot, e.g. '.txt', '.xml'.

    Returns:
        Sorted list of matching paths.
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    return sorted(directory.rglob(f"*{extension}"))


def compute_file_hash(path: Path, algorithm: str = "md5") -> str:
    """
    Compute the hash of a file for duplicate detection.

    Args:
        path:      Path to the file.
        algorithm: Hash algorithm — 'md5' | 'sha256'.

    Returns:
        Hex digest string.
    """
    h = hashlib.new(algorithm)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError) as e:
        logger.error(f"Cannot hash file {path}: {e}")
        return ""


def save_json(data: dict, path: Path) -> None:
    """
    Save a dictionary as a pretty-printed JSON file.

    Args:
        data: Dictionary to serialise.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved JSON report: {path}")


def load_json(path: Path) -> dict:
    """
    Load a JSON file and return its contents.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_stem(path: Path) -> str:
    """Return the file stem (name without extension)."""
    return path.stem


def resolve_project_root() -> Path:
    """
    Resolve the project root directory.
    Defined as the directory containing the 'configs' folder.

    Returns:
        Absolute Path to the project root.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists():
            return parent
    # Fallback: two levels up from utils/
    return current.parent.parent.parent