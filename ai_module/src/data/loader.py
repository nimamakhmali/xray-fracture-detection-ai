"""
Dataset loader for YOLO-format fracture detection datasets.

Provides a clean Python API for:
  - Loading dataset configuration from YAML
  - Discovering image/label pairs per split
  - Computing dataset statistics
  - Returning structured sample lists

Does NOT re-implement Ultralytics' DataLoader.
Used for inspection, statistics, and project-level pipeline control.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.utils.logger import get_logger
from src.utils.file_utils import IMAGE_EXTENSIONS

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetSample:
    """Represents one image-label pair in the dataset."""
    image_path: Path
    label_path: Optional[Path]
    split: str
    has_label: bool
    num_boxes: int
    fracture_positive: bool
    source_dataset: Optional[str] = None
    original_filename: Optional[str] = None


@dataclass
class SplitSummary:
    split: str
    total: int = 0
    positive: int = 0
    negative: int = 0
    missing_labels: int = 0
    total_boxes: int = 0
    positive_ratio: float = 0.0


@dataclass
class DatasetSummary:
    dataset_name: str
    nc: int
    class_names: List[str]
    splits: Dict[str, SplitSummary] = field(default_factory=dict)
    total_images: int = 0
    total_positive: int = 0
    total_negative: int = 0
    total_boxes: int = 0
    overall_positive_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Project-level dataset loader.

    Args:
        config_path: Path to dataset YAML config (e.g. configs/dataset.yaml).
        project_root: Optional override for project root resolution.

    Example:
        loader = DatasetLoader(config_path=Path("configs/dataset.yaml"))
        summary = loader.summary()
        train_samples = loader.get_split("train")
    """

    def __init__(
        self,
        config_path: Path,
        project_root: Optional[Path] = None,
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Resolve base data path
        if project_root is not None:
            self._root = Path(project_root)
        else:
            self._root = self._resolve_root()

        # Resolve processed data path
        raw_path = self.config.get("path", "data/processed")
        if Path(raw_path).is_absolute():
            self._data_root = Path(raw_path)
        else:
            self._data_root = self._root / raw_path

        self.nc: int = self.config.get("nc", 1)
        self.class_names: List[str] = self.config.get("names", ["fracture"])

        logger.info(f"DatasetLoader initialised — root: {self._data_root}")
        logger.info(f"Classes ({self.nc}): {self.class_names}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_split(self, split: str) -> List[DatasetSample]:
        """
        Return all samples for a given split.

        Args:
            split: One of 'train', 'val', 'test'.

        Returns:
            List of DatasetSample objects.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'. Must be train | val | test.")

        split_key_map = {"train": "train", "val": "val", "test": "test"}
        rel_images = self.config.get(split_key_map[split], f"{split}/images")

        images_dir = self._data_root / rel_images
        labels_dir = images_dir.parent.parent / split / "labels"

        if not images_dir.exists():
            logger.warning(f"images dir not found: {images_dir}")
            return []

        image_files = sorted([
            p for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        samples = []
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            has_label = label_path.exists()
            num_boxes, positive = self._parse_label(label_path) if has_label else (0, False)

            samples.append(DatasetSample(
                image_path=img_path,
                label_path=label_path if has_label else None,
                split=split,
                has_label=has_label,
                num_boxes=num_boxes,
                fracture_positive=positive,
            ))

        logger.info(
            f"[{split}] Loaded {len(samples)} samples — "
            f"positive={sum(s.fracture_positive for s in samples)}, "
            f"negative={sum(not s.fracture_positive for s in samples)}"
        )
        return samples

    def summary(self, print_report: bool = True) -> DatasetSummary:
        """
        Compute and optionally print a full dataset summary.

        Args:
            print_report: Whether to log the summary.

        Returns:
            DatasetSummary object.
        """
        dataset_name = self.config.get("dataset", {}).get("name", "unknown")
        summary = DatasetSummary(
            dataset_name=dataset_name,
            nc=self.nc,
            class_names=self.class_names,
        )

        for split in ("train", "val", "test"):
            samples = self.get_split(split)
            positive = sum(s.fracture_positive for s in samples)
            negative = sum(not s.fracture_positive for s in samples)
            boxes = sum(s.num_boxes for s in samples)
            missing = sum(not s.has_label for s in samples)

            split_summary = SplitSummary(
                split=split,
                total=len(samples),
                positive=positive,
                negative=negative,
                missing_labels=missing,
                total_boxes=boxes,
                positive_ratio=positive / len(samples) if samples else 0.0,
            )
            summary.splits[split] = split_summary
            summary.total_images += len(samples)
            summary.total_positive += positive
            summary.total_negative += negative
            summary.total_boxes += boxes

        if summary.total_images > 0:
            summary.overall_positive_ratio = (
                summary.total_positive / summary.total_images
            )

        if print_report:
            self._print_summary(summary)

        return summary

    def get_class_names(self) -> List[str]:
        """Return list of class names from config."""
        return self.class_names

    def get_data_root(self) -> Path:
        """Return resolved data root path."""
        return self._data_root

    def get_config(self) -> dict:
        """Return raw config dictionary."""
        return self.config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_root(self) -> Path:
        """
        Resolve project root as the directory containing 'configs/'.
        Walk up from this file's location.
        """
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "configs").exists():
                return parent
        return Path.cwd()

    def _parse_label(self, label_path: Path) -> Tuple[int, bool]:
        """
        Parse a YOLO label file and return (num_boxes, is_positive).

        Args:
            label_path: Path to .txt label file.

        Returns:
            Tuple (number of valid boxes, fracture_positive flag).
        """
        try:
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            valid_lines = [l for l in lines if l.strip()]
            return len(valid_lines), len(valid_lines) > 0
        except Exception as e:
            logger.warning(f"Could not parse label {label_path}: {e}")
            return 0, False

    def _print_summary(self, summary: DatasetSummary) -> None:
        """Log formatted dataset summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"DATASET SUMMARY — {summary.dataset_name}")
        logger.info("=" * 60)
        logger.info(f"  Classes ({summary.nc}): {summary.class_names}")
        logger.info(f"  Total images       : {summary.total_images}")
        logger.info(f"  Total positive     : {summary.total_positive}")
        logger.info(f"  Total negative     : {summary.total_negative}")
        logger.info(f"  Total boxes        : {summary.total_boxes}")
        logger.info(
            f"  Positive ratio     : {summary.overall_positive_ratio:.2%}"
        )
        logger.info("")

        header = f"  {'Split':8s} {'Total':>7} {'Pos':>7} {'Neg':>7} {'Boxes':>8} {'PosRatio':>10}"
        logger.info(header)
        logger.info("  " + "-" * 50)

        for split in ("train", "val", "test"):
            if split not in summary.splits:
                continue
            s = summary.splits[split]
            logger.info(
                f"  {s.split:8s} "
                f"{s.total:>7d} "
                f"{s.positive:>7d} "
                f"{s.negative:>7d} "
                f"{s.total_boxes:>8d} "
                f"{s.positive_ratio:>10.2%}"
            )

        logger.info("=" * 60)