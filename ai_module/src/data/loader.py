"""
Dataset loader — reads the canonical manifest.csv as the single source
of truth. Directory rescanning is only used for consistency verification,
never as the primary source of facts (dataset origin, patient_id, etc).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.utils.logger import get_logger
from src.data.manifest import ManifestStore, UNAVAILABLE

logger = get_logger(__name__)


@dataclass
class DatasetSample:
    sample_id: str
    image_path: Path
    label_path: Optional[Path]
    split: str
    source_dataset: str
    patient_id: str
    study_id: str
    fracture_positive: bool
    num_boxes: int
    annotation_source: str
    annotation_status: str
    width: Optional[int]
    height: Optional[int]


@dataclass
class SplitSummary:
    split: str
    total: int = 0
    positive: int = 0
    negative: int = 0
    total_boxes: int = 0
    positive_ratio: float = 0.0
    by_dataset: Dict[str, int] = field(default_factory=dict)


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


class DatasetLoader:
    def __init__(self, config_path: Path, project_root: Optional[Path] = None):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        raw_path = self.config.get("path", "data/processed")
        self._data_root = (
            Path(raw_path) if Path(raw_path).is_absolute()
            else self.config_path.parent.parent / raw_path
        )

        self.nc = self.config.get("nc", 1)
        self.class_names = self.config.get("names", ["fracture"])

        manifest_path = self._data_root / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Canonical manifest not found: {manifest_path}. "
                f"Run scripts/prepare_dataset.py first."
            )
        self.manifest = ManifestStore.load(manifest_path)

        logger.info(f"DatasetLoader initialised — root: {self._data_root}")
        logger.info(f"Manifest loaded: {len(self.manifest.records)} samples")

    def get_split(self, split: str) -> List[DatasetSample]:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split '{split}'")

        samples = []
        for r in self.manifest.by_split(split):
            img_path = self._data_root / split / "images" / r.image_path
            label_path = self._data_root / split / "labels" / (Path(r.image_path).stem + ".txt")
            samples.append(DatasetSample(
                sample_id=r.sample_id,
                image_path=img_path,
                label_path=label_path if label_path.exists() else None,
                split=split,
                source_dataset=r.dataset,
                patient_id=r.patient_id,
                study_id=r.study_id,
                fracture_positive=(r.fracture_positive == "True"),
                num_boxes=int(r.num_boxes) if r.num_boxes.isdigit() else 0,
                annotation_source=r.annotation_source,
                annotation_status=r.annotation_status,
                width=int(r.width) if r.width not in ("", UNAVAILABLE) else None,
                height=int(r.height) if r.height not in ("", UNAVAILABLE) else None,
            ))
        return samples

    def get_by_dataset(self, split: str, dataset: str) -> List[DatasetSample]:
        """Required for per-dataset evaluation breakdown (FracAtlas vs GRAZPEDWRI)."""
        return [s for s in self.get_split(split) if s.source_dataset == dataset]

    def summary(self, print_report: bool = True) -> DatasetSummary:
        dataset_name = self.config.get("dataset", {}).get("name", "unknown")
        summary = DatasetSummary(dataset_name=dataset_name, nc=self.nc, class_names=self.class_names)

        for split in ("train", "val", "test"):
            samples = self.get_split(split)
            positive = sum(s.fracture_positive for s in samples)
            negative = len(samples) - positive
            boxes = sum(s.num_boxes for s in samples)
            by_dataset: Dict[str, int] = {}
            for s in samples:
                by_dataset[s.source_dataset] = by_dataset.get(s.source_dataset, 0) + 1

            summary.splits[split] = SplitSummary(
                split=split, total=len(samples), positive=positive, negative=negative,
                total_boxes=boxes,
                positive_ratio=positive / len(samples) if samples else 0.0,
                by_dataset=by_dataset,
            )
            summary.total_images += len(samples)
            summary.total_positive += positive
            summary.total_negative += negative
            summary.total_boxes += boxes

        if summary.total_images:
            summary.overall_positive_ratio = summary.total_positive / summary.total_images

        if print_report:
            self._print_summary(summary)
        return summary

    def verify_against_disk(self) -> dict:
        """Cross-checks manifest entries against actual files on disk."""
        missing_images, missing_labels = [], []
        for split in ("train", "val", "test"):
            for s in self.get_split(split):
                if not s.image_path.exists():
                    missing_images.append(str(s.image_path))
                expected_label = self._data_root / split / "labels" / (s.image_path.stem + ".txt")
                if not expected_label.exists():
                    missing_labels.append(str(expected_label))
        return {
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "is_consistent": not missing_images and not missing_labels,
        }

    def get_class_names(self) -> List[str]:
        return self.class_names

    def get_data_root(self) -> Path:
        return self._data_root

    def get_config(self) -> dict:
        return self.config

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _print_summary(self, summary: DatasetSummary) -> None:
        logger.info("=" * 70)
        logger.info(f"DATASET SUMMARY — {summary.dataset_name}")
        logger.info(f"  Classes ({summary.nc}): {summary.class_names}")
        logger.info(
            f"  Total images: {summary.total_images}  Positive: {summary.total_positive}  "
            f"Negative: {summary.total_negative}  Boxes: {summary.total_boxes}"
        )
        for split, s in summary.splits.items():
            logger.info(
                f"  [{split:5s}] total={s.total:6d} pos={s.positive:6d} neg={s.negative:6d} "
                f"boxes={s.total_boxes:6d} by_dataset={s.by_dataset}"
            )
        logger.info("=" * 70)