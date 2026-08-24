"""
Dataset validator for YOLO-format fracture detection datasets.

Usage as module:
    from src.data.validator import DatasetValidator
    v = DatasetValidator(processed_dir=Path("data/processed"))
    report = v.validate()

Usage as CLI:
    python -m src.data.validator --data data/processed
    python scripts/validate_dataset.py
"""

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.file_utils import save_json, IMAGE_EXTENSIONS

logger = get_logger(__name__)

VALID_CLASS_IDS = {0}  # Phase 1: fracture only
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LabelIssue:
    image_path: str
    label_path: str
    issue_type: str          # MISSING | MALFORMED | OUT_OF_RANGE | ZERO_AREA | UNKNOWN_CLASS
    line_number: Optional[int] = None
    raw_line: Optional[str] = None
    detail: Optional[str] = None
    severity: str = "ERROR"  # ERROR | WARNING | INFO


@dataclass
class SplitStats:
    split: str
    total_images: int = 0
    total_labels: int = 0
    positive_images: int = 0
    negative_images: int = 0
    total_boxes: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    corrupted_images: int = 0
    invalid_label_files: int = 0


@dataclass
class ValidationReport:
    status: str = "NOT_READY_FOR_TRAINING"
    total_images: int = 0
    total_labels: int = 0
    positive_images: int = 0
    negative_images: int = 0
    total_boxes: int = 0
    boxes_by_class: Dict[int, int] = field(default_factory=dict)
    missing_labels: int = 0
    orphan_labels: int = 0
    corrupted_images: int = 0
    invalid_label_files: int = 0
    out_of_range_boxes: int = 0
    zero_area_boxes: int = 0
    duplicate_images: int = 0
    duplicate_annotations: int = 0
    split_stats: Dict[str, dict] = field(default_factory=dict)
    issues: List[dict] = field(default_factory=list)
    critical_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """
    Validates a processed YOLO-format dataset.

    The validator checks:
    - Image-label correspondence
    - YOLO label format correctness
    - Coordinate normalisation
    - Class ID validity
    - Image readability
    - Duplicate detection
    - Positive/negative sample distribution

    Empty label files are treated as valid negative samples
    (configurable via allow_empty_labels parameter).
    """

    def __init__(
        self,
        processed_dir: Path,
        allow_empty_labels: bool = True,
        report_dir: Optional[Path] = None,
    ):
        """
        Args:
            processed_dir:     Root of data/processed/ directory.
            allow_empty_labels: If True, images with no annotations are valid
                                negative samples (not errors).
            report_dir:        Where to save the JSON report. Defaults to
                               reports/ in project root.
        """
        self.processed_dir = Path(processed_dir)
        self.allow_empty_labels = allow_empty_labels

        if report_dir is None:
            self.report_dir = self.processed_dir.parent.parent / "reports"
        else:
            self.report_dir = Path(report_dir)

        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.report = ValidationReport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, save_report: bool = True) -> ValidationReport:
        """
        Run full validation pipeline.

        Args:
            save_report: If True, saves JSON report to reports/.

        Returns:
            Populated ValidationReport.
        """
        logger.info("=" * 60)
        logger.info("Starting dataset validation")
        logger.info(f"Dataset root: {self.processed_dir}")
        logger.info("=" * 60)

        all_image_hashes: Dict[str, str] = {}  # hash -> path for duplicate detection

        for split in SPLITS:
            split_dir = self.processed_dir / split
            if not split_dir.exists():
                self.report.critical_errors.append(
                    f"Split directory missing: {split_dir}"
                )
                logger.error(f"Split directory missing: {split_dir}")
                continue

            stats = self._validate_split(split, split_dir, all_image_hashes)
            self.report.split_stats[split] = asdict(stats)

            # Aggregate totals
            self.report.total_images += stats.total_images
            self.report.total_labels += stats.total_labels
            self.report.positive_images += stats.positive_images
            self.report.negative_images += stats.negative_images
            self.report.total_boxes += stats.total_boxes
            self.report.missing_labels += stats.missing_labels
            self.report.orphan_labels += stats.orphan_labels
            self.report.corrupted_images += stats.corrupted_images
            self.report.invalid_label_files += stats.invalid_label_files

        # Determine final status
        self.report.status = self._determine_status()

        if save_report:
            report_path = self.report_dir / "validation_report.json"
            save_json(asdict(self.report), report_path)

        self._print_summary()
        return self.report

    def validate_single_label(
        self, label_path: Path, image_path: Optional[Path] = None
    ) -> List[LabelIssue]:
        """
        Validate a single label file.

        Args:
            label_path:  Path to .txt YOLO label.
            image_path:  Corresponding image for dimension checks (optional).

        Returns:
            List of LabelIssue objects.
        """
        issues = []

        if not label_path.exists():
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="MISSING",
                detail="Label file does not exist",
                severity="ERROR",
            ))
            return issues

        try:
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        except Exception as e:
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="MALFORMED",
                detail=f"Cannot read label file: {e}",
                severity="ERROR",
            ))
            return issues

        # Empty label file — valid negative sample if configured
        if not lines:
            return issues

        seen_boxes = set()

        for line_no, line in enumerate(lines, start=1):
            line_issues = self._validate_label_line(
                line=line,
                line_no=line_no,
                label_path=label_path,
                image_path=image_path,
                seen_boxes=seen_boxes,
            )
            issues.extend(line_issues)

        return issues

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _validate_split(
        self,
        split: str,
        split_dir: Path,
        all_image_hashes: Dict[str, str],
    ) -> SplitStats:
        """Validate one split (train/val/test)."""
        stats = SplitStats(split=split)
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            self.report.critical_errors.append(
                f"images/ directory missing in split '{split}'"
            )
            return stats

        if not labels_dir.exists():
            self.report.critical_errors.append(
                f"labels/ directory missing in split '{split}'"
            )
            return stats

        image_paths = [
            p for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        label_paths = {p.stem: p for p in labels_dir.glob("*.txt")}

        stats.total_images = len(image_paths)
        logger.info(f"[{split}] Found {stats.total_images} images, "
                    f"{len(label_paths)} label files")

        # Check for orphan labels (label without image)
        image_stems = {p.stem for p in image_paths}
        for stem, lp in label_paths.items():
            if stem not in image_stems:
                stats.orphan_labels += 1
                self.report.issues.append(asdict(LabelIssue(
                    image_path="N/A",
                    label_path=str(lp),
                    issue_type="ORPHAN_LABEL",
                    detail="Label file has no corresponding image",
                    severity="WARNING",
                )))

        for img_path in image_paths:
            # --- Image validity ---
            if not self._check_image(img_path, stats):
                continue

            # --- Duplicate detection ---
            self._check_duplicate(img_path, all_image_hashes, stats)

            # --- Label correspondence ---
            label_path = labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                stats.missing_labels += 1
                self.report.issues.append(asdict(LabelIssue(
                    image_path=str(img_path),
                    label_path=str(label_path),
                    issue_type="MISSING",
                    detail="No label file for this image",
                    severity="WARNING" if self.allow_empty_labels else "ERROR",
                )))
                stats.negative_images += 1
                continue

            # --- Label content ---
            issues = self.validate_single_label(label_path, img_path)
            error_issues = [i for i in issues if i.severity == "ERROR"]

            if error_issues:
                stats.invalid_label_files += 1
                for iss in issues:
                    self.report.issues.append(asdict(iss))
            else:
                # Count boxes
                lines = label_path.read_text().strip().splitlines()
                valid_lines = [l for l in lines if l.strip()]
                box_count = len(valid_lines)
                stats.total_boxes += box_count

                # Track per-class counts
                for line in valid_lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls = int(parts[0])
                            self.report.boxes_by_class[cls] = (
                                self.report.boxes_by_class.get(cls, 0) + 1
                            )
                        except ValueError:
                            pass

                if box_count > 0:
                    stats.positive_images += 1
                else:
                    stats.negative_images += 1

                # Add warnings even for otherwise valid labels
                for iss in issues:
                    if iss.severity == "WARNING":
                        self.report.issues.append(asdict(iss))

        stats.total_labels = len(label_paths)
        return stats

    def _check_image(self, img_path: Path, stats: SplitStats) -> bool:
        """
        Check image readability and basic validity.

        Returns:
            True if image is usable, False if corrupted/unreadable.
        """
        if img_path.stat().st_size == 0:
            stats.corrupted_images += 1
            self.report.issues.append(asdict(LabelIssue(
                image_path=str(img_path),
                label_path="N/A",
                issue_type="CORRUPTED",
                detail="Zero-byte file",
                severity="ERROR",
            )))
            return False

        img = cv2.imread(str(img_path))
        if img is None:
            stats.corrupted_images += 1
            self.report.issues.append(asdict(LabelIssue(
                image_path=str(img_path),
                label_path="N/A",
                issue_type="CORRUPTED",
                detail="cv2 cannot open image",
                severity="ERROR",
            )))
            return False

        return True

    def _check_duplicate(
        self,
        img_path: Path,
        all_hashes: Dict[str, str],
        stats: SplitStats,
    ) -> None:
        """Detect exact duplicate files via MD5 hash."""
        from src.utils.file_utils import compute_file_hash
        file_hash = compute_file_hash(img_path)
        if not file_hash:
            return

        if file_hash in all_hashes:
            self.report.duplicate_images += 1
            self.report.issues.append(asdict(LabelIssue(
                image_path=str(img_path),
                label_path="N/A",
                issue_type="DUPLICATE",
                detail=f"Exact duplicate of {all_hashes[file_hash]}",
                severity="WARNING",
            )))
        else:
            all_hashes[file_hash] = str(img_path)

    def _validate_label_line(
        self,
        line: str,
        line_no: int,
        label_path: Path,
        image_path: Optional[Path],
        seen_boxes: set,
    ) -> List[LabelIssue]:
        """Validate a single line in a YOLO label file."""
        issues = []
        stripped = line.strip()

        if not stripped:
            return issues

        parts = stripped.split()

        # Must have exactly 5 fields
        if len(parts) != 5:
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="MALFORMED",
                line_number=line_no,
                raw_line=stripped,
                detail=f"Expected 5 fields, got {len(parts)}",
                severity="ERROR",
            ))
            return issues

        # Parse fields
        try:
            cls_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError as e:
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="MALFORMED",
                line_number=line_no,
                raw_line=stripped,
                detail=f"Non-numeric value: {e}",
                severity="ERROR",
            ))
            return issues

        # Check NaN/Inf
        for val, name in [(xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")]:
            if not np.isfinite(val):
                issues.append(LabelIssue(
                    image_path=str(image_path or "unknown"),
                    label_path=str(label_path),
                    issue_type="MALFORMED",
                    line_number=line_no,
                    raw_line=stripped,
                    detail=f"{name} is NaN or Inf: {val}",
                    severity="ERROR",
                ))
                return issues

        # Unknown class
        if cls_id not in VALID_CLASS_IDS:
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="UNKNOWN_CLASS",
                line_number=line_no,
                raw_line=stripped,
                detail=f"Class ID {cls_id} not in valid set {VALID_CLASS_IDS}",
                severity="ERROR",
            ))

        # Coordinate range check
        out_of_range = False
        if not (0.0 <= xc <= 1.0):
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="OUT_OF_RANGE",
                line_number=line_no,
                raw_line=stripped,
                detail=f"x_center={xc:.6f} outside [0, 1]",
                severity="ERROR",
            ))
            out_of_range = True

        if not (0.0 <= yc <= 1.0):
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="OUT_OF_RANGE",
                line_number=line_no,
                raw_line=stripped,
                detail=f"y_center={yc:.6f} outside [0, 1]",
                severity="ERROR",
            ))
            out_of_range = True

        if not (0.0 < w <= 1.0):
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="OUT_OF_RANGE",
                line_number=line_no,
                raw_line=stripped,
                detail=f"width={w:.6f} outside (0, 1]",
                severity="ERROR",
            ))
            out_of_range = True

        if not (0.0 < h <= 1.0):
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="OUT_OF_RANGE",
                line_number=line_no,
                raw_line=stripped,
                detail=f"height={h:.6f} outside (0, 1]",
                severity="ERROR",
            ))
            out_of_range = True

        if out_of_range:
            self.report.out_of_range_boxes += 1

        # Zero-area box
        if w <= 0 or h <= 0:
            self.report.zero_area_boxes += 1
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="ZERO_AREA",
                line_number=line_no,
                raw_line=stripped,
                detail=f"width={w:.6f}, height={h:.6f}",
                severity="ERROR",
            ))

        # Duplicate annotation within same label file
        box_key = (cls_id, round(xc, 6), round(yc, 6), round(w, 6), round(h, 6))
        if box_key in seen_boxes:
            self.report.duplicate_annotations += 1
            issues.append(LabelIssue(
                image_path=str(image_path or "unknown"),
                label_path=str(label_path),
                issue_type="DUPLICATE_ANNOTATION",
                line_number=line_no,
                raw_line=stripped,
                detail="Duplicate bounding box within label file",
                severity="WARNING",
            ))
        else:
            seen_boxes.add(box_key)

        return issues

    def _determine_status(self) -> str:
        """
        Determine final readiness status.

        Rules:
        - Any critical_errors → NOT_READY
        - Any ERROR-severity issues → NOT_READY
        - Otherwise → READY_FOR_TRAINING
        """
        if self.report.critical_errors:
            return "NOT_READY_FOR_TRAINING"

        error_issues = [
            i for i in self.report.issues
            if i.get("severity") == "ERROR"
        ]
        if error_issues:
            return "NOT_READY_FOR_TRAINING"

        if self.report.total_images == 0:
            return "NOT_READY_FOR_TRAINING"

        return "READY_FOR_TRAINING"

    def _print_summary(self) -> None:
        """Print human-readable validation summary."""
        r = self.report
        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total images       : {r.total_images}")
        logger.info(f"  Total labels       : {r.total_labels}")
        logger.info(f"  Positive images    : {r.positive_images}")
        logger.info(f"  Negative images    : {r.negative_images}")
        logger.info(f"  Total boxes        : {r.total_boxes}")
        logger.info(f"  Missing labels     : {r.missing_labels}")
        logger.info(f"  Orphan labels      : {r.orphan_labels}")
        logger.info(f"  Corrupted images   : {r.corrupted_images}")
        logger.info(f"  Invalid labels     : {r.invalid_label_files}")
        logger.info(f"  Out-of-range boxes : {r.out_of_range_boxes}")
        logger.info(f"  Zero-area boxes    : {r.zero_area_boxes}")
        logger.info(f"  Duplicate images   : {r.duplicate_images}")
        logger.info(f"  Duplicate annots   : {r.duplicate_annotations}")
        logger.info(f"  Issues logged      : {len(r.issues)}")
        logger.info("")

        for split, stats in r.split_stats.items():
            logger.info(
                f"  [{split:5s}] "
                f"images={stats['total_images']:5d}  "
                f"pos={stats['positive_images']:5d}  "
                f"neg={stats['negative_images']:5d}  "
                f"boxes={stats['total_boxes']:6d}"
            )

        logger.info("")
        logger.info(f"  STATUS: {r.status}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO-format dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed",
        help="Path to processed dataset root (default: data/processed)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory to save validation report (default: reports)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat empty label files as errors (no negative samples allowed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validator = DatasetValidator(
        processed_dir=Path(args.data),
        allow_empty_labels=not args.strict,
        report_dir=Path(args.report_dir),
    )
    report = validator.validate()
    exit(0 if report.status == "READY_FOR_TRAINING" else 1)