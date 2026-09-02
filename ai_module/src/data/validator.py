"""
Final dataset validator — the last gate before training.

Extends the original file-level checks with the checks explicitly
required by the audit spec but previously MISSING:
    - patient/study leakage across splits (via canonical manifest)
    - exact-hash content leakage across splits
    - dataset.yaml vs actual computed statistics consistency
    - train/val/test ratio sanity check
    - annotation_status distribution (flags negative_from_invalid_boxes)

TRAINING_STATUS is only READY when ALL of these pass.
"""
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from src.utils.logger import get_logger
from src.utils.file_utils import save_json, IMAGE_EXTENSIONS, compute_file_hash
from src.data.manifest import ManifestStore, UNAVAILABLE
from src.utils.image_utils import check_image_integrity


logger = get_logger(__name__)

VALID_CLASS_IDS = {0}
SPLITS = ["train", "val", "test"]
SPLIT_RATIO_TOLERANCE = 0.05  # allow +/-5% deviation from target ratios (group splits won't be exact)
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_ACCEPTABLE_INVALID_ANNOTATION_RATIO = 0.02  # 2% negative_from_invalid_boxes triggers concern

NOT_CHECKED = "NOT_CHECKED_DUE_TO_PRIOR_ERROR"


@dataclass
class LabelIssue:
    image_path: str
    label_path: str
    issue_type: str
    line_number: Optional[int] = None
    raw_line: Optional[str] = None
    detail: Optional[str] = None
    severity: str = "ERROR"


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
    # --- NEW fields ---

    # NEW fields — explicitly typed as "not checked yet" by default, never {}
    patient_leakage: object = field(default_factory=lambda: NOT_CHECKED)
    hash_leakage: object = field(default_factory=lambda: NOT_CHECKED)
    dataset_yaml_consistency: object = field(default_factory=lambda: NOT_CHECKED)
    split_ratio_check: object = field(default_factory=lambda: NOT_CHECKED)
    annotation_status_distribution: object = field(default_factory=lambda: NOT_CHECKED)
    manifest_summary: object = field(default_factory=lambda: NOT_CHECKED)


class DatasetValidator:
    def __init__(
        self,
        processed_dir: Path,
        allow_empty_labels: bool = True,
        report_dir: Optional[Path] = None,
        dataset_yaml_path: Optional[Path] = None,
        skip_integrity_recheck: bool = False,  # NEW
    ):
        
        self.processed_dir = Path(processed_dir)
        self.allow_empty_labels = allow_empty_labels
        self.skip_integrity_recheck = skip_integrity_recheck  # NEW
        self.report_dir = Path(report_dir) if report_dir else self.processed_dir.parent.parent / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_yaml_path = (
            Path(dataset_yaml_path) if dataset_yaml_path
            else self.processed_dir.parent / "configs" / "dataset.yaml"
        )
        self.report = ValidationReport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, save_report: bool = True) -> ValidationReport:
        logger.info("=" * 60)
        logger.info("Starting dataset validation")
        logger.info(f"Dataset root: {self.processed_dir}")
        logger.info("=" * 60)

        all_image_hashes: Dict[str, str] = {}

        for split in SPLITS:
            split_dir = self.processed_dir / split
            if not split_dir.exists():
                self.report.critical_errors.append(f"Split directory missing: {split_dir}")
                continue
            stats = self._validate_split(split, split_dir, all_image_hashes)
            self.report.split_stats[split] = asdict(stats)
            self.report.total_images += stats.total_images
            self.report.total_labels += stats.total_labels
            self.report.positive_images += stats.positive_images
            self.report.negative_images += stats.negative_images
            self.report.total_boxes += stats.total_boxes
            self.report.missing_labels += stats.missing_labels
            self.report.orphan_labels += stats.orphan_labels
            self.report.corrupted_images += stats.corrupted_images
            self.report.invalid_label_files += stats.invalid_label_files

        # --- NEW: manifest-based checks (leakage + consistency) ---
        self._validate_manifest_and_leakage()
        self._validate_against_dataset_yaml()
        self._validate_split_ratios()

        self.report.status = self._determine_status()

        if save_report:
            save_json(asdict(self.report), self.report_dir / "validation_report.json")

        self._print_summary()
        return self.report

    # ------------------------------------------------------------------
    # NEW: manifest / leakage / consistency checks
    # ------------------------------------------------------------------

    def _validate_manifest_and_leakage(self) -> None:
        manifest_path = self.processed_dir / "manifest.csv"
        if not manifest_path.exists():
            self.report.critical_errors.append(
                "manifest.csv not found — cannot verify patient/study leakage. "
                "TRAINING MUST BE BLOCKED."
            )
            self.report.patient_leakage = NOT_CHECKED
            self.report.hash_leakage = NOT_CHECKED
            self.report.manifest_summary = NOT_CHECKED
            self.report.annotation_status_distribution = NOT_CHECKED
            return

        try:
            store = ManifestStore.load(manifest_path)
        except Exception as e:
            self.report.critical_errors.append(f"manifest.csv failed to load: {e}")
            self.report.patient_leakage = NOT_CHECKED
            self.report.hash_leakage = NOT_CHECKED
            self.report.manifest_summary = NOT_CHECKED
            self.report.annotation_status_distribution = NOT_CHECKED
            return

        self.report.manifest_summary = store.summary()

        patient_leak = store.check_patient_leakage()
        self.report.patient_leakage = patient_leak
        if patient_leak["train_val_overlap"] or patient_leak["train_test_overlap"] or patient_leak["val_test_overlap"]:
            self.report.critical_errors.append(
                f"PATIENT-LEVEL LEAKAGE DETECTED: {patient_leak}. TRAINING BLOCKED."
            )
        if patient_leak.get("datasets_with_unknown_patient_id"):
            self.report.warnings.append(
                f"Datasets with UNVERIFIABLE patient_id (leakage cannot be proven, only "
                f"assumed absent): {patient_leak['datasets_with_unknown_patient_id']}"
            )

        hash_leak = store.check_hash_leakage()
        self.report.hash_leakage = hash_leak
        if any(hash_leak.values()):
            self.report.critical_errors.append(
                f"EXACT-DUPLICATE IMAGE CONTENT LEAKS ACROSS SPLITS: {hash_leak}. TRAINING BLOCKED."
            )

        status_dist = self.report.manifest_summary.get("annotation_status_distribution", {})
        self.report.annotation_status_distribution = status_dist
        bad = status_dist.get("negative_from_invalid_boxes", 0)
        total = sum(status_dist.values()) or 1
        ratio = bad / total
        if ratio > MAX_ACCEPTABLE_INVALID_ANNOTATION_RATIO:
            self.report.critical_errors.append(
                f"{bad} samples ({ratio:.2%}) are labeled negative ONLY because their "
                f"original fracture annotation was geometrically invalid. Exceeds "
                f"{MAX_ACCEPTABLE_INVALID_ANNOTATION_RATIO:.0%} tolerance. TRAINING BLOCKED."
            )
        elif bad > 0:
            self.report.warnings.append(
                f"{bad} samples ({ratio:.2%}) are 'negative_from_invalid_boxes' — "
                f"within tolerance but should be manually spot-checked."
            )



    def _validate_against_dataset_yaml(self) -> None:
        if not self.dataset_yaml_path.exists():
            self.report.warnings.append(
                f"dataset.yaml not found at {self.dataset_yaml_path} — cannot cross-check."
            )
            return
        with open(self.dataset_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        stats = cfg.get("stats", {})

        mismatches = {}
        if stats.get("total_images") != self.report.total_images:
            mismatches["total_images"] = {
                "dataset_yaml": stats.get("total_images"),
                "actual_computed": self.report.total_images,
            }
        if stats.get("fracture_positive") != self.report.positive_images:
            mismatches["fracture_positive"] = {
                "dataset_yaml": stats.get("fracture_positive"),
                "actual_computed": self.report.positive_images,
            }

        self.report.dataset_yaml_consistency = {
            "checked": True,
            "mismatches": mismatches,
            "is_consistent": not mismatches,
        }
        if mismatches:
            self.report.critical_errors.append(
                f"dataset.yaml does NOT match actual processed dataset: {mismatches}. "
                f"TRAINING BLOCKED — regenerate dataset.yaml via prepare_dataset.py."
            )

    def _validate_split_ratios(self) -> None:
        total = self.report.total_images
        if total == 0:
            return
        result = {}
        for split, target in TARGET_RATIOS.items():
            actual = self.report.split_stats.get(split, {}).get("total_images", 0) / total
            deviation = abs(actual - target)
            result[split] = {
                "target": target, "actual": round(actual, 4),
                "deviation": round(deviation, 4),
                "within_tolerance": deviation <= SPLIT_RATIO_TOLERANCE,
            }
            if deviation > SPLIT_RATIO_TOLERANCE:
                self.report.warnings.append(
                    f"Split '{split}' ratio {actual:.1%} deviates from target {target:.0%} "
                    f"by more than {SPLIT_RATIO_TOLERANCE:.0%} (expected with group-level "
                    f"splitting — verify this is intentional, not a bug)."
                )
        self.report.split_ratio_check = result

    # ------------------------------------------------------------------
    # Existing per-file checks (kept, unchanged in spirit)
    # ------------------------------------------------------------------

    def validate_single_label(self, label_path: Path, image_path: Optional[Path] = None) -> List[LabelIssue]:
        issues = []
        if not label_path.exists():
            issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "MISSING",
                                      detail="Label file does not exist", severity="ERROR"))
            return issues
        try:
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        except Exception as e:
            issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "MALFORMED",
                                      detail=f"Cannot read label file: {e}", severity="ERROR"))
            return issues
        if not lines:
            return issues
        seen_boxes = set()
        for line_no, line in enumerate(lines, start=1):
            issues.extend(self._validate_label_line(line, line_no, label_path, image_path, seen_boxes))
        return issues

    def _validate_split(self, split: str, split_dir: Path, all_image_hashes: Dict[str, str]) -> SplitStats:
        stats = SplitStats(split=split)
        images_dir, labels_dir = split_dir / "images", split_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            self.report.critical_errors.append(f"images/ or labels/ missing in split '{split}'")
            return stats

        image_paths = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
        label_paths = {p.stem: p for p in labels_dir.glob("*.txt")}
        stats.total_images = len(image_paths)

        image_stems = {p.stem for p in image_paths}
        for stem, lp in label_paths.items():
            if stem not in image_stems:
                stats.orphan_labels += 1
                self.report.issues.append(asdict(LabelIssue(
                    "N/A", str(lp), "ORPHAN_LABEL",
                    detail="Label file has no corresponding image", severity="WARNING",
                )))

        for img_path in image_paths:
            if not self._check_image(img_path, stats):
                continue
            self._check_duplicate(img_path, all_image_hashes, stats)

            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                # NOTE: by pipeline design, EVERY valid sample gets a label file
                # (even empty ones for negatives). A missing label here means a
                # real bug, not a legitimate negative — treat as ERROR.
                stats.missing_labels += 1
                self.report.issues.append(asdict(LabelIssue(
                    str(img_path), str(label_path), "MISSING",
                    detail="No label file — pipeline should always write one (even if empty)",
                    severity="ERROR",
                )))
                continue

            issues = self.validate_single_label(label_path, img_path)
            error_issues = [i for i in issues if i.severity == "ERROR"]
            if error_issues:
                stats.invalid_label_files += 1
                for iss in issues:
                    self.report.issues.append(asdict(iss))
            else:
                lines = [l for l in label_path.read_text().strip().splitlines() if l.strip()]
                stats.total_boxes += len(lines)
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls = int(parts[0])
                            self.report.boxes_by_class[cls] = self.report.boxes_by_class.get(cls, 0) + 1
                        except ValueError:
                            pass
                if lines:
                    stats.positive_images += 1
                else:
                    stats.negative_images += 1
                for iss in issues:
                    if iss.severity == "WARNING":
                        self.report.issues.append(asdict(iss))

        stats.total_labels = len(label_paths)
        return stats

    def _check_image(self, img_path: Path, stats: SplitStats) -> bool:
        """
        Uses the deep integrity check by default (catches truncated JPEGs
        that cv2.imread() silently tolerates). This is slower but
        authoritative — see src/utils/image_utils.check_image_integrity.

        Use --skip-integrity-recheck for faster iteration ONLY when you
        already trust these images were verified during prepare_dataset.py
        (e.g. re-running validator just to check labels/manifest after a
        clean prepare run).
        """
        if self.skip_integrity_recheck:
            if img_path.stat().st_size == 0:
                stats.corrupted_images += 1
                self.report.issues.append(asdict(LabelIssue(
                    str(img_path), "N/A", "CORRUPTED", detail="Zero-byte file", severity="ERROR")))
                return False
            return True

        is_ok, status = check_image_integrity(img_path)
        if not is_ok:
            stats.corrupted_images += 1
            self.report.issues.append(asdict(LabelIssue(
                str(img_path), "N/A", "CORRUPTED", detail=status, severity="ERROR")))
            return False
        return True

    
    def _check_duplicate(self, img_path: Path, all_hashes: Dict[str, str], stats: SplitStats) -> None:
        file_hash = compute_file_hash(img_path)
        if not file_hash:
            return
        if file_hash in all_hashes:
            self.report.duplicate_images += 1
            self.report.issues.append(asdict(LabelIssue(
                str(img_path), "N/A", "DUPLICATE",
                detail=f"Exact duplicate of {all_hashes[file_hash]}", severity="WARNING")))
        else:
            all_hashes[file_hash] = str(img_path)

    def _validate_label_line(self, line, line_no, label_path, image_path, seen_boxes) -> List[LabelIssue]:
        issues = []
        stripped = line.strip()
        if not stripped:
            return issues
        parts = stripped.split()
        if len(parts) != 5:
            issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "MALFORMED",
                                      line_no, stripped, f"Expected 5 fields, got {len(parts)}", "ERROR"))
            return issues
        try:
            cls_id, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError as e:
            issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "MALFORMED",
                                      line_no, stripped, f"Non-numeric value: {e}", "ERROR"))
            return issues
        for val, name in [(xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")]:
            if not np.isfinite(val):
                issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "MALFORMED",
                                          line_no, stripped, f"{name} is NaN or Inf: {val}", "ERROR"))
                return issues
        if cls_id not in VALID_CLASS_IDS:
            issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "UNKNOWN_CLASS",
                                      line_no, stripped, f"Class ID {cls_id} not in {VALID_CLASS_IDS}", "ERROR"))
        out_of_range = False
        for val, lo, hi, name in [(xc, 0.0, 1.0, "x_center"), (yc, 0.0, 1.0, "y_center")]:
            if not (lo <= val <= hi):
                issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "OUT_OF_RANGE",
                                          line_no, stripped, f"{name}={val:.6f} outside [0,1]", "ERROR"))
                out_of_range = True
        for val, name in [(w, "width"), (h, "height")]:
            if not (0.0 < val <= 1.0):
                issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "OUT_OF_RANGE",
                                          line_no, stripped, f"{name}={val:.6f} outside (0,1]", "ERROR"))
                out_of_range = True
        if out_of_range:
            self.report.out_of_range_boxes += 1
        if w <= 0 or h <= 0:
            self.report.zero_area_boxes += 1
            issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "ZERO_AREA",
                                      line_no, stripped, f"width={w:.6f}, height={h:.6f}", "ERROR"))
        box_key = (cls_id, round(xc, 6), round(yc, 6), round(w, 6), round(h, 6))
        if box_key in seen_boxes:
            self.report.duplicate_annotations += 1
            issues.append(LabelIssue(str(image_path or "unknown"), str(label_path), "DUPLICATE_ANNOTATION",
                                      line_no, stripped, "Duplicate bbox within label file", "WARNING"))
        else:
            seen_boxes.add(box_key)
        return issues

    def _determine_status(self) -> str:
        if self.report.critical_errors:
            return "NOT_READY_FOR_TRAINING"
        if any(i.get("severity") == "ERROR" for i in self.report.issues):
            return "NOT_READY_FOR_TRAINING"
        if self.report.total_images == 0:
            return "NOT_READY_FOR_TRAINING"
        return "READY_FOR_TRAINING"

    def _print_summary(self) -> None:
        r = self.report

        def _fmt(value):
            return "⏭️  NOT CHECKED (blocked earlier — see critical_errors)" if value == NOT_CHECKED else value

        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Total images        : {r.total_images}")
        logger.info(f"  Positive / Negative : {r.positive_images} / {r.negative_images}")
        logger.info(f"  Total boxes         : {r.total_boxes}")
        logger.info(f"  Missing labels      : {r.missing_labels}")
        logger.info(f"  Orphan labels       : {r.orphan_labels}")
        logger.info(f"  Corrupted images    : {r.corrupted_images}")
        logger.info(f"  Out-of-range boxes  : {r.out_of_range_boxes}")
        logger.info(f"  Zero-area boxes     : {r.zero_area_boxes}")
        logger.info(f"  Duplicate images    : {r.duplicate_images}")
        logger.info(f"  Patient leakage     : {_fmt(r.patient_leakage)}")
        logger.info(f"  Hash leakage        : {_fmt(r.hash_leakage)}")
        logger.info(f"  dataset.yaml match  : {_fmt(r.dataset_yaml_consistency)}")
        logger.info(f"  Annotation status   : {_fmt(r.annotation_status_distribution)}")
        logger.info(f"  Critical errors     : {len(r.critical_errors)}")
        for ce in r.critical_errors:
            logger.error(f"    ❌ {ce}")
        for w in r.warnings:
            logger.warning(f"    ⚠️  {w}")
        logger.info("")
        logger.info(f"  STATUS: {r.status}")
        logger.info("=" * 60)


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO-format dataset")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--report-dir", type=str, default="reports")
    parser.add_argument("--dataset-yaml", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--skip-integrity-recheck", action="store_true",
        help="Skip slow deep image decode check (only use if prepare_dataset.py "
             "already verified these exact files' integrity in this run)."
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = _parse_args()
    validator = DatasetValidator(
        processed_dir=Path(args.data),
        allow_empty_labels=not args.strict,
        report_dir=Path(args.report_dir),
        dataset_yaml_path=Path(args.dataset_yaml) if args.dataset_yaml else None,
        skip_integrity_recheck=args.skip_integrity_recheck,
    )
    report = validator.validate()
    exit(0 if report.status == "READY_FOR_TRAINING" else 1)
