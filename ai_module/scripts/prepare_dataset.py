"""
Dataset preparation pipeline for Phase 1.

Actual dataset structures discovered via inspect_datasets.py:

FracAtlas:
    fracatlas/FracAtlas/Annotations/YOLO/          ← YOLO labels
    fracatlas/FracAtlas/images/Fractured/           ← positive images
    fracatlas/FracAtlas/images/Non_fractured/       ← negative images
    fracatlas/FracAtlas/dataset.csv

GRAZPEDWRI-DX:
    GRAZPEDWRI-DX/folder_structure/pascalvoc/      ← canonical VOC annotations
    GRAZPEDWRI-DX/folder_structure/yolov5/labels/  ← multi-class YOLO (audit only)
    GRAZPEDWRI-DX/folder_structure/supervisely/    ← audit only
    GRAZPEDWRI-DX/images_part1..4/                 ← images

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --dry-run --verbose
    python scripts/prepare_dataset.py --source fracatlas
"""

import argparse
import csv
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.file_utils import (
    compute_file_hash,
    find_images,
    save_json,
    IMAGE_EXTENSIONS,
)
from src.utils.image_utils import get_image_dimensions

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Class 0 = fracture only
FRACTURE_CLASS_NAMES = {
    "fracture", "Fracture", "FRACTURE",
    "bone fracture", "Bone Fracture",
    "fraktur", "fraktura",
}

IGNORED_CLASS_NAMES = {
    "text", "Text", "TEXT",
    "ruler", "scale", "marker",
    "artifact", "Artifact",
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_project_root() -> Path:
    """
    Walk up from this script to find the directory containing
    both ai_module/ and at least one dataset directory.
    """
    current = Path(__file__).resolve()
    for parent in list(current.parents)[:6]:
        has_ai = (parent / "ai_module").exists()
        has_data = any([
            (parent / "fracatlas").exists(),
            (parent / "FracAtlas").exists(),
            (parent / "GRAZPEDWRI-DX").exists(),
        ])
        if has_ai and has_data:
            logger.info(f"Project root detected: {parent}")
            return parent
    fallback = current.parent.parent.parent
    logger.warning(f"Could not auto-detect project root. Using: {fallback}")
    return fallback


def resolve_dataset_path(raw: str, project_root: Path, ai_root: Path) -> Path:
    """Resolve a path from config, trying multiple base directories."""
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in [ai_root, project_root]:
        candidate = (base / raw).resolve()
        if candidate.exists():
            return candidate
    return (ai_root / raw).resolve()


# ---------------------------------------------------------------------------
# Manifest row
# ---------------------------------------------------------------------------

class ManifestRow:
    __slots__ = [
        "sample_id", "source_dataset", "original_filename",
        "processed_filename", "split", "fracture_positive",
        "num_boxes", "annotation_source", "patient_id",
        "study_id", "width", "height",
    ]

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k, ""))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


# ---------------------------------------------------------------------------
# YOLO annotation validator
# ---------------------------------------------------------------------------

class YOLOAnnotationValidator:
    """Validates a single YOLO label line."""

    def validate_line(self, line: str) -> Tuple[bool, Optional[list], str]:
        parts = line.strip().split()
        if len(parts) != 5:
            return False, None, f"Expected 5 fields, got {len(parts)}"
        try:
            cls  = int(parts[0])
            xc   = float(parts[1])
            yc   = float(parts[2])
            w    = float(parts[3])
            h    = float(parts[4])
        except ValueError as e:
            return False, None, f"Parse error: {e}"

        if not all(np.isfinite(v) for v in [xc, yc, w, h]):
            return False, None, "Contains NaN or Inf"
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            return False, None, f"x_center={xc:.4f} or y_center={yc:.4f} out of [0,1]"
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return False, None, f"width={w:.4f} or height={h:.4f} not in (0,1]"
        return True, [cls, xc, yc, w, h], "OK"


# ---------------------------------------------------------------------------
# Pascal VOC converter
# ---------------------------------------------------------------------------

class PascalVOCConverter:
    """Converts Pascal VOC XML objects to YOLO format."""

    def __init__(self, fracture_names: set, ignored_names: set):
        self.fracture_names = fracture_names
        self.ignored_names  = ignored_names

    def convert(
        self,
        xml_path: Path,
        image_width: int,
        image_height: int,
    ) -> Tuple[List[list], List[dict]]:
        yolo_boxes = []
        issues     = []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            return [], [{"type": "PARSE_ERROR", "detail": str(e)}]

        # Use annotation size if image dims not provided
        size_el = root.find("size")
        if size_el is not None:
            try:
                aw = int(size_el.findtext("width",  "0"))
                ah = int(size_el.findtext("height", "0"))
                if aw > 0 and ah > 0:
                    image_width  = aw
                    image_height = ah
            except ValueError:
                pass

        for obj in root.findall("object"):
            name    = obj.findtext("name", "").strip()
            bndbox  = obj.find("bndbox")

            if bndbox is None:
                issues.append({"type": "MISSING_BNDBOX", "class_name": name})
                continue

            try:
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))
            except ValueError as e:
                issues.append({"type": "INVALID_COORDS", "class_name": name, "detail": str(e)})
                continue

            if name in self.fracture_names:
                box = self._convert_box(xmin, ymin, xmax, ymax, image_width, image_height)
                if box is not None:
                    yolo_boxes.append([0] + box)
                else:
                    issues.append({
                        "type": "INVALID_BOX", "class_name": name,
                        "detail": f"xmin={xmin},ymin={ymin},xmax={xmax},ymax={ymax}",
                    })
            elif name in self.ignored_names:
                issues.append({"type": "IGNORED_CLASS", "class_name": name})
            else:
                issues.append({"type": "UNKNOWN_CLASS", "class_name": name})

        return yolo_boxes, issues

    @staticmethod
    def _convert_box(
        xmin: float, ymin: float, xmax: float, ymax: float,
        img_w: int, img_h: int,
    ) -> Optional[list]:
        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 0 or bh <= 0:
            return None

        xc = (xmin + xmax) / 2.0 / img_w
        yc = (ymin + ymax) / 2.0 / img_h
        w  = bw / img_w
        h  = bh / img_h

        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return None
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            return None
        return [xc, yc, w, h]


# ---------------------------------------------------------------------------
# FracAtlas processor
# ---------------------------------------------------------------------------

class FracAtlasProcessor:
    """
    Processes FracAtlas dataset.

    Actual structure:
        fracatlas/
            FracAtlas/
                Annotations/YOLO/           ← existing YOLO labels (validated)
                images/
                    Fractured/              ← positive images (717)
                    Non_fractured/          ← negative images (3366)
                dataset.csv

    Strategy:
        1. Use existing YOLO labels from Annotations/YOLO/
        2. Validate every label line
        3. Remap class to 0 (fracture) explicitly
        4. Fractured/ images → positive samples
        5. Non_fractured/ images → negative samples (empty labels expected)
    """

    def __init__(self, raw_root: Path, verbose: bool = False):
        self.raw_root  = raw_root
        self.verbose   = verbose

        # Actual paths — one extra FracAtlas/ subdirectory
        self._inner    = raw_root / "FracAtlas"
        self.yolo_dir  = self._inner / "Annotations" / "YOLO"
        self.images_fractured     = self._inner / "images" / "Fractured"
        self.images_non_fractured = self._inner / "images" / "Non_fractured"
        self.csv_path  = self._inner / "dataset.csv"

        self.validator = YOLOAnnotationValidator()

    def process(self) -> List[dict]:
        """Process all FracAtlas samples. Returns list of sample dicts."""
        samples = []

        if not self.yolo_dir.exists():
            logger.error(f"FracAtlas YOLO dir not found: {self.yolo_dir}")
            return samples

        label_files = sorted(self.yolo_dir.glob("*.txt"))
        logger.info(
            f"FracAtlas: found {len(label_files)} YOLO label files "
            f"in {self.yolo_dir}"
        )

        # Build image index from BOTH Fractured/ and Non_fractured/
        image_index: Dict[str, Path] = {}
        for img_dir in [self.images_fractured, self.images_non_fractured]:
            if img_dir.exists():
                for img_path in find_images(img_dir):
                    image_index[img_path.stem] = img_path

        logger.info(
            f"FracAtlas: indexed {len(image_index)} images "
            f"(fractured + non_fractured)"
        )

        # Load CSV metadata
        metadata = self._load_csv()

        for label_path in label_files:
            stem       = label_path.stem
            image_path = image_index.get(stem)
            sample_meta = metadata.get(stem, {})

            # Read and validate label
            valid_lines = []
            issues      = []

            try:
                raw_text = label_path.read_text(encoding="utf-8").strip()
                raw_lines = [l for l in raw_text.splitlines() if l.strip()]
            except Exception as e:
                issues.append({"type": "READ_ERROR", "detail": str(e)})
                samples.append(self._make_sample(
                    stem, image_path, [], False, issues, sample_meta
                ))
                continue

            for line in raw_lines:
                is_valid, parsed, msg = self.validator.validate_line(line)
                if is_valid:
                    # Force class 0 = fracture
                    parsed[0] = 0
                    valid_lines.append(parsed)
                else:
                    issues.append({"type": "INVALID_LINE", "raw": line, "detail": msg})
                    if self.verbose:
                        logger.debug(f"FracAtlas {stem}: invalid line — {msg}")

            fracture_positive = len(valid_lines) > 0
            is_valid_sample   = image_path is not None

            samples.append(self._make_sample(
                stem, image_path, valid_lines, fracture_positive, issues, sample_meta
            ))

        # Count stats
        valid_count = sum(1 for s in samples if s["valid"])
        pos_count   = sum(1 for s in samples if s.get("fracture_positive", False))
        neg_count   = sum(1 for s in samples if not s.get("fracture_positive", False))

        logger.info(
            f"FracAtlas: processed {len(samples)} samples — "
            f"valid={valid_count}, positive={pos_count}, negative={neg_count}"
        )
        return samples

    def _make_sample(
        self,
        stem: str,
        image_path: Optional[Path],
        label_lines: list,
        fracture_positive: bool,
        issues: list,
        metadata: dict,
    ) -> dict:
        return {
            "source":            "fracatlas",
            "stem":              stem,
            "image_path":        image_path,
            "label_lines":       label_lines,
            "valid":             image_path is not None,
            "issues":            issues,
            "metadata":          metadata,
            "fracture_positive": fracture_positive,
            "annotation_source": "yolo_existing",
        }

    def _load_csv(self) -> Dict[str, dict]:
        """Load FracAtlas dataset.csv metadata."""
        result = {}
        if not self.csv_path.exists():
            return result
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = Path(row.get("image_id", "")).stem or str(
                        list(row.values())[0]
                    )
                    result[key] = dict(row)
            logger.info(f"FracAtlas: loaded {len(result)} CSV metadata rows")
        except Exception as e:
            logger.warning(f"FracAtlas: could not load dataset.csv — {e}")
        return result


# ---------------------------------------------------------------------------
# GRAZPEDWRI-DX processor
# ---------------------------------------------------------------------------

class GRAZPEDWRIProcessor:
    """
    Processes GRAZPEDWRI-DX dataset.

    Actual structure:
        GRAZPEDWRI-DX/
            folder_structure/
                pascalvoc/          ← canonical VOC annotations (20327 xml)
                yolov5/labels/      ← multi-class YOLO (audit only — classes 3,5,8...)
                supervisely/        ← audit only
            images_part1..4/        ← PNG images (20327 total)
            dataset.csv             ← patient/study metadata

    Strategy:
        - Pascal VOC → canonical annotation source (explicit class names)
        - yolov5/labels → inspect only (multi-class, NOT class-0-only)
        - fracture objects → class 0
        - text, ruler, etc. → ignored
        - Patient ID extracted from filename pattern: XXXX_XXXXXXXXXX_...
    """

    def __init__(self, raw_root: Path, verbose: bool = False):
        self.raw_root  = raw_root
        self.verbose   = verbose

        # Actual paths
        self.voc_dir        = raw_root / "folder_structure" / "pascalvoc"
        self.supervisely_dir = raw_root / "folder_structure" / "supervisely" / "wrist" / "ann"
        self.yolov5_dir     = raw_root / "folder_structure" / "yolov5" / "labels"
        self.csv_path       = raw_root / "dataset.csv"

        self.converter      = PascalVOCConverter(FRACTURE_CLASS_NAMES, IGNORED_CLASS_NAMES)

        # Build image index from all parts
        self._image_map: Dict[str, Path] = {}
        self._build_image_map()

    def _build_image_map(self) -> None:
        """Index all images from images_part1..4."""
        for child in sorted(self.raw_root.iterdir()):
            if child.is_dir() and child.name.lower().startswith("images"):
                for img_path in find_images(child):
                    stem = img_path.stem
                    if stem not in self._image_map:
                        self._image_map[stem] = img_path
        logger.info(
            f"GRAZPEDWRI-DX: indexed {len(self._image_map)} images "
            f"from images_part* directories"
        )

    def process(self) -> List[dict]:
        """Process all GRAZPEDWRI-DX samples."""
        samples = []

        if not self.voc_dir.exists():
            logger.error(f"GRAZPEDWRI-DX VOC dir not found: {self.voc_dir}")
            return samples

        xml_files = sorted(self.voc_dir.glob("*.xml"))
        logger.info(
            f"GRAZPEDWRI-DX: found {len(xml_files)} Pascal VOC XML files "
            f"in {self.voc_dir}"
        )

        # Load CSV metadata
        metadata = self._load_csv()

        # Audit yolov5 labels (for reference only — NOT used as training labels)
        yolov5_classes = self._audit_yolov5_labels()
        if yolov5_classes:
            logger.info(
                f"GRAZPEDWRI-DX yolov5 label class distribution: {dict(yolov5_classes)}"
            )
            logger.info(
                "GRAZPEDWRI-DX: yolov5 labels are multi-class (not class-0-only). "
                "Using Pascal VOC as canonical source."
            )

        unknown_classes: Dict[str, int] = defaultdict(int)

        for xml_path in xml_files:
            stem       = xml_path.stem
            image_path = self._image_map.get(stem)
            sample_meta = metadata.get(stem, {})

            # Add patient_id from filename pattern if not in CSV
            # GRAZPEDWRI pattern: XXXX_XXXXXXXXXX_NN_WRI-XX_XXXX
            if "patient_id" not in sample_meta:
                patient_id = self._extract_patient_id(stem)
                if patient_id:
                    sample_meta["patient_id"] = patient_id

            # Get image dimensions for accurate VOC conversion
            img_w, img_h = 0, 0
            if image_path is not None:
                dims = get_image_dimensions(image_path)
                if dims:
                    img_h, img_w = dims[0], dims[1]
            else:
                logger.debug(f"GRAZPEDWRI-DX: no image found for {stem}")

            yolo_boxes, issues = self.converter.convert(xml_path, img_w, img_h)

            # Track unknown classes
            for iss in issues:
                if iss["type"] == "UNKNOWN_CLASS":
                    unknown_classes[iss["class_name"]] += 1

            if self.verbose:
                non_trivial = [i for i in issues if i["type"] not in {"IGNORED_CLASS"}]
                if non_trivial:
                    logger.debug(f"GRAZPEDWRI-DX {stem}: {non_trivial}")

            samples.append({
                "source":            "grazpedwri",
                "stem":              stem,
                "image_path":        image_path,
                "label_lines":       yolo_boxes,
                "valid":             image_path is not None,
                "issues":            issues,
                "metadata":          sample_meta,
                "fracture_positive": len(yolo_boxes) > 0,
                "annotation_source": "pascal_voc",
            })

        if unknown_classes:
            logger.warning(
                f"GRAZPEDWRI-DX: unknown class names encountered: {dict(unknown_classes)}"
            )

        valid_count = sum(1 for s in samples if s["valid"])
        pos_count   = sum(1 for s in samples if s.get("fracture_positive", False))
        neg_count   = sum(1 for s in samples if not s.get("fracture_positive", False))

        logger.info(
            f"GRAZPEDWRI-DX: processed {len(samples)} samples — "
            f"valid={valid_count}, positive={pos_count}, negative={neg_count}"
        )
        return samples

    @staticmethod
    def _extract_patient_id(stem: str) -> Optional[str]:
        """
        Extract patient ID from GRAZPEDWRI filename.
        Pattern: XXXX_XXXXXXXXXX_NN_WRI-XX_XXXX
        Patient ID = first segment (XXXX).
        """
        parts = stem.split("_")
        if len(parts) >= 2:
            return parts[0]
        return None

    def _load_csv(self) -> Dict[str, dict]:
        """Load GRAZPEDWRI-DX dataset.csv metadata."""
        result = {}
        if not self.csv_path.exists():
            logger.warning(f"GRAZPEDWRI-DX: dataset.csv not found at {self.csv_path}")
            return result
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common filename columns
                    filename = (
                        row.get("filestem", "") or
                        row.get("filename", "") or
                        row.get("image_id", "") or
                        list(row.values())[0]
                    )
                    key = Path(filename).stem if filename else ""
                    if key:
                        result[key] = dict(row)
            logger.info(
                f"GRAZPEDWRI-DX: loaded {len(result)} CSV metadata rows"
            )
        except Exception as e:
            logger.warning(f"GRAZPEDWRI-DX: could not load dataset.csv — {e}")
        return result

    def _audit_yolov5_labels(self) -> Dict[str, int]:
        """
        Audit existing yolov5 labels (NOT used for training).
        Just to understand what classes they contain.
        Returns class_id → count.
        """
        class_counts: Dict[int, int] = defaultdict(int)
        if not self.yolov5_dir.exists():
            return {}
        sample_files = list(self.yolov5_dir.glob("*.txt"))[:50]
        for lp in sample_files:
            try:
                for line in lp.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cls = int(parts[0])
                            class_counts[cls] += 1
                        except ValueError:
                            pass
            except Exception:
                pass
        return dict(class_counts)


# ---------------------------------------------------------------------------
# Leakage-aware splitter
# ---------------------------------------------------------------------------

class LeakageAwareSplitter:
    """
    Splits samples into train/val/test.

    Priority:
    1. Group-level split on patient_id if available (≥80% coverage)
    2. Fall back to stratified image-level split

    For GRAZPEDWRI-DX, patient_id is extracted from the filename:
    XXXX_... → patient XXXX
    """

    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio:   float = VAL_RATIO,
        seed:        int   = RANDOM_SEED,
    ):
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.seed        = seed
        random.seed(seed)
        np.random.seed(seed)

    def split(self, samples: List[dict]) -> Dict[str, List[dict]]:
        group_field = self._detect_group_field(samples)

        if group_field:
            logger.info(f"Using group-level split on field: '{group_field}'")
            return self._group_split(samples, group_field)
        else:
            logger.info(
                "No reliable group field — using stratified image-level split"
            )
            return self._stratified_split(samples)

    def _detect_group_field(self, samples: List[dict]) -> Optional[str]:
        candidates = [
            "patient_id", "patientid", "patient",
            "study_id", "studyid",
        ]
        for field in candidates:
            values = [
                str(s.get("metadata", {}).get(field, "")).strip()
                for s in samples
            ]
            non_empty = [v for v in values if v and v not in {"", "nan", "None"}]
            coverage  = len(non_empty) / len(samples) if samples else 0
            if coverage >= 0.8:
                unique_count = len(set(non_empty))
                logger.info(
                    f"Group field '{field}': {unique_count} groups, "
                    f"{coverage:.1%} coverage → USABLE"
                )
                return field
            elif non_empty:
                logger.info(
                    f"Group field '{field}': {coverage:.1%} coverage → too low"
                )
        return None

    def _group_split(
        self, samples: List[dict], group_field: str
    ) -> Dict[str, List[dict]]:
        """Group-aware stratified split."""
        group_map: Dict[str, List[dict]] = defaultdict(list)
        for s in samples:
            gid = str(s.get("metadata", {}).get(group_field, f"_unk_{s['stem']}"))
            group_map[gid].append(s)

        positive_groups = []
        negative_groups = []
        for gid, group_samples in group_map.items():
            if any(s.get("fracture_positive", False) for s in group_samples):
                positive_groups.append(gid)
            else:
                negative_groups.append(gid)

        random.shuffle(positive_groups)
        random.shuffle(negative_groups)

        splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
        for group_list in [positive_groups, negative_groups]:
            n       = len(group_list)
            n_train = int(n * self.train_ratio)
            n_val   = int(n * self.val_ratio)

            for gid in group_list[:n_train]:
                splits["train"].extend(group_map[gid])
            for gid in group_list[n_train: n_train + n_val]:
                splits["val"].extend(group_map[gid])
            for gid in group_list[n_train + n_val:]:
                splits["test"].extend(group_map[gid])

        self._log_split_stats(splits)
        return splits

    def _stratified_split(
        self, samples: List[dict]
    ) -> Dict[str, List[dict]]:
        """Image-level stratified split."""
        positive = [s for s in samples if s.get("fracture_positive", False)]
        negative = [s for s in samples if not s.get("fracture_positive", False)]

        random.shuffle(positive)
        random.shuffle(negative)

        splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
        for group in [positive, negative]:
            n       = len(group)
            n_train = int(n * self.train_ratio)
            n_val   = int(n * self.val_ratio)
            splits["train"].extend(group[:n_train])
            splits["val"].extend(group[n_train: n_train + n_val])
            splits["test"].extend(group[n_train + n_val:])

        self._log_split_stats(splits)
        return splits

    @staticmethod
    def _log_split_stats(splits: Dict[str, List[dict]]) -> None:
        total = sum(len(v) for v in splits.values())
        for split, slist in splits.items():
            pos = sum(s.get("fracture_positive", False) for s in slist)
            pct = len(slist) / total * 100 if total else 0
            logger.info(
                f"  {split:6s}: {len(slist):6d} samples ({pct:5.1f}%) — "
                f"pos={pos}, neg={len(slist)-pos}"
            )


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------

class DatasetWriter:
    """Writes processed samples to data/processed/ in YOLO format."""

    def __init__(self, output_dir: Path, dry_run: bool = False):
        self.output_dir = output_dir
        self.dry_run    = dry_run
        self.manifest:  List[ManifestRow] = []
        self._name_counter: Dict[str, int] = defaultdict(int)

    def write(self, splits: Dict[str, List[dict]]) -> List[ManifestRow]:
        if not self.dry_run:
            self._create_dirs()

        sample_id = 0
        for split, samples in splits.items():
            images_dir = self.output_dir / split / "images"
            labels_dir = self.output_dir / split / "labels"

            for sample in samples:
                if not sample.get("valid", False):
                    continue

                img_src: Optional[Path] = sample.get("image_path")
                if img_src is None or not img_src.exists():
                    logger.debug(f"Skipping — image not found: {sample['stem']}")
                    continue

                safe_name = self._safe_filename(
                    sample["source"], img_src.stem, img_src.suffix
                )
                img_dst   = images_dir / safe_name
                label_dst = labels_dir / (Path(safe_name).stem + ".txt")

                if not self.dry_run:
                    shutil.copy2(img_src, img_dst)
                    self._write_label(label_dst, sample.get("label_lines", []))

                meta = sample.get("metadata", {})
                self.manifest.append(ManifestRow(
                    sample_id=str(sample_id),
                    source_dataset=sample["source"],
                    original_filename=img_src.name,
                    processed_filename=safe_name,
                    split=split,
                    fracture_positive=str(sample.get("fracture_positive", False)),
                    num_boxes=str(len(sample.get("label_lines", []))),
                    annotation_source=sample.get("annotation_source", "unknown"),
                    patient_id=meta.get("patient_id", ""),
                    study_id=meta.get("study_id", meta.get("studyid", "")),
                ))
                sample_id += 1

        logger.info(
            f"DatasetWriter: "
            f"{'[DRY RUN] ' if self.dry_run else ''}"
            f"wrote {sample_id} samples"
        )
        return self.manifest

    def save_manifest(self, manifest_path: Path) -> None:
        if self.dry_run:
            logger.info(f"[DRY RUN] Would save manifest: {manifest_path}")
            return
        if not self.manifest:
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(self.manifest[0].to_dict().keys())
            )
            writer.writeheader()
            for row in self.manifest:
                writer.writerow(row.to_dict())
        logger.info(
            f"Manifest saved: {manifest_path} ({len(self.manifest)} rows)"
        )

    def _create_dirs(self) -> None:
        for split in ("train", "val", "test"):
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output structure: {self.output_dir}")

    def _safe_filename(self, source: str, stem: str, suffix: str) -> str:
        prefix = {"fracatlas": "fa", "grazpedwri": "grz"}.get(source, source[:3])
        base   = f"{prefix}_{stem}{suffix}"
        if base not in self._name_counter:
            self._name_counter[base] = 0
            return base
        self._name_counter[base] += 1
        return f"{prefix}_{stem}_{self._name_counter[base]}{suffix}"

    @staticmethod
    def _write_label(label_path: Path, lines: List[list]) -> None:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            for box in lines:
                cls = int(box[0])
                xc, yc, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class DatasetPreparationPipeline:
    """Orchestrates Phase 1 dataset preparation."""

    def __init__(
        self,
        config_path:      Path,
        output_dir:       Path,
        reports_dir:      Path,
        fracatlas_root:   Optional[Path] = None,
        grazpedwri_root:  Optional[Path] = None,
        source:           Optional[str]  = None,
        dry_run:          bool           = False,
        verbose:          bool           = False,
        seed:             int            = RANDOM_SEED,
    ):
        self.config_path     = Path(config_path)
        self.output_dir      = Path(output_dir)
        self.reports_dir     = Path(reports_dir)
        self.source          = source
        self.dry_run         = dry_run
        self.verbose         = verbose
        self.seed            = seed
        self.config          = self._load_config()

        # Resolve roots
        self._project_root  = resolve_project_root()
        self._ai_module_root = Path(__file__).resolve().parent.parent

        logger.info(f"Project root  : {self._project_root}")
        logger.info(f"AI module root: {self._ai_module_root}")

        sources_cfg: Dict[str, str] = {
            s["name"]: s["local_path"]
            for s in self.config.get("sources", [])
        }

        if fracatlas_root is not None:
            self.fracatlas_root = Path(fracatlas_root)
        else:
            raw = sources_cfg.get("FracAtlas", "../fracatlas")
            self.fracatlas_root = resolve_dataset_path(
                raw, self._project_root, self._ai_module_root
            )

        if grazpedwri_root is not None:
            self.grazpedwri_root = Path(grazpedwri_root)
        else:
            raw = sources_cfg.get("GRAZPEDWRI-DX", "../GRAZPEDWRI-DX")
            self.grazpedwri_root = resolve_dataset_path(
                raw, self._project_root, self._ai_module_root
            )

        logger.info(f"FracAtlas root    : {self.fracatlas_root}")
        logger.info(f"FracAtlas exists  : {self.fracatlas_root.exists()}")
        logger.info(f"GRAZPEDWRI root   : {self.grazpedwri_root}")
        logger.info(f"GRAZPEDWRI exists : {self.grazpedwri_root.exists()}")

        self.report: Dict = {
            "pipeline":         "phase1_dataset_preparation",
            "seed":             seed,
            "dry_run":          dry_run,
            "project_root":     str(self._project_root),
            "fracatlas_root":   str(self.fracatlas_root),
            "grazpedwri_root":  str(self.grazpedwri_root),
            "sources":          {},
            "deduplication":    {},
            "splitting":        {},
            "writing":          {},
            "skipped":          [],
        }

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("Phase 1 — Dataset Preparation Pipeline")
        logger.info(f"Seed: {self.seed} | Dry-run: {self.dry_run}")
        logger.info("=" * 60)

        all_samples: List[dict] = []

        # Step 1 — FracAtlas
        if self.source in (None, "fracatlas"):
            fa_samples = self._run_fracatlas()
            all_samples.extend(fa_samples)
            self.report["sources"]["fracatlas"] = {
                "total":    len(fa_samples),
                "valid":    sum(s["valid"] for s in fa_samples),
                "positive": sum(s.get("fracture_positive", False) for s in fa_samples),
            }

        # Step 2 — GRAZPEDWRI-DX
        if self.source in (None, "grazpedwri"):
            grz_samples = self._run_grazpedwri()
            all_samples.extend(grz_samples)
            self.report["sources"]["grazpedwri"] = {
                "total":    len(grz_samples),
                "valid":    sum(s["valid"] for s in grz_samples),
                "positive": sum(s.get("fracture_positive", False) for s in grz_samples),
            }

        logger.info(f"Total samples before deduplication: {len(all_samples)}")

        # Step 3 — Deduplicate
        all_samples, dup_report = self._deduplicate(all_samples)
        self.report["deduplication"] = dup_report

        # Step 4 — Filter valid
        valid_samples = [s for s in all_samples if s.get("valid", False)]
        logger.info(
            f"Valid samples: {len(valid_samples)} | "
            f"Invalid/skipped: {len(all_samples) - len(valid_samples)}"
        )

        # Step 5 — Split
        splitter = LeakageAwareSplitter(seed=self.seed)
        splits   = splitter.split(valid_samples)
        self.report["splitting"] = {
            split: {
                "count":    len(sl),
                "positive": sum(s.get("fracture_positive", False) for s in sl),
                "negative": sum(not s.get("fracture_positive", False) for s in sl),
            }
            for split, sl in splits.items()
        }

        # Step 6 — Write
        writer   = DatasetWriter(self.output_dir, dry_run=self.dry_run)
        manifest = writer.write(splits)

        # Step 7 — Manifest
        writer.save_manifest(self.output_dir / "manifest.csv")

        # Step 8 — Update dataset.yaml
        if not self.dry_run:
            self._update_dataset_yaml(splits)

        # Step 9 — Save report
        if not self.dry_run:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            save_json(self.report, self.reports_dir / "dataset_preparation_report.json")

        self._print_final_summary(splits, manifest)

    def _run_fracatlas(self) -> List[dict]:
        logger.info("-" * 40)
        logger.info("Processing FracAtlas...")
        if not self.fracatlas_root.exists():
            logger.error(f"FracAtlas root not found: {self.fracatlas_root}")
            return []
        return FracAtlasProcessor(
            self.fracatlas_root, verbose=self.verbose
        ).process()

    def _run_grazpedwri(self) -> List[dict]:
        logger.info("-" * 40)
        logger.info("Processing GRAZPEDWRI-DX...")
        if not self.grazpedwri_root.exists():
            logger.error(f"GRAZPEDWRI root not found: {self.grazpedwri_root}")
            return []
        return GRAZPEDWRIProcessor(
            self.grazpedwri_root, verbose=self.verbose
        ).process()

    def _deduplicate(
        self, samples: List[dict]
    ) -> Tuple[List[dict], dict]:
        seen:       Dict[str, str] = {}
        unique:     List[dict]     = []
        duplicates: List[dict]     = []

        for s in samples:
            img_path = s.get("image_path")
            if img_path is None or not img_path.exists():
                unique.append(s)
                continue
            file_hash = compute_file_hash(img_path)
            if not file_hash:
                unique.append(s)
                continue
            if file_hash in seen:
                duplicates.append({
                    "stem":          s["stem"],
                    "source":        s["source"],
                    "duplicate_of":  seen[file_hash],
                })
                logger.info(
                    f"Duplicate: {s['source']}:{s['stem']} == {seen[file_hash]}"
                )
            else:
                seen[file_hash] = f"{s['source']}:{s['stem']}"
                unique.append(s)

        logger.info(
            f"Deduplication: {len(samples)} → {len(unique)} "
            f"({len(duplicates)} removed)"
        )
        return unique, {
            "total_before":       len(samples),
            "total_after":        len(unique),
            "duplicates_removed": len(duplicates),
            "duplicate_list":     duplicates,
        }

    def _update_dataset_yaml(self, splits: Dict[str, List[dict]]) -> None:
        if not self.config_path.exists():
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        total     = sum(len(v) for v in splits.values())
        total_pos = sum(
            s.get("fracture_positive", False)
            for sl in splits.values()
            for s in sl
        )
        if "stats" in cfg:
            cfg["stats"].update({
                "total_images":     total,
                "train_images":     len(splits.get("train", [])),
                "val_images":       len(splits.get("val",   [])),
                "test_images":      len(splits.get("test",  [])),
                "fracture_positive": total_pos,
                "fracture_negative": total - total_pos,
            })
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
        logger.info(f"Updated dataset.yaml: {self.config_path}")

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _print_final_summary(
        self, splits: Dict[str, List[dict]], manifest: List[ManifestRow]
    ) -> None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        total = sum(len(v) for v in splits.values())
        for split, sl in splits.items():
            pos = sum(s.get("fracture_positive", False) for s in sl)
            pct = len(sl) / total * 100 if total else 0
            logger.info(
                f"  {split:6s}: {len(sl):6d} ({pct:5.1f}%) — "
                f"positive={pos}, negative={len(sl)-pos}"
            )
        logger.info(f"  Manifest rows : {len(manifest)}")
        logger.info(f"  Output dir    : {self.output_dir}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 — Dataset preparation")
    p.add_argument("--config",      default="configs/dataset.yaml")
    p.add_argument("--output",      default="data/processed")
    p.add_argument("--reports",     default="reports")
    p.add_argument("--fracatlas",   default=None)
    p.add_argument("--grazpedwri",  default=None)
    p.add_argument("--source",      default=None, choices=["fracatlas", "grazpedwri"])
    p.add_argument("--seed",        type=int, default=RANDOM_SEED)
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    DatasetPreparationPipeline(
        config_path     = Path(args.config),
        output_dir      = Path(args.output),
        reports_dir     = Path(args.reports),
        fracatlas_root  = Path(args.fracatlas)  if args.fracatlas  else None,
        grazpedwri_root = Path(args.grazpedwri) if args.grazpedwri else None,
        source          = args.source,
        dry_run         = args.dry_run,
        verbose         = args.verbose,
        seed            = args.seed,
    ).run()


if __name__ == "__main__":
    main()