"""
Dataset preparation pipeline for Phase 1 — REVISED after data audit.

Key differences from the previous version:
  - FracAtlas raw class IDs are pre-scanned BEFORE remapping to class 0.
  - GRAZPEDWRI-DX unknown VOC class names => BLOCKER.
  - annotation_status distinguishes true negatives from invalid-box negatives.
  - Splitting is performed INDEPENDENTLY per source dataset.
  - The canonical manifest is the only output schema.
  - dataset.yaml is regenerated from records (post-drop), not from splits (pre-drop).
  - DatasetWriter is instantiated and called ONCE only.

Usage:
    python scripts/prepare_dataset.py --verbose
    python scripts/prepare_dataset.py --dry-run
    python scripts/prepare_dataset.py --allow-multiclass-fracatlas
    python scripts/prepare_dataset.py --allow-unknown-grz-classes
"""

import argparse
import csv
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
from src.utils.file_utils import compute_file_hash, find_images, save_json
from src.utils.image_utils import get_image_dimensions, check_image_integrity
from src.data.manifest import ManifestRecord, ManifestStore, UNAVAILABLE

logger = get_logger(__name__)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FRACTURE_CLASS_NAMES = {
    "fracture", "Fracture", "FRACTURE",
    "bone fracture", "Bone Fracture",
    "fraktur", "fraktura",
}
IGNORED_CLASS_NAMES = {
    "text", "Text", "TEXT",
    "metal", "Metal",
    "periostealreaction", "PeriostealReaction",
    "pronatorsign", "PronatorSign",
    "boneanomaly", "BoneAnomaly",
    "bonelesion", "BoneLesion",
    "softtissue", "SoftTissue",
    "foreignbody", "ForeignBody",
}
CLINICALLY_ADJACENT_TO_FRACTURE = {
    "pronatorsign", "PronatorSign",
    "periostealreaction", "PeriostealReaction",
    "boneanomaly", "BoneAnomaly",
}


class DatasetIntegrityError(Exception):
    """Raised when the pipeline detects an unresolved BLOCKER-level issue."""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_project_root() -> Path:
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
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in [ai_root, project_root]:
        candidate = (base / raw).resolve()
        if candidate.exists():
            return candidate
    return (ai_root / raw).resolve()


# ---------------------------------------------------------------------------
# Annotation status helper
# ---------------------------------------------------------------------------

def determine_annotation_status(num_final_boxes: int, issues: List[dict]) -> str:
    dropped_fracture = any(
        iss.get("type") in ("INVALID_BOX", "INVALID_LINE") for iss in issues
    )
    if num_final_boxes > 0:
        return "positive_with_dropped_boxes" if dropped_fracture else "positive_clean"
    return "negative_from_invalid_boxes" if dropped_fracture else "negative_clean"


# ---------------------------------------------------------------------------
# YOLO annotation validator
# ---------------------------------------------------------------------------

class YOLOAnnotationValidator:
    def validate_line(self, line: str) -> Tuple[bool, Optional[list], str]:
        parts = line.strip().split()
        if len(parts) != 5:
            return False, None, f"Expected 5 fields, got {len(parts)}"
        try:
            cls = int(parts[0])
            xc, yc, w, h = (
                float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]),
            )
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
    def __init__(self, fracture_names: set, ignored_names: set):
        self.fracture_names = fracture_names
        self.ignored_names = ignored_names

    def convert(
        self,
        xml_path: Path,
        image_width: int,
        image_height: int,
    ) -> Tuple[List[list], List[dict]]:
        yolo_boxes, issues = [], []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            return [], [{"type": "PARSE_ERROR", "detail": str(e)}]

        size_el = root.find("size")
        if size_el is not None:
            try:
                aw = int(size_el.findtext("width", "0"))
                ah = int(size_el.findtext("height", "0"))
                if aw > 0 and ah > 0:
                    image_width, image_height = aw, ah
            except ValueError:
                pass

        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip()
            bndbox = obj.find("bndbox")
            if bndbox is None:
                issues.append({"type": "MISSING_BNDBOX", "class_name": name})
                continue
            try:
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))
            except ValueError as e:
                issues.append({
                    "type": "INVALID_COORDS",
                    "class_name": name,
                    "detail": str(e),
                })
                continue

            if name in self.fracture_names:
                box = self._convert_box(
                    xmin, ymin, xmax, ymax, image_width, image_height
                )
                if box is not None:
                    yolo_boxes.append([0] + box)
                else:
                    issues.append({
                        "type": "INVALID_BOX",
                        "class_name": name,
                        "detail": (
                            f"xmin={xmin},ymin={ymin},xmax={xmax},ymax={ymax},"
                            f"img_w={image_width},img_h={image_height}"
                        ),
                    })
            elif name in self.ignored_names:
                issues.append({"type": "IGNORED_CLASS", "class_name": name})
            else:
                issues.append({"type": "UNKNOWN_CLASS", "class_name": name})

        return yolo_boxes, issues

    @staticmethod
    def _convert_box(
        xmin: float, ymin: float,
        xmax: float, ymax: float,
        img_w: int, img_h: int,
    ) -> Optional[list]:
        if img_w <= 0 or img_h <= 0:
            return None
        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 0 or bh <= 0:
            return None
        xc = (xmin + xmax) / 2.0 / img_w
        yc = (ymin + ymax) / 2.0 / img_h
        w, h = bw / img_w, bh / img_h
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return None
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            return None
        return [xc, yc, w, h]


# ---------------------------------------------------------------------------
# FracAtlas processor
# ---------------------------------------------------------------------------

class FracAtlasProcessor:
    def __init__(
        self,
        raw_root: Path,
        verbose: bool = False,
        allow_multiclass: bool = False,
    ):
        self.raw_root = raw_root
        self.verbose = verbose
        self.allow_multiclass = allow_multiclass

        self._inner = raw_root / "FracAtlas"
        self.yolo_dir = self._inner / "Annotations" / "YOLO"
        self.images_fractured = self._inner / "images" / "Fractured"
        self.images_non_fractured = self._inner / "images" / "Non_fractured"
        self.csv_path = self._inner / "dataset.csv"
        self.validator = YOLOAnnotationValidator()

    def process(self) -> List[dict]:
        samples = []
        if not self.yolo_dir.exists():
            logger.error(f"FracAtlas YOLO dir not found: {self.yolo_dir}")
            return samples

        label_files = sorted(self.yolo_dir.glob("*.txt"))
        logger.info(
            f"FracAtlas: found {len(label_files)} YOLO label files in {self.yolo_dir}"
        )

        # PASS 1: raw class audit BEFORE any remap
        raw_class_counter: Dict[int, int] = defaultdict(int)
        for lp in label_files:
            try:
                for line in lp.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            raw_class_counter[int(parts[0])] += 1
                        except ValueError:
                            pass
            except Exception:
                pass

        logger.info(
            f"FracAtlas: raw class distribution BEFORE remap = {dict(raw_class_counter)}"
        )
        if len(raw_class_counter) > 1 and not self.allow_multiclass:
            raise DatasetIntegrityError(
                f"FracAtlas contains {len(raw_class_counter)} distinct raw class IDs "
                f"{dict(raw_class_counter)}, but the pipeline force-remaps everything "
                f"to class 0. Re-run with --allow-multiclass-fracatlas only after "
                f"manually confirming all classes represent fracture."
            )

        image_index: Dict[str, Path] = {}
        for img_dir in [self.images_fractured, self.images_non_fractured]:
            if img_dir.exists():
                for img_path in find_images(img_dir):
                    image_index[img_path.stem] = img_path
        logger.info(
            f"FracAtlas: indexed {len(image_index)} images "
            f"(fractured + non_fractured)"
        )

        metadata = self._load_csv()

        for label_path in label_files:
            stem = label_path.stem
            image_path = image_index.get(stem)
            sample_meta = dict(metadata.get(stem, {}))

            valid_lines, issues = [], []
            try:
                raw_lines = [
                    ln for ln in
                    label_path.read_text(encoding="utf-8").strip().splitlines()
                    if ln.strip()
                ]
            except Exception as e:
                issues.append({"type": "READ_ERROR", "detail": str(e)})
                samples.append(
                    self._make_sample(stem, image_path, [], issues, sample_meta)
                )
                continue

            for line in raw_lines:
                is_valid, parsed, msg = self.validator.validate_line(line)
                if is_valid:
                    parsed[0] = 0  # remap to class 0 (proven safe by PASS 1)
                    valid_lines.append(parsed)
                else:
                    issues.append({
                        "type": "INVALID_LINE",
                        "raw": line,
                        "detail": msg,
                    })
                    if self.verbose:
                        logger.debug(f"FracAtlas {stem}: invalid line — {msg}")

            samples.append(
                self._make_sample(stem, image_path, valid_lines, issues, sample_meta)
            )

        valid_count = sum(1 for s in samples if s["valid"])
        pos_count = sum(1 for s in samples if s.get("fracture_positive", False))
        logger.info(
            f"FracAtlas: processed {len(samples)} samples — "
            f"valid={valid_count}, positive={pos_count}, "
            f"negative={len(samples) - pos_count}"
        )
        return samples

    def _make_sample(
        self,
        stem: str,
        image_path: Optional[Path],
        label_lines: List[list],
        issues: List[dict],
        metadata: dict,
    ) -> dict:
        return {
            "source": "fracatlas",
            "stem": stem,
            "image_path": image_path,
            "label_lines": label_lines,
            "valid": image_path is not None,
            "issues": issues,
            "metadata": metadata,
            "fracture_positive": len(label_lines) > 0,
            "annotation_source": "yolo_existing",
            "annotation_status": determine_annotation_status(
                len(label_lines), issues
            ),
        }

    def _load_csv(self) -> Dict[str, dict]:
        result = {}
        if not self.csv_path.exists():
            return result
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (
                        Path(row.get("image_id", "")).stem
                        or str(list(row.values())[0])
                    )
                    result[key] = dict(row)
            logger.info(f"FracAtlas: loaded {len(result)} CSV metadata rows")
            grouping_candidates = {
                "patient_id", "patientid", "patient", "study_id", "studyid"
            }
            found = [
                c for c in (reader.fieldnames or [])
                if c.lower() in grouping_candidates
            ]
            if not found:
                logger.warning(
                    "FracAtlas: dataset.csv has NO patient/study grouping column. "
                    "Patient-level leakage CANNOT be verified for this dataset — "
                    "split will fall back to image-level stratified split."
                )
        except Exception as e:
            logger.warning(f"FracAtlas: could not load dataset.csv — {e}")
        return result


# ---------------------------------------------------------------------------
# GRAZPEDWRI-DX processor
# ---------------------------------------------------------------------------

class GRAZPEDWRIProcessor:
    def __init__(
        self,
        raw_root: Path,
        verbose: bool = False,
        allow_unknown_classes: bool = False,
    ):
        self.raw_root = raw_root
        self.verbose = verbose
        self.allow_unknown_classes = allow_unknown_classes

        self.voc_dir = raw_root / "folder_structure" / "pascalvoc"
        self.csv_path = raw_root / "dataset.csv"
        self.converter = PascalVOCConverter(FRACTURE_CLASS_NAMES, IGNORED_CLASS_NAMES)

        self._image_map: Dict[str, Path] = {}
        self._clinically_adjacent_negatives: int = 0
        self._build_image_map()

    def _build_image_map(self) -> None:
        for child in sorted(self.raw_root.iterdir()):
            if child.is_dir() and child.name.lower().startswith("images"):
                for img_path in find_images(child):
                    self._image_map.setdefault(img_path.stem, img_path)
        logger.info(f"GRAZPEDWRI-DX: indexed {len(self._image_map)} images")

    def process(self) -> List[dict]:
        samples = []
        if not self.voc_dir.exists():
            logger.error(f"GRAZPEDWRI-DX VOC dir not found: {self.voc_dir}")
            return samples

        xml_files = sorted(self.voc_dir.glob("*.xml"))
        logger.info(f"GRAZPEDWRI-DX: found {len(xml_files)} Pascal VOC XML files")

        metadata = self._load_csv()
        unknown_classes: Dict[str, int] = defaultdict(int)
        clinically_adjacent_negatives = 0

        for xml_path in xml_files:
            stem = xml_path.stem
            image_path = self._image_map.get(stem)
            sample_meta = dict(metadata.get(stem, {}))

            # patient_id: prefer CSV, fallback to filename-derived
            if not sample_meta.get("patient_id"):
                derived = self._extract_patient_id(stem)
                if derived:
                    sample_meta["patient_id"] = derived
                    sample_meta["_patient_id_source"] = "filename_derived"
            else:
                sample_meta["_patient_id_source"] = "csv"

            img_w, img_h = 0, 0
            if image_path is not None:
                dims = get_image_dimensions(image_path)
                if dims:
                    img_h, img_w = dims[0], dims[1]

            yolo_boxes, issues = self.converter.convert(xml_path, img_w, img_h)

            # track clinically-adjacent-but-fracture-negative samples
            has_adjacent = any(
                iss.get("class_name") in CLINICALLY_ADJACENT_TO_FRACTURE
                for iss in issues
                if iss["type"] == "IGNORED_CLASS"
            )
            if len(yolo_boxes) == 0 and has_adjacent:
                clinically_adjacent_negatives += 1

            for iss in issues:
                if iss["type"] == "UNKNOWN_CLASS":
                    unknown_classes[iss["class_name"]] += 1

            samples.append({
                "source": "grazpedwri",
                "stem": stem,
                "image_path": image_path,
                "label_lines": yolo_boxes,
                "valid": image_path is not None,
                "issues": issues,
                "metadata": sample_meta,
                "fracture_positive": len(yolo_boxes) > 0,
                "annotation_source": "pascal_voc",
                "annotation_status": determine_annotation_status(
                    len(yolo_boxes), issues
                ),
                "width": img_w or None,
                "height": img_h or None,
            })

        if unknown_classes and not self.allow_unknown_classes:
            raise DatasetIntegrityError(
                f"GRAZPEDWRI-DX contains {len(unknown_classes)} unknown class "
                f"name(s) not mapped to fracture nor ignored: "
                f"{dict(unknown_classes)}. Re-run with --allow-unknown-grz-classes "
                f"to treat them as ignored."
            )
        elif unknown_classes:
            logger.warning(
                f"GRAZPEDWRI-DX: proceeding with unknown classes IGNORED: "
                f"{dict(unknown_classes)}"
            )

        self._clinically_adjacent_negatives = clinically_adjacent_negatives
        if clinically_adjacent_negatives > 0:
            logger.warning(
                f"GRAZPEDWRI-DX: {clinically_adjacent_negatives} samples are "
                f"fracture-NEGATIVE but contain a clinically fracture-adjacent "
                f"finding. Documented Phase-1 scope limitation."
            )

        valid_count = sum(1 for s in samples if s["valid"])
        pos_count = sum(1 for s in samples if s.get("fracture_positive", False))
        logger.info(
            f"GRAZPEDWRI-DX: processed {len(samples)} samples — "
            f"valid={valid_count}, positive={pos_count}, "
            f"negative={len(samples) - pos_count}"
        )
        return samples

    @staticmethod
    def _extract_patient_id(stem: str) -> Optional[str]:
        parts = stem.split("_")
        return parts[0] if len(parts) >= 2 else None

    def _load_csv(self) -> Dict[str, dict]:
        result = {}
        if not self.csv_path.exists():
            logger.warning(
                f"GRAZPEDWRI-DX: dataset.csv not found at {self.csv_path}"
            )
            return result
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = (
                        row.get("filestem", "")
                        or row.get("filename", "")
                        or row.get("image_id", "")
                        or list(row.values())[0]
                    )
                    key = Path(filename).stem if filename else ""
                    if key:
                        result[key] = dict(row)
            logger.info(
                f"GRAZPEDWRI-DX: loaded {len(result)} CSV metadata rows"
            )
        except Exception as e:
            logger.warning(
                f"GRAZPEDWRI-DX: could not load dataset.csv — {e}"
            )
        return result


# ---------------------------------------------------------------------------
# Leakage-aware splitter — splits INDEPENDENTLY per source dataset
# ---------------------------------------------------------------------------

class LeakageAwareSplitter:
    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        seed: int = RANDOM_SEED,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.strategy_report: Dict[str, str] = {}

    def split(self, samples: List[dict]) -> Dict[str, List[dict]]:
        by_source: Dict[str, List[dict]] = defaultdict(list)
        for s in samples:
            by_source[s["source"]].append(s)

        combined: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

        for source, source_samples in by_source.items():
            group_field = self._detect_group_field(source, source_samples)
            if group_field:
                logger.info(
                    f"[{source}] Using GROUP-LEVEL split on '{group_field}'"
                )
                self.strategy_report[source] = f"group_level:{group_field}"
                s_splits = self._group_split(source_samples, group_field)
            else:
                logger.info(
                    f"[{source}] No reliable group field — "
                    f"STRATIFIED IMAGE-LEVEL split"
                )
                self.strategy_report[source] = "image_level_stratified"
                s_splits = self._stratified_split(source_samples)

            for split in ("train", "val", "test"):
                combined[split].extend(s_splits[split])

        self._log_split_stats(combined)
        return combined

    def _detect_group_field(
        self, source: str, samples: List[dict]
    ) -> Optional[str]:
        candidates = [
            "patient_id", "patientid", "patient", "study_id", "studyid"
        ]
        for field in candidates:
            values = [
                str(s.get("metadata", {}).get(field, "")).strip()
                for s in samples
            ]
            non_empty = [
                v for v in values
                if v and v not in {"", "nan", "None", "UNAVAILABLE"}
            ]
            coverage = len(non_empty) / len(samples) if samples else 0
            if coverage >= 0.8:
                logger.info(
                    f"  Group field '{field}': "
                    f"{len(set(non_empty))} groups, "
                    f"{coverage:.1%} coverage → USABLE"
                )
                return field
            elif non_empty:
                logger.info(
                    f"  Group field '{field}': "
                    f"{coverage:.1%} coverage → too low, skipping"
                )
        return None

    def _group_split(
        self, samples: List[dict], group_field: str
    ) -> Dict[str, List[dict]]:
        group_map: Dict[str, List[dict]] = defaultdict(list)
        for s in samples:
            gid = str(
                s.get("metadata", {}).get(
                    group_field, f"_unk_{s['stem']}"
                )
            )
            group_map[gid].append(s)

        positive_groups, negative_groups = [], []
        for gid, group_samples in group_map.items():
            if any(s.get("fracture_positive", False) for s in group_samples):
                positive_groups.append(gid)
            else:
                negative_groups.append(gid)

        random.shuffle(positive_groups)
        random.shuffle(negative_groups)

        splits: Dict[str, List[dict]] = {
            "train": [], "val": [], "test": []
        }
        for group_list in [positive_groups, negative_groups]:
            n = len(group_list)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            for gid in group_list[:n_train]:
                splits["train"].extend(group_map[gid])
            for gid in group_list[n_train: n_train + n_val]:
                splits["val"].extend(group_map[gid])
            for gid in group_list[n_train + n_val:]:
                splits["test"].extend(group_map[gid])
        return splits

    def _stratified_split(
        self, samples: List[dict]
    ) -> Dict[str, List[dict]]:
        positive = [s for s in samples if s.get("fracture_positive", False)]
        negative = [s for s in samples if not s.get("fracture_positive", False)]
        random.shuffle(positive)
        random.shuffle(negative)

        splits: Dict[str, List[dict]] = {
            "train": [], "val": [], "test": []
        }
        for group in [positive, negative]:
            n = len(group)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            splits["train"].extend(group[:n_train])
            splits["val"].extend(group[n_train: n_train + n_val])
            splits["test"].extend(group[n_train + n_val:])
        return splits

    @staticmethod
    def _log_split_stats(splits: Dict[str, List[dict]]) -> None:
        total = sum(len(v) for v in splits.values())
        for split, slist in splits.items():
            pos = sum(
                1 for s in slist if s.get("fracture_positive", False)
            )
            pct = len(slist) / total * 100 if total else 0
            logger.info(
                f"  {split:6s}: {len(slist):6d} ({pct:5.1f}%) — "
                f"pos={pos}, neg={len(slist) - pos}"
            )


# ---------------------------------------------------------------------------
# Dataset writer — called EXACTLY ONCE
# ---------------------------------------------------------------------------

class DatasetWriter:
    def __init__(self, output_dir: Path, dry_run: bool = False):
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.records: List[ManifestRecord] = []
        self._name_counter: Dict[str, int] = defaultdict(int)

    def write(
        self, splits: Dict[str, List[dict]]
    ) -> Tuple[List[ManifestRecord], List[dict]]:
        if not self.dry_run:
            self._create_dirs()

        dropped_samples: List[dict] = []
        sample_id = 0

        for split, samples in splits.items():
            images_dir = self.output_dir / split / "images"
            labels_dir = self.output_dir / split / "labels"

            for sample in samples:
                if not sample.get("valid", False):
                    dropped_samples.append({
                        "stem": sample["stem"],
                        "source": sample["source"],
                        "reason": "no_matching_image_orphan_annotation",
                    })
                    continue

                img_src: Optional[Path] = sample.get("image_path")
                if img_src is None or not img_src.exists():
                    dropped_samples.append({
                        "stem": sample["stem"],
                        "source": sample["source"],
                        "reason": "image_path_missing_on_disk",
                    })
                    continue

                # deep integrity check BEFORE copying
                is_readable, integrity_status = check_image_integrity(img_src)
                if not is_readable:
                    dropped_samples.append({
                        "stem": sample["stem"],
                        "source": sample["source"],
                        "reason": f"image_integrity_failed:{integrity_status}",
                    })
                    logger.warning(
                        f"Dropping {sample['source']}:{sample['stem']} "
                        f"— {integrity_status}"
                    )
                    continue

                dims = get_image_dimensions(img_src)
                height = dims[0] if dims else None
                width = dims[1] if dims else None

                # reuse hash computed during deduplication if available
                img_hash = (
                    sample.get("image_hash")
                    or compute_file_hash(img_src)
                    or UNAVAILABLE
                )

                safe_name = self._safe_filename(
                    sample["source"], img_src.stem, img_src.suffix
                )
                img_dst = images_dir / safe_name
                label_dst = labels_dir / (Path(safe_name).stem + ".txt")

                if not self.dry_run:
                    shutil.copy2(img_src, img_dst)
                    self._write_label(label_dst, sample.get("label_lines", []))

                meta = sample.get("metadata", {})
                self.records.append(ManifestRecord(
                    sample_id=str(sample_id),
                    dataset=sample["source"],
                    image_path=safe_name,
                    original_filename=img_src.name,
                    image_hash=img_hash,
                    patient_id=(
                        str(meta.get("patient_id", "")).strip() or UNAVAILABLE
                    ),
                    study_id=(
                        str(
                            meta.get("study_id", meta.get("studyid", ""))
                        ).strip() or UNAVAILABLE
                    ),
                    fracture_positive=str(
                        sample.get("fracture_positive", False)
                    ),
                    num_boxes=str(len(sample.get("label_lines", []))),
                    annotation_source=sample.get(
                        "annotation_source", "unknown"
                    ),
                    annotation_status=sample.get(
                        "annotation_status", "unknown"
                    ),
                    width=str(width) if width is not None else UNAVAILABLE,
                    height=str(height) if height is not None else UNAVAILABLE,
                    split=split,
                ))
                sample_id += 1

        prefix = "[DRY RUN] " if self.dry_run else ""
        logger.info(
            f"DatasetWriter: {prefix}wrote {sample_id} samples, "
            f"dropped {len(dropped_samples)} invalid samples"
        )
        return self.records, dropped_samples

    def save_manifest(self, manifest_path: Path) -> None:
        if self.dry_run:
            logger.info(f"[DRY RUN] Would save manifest: {manifest_path}")
            return
        store = ManifestStore(self.records)
        store.save(manifest_path)
        logger.info(
            f"Manifest saved: {manifest_path} ({len(self.records)} rows)"
        )

    def _create_dirs(self) -> None:
        for split in ("train", "val", "test"):
            (self.output_dir / split / "images").mkdir(
                parents=True, exist_ok=True
            )
            (self.output_dir / split / "labels").mkdir(
                parents=True, exist_ok=True
            )

    def _safe_filename(
        self, source: str, stem: str, suffix: str
    ) -> str:
        prefix = {
            "fracatlas": "fa", "grazpedwri": "grz"
        }.get(source, source[:3])
        base = f"{prefix}_{stem}{suffix}"
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
                xc = float(box[1])
                yc = float(box[2])
                w = float(box[3])
                h = float(box[4])
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class DatasetPreparationPipeline:
    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        reports_dir: Path,
        fracatlas_root: Optional[Path] = None,
        grazpedwri_root: Optional[Path] = None,
        source: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False,
        seed: int = RANDOM_SEED,
        allow_multiclass_fracatlas: bool = False,
        allow_unknown_grz_classes: bool = False,
    ):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.source = source
        self.dry_run = dry_run
        self.verbose = verbose
        self.seed = seed
        self.allow_multiclass_fracatlas = allow_multiclass_fracatlas
        self.allow_unknown_grz_classes = allow_unknown_grz_classes

        # _load_config must be defined before this line
        self.config = self._load_config()

        self._project_root = resolve_project_root()
        self._ai_module_root = Path(__file__).resolve().parent.parent

        sources_cfg = {
            s["name"]: s["local_path"]
            for s in self.config.get("sources", [])
        }

        self.fracatlas_root = fracatlas_root or resolve_dataset_path(
            sources_cfg.get("FracAtlas", "../fracatlas"),
            self._project_root,
            self._ai_module_root,
        )
        self.grazpedwri_root = grazpedwri_root or resolve_dataset_path(
            sources_cfg.get("GRAZPEDWRI-DX", "../GRAZPEDWRI-DX"),
            self._project_root,
            self._ai_module_root,
        )

        logger.info(
            f"FracAtlas root : {self.fracatlas_root} "
            f"(exists={self.fracatlas_root.exists()})"
        )
        logger.info(
            f"GRAZPEDWRI root: {self.grazpedwri_root} "
            f"(exists={self.grazpedwri_root.exists()})"
        )

        self._grz_processor_clinically_adjacent: Optional[int] = None

        self.report: Dict = {
            "pipeline": "phase1_dataset_preparation_v2",
            "seed": seed,
            "dry_run": dry_run,
            "sources": {},
            "deduplication": {},
            "splitting": {},
            "split_strategy_per_source": {},
            "dropped_samples": [],
            "dropped_samples_by_reason": {},
            "reconciliation": {},
            "clinical_notes": {},
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("=" * 60)
        logger.info("Phase 1 — Dataset Preparation Pipeline (v2, audit-hardened)")
        logger.info("=" * 60)

        all_samples: List[dict] = []

        if self.source in (None, "fracatlas"):
            fa_samples = self._run_fracatlas()
            all_samples.extend(fa_samples)
            self.report["sources"]["fracatlas"] = {
                "total": len(fa_samples),
                "valid": sum(s["valid"] for s in fa_samples),
                "positive": sum(
                    1 for s in fa_samples if s.get("fracture_positive", False)
                ),
            }

        if self.source in (None, "grazpedwri"):
            grz_samples = self._run_grazpedwri()
            all_samples.extend(grz_samples)
            self.report["sources"]["grazpedwri"] = {
                "total": len(grz_samples),
                "valid": sum(s["valid"] for s in grz_samples),
                "positive": sum(
                    1 for s in grz_samples if s.get("fracture_positive", False)
                ),
            }

        total_before_dedup = len(all_samples)
        logger.info(f"Total samples before deduplication: {total_before_dedup}")

        all_samples, dup_report = self._deduplicate(all_samples)
        self.report["deduplication"] = dup_report

        valid_samples = [s for s in all_samples if s.get("valid", False)]
        invalid_samples = [s for s in all_samples if not s.get("valid", False)]
        logger.info(
            f"Valid samples: {len(valid_samples)} | "
            f"Invalid (orphan): {len(invalid_samples)}"
        )

        splitter = LeakageAwareSplitter(seed=self.seed)
        splits = splitter.split(valid_samples)
        self.report["split_strategy_per_source"] = splitter.strategy_report
        self.report["splitting"] = {
            split: {
                "count": len(sl),
                "positive": sum(
                    1 for s in sl if s.get("fracture_positive", False)
                ),
                "negative": sum(
                    1 for s in sl if not s.get("fracture_positive", False)
                ),
            }
            for split, sl in splits.items()
        }

        # FIX: DatasetWriter instantiated and called EXACTLY ONCE
        writer = DatasetWriter(self.output_dir, dry_run=self.dry_run)
        records, dropped = writer.write(splits)

        # clinical notes — attribute set by _run_grazpedwri
        self.report["clinical_notes"] = {
            "grazpedwri_clinically_adjacent_negatives": (
                self._grz_processor_clinically_adjacent
            ),
        }

        # breakdown of drop reasons
        drop_reason_counts: Dict[str, int] = defaultdict(int)
        for d in dropped:
            reason_key = d["reason"].split(":")[0]
            drop_reason_counts[reason_key] += 1

        self.report["dropped_samples"] = dropped
        self.report["dropped_samples_by_reason"] = dict(drop_reason_counts)

        writer.save_manifest(self.output_dir / "manifest.csv")

        if not self.dry_run:
            # FIX: pass records (post-drop) not splits (pre-drop)
            self._write_dataset_yaml(records, dropped)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                self.report,
                self.reports_dir / "dataset_preparation_report.json",
            )

        self._print_final_summary(records)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_fracatlas(self) -> List[dict]:
        if not self.fracatlas_root.exists():
            logger.error(
                f"FracAtlas root not found: {self.fracatlas_root}"
            )
            return []
        return FracAtlasProcessor(
            self.fracatlas_root,
            verbose=self.verbose,
            allow_multiclass=self.allow_multiclass_fracatlas,
        ).process()

    def _run_grazpedwri(self) -> List[dict]:
        if not self.grazpedwri_root.exists():
            logger.error(
                f"GRAZPEDWRI root not found: {self.grazpedwri_root}"
            )
            return []
        processor = GRAZPEDWRIProcessor(
            self.grazpedwri_root,
            verbose=self.verbose,
            allow_unknown_classes=self.allow_unknown_grz_classes,
        )
        samples = processor.process()
        # store for clinical_notes report
        self._grz_processor_clinically_adjacent = (
            processor._clinically_adjacent_negatives
        )
        return samples

    def _deduplicate(
        self, samples: List[dict]
    ) -> Tuple[List[dict], dict]:
        seen: Dict[str, dict] = {}
        unique, duplicates = [], []

        for s in samples:
            img_path = s.get("image_path")
            if img_path is None or not img_path.exists():
                unique.append(s)
                continue
            file_hash = compute_file_hash(img_path)
            if not file_hash:
                unique.append(s)
                continue

            # store hash in sample so DatasetWriter reuses it (no second hash)
            s["image_hash"] = file_hash

            if file_hash in seen:
                duplicates.append({
                    "stem": s["stem"],
                    "source": s["source"],
                    "duplicate_of": seen[file_hash]["ref"],
                    "was_positive": s.get("fracture_positive", False),
                    "kept_sample_was_positive": seen[file_hash]["positive"],
                })
            else:
                seen[file_hash] = {
                    "ref": f"{s['source']}:{s['stem']}",
                    "positive": s.get("fracture_positive", False),
                }
                unique.append(s)

        logger.info(
            f"Deduplication: {len(samples)} → {len(unique)} "
            f"({len(duplicates)} removed)"
        )
        return unique, {
            "total_before": len(samples),
            "total_after": len(unique),
            "duplicates_removed": len(duplicates),
            "duplicates_that_were_positive": sum(
                1 for d in duplicates if d["was_positive"]
            ),
            "duplicate_list": duplicates,
        }

    def _write_dataset_yaml(
        self,
        records: List[ManifestRecord],
        dropped: List[dict],
    ) -> None:
        """
        Regenerates dataset.yaml from records (post-drop actual counts).

        IMPORTANT: split counts come from records, NOT from splits dict.
        splits contains pre-drop counts; records reflects what was
        actually written to disk after integrity filtering.
        """
        total = len(records)
        total_pos = sum(1 for r in records if r.fracture_positive == "True")

        split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        by_dataset: Dict[str, int] = defaultdict(int)
        status_dist: Dict[str, int] = defaultdict(int)
        widths: List[int] = []
        heights: List[int] = []
        boxes: List[int] = []

        for r in records:
            split_counts[r.split] = split_counts.get(r.split, 0) + 1
            by_dataset[r.dataset] += 1
            status_dist[r.annotation_status] += 1
            if r.width not in ("", UNAVAILABLE):
                try:
                    widths.append(int(r.width))
                except ValueError:
                    pass
            if r.height not in ("", UNAVAILABLE):
                try:
                    heights.append(int(r.height))
                except ValueError:
                    pass
            if r.num_boxes.isdigit():
                boxes.append(int(r.num_boxes))

        # build drop reason summary
        drop_reasons: Dict[str, int] = defaultdict(int)
        for d in dropped:
            drop_reasons[d["reason"].split(":")[0]] += 1

        logger.info(
            f"dataset.yaml stats: total={total} "
            f"train={split_counts['train']} "
            f"val={split_counts['val']} "
            f"test={split_counts['test']} "
            f"dropped={len(dropped)}"
        )

        cfg = self._load_config()
        cfg["stats"] = {
            "total_images": total,
            "train_images": split_counts["train"],
            "val_images": split_counts["val"],
            "test_images": split_counts["test"],
            "fracture_positive": total_pos,
            "fracture_negative": total - total_pos,
            "by_dataset": dict(by_dataset),
            "annotation_status_distribution": dict(status_dist),
            "avg_image_width": (
                round(sum(widths) / len(widths), 1) if widths else None
            ),
            "avg_image_height": (
                round(sum(heights) / len(heights), 1) if heights else None
            ),
            "avg_annotations_per_image": (
                round(sum(boxes) / total, 3) if total else None
            ),
            "generated_by": "scripts/prepare_dataset.py (v2)",
            "random_seed": self.seed,
            "dropped_total": len(dropped),
            "dropped_by_reason": dict(drop_reasons),
        }

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
        logger.info(f"dataset.yaml fully regenerated: {self.config_path}")

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _print_final_summary(self, records: List[ManifestRecord]) -> None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Manifest rows : {len(records)}")
        logger.info(f"  Output dir    : {self.output_dir}")
        logger.info(
            f"  Split strategy per source: "
            f"{self.report.get('split_strategy_per_source')}"
        )
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 — Dataset preparation (audit-hardened)"
    )
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--reports", default="reports")
    p.add_argument("--fracatlas", default=None)
    p.add_argument("--grazpedwri", default=None)
    p.add_argument(
        "--source", default=None, choices=["fracatlas", "grazpedwri"]
    )
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--allow-multiclass-fracatlas", action="store_true")
    p.add_argument("--allow-unknown-grz-classes", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        DatasetPreparationPipeline(
            config_path=Path(args.config),
            output_dir=Path(args.output),
            reports_dir=Path(args.reports),
            fracatlas_root=Path(args.fracatlas) if args.fracatlas else None,
            grazpedwri_root=(
                Path(args.grazpedwri) if args.grazpedwri else None
            ),
            source=args.source,
            dry_run=args.dry_run,
            verbose=args.verbose,
            seed=args.seed,
            allow_multiclass_fracatlas=args.allow_multiclass_fracatlas,
            allow_unknown_grz_classes=args.allow_unknown_grz_classes,
        ).run()
    except DatasetIntegrityError as e:
        logger.error(f"BLOCKER — pipeline aborted: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()