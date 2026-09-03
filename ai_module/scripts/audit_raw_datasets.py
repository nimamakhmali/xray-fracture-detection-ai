"""
Independent forensic audit of raw datasets — FracAtlas & GRAZPEDWRI-DX.

CRITICAL: This script does NOT import from or trust prepare_dataset.py,
dataset.yaml, or any previously generated report. It reads raw files
directly. Its output is the independent ground truth used to validate
(or invalidate) everything prepare_dataset.py claims.

v2 changes:
  - Full-file image integrity scan (catches truncated JPEGs that
    zero-byte checks miss entirely).
  - Confirmed 9-class GRAZPEDWRI-DX taxonomy from full XML scan.

Usage:
    python scripts/audit_raw_datasets.py \
        --fracatlas ../fracatlas --grazpedwri ../GRAZPEDWRI-DX
"""
import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.file_utils import find_images, compute_file_hash, save_json
from src.utils.image_utils import check_image_integrity
from src.utils.logger import get_logger

logger = get_logger("raw_audit")

KNOWN_FRACTURE_NAMES = {
    "fracture", "Fracture", "FRACTURE", "bone fracture", "Bone Fracture",
    "fraktur", "fraktura",
}
# Confirmed via full-scan audit against the real GRAZPEDWRI-DX taxonomy
# (9 classes total: fracture + text + 7 others below).
KNOWN_IGNORED_NAMES = {
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


def scan_image_integrity(images: List[Path]) -> dict:
    """Full scan, no sampling. Slow but authoritative — this is a one-time
    forensic audit, not a per-training-run check."""
    zero_byte, truncated, corrupted, ok = [], [], [], 0
    for p in images:
        is_ok, status = check_image_integrity(p)
        if is_ok:
            ok += 1
        elif status == "zero_byte":
            zero_byte.append(str(p))
        elif status.startswith("truncated_or_corrupted"):
            truncated.append({"path": str(p), "detail": status})
        else:
            corrupted.append({"path": str(p), "detail": status})
    return {
        "total_scanned": len(images),
        "ok": ok,
        "zero_byte_count": len(zero_byte),
        "zero_byte_files": zero_byte[:50],
        "truncated_or_corrupted_count": len(truncated) + len(corrupted),
        "truncated_or_corrupted_files": (truncated + corrupted)[:50],
    }


def audit_fracatlas(root: Path) -> dict:
    report = {"root": str(root), "exists": root.exists()}
    if not root.exists():
        return report

    inner = root / "FracAtlas"
    yolo_dir = inner / "Annotations" / "YOLO"
    coco_path = inner / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"
    img_fractured = inner / "images" / "Fractured"
    img_non_fractured = inner / "images" / "Non_fractured"
    csv_path = inner / "dataset.csv"

    images = []
    for d in (img_fractured, img_non_fractured):
        if d.exists():
            images.extend(find_images(d))

    stems = [p.stem for p in images]
    stem_counts = Counter(stems)
    duplicate_stems = {s: c for s, c in stem_counts.items() if c > 1}

    logger.info(f"FracAtlas: running deep integrity scan on {len(images)} images (this can take a while)...")
    integrity = scan_image_integrity(images)
    logger.info(
        f"FracAtlas integrity: ok={integrity['ok']} "
        f"zero_byte={integrity['zero_byte_count']} "
        f"truncated_or_corrupted={integrity['truncated_or_corrupted_count']}"
    )

    hashes: Dict[str, list] = defaultdict(list)
    for p in images:
        if p.stat().st_size == 0:
            continue
        h = compute_file_hash(p)
        if h:
            hashes[h].append(str(p))
    duplicate_content = {h: v for h, v in hashes.items() if len(v) > 1}

    report["images"] = {
        "total": len(images),
        "fractured_dir_count": len(find_images(img_fractured)) if img_fractured.exists() else 0,
        "non_fractured_dir_count": len(find_images(img_non_fractured)) if img_non_fractured.exists() else 0,
        "formats": dict(Counter(p.suffix.lower() for p in images)),
        "duplicate_filename_stems": duplicate_stems,
        "duplicate_content_groups": len(duplicate_content),
        "duplicate_content_examples": dict(list(duplicate_content.items())[:10]),
        "integrity": integrity,
    }
    if integrity["truncated_or_corrupted_count"] > 0:
        report["images"]["INTEGRITY_BLOCKER"] = (
            f"{integrity['truncated_or_corrupted_count']} truncated/corrupted images found. "
            f"These MUST be excluded from the processed dataset, not silently included."
        )

    label_files = sorted(yolo_dir.glob("*.txt")) if yolo_dir.exists() else []
    raw_class_counter: Counter = Counter()
    empty_count = non_empty_count = malformed = total_boxes = 0
    image_stem_set = set(stems)
    orphan_labels, missing_labels_for_images = [], []

    for lp in label_files:
        try:
            lines = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception:
            malformed += 1
            continue
        if not lines:
            empty_count += 1
        else:
            non_empty_count += 1
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                malformed += 1
                continue
            try:
                raw_class_counter[int(parts[0])] += 1
                total_boxes += 1
            except ValueError:
                malformed += 1
        if lp.stem not in image_stem_set:
            orphan_labels.append(lp.stem)

    label_stem_set = {lp.stem for lp in label_files}
    for stem in image_stem_set:
        if stem not in label_stem_set:
            missing_labels_for_images.append(stem)

    report["yolo_annotations"] = {
        "dir": str(yolo_dir),
        "exists": yolo_dir.exists(),
        "total_label_files": len(label_files),
        "empty_label_files": empty_count,
        "non_empty_label_files": non_empty_count,
        "malformed_lines": malformed,
        "total_boxes": total_boxes,
        "raw_class_id_distribution": dict(raw_class_counter),
        "orphan_labels_no_image": orphan_labels[:50],
        "orphan_labels_count": len(orphan_labels),
        "images_missing_label": missing_labels_for_images[:50],
        "images_missing_label_count": len(missing_labels_for_images),
    }
    if len(raw_class_counter) > 1:
        report["yolo_annotations"]["BLOCKER"] = (
            f"Multiple raw class IDs found: {dict(raw_class_counter)}. "
            f"prepare_dataset.py currently force-remaps ALL of them to class 0. "
            f"This MUST be manually reviewed before trusting class-0=fracture."
        )

    report["coco"] = {"exists": coco_path.exists()}
    if coco_path.exists():
        try:
            with open(coco_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
            report["coco"].update({
                "images": len(coco.get("images", [])),
                "annotations": len(coco.get("annotations", [])),
                "categories": coco.get("categories", []),
            })
        except Exception as e:
            report["coco"]["parse_error"] = str(e)

    report["csv"] = {"exists": csv_path.exists()}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        grouping_candidates = {
            "patient_id", "patientid", "patient", "study_id", "studyid",
            "hospital", "source", "case_id", "subject_id",
        }
        found = [c for c in (reader.fieldnames or []) if c.lower() in grouping_candidates]
        report["csv"].update({
            "rows": len(rows),
            "columns": reader.fieldnames,
            "potential_grouping_fields": found,
            "NOTE": (
                "No grouping field found — patient-level leakage cannot be verified "
                "for FracAtlas. Splitter MUST fall back to image-level split for this "
                "dataset, not silently inherit GRAZPEDWRI's patient coverage."
            ) if not found else None,
        })

    return report


def audit_grazpedwri(root: Path) -> dict:
    report = {"root": str(root), "exists": root.exists()}
    if not root.exists():
        return report

    voc_dir = root / "folder_structure" / "pascalvoc"
    yolov5_dir = root / "folder_structure" / "yolov5" / "labels"
    superv_dir = root / "folder_structure" / "supervisely" / "wrist" / "ann"
    csv_path = root / "dataset.csv"

    images = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.lower().startswith("images"):
            images.extend(find_images(child))

    stems = [p.stem for p in images]
    duplicate_stems = {s: c for s, c in Counter(stems).items() if c > 1}

    logger.info(f"GRAZPEDWRI-DX: running deep integrity scan on {len(images)} images (this can take a while)...")
    integrity = scan_image_integrity(images)
    logger.info(
        f"GRAZPEDWRI-DX integrity: ok={integrity['ok']} "
        f"zero_byte={integrity['zero_byte_count']} "
        f"truncated_or_corrupted={integrity['truncated_or_corrupted_count']}"
    )

    report["images"] = {
        "total": len(images),
        "formats": dict(Counter(p.suffix.lower() for p in images)),
        "duplicate_filename_stems": duplicate_stems,
        "integrity": integrity,
    }
    if integrity["truncated_or_corrupted_count"] > 0:
        report["images"]["INTEGRITY_BLOCKER"] = (
            f"{integrity['truncated_or_corrupted_count']} truncated/corrupted images found."
        )

    xml_files = sorted(voc_dir.glob("*.xml")) if voc_dir.exists() else []
    class_counter: Counter = Counter()
    class_image_counter: Dict[str, set] = defaultdict(set)
    parse_errors = empty_xml = total_objects = 0
    stem_set = set(stems)
    orphan_xml, missing_xml_for_images = [], []
    clinically_adjacent_negative_stems = []

    for xp in xml_files:
        try:
            tree = ET.parse(xp)
            root_el = tree.getroot()
        except ET.ParseError:
            parse_errors += 1
            continue

        objs = root_el.findall("object")
        if not objs:
            empty_xml += 1

        names_in_this_file = []
        for obj in objs:
            name = obj.findtext("name", "").strip()
            class_counter[name] += 1
            class_image_counter[name].add(xp.stem)
            total_objects += 1
            names_in_this_file.append(name)

        has_fracture = any(n in KNOWN_FRACTURE_NAMES for n in names_in_this_file)
        has_adjacent = any(n in CLINICALLY_ADJACENT_TO_FRACTURE for n in names_in_this_file)
        if not has_fracture and has_adjacent:
            clinically_adjacent_negative_stems.append(xp.stem)

        if xp.stem not in stem_set:
            orphan_xml.append(xp.stem)

    xml_stem_set = {xp.stem for xp in xml_files}
    for stem in stem_set:
        if stem not in xml_stem_set:
            missing_xml_for_images.append(stem)

    unknown = {
        k: v for k, v in class_counter.items()
        if k not in KNOWN_FRACTURE_NAMES and k not in KNOWN_IGNORED_NAMES
    }

    report["voc_annotations"] = {
        "dir": str(voc_dir),
        "exists": voc_dir.exists(),
        "total_xml_files": len(xml_files),
        "parse_errors": parse_errors,
        "empty_xml_files": empty_xml,
        "total_objects": total_objects,
        "FULL_class_distribution_objects": dict(class_counter),
        "FULL_class_distribution_images": {k: len(v) for k, v in class_image_counter.items()},
        "orphan_xml_no_image": orphan_xml[:50],
        "orphan_xml_count": len(orphan_xml),
        "images_missing_xml": missing_xml_for_images[:50],
        "images_missing_xml_count": len(missing_xml_for_images),
        "UNKNOWN_CLASSES_BLOCKER": unknown,
        "clinical_notes": {
            "clinically_adjacent_classes": sorted(CLINICALLY_ADJACENT_TO_FRACTURE),
            "images_negative_for_fracture_but_clinically_adjacent_finding": len(clinically_adjacent_negative_stems),
            "example_stems": clinically_adjacent_negative_stems[:20],
            "SCOPE_NOTE": (
                "Phase 1 targets single-class fracture detection only. Images with "
                "pronatorsign/periostealreaction/boneanomaly but no explicit fracture "
                "annotation are treated as fracture-negative. This is a documented "
                "scope limitation, not a data quality bug."
            ),
        },
    }
    if unknown:
        report["voc_annotations"]["BLOCKER"] = (
            f"{len(unknown)} unknown class name(s) neither mapped to fracture "
            f"nor ignored: {list(unknown.keys())}. Training BLOCKED until resolved."
        )

    yolov5_files = sorted(yolov5_dir.glob("*.txt")) if yolov5_dir.exists() else []
    yolov5_class_counter: Counter = Counter()
    for lp in yolov5_files:
        try:
            for line in lp.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        yolov5_class_counter[int(parts[0])] += 1
                    except ValueError:
                        pass
        except Exception:
            pass
    report["yolov5_audit_only"] = {
        "dir": str(yolov5_dir),
        "exists": yolov5_dir.exists(),
        "total_files": len(yolov5_files),
        "FULL_class_id_distribution": dict(yolov5_class_counter),
    }

    superv_files = sorted(superv_dir.glob("*.json")) if superv_dir.exists() else []
    superv_class_counter: Counter = Counter()
    for sp in superv_files:
        try:
            with open(sp, "r", encoding="utf-8") as f:
                data = json.load(f)
            for obj in data.get("objects", []):
                superv_class_counter[obj.get("classTitle", obj.get("class", "unknown"))] += 1
        except Exception:
            pass
    report["supervisely_audit_only"] = {
        "dir": str(superv_dir),
        "exists": superv_dir.exists(),
        "total_files": len(superv_files),
        "FULL_class_distribution": dict(superv_class_counter),
    }

    report["csv"] = {"exists": csv_path.exists()}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        grouping_candidates = {"patient_id", "patientid", "patient", "study_id", "studyid"}
        found = [c for c in (reader.fieldnames or []) if c.lower() in grouping_candidates]
        report["csv"].update({
            "rows": len(csv_rows),
            "columns": reader.fieldnames,
            "potential_grouping_fields": found,
        })

        if found:
            csv_patient_col = found[0]
            checked = mismatches = 0
            for row in csv_rows:
                filename = row.get("filestem") or row.get("filename") or ""
                stem = Path(filename).stem if filename else ""
                if not stem or "_" not in stem:
                    continue
                derived = stem.split("_")[0]
                csv_val = str(row.get(csv_patient_col, "")).strip()
                if derived and csv_val:
                    checked += 1
                    if derived != csv_val:
                        mismatches += 1
            report["csv"]["patient_id_filename_vs_csv_check"] = {
                "column_compared": csv_patient_col,
                "checked": checked,
                "mismatches": mismatches,
                "mismatch_rate": round(mismatches / checked, 4) if checked else None,
                "VERDICT": (
                    "Filename-derived patient_id is UNRELIABLE — do not use for leakage split."
                    if checked and mismatches / checked > 0.02 else
                    "Filename-derived patient_id appears consistent with CSV."
                    if checked else
                    "Could not cross-validate — no comparable rows found."
                ),
            }
        else:
            report["csv"]["WARNING"] = (
                "No official patient/study column found in CSV. The filename-derived "
                "patient_id used by prepare_dataset.py is UNVERIFIED against any ground truth."
            )

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fracatlas", default="../fracatlas")
    parser.add_argument("--grazpedwri", default="../GRAZPEDWRI-DX")
    parser.add_argument("--output", default="reports/raw_data_audit.json")
    args = parser.parse_args()

    fa_root = Path(args.fracatlas).resolve()
    grz_root = Path(args.grazpedwri).resolve()

    logger.info(f"Auditing FracAtlas: {fa_root}")
    fa_report = audit_fracatlas(fa_root)

    logger.info(f"Auditing GRAZPEDWRI-DX: {grz_root}")
    grz_report = audit_grazpedwri(grz_root)

    full_report = {"fracatlas": fa_report, "grazpedwri": grz_report}
    out_path = Path(args.output)
    save_json(full_report, out_path)
    logger.info(f"Raw audit report saved: {out_path}")

    blockers = []
    if grz_report.get("voc_annotations", {}).get("UNKNOWN_CLASSES_BLOCKER"):
        blockers.append("GRAZPEDWRI unknown VOC classes")
    if len(fa_report.get("yolo_annotations", {}).get("raw_class_id_distribution", {})) > 1:
        blockers.append("FracAtlas multiple raw class IDs")
    if fa_report.get("images", {}).get("integrity", {}).get("truncated_or_corrupted_count", 0) > 0:
        blockers.append("FracAtlas truncated/corrupted images")
    if grz_report.get("images", {}).get("integrity", {}).get("truncated_or_corrupted_count", 0) > 0:
        blockers.append("GRAZPEDWRI truncated/corrupted images")

    if blockers:
        logger.error(f"BLOCKERS FOUND: {blockers}")
        sys.exit(1)
    logger.info("No BLOCKER-level issues found in raw class/annotation/integrity audit.")


if __name__ == "__main__":
    main()