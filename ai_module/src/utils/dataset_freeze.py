"""
src/utils/dataset_freeze.py

Frozen-dataset contract (Phase 2).

A freeze record captures:
  * hash of manifest.csv, configs/dataset.yaml and the validation report it was gated on
  * per-file content hash of EVERY image and label listed in the manifest
  * aggregate statistics recomputed from the manifest

Verification modes:
  quick : manifest hash + dataset.yaml hash          (used at training start)
  full  : re-hash every file                         (authoritative; freeze_dataset.py --verify --full)

`*.cache` files written by Ultralytics inside the dataset tree are tolerated
(they are caches, not data). Any other new / changed / missing file is a violation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

TOLERATED_SUFFIXES = {".cache"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"}
_HASH_LEN_TO_ALGO = {32: "md5", 40: "sha1", 64: "sha256"}


# ── helpers ────────────────────────────────────────────────────────────────

def hash_file(path: Path, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_hash_algorithm(sample_hash: str) -> str:
    algo = _HASH_LEN_TO_ALGO.get(len(sample_hash.strip()))
    if algo is None:
        raise ValueError(f"Cannot infer hash algorithm from a hash of length {len(sample_hash)}")
    return algo


def read_manifest_rows(manifest_path: Path) -> List[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def image_rel_path(row: dict) -> str:
    return f"{row['split']}/images/{row['image_path']}"


def label_rel_path(row: dict) -> str:
    return f"{row['split']}/labels/{Path(row['image_path']).stem}.txt"


def files_path_for(record_path: Path) -> Path:
    return record_path.with_name(record_path.stem + "_files.json")


# ── record ─────────────────────────────────────────────────────────────────

@dataclass
class FrozenDatasetRecord:
    version: str
    created_at: str
    seed: int
    hash_algorithm: str
    dataset_yaml_hash: str
    manifest_hash: str
    validation_report_hash: str
    total_images: int
    total_labels: int
    total_boxes: int
    train_images: int
    val_images: int
    test_images: int
    positive_images: int
    negative_images: int
    by_dataset: Dict[str, int]
    by_dataset_split: Dict[str, Dict[str, int]]
    annotation_status: Dict[str, int]
    file_count: int
    files_record: str
    jpeg_audit_summary: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)


# ── manifest ↔ disk reconciliation ─────────────────────────────────────────

def compare_manifest_to_disk(processed_dir: Path, rows: List[dict], algorithm: str) -> Dict:
    """Which manifest images differ from disk, are missing, or exist on disk but not in manifest."""
    mismatched: List[dict] = []
    missing: List[str] = []
    listed = set()
    for row in rows:
        rel = image_rel_path(row)
        listed.add(rel)
        p = processed_dir / rel
        if not p.exists():
            missing.append(rel)
            continue
        current = hash_file(p, algorithm)
        if current != row["image_hash"]:
            mismatched.append({"path": rel, "manifest_hash": row["image_hash"], "disk_hash": current})
    on_disk = {
        str(p.relative_to(processed_dir))
        for split in ("train", "val", "test")
        for p in (processed_dir / split / "images").glob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    return {
        "checked": len(rows),
        "mismatched": sorted(mismatched, key=lambda d: d["path"]),
        "missing": sorted(missing),
        "unlisted_on_disk": sorted(on_disk - listed),
        "clean": not mismatched and not missing and not (on_disk - listed),
    }


# ── create ─────────────────────────────────────────────────────────────────

def create_freeze_record(
    processed_dir: Path,
    dataset_yaml: Path,
    validation_report: Path,
    version: str = "frozen_v1",
    seed: int = 42,
    jpeg_audit_summary: Optional[Dict] = None,
    notes: Optional[List[str]] = None,
    files_record_name: Optional[str] = None,
) -> Tuple[FrozenDatasetRecord, Dict[str, str]]:
    """
    Build the freeze record. Raises if manifest and disk disagree — the caller
    must reconcile (freeze_dataset.py --sync-manifest-hashes) and re-validate first.
    """
    manifest_path = processed_dir / "manifest.csv"
    rows = read_manifest_rows(manifest_path)
    if not rows:
        raise ValueError("Manifest is empty.")
    algorithm = detect_hash_algorithm(rows[0]["image_hash"])

    diff = compare_manifest_to_disk(processed_dir, rows, algorithm)
    if not diff["clean"]:
        raise RuntimeError(
            f"Manifest/disk mismatch — refusing to freeze. "
            f"mismatched={len(diff['mismatched'])} missing={len(diff['missing'])} "
            f"unlisted={len(diff['unlisted_on_disk'])}. "
            f"Run: python scripts/freeze_dataset.py --sync-manifest-hashes, then validate, then freeze."
        )

    file_hashes: Dict[str, str] = {}
    total_boxes = total_labels = positive = 0
    by_dataset: Dict[str, int] = {}
    by_dataset_split: Dict[str, Dict[str, int]] = {}
    annotation_status: Dict[str, int] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for row in rows:
        img_rel = image_rel_path(row)
        file_hashes[img_rel] = row["image_hash"]          # verified equal to disk above

        lbl_rel = label_rel_path(row)
        lbl = processed_dir / lbl_rel
        if not lbl.exists():
            raise RuntimeError(f"Label file missing for {img_rel} (pipeline always writes one).")
        file_hashes[lbl_rel] = hash_file(lbl, algorithm)
        total_labels += 1
        total_boxes += sum(1 for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip())

        ds, sp = row["dataset"], row["split"]
        by_dataset[ds] = by_dataset.get(ds, 0) + 1
        by_dataset_split.setdefault(ds, {}).setdefault(sp, 0)
        by_dataset_split[ds][sp] += 1
        if sp in split_counts:
            split_counts[sp] += 1
        if row["fracture_positive"] == "True":
            positive += 1
        st = row.get("annotation_status", "UNAVAILABLE")
        annotation_status[st] = annotation_status.get(st, 0) + 1

    total = len(rows)
    record = FrozenDatasetRecord(
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        hash_algorithm=algorithm,
        dataset_yaml_hash=hash_file(dataset_yaml),
        manifest_hash=hash_file(manifest_path),
        validation_report_hash=hash_file(validation_report),
        total_images=total,
        total_labels=total_labels,
        total_boxes=total_boxes,
        train_images=split_counts["train"],
        val_images=split_counts["val"],
        test_images=split_counts["test"],
        positive_images=positive,
        negative_images=total - positive,
        by_dataset=by_dataset,
        by_dataset_split=by_dataset_split,
        annotation_status=annotation_status,
        file_count=len(file_hashes),
        files_record=files_record_name or f"{version}_files.json",
        jpeg_audit_summary=jpeg_audit_summary or {},
        notes=notes or [],
        environment={"python": platform.python_version(), "platform": platform.platform()},
    )
    return record, file_hashes


def save_freeze_record(record: FrozenDatasetRecord, path: Path, file_hashes: Dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    files_path = files_path_for(path)
    record.files_record = files_path.name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2, ensure_ascii=False)
    with open(files_path, "w", encoding="utf-8") as f:
        json.dump({"hash_algorithm": record.hash_algorithm, "files": file_hashes}, f, indent=1)
    logger.info(f"Freeze record saved: {path}  (+ {files_path.name}, {len(file_hashes)} files)")
    return path


def load_freeze_record(path: Path) -> Tuple[dict, Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    files_path = path.parent / record.get("files_record", files_path_for(path).name)
    with open(files_path, encoding="utf-8") as f:
        files = json.load(f)["files"]
    return record, files


# ── verify ─────────────────────────────────────────────────────────────────

def verify_freeze(
    freeze_record_path: Path,
    processed_dir: Path,
    dataset_yaml: Path,
    full: bool = False,
) -> Dict:
    """
    quick: manifest.csv + dataset.yaml hashes.
    full : additionally re-hash every recorded file and scan for unrecorded files.
    """
    if not freeze_record_path.exists():
        return {"ok": False, "mode": "quick", "reason": f"Freeze record not found: {freeze_record_path}"}

    try:
        saved, files = load_freeze_record(freeze_record_path)
    except Exception as e:
        return {"ok": False, "mode": "quick", "reason": f"Freeze record unreadable: {e}"}

    problems: List[str] = []
    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        problems.append("manifest.csv missing")
    elif hash_file(manifest_path) != saved["manifest_hash"]:
        problems.append("manifest.csv hash changed since freeze")
    if not dataset_yaml.exists():
        problems.append("dataset.yaml missing")
    elif hash_file(dataset_yaml) != saved["dataset_yaml_hash"]:
        problems.append("dataset.yaml hash changed since freeze")

    result: Dict = {"mode": "full" if full else "quick", "version": saved.get("version"),
                    "created_at": saved.get("created_at")}

    if full:
        algo = saved["hash_algorithm"]
        changed, missing = [], []
        for rel, expected in files.items():
            p = processed_dir / rel
            if not p.exists():
                missing.append(rel)
            elif hash_file(p, algo) != expected:
                changed.append(rel)
        recorded = set(files)
        unrecorded = []
        for p in processed_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(processed_dir))
            if rel == "manifest.csv" or p.suffix.lower() in TOLERATED_SUFFIXES:
                continue
            if rel not in recorded:
                unrecorded.append(rel)
        result.update({"files_checked": len(files), "changed": sorted(changed),
                       "missing": sorted(missing), "unrecorded": sorted(unrecorded)})
        if changed:
            problems.append(f"{len(changed)} recorded file(s) changed")
        if missing:
            problems.append(f"{len(missing)} recorded file(s) missing")
        if unrecorded:
            problems.append(f"{len(unrecorded)} unrecorded file(s) present in dataset tree")

    result["ok"] = not problems
    result["mismatches"] = problems
    (logger.info if result["ok"] else logger.warning)(
        f"Freeze verification {'PASSED' if result['ok'] else 'FAILED'} "
        f"[{result['mode']}] version={saved.get('version')} {problems if problems else ''}"
    )
    return result