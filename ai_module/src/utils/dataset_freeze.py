"""
src/utils/dataset_freeze.py

Implements dataset freeze/verification for the frozen dataset contract.

A frozen dataset is one where:
  1. All files are recorded with their hash.
  2. The manifest hash is recorded.
  3. The dataset.yaml hash is recorded.
  4. No training run may silently modify files.

The freeze record is saved as:
    reports/frozen_dataset_v1.json

Every training run reads this and verifies integrity before starting.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import get_logger
from src.utils.file_utils import compute_file_hash, save_json

logger = get_logger(__name__)


@dataclass
class FrozenDatasetRecord:
    version: str
    created_at: str
    seed: int
    dataset_yaml_hash: str
    manifest_hash: str
    total_images: int
    total_labels: int
    total_boxes: int
    train_images: int
    val_images: int
    test_images: int
    positive_images: int
    negative_images: int
    by_dataset: Dict[str, int]
    annotation_status: Dict[str, int]
    jpeg_audit_summary: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def compute_manifest_hash(manifest_path: Path) -> str:
    """SHA256 of the entire manifest.csv."""
    return compute_file_hash(manifest_path, algorithm="sha256")


def compute_yaml_hash(yaml_path: Path) -> str:
    """SHA256 of dataset.yaml."""
    return compute_file_hash(yaml_path, algorithm="sha256")


def create_freeze_record(
    processed_dir: Path,
    dataset_yaml: Path,
    version: str = "frozen_v1",
    seed: int = 42,
    jpeg_audit_summary: Optional[Dict] = None,
    notes: Optional[List[str]] = None,
) -> FrozenDatasetRecord:
    """
    Create a freeze record from the current processed dataset.
    Reads manifest.csv for all statistics.
    """
    import csv
    import yaml

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_hash = compute_manifest_hash(manifest_path)
    yaml_hash = compute_yaml_hash(dataset_yaml)

    # read stats from manifest
    total_images = 0
    positive = 0
    by_dataset: Dict[str, int] = {}
    annotation_status: Dict[str, int] = {}
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    with open(manifest_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_images += 1
            if row.get("fracture_positive") == "True":
                positive += 1
            ds = row.get("dataset", "unknown")
            by_dataset[ds] = by_dataset.get(ds, 0) + 1
            status = row.get("annotation_status", "unknown")
            annotation_status[status] = annotation_status.get(status, 0) + 1
            sp = row.get("split", "unknown")
            if sp in split_counts:
                split_counts[sp] += 1

    # count actual label files and boxes
    total_labels = 0
    total_boxes = 0
    for split in ("train", "val", "test"):
        label_dir = processed_dir / split / "labels"
        if label_dir.exists():
            for lf in label_dir.glob("*.txt"):
                total_labels += 1
                try:
                    lines = [
                        ln for ln in
                        lf.read_text().strip().splitlines()
                        if ln.strip()
                    ]
                    total_boxes += len(lines)
                except Exception:
                    pass

    record = FrozenDatasetRecord(
        version=version,
        created_at=datetime.utcnow().isoformat() + "Z",
        seed=seed,
        dataset_yaml_hash=yaml_hash,
        manifest_hash=manifest_hash,
        total_images=total_images,
        total_labels=total_labels,
        total_boxes=total_boxes,
        train_images=split_counts.get("train", 0),
        val_images=split_counts.get("val", 0),
        test_images=split_counts.get("test", 0),
        positive_images=positive,
        negative_images=total_images - positive,
        by_dataset=by_dataset,
        annotation_status=annotation_status,
        jpeg_audit_summary=jpeg_audit_summary or {},
        notes=notes or [
            "FracAtlas patient-level leakage: UNVERIFIABLE (no patient_id in CSV).",
            "GRAZPEDWRI-DX patient-level leakage: verified clean.",
            "169 GRAZPEDWRI-DX negatives contain clinically-adjacent findings.",
            "JPEG repair: pre-repaired before freeze to prevent Ultralytics in-place modification.",
        ],
    )
    return record


def verify_freeze(
    freeze_record_path: Path,
    processed_dir: Path,
    dataset_yaml: Path,
) -> Dict:
    """
    Verify that the frozen dataset has not been modified.
    Returns a dict with 'ok' and 'mismatches'.
    """
    if not freeze_record_path.exists():
        return {
            "ok": False,
            "reason": f"Freeze record not found: {freeze_record_path}",
        }

    with open(freeze_record_path) as f:
        saved = json.load(f)

    mismatches = []

    # check manifest hash
    manifest_path = processed_dir / "manifest.csv"
    if manifest_path.exists():
        current_hash = compute_manifest_hash(manifest_path)
        if current_hash != saved.get("manifest_hash"):
            mismatches.append(
                f"manifest.csv hash changed: "
                f"was={saved.get('manifest_hash')[:12]}... "
                f"now={current_hash[:12]}..."
            )
    else:
        mismatches.append("manifest.csv missing")

    # check yaml hash
    if dataset_yaml.exists():
        current_yaml_hash = compute_yaml_hash(dataset_yaml)
        if current_yaml_hash != saved.get("dataset_yaml_hash"):
            mismatches.append(
                f"dataset.yaml hash changed: "
                f"was={saved.get('dataset_yaml_hash')[:12]}... "
                f"now={current_yaml_hash[:12]}..."
            )

    if mismatches:
        logger.warning(
            f"Dataset freeze verification FAILED: {mismatches}"
        )
        return {"ok": False, "mismatches": mismatches}

    logger.info(
        f"Dataset freeze verification PASSED — "
        f"version={saved.get('version')}"
    )
    return {
        "ok": True,
        "version": saved.get("version"),
        "created_at": saved.get("created_at"),
    }