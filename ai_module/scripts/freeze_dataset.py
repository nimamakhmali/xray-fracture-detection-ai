#!/usr/bin/env python3
"""
scripts/freeze_dataset.py

Creates the official frozen dataset record for Phase 2 baseline.

This script must be run ONCE after:
  1. prepare_dataset.py has been executed
  2. validate_dataset.py has passed
  3. JPEG audit has been performed

After running this script, the dataset is considered FROZEN.
Training runs will verify against this record.

Usage:
    python scripts/freeze_dataset.py
    python scripts/freeze_dataset.py --repair-jpegs
    python scripts/freeze_dataset.py --repair-jpegs --backup-dir data/jpeg_originals
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.dataset_freeze import create_freeze_record, save_freeze_record
from src.utils.jpeg_audit import audit_and_repair_directory
from src.utils.file_utils import save_json
from src.utils.logger import get_logger

logger = get_logger("freeze_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Freeze the processed dataset for Phase 2 baseline."
    )
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    p.add_argument("--output", default="reports/frozen_dataset_v1.json")
    p.add_argument(
        "--repair-jpegs", action="store_true",
        help="Pre-repair JPEG files to prevent Ultralytics in-place modification.",
    )
    p.add_argument(
        "--backup-dir", default="data/jpeg_originals",
        help="Directory to back up original JPEGs before repair.",
    )
    p.add_argument("--version", default="frozen_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    processed_dir = root / args.processed_dir
    dataset_yaml = root / args.dataset_yaml
    output_path = root / args.output

    if not processed_dir.exists():
        logger.error(f"Processed dir not found: {processed_dir}")
        sys.exit(1)

    if not dataset_yaml.exists():
        logger.error(f"dataset.yaml not found: {dataset_yaml}")
        sys.exit(1)

    jpeg_audit_summary = {}

    if args.repair_jpegs:
        logger.info("Running JPEG audit and repair...")
        backup_dir = root / args.backup_dir if args.backup_dir else None
        audit_report = audit_and_repair_directory(
            directory=processed_dir,
            repair=True,
            backup_dir=backup_dir,
            report_path=root / "reports" / "jpeg_audit.json",
        )
        jpeg_audit_summary = {
            "total_scanned": audit_report.total_scanned,
            "ok": audit_report.ok_count,
            "repairable_found": audit_report.repairable_count,
            "repaired": audit_report.repaired_count,
            "unreadable": audit_report.unreadable_count,
        }
        logger.info(f"JPEG audit complete: {jpeg_audit_summary}")
    else:
        logger.warning(
            "JPEG repair skipped. Run with --repair-jpegs to prevent "
            "Ultralytics from modifying frozen dataset during training."
        )
        jpeg_audit_summary = {"status": "skipped"}

    logger.info("Creating freeze record...")
    record = create_freeze_record(
        processed_dir=processed_dir,
        dataset_yaml=dataset_yaml,
        version=args.version,
        jpeg_audit_summary=jpeg_audit_summary,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        record.__dict__ if hasattr(record, '__dict__') else vars(record),
        output_path,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"DATASET FROZEN: {args.version}")
    logger.info(f"  Total images  : {record.total_images}")
    logger.info(f"  Total boxes   : {record.total_boxes}")
    logger.info(f"  Manifest hash : {record.manifest_hash[:16]}...")
    logger.info(f"  YAML hash     : {record.dataset_yaml_hash[:16]}...")
    logger.info(f"  Saved to      : {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()