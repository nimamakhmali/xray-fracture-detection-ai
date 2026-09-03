#!/usr/bin/env python3
"""
scripts/validate_dataset.py

Convenience wrapper around DatasetValidator with --skip-integrity-recheck.

Usage:
    # بار اول یا CI — کامل و کند:
    python scripts/validate_dataset.py

    # بلافاصله بعد از prepare_dataset.py — سریع:
    python scripts/validate_dataset.py --skip-integrity-recheck
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.validator import DatasetValidator
from src.utils.logger import get_logger

logger = get_logger("validate_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--dataset-yaml", default=None,
                   help="اگر نداری، auto-detect می‌کند")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--strict", action="store_true")
    p.add_argument(
        "--skip-integrity-recheck", action="store_true",
        help=(
            "از PIL integrity scan صرف نظر کن. "
            "فقط وقتی مطمئنی prepare_dataset.py همین لحظه اجرا شده استفاده کن."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.data)

    if not processed_dir.exists():
        logger.error(f"Processed dir not found: {processed_dir}")
        logger.error("Run: python scripts/prepare_dataset.py --verbose")
        sys.exit(1)

    if not (processed_dir / "manifest.csv").exists():
        logger.error(f"manifest.csv not found in {processed_dir}")
        logger.error("Run: python scripts/prepare_dataset.py --verbose")
        sys.exit(1)

    if args.skip_integrity_recheck:
        logger.warning(
            "INTEGRITY RECHECK SKIPPED — "
            "assuming prepare_dataset.py already filtered bad files."
        )

    validator = DatasetValidator(
        processed_dir=processed_dir,
        allow_empty_labels=not args.strict,
        report_dir=Path(args.report_dir),
        dataset_yaml_path=Path(args.dataset_yaml) if args.dataset_yaml else None,
        skip_integrity_recheck=args.skip_integrity_recheck,
    )
    report = validator.validate()

    print("\n" + "=" * 60)
    is_ready = report.status == "READY_FOR_TRAINING"
    print(f"DATASET AUDIT STATUS : {'✅ PASS' if is_ready else '❌ FAIL'}")
    print(f"TRAINING STATUS      : {'✅ READY' if is_ready else '🚫 BLOCKED'}")
    print("=" * 60)

    if not is_ready:
        print("\nBLOCKERS:")
        for err in report.critical_errors:
            print(f"  ❌ {err}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()