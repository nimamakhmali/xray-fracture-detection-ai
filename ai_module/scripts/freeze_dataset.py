#!/usr/bin/env python3
"""
scripts/freeze_dataset.py

Freeze / verify the processed dataset for Phase 2.

Mandatory order:
    1. --audit-only              : see JPEG state + manifest/disk hash diff (no writes)
    2. --repair-jpegs            : lossless EOI repair (originals backed up)
    3. --sync-manifest-hashes    : write current disk hashes into manifest.csv (old manifest backed up)
    4. python scripts/validate_dataset.py      (must be re-run after 2/3)
    5. (no flags)                : create freeze record (gated on a READY, up-to-date validation report)
    6. --verify [--full]         : verify an existing freeze record

Nothing here regenerates or re-splits the dataset.
"""
import argparse
import shutil
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.dataset_freeze import (
    compare_manifest_to_disk, create_freeze_record, detect_hash_algorithm,
    hash_file, read_manifest_rows, save_freeze_record, verify_freeze,
)
from src.utils.jpeg_audit import audit_and_repair_directory
from src.utils.file_utils import save_json
from src.utils.logger import get_logger

logger = get_logger("freeze_dataset")
ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze / verify the processed dataset.")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    p.add_argument("--validation-report", default="reports/validation_report.json")
    p.add_argument("--version", default="frozen_v1")
    p.add_argument("--output", default=None, help="default: reports/dataset/<version>.json")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--repair-jpegs", action="store_true")
    p.add_argument("--backup-dir", default="data/jpeg_originals")
    p.add_argument("--sync-manifest-hashes", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--full", action="store_true", help="with --verify: re-hash every file")
    return p.parse_args()


def _jpeg_audit(processed_dir: Path, repair: bool, backup_dir: Path, version: str) -> dict:
    rep = audit_and_repair_directory(
        processed_dir, repair=repair, backup_dir=backup_dir if repair else None,
        report_path=ROOT / "reports" / "dataset" / f"{version}_jpeg_audit.json",
    )
    return {"total_scanned": rep.total_scanned, "ok": rep.ok, "needs_repair": rep.needs_repair,
            "repaired": rep.repaired, "unreadable": rep.unreadable, "not_jpeg": rep.not_jpeg,
            "repair_method": rep.repair_method}


def _manifest_diff(processed_dir: Path, version: str) -> dict:
    rows = read_manifest_rows(processed_dir / "manifest.csv")
    algo = detect_hash_algorithm(rows[0]["image_hash"])
    diff = compare_manifest_to_disk(processed_dir, rows, algo)
    save_json(diff, ROOT / "reports" / "dataset" / f"{version}_manifest_disk_diff.json")
    logger.info(f"Manifest/disk diff: checked={diff['checked']} mismatched={len(diff['mismatched'])} "
                f"missing={len(diff['missing'])} unlisted={len(diff['unlisted_on_disk'])} clean={diff['clean']}")
    for m in diff["mismatched"][:10]:
        logger.info(f"  mismatched: {m['path']}")
    return diff


def _sync_manifest(processed_dir: Path, diff: dict) -> int:
    """Rewrite manifest.csv image_hash for mismatched rows only. Old manifest backed up."""
    import csv
    if diff["missing"] or diff["unlisted_on_disk"]:
        logger.error("Refusing to sync: manifest has missing/unlisted images — that is a preparation problem, not a hash drift.")
        sys.exit(1)
    if not diff["mismatched"]:
        logger.info("Nothing to sync.")
        return 0
    manifest = processed_dir / "manifest.csv"
    backup_dir = ROOT / "data" / "manifest_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(manifest, backup_dir / f"manifest_{stamp}.csv")

    new_hash = {m["path"]: m["disk_hash"] for m in diff["mismatched"]}
    with open(manifest, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    changed = 0
    for r in rows:
        rel = f"{r['split']}/images/{r['image_path']}"
        if rel in new_hash:
            r["image_hash"] = new_hash[rel]
            changed += 1
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    save_json({"synced_at": stamp, "backup": str(backup_dir / f"manifest_{stamp}.csv"),
               "updated_rows": changed, "updated": diff["mismatched"]},
              ROOT / "reports" / "dataset" / f"manifest_hash_sync_{stamp}.json")
    logger.info(f"Manifest hashes synced: {changed} row(s). Backup: {backup_dir / f'manifest_{stamp}.csv'}")
    return changed


def main() -> None:
    a = parse_args()
    processed_dir = ROOT / a.processed_dir
    dataset_yaml = ROOT / a.dataset_yaml
    validation_report = ROOT / a.validation_report
    output = ROOT / (a.output or f"reports/dataset/{a.version}.json")
    (ROOT / "reports" / "dataset").mkdir(parents=True, exist_ok=True)

    for p in (processed_dir, dataset_yaml):
        if not p.exists():
            logger.error(f"Not found: {p}"); sys.exit(1)

    if a.verify:
        res = verify_freeze(output, processed_dir, dataset_yaml, full=a.full)
        save_json(res, ROOT / "reports" / "dataset" / f"{a.version}_verify_{res['mode']}.json")
        sys.exit(0 if res["ok"] else 1)

    if a.audit_only:
        _jpeg_audit(processed_dir, repair=False, backup_dir=None, version=a.version)
        _manifest_diff(processed_dir, a.version)
        return

    wrote = False
    jpeg_summary = None
    if a.repair_jpegs:
        jpeg_summary = _jpeg_audit(processed_dir, True, ROOT / a.backup_dir, a.version)
        wrote |= jpeg_summary["repaired"] > 0
    if a.sync_manifest_hashes:
        wrote |= _sync_manifest(processed_dir, _manifest_diff(processed_dir, a.version)) > 0
    if wrote:
        logger.warning("Dataset files/manifest were modified. Re-run scripts/validate_dataset.py, "
                       "then run this script with no flags to freeze.")
        return
    if a.repair_jpegs or a.sync_manifest_hashes:
        logger.info("No changes were necessary.")
        return

    # ── freeze ──────────────────────────────────────────────────────────────
    if not validation_report.exists():
        logger.error(f"Validation report missing: {validation_report}"); sys.exit(1)
    vr = json.loads(validation_report.read_text())
    if vr.get("status") != "READY_FOR_TRAINING":
        logger.error(f"Validation status is '{vr.get('status')}' — cannot freeze."); sys.exit(1)
    if validation_report.stat().st_mtime < (processed_dir / "manifest.csv").stat().st_mtime:
        logger.error("validation_report.json is OLDER than manifest.csv — re-run validate_dataset.py."); sys.exit(1)

    jpeg_summary = jpeg_summary or _jpeg_audit(processed_dir, False, None, a.version)
    if jpeg_summary["needs_repair"] or jpeg_summary["unreadable"]:
        logger.error("JPEG audit not clean — run with --repair-jpegs first."); sys.exit(1)

    record, file_hashes = create_freeze_record(
        processed_dir=processed_dir, dataset_yaml=dataset_yaml, validation_report=validation_report,
        version=a.version, seed=int(vr.get("manifest_summary", {}).get("random_seed", 42)),
        jpeg_audit_summary=jpeg_summary,
        notes=[
            f"patient_leakage (from validation report): {vr.get('patient_leakage')}",
            f"hash_leakage (from validation report): {vr.get('hash_leakage')}",
            "FracAtlas patient_id is UNAVAILABLE in source metadata; patient-level leakage unverifiable for that source.",
        ],
    )
    save_freeze_record(record, output, file_hashes)
    logger.info("=" * 60)
    logger.info(f"DATASET FROZEN: {record.version}")
    logger.info(f"  images={record.total_images} boxes={record.total_boxes} files_hashed={record.file_count}")
    logger.info(f"  by_dataset={record.by_dataset}")
    logger.info(f"  manifest_hash={record.manifest_hash[:16]}...  yaml_hash={record.dataset_yaml_hash[:16]}...")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()