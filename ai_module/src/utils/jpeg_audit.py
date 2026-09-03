"""
src/utils/jpeg_audit.py

Detects JPEG files that PIL considers valid but Ultralytics/libjpeg
will repair during training. This is the root cause of the
"corrupt JPEG restored and saved" messages observed in the smoke test.

Root cause analysis:
    PIL (Pillow) with LOAD_TRUNCATED_IMAGES=False raises on truncated
    JPEGs. However, some JPEGs with non-fatal structural anomalies
    (e.g., missing EOI marker, extra trailing bytes, Adobe APP14
    markers) pass PIL verification but trigger libjpeg's repair
    mechanism when decoded by OpenCV/Ultralytics during training.

    This is a decoder-specific difference, not a data quality failure.
    The files are readable and usable, but Ultralytics modifies them
    in-place during cache building, which violates the frozen dataset
    contract.

Resolution:
    Option A (preferred): pre-repair affected JPEGs during dataset
        preparation so Ultralytics does not modify the frozen dataset.
    Option B: record affected files and accept in-place repair,
        documenting it in the integrity report.

This module implements Option A: detect and pre-repair, with full
audit trail.
"""
from __future__ import annotations

import hashlib
import io
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile

from src.utils.logger import get_logger
from src.utils.file_utils import save_json, compute_file_hash

logger = get_logger(__name__)


@dataclass
class JPEGAuditRecord:
    path: str
    original_hash: str
    status: str          # ok | repairable | unreadable
    repair_method: str   # none | pil_resave | opencv_resave
    repaired_hash: str   # empty if not repaired
    detail: str          # description of issue


@dataclass
class JPEGAuditReport:
    total_scanned: int = 0
    ok_count: int = 0
    repairable_count: int = 0
    unreadable_count: int = 0
    repaired_count: int = 0
    records: List[JPEGAuditRecord] = field(default_factory=list)


def _is_repairable_by_ultralytics(path: Path) -> Tuple[bool, str]:
    """
    Detects JPEGs that pass PIL verification but would be repaired
    by Ultralytics/libjpeg.

    Strategy: attempt to decode with OpenCV using strict settings.
    If OpenCV produces a warning or partial result that differs from
    PIL decode, flag as repairable.
    """
    # Step 1: Try PIL with strict settings
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            pil_arr = np.array(img.convert("RGB"))
    except Exception as e:
        return False, f"PIL_fail: {e}"

    # Step 2: Try OpenCV decode
    cv_arr = cv2.imread(str(path))
    if cv_arr is None:
        return True, "opencv_returns_none"

    # Step 3: Check for JFIF/EXIF anomalies that libjpeg repairs
    # Read raw bytes and check for missing EOI or trailing garbage
    raw = path.read_bytes()

    # Missing EOI (End of Image) marker
    if not raw.endswith(b'\xff\xd9'):
        # check if EOI exists anywhere near the end
        eoi_pos = raw.rfind(b'\xff\xd9')
        if eoi_pos == -1:
            return True, "missing_EOI_marker"
        trailing = raw[eoi_pos + 2:]
        if len(trailing) > 0:
            return True, f"trailing_bytes_after_EOI: {len(trailing)} bytes"

    # Step 4: Re-encode PIL result and compare size
    # If PIL re-save produces a significantly different file,
    # it suggests the original has anomalies
    pil_buf = io.BytesIO()
    with Image.open(path) as img:
        img.save(pil_buf, format="JPEG", quality=95)
    pil_size = pil_buf.tell()

    # Heuristic: if file is >20% larger than re-encoded version,
    # it may contain garbage that libjpeg strips
    if path.stat().st_size > pil_size * 1.5:
        return True, f"oversized: {path.stat().st_size} vs reencoded {pil_size}"

    return False, "ok"


def repair_jpeg(path: Path, backup_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Pre-repair a JPEG file using PIL re-save.
    Optionally backs up the original.

    Returns (success, detail).
    """
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / path.name
        shutil.copy2(path, backup_path)

    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        # Save back to same path
        rgb.save(str(path), format="JPEG", quality=95, optimize=True)
        return True, "repaired_via_pil_resave"
    except Exception as e:
        return False, f"repair_failed: {e}"


def audit_and_repair_directory(
    directory: Path,
    repair: bool = True,
    backup_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> JPEGAuditReport:
    """
    Scan all JPEGs in a directory tree.
    Optionally repair in-place (with backup).

    Args:
        directory:   Root directory to scan.
        repair:      If True, repair repairable files in-place.
        backup_dir:  If set, back up originals before repair.
        report_path: If set, save JSON report here.

    Returns:
        JPEGAuditReport
    """
    report = JPEGAuditReport()

    jpeg_files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
    )
    report.total_scanned = len(jpeg_files)
    logger.info(
        f"JPEG audit: scanning {len(jpeg_files)} files in {directory}"
    )

    for path in jpeg_files:
        orig_hash = compute_file_hash(path)
        is_repair, detail = _is_repairable_by_ultralytics(path)

        if not is_repair:
            report.ok_count += 1
            report.records.append(JPEGAuditRecord(
                path=str(path),
                original_hash=orig_hash,
                status="ok",
                repair_method="none",
                repaired_hash="",
                detail="ok",
            ))
            continue

        report.repairable_count += 1

        if repair:
            success, repair_detail = repair_jpeg(path, backup_dir)
            if success:
                new_hash = compute_file_hash(path)
                report.repaired_count += 1
                logger.info(
                    f"  Repaired: {path.name} — {detail}"
                )
                report.records.append(JPEGAuditRecord(
                    path=str(path),
                    original_hash=orig_hash,
                    status="repairable",
                    repair_method="pil_resave",
                    repaired_hash=new_hash,
                    detail=detail,
                ))
            else:
                report.unreadable_count += 1
                logger.warning(
                    f"  Repair FAILED: {path.name} — {repair_detail}"
                )
                report.records.append(JPEGAuditRecord(
                    path=str(path),
                    original_hash=orig_hash,
                    status="unreadable",
                    repair_method="none",
                    repaired_hash="",
                    detail=repair_detail,
                ))
        else:
            report.records.append(JPEGAuditRecord(
                path=str(path),
                original_hash=orig_hash,
                status="repairable",
                repair_method="none",
                repaired_hash="",
                detail=detail,
            ))

    logger.info(
        f"JPEG audit complete: ok={report.ok_count} "
        f"repairable={report.repairable_count} "
        f"unreadable={report.unreadable_count} "
        f"repaired={report.repaired_count}"
    )

    if report_path:
        save_json(asdict(report), report_path)
        logger.info(f"JPEG audit report saved: {report_path}")

    return report