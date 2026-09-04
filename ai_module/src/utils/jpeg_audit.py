"""
src/utils/jpeg_audit.py

Why this exists
---------------
Ultralytics (ultralytics/data/utils.py::verify_image_label) does exactly one
structural check on JPEGs and REWRITES the file in place if it fails:

    f.seek(-2, 2); if f.read() != b"\\xff\\xd9":   # missing EOI marker
        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
        -> "corrupt JPEG restored and saved"

That rewrite is a lossy re-encode and silently mutates a frozen dataset.
Pillow with LOAD_TRUNCATED_IMAGES=False decodes such files fine (libjpeg treats
a missing EOI as a warning when the scan data is complete), which is why our
validator reported 0 corrupt images while Ultralytics rewrote 26 of them.

What this module does
---------------------
* detect: exactly the Ultralytics criterion (PIL format == JPEG and last 2 bytes != FF D9)
* repair: LOSSLESS — append the 2-byte EOI marker; pixel data is untouched
* guard : a file must fully decode with PIL before and after repair, else it is
          reported as unreadable and NOT modified
* audit trail: original hash, repaired hash, backup path
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageFile

from src.utils.logger import get_logger
from src.utils.file_utils import save_json
from src.utils.dataset_freeze import hash_file

logger = get_logger(__name__)

EOI = b"\xff\xd9"


@dataclass
class JPEGAuditRecord:
    path: str                 # relative to scanned directory
    status: str               # ok | needs_repair | repaired | unreadable | not_jpeg
    detail: str
    original_hash: str
    repaired_hash: str = ""
    backup_path: str = ""


@dataclass
class JPEGAuditReport:
    directory: str = ""
    total_scanned: int = 0
    ok: int = 0
    needs_repair: int = 0
    repaired: int = 0
    unreadable: int = 0
    not_jpeg: int = 0
    repair_method: str = "append_eoi_marker_lossless"
    records: List[JPEGAuditRecord] = field(default_factory=list)


def _pil_full_decode_ok(path: Path) -> Tuple[bool, str]:
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as im:
            fmt = (im.format or "").upper()
        with Image.open(path) as im:
            im.load()
        return True, fmt
    except Exception as e:
        return False, f"pil_decode_failed: {e}"
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev


def ultralytics_will_rewrite(path: Path) -> Tuple[bool, str]:
    """Mirror of the Ultralytics check. Returns (will_rewrite, detail)."""
    ok, fmt_or_err = _pil_full_decode_ok(path)
    if not ok:
        return False, fmt_or_err
    if fmt_or_err != "JPEG":
        return False, f"not_jpeg:{fmt_or_err}"
    with open(path, "rb") as f:
        f.seek(-2, 2)
        tail = f.read()
    return (tail != EOI), ("missing_eoi_marker" if tail != EOI else "ok")


def repair_jpeg_lossless(path: Path, backup_dir: Optional[Path], rel: str) -> Tuple[bool, str, str]:
    """Append EOI marker. Returns (success, detail, backup_path)."""
    backup_path = ""
    if backup_dir is not None:
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():                      # never overwrite an earlier original
            shutil.copy2(path, dst)
        backup_path = str(dst)
    with open(path, "ab") as f:
        f.write(EOI)
    ok, err = _pil_full_decode_ok(path)
    if not ok:
        # roll back from backup if we have one
        if backup_path:
            shutil.copy2(backup_path, path)
        return False, f"post_repair_decode_failed_rolled_back: {err}", backup_path
    still, _ = ultralytics_will_rewrite(path)
    if still:
        return False, "post_repair_still_flagged", backup_path
    return True, "eoi_appended", backup_path


def audit_and_repair_directory(
    directory: Path,
    repair: bool = False,
    backup_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    hash_algorithm: str = "sha256",
) -> JPEGAuditReport:
    directory = Path(directory)
    report = JPEGAuditReport(directory=str(directory))
    files = sorted(p for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    report.total_scanned = len(files)
    logger.info(f"JPEG audit: {len(files)} files under {directory} (repair={repair})")

    for p in files:
        rel = str(p.relative_to(directory))
        orig_hash = hash_file(p, hash_algorithm)
        will_rewrite, detail = ultralytics_will_rewrite(p)

        if detail.startswith("pil_decode_failed"):
            report.unreadable += 1
            report.records.append(JPEGAuditRecord(rel, "unreadable", detail, orig_hash))
            continue
        if detail.startswith("not_jpeg"):
            report.not_jpeg += 1
            report.records.append(JPEGAuditRecord(rel, "not_jpeg", detail, orig_hash))
            continue
        if not will_rewrite:
            report.ok += 1
            report.records.append(JPEGAuditRecord(rel, "ok", "ok", orig_hash))
            continue

        report.needs_repair += 1
        if not repair:
            report.records.append(JPEGAuditRecord(rel, "needs_repair", detail, orig_hash))
            continue

        success, rdetail, backup = repair_jpeg_lossless(p, backup_dir, rel)
        if success:
            report.repaired += 1
            report.records.append(JPEGAuditRecord(rel, "repaired", f"{detail} -> {rdetail}",
                                                  orig_hash, hash_file(p, hash_algorithm), backup))
            logger.info(f"  repaired (lossless): {rel}")
        else:
            report.unreadable += 1
            report.records.append(JPEGAuditRecord(rel, "unreadable", f"{detail} -> {rdetail}",
                                                  orig_hash, "", backup))
            logger.warning(f"  repair FAILED: {rel} — {rdetail}")

    logger.info(f"JPEG audit done: ok={report.ok} needs_repair={report.needs_repair} "
                f"repaired={report.repaired} unreadable={report.unreadable} not_jpeg={report.not_jpeg}")
    if report_path:
        save_json(asdict(report), report_path)
    return report