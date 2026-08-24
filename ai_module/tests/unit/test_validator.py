"""
Unit tests for src/data/validator.py

All tests use synthetic temporary data.
No real medical dataset required.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.validator import DatasetValidator, VALID_CLASS_IDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path: Path, spec: dict) -> Path:
    """
    Create a minimal YOLO dataset from a spec dict.

    spec = {
        "train": {
            "img_001": "0 0.5 0.5 0.2 0.2\n",   # valid positive
            "img_002": "",                          # empty label = negative
            "img_003": None,                        # no label file at all
        }
    }

    Returns the processed root path.
    """
    processed = tmp_path / "processed"
    for split, samples in spec.items():
        (processed / split / "images").mkdir(parents=True, exist_ok=True)
        (processed / split / "labels").mkdir(parents=True, exist_ok=True)

        for stem, label_content in samples.items():
            # Create a tiny valid image
            img = np.ones((64, 64, 3), dtype=np.uint8) * 200
            cv2.imwrite(str(processed / split / "images" / f"{stem}.jpg"), img)

            if label_content is not None:
                (processed / split / "labels" / f"{stem}.txt").write_text(
                    label_content, encoding="utf-8"
                )

    return processed


# ---------------------------------------------------------------------------
# Tests — basic structure
# ---------------------------------------------------------------------------

class TestValidatorStructure:

    def test_missing_split_dir_is_critical_error(self, tmp_path):
        processed = tmp_path / "empty_processed"
        processed.mkdir()
        v = DatasetValidator(processed_dir=processed)
        report = v.validate(save_report=False)
        assert report.status == "NOT_READY_FOR_TRAINING"
        assert len(report.critical_errors) > 0

    def test_empty_dataset_not_ready(self, tmp_path):
        spec = {"train": {}, "val": {}, "test": {}}
        processed = _make_dataset(tmp_path, spec)
        v = DatasetValidator(processed_dir=processed)
        report = v.validate(save_report=False)
        assert report.status == "NOT_READY_FOR_TRAINING"

    def test_valid_dataset_is_ready(self, tmp_path):
        spec = {
            "train": {f"img_{i:03d}": "0 0.5 0.5 0.2 0.2\n" for i in range(10)},
            "val":   {f"img_{i:03d}": "0 0.4 0.4 0.1 0.1\n" for i in range(3)},
            "test":  {f"img_{i:03d}": "0 0.6 0.6 0.15 0.15\n" for i in range(3)},
        }
        processed = _make_dataset(tmp_path, spec)
        v = DatasetValidator(processed_dir=processed)
        report = v.validate(save_report=False)
        assert report.status == "READY_FOR_TRAINING"
        assert report.total_images == 16
        assert report.positive_images == 16


# ---------------------------------------------------------------------------
# Tests — negative samples
# ---------------------------------------------------------------------------

class TestNegativeSamples:

    def test_empty_label_file_is_negative_not_error(self, tmp_path):
        spec = {
            "train": {"neg_001": "", "neg_002": ""},
            "val": {"neg_003": ""},
            "test": {"neg_004": ""},
        }
        processed = _make_dataset(tmp_path, spec)
        v = DatasetValidator(processed_dir=processed, allow_empty_labels=True)
        report = v.validate(save_report=False)
        # Negative images should be counted, not flagged as errors
        assert report.negative_images == 4
        assert report.positive_images == 0
        # Status depends on whether we have at least some images
        # Empty labels are not ERROR → can be READY if no critical issues
        error_issues = [i for i in report.issues if i.get("severity") == "ERROR"]
        assert len(error_issues) == 0

    def test_missing_label_flagged_as_warning_in_permissive_mode(self, tmp_path):
        spec = {
            "train": {"img_001": None},  # no label file
            "val": {"img_002": "0 0.5 0.5 0.2 0.2\n"},
            "test": {"img_003": "0 0.5 0.5 0.2 0.2\n"},
        }
        processed = _make_dataset(tmp_path, spec)
        v = DatasetValidator(processed_dir=processed, allow_empty_labels=True)
        report = v.validate(save_report=False)
        assert report.missing_labels == 1


# ---------------------------------------------------------------------------
# Tests — label validation
# ---------------------------------------------------------------------------

class TestLabelValidation:

    def test_valid_yolo_line_passes(self, tmp_path):
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label.__func__(v, tmp_path / "dummy.txt", None)
        # No file = missing issue
        assert any(i.issue_type == "MISSING" for i in issues)

    def test_validate_valid_label_content(self, tmp_path):
        label = tmp_path / "good.txt"
        label.write_text("0 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\n")
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        assert issues == []

    def test_wrong_field_count(self, tmp_path):
        label = tmp_path / "bad.txt"
        label.write_text("0 0.5 0.5 0.2\n")  # only 4 fields
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        assert any(i.issue_type == "MALFORMED" for i in issues)

    def test_out_of_range_x_center(self, tmp_path):
        label = tmp_path / "bad.txt"
        label.write_text("0 1.5 0.5 0.2 0.2\n")  # x_center > 1
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        assert any(i.issue_type == "OUT_OF_RANGE" for i in issues)

    def test_zero_width_box(self, tmp_path):
        label = tmp_path / "bad.txt"
        label.write_text("0 0.5 0.5 0.0 0.2\n")  # width = 0
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        types = {i.issue_type for i in issues}
        assert "OUT_OF_RANGE" in types or "ZERO_AREA" in types

    def test_negative_height(self, tmp_path):
        label = tmp_path / "bad.txt"
        label.write_text("0 0.5 0.5 0.2 -0.1\n")
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        assert any(i.issue_type == "OUT_OF_RANGE" for i in issues)

    def test_unknown_class_id(self, tmp_path):
        label = tmp_path / "bad.txt"
        label.write_text("5 0.5 0.5 0.2 0.2\n")  # class 5 not in VALID_CLASS_IDS
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        assert any(i.issue_type == "UNKNOWN_CLASS" for i in issues)

    def test_nan_value_detected(self, tmp_path):
        label = tmp_path / "bad.txt"
        label.write_text("0 nan 0.5 0.2 0.2\n")
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        assert len(issues) > 0

    def test_duplicate_annotation_in_same_file(self, tmp_path):
        label = tmp_path / "dup.txt"
        label.write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        assert any(i.issue_type == "DUPLICATE_ANNOTATION" for i in issues)

    def test_non_numeric_field(self, tmp_path):
        label = tmp_path / "bad.txt"
        label.write_text("0 abc 0.5 0.2 0.2\n")
        v = DatasetValidator(processed_dir=tmp_path)
        issues = v.validate_single_label(label)
        assert any(i.issue_type == "MALFORMED" for i in issues)


# ---------------------------------------------------------------------------
# Tests — corrupted images
# ---------------------------------------------------------------------------

class TestCorruptedImages:

    def test_zero_byte_image_flagged(self, tmp_path):
        processed = tmp_path / "processed"
        for split in ("train", "val", "test"):
            (processed / split / "images").mkdir(parents=True, exist_ok=True)
            (processed / split / "labels").mkdir(parents=True, exist_ok=True)

        # Create a zero-byte image
        img_path = processed / "train" / "images" / "corrupt.jpg"
        img_path.write_bytes(b"")

        v = DatasetValidator(processed_dir=processed)
        report = v.validate(save_report=False)
        assert report.corrupted_images >= 1


# ---------------------------------------------------------------------------
# Tests — report output
# ---------------------------------------------------------------------------

class TestReportOutput:

    def test_report_saved_to_disk(self, tmp_path):
        spec = {
            "train": {"img_001": "0 0.5 0.5 0.2 0.2\n"},
            "val":   {"img_002": "0 0.4 0.4 0.1 0.1\n"},
            "test":  {"img_003": "0 0.6 0.6 0.1 0.1\n"},
        }
        processed = _make_dataset(tmp_path, spec)
        report_dir = tmp_path / "reports"
        v = DatasetValidator(
            processed_dir=processed,
            report_dir=report_dir,
        )
        v.validate(save_report=True)
        assert (report_dir / "validation_report.json").exists()

    def test_split_stats_present(self, tmp_path):
        spec = {
            "train": {"img_001": "0 0.5 0.5 0.2 0.2\n"},
            "val":   {"img_002": ""},
            "test":  {"img_003": "0 0.6 0.6 0.1 0.1\n"},
        }
        processed = _make_dataset(tmp_path, spec)
        v = DatasetValidator(processed_dir=processed)
        report = v.validate(save_report=False)
        for split in ("train", "val", "test"):
            assert split in report.split_stats