"""
src/training/evaluator.py

Evaluation pipeline for the fracture detection model.

Key design principles:
  1. Test set is SACRED — never used for model selection.
  2. Metrics are reported SEPARATELY for:
       - Combined dataset
       - FracAtlas subset
       - GRAZPEDWRI-DX subset
     to expose domain shift.
  3. Visual samples (TP/FP/FN) are saved for qualitative analysis.
  4. The 169 clinically-adjacent GRAZPEDWRI negatives are tracked
     separately in error analysis.
  5. No metric inflation — confidence threshold is not tuned on test set.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from src.utils.logger import get_logger
from src.utils.file_utils import save_json
from src.data.manifest import ManifestStore, UNAVAILABLE

logger = get_logger(__name__)


# ── result structures ─────────────────────────────────────────────────────────

@dataclass
class DetectionMetrics:
    """Standard object detection metrics for one evaluation subset."""
    subset: str                    # combined | fracatlas | grazpedwri
    split: str                     # val | test
    num_images: int = 0
    num_positive_images: int = 0
    num_negative_images: int = 0
    num_gt_boxes: int = 0
    num_predicted_boxes: int = 0
    precision: float = -1.0
    recall: float = -1.0
    f1: float = -1.0
    map50: float = -1.0
    map50_95: float = -1.0
    # image-level (presence/absence)
    image_level_tp: int = 0
    image_level_fp: int = 0
    image_level_fn: int = 0
    image_level_tn: int = 0
    # timing
    inference_time_ms_mean: float = -1.0
    inference_time_ms_total: float = -1.0


@dataclass
class EvaluationReport:
    weights: str
    split: str
    confidence_threshold: float
    iou_threshold: float
    dataset_version: str
    combined: DetectionMetrics = field(
        default_factory=lambda: DetectionMetrics("combined", "")
    )
    fracatlas: DetectionMetrics = field(
        default_factory=lambda: DetectionMetrics("fracatlas", "")
    )
    grazpedwri: DetectionMetrics = field(
        default_factory=lambda: DetectionMetrics("grazpedwri", "")
    )
    error_analysis: Dict = field(default_factory=dict)
    domain_shift_summary: Dict = field(default_factory=dict)
    known_limitations: List[str] = field(default_factory=list)


# ── evaluator ─────────────────────────────────────────────────────────────────

class FractureDetectionEvaluator:
    """
    Evaluates a YOLOv8 fracture detection model.

    Provides:
      - Combined metrics (val or test)
      - Per-source metrics (FracAtlas / GRAZPEDWRI-DX)
      - Visual samples (TP / FP / FN)
      - Error analysis
      - Domain shift summary
    """

    def __init__(
        self,
        weights: Path,
        dataset_yaml: Path,
        manifest_path: Path,
        output_dir: Path,
        reports_dir: Optional[Path] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        max_visual_samples: int = 20,
    ):
        self.weights = Path(weights).resolve()
        self.dataset_yaml = Path(dataset_yaml).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.reports_dir = (
            Path(reports_dir).resolve() if reports_dir
            else self.output_dir.parent / "reports" / "evaluation"
        )
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self.max_visual_samples = max_visual_samples

        self._root = Path(__file__).resolve().parent.parent.parent
        self._processed_dir = self._root / "data" / "processed"

        # load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )
        self._manifest = ManifestStore.load(self.manifest_path)
        logger.info(
            f"Manifest loaded: {len(self._manifest.records)} records"
        )

        # load dataset cfg
        with open(self.dataset_yaml) as f:
            self._dataset_cfg = yaml.safe_load(f) or {}

    # ── public ───────────────────────────────────────────────────────────────

    def evaluate(self, split: str = "val") -> EvaluationReport:
        """
        Full evaluation on the given split.

        Args:
            split: 'val' or 'test'
                   Use 'val' for model selection / threshold tuning.
                   Use 'test' ONLY for final unbiased evaluation.
        """
        if split not in ("val", "test"):
            raise ValueError(f"split must be 'val' or 'test', got '{split}'")

        if split == "test":
            logger.warning(
                "TEST SET EVALUATION — this should only be run once "
                "after model/threshold selection is finalized on val set. "
                "Do NOT use test metrics for any further tuning."
            )

        logger.info("=" * 60)
        logger.info(f"EVALUATION — split={split}")
        logger.info(f"  Weights : {self.weights}")
        logger.info(f"  Conf    : {self.conf}")
        logger.info(f"  IoU     : {self.iou}")
        logger.info("=" * 60)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # step 1: run Ultralytics val on full split
        ultralytics_metrics = self._run_ultralytics_val(split)

        # step 2: run per-image inference for per-source metrics + visuals
        split_records = self._manifest.by_split(split)
        fa_records = [r for r in split_records if r.dataset == "fracatlas"]
        grz_records = [r for r in split_records if r.dataset == "grazpedwri"]

        logger.info(
            f"Split '{split}': total={len(split_records)}, "
            f"fracatlas={len(fa_records)}, "
            f"grazpedwri={len(grz_records)}"
        )

        # step 3: per-source inference
        fa_metrics, fa_samples = self._evaluate_subset(
            fa_records, split, "fracatlas"
        )
        grz_metrics, grz_samples = self._evaluate_subset(
            grz_records, split, "grazpedwri"
        )

        # step 4: build combined metrics from Ultralytics output
        combined = self._build_combined_metrics(
            ultralytics_metrics, split, split_records
        )

        # step 5: visual samples
        visual_dir = self.output_dir / split / "visual_samples"
        visual_dir.mkdir(parents=True, exist_ok=True)
        all_samples = fa_samples + grz_samples
        self._save_visual_samples(all_samples, visual_dir)

        # step 6: error analysis
        error_analysis = self._build_error_analysis(
            all_samples, fa_records, grz_records
        )

        # step 7: domain shift summary
        domain_shift = self._build_domain_shift_summary(
            fa_metrics, grz_metrics, combined
        )

        report = EvaluationReport(
            weights=str(self.weights),
            split=split,
            confidence_threshold=self.conf,
            iou_threshold=self.iou,
            dataset_version="frozen_v1",
            combined=combined,
            fracatlas=fa_metrics,
            grazpedwri=grz_metrics,
            error_analysis=error_analysis,
            domain_shift_summary=domain_shift,
            known_limitations=self._known_limitations(),
        )

        # save report
        report_path = (
            self.reports_dir / f"evaluation_{split}.json"
        )
        save_json(asdict(report), report_path)
        self._print_evaluation_summary(report)
        return report

    # ── ultralytics val ───────────────────────────────────────────────────────

    def _run_ultralytics_val(self, split: str) -> dict:
        """
        Run Ultralytics model.val() on full split.
        Returns a dict of raw metrics.
        """
        from ultralytics import YOLO

        model = YOLO(str(self.weights))

        # Ultralytics val uses the split defined in dataset.yaml
        # We need to point it at val or test
        val_result = model.val(
            data=str(self.dataset_yaml),
            split=split,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            plots=True,
            save_json=False,
            verbose=False,
        )

        metrics = {}
        try:
            metrics["map50"] = float(val_result.box.map50)
            metrics["map50_95"] = float(val_result.box.map)
            metrics["precision"] = float(val_result.box.mp)
            metrics["recall"] = float(val_result.box.mr)
            metrics["speed"] = dict(val_result.speed)
        except Exception as e:
            logger.warning(f"Could not extract Ultralytics metrics: {e}")

        logger.info(
            f"Ultralytics val [{split}]: "
            f"mAP50={metrics.get('map50', -1):.4f} "
            f"mAP50-95={metrics.get('map50_95', -1):.4f} "
            f"P={metrics.get('precision', -1):.4f} "
            f"R={metrics.get('recall', -1):.4f}"
        )
        return metrics

    # ── per-source inference ──────────────────────────────────────────────────

    def _evaluate_subset(
        self,
        records: list,
        split: str,
        source: str,
    ) -> Tuple[DetectionMetrics, List[dict]]:
        """
        Run inference on a subset of records.
        Returns metrics + per-image sample info for visual/error analysis.
        """
        from ultralytics import YOLO

        model = YOLO(str(self.weights))
        metrics = DetectionMetrics(subset=source, split=split)
        samples = []

        positive_records = [r for r in records if r.fracture_positive == "True"]
        negative_records = [r for r in records if r.fracture_positive == "False"]
        metrics.num_images = len(records)
        metrics.num_positive_images = len(positive_records)
        metrics.num_negative_images = len(negative_records)

        tp_images = fp_images = fn_images = tn_images = 0
        total_gt_boxes = 0
        total_pred_boxes = 0
        inference_times = []

        for record in records:
            img_path = (
                self._processed_dir / split / "images" / record.image_path
            )
            label_path = (
                self._processed_dir / split / "labels"
                / (Path(record.image_path).stem + ".txt")
            )

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            # ground truth boxes
            gt_boxes = self._load_yolo_labels(label_path)
            total_gt_boxes += len(gt_boxes)
            is_positive = record.fracture_positive == "True"

            # inference
            t0 = time.perf_counter()
            results = model.predict(
                str(img_path),
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) * 1000)

            pred_boxes = self._extract_predictions(results)
            total_pred_boxes += len(pred_boxes)
            has_prediction = len(pred_boxes) > 0

            # image-level classification
            if is_positive and has_prediction:
                tp_images += 1
                sample_type = "TP"
            elif not is_positive and has_prediction:
                fp_images += 1
                sample_type = "FP"
            elif is_positive and not has_prediction:
                fn_images += 1
                sample_type = "FN"
            else:
                tn_images += 1
                sample_type = "TN"

            samples.append({
                "sample_id": record.sample_id,
                "dataset": record.dataset,
                "image_path": str(img_path),
                "annotation_status": record.annotation_status,
                "patient_id": record.patient_id,
                "split": split,
                "is_positive": is_positive,
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes,
                "sample_type": sample_type,
                "inference_time_ms": round((t1 - t0) * 1000, 1),
            })

        # compute image-level metrics
        metrics.image_level_tp = tp_images
        metrics.image_level_fp = fp_images
        metrics.image_level_fn = fn_images
        metrics.image_level_tn = tn_images
        metrics.num_gt_boxes = total_gt_boxes
        metrics.num_predicted_boxes = total_pred_boxes

        prec, rec, f1 = self._compute_prf(tp_images, fp_images, fn_images)
        metrics.precision = prec
        metrics.recall = rec
        metrics.f1 = f1

        if inference_times:
            metrics.inference_time_ms_mean = round(
                float(np.mean(inference_times)), 1
            )
            metrics.inference_time_ms_total = round(
                float(np.sum(inference_times)), 1
            )

        logger.info(
            f"[{source}] {split}: "
            f"images={metrics.num_images} "
            f"TP={tp_images} FP={fp_images} "
            f"FN={fn_images} TN={tn_images} "
            f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}"
        )
        return metrics, samples

    # ── visual samples ────────────────────────────────────────────────────────

    def _save_visual_samples(
        self, samples: List[dict], output_dir: Path
    ) -> None:
        """Save representative TP/FP/FN visual examples."""
        by_type: Dict[str, List[dict]] = {
            "TP": [], "FP": [], "FN": [], "TN": []
        }
        for s in samples:
            by_type[s["sample_type"]].append(s)

        per_type = max(1, self.max_visual_samples // 4)

        for sample_type, sample_list in by_type.items():
            if not sample_list:
                continue
            type_dir = output_dir / sample_type
            type_dir.mkdir(parents=True, exist_ok=True)
            # take first N — deterministic (manifest order is fixed)
            selected = sample_list[:per_type]
            for s in selected:
                self._save_annotated_image(s, type_dir)

        logger.info(f"Visual samples saved: {output_dir}")

    def _save_annotated_image(self, sample: dict, output_dir: Path) -> None:
        img = cv2.imread(sample["image_path"])
        if img is None:
            return
        h, w = img.shape[:2]

        # draw GT (green)
        for box in sample["gt_boxes"]:
            _, xc, yc, bw, bh = box
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, "GT", (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        # draw predictions (red)
        for pred in sample["pred_boxes"]:
            x1, y1, x2, y2, conf = (
                int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]),
                pred[4],
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img, f"{conf:.2f}", (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )

        label = (
            f"{sample['sample_type']}_"
            f"{sample['dataset']}_"
            f"{Path(sample['image_path']).stem}"
        )
        out_path = output_dir / f"{label}.jpg"
        cv2.imwrite(str(out_path), img)

    # ── error analysis ────────────────────────────────────────────────────────

    def _build_error_analysis(
        self,
        samples: List[dict],
        fa_records: list,
        grz_records: list,
    ) -> dict:
        fp_samples = [s for s in samples if s["sample_type"] == "FP"]
        fn_samples = [s for s in samples if s["sample_type"] == "FN"]

        # clinically-adjacent false positives
        # (samples marked negative but have adjacent findings)
        adjacent_statuses = {"negative_clean"}
        clinically_adj_fp = [
            s for s in fp_samples
            if s.get("annotation_status") not in adjacent_statuses
        ]

        return {
            "total_fp": len(fp_samples),
            "total_fn": len(fn_samples),
            "fp_by_source": {
                "fracatlas": sum(
                    1 for s in fp_samples if s["dataset"] == "fracatlas"
                ),
                "grazpedwri": sum(
                    1 for s in fp_samples if s["dataset"] == "grazpedwri"
                ),
            },
            "fn_by_source": {
                "fracatlas": sum(
                    1 for s in fn_samples if s["dataset"] == "fracatlas"
                ),
                "grazpedwri": sum(
                    1 for s in fn_samples if s["dataset"] == "grazpedwri"
                ),
            },
            "clinically_adjacent_fp_count": len(clinically_adj_fp),
            "clinically_adjacent_fp_note": (
                "FP samples where annotation_status != negative_clean. "
                "These may include the 169 fracture-adjacent findings in "
                "GRAZPEDWRI-DX (periosteal reaction, pronator sign, etc.). "
                "Do not automatically relabel these samples."
            ),
            "fp_example_ids": [s["sample_id"] for s in fp_samples[:10]],
            "fn_example_ids": [s["sample_id"] for s in fn_samples[:10]],
        }

    def _build_domain_shift_summary(
        self,
        fa: DetectionMetrics,
        grz: DetectionMetrics,
        combined: DetectionMetrics,
    ) -> dict:
        def delta(a: float, b: float) -> str:
            if a < 0 or b < 0:
                return "N/A"
            return f"{(a - b):+.3f}"

        return {
            "note": (
                "FracAtlas covers multi-region anatomy (hand/leg/hip/shoulder). "
                "GRAZPEDWRI-DX covers pediatric wrist imaging. "
                "Differences in per-source metrics indicate domain shift."
            ),
            "recall_delta_fa_minus_grz": delta(fa.recall, grz.recall),
            "precision_delta_fa_minus_grz": delta(fa.precision, grz.precision),
            "f1_delta_fa_minus_grz": delta(fa.f1, grz.f1),
            "inference_time_delta_ms": delta(
                fa.inference_time_ms_mean, grz.inference_time_ms_mean
            ),
            "fracatlas_positive_rate": (
                round(fa.num_positive_images / fa.num_images, 3)
                if fa.num_images > 0 else -1
            ),
            "grazpedwri_positive_rate": (
                round(grz.num_positive_images / grz.num_images, 3)
                if grz.num_images > 0 else -1
            ),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_combined_metrics(
        self,
        ultralytics_metrics: dict,
        split: str,
        records: list,
    ) -> DetectionMetrics:
        m = DetectionMetrics(subset="combined", split=split)
        m.num_images = len(records)
        m.num_positive_images = sum(
            1 for r in records if r.fracture_positive == "True"
        )
        m.num_negative_images = m.num_images - m.num_positive_images
        m.map50 = ultralytics_metrics.get("map50", -1.0)
        m.map50_95 = ultralytics_metrics.get("map50_95", -1.0)
        m.precision = ultralytics_metrics.get("precision", -1.0)
        m.recall = ultralytics_metrics.get("recall", -1.0)
        if m.precision >= 0 and m.recall >= 0:
            m.f1 = self._compute_f1(m.precision, m.recall)
        speed = ultralytics_metrics.get("speed", {})
        m.inference_time_ms_mean = speed.get("inference", -1.0)
        return m

    @staticmethod
    def _load_yolo_labels(label_path: Path) -> List[list]:
        if not label_path.exists():
            return []
        lines = []
        try:
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    lines.append([float(x) for x in parts])
        except Exception:
            pass
        return lines

    @staticmethod
    def _extract_predictions(results) -> List[list]:
        """Extract [x1, y1, x2, y2, conf] from Ultralytics Results."""
        preds = []
        try:
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        preds.append([
                            float(box[0]), float(box[1]),
                            float(box[2]), float(box[3]),
                            float(conf),
                        ])
        except Exception as e:
            logger.warning(f"Could not extract predictions: {e}")
        return preds

    @staticmethod
    def _compute_prf(
        tp: int, fp: int, fn: int
    ) -> Tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return round(precision, 4), round(recall, 4), round(f1, 4)

    @staticmethod
    def _compute_f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def _known_limitations() -> List[str]:
        return [
            "Patient-level leakage verified for GRAZPEDWRI-DX only. "
            "FracAtlas lacks patient identifiers — image-level split used.",
            "169 GRAZPEDWRI-DX samples are fracture-negative but contain "
            "clinically adjacent findings (periosteal reaction, pronator sign, "
            "bone anomaly). These may cause systematic false positives.",
            "GRAZPEDWRI-DX is pediatric wrist imaging only. "
            "FracAtlas covers multi-region anatomy. "
            "Model performance may differ significantly by anatomical region.",
            "Model is NOT clinically validated. "
            "Results are research/development metrics only.",
        ]

    def _print_evaluation_summary(self, report: EvaluationReport) -> None:
        c = report.combined
        fa = report.fracatlas
        grz = report.grazpedwri
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"EVALUATION SUMMARY — split={report.split}")
        logger.info("=" * 60)
        logger.info(
            f"  Combined  : mAP50={c.map50:.4f} "
            f"P={c.precision:.4f} R={c.recall:.4f} F1={c.f1:.4f}"
        )
        logger.info(
            f"  FracAtlas : "
            f"P={fa.precision:.4f} R={fa.recall:.4f} F1={fa.f1:.4f} "
            f"TP={fa.image_level_tp} FP={fa.image_level_fp} "
            f"FN={fa.image_level_fn}"
        )
        logger.info(
            f"  GRAZPEDWRI: "
            f"P={grz.precision:.4f} R={grz.recall:.4f} F1={grz.f1:.4f} "
            f"TP={grz.image_level_tp} FP={grz.image_level_fp} "
            f"FN={grz.image_level_fn}"
        )
        logger.info(
            f"  Inference : {c.inference_time_ms_mean:.1f} ms/image (mean)"
        )
        logger.info("=" * 60)