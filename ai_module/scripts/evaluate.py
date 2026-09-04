#!/usr/bin/env python3
"""
scripts/evaluate.py

  val evaluation      : python scripts/evaluate.py --weights <best.pt> --split val
  threshold analysis  : python scripts/evaluate.py --weights <best.pt> --analyze-thresholds
  val at selected τ   : python scripts/evaluate.py --weights <best.pt> --split val --use-selected-threshold
  FINAL test (once)   : python scripts/evaluate.py --weights <best.pt> --split test --use-selected-threshold
  dev smoke (val only): python scripts/evaluate.py --weights <best.pt> --split val --limit 100
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.evaluator import FractureDetectionEvaluator, SELECTION_FILE
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    p.add_argument("--manifest", default="data/processed/manifest.csv")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--use-selected-threshold", action="store_true")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU for the operating-point predict pass")
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--output-dir", default="runs/evaluate")
    p.add_argument("--reports-dir", default="reports/evaluation")
    p.add_argument("--max-visuals", type=int, default=25)
    p.add_argument("--limit", type=int, default=None, help="DEV ONLY: subset of val")
    p.add_argument("--analyze-thresholds", action="store_true")
    p.add_argument("--thresholds", default=None, help="comma list, e.g. 0.1,0.2,0.3")
    p.add_argument("--selection-rule", default="max_image_f1", choices=["max_image_f1", "max_object_f1"])
    p.add_argument("--force-test-rerun", action="store_true")
    p.add_argument("--reason", default="")
    p.add_argument("--no-post-verify", action="store_true", help="skip full freeze re-hash after run")
    return p.parse_args()


def main():
    a = parse_args()
    root = Path(__file__).resolve().parent.parent
    weights = Path(a.weights) if Path(a.weights).is_absolute() else root / a.weights
    conf = a.conf
    if a.use_selected_threshold:
        sel = root / a.reports_dir / SELECTION_FILE
        if not sel.exists():
            logger.error(f"{sel} not found — run --analyze-thresholds first."); sys.exit(1)
        conf = json.loads(sel.read_text())["selected_threshold"]
        logger.info(f"Using selected threshold from val analysis: {conf}")
    try:
        ev = FractureDetectionEvaluator(
            weights=weights, dataset_yaml=root / a.dataset_yaml, manifest_path=root / a.manifest,
            output_dir=root / a.output_dir, reports_dir=root / a.reports_dir, confidence_threshold=conf,
            iou_threshold=a.iou, match_iou=a.match_iou, device=a.device, image_size=a.image_size,
            max_visual_samples=a.max_visuals, limit=a.limit, verify_after_full=not a.no_post_verify)
        if a.analyze_thresholds:
            th = [float(x) for x in a.thresholds.split(",")] if a.thresholds else None
            ev.analyze_thresholds("val", th, a.selection_rule)
        else:
            ev.evaluate(a.split, force_test=a.force_test_rerun, force_reason=a.reason)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()