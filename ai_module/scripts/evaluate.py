#!/usr/bin/env python3
"""
scripts/evaluate.py

Evaluation entry point.

Usage:
    # Validate on validation set (model selection)
    python scripts/evaluate.py --weights runs/detect/<exp>/weights/best.pt

    # Final test evaluation (run ONLY once after model is finalized)
    python scripts/evaluate.py \
        --weights models/production/best_model.pt \
        --split test

    # Evaluate with custom threshold
    python scripts/evaluate.py \
        --weights models/production/best_model.pt \
        --conf 0.3 --split val
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.evaluator import FractureDetectionEvaluator
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate fracture detection model."
    )
    p.add_argument(
        "--weights", required=True,
        help="Path to model weights (.pt)",
    )
    p.add_argument(
        "--dataset-yaml", default="configs/dataset.yaml",
    )
    p.add_argument(
        "--manifest", default="data/processed/manifest.csv",
    )
    p.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Which split to evaluate. Use 'test' only for final evaluation.",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="runs/evaluate")
    p.add_argument("--reports-dir", default="reports/evaluation")
    p.add_argument("--max-visuals", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = root / weights
    if not weights.exists():
        logger.error(f"Weights not found: {weights}")
        sys.exit(1)

    evaluator = FractureDetectionEvaluator(
        weights=weights,
        dataset_yaml=root / args.dataset_yaml,
        manifest_path=root / args.manifest,
        output_dir=root / args.output_dir,
        reports_dir=root / args.reports_dir,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        max_visual_samples=args.max_visuals,
    )

    try:
        report = evaluator.evaluate(split=args.split)
        c = report.combined
        logger.info(
            f"mAP@50={c.map50:.4f} | "
            f"P={c.precision:.4f} | R={c.recall:.4f} | F1={c.f1:.4f}"
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()