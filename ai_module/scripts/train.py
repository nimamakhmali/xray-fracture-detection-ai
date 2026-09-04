#!/usr/bin/env python3
"""
scripts/train.py — the only entry point for training.

  Smoke test  : python scripts/train.py --smoke-test --device cpu --fraction 0.02
  CPU baseline: python scripts/train.py --config configs/model_config_cpu.yaml --device cpu --name baseline_yolov8n_cpu
  GPU baseline: python scripts/train.py --config configs/model_config.yaml --device 0
  Resume      : python scripts/train.py --resume --name <experiment_id>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import FractureDetectionTrainer
from src.utils.logger import get_logger

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 fracture detection baseline.")
    p.add_argument("--dataset-yaml", default="configs/dataset.yaml")
    p.add_argument("--config", "--model-config", dest="model_config", default="configs/model_config.yaml")
    p.add_argument("--validation-report", default="reports/validation_report.json")
    p.add_argument("--weights", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--output-dir", default="runs/detect")
    p.add_argument("--reports-dir", default="reports/training")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu | auto | 0 | 0,1")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fraction", type=float, default=None, help="train-set fraction (SMOKE TEST ONLY)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="no gates; default 1 epoch, batch 4")
    p.add_argument("--promote", action="store_true", help="copy best.pt to models/production/ (official runs only)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    root = Path(__file__).resolve().parent.parent
    epochs, batch = a.epochs, a.batch_size
    if a.smoke_test:
        logger.warning("SMOKE TEST MODE — not a baseline experiment.")
        epochs, batch = epochs or 1, batch or 4
    elif a.fraction is not None and a.fraction != 1.0:
        logger.error("--fraction is only permitted with --smoke-test."); sys.exit(2)

    try:
        trainer = FractureDetectionTrainer(
            dataset_yaml=root / a.dataset_yaml, model_config_yaml=root / a.model_config,
            validation_report=None if a.smoke_test else root / a.validation_report,
            output_dir=root / a.output_dir, reports_dir=root / a.reports_dir,
            epochs=epochs, batch_size=batch, image_size=a.image_size, device=a.device,
            workers=a.workers, seed=a.seed, resume=a.resume,
            experiment_name=a.name or ("smoke_test_cpu" if a.smoke_test else None),
            pretrained_weights=a.weights, fraction=a.fraction, promote_to_production=a.promote,
        )
        result = trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    logger.info(f"Done. best.pt={result.best_checkpoint} val mAP50={result.best_map50:.4f}")


if __name__ == "__main__":
    main()