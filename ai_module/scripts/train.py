#!/usr/bin/env python3
"""
scripts/train.py

Training entry point for the fracture detection baseline.

This script is the ONLY way to start a training run.
It enforces:
  1. Dataset validation gate (training blocked if dataset not READY)
  2. Reproducible configuration via model_config.yaml
  3. CLI overrides for experimentation
  4. Clean experiment directory
  5. Saved experiment metadata

Usage:
    # Smoke test (2 epochs, fast verification)
    python scripts/train.py --smoke-test

    # Baseline training (50 epochs)
    python scripts/train.py

    # With explicit device
    python scripts/train.py --device cpu
    python scripts/train.py --device 0

    # Resume interrupted training
    python scripts/train.py --resume --name <experiment_id>

    # Override specific params
    python scripts/train.py --epochs 30 --batch-size 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import FractureDetectionTrainer
from src.utils.logger import get_logger

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8 fracture detection baseline."
    )
    # paths
    p.add_argument(
        "--dataset-yaml", default="configs/dataset.yaml",
        help="Path to dataset YAML (default: configs/dataset.yaml)",
    )
    p.add_argument(
        "--model-config", default="configs/model_config.yaml",
        help="Path to model config YAML",
    )
    p.add_argument(
        "--validation-report", default="reports/validation_report.json",
        help="Path to validation report (dataset gate)",
    )
    p.add_argument(
        "--weights", default=None,
        help="Pretrained weights (overrides model_config.yaml)",
    )
    # experiment
    p.add_argument("--name", default=None, help="Experiment name")
    p.add_argument("--output-dir", default="runs/detect", help="Output root")
    p.add_argument("--reports-dir", default="reports/training")
    # training params (all override model_config.yaml)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu | 0 | 0,1 | auto")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    # flags
    p.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Run 2 epochs with batch=4 to verify pipeline (no gate check)",
    )
    p.add_argument(
        "--skip-validation-gate", action="store_true",
        help="Skip dataset validation report check (use only for debugging)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # resolve paths from repo root (ai_module/)
    root = Path(__file__).resolve().parent.parent

    dataset_yaml = root / args.dataset_yaml
    model_config = root / args.model_config
    validation_report = root / args.validation_report
    output_dir = root / args.output_dir
    reports_dir = root / args.reports_dir

    # smoke test overrides
    epochs = args.epochs
    batch_size = args.batch_size
    skip_gate = args.skip_validation_gate

    if args.smoke_test:
        logger.warning(
            "SMOKE TEST MODE — 2 epochs, batch=4. "
            "This is NOT the baseline experiment."
        )
        epochs = epochs or 2
        batch_size = batch_size or 4
        skip_gate = True  # smoke test doesn't need full gate

    # build trainer
    trainer = FractureDetectionTrainer(
        dataset_yaml=dataset_yaml,
        model_config_yaml=model_config,
        validation_report=None if skip_gate else validation_report,
        output_dir=output_dir,
        reports_dir=reports_dir,
        epochs=epochs,
        batch_size=batch_size,
        image_size=args.image_size,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        resume=args.resume,
        experiment_name=args.name,
        pretrained_weights=args.weights,
    )

    try:
        result = trainer.train()
        logger.info(f"Training completed successfully.")
        logger.info(f"Best mAP@50 : {result.best_map50:.4f}")
        logger.info(f"Checkpoint  : {result.best_checkpoint}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()