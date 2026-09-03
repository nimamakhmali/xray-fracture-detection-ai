"""
src/training/trainer.py
"""
from __future__ import annotations

import csv as csv_module
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.file_utils import save_json

logger = get_logger(__name__)

VALIDATION_REPORT_READY_STATUS = "READY_FOR_TRAINING"
DATASET_VERSION = "frozen_v1"


@dataclass
class ExperimentConfig:
    experiment_id: str
    dataset_version: str
    model_architecture: str
    pretrained_weights: str
    num_classes: int
    class_names: list
    dataset_yaml: str
    image_size: int
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    momentum: float
    device: str
    seed: int
    early_stopping_patience: int
    workers: int
    project_dir: str
    run_name: str
    resume: bool
    use_custom_augmentation: bool
    augmentation_probability: float
    best_epoch: int = -1
    best_map50: float = -1.0
    training_duration_seconds: float = -1.0
    checkpoint_best: str = ""
    checkpoint_last: str = ""
    status: str = "initialized"
    error: str = ""
    framework_versions: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingResult:
    success: bool
    experiment_id: str
    best_checkpoint: Path
    last_checkpoint: Path
    best_map50: float
    best_epoch: int
    training_duration_seconds: float
    results_csv: Path
    experiment_dir: Path
    config: ExperimentConfig


class FractureDetectionTrainer:

    def __init__(
        self,
        dataset_yaml: Path,
        model_config_yaml: Path,
        validation_report: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        image_size: Optional[int] = None,
        device: Optional[str] = None,
        workers: Optional[int] = None,
        seed: Optional[int] = None,
        resume: bool = False,
        experiment_name: Optional[str] = None,
        pretrained_weights: Optional[str] = None,
    ):
        self.dataset_yaml = Path(dataset_yaml).resolve()
        self.model_config_yaml = Path(model_config_yaml).resolve()
        self.validation_report = (
            Path(validation_report).resolve() if validation_report else None
        )
        self.resume = resume

        self._root = Path(__file__).resolve().parent.parent.parent
        self._reports_dir = (
            Path(reports_dir).resolve() if reports_dir
            else self._root / "reports" / "training"
        )
        self._output_dir = (
            Path(output_dir).resolve() if output_dir
            else self._root / "runs" / "detect"
        )

        self._model_cfg = self._load_yaml(self.model_config_yaml)
        self._dataset_cfg = self._load_yaml(self.dataset_yaml)

        train_cfg = self._model_cfg.get("training", {})
        model_cfg = self._model_cfg.get("model", {})

        _seed = seed if seed is not None else train_cfg.get("seed", 42)
        _epochs = epochs if epochs is not None else train_cfg.get("epochs", 50)
        _batch = batch_size if batch_size is not None else train_cfg.get("batch_size", 16)
        _imgsz = image_size if image_size is not None else train_cfg.get("image_size", 640)
        _device = device if device is not None else train_cfg.get("device", "auto")
        _workers = workers if workers is not None else train_cfg.get("workers", 8)
        _weights = (
            pretrained_weights
            or model_cfg.get("pretrained_weights", "yolov8n.pt")
        )

        import datetime
        _exp_name = (
            experiment_name
            or f"baseline_{model_cfg.get('architecture', 'yolov8n')}"
        )
        exp_id = f"{_exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.cfg = ExperimentConfig(
            experiment_id=exp_id,
            dataset_version=DATASET_VERSION,
            model_architecture=model_cfg.get("architecture", "yolov8n"),
            pretrained_weights=_weights,
            num_classes=model_cfg.get("num_classes", 1),
            class_names=model_cfg.get("class_names", ["fracture"]),
            dataset_yaml=str(self.dataset_yaml),
            image_size=_imgsz,
            epochs=_epochs,
            batch_size=_batch,
            optimizer=train_cfg.get("optimizer", "SGD"),
            learning_rate=train_cfg.get("learning_rate", 0.01),
            weight_decay=train_cfg.get("weight_decay", 0.0005),
            momentum=train_cfg.get("momentum", 0.937),
            device=_device,
            seed=_seed,
            early_stopping_patience=train_cfg.get("early_stopping_patience", 15),
            workers=_workers,
            project_dir=str(self._output_dir),
            run_name=exp_id,
            resume=resume,
            use_custom_augmentation=train_cfg.get("use_custom_augmentation", False),
            augmentation_probability=train_cfg.get("augmentation_probability", 0.5),
            framework_versions=self._collect_versions(),
        )


   # --------------------------------------------------------------------------------
    
    def _setup_mlflow(self) -> None:
        """
        Configure MLflow filesystem backend.

        The newer MLflow versions raise by default when using the filesystem
        backend. Setting MLFLOW_ALLOW_FILE_STORE=true restores previous
        behavior. This is a local development environment — no cloud
        MLflow server is available.
        """
        import os
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        logger.info("MLflow: filesystem backend enabled (MLFLOW_ALLOW_FILE_STORE=true)")

    def _verify_dataset_freeze(self) -> None:
        """
        Verify that the frozen dataset has not been modified since freeze.
        Only runs for non-smoke-test experiments.
        """
        from src.utils.dataset_freeze import verify_freeze

        freeze_record = self._root / "reports" / "frozen_dataset_v1.json"
        if not freeze_record.exists():
            logger.warning(
                "Freeze record not found — dataset freeze not verified. "
                "Run scripts/freeze_dataset.py --repair-jpegs first."
            )
            return

        result = verify_freeze(
            freeze_record_path=freeze_record,
            processed_dir=self._root / "data" / "processed",
            dataset_yaml=self.dataset_yaml,
        )
        if not result["ok"]:
            raise RuntimeError(
                f"Dataset freeze verification FAILED: "
                f"{result.get('mismatches', result.get('reason'))}. "
                f"Dataset may have been modified. Re-run prepare_dataset.py."
            )
        logger.info(
            f"Dataset freeze verified: version={result.get('version')}"
        )
        


    # ── public ───────────────────────────────────────────────────────────────

    def train(self) -> TrainingResult:
        """Run training. Returns TrainingResult with all artifact paths."""
        # setup MLflow BEFORE Ultralytics initializes it
        self._setup_mlflow()

        self._pre_flight_checks()

        # verify frozen dataset (skipped for smoke tests via no validation_report)
        if self.validation_report:
            self._verify_dataset_freeze()

        self._setup_seed()
        self._log_experiment_header()

        self._reports_dir.mkdir(parents=True, exist_ok=True)

        meta_path = (
            self._reports_dir / f"{self.cfg.experiment_id}_config.json"
        )
        save_json(asdict(self.cfg), meta_path)
        logger.info(f"Experiment config saved: {meta_path}")

        start_time = time.time()
        self.cfg.status = "running"

        try:
            result = self._run_ultralytics_training()
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
            self.cfg.status = "interrupted"
            self._save_final_metadata()
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.cfg.status = "failed"
            self.cfg.error = str(e)
            self._save_final_metadata()
            raise

        duration = time.time() - start_time
        self.cfg.training_duration_seconds = round(duration, 1)
        self.cfg.status = "completed"

        self._extract_best_metrics(result)
        self._save_final_metadata()

        # FIXED: only promote to production if NOT a smoke test
        # (smoke test has no validation_report)
        if self.validation_report is not None:
            self._copy_best_to_production()
        else:
            logger.warning(
                "Smoke test checkpoint NOT copied to production/. "
                "Only officially validated experiments are promoted."
            )

        training_result = self._build_result(result)
        self._log_training_summary(training_result)
        return training_result


    # ── private ───────────────────────────────────────────────────────────────
    def _pre_flight_checks(self) -> None:
        """Block training if any prerequisite fails."""
        errors = []

        if not self.dataset_yaml.exists():
            errors.append(f"dataset.yaml not found: {self.dataset_yaml}")

        if self.validation_report:
            if not self.validation_report.exists():
                errors.append(
                    f"Validation report not found: {self.validation_report}."
                )
            else:
                try:
                    with open(self.validation_report) as f:
                        vr = json.load(f)
                    status = vr.get("status", "UNKNOWN")
                    if status != VALIDATION_REPORT_READY_STATUS:
                        errors.append(
                            f"Dataset validation status is '{status}' — "
                            f"expected '{VALIDATION_REPORT_READY_STATUS}'."
                        )
                    else:
                        logger.info(
                            f"Dataset validation gate: PASS (status={status})"
                        )
                except Exception as e:
                    errors.append(f"Cannot read validation report: {e}")
        else:
            logger.warning(
                "No validation report path — skipping gate. "
                "Acceptable only for smoke tests."
            )

        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            errors.append("ultralytics not installed.")

        stats = self._dataset_cfg.get("stats", {})
        if stats:
            total = stats.get("total_images", 0)
            if total < 100:
                errors.append(
                    f"dataset.yaml reports only {total} images — suspicious."
                )

        if errors:
            for err in errors:
                logger.error(f"PRE-FLIGHT FAIL: {err}")
            raise RuntimeError(
                f"Training blocked — {len(errors)} pre-flight check(s) failed."
            )

        logger.info("All pre-flight checks passed.")


    def _setup_seed(self) -> None:
        import random
        import numpy as np
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        logger.info(f"Random seed set: {self.cfg.seed}")

    def _run_ultralytics_training(self):
        from ultralytics import YOLO

        weights = self.cfg.pretrained_weights
        if not Path(weights).exists():
            logger.info(
                f"Weights '{weights}' not found locally — "
                f"Ultralytics will download from hub."
            )

        model = YOLO(weights)
        logger.info(f"Model loaded: {weights}")

        device = self._resolve_device(self.cfg.device)
        logger.info(f"Training device: {device}")

        resolved_yaml = self._resolve_dataset_yaml_for_ultralytics()

        train_kwargs = dict(
            data=str(resolved_yaml),
            epochs=self.cfg.epochs,
            batch=self.cfg.batch_size,
            imgsz=self.cfg.image_size,
            optimizer=self.cfg.optimizer,
            lr0=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum,
            device=device,
            workers=self.cfg.workers,
            seed=self.cfg.seed,
            patience=self.cfg.early_stopping_patience,
            project=str(self._output_dir),
            name=self.cfg.experiment_id,
            exist_ok=self.resume,
            resume=self.resume,
            plots=True,
            save=True,
            save_period=self._model_cfg.get("training", {}).get(
                "save_period", 10
            ),
            deterministic=False,
            verbose=True,
        )

        logger.info(f"Starting Ultralytics training...")
        result = model.train(**train_kwargs)
        return result

    def _resolve_dataset_yaml_for_ultralytics(self) -> Path:
        """
        Write a temp yaml with absolute 'path' so Ultralytics resolves
        train/val/test correctly regardless of cwd.
        Original dataset.yaml is never modified.
        """
        cfg = dict(self._dataset_cfg)

        raw_path = cfg.get("path", "data/processed")
        if not Path(raw_path).is_absolute():
            abs_path = (
                self.dataset_yaml.parent.parent / raw_path
            ).resolve()
        else:
            abs_path = Path(raw_path)

        if not abs_path.exists():
            raise RuntimeError(
                f"Dataset path does not exist: {abs_path}. "
                f"Run prepare_dataset.py first."
            )

        cfg["path"] = str(abs_path)

        tmp_yaml = self.dataset_yaml.parent / "_ultralytics_resolved.yaml"
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

        logger.info(f"Resolved dataset yaml: {tmp_yaml}")
        logger.info(f"  Absolute path: {abs_path}")
        return tmp_yaml

    def _extract_best_metrics(self, result) -> None:
        try:
            if hasattr(result, "results_dict"):
                rd = result.results_dict
                self.cfg.best_map50 = float(
                    rd.get("metrics/mAP50(B)", -1.0)
                )
        except Exception:
            pass

        exp_dir = self._output_dir / self.cfg.experiment_id
        csv_path = exp_dir / "results.csv"
        if csv_path.exists():
            try:
                with open(csv_path) as f:
                    rows = list(csv_module.DictReader(f))
                if rows:
                    rows = [
                        {k.strip(): v.strip() for k, v in r.items()}
                        for r in rows
                    ]
                    map50_key = next(
                        (
                            k for k in rows[0]
                            if "mAP50" in k and "95" not in k
                        ),
                        None,
                    )
                    if map50_key:
                        best_row = max(
                            rows,
                            key=lambda r: float(r.get(map50_key, 0) or 0),
                        )
                        self.cfg.best_map50 = float(
                            best_row.get(map50_key, -1.0)
                        )
                        self.cfg.best_epoch = int(
                            float(best_row.get("epoch", -1))
                        )
            except Exception as e:
                logger.warning(f"Could not parse results.csv: {e}")

        exp_dir = self._output_dir / self.cfg.experiment_id
        best_pt = exp_dir / "weights" / "best.pt"
        last_pt = exp_dir / "weights" / "last.pt"
        self.cfg.checkpoint_best = (
            str(best_pt) if best_pt.exists() else ""
        )
        self.cfg.checkpoint_last = (
            str(last_pt) if last_pt.exists() else ""
        )

    def _copy_best_to_production(self) -> None:
        best_pt = Path(self.cfg.checkpoint_best)
        if not best_pt.exists():
            logger.warning(
                "best.pt not found — skipping copy to production/"
            )
            return
        prod_dir = self._root / "models" / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)
        dest = prod_dir / "best_model.pt"
        shutil.copy2(best_pt, dest)
        logger.info(f"Best checkpoint copied to: {dest}")

    def _save_final_metadata(self) -> None:
        meta_path = (
            self._reports_dir
            / f"{self.cfg.experiment_id}_final.json"
        )
        save_json(asdict(self.cfg), meta_path)
        logger.info(f"Final metadata saved: {meta_path}")

    def _build_result(self, ultralytics_result) -> TrainingResult:
        exp_dir = self._output_dir / self.cfg.experiment_id
        return TrainingResult(
            success=True,
            experiment_id=self.cfg.experiment_id,
            best_checkpoint=Path(self.cfg.checkpoint_best),
            last_checkpoint=Path(self.cfg.checkpoint_last),
            best_map50=self.cfg.best_map50,
            best_epoch=self.cfg.best_epoch,
            training_duration_seconds=self.cfg.training_duration_seconds,
            results_csv=exp_dir / "results.csv",
            experiment_dir=exp_dir,
            config=self.cfg,
        )

    def _log_experiment_header(self) -> None:
        logger.info("=" * 60)
        logger.info("FRACTURE DETECTION — BASELINE TRAINING")
        logger.info("=" * 60)
        logger.info(f"  Experiment ID : {self.cfg.experiment_id}")
        logger.info(f"  Dataset       : {self.cfg.dataset_version}")
        logger.info(f"  Architecture  : {self.cfg.model_architecture}")
        logger.info(f"  Weights       : {self.cfg.pretrained_weights}")
        logger.info(
            f"  Classes       : {self.cfg.num_classes} "
            f"{self.cfg.class_names}"
        )
        logger.info(f"  Image size    : {self.cfg.image_size}")
        logger.info(f"  Epochs        : {self.cfg.epochs}")
        logger.info(f"  Batch size    : {self.cfg.batch_size}")
        logger.info(f"  Optimizer     : {self.cfg.optimizer}")
        logger.info(f"  LR            : {self.cfg.learning_rate}")
        logger.info(f"  Device        : {self.cfg.device}")
        logger.info(f"  Seed          : {self.cfg.seed}")
        logger.info("=" * 60)

    def _log_training_summary(self, result: TrainingResult) -> None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(
            f"  Duration    : {result.training_duration_seconds:.0f}s"
        )
        logger.info(f"  Best mAP@50 : {result.best_map50:.4f}")
        logger.info(f"  Best epoch  : {result.best_epoch}")
        logger.info(f"  Checkpoint  : {result.best_checkpoint}")
        logger.info("=" * 60)

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return device_str

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _collect_versions() -> Dict[str, str]:
        versions: Dict[str, str] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": (
                torch.version.cuda
                if torch.cuda.is_available()
                else "N/A"
            ),
        }
        for pkg in ("ultralytics", "cv2", "numpy"):
            try:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[pkg] = "not installed"
        return versions