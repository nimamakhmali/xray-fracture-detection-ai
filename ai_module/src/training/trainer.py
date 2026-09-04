"""
src/training/trainer.py

Phase 2 training orchestrator.

Engine stack:  PyTorch  ->  Ultralytics YOLO (high-level API)  ->  YOLOv8n
This module does not re-implement the detector; it wraps Ultralytics with the
project's gates, provenance and reproducibility metadata.

Run kinds
  smoke_test : validation_report=None. No gates, may use --fraction, never promoted.
  official   : gates enforced: validation READY, frozen dataset quick-verify,
               requested device actually available, fraction == 1.0.
"""
from __future__ import annotations

import csv as csv_module
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.file_utils import save_json

logger = get_logger(__name__)

VALIDATION_REPORT_READY_STATUS = "READY_FOR_TRAINING"
FREEZE_RECORD_REL = Path("reports/dataset/frozen_v1.json")

STATUS_INITIALIZED, STATUS_RUNNING = "INITIALIZED", "RUNNING"
STATUS_SUCCESS, STATUS_FAILED, STATUS_INTERRUPTED = "SUCCESS", "FAILED", "INTERRUPTED"

# Ultralytics DetMetrics.fitness = 0.1*mAP50 + 0.9*mAP50-95 ; best.pt is chosen by this.
FITNESS_W_MAP50, FITNESS_W_MAP5095 = 0.1, 0.9


# ── small helpers ──────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "UNAVAILABLE"


def _pkg_version(name: str) -> str:
    try:
        return getattr(__import__(name), "__version__", "unknown")
    except ImportError:
        return "not installed"


def resolve_device(requested: str) -> str:
    """
    Deterministic, hardware-aware device resolution. Never silently falls back.
      auto        -> "0" if CUDA available else "cpu"
      cpu         -> "cpu"
      0 / 0,1 / cuda:0 -> validated against torch.cuda
    """
    req = str(requested).strip().lower()
    cuda_ok = torch.cuda.is_available()
    if req in ("", "auto"):
        return "0" if cuda_ok else "cpu"
    if req == "cpu":
        return "cpu"
    if req == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("device=mps requested but MPS is not available.")
    try:
        idx = [int(x) for x in req.replace("cuda:", "").split(",") if x.strip()]
    except ValueError:
        raise RuntimeError(f"Unrecognised device spec: '{requested}' (use cpu | auto | 0 | 0,1)")
    if not cuda_ok:
        raise RuntimeError(
            f"CUDA device(s) {idx} requested but torch.cuda.is_available() is False "
            f"(torch={torch.__version__}). Use --device cpu (or auto)."
        )
    n = torch.cuda.device_count()
    bad = [i for i in idx if i >= n]
    if bad:
        raise RuntimeError(f"CUDA device index {bad} out of range; {n} device(s) visible.")
    return ",".join(str(i) for i in idx)


# ── dataclasses ────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    experiment_id: str
    run_kind: str                      # smoke_test | official
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
    device_requested: str
    device: str                        # resolved
    seed: int
    early_stopping_patience: int
    workers: int
    fraction: float
    deterministic: bool
    project_dir: str
    run_name: str
    resume: bool
    use_custom_augmentation: bool
    augmentation_probability: float
    provenance: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    mlflow: Dict[str, str] = field(default_factory=dict)
    epochs_completed: int = 0
    best_epoch: int = -1
    best_fitness: float = -1.0
    best_map50: float = -1.0
    best_map50_95: float = -1.0
    best_precision: float = -1.0
    best_recall: float = -1.0
    final_epoch_metrics: Dict[str, float] = field(default_factory=dict)
    training_duration_seconds: float = -1.0
    checkpoint_best: str = ""
    checkpoint_last: str = ""
    status: str = STATUS_INITIALIZED
    error: str = ""


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


# ── trainer ────────────────────────────────────────────────────────────────

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
        fraction: Optional[float] = None,
        promote_to_production: bool = False,
    ):
        self.dataset_yaml = Path(dataset_yaml).resolve()
        self.model_config_yaml = Path(model_config_yaml).resolve()
        self.validation_report = Path(validation_report).resolve() if validation_report else None
        self.resume = resume
        self.promote = promote_to_production
        self._root = Path(__file__).resolve().parent.parent.parent
        self._reports_dir = Path(reports_dir).resolve() if reports_dir else self._root / "reports" / "training"
        self._output_dir = Path(output_dir).resolve() if output_dir else self._root / "runs" / "detect"
        self._freeze_record = self._root / FREEZE_RECORD_REL

        self._model_cfg = self._load_yaml(self.model_config_yaml)
        self._dataset_cfg = self._load_yaml(self.dataset_yaml)
        train_cfg = self._model_cfg.get("training", {})
        model_cfg = self._model_cfg.get("model", {})

        run_kind = "official" if self.validation_report else "smoke_test"
        requested_device = device if device is not None else train_cfg.get("device", "auto")
        resolved_device = resolve_device(requested_device)      # raises early if impossible

        if resume:
            if not experiment_name:
                raise ValueError("--resume requires --name <existing experiment_id>")
            exp_id = experiment_name
        else:
            base = experiment_name or f"baseline_{model_cfg.get('architecture', 'yolov8n')}"
            exp_id = f"{base}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        _fraction = float(fraction) if fraction is not None else 1.0

        self.cfg = ExperimentConfig(
            experiment_id=exp_id,
            run_kind=run_kind,
            dataset_version="UNAVAILABLE",   # filled from freeze record in pre-flight
            model_architecture=model_cfg.get("architecture", "yolov8n"),
            pretrained_weights=pretrained_weights or model_cfg.get("pretrained_weights", "yolov8n.pt"),
            num_classes=model_cfg.get("num_classes", 1),
            class_names=model_cfg.get("class_names", ["fracture"]),
            dataset_yaml=str(self.dataset_yaml),
            image_size=image_size if image_size is not None else train_cfg.get("image_size", 640),
            epochs=epochs if epochs is not None else train_cfg.get("epochs", 50),
            batch_size=batch_size if batch_size is not None else train_cfg.get("batch_size", 16),
            optimizer=train_cfg.get("optimizer", "SGD"),
            learning_rate=train_cfg.get("learning_rate", 0.01),
            weight_decay=train_cfg.get("weight_decay", 0.0005),
            momentum=train_cfg.get("momentum", 0.937),
            device_requested=str(requested_device),
            device=resolved_device,
            seed=seed if seed is not None else train_cfg.get("seed", 42),
            early_stopping_patience=train_cfg.get("early_stopping_patience", 15),
            workers=workers if workers is not None else train_cfg.get("workers", 8),
            fraction=_fraction,
            deterministic=bool(train_cfg.get("deterministic", True)),
            project_dir=str(self._output_dir),
            run_name=exp_id,
            resume=resume,
            use_custom_augmentation=train_cfg.get("use_custom_augmentation", False),
            augmentation_probability=train_cfg.get("augmentation_probability", 0.5),
            environment=self._collect_environment(resolved_device),
            provenance=self._collect_provenance(),
        )
        self._mlflow_uri = str((self._root / self._model_cfg.get("paths", {}).get("mlflow_tracking", "mlruns/")).resolve())
        self._mlflow_experiment = self._model_cfg.get("paths", {}).get("mlflow_experiment", "fracture-detection-phase2")

    # ── public ────────────────────────────────────────────────────────────

    def train(self) -> TrainingResult:
        self._setup_mlflow()
        self._pre_flight_checks()
        self._setup_seed()
        self._log_experiment_header()

        self._reports_dir.mkdir(parents=True, exist_ok=True)
        save_json(asdict(self.cfg), self._reports_dir / f"{self.cfg.experiment_id}_config.json")

        start = time.time()
        self.cfg.status = STATUS_RUNNING
        try:
            result = self._run_ultralytics_training()
        except KeyboardInterrupt:
            self.cfg.status, self.cfg.training_duration_seconds = STATUS_INTERRUPTED, round(time.time() - start, 1)
            self._extract_best_metrics(); self._save_final_metadata()
            raise
        except Exception as e:
            self.cfg.status, self.cfg.error = STATUS_FAILED, f"{type(e).__name__}: {e}"
            self.cfg.training_duration_seconds = round(time.time() - start, 1)
            self._extract_best_metrics(); self._save_final_metadata()
            raise

        self.cfg.training_duration_seconds = round(time.time() - start, 1)
        self._extract_best_metrics()
        if not self.cfg.checkpoint_best:
            self.cfg.status, self.cfg.error = STATUS_FAILED, "Training returned but best.pt was not produced."
            self._save_final_metadata()
            raise RuntimeError(self.cfg.error)

        self.cfg.status = STATUS_SUCCESS
        self._log_provenance_to_mlflow()
        self._save_final_metadata()

        if self.promote and self.cfg.run_kind == "official":
            self._promote_to_production()
        elif self.promote:
            logger.warning("Smoke-test checkpoints are never promoted to models/production/.")

        tr = self._build_result()
        self._log_training_summary(tr)
        return tr

    # ── gates ─────────────────────────────────────────────────────────────

    def _pre_flight_checks(self) -> None:
        errors: List[str] = []
        if not self.dataset_yaml.exists():
            errors.append(f"dataset.yaml not found: {self.dataset_yaml}")
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            errors.append("ultralytics not installed.")

        if self.cfg.run_kind == "official":
            if self.cfg.fraction != 1.0:
                errors.append("fraction != 1.0 is only allowed for smoke tests.")
            vr_path = self.validation_report
            if not vr_path.exists():
                errors.append(f"Validation report not found: {vr_path}")
            else:
                try:
                    status = json.loads(vr_path.read_text()).get("status", "UNKNOWN")
                    if status != VALIDATION_REPORT_READY_STATUS:
                        errors.append(f"Dataset validation status '{status}' != '{VALIDATION_REPORT_READY_STATUS}'.")
                    else:
                        logger.info("Gate: dataset validation READY")
                except Exception as e:
                    errors.append(f"Cannot read validation report: {e}")
            errors.extend(self._verify_dataset_freeze())
        else:
            logger.warning("SMOKE TEST — validation gate and freeze verification skipped.")
            if self._freeze_record.exists():
                self.cfg.dataset_version = json.loads(self._freeze_record.read_text()).get("version", "UNAVAILABLE")

        if self.cfg.device == "cpu" and self.cfg.workers != 0:
            logger.info("Note: Ultralytics forces workers=0 on CPU; the configured value is recorded but inert.")

        if errors:
            for e in errors:
                logger.error(f"PRE-FLIGHT FAIL: {e}")
            raise RuntimeError(f"Training blocked — {len(errors)} pre-flight check(s) failed.")
        logger.info("All pre-flight checks passed.")

    def _verify_dataset_freeze(self) -> List[str]:
        from src.utils.dataset_freeze import verify_freeze
        if not self._freeze_record.exists():
            return [f"Freeze record not found: {self._freeze_record}. Run scripts/freeze_dataset.py first."]
        res = verify_freeze(self._freeze_record, self._root / "data" / "processed", self.dataset_yaml, full=False)
        if not res["ok"]:
            return [f"Dataset freeze verification FAILED: {res.get('mismatches') or res.get('reason')}"]
        self.cfg.dataset_version = res["version"]
        self.cfg.provenance["freeze_record_sha256"] = _sha256(self._freeze_record)
        logger.info(f"Gate: frozen dataset verified (version={res['version']})")
        return []

    # ── setup ─────────────────────────────────────────────────────────────

    def _setup_mlflow(self) -> None:
        Path(self._mlflow_uri).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ["MLFLOW_TRACKING_URI"] = self._mlflow_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self._mlflow_experiment
        os.environ["MLFLOW_RUN"] = self.cfg.experiment_id
        try:
            from ultralytics import settings
            settings.update({"mlflow": True})
        except Exception as e:
            logger.warning(f"Could not enable Ultralytics MLflow integration: {e}")
        self.cfg.mlflow = {"tracking_uri": self._mlflow_uri, "experiment": self._mlflow_experiment,
                           "run_name": self.cfg.experiment_id, "status": "configured"}
        logger.info(f"MLflow: uri={self._mlflow_uri} experiment={self._mlflow_experiment}")

    def _setup_seed(self) -> None:
        import random, numpy as np
        random.seed(self.cfg.seed); np.random.seed(self.cfg.seed); torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        logger.info(f"Random seed set: {self.cfg.seed}")

    # ── training ──────────────────────────────────────────────────────────

    def _run_ultralytics_training(self):
        from ultralytics import YOLO
        exp_dir = self._output_dir / self.cfg.experiment_id

        if self.resume:
            last = exp_dir / "weights" / "last.pt"
            if not last.exists():
                raise RuntimeError(f"Cannot resume — {last} not found.")
            logger.info(f"Resuming from {last}")
            return YOLO(str(last)).train(resume=True)

        weights = self.cfg.pretrained_weights
        if not Path(weights).exists():
            logger.info(f"Weights '{weights}' not local — Ultralytics will download.")
        model = YOLO(weights)
        resolved_yaml = self._resolve_dataset_yaml_for_ultralytics()

        kwargs = dict(
            data=str(resolved_yaml), epochs=self.cfg.epochs, batch=self.cfg.batch_size,
            imgsz=self.cfg.image_size, optimizer=self.cfg.optimizer, lr0=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum,
            device=self.cfg.device, workers=self.cfg.workers, seed=self.cfg.seed,
            deterministic=self.cfg.deterministic, patience=self.cfg.early_stopping_patience,
            project=str(self._output_dir), name=self.cfg.experiment_id, exist_ok=False,
            plots=True, save=True, cache=False,               # cache=disk would write .npy into the frozen tree
            save_period=self._model_cfg.get("training", {}).get("save_period", 10),
            fraction=self.cfg.fraction, verbose=True,
        )
        logger.info(f"Starting Ultralytics training on device={self.cfg.device} ...")
        return model.train(**kwargs)

    def _resolve_dataset_yaml_for_ultralytics(self) -> Path:
        """Write a per-experiment copy with an absolute 'path'. Original dataset.yaml is never modified."""
        cfg = dict(self._dataset_cfg)
        raw = cfg.get("path", "data/processed")
        abs_path = Path(raw) if Path(raw).is_absolute() else (self.dataset_yaml.parent.parent / raw).resolve()
        if not abs_path.exists():
            raise RuntimeError(f"Dataset path does not exist: {abs_path}")
        cfg["path"] = str(abs_path)
        out = self._reports_dir / f"{self.cfg.experiment_id}_dataset_resolved.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
        return out

    # ── results ───────────────────────────────────────────────────────────

    def _extract_best_metrics(self) -> None:
        exp_dir = self._output_dir / self.cfg.experiment_id
        best_pt, last_pt = exp_dir / "weights" / "best.pt", exp_dir / "weights" / "last.pt"
        self.cfg.checkpoint_best = str(best_pt) if best_pt.exists() else ""
        self.cfg.checkpoint_last = str(last_pt) if last_pt.exists() else ""

        csv_path = exp_dir / "results.csv"
        if not csv_path.exists():
            return
        try:
            with open(csv_path) as f:
                rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv_module.DictReader(f)]
        except Exception as e:
            logger.warning(f"Could not parse results.csv: {e}"); return
        if not rows:
            return

        def col(pred):
            return next((k for k in rows[0] if pred(k)), None)
        k_p = col(lambda k: "precision" in k)
        k_r = col(lambda k: "recall" in k)
        k_m50 = col(lambda k: "mAP50" in k and "95" not in k)
        k_m5095 = col(lambda k: "mAP50-95" in k)
        fnum = lambda r, k: float(r.get(k, 0) or 0) if k else 0.0

        scored = [(FITNESS_W_MAP50 * fnum(r, k_m50) + FITNESS_W_MAP5095 * fnum(r, k_m5095), i) for i, r in enumerate(rows)]
        best_fit, bi = max(scored)
        b = rows[bi]
        self.cfg.epochs_completed = len(rows)
        self.cfg.best_epoch = int(float(b.get("epoch", bi + 1)))
        self.cfg.best_fitness = round(best_fit, 6)
        self.cfg.best_map50, self.cfg.best_map50_95 = fnum(b, k_m50), fnum(b, k_m5095)
        self.cfg.best_precision, self.cfg.best_recall = fnum(b, k_p), fnum(b, k_r)
        last = rows[-1]
        self.cfg.final_epoch_metrics = {k: fnum(last, k) for k in (k_p, k_r, k_m50, k_m5095) if k}

    def _log_provenance_to_mlflow(self) -> None:
        """Attach project provenance to the run Ultralytics created. Failure is recorded, never hidden."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self._mlflow_uri)
            exp = mlflow.get_experiment_by_name(self._mlflow_experiment)
            if exp is None:
                raise RuntimeError("MLflow experiment not found — Ultralytics callback did not log this run.")
            runs = mlflow.search_runs([exp.experiment_id],
                                      filter_string=f"tags.mlflow.runName = '{self.cfg.experiment_id}'",
                                      output_format="list")
            if not runs:
                raise RuntimeError("MLflow run not found by name.")
            run_id = runs[0].info.run_id
            with mlflow.start_run(run_id=run_id):
                tags = {f"prov.{k}": str(v) for k, v in self.cfg.provenance.items()}
                tags.update({"run_kind": self.cfg.run_kind, "dataset_version": self.cfg.dataset_version,
                             "device_resolved": self.cfg.device, "status": self.cfg.status})
                mlflow.set_tags(tags)
                mlflow.log_dict(asdict(self.cfg), "experiment_config.json")
            self.cfg.mlflow.update({"status": "logged", "run_id": run_id})
        except Exception as e:
            logger.warning(f"MLflow provenance logging failed: {e}")
            self.cfg.mlflow.update({"status": "failed", "error": str(e)})

    def _promote_to_production(self) -> None:
        prod = self._root / "models" / "production"
        prod.mkdir(parents=True, exist_ok=True)
        dest = prod / "best_model.pt"
        shutil.copy2(self.cfg.checkpoint_best, dest)
        save_json({
            "selected_checkpoint": self.cfg.checkpoint_best, "checkpoint_sha256": _sha256(dest),
            "experiment_id": self.cfg.experiment_id, "dataset_version": self.cfg.dataset_version,
            "best_epoch": self.cfg.best_epoch, "val_mAP50": self.cfg.best_map50, "val_mAP50_95": self.cfg.best_map50_95,
            "selection_reason": "explicit --promote at training time (pre threshold analysis)",
            "promoted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }, prod / "PROVENANCE.json")
        logger.info(f"Promoted to {dest} (PROVENANCE.json written)")

    def _save_final_metadata(self) -> None:
        save_json(asdict(self.cfg), self._reports_dir / f"{self.cfg.experiment_id}_final.json")

    def _build_result(self) -> TrainingResult:
        exp_dir = self._output_dir / self.cfg.experiment_id
        return TrainingResult(
            success=self.cfg.status == STATUS_SUCCESS, experiment_id=self.cfg.experiment_id,
            best_checkpoint=Path(self.cfg.checkpoint_best), last_checkpoint=Path(self.cfg.checkpoint_last),
            best_map50=self.cfg.best_map50, best_epoch=self.cfg.best_epoch,
            training_duration_seconds=self.cfg.training_duration_seconds,
            results_csv=exp_dir / "results.csv", experiment_dir=exp_dir, config=self.cfg,
        )

    # ── provenance / env ──────────────────────────────────────────────────

    def _collect_provenance(self) -> Dict[str, str]:
        p = {
            "git_commit": _git(self._root, "rev-parse", "HEAD"),
            "git_dirty": str(bool(_git(self._root, "status", "--porcelain"))),
            "model_config_sha256": _sha256(self.model_config_yaml),
            "dataset_yaml_sha256": _sha256(self.dataset_yaml),
        }
        manifest = self._root / "data" / "processed" / "manifest.csv"
        p["manifest_sha256"] = _sha256(manifest) if manifest.exists() else "UNAVAILABLE"
        p["validation_report_sha256"] = (_sha256(self.validation_report)
                                         if self.validation_report and self.validation_report.exists() else "UNAVAILABLE")
        return p

    @staticmethod
    def _collect_environment(device: str) -> Dict[str, str]:
        return {
            "python": platform.python_version(), "platform": platform.platform(), "cpu_model": _cpu_model(),
            "torch": torch.__version__, "cuda_available": str(torch.cuda.is_available()),
            "cuda_build": torch.version.cuda or "N/A", "device_resolved": device,
            "ultralytics": _pkg_version("ultralytics"), "opencv": _pkg_version("cv2"),
            "numpy": _pkg_version("numpy"), "pillow": _pkg_version("PIL"), "mlflow": _pkg_version("mlflow"),
        }

    # ── logging ───────────────────────────────────────────────────────────

    def _log_experiment_header(self) -> None:
        c = self.cfg
        logger.info("=" * 60)
        logger.info(f"FRACTURE DETECTION — {c.run_kind.upper()} TRAINING")
        logger.info("=" * 60)
        for k, v in (("Experiment", c.experiment_id), ("Dataset", c.dataset_version), ("Arch", c.model_architecture),
                     ("Weights", c.pretrained_weights), ("imgsz", c.image_size), ("Epochs", c.epochs),
                     ("Batch", c.batch_size), ("Optimizer", f"{c.optimizer} lr0={c.learning_rate}"),
                     ("Device", f"{c.device} (requested: {c.device_requested})"), ("Seed", c.seed),
                     ("Deterministic", c.deterministic), ("Fraction", c.fraction),
                     ("Git", f"{c.provenance.get('git_commit','')[:12]} dirty={c.provenance.get('git_dirty')}")):
            logger.info(f"  {k:<13}: {v}")
        logger.info("=" * 60)

    def _log_training_summary(self, r: TrainingResult) -> None:
        c = self.cfg
        logger.info("=" * 60)
        logger.info(f"TRAINING {c.status}")
        logger.info(f"  Duration      : {r.training_duration_seconds:.0f}s  ({r.training_duration_seconds/3600:.2f} h)")
        logger.info(f"  Epochs done   : {c.epochs_completed}/{c.epochs}")
        logger.info(f"  Best epoch    : {c.best_epoch} (fitness={c.best_fitness:.4f})")
        logger.info(f"  Val P/R       : {c.best_precision:.4f} / {c.best_recall:.4f}")
        logger.info(f"  Val mAP50     : {c.best_map50:.4f}   mAP50-95: {c.best_map50_95:.4f}")
        logger.info(f"  best.pt       : {r.best_checkpoint}")
        logger.info(f"  MLflow        : {c.mlflow.get('status')} {c.mlflow.get('run_id','')}")
        logger.info("=" * 60)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}