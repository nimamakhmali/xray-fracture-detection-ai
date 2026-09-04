"""
src/training/evaluator.py

Two strictly separated metric families, each reported for combined / fracatlas / grazpedwri:

  OBJECT-LEVEL (detection)
    mAP@50, mAP@50-95           Ultralytics val at conf=0.001 (full PR curve, never truncated
                                by the operating threshold).
    P/R/F1, TP/FP/FN, IoU stats our greedy matching at the OPERATING threshold, match IoU>=0.5.
  IMAGE-LEVEL (presence/absence)
    TP/FP/FN/TN, P/R/Spec/F1    fracture-positive image  vs  any detection >= threshold.

Guards
  * frozen dataset verified before (quick) and after (full) every run
  * test split: only with the threshold selected on val for the SAME checkpoint, once (lock)
  * threshold analysis: val only, offline sweep over cached predictions
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.utils.logger import get_logger
from src.utils.file_utils import save_json
from src.data.manifest import ManifestStore
from src.training.trainer import resolve_device
from src.utils.dataset_freeze import verify_freeze
from src.utils.provenance import (current_freeze_version, derive_run_id, git_commit,
                                  lookup_training_metadata, sha256_file)

logger = get_logger(__name__)

MAP_CONF = 0.001          # standard confidence floor for PR-curve integration
MAP_NMS_IOU = 0.7         # Ultralytics default NMS IoU for val/mAP
SOURCES = ("fracatlas", "grazpedwri")
SUBSETS = ("combined",) + SOURCES
DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
FREEZE_RECORD_REL = Path("reports/dataset/frozen_v1.json")
SELECTION_FILE = "threshold_selection_val.json"

SAMPLE_TYPES = ("TP_correct_localization", "TP_poor_or_partial_localization",
                "FN_missed", "FP_false_detection", "TN")


# ── pure helpers ────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def match_greedy(gt: List[list], preds: List[list], tau: float, iou_thr: float):
    """Greedy matching by confidence. Returns tp, fp, fn, matched_ious, poor_localization_count."""
    cand = sorted((q for q in preds if q[4] >= tau), key=lambda q: -q[4])
    used, tp, fp, ious, poor = set(), 0, 0, [], 0
    for q in cand:
        best, bi = 0.0, -1
        for gi, g in enumerate(gt):
            if gi in used:
                continue
            v = _iou(q, g)
            if v > best:
                best, bi = v, gi
        if best >= iou_thr:
            used.add(bi); tp += 1; ious.append(best)
        else:
            fp += 1
            if gt and best > 0:
                poor += 1                      # overlaps a GT but below match IoU
    return tp, fp, len(gt) - len(used), ious, poor


def image_level_metrics(preds: List[dict], tau: float, subset: str) -> dict:
    tp = fp = fn = tn = 0
    for p in preds:
        det = any(q[4] >= tau for q in p["preds"])
        if p["is_positive"]:
            tp += det; fn += (not det)
        else:
            fp += det; tn += (not det)
    P, R, F = _prf(tp, fp, fn)
    return {"level": "image", "subset": subset, "operating_threshold": tau,
            "num_images": len(preds), "num_positive": tp + fn, "num_negative": fp + tn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": P, "recall": R,
            "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0, "f1": F}


def object_level_at_threshold(preds: List[dict], tau: float, match_iou: float, subset: str) -> dict:
    TP = FP = FN = poor = 0
    ious: List[float] = []
    for p in preds:
        tp, fp, fn, ii, pr = match_greedy(p["gt_xyxy"], p["preds"], tau, match_iou)
        TP += tp; FP += fp; FN += fn; poor += pr; ious += ii
    P, R, F = _prf(TP, FP, FN)
    return {"level": "object", "subset": subset, "operating_threshold": tau, "match_iou": match_iou,
            "num_gt_boxes": TP + FN, "num_pred_boxes": TP + FP, "tp": TP, "fp": FP, "fn": FN,
            "precision": P, "recall": R, "f1": F,
            "matched_iou_mean": round(float(np.mean(ious)), 4) if ious else -1.0,
            "matched_iou_median": round(float(np.median(ious)), 4) if ious else -1.0,
            "poor_localization_preds": poor}


def classify_sample(p: dict, tau: float, match_iou: float) -> str:
    det = any(q[4] >= tau for q in p["preds"])
    if not p["is_positive"]:
        return "FP_false_detection" if det else "TN"
    if not det:
        return "FN_missed"
    tp, fp, fn, _, _ = match_greedy(p["gt_xyxy"], p["preds"], tau, match_iou)
    return "TP_correct_localization" if (fn == 0 and fp == 0) else "TP_poor_or_partial_localization"


# ── evaluator ───────────────────────────────────────────────────────────────

class FractureDetectionEvaluator:

    def __init__(
        self,
        weights: Path,
        dataset_yaml: Path,
        manifest_path: Path,
        output_dir: Path,
        reports_dir: Optional[Path] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        match_iou: float = 0.5,
        device: Optional[str] = None,
        image_size: int = 640,
        val_batch: int = 16,
        max_visual_samples: int = 20,
        limit: Optional[int] = None,
        verify_after_full: bool = True,
    ):
        self._root = Path(__file__).resolve().parent.parent.parent
        self.weights = Path(weights).resolve()
        if not self.weights.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights}")
        self.dataset_yaml = Path(dataset_yaml).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.reports_dir = Path(reports_dir).resolve() if reports_dir else self._root / "reports" / "evaluation"
        self.conf, self.iou, self.match_iou = float(confidence_threshold), float(iou_threshold), float(match_iou)
        self.imgsz, self.val_batch = image_size, val_batch
        self.max_visual_samples = max_visual_samples
        self.limit = limit
        self.verify_after_full = verify_after_full
        self.device = resolve_device(device or "auto")
        self._processed = self._root / "data" / "processed"

        self.weights_sha = sha256_file(self.weights)
        self.run_id = derive_run_id(self.weights, self.weights_sha)
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = self.reports_dir / self.run_id
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.training_meta = lookup_training_metadata(self._root, self.weights)
        if self.training_meta["training_status"] not in ("SUCCESS", "UNAVAILABLE"):
            logger.warning(f"Checkpoint comes from a run with status={self.training_meta['training_status']}")
        if self.training_meta["run_kind"] == "smoke_test":
            logger.warning("Checkpoint is a SMOKE-TEST checkpoint — results are not baseline results.")

        self._manifest = ManifestStore.load(self.manifest_path)
        with open(self.dataset_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self._nc, self._names = int(cfg.get("nc", 1)), list(cfg.get("names", ["fracture"]))
        logger.info(f"Evaluator ready: run_id={self.run_id} device={self.device} "
                    f"experiment={self.training_meta['experiment_id']} conf={self.conf}")

    # ── public ───────────────────────────────────────────────────────────

    def evaluate(self, split: str = "val", force_test: bool = False, force_reason: str = "") -> dict:
        if split not in ("val", "test"):
            raise ValueError("split must be val|test")
        dev_partial = self.limit is not None
        if split == "test":
            self._test_guard(force_test, force_reason)

        pre = verify_freeze(self._root / FREEZE_RECORD_REL, self._processed, self.dataset_yaml, full=False)
        if not pre["ok"]:
            raise RuntimeError(f"Frozen dataset verification failed before evaluation: {pre}")

        records = self._split_records(split)
        logger.info(f"Evaluating split={split} n={len(records)} "
                    f"(fracatlas={sum(r.dataset=='fracatlas' for r in records)}, "
                    f"grazpedwri={sum(r.dataset=='grazpedwri' for r in records)})"
                    + (" DEV-PARTIAL" if dev_partial else ""))

        t0 = time.time()
        object_level = {s: self._ultralytics_map(split, s, [r for r in records if s == "combined" or r.dataset == s])
                        for s in SUBSETS}
        preds = self._predict(split, records)
        image_level = {}
        for s in SUBSETS:
            sp = [p for p in preds if s == "combined" or p["dataset"] == s]
            image_level[s] = image_level_metrics(sp, self.conf, s)
            object_level[s].update(object_level_at_threshold(sp, self.conf, self.match_iou, s))

        types = {p["sample_id"]: classify_sample(p, self.conf, self.match_iou) for p in preds}
        self._save_visual_samples(preds, types, split)
        post = verify_freeze(self._root / FREEZE_RECORD_REL, self._processed, self.dataset_yaml,
                             full=self.verify_after_full)
        if not post["ok"]:
            raise RuntimeError(f"Dataset was modified during evaluation: {post}")

        report = {
            "run_id": self.run_id, "split": split, "dev_partial": dev_partial,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "weights": str(self.weights), "weights_sha256": self.weights_sha,
            "experiment_id": self.training_meta["experiment_id"],
            "dataset_version": self.training_meta["dataset_version"] if self.training_meta["dataset_version"] != "UNAVAILABLE"
                               else current_freeze_version(self._root),
            "thresholds": {"operating_confidence": self.conf, "nms_iou": self.iou,
                           "match_iou": self.match_iou, "map_conf": MAP_CONF, "map_nms_iou": MAP_NMS_IOU},
            "num_images": len(records),
            "object_level": object_level,
            "image_level": image_level,
            "timing": self._timing(preds, object_level["combined"]),
            "error_analysis": self._error_analysis(preds, types),
            "domain_shift": self._domain_shift(object_level, image_level),
            "freeze_verification": {"before": pre, "after": {k: v for k, v in post.items() if k != "changed"}
                                    | {"changed_count": len(post.get("changed", []))}},
            "provenance": {"git_commit": git_commit(self._root), "manifest_sha256": sha256_file(self.manifest_path),
                           "dataset_yaml_sha256": sha256_file(self.dataset_yaml),
                           "training_status": self.training_meta["training_status"],
                           "run_kind": self.training_meta["run_kind"]},
            "environment": self._environment(),
            "known_limitations": self._known_limitations(),
            "evaluation_duration_seconds": round(time.time() - t0, 1),
            "artifacts": {"predictions": str(self.run_dir / f"predictions_{split}.jsonl"),
                          "visual_samples": str(self.run_dir / f"visual_samples_{split}")},
        }
        name = f"evaluation_{split}{'_DEVPARTIAL' if dev_partial else ''}.json"
        save_json(report, self.report_dir / name)
        save_json(report, self.run_dir / name)
        if split == "test" and not dev_partial:
            self._write_test_lock(report, force_reason)
        self._print_summary(report)
        return report

    def analyze_thresholds(self, split: str = "val", thresholds: Optional[List[float]] = None,
                           selection_rule: str = "max_image_f1") -> dict:
        if split != "val":
            raise ValueError("Threshold analysis is permitted on the validation split only.")
        if self.limit is not None:
            raise ValueError("Threshold analysis must use the full validation split (no --limit).")
        thresholds = sorted(thresholds or DEFAULT_THRESHOLDS)
        records = self._split_records("val")
        preds = self._predict("val", records)

        rows = []
        for t in thresholds:
            row = {"threshold": t}
            for s in SUBSETS:
                sp = [p for p in preds if s == "combined" or p["dataset"] == s]
                row[f"image_{s}"] = image_level_metrics(sp, t, s)
                row[f"object_{s}"] = object_level_at_threshold(sp, t, self.match_iou, s)
            rows.append(row)
            ic, oc = row["image_combined"], row["object_combined"]
            logger.info(f"  τ={t:.2f}  image P={ic['precision']:.3f} R={ic['recall']:.3f} "
                        f"Spec={ic['specificity']:.3f} F1={ic['f1']:.3f} | object P={oc['precision']:.3f} "
                        f"R={oc['recall']:.3f} F1={oc['f1']:.3f}")

        rules = {"max_image_f1": lambda r: r["image_combined"]["f1"],
                 "max_object_f1": lambda r: r["object_combined"]["f1"]}
        if selection_rule not in rules:
            raise ValueError(f"selection_rule must be one of {list(rules)}")
        best = max(rows, key=rules[selection_rule])          # ties → lowest threshold
        selection = {
            "selected_threshold": best["threshold"], "selection_rule": selection_rule, "split": "val",
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "checkpoint_sha256": self.weights_sha, "weights": str(self.weights), "run_id": self.run_id,
            "experiment_id": self.training_meta["experiment_id"],
            "metrics_at_selected": {k: v for k, v in best.items() if k != "threshold"},
            "note": "Selected on validation only. Test evaluation must use exactly this threshold.",
        }
        analysis = {"run_id": self.run_id, "weights_sha256": self.weights_sha, "split": "val",
                    "match_iou": self.match_iou, "nms_iou": self.iou, "thresholds": thresholds, "rows": rows,
                    "selection": selection}
        save_json(analysis, self.report_dir / "threshold_analysis_val.json")
        save_json(selection, self.report_dir / SELECTION_FILE)
        save_json(selection, self.reports_dir / SELECTION_FILE)
        logger.info(f"Selected threshold={best['threshold']} by {selection_rule} "
                    f"(image F1={best['image_combined']['f1']}, object F1={best['object_combined']['f1']})")
        return analysis

    # ── guards ───────────────────────────────────────────────────────────

    def _test_guard(self, force: bool, reason: str) -> None:
        if self.limit is not None:
            raise RuntimeError("--limit is not allowed on the test split.")
        sel_path = self.reports_dir / SELECTION_FILE
        if not sel_path.exists():
            raise RuntimeError("No threshold_selection_val.json — run threshold analysis on val first.")
        sel = json.loads(sel_path.read_text())
        if sel["checkpoint_sha256"] != self.weights_sha:
            raise RuntimeError("Threshold selection belongs to a different checkpoint. Re-run val analysis for this one.")
        if abs(sel["selected_threshold"] - self.conf) > 1e-9:
            raise RuntimeError(f"conf={self.conf} != selected {sel['selected_threshold']}. Use --use-selected-threshold.")
        lock = self.report_dir / "TEST_LOCK.json"
        if lock.exists() and not (force and reason.strip()):
            raise RuntimeError("Test split already evaluated for this checkpoint (TEST_LOCK.json). "
                               "Re-run requires --force-test-rerun --reason '...' and is recorded.")
        if lock.exists():
            logger.warning(f"TEST RE-RUN FORCED — reason will be recorded: {reason}")

    def _write_test_lock(self, report: dict, reason: str) -> None:
        lock = self.report_dir / "TEST_LOCK.json"
        hist = json.loads(lock.read_text())["history"] if lock.exists() else []
        hist.append({"at": report["created_at"], "threshold": self.conf, "reason": reason or "first_and_only_run",
                     "map50_combined": report["object_level"]["combined"]["map50"]})
        save_json({"weights_sha256": self.weights_sha, "history": hist}, lock)

    # ── inference ────────────────────────────────────────────────────────

    def _split_records(self, split: str) -> list:
        recs = sorted(self._manifest.by_split(split), key=lambda r: r.sample_id)
        if self.limit:
            fa = [r for r in recs if r.dataset == "fracatlas"][: self.limit // 2]
            gz = [r for r in recs if r.dataset == "grazpedwri"][: self.limit - len(fa)]
            recs = fa + gz
        return recs

    def _ultralytics_map(self, split: str, subset: str, records: list) -> dict:
        base = {"level": "object", "subset": subset, "num_images": len(records),
                "map50": -1.0, "map50_95": -1.0, "ultralytics_precision_f1opt": -1.0,
                "ultralytics_recall_f1opt": -1.0, "ultralytics_speed_ms": {}, "ultralytics_val_batch": self.val_batch,
                "note": "P/R from Ultralytics are at its F1-optimal confidence, NOT at the operating threshold."}
        if not records:
            return base
        from ultralytics import YOLO
        lst = self.run_dir / f"{split}_{subset}_images.txt"
        lst.write_text("\n".join(str(self._processed / split / "images" / r.image_path) for r in records))
        data_yaml = self.run_dir / f"{split}_{subset}_data.yaml"
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.dump({"path": str(self._processed), "train": "train/images", "val": str(lst),
                       "nc": self._nc, "names": self._names}, f)
        m = YOLO(str(self.weights)).val(
            data=str(data_yaml), split="val", conf=MAP_CONF, iou=MAP_NMS_IOU, imgsz=self.imgsz,
            batch=self.val_batch, device=self.device, plots=(subset == "combined"),
            project=str(self.run_dir), name=f"ultralytics_val_{split}_{subset}", exist_ok=True, verbose=False)
        base.update(map50=round(float(m.box.map50), 4), map50_95=round(float(m.box.map), 4),
                    ultralytics_precision_f1opt=round(float(m.box.mp), 4),
                    ultralytics_recall_f1opt=round(float(m.box.mr), 4),
                    ultralytics_speed_ms={k: round(float(v), 2) for k, v in m.speed.items()})
        logger.info(f"[{subset}/{split}] mAP50={base['map50']:.4f} mAP50-95={base['map50_95']:.4f} n={len(records)}")
        return base

    def _predict(self, split: str, records: list) -> List[dict]:
        cache = self.run_dir / f"predictions_{split}.jsonl"
        header = {"weights_sha256": self.weights_sha, "nms_iou": self.iou, "min_conf": MAP_CONF,
                  "imgsz": self.imgsz, "n": len(records), "first": records[0].sample_id if records else ""}
        if cache.exists():
            lines = cache.read_text(encoding="utf-8").splitlines()
            if lines and json.loads(lines[0]) == header:
                logger.info(f"Reusing cached predictions: {cache}")
                return [json.loads(l) for l in lines[1:]]
        from ultralytics import YOLO
        model = YOLO(str(self.weights))
        paths = [str(self._processed / split / "images" / r.image_path) for r in records]
        out: List[dict] = []
        logger.info(f"Predicting {len(paths)} images (batch=1, conf={MAP_CONF}, nms_iou={self.iou}) ...")
        stream = model.predict(source=paths, stream=True, conf=MAP_CONF, iou=self.iou, imgsz=self.imgsz,
                               max_det=100, device=self.device, verbose=False)
        for i, (r, res) in enumerate(zip(records, stream)):
            h, w = (int(v) for v in res.orig_shape)
            preds = [] if res.boxes is None else [
                [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2]), 2), round(float(b[3]), 2),
                 round(float(c), 4)] for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
            out.append({"sample_id": r.sample_id, "dataset": r.dataset, "image": f"{split}/images/{r.image_path}",
                        "is_positive": r.fracture_positive == "True", "w": w, "h": h,
                        "gt_xyxy": self._load_gt(split, r.image_path, w, h), "preds": preds,
                        "speed_ms": {k: round(float(v), 2) for k, v in res.speed.items()}})
            if (i + 1) % 500 == 0:
                logger.info(f"  {i + 1}/{len(paths)}")
        with open(cache, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for p in out:
                f.write(json.dumps(p) + "\n")
        return out

    def _load_gt(self, split: str, image_name: str, w: int, h: int) -> List[list]:
        lp = self._processed / split / "labels" / (Path(image_name).stem + ".txt")
        boxes = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    _, xc, yc, bw, bh = map(float, parts)
                    boxes.append([round((xc - bw / 2) * w, 2), round((yc - bh / 2) * h, 2),
                                  round((xc + bw / 2) * w, 2), round((yc + bh / 2) * h, 2)])
        return boxes

    # ── analysis / artifacts ────────────────────────────────────────────

    def _timing(self, preds: List[dict], combined_obj: dict) -> dict:
        inf = [p["speed_ms"].get("inference", 0.0) for p in preds]
        tot = [sum(p["speed_ms"].values()) for p in preds]
        return {"note": "batch-1 predict pass (no warm-up exclusion) — scripts/benchmark.py is authoritative",
                "device": self.device, "images": len(preds),
                "inference_ms_mean": round(statistics.fmean(inf), 2) if inf else -1,
                "inference_ms_median": round(statistics.median(inf), 2) if inf else -1,
                "total_ms_mean": round(statistics.fmean(tot), 2) if tot else -1,
                "ultralytics_val_speed_ms_per_image": combined_obj.get("ultralytics_speed_ms", {}),
                "ultralytics_val_batch": self.val_batch}

    def _error_analysis(self, preds: List[dict], types: Dict[str, str]) -> dict:
        counts = {t: {s: 0 for s in SOURCES} for t in SAMPLE_TYPES}
        examples = {t: [] for t in SAMPLE_TYPES}
        poor = {s: 0 for s in SOURCES}
        for p in preds:
            t = types[p["sample_id"]]
            counts[t][p["dataset"]] += 1
            if len(examples[t]) < 10:
                examples[t].append(p["sample_id"])
            if p["is_positive"]:
                poor[p["dataset"]] += match_greedy(p["gt_xyxy"], p["preds"], self.conf, self.match_iou)[4]
        pos = {s: sum(1 for p in preds if p["dataset"] == s and p["is_positive"]) for s in SOURCES}
        return {
            "operating_threshold": self.conf, "match_iou": self.match_iou,
            "sample_type_counts_by_source": counts,
            "image_level_miss_rate_by_source": {s: round(counts["FN_missed"][s] / pos[s], 4) if pos[s] else -1 for s in SOURCES},
            "poor_localization_preds_by_source": poor,
            "example_ids": examples,
            "clinically_adjacent_negatives": {
                "status": "NOT_TRACKED_PER_SAMPLE",
                "note": ("Phase 1 reported 169 GRAZPEDWRI-DX fracture-negative images with fracture-adjacent findings "
                         "(periosteal reaction, pronator sign, bone anomaly). The manifest annotation_status only "
                         "distinguishes positive_clean/negative_clean, so FPs cannot be attributed to these cases "
                         "per-sample in Phase 2. Recorded as a limitation; no relabeling was performed.")},
        }

    @staticmethod
    def _domain_shift(obj: dict, img: dict) -> dict:
        fa, gz = obj["fracatlas"], obj["grazpedwri"]
        ifa, igz = img["fracatlas"], img["grazpedwri"]
        d = lambda a, b: round(a - b, 4) if a >= 0 and b >= 0 else "N/A"
        return {
            "object_map50": {"fracatlas": fa["map50"], "grazpedwri": gz["map50"], "delta_fa_minus_grz": d(fa["map50"], gz["map50"])},
            "object_map50_95": {"fracatlas": fa["map50_95"], "grazpedwri": gz["map50_95"], "delta_fa_minus_grz": d(fa["map50_95"], gz["map50_95"])},
            "object_recall_at_threshold": {"fracatlas": fa["recall"], "grazpedwri": gz["recall"], "delta": d(fa["recall"], gz["recall"])},
            "image_recall": {"fracatlas": ifa["recall"], "grazpedwri": igz["recall"], "delta": d(ifa["recall"], igz["recall"])},
            "image_precision": {"fracatlas": ifa["precision"], "grazpedwri": igz["precision"], "delta": d(ifa["precision"], igz["precision"])},
            "positive_rate": {"fracatlas": round(ifa["num_positive"] / ifa["num_images"], 4) if ifa["num_images"] else -1,
                              "grazpedwri": round(igz["num_positive"] / igz["num_images"], 4) if igz["num_images"] else -1},
            "training_share": "grazpedwri ≈ 83.5% of training images, fracatlas ≈ 16.5% (frozen_v1)",
            "hypotheses_not_conclusions": [
                "anatomical scope: pediatric wrist only (GRAZ) vs multi-region adult (FracAtlas)",
                "class prior: ~67% positive (GRAZ) vs ~18% positive (FracAtlas)",
                "training data share strongly favours GRAZ",
                "annotation style / box tightness may differ between sources",
                "FracAtlas IMG0004000+ batch had systematic JPEG issues (58 dropped, 27 EOI-less)"],
            "note": "Differences are observations; causes are hypotheses requiring dedicated analysis.",
        }

    def _save_visual_samples(self, preds: List[dict], types: Dict[str, str], split: str) -> None:
        out_root = self.run_dir / f"visual_samples_{split}"
        per_type = max(2, self.max_visual_samples // len(SAMPLE_TYPES))
        by_type: Dict[str, List[dict]] = {t: [] for t in SAMPLE_TYPES}
        for p in preds:
            by_type[types[p["sample_id"]]].append(p)
        for t, lst in by_type.items():
            if not lst:
                continue
            fa = [p for p in lst if p["dataset"] == "fracatlas"]
            gz = [p for p in lst if p["dataset"] == "grazpedwri"]
            chosen, i = [], 0
            while len(chosen) < per_type and (i < len(fa) or i < len(gz)):
                if i < len(fa): chosen.append(fa[i])
                if i < len(gz) and len(chosen) < per_type: chosen.append(gz[i])
                i += 1
            d = out_root / t
            d.mkdir(parents=True, exist_ok=True)
            for p in chosen:
                self._draw(p, t, d / f"{p['dataset']}_{Path(p['image']).stem}.jpg")
        logger.info(f"Visual samples: {out_root}")

    def _draw(self, p: dict, sample_type: str, out: Path) -> None:
        img = cv2.imread(str(self._processed / p["image"]))
        if img is None:
            return
        for g in p["gt_xyxy"]:
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), 2)
        for q in p["preds"]:
            if q[4] < self.conf:
                continue
            best = max((_iou(q, g) for g in p["gt_xyxy"]), default=0.0)
            cv2.rectangle(img, (int(q[0]), int(q[1])), (int(q[2]), int(q[3])), (0, 0, 255), 2)
            cv2.putText(img, f"{q[4]:.2f} IoU={best:.2f}", (int(q[0]), max(int(q[1]) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, f"{sample_type} | GT=green PRED=red | tau={self.conf}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.imwrite(str(out), img)

    def _environment(self) -> dict:
        import torch, ultralytics
        return {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "device": self.device, "platform": platform.platform()}

    @staticmethod
    def _known_limitations() -> List[str]:
        return [
            "FracAtlas: patient identifiers unavailable → image-level split; patient-level leakage unverifiable.",
            "GRAZPEDWRI-DX: 169 fracture-negative images contain fracture-adjacent findings (not tracked per sample).",
            "Domain: GRAZPEDWRI-DX is pediatric wrist only; FracAtlas is multi-region. Performance is not uniform across anatomy.",
            "Test split was touched once by an exploratory evaluation on 2026-09-03 (file EXPLORATORY_evaluation_test_20260903_DO_NOT_CITE.json); "
            "no tuning decision was derived from it.",
            "Research prototype — not clinically validated.",
        ]

    def _print_summary(self, r: dict) -> None:
        logger.info("=" * 64)
        logger.info(f"EVALUATION {r['split'].upper()}  run_id={r['run_id']}  τ={self.conf}")
        for s in SUBSETS:
            o, i = r["object_level"][s], r["image_level"][s]
            logger.info(f"  {s:<10} OBJ mAP50={o['map50']:.4f} mAP50-95={o['map50_95']:.4f} "
                        f"P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} | "
                        f"IMG P={i['precision']:.3f} R={i['recall']:.3f} Spec={i['specificity']:.3f} F1={i['f1']:.3f} "
                        f"(n={i['num_images']})")
        logger.info(f"  timing: {r['timing']['inference_ms_mean']} ms/img inference (batch 1, {self.device})")
        logger.info("=" * 64)