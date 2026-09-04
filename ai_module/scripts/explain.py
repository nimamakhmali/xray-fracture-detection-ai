#!/usr/bin/env python3
"""
scripts/explain.py
  single image : python scripts/explain.py --weights <best.pt> --image <path>
  val TP set   : python scripts/explain.py --weights <best.pt> --val-samples 8 --conf <selected τ>
Outputs: runs/explainability/<run_id>/<image_stem>/*  +  reports/explainability/<run_id>_summary.json
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability.cam import YOLOExplainability, DEFAULT_TARGET_LAYER
from src.data.manifest import ManifestStore
from src.utils.file_utils import save_json
from src.utils.provenance import derive_run_id, sha256_file
from src.utils.logger import get_logger

logger = get_logger("explain")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--val-samples", type=int, default=0, help="explain N val-positive images with a detection (TPs)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--detection-index", type=int, default=0, help="-1 = all detections")
    p.add_argument("--target-layer", default=DEFAULT_TARGET_LAYER)
    p.add_argument("--output-dir", default="runs/explainability")
    p.add_argument("--reports-dir", default="reports/explainability")
    p.add_argument("--manifest", default="data/processed/manifest.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    root = Path(__file__).resolve().parent.parent
    weights = Path(a.weights) if Path(a.weights).is_absolute() else root / a.weights
    if not weights.exists():
        logger.error(f"Weights not found: {weights}"); sys.exit(1)
    run_id = derive_run_id(weights, sha256_file(weights))
    out_root = root / a.output_dir / run_id
    ex = YOLOExplainability(weights, target_layer=a.target_layer, device=a.device)

    todo = []
    if a.image:
        todo.append(Path(a.image) if Path(a.image).is_absolute() else root / a.image)
    skipped_no_det = 0
    if a.val_samples > 0:
        store = ManifestStore.load(root / a.manifest)
        pos = sorted((r for r in store.by_split("val") if r.fracture_positive == "True"), key=lambda r: r.sample_id)
        random.Random(a.seed).shuffle(pos)
        for r in pos:
            if len(todo) >= a.val_samples + (1 if a.image else 0):
                break
            img = root / "data" / "processed" / "val" / "images" / r.image_path
            b = ex.detect(img, a.conf, a.iou)
            if b is None or len(b) == 0:
                skipped_no_det += 1; continue
            todo.append(img)
    if not todo:
        logger.error("Nothing to explain (use --image or --val-samples)."); sys.exit(1)

    entries, methods = [], {"gradcam": 0, "activation": 0}
    for img in todo:
        try:
            res = ex.explain(img, a.conf, a.iou, a.detection_index, out_root / img.stem)
            for r in res:
                methods[r.heatmap_method] = methods.get(r.heatmap_method, 0) + 1
            entries.append({"image": str(img), "detections_explained": len(res),
                            "methods": [r.heatmap_method for r in res], "confidences": [round(r.confidence, 4) for r in res],
                            "anchor_iou": [round(r.anchor_iou_with_detection, 3) for r in res], "output_dir": str(out_root / img.stem)})
            logger.info(f"{img.name}: {len(res)} explained {[r.heatmap_method for r in res]}")
        except Exception as e:
            entries.append({"image": str(img), "error": str(e)})
            logger.error(f"{img.name}: FAILED — {e}")

    summary = {"run_id": run_id, "weights": str(weights), "checkpoint_sha256": ex.checkpoint_sha256,
               "experiment_id": ex.experiment_id, "dataset_version": ex.dataset_version,
               "target_layer": ex.target_layer, "conf": a.conf, "iou": a.iou, "seed": a.seed,
               "val_positives_skipped_no_detection": skipped_no_det, "method_counts": methods, "entries": entries,
               "disclaimer": "Heatmaps are model-debugging artifacts; not medically validated."}
    save_json(summary, out_root / "index.json")
    save_json(summary, root / a.reports_dir / f"{run_id}_summary.json")
    logger.info(f"Explainability done: {sum('error' not in e for e in entries)}/{len(entries)} ok → {out_root}")


if __name__ == "__main__":
    main()