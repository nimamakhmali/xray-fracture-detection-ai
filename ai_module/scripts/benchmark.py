#!/usr/bin/env python3
"""
scripts/benchmark.py — CPU inference benchmark (batch 1). Warm-up iterations excluded.
  python scripts/benchmark.py --weights <best.pt> --num-images 300 --warmup 20
Output: reports/evaluation/<run_id>/benchmark_<device>.json
"""
import argparse
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.manifest import ManifestStore
from src.training.trainer import resolve_device, _cpu_model
from src.utils.file_utils import save_json
from src.utils.provenance import derive_run_id, lookup_training_metadata, sha256_file
from src.utils.logger import get_logger

logger = get_logger("benchmark")


def stats(xs):
    xs = sorted(xs)
    return {"mean": round(statistics.fmean(xs), 2), "median": round(statistics.median(xs), 2),
            "p95": round(xs[int(0.95 * (len(xs) - 1))], 2), "min": round(xs[0], 2), "max": round(xs[-1], 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--num-images", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reports-dir", default="reports/evaluation")
    a = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    weights = Path(a.weights) if Path(a.weights).is_absolute() else root / a.weights
    device = resolve_device(a.device or "auto")
    sha = sha256_file(weights)
    run_id = derive_run_id(weights, sha)

    import torch, ultralytics
    from ultralytics import YOLO
    store = ManifestStore.load(root / "data" / "processed" / "manifest.csv")
    recs = sorted(store.by_split(a.split), key=lambda r: r.sample_id)
    random.Random(a.seed).shuffle(recs)
    paths = [root / "data" / "processed" / a.split / "images" / r.image_path for r in recs[: a.warmup + a.num_images]]
    model = YOLO(str(weights))

    pre, inf, post, wall = [], [], [], []
    for i, pth in enumerate(paths):
        t0 = time.perf_counter()
        res = model.predict(str(pth), conf=a.conf, iou=a.iou, imgsz=a.image_size, device=device, verbose=False)[0]
        t1 = time.perf_counter()
        if i < a.warmup:
            continue
        pre.append(res.speed["preprocess"]); inf.append(res.speed["inference"]); post.append(res.speed["postprocess"])
        wall.append((t1 - t0) * 1000)
    report = {
        "run_id": run_id, "weights": str(weights), "checkpoint_sha256": sha,
        "training": lookup_training_metadata(root, weights),
        "device": device, "batch_size": 1, "image_size": a.image_size, "conf": a.conf, "nms_iou": a.iou,
        "warmup_excluded": a.warmup, "measured_images": len(wall), "split": a.split, "seed": a.seed,
        "ms_per_image": {"preprocess": stats(pre), "inference": stats(inf), "postprocess": stats(post),
                         "end_to_end_wall": stats(wall)},
        "images_per_second_end_to_end": round(1000 / statistics.fmean(wall), 2),
        "environment": {"cpu_model": _cpu_model(), "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "ultralytics": ultralytics.__version__, "platform": platform.platform()},
        "note": "Single-image latency incl. Ultralytics letterbox preprocess and NMS; CPU only.",
    }
    out = root / a.reports_dir / run_id / f"benchmark_{'cpu' if device == 'cpu' else 'gpu'}.json"
    save_json(report, out)
    logger.info(f"{report['ms_per_image']['end_to_end_wall']['mean']} ms/img end-to-end "
                f"({report['images_per_second_end_to_end']} img/s), inference mean "
                f"{report['ms_per_image']['inference']['mean']} ms on {_cpu_model()} → {out}")


if __name__ == "__main__":
    main()