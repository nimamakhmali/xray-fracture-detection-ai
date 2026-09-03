#!/usr/bin/env python3
"""
scripts/explain.py

Generate explainability visualizations (Grad-CAM heatmaps + bounding boxes)
for fracture detections.

Usage:
    # explain single image
    python scripts/explain.py \
        --weights models/production/best_model.pt \
        --image data/processed/val/images/grz_0001_....png

    # explain multiple images from val set
    python scripts/explain.py \
        --weights models/production/best_model.pt \
        --val-samples 5

    # explain with lower confidence threshold
    python scripts/explain.py \
        --weights models/production/best_model.pt \
        --image <path> --conf 0.15
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability.cam import YOLOExplainability
from src.data.manifest import ManifestStore
from src.utils.logger import get_logger

logger = get_logger("explain")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Grad-CAM explainability for fracture detections."
    )
    p.add_argument(
        "--weights", required=True,
        help="Path to model weights (.pt)",
    )
    p.add_argument("--image", default=None, help="Single image path")
    p.add_argument(
        "--val-samples", type=int, default=0,
        help="Explain N random val-set TP samples",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument(
        "--detection-index", type=int, default=0,
        help="Which detection to explain (0=highest confidence)",
    )
    p.add_argument(
        "--output-dir", default="runs/explainability",
    )
    p.add_argument(
        "--manifest", default="data/processed/manifest.csv",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    weights = root / args.weights
    if not weights.exists():
        logger.error(f"Weights not found: {weights}")
        sys.exit(1)

    output_dir = root / args.output_dir
    explainer = YOLOExplainability(
        model_path=weights,
        device=args.device,
    )

    images_to_explain = []

    if args.image:
        images_to_explain.append(Path(args.image))

    if args.val_samples > 0:
        manifest_path = root / args.manifest
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            sys.exit(1)
        store = ManifestStore.load(manifest_path)
        val_positive = [
            r for r in store.by_split("val")
            if r.fracture_positive == "True"
        ]
        import random
        random.seed(42)
        selected = random.sample(
            val_positive, min(args.val_samples, len(val_positive))
        )
        processed_dir = root / "data" / "processed"
        for r in selected:
            img_path = processed_dir / "val" / "images" / r.image_path
            if img_path.exists():
                images_to_explain.append(img_path)

    if not images_to_explain:
        logger.error("No images to explain. Use --image or --val-samples.")
        sys.exit(1)

    success = 0
    for img_path in images_to_explain:
        logger.info(f"Explaining: {img_path.name}")
        try:
            results = explainer.explain(
                image_path=img_path,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                detection_index=args.detection_index,
                output_dir=output_dir / img_path.stem,
            )
            if results:
                success += 1
                logger.info(
                    f"  → {len(results)} detection(s) explained, "
                    f"method={results[0].heatmap_method}"
                )
            else:
                logger.info("  → No detections found.")
        except Exception as e:
            logger.error(f"  → Failed: {e}")

    logger.info(f"Explainability complete: {success}/{len(images_to_explain)} images.")
    sys.exit(0)


if __name__ == "__main__":
    main()