"""
src/explainability/cam.py

Detection-conditioned Grad-CAM for YOLOv8.

Method
  1. Ultralytics predict → post-NMS detections (original image coordinates).
  2. Letterbox the image exactly as Ultralytics does (size 640, pad 114) and run the raw
     DetectionModel with gradients enabled. The raw head output y has shape [1, 4+nc, N]:
     decoded xywh boxes (letterbox pixels) + sigmoid class scores per anchor.
  3. Map the chosen detection into letterbox coordinates and select the ANCHOR whose decoded
     box best overlaps it (IoU>=0.5 → highest score among those; else max IoU).
     This is what makes the heatmap specific to THAT detection.
  4. Backprop that anchor's class score to the target layer (default SPPF, model.9);
     CAM = ReLU(Σ_c mean(∂score/∂A_c) · A_c). Upsample, crop the letterbox padding, resize
     to the original image.
  5. Fallback ONLY if gradients fail but activations were captured: activation-magnitude map,
     clearly labelled "activation". If nothing was captured the call raises — no synthetic maps.

Heatmaps are a model-debugging / transparency tool. They are NOT medically validated.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.utils.logger import get_logger
from src.utils.file_utils import save_json
from src.utils.provenance import current_freeze_version, lookup_training_metadata, sha256_file

logger = get_logger(__name__)

DEFAULT_TARGET_LAYER = "model.9"   # SPPF — last backbone feature map (stride 32) in YOLOv8n
PAD_VALUE = 114
DISCLAIMER = ("This heatmap shows which image regions most influenced the selected detection's "
              "confidence score. It is NOT medically validated and NOT a clinical finding.")


def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    """Min-max to [0,1]; constant/zero/NaN maps become all-zeros (never NaN)."""
    hm = np.nan_to_num(np.asarray(hm, dtype=np.float32))
    mn, mx = float(hm.min()), float(hm.max())
    return np.zeros_like(hm) if mx - mn < 1e-12 else (hm - mn) / (mx - mn)


def letterbox_params(h: int, w: int, size: int) -> Tuple[float, int, int, int, int]:
    """ratio, new_h, new_w, pad_top, pad_left — mirrors Ultralytics LetterBox(auto=False, center=True)."""
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    return r, nh, nw, (size - nh) // 2, (size - nw) // 2


def _iou_1_to_n(boxes: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ix1 = torch.maximum(boxes[:, 0], b[0]); iy1 = torch.maximum(boxes[:, 1], b[1])
    ix2 = torch.minimum(boxes[:, 2], b[2]); iy2 = torch.minimum(boxes[:, 3], b[3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area.clamp(min=1e-9)


@dataclass
class ExplainabilityResult:
    image_path: str
    detection_index: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_normalized: List[float]
    heatmap_method: str                       # gradcam | activation
    target_layer: str
    heatmap_normalized: Optional[np.ndarray] = field(default=None, repr=False)
    overlay: Optional[np.ndarray] = field(default=None, repr=False)
    model_architecture: str = "yolov8n"
    image_size: int = 640
    notes: List[str] = field(default_factory=list)
    anchor_index: int = -1
    anchor_iou_with_detection: float = -1.0
    checkpoint_sha256: str = ""
    experiment_id: str = "UNAVAILABLE"
    dataset_version: str = "UNAVAILABLE"
    letterbox: Dict[str, float] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        return {
            "image_path": self.image_path, "detection_index": self.detection_index,
            "class": self.class_name, "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(x, 1) for x in self.bbox_xyxy],
            "bbox_normalized": [round(x, 4) for x in self.bbox_normalized],
            "heatmap_method": self.heatmap_method, "target_layer": self.target_layer,
            "anchor_index": self.anchor_index, "anchor_iou_with_detection": round(self.anchor_iou_with_detection, 4),
            "model_architecture": self.model_architecture, "image_size": self.image_size,
            "checkpoint_sha256": self.checkpoint_sha256, "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version, "letterbox": self.letterbox,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "notes": self.notes, "disclaimer": DISCLAIMER,
        }


class YOLOExplainability:

    def __init__(self, model_path: Path, target_layer: str = DEFAULT_TARGET_LAYER,
                 device: Optional[str] = None, class_names: Optional[List[str]] = None, image_size: int = 640):
        from ultralytics import YOLO
        from src.training.trainer import resolve_device
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        self.target_layer = target_layer
        self.image_size = image_size
        dev = resolve_device(device or "auto")
        self.device = "cpu" if dev == "cpu" else f"cuda:{dev.split(',')[0]}"
        self._yolo = YOLO(str(self.model_path))
        self._model = self._yolo.model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)                       # gradients flow via the input only
        self._layer = self._get_layer_by_name(target_layer)
        if self._layer is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model.")
        names = getattr(self._model, "names", None)
        self.class_names = class_names or ([names[i] for i in sorted(names)] if isinstance(names, dict) else ["fracture"])
        root = Path(__file__).resolve().parent.parent.parent
        self.checkpoint_sha256 = sha256_file(self.model_path)
        meta = lookup_training_metadata(root, self.model_path)
        self.experiment_id = meta["experiment_id"]
        self.dataset_version = meta["dataset_version"] if meta["dataset_version"] != "UNAVAILABLE" else current_freeze_version(root)
        logger.info(f"Explainability: ckpt={self.model_path.name} sha={self.checkpoint_sha256[:8]} "
                    f"experiment={self.experiment_id} layer={target_layer} device={self.device}")

    # ── public ───────────────────────────────────────────────────────────

    def detect(self, image_path: Path, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        res = self._yolo.predict(str(image_path), conf=conf_threshold, iou=iou_threshold,
                                 imgsz=self.image_size, device=self.device, verbose=False)
        return res[0].boxes if res and res[0].boxes is not None else None

    def explain(self, image_path: Path, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                detection_index: int = 0, output_dir: Optional[Path] = None) -> List[ExplainabilityResult]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        boxes = self.detect(image_path, conf_threshold, iou_threshold)
        n = 0 if boxes is None else len(boxes)
        if n == 0:
            logger.info(f"No detections ≥{conf_threshold}: {image_path.name}")
            return []
        if detection_index == -1:
            idxs = list(range(n))
        elif detection_index >= n:
            logger.warning(f"detection_index={detection_index} out of range ({n}); explaining index 0")
            idxs = [0]
        else:
            idxs = [detection_index]
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"cv2 could not read {image_path}")
        return [self._explain_one(image_path, img, boxes, i, output_dir) for i in idxs]

    # ── core ─────────────────────────────────────────────────────────────

    def _explain_one(self, image_path, img, boxes, idx, output_dir) -> ExplainabilityResult:
        conf = float(boxes.conf[idx]); xyxy = [float(v) for v in boxes.xyxy[idx]]
        cls_id = int(boxes.cls[idx])
        name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
        h, w = img.shape[:2]
        r, nh, nw, top, left = letterbox_params(h, w, self.image_size)

        cam_lb, method, notes, a_idx, a_iou = self._gradcam(img, xyxy, cls_id, (r, nh, nw, top, left))
        crop = cam_lb[top:top + nh, left:left + nw]
        heat = normalize_heatmap(cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR))
        if float(heat.max()) == 0.0:
            notes.append("WARNING: heatmap is all-zero after ReLU/normalisation (no positive attribution).")

        heat_u8 = (heat * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.55, heat_color, 0.45, 0)
        boxed = overlay.copy()
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(boxed, f"{name} {conf:.2f} [{method}]", (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        res = ExplainabilityResult(
            image_path=str(image_path), detection_index=idx, class_name=name, confidence=conf, bbox_xyxy=xyxy,
            bbox_normalized=[(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h],
            heatmap_method=method, target_layer=self.target_layer, heatmap_normalized=heat, overlay=overlay,
            image_size=self.image_size, notes=notes, anchor_index=a_idx, anchor_iou_with_detection=a_iou,
            checkpoint_sha256=self.checkpoint_sha256, experiment_id=self.experiment_id,
            dataset_version=self.dataset_version,
            letterbox={"ratio": round(r, 6), "pad_top": top, "pad_left": left, "orig_h": h, "orig_w": w})
        if output_dir:
            self._save(res, img, heat_color, overlay, boxed, Path(output_dir), image_path.stem, idx)
        return res

    def _gradcam(self, img_bgr, det_xyxy, cls_id, lb) -> Tuple[np.ndarray, str, List[str], int, float]:
        r, nh, nw, top, left = lb
        S = self.image_size
        holder: Dict[str, torch.Tensor] = {}

        def fwd_hook(_m, _i, out):
            out.retain_grad(); holder["a"] = out

        handle = self._layer.register_forward_hook(fwd_hook)
        notes: List[str] = []
        try:
            x = self._preprocess(img_bgr, lb).requires_grad_(True)
            with torch.enable_grad():
                out = self._model(x)
                y = out[0] if isinstance(out, (list, tuple)) else out
                if y.dim() != 3 or y.shape[1] < 4 + 1 + cls_id:
                    raise RuntimeError(f"Unexpected raw head output shape {tuple(y.shape)}")
                xywh = y[0, :4].T
                anchors = torch.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                                       xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
                scores = y[0, 4 + cls_id]
                tb = torch.tensor([det_xyxy[0] * r + left, det_xyxy[1] * r + top,
                                   det_xyxy[2] * r + left, det_xyxy[3] * r + top], device=y.device)
                ious = _iou_1_to_n(anchors.detach(), tb)
                good = ious >= 0.5
                a_idx = int((scores.detach() * good).argmax()) if bool(good.any()) else int(ious.argmax())
                a_iou = float(ious[a_idx])
                if not bool(good.any()):
                    notes.append(f"No raw anchor reached IoU≥0.5 with the detection; used max-IoU anchor (IoU={a_iou:.3f}).")
                self._model.zero_grad(set_to_none=True)
                scores[a_idx].backward()
            act = holder["a"]
            if act.grad is None:
                raise RuntimeError("No gradient reached the target layer.")
            wts = act.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((wts * act).sum(1))[0].detach().cpu().numpy()
            notes.append(f"Grad-CAM on {self.target_layer} ({tuple(act.shape[1:])}), target = raw class score of "
                         f"anchor #{a_idx} matched to the detection (IoU={a_iou:.3f}), pre-NMS.")
            return cv2.resize(cam, (S, S), interpolation=cv2.INTER_LINEAR), "gradcam", notes, a_idx, a_iou
        except Exception as e:
            if "a" in holder:
                logger.warning(f"Grad-CAM failed ({e}); using activation-magnitude map from the same forward pass.")
                act = holder["a"].detach().abs().mean(1)[0].cpu().numpy()
                notes.append(f"FALLBACK activation-magnitude map (not detection-specific). Reason: {e}")
                return cv2.resize(act, (S, S), interpolation=cv2.INTER_LINEAR), "activation", notes, -1, -1.0
            raise RuntimeError(f"Explainability failed and no activations were captured: {e}") from e
        finally:
            handle.remove()

    def _preprocess(self, img_bgr: np.ndarray, lb) -> torch.Tensor:
        r, nh, nw, top, left = lb
        S = self.image_size
        canvas = np.full((S, S, 3), PAD_VALUE, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(self.device)

    def _get_layer_by_name(self, name: str) -> Optional[torch.nn.Module]:
        try:
            m = self._model
            for part in name.split("."):
                m = m[int(part)] if part.isdigit() else getattr(m, part)
            return m
        except (AttributeError, IndexError, KeyError, TypeError):
            return None

    def _save(self, res, img, heat_color, overlay, boxed, out_dir: Path, stem: str, idx: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = f"{stem}_det{idx}"
        cv2.imwrite(str(out_dir / f"{p}_original.jpg"), img)
        cv2.imwrite(str(out_dir / f"{p}_heatmap.jpg"), heat_color)
        cv2.imwrite(str(out_dir / f"{p}_overlay.jpg"), overlay)
        cv2.imwrite(str(out_dir / f"{p}_boxed_overlay.jpg"), boxed)
        cv2.imwrite(str(out_dir / f"{p}_panel.jpg"), np.hstack([img, boxed]))
        save_json(res.to_metadata_dict(), out_dir / f"{p}_metadata.json")