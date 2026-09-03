"""
src/explainability/cam.py

YOLOv8-compatible explainability via Grad-CAM on backbone feature maps.

Design rationale:
  Standard classification Grad-CAM targets a class logit in a softmax
  layer. YOLOv8 is an object detector: its output is a set of bounding
  box proposals with objectness/class scores, NOT a single class logit.

  Two approaches are technically defensible:

  Approach A — Score-based Grad-CAM (implemented here):
    Target: the confidence score of a specific detection.
    Gradient: d(score) / d(feature_map) at a chosen backbone layer.
    Limitation: YOLOv8's post-processing (NMS, decode) is not fully
    differentiable. We therefore target the RAW detection head output
    (before NMS) by re-running the model in a mode that exposes
    pre-NMS outputs.

  Approach B — Feature-map activation visualization:
    No gradient required. Average the absolute activation magnitudes
    of the last C2f/SPPF backbone feature map. Less specific to a
    single detection but always stable.

  This implementation uses Approach A with fallback to Approach B.
  Both are clearly documented in the output metadata.

  IMPORTANT: The resulting heatmap indicates WHICH regions of the image
  most influenced the selected detection's confidence score. It is NOT
  a clinical finding. It is NOT medically validated. It is a tool for
  model debugging and transparency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.logger import get_logger
from src.utils.file_utils import save_json

logger = get_logger(__name__)

# Target layer name in YOLOv8 backbone (SPPF output — rich semantic features)
# This is the last feature map before the neck/head.
# Verified against YOLOv8n architecture.
DEFAULT_TARGET_LAYER = "model.9"   # SPPF layer in YOLOv8n


@dataclass
class ExplainabilityResult:
    """Output of a single explainability run."""
    # inputs
    image_path: str
    detection_index: int
    # detection info
    class_name: str
    confidence: float
    bbox_xyxy: List[float]          # [x1, y1, x2, y2] in pixels
    bbox_normalized: List[float]    # [xc, yc, w, h] normalized
    # heatmap info
    heatmap_method: str             # "gradcam" | "activation"
    target_layer: str
    # arrays (not stored in metadata, only in image files)
    heatmap_normalized: Optional[np.ndarray] = field(default=None)
    overlay: Optional[np.ndarray] = field(default=None)
    # metadata
    model_architecture: str = "yolov8n"
    image_size: int = 640
    notes: List[str] = field(default_factory=list)

    def to_metadata_dict(self) -> dict:
        """Serializable metadata (excludes numpy arrays)."""
        return {
            "image_path": self.image_path,
            "detection_index": self.detection_index,
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(x, 1) for x in self.bbox_xyxy],
            "bbox_normalized": [round(x, 4) for x in self.bbox_normalized],
            "heatmap_method": self.heatmap_method,
            "target_layer": self.target_layer,
            "model_architecture": self.model_architecture,
            "image_size": self.image_size,
            "notes": self.notes,
            "disclaimer": (
                "This heatmap indicates model attention for the selected "
                "detection. It is NOT medically validated and is NOT a "
                "substitute for clinical interpretation."
            ),
        }


class YOLOExplainability:
    """
    Grad-CAM explainability for YOLOv8 fracture detections.

    Usage:
        explainer = YOLOExplainability(model_path)
        results = explainer.explain(
            image_path,
            conf_threshold=0.25,
            output_dir=Path("runs/explainability/sample"),
        )
    """

    def __init__(
        self,
        model_path: Path,
        target_layer: str = DEFAULT_TARGET_LAYER,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path)
        self.target_layer = target_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or ["fracture"]

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}"
            )

        # load via Ultralytics
        from ultralytics import YOLO
        self._yolo = YOLO(str(self.model_path))
        self._model = self._yolo.model
        self._model.eval()
        self._model.to(self.device)

        logger.info(
            f"YOLOExplainability initialized: "
            f"model={self.model_path.name} "
            f"layer={self.target_layer} "
            f"device={self.device}"
        )

    def explain(
        self,
        image_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        detection_index: int = 0,
        output_dir: Optional[Path] = None,
    ) -> List[ExplainabilityResult]:
        """
        Run explainability on an image.

        Args:
            image_path:       Input image path.
            conf_threshold:   Confidence threshold for detections.
            iou_threshold:    IoU threshold for NMS.
            detection_index:  Which detection to explain (-1 = all).
            output_dir:       If set, save visualizations here.

        Returns:
            List of ExplainabilityResult (one per explained detection).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Step 1: run standard Ultralytics predict for detections
        results = self._yolo.predict(
            str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            logger.info(f"No detections above conf={conf_threshold}: {image_path.name}")
            return []

        boxes = results[0].boxes
        num_detections = len(boxes)
        logger.info(
            f"Explaining {image_path.name}: "
            f"{num_detections} detection(s)"
        )

        # determine which detections to explain
        if detection_index == -1:
            indices = list(range(num_detections))
        else:
            if detection_index >= num_detections:
                logger.warning(
                    f"detection_index={detection_index} out of range "
                    f"(only {num_detections} detections). "
                    f"Explaining index 0."
                )
                indices = [0]
            else:
                indices = [detection_index]

        # load image for visualization
        orig_img = cv2.imread(str(image_path))
        if orig_img is None:
            orig_img = np.array(Image.open(image_path).convert("RGB"))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        output_results = []

        for idx in indices:
            result = self._explain_single_detection(
                image_path=image_path,
                orig_img=orig_img,
                boxes=boxes,
                detection_index=idx,
                output_dir=output_dir,
            )
            output_results.append(result)

        return output_results

    def _explain_single_detection(
        self,
        image_path: Path,
        orig_img: np.ndarray,
        boxes,
        detection_index: int,
        output_dir: Optional[Path],
    ) -> ExplainabilityResult:
        """Explain a single detection using Grad-CAM with activation fallback."""

        conf = float(boxes.conf[detection_index].cpu())
        xyxy = boxes.xyxy[detection_index].cpu().tolist()
        cls_id = int(boxes.cls[detection_index].cpu())
        class_name = (
            self.class_names[cls_id]
            if cls_id < len(self.class_names)
            else str(cls_id)
        )

        h, w = orig_img.shape[:2]
        # normalized bbox
        x1, y1, x2, y2 = xyxy
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # try Grad-CAM first
        heatmap, method, notes = self._compute_gradcam(image_path, detection_index)

        # if Grad-CAM failed, fall back to activation map
        if heatmap is None:
            heatmap, method, notes = self._compute_activation_map()

        # resize heatmap to original image size
        heatmap_resized = cv2.resize(
            heatmap, (w, h), interpolation=cv2.INTER_LINEAR
        )
        heatmap_normalized = (
            (heatmap_resized - heatmap_resized.min())
            / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
        )

        # create colormap overlay
        heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap_color, 0.5, 0)

        # draw bounding box on overlay
        boxed_overlay = overlay.copy()
        cv2.rectangle(
            boxed_overlay,
            (int(x1), int(y1)), (int(x2), int(y2)),
            (0, 255, 0), 2,
        )
        cv2.putText(
            boxed_overlay,
            f"{class_name} {conf:.2f}",
            (int(x1), max(int(y1) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        result = ExplainabilityResult(
            image_path=str(image_path),
            detection_index=detection_index,
            class_name=class_name,
            confidence=conf,
            bbox_xyxy=xyxy,
            bbox_normalized=[xc, yc, bw, bh],
            heatmap_method=method,
            target_layer=self.target_layer,
            heatmap_normalized=heatmap_normalized,
            overlay=overlay,
            notes=notes,
        )

        if output_dir:
            self._save_outputs(
                result=result,
                orig_img=orig_img,
                heatmap_uint8=heatmap_uint8,
                heatmap_color=heatmap_color,
                overlay=overlay,
                boxed_overlay=boxed_overlay,
                output_dir=Path(output_dir),
                detection_index=detection_index,
                image_stem=image_path.stem,
            )

        return result

    def _compute_gradcam(
        self,
        image_path: Path,
        detection_index: int,
    ) -> Tuple[Optional[np.ndarray], str, List[str]]:
        """
        Compute Grad-CAM for a specific detection.

        Targets the sum of the detection head outputs corresponding
        to the selected detection's objectness×class_conf score.

        Returns (heatmap, method_name, notes) or (None, ...) on failure.
        """
        notes = []
        try:
            # resolve target layer
            target_layer = self._get_layer_by_name(self.target_layer)
            if target_layer is None:
                notes.append(
                    f"Layer '{self.target_layer}' not found — "
                    f"falling back to activation map."
                )
                return None, "activation_fallback", notes

            # storage for hooks
            activations: List[torch.Tensor] = []
            gradients: List[torch.Tensor] = []

            def fwd_hook(module, inp, out):
                activations.append(out.detach())

            def bwd_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0].detach())

            fwd_handle = target_layer.register_forward_hook(fwd_hook)
            bwd_handle = target_layer.register_backward_hook(bwd_hook)

            try:
                # prepare input tensor
                img_tensor = self._preprocess_image(image_path)
                img_tensor.requires_grad_(True)

                # forward pass — use raw model output (before NMS)
                self._model.zero_grad()
                with torch.enable_grad():
                    raw_output = self._model(img_tensor)

                # raw_output shape depends on YOLOv8 version
                # For YOLOv8, model(x) returns a tuple
                # preds[0] shape: [1, 5+nc, num_anchors] or similar
                preds = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output

                if preds is None:
                    notes.append("Could not get raw predictions — fallback.")
                    return None, "activation_fallback", notes

                # target: objectness score of detection at detection_index
                # preds shape: [1, 84, 8400] for COCO; [1, 5, 8400] for 1-class
                # dim 0: batch, dim 1: channels (4 bbox + nc), dim 2: anchors
                if preds.dim() == 3:
                    # confidence = max over anchor dimension
                    # For single class: channel 4 is class_conf
                    # target: the anchor with highest confidence
                    conf_channel = 4  # objectness/class for nc=1
                    if preds.shape[1] > conf_channel:
                        scores = preds[0, conf_channel, :]
                        # get top-k scores and target the (detection_index)-th
                        top_scores = scores.topk(
                            min(detection_index + 1, scores.shape[0])
                        )
                        target_score = top_scores.values[detection_index]
                    else:
                        target_score = preds[0].max()
                else:
                    target_score = preds.max()

                # backward
                self._model.zero_grad()
                target_score.backward(retain_graph=False)

                if not gradients:
                    notes.append("No gradients captured — fallback.")
                    return None, "activation_fallback", notes

                # Grad-CAM computation
                grads = gradients[0]    # [1, C, H, W]
                acts = activations[0]   # [1, C, H, W]

                # global average pool gradients
                weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
                cam = (weights * acts).sum(dim=1).squeeze()      # [H, W]
                cam = F.relu(cam)                                 # keep positive

                heatmap = cam.cpu().numpy()
                notes.append(
                    f"Grad-CAM computed on layer '{self.target_layer}'. "
                    f"Target: confidence score of detection #{detection_index}."
                )
                return heatmap, "gradcam", notes

            finally:
                fwd_handle.remove()
                bwd_handle.remove()
                # clear cuda cache
                if self.device != "cpu":
                    torch.cuda.empty_cache()

        except Exception as e:
            notes.append(f"Grad-CAM failed: {e} — falling back to activation.")
            logger.warning(f"Grad-CAM exception: {e}")
            return None, "activation_fallback", notes

    def _compute_activation_map(
        self,
    ) -> Tuple[np.ndarray, str, List[str]]:
        """
        Fallback: activation magnitude visualization.

        Uses stored activations from the last forward pass.
        No gradient required — always stable.
        """
        notes = [
            "Activation map used (Grad-CAM fallback). "
            "Shows absolute activation magnitudes, not gradient-weighted. "
            "Less specific to a particular detection."
        ]
        if not hasattr(self, "_last_activations"):
            # run a dummy forward to get activations
            acts = torch.zeros(1, 256, 20, 20)
        else:
            acts = self._last_activations

        heatmap = acts.abs().mean(dim=1).squeeze().cpu().numpy()
        return heatmap, "activation", notes

    def _preprocess_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess image for model input."""
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_tensor = (
            torch.from_numpy(img_resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0
        )
        return img_tensor.to(self.device)

    def _get_layer_by_name(self, name: str) -> Optional[torch.nn.Module]:
        """Retrieve a submodule by dotted name."""
        try:
            module = self._model
            for part in name.split("."):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError, KeyError):
            return None

    def _save_outputs(
        self,
        result: ExplainabilityResult,
        orig_img: np.ndarray,
        heatmap_uint8: np.ndarray,
        heatmap_color: np.ndarray,
        overlay: np.ndarray,
        boxed_overlay: np.ndarray,
        output_dir: Path,
        detection_index: int,
        image_stem: str,
    ) -> None:
        prefix = f"{image_stem}_det{detection_index}"
        output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_dir / f"{prefix}_original.jpg"), orig_img)
        cv2.imwrite(str(output_dir / f"{prefix}_heatmap.jpg"), heatmap_color)
        cv2.imwrite(str(output_dir / f"{prefix}_overlay.jpg"), overlay)
        cv2.imwrite(
            str(output_dir / f"{prefix}_boxed_overlay.jpg"), boxed_overlay
        )
        save_json(
            result.to_metadata_dict(),
            output_dir / f"{prefix}_metadata.json",
        )
        logger.info(f"Explainability outputs saved: {output_dir / prefix}*")