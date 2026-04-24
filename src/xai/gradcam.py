"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for U-Net.

Hooks into the bottleneck layer to produce a heatmap showing which
spatial regions most influenced the segmentation / fertility score.

Class mapping for the 3-class model:
  0 = background
  1 = vascularization  (fertile indicator)
  2 = embryo core      (fertile indicator)

Usage:
    explainer = GradCAMExplainer(model)
    heatmap, overlay, confidence = explainer.explain(image_tensor, score_mode="fertile")
    explainer.save("output.png", original_image, heatmap, overlay, label="fertile", confidence=confidence)
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Literal, Optional, Tuple


ScoreMode = Literal["fertile", "infertile", "mean"]


class GradCAM:
    """Grad-CAM hook for a single convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        score_mode: ScoreMode = "fertile",
    ) -> Tuple[np.ndarray, float]:
        """Return a ((H, W) heatmap normalised to [0, 1], confidence) tuple.

        Args:
            input_tensor: (1, C, H, W) float tensor on the model's device.
            score_mode:
                "fertile"   – score = mean prob of egg-region classes (1+2 for multiclass,
                              or mean logit for binary)
                "infertile" – score = mean prob of background class (0)
                "mean"      – class-agnostic spatial mean logit
        """
        self.model.eval()
        self.model.zero_grad()
        self._gradients = None
        self._activations = None

        with torch.enable_grad():
            output = self.model(input_tensor)   # (1, n_classes, H, W)
            n_classes = output.shape[1]

            if n_classes > 1:
                # Multiclass segmentation (softmax over class dim)
                prob = F.softmax(output, dim=1)   # (1, n_classes, H, W)
                if score_mode == "fertile":
                    # Class 1 = vascularization — the UNIQUE fertility indicator
                    # (class 2 = egg body, present in both fertile AND infertile)
                    score = prob[:, 1, :, :].mean()
                    confidence = float(score.detach().cpu().item())
                elif score_mode == "infertile":
                    # Inverse of vascularization: highlights where fertility is ABSENT
                    score = (1.0 - prob[:, 1, :, :]).mean()
                    confidence = float(1.0 - prob[:, 1, :, :].mean().detach().cpu().item())
                else:
                    # class-agnostic: highlight any foreground structure
                    score = output[:, 1:, :, :].mean()
                    confidence = float(prob[:, 1:, :, :].sum(dim=1).mean().detach().cpu().item())
            else:
                # Binary segmentation (sigmoid)
                prob = torch.sigmoid(output)
                if score_mode == "fertile":
                    score = output.mean()   # logits — more stable gradients than prob
                    confidence = float(prob.mean().detach().cpu().item())
                elif score_mode == "infertile":
                    score = (-output).mean()
                    confidence = float(1.0 - prob.mean().detach().cpu().item())
                else:
                    score = output.abs().mean()
                    confidence = float(prob.mean().detach().cpu().item())

            score.backward()

        if self._gradients is None or self._activations is None:
            return np.zeros(input_tensor.shape[2:], dtype=np.float32), confidence

        # Grad-CAM: weight each channel by its global-average gradient
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        vmin, vmax = cam.min(), cam.max()
        if vmax - vmin > 1e-8:
            cam = (cam - vmin) / (vmax - vmin)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), confidence

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


class GradCAMExplainer:
    """High-level wrapper: load a U-Net, run Grad-CAM, produce overlays.

    Example
    -------
    >>> from src.xai.gradcam import GradCAMExplainer
    >>> explainer = GradCAMExplainer(model)
    >>> heatmap, overlay, confidence = explainer.explain(img_tensor)
    >>> explainer.save("egg_gradcam.png", original_bgr, heatmap, overlay,
    ...                label="fertile", confidence=confidence)
    """

    COLORMAP = cv2.COLORMAP_JET

    def __init__(
        self,
        model: torch.nn.Module,
        layer: str = "bottleneck",
    ) -> None:
        self.model = model
        target = self._resolve_layer(model, layer)
        self._cam = GradCAM(model, target)
        self._layer_name = layer

    @staticmethod
    def _resolve_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
        if hasattr(model, name):
            return getattr(model, name)
        # Graceful fallback for lightweight vs full model
        fallbacks = ["bottleneck", "down3", "down2", "down1", "inc"]
        for fb in fallbacks:
            if hasattr(model, fb):
                return getattr(model, fb)
        raise ValueError(
            f"Layer '{name}' not found on model. "
            f"Available: {list(model._modules.keys())}"
        )

    def explain(
        self,
        image_tensor: torch.Tensor,
        score_mode: ScoreMode = "fertile",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run Grad-CAM.

        Returns:
            heatmap    : (H, W) float32 array in [0, 1]
            overlay    : (H, W, 3) uint8 BGR heatmap-only coloured image
            confidence : float, model's fertility probability for this image
        """
        heatmap, confidence = self._cam.generate(image_tensor, score_mode)
        overlay = self._make_overlay(heatmap)
        return heatmap, overlay, confidence

    def explain_image(
        self,
        bgr_image: np.ndarray,
        score_mode: ScoreMode = "fertile",
        image_size: int = 256,
        device: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Convenience: accept a raw BGR OpenCV image.

        Returns:
            heatmap    : (H, W) float32 [0, 1]
            overlay    : (H, W, 3) uint8 BGR heatmap blended on original
            resized    : (H, W, 3) uint8 BGR resized original
            confidence : float, model's fertility probability for this image
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        resized = cv2.resize(bgr_image, (image_size, image_size))

        # Feed original-normalized image to model (preserves training distribution)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        heatmap, _, confidence = self.explain(tensor, score_mode)

        # CLAHE only for display — so dark candling images are visible to humans
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        display_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        overlay = self._blend(display_bgr, heatmap)
        return heatmap, overlay, display_bgr, confidence

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _make_overlay(self, heatmap: np.ndarray) -> np.ndarray:
        """Convert float heatmap to colour BGR image."""
        uint8 = (heatmap * 255).astype(np.uint8)
        return cv2.applyColorMap(uint8, self.COLORMAP)

    @staticmethod
    def _blend(
        bgr_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend heatmap onto original image."""
        uint8 = (heatmap * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(uint8, cv2.COLORMAP_JET)
        coloured = cv2.resize(coloured, (bgr_image.shape[1], bgr_image.shape[0]))
        return cv2.addWeighted(bgr_image, 1 - alpha, coloured, alpha, 0)

    def save(
        self,
        output_path: str,
        original_bgr: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        label: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Save a side-by-side panel: Original | Heatmap | Overlay."""
        H, W = original_bgr.shape[:2]

        orig_panel = cv2.resize(original_bgr, (W, H))
        heat_panel = cv2.resize(self._make_overlay(heatmap), (W, H))
        over_panel = cv2.resize(overlay, (W, H))

        def _put_text(img, text):
            out = img.copy()
            cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            return out

        orig_panel = _put_text(orig_panel, "Original")
        heat_panel = _put_text(heat_panel, f"Grad-CAM [{self._layer_name}]")
        caption = f"{label} ({confidence:.1%})" if label else f"conf={confidence:.1%}"
        over_panel = _put_text(over_panel, caption)

        panel = np.concatenate([orig_panel, heat_panel, over_panel], axis=1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, panel)

    def cleanup(self) -> None:
        self._cam.remove_hooks()
