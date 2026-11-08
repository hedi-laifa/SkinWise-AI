"""Mathematical Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet-B0.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.transforms import get_val_test_transforms
from src.models.classifier import build_efficientnet_b0, load_trained_model


class GradCAM:
    """True mathematical Grad-CAM hooking into the final convolutional layer of EfficientNet-B0.

    Given an input tensor $x$ and target class $c$ with pre-softmax score $y^c$:
      1. Forward hook captures final convolutional feature maps $A^k \\in \\mathbb{R}^{H \\times W}$.
      2. Backward hook captures gradients $\\frac{\\partial y^c}{\\partial A^k_{i,j}}$.
      3. Neuron importance weights are computed via global average pooling over spatial dims:
         $\\alpha_k^c = \\frac{1}{H \\cdot W} \\sum_{i=1}^{H} \\sum_{j=1}^{W} `
         `\\frac{\\partial y^c}{\\partial A^k_{i,j}}$
      4. Spatial localization map is computed via weighted combination and ReLU:
         $L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$, normalized to $[0, 1]$.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        """Initialize Grad-CAM and register forward/backward hooks on `target_layer`.

        Args:
            model: PyTorch `EfficientNet-B0` model (`nn.Module`).
            target_layer: Convolutional block to hook. If `None`, defaults to
                `model.features[-1]` (the final convolutional stage of EfficientNet-B0).
        """
        self.model: nn.Module = model
        self.model.eval()
        self._disable_inplace_ops(self.model)

        if target_layer is None:
            if not hasattr(self.model, "features"):
                raise ValueError(
                    "Model does not have a `.features` attribute; provide `target_layer` explicitly."
                )
            target_layer = self.model.features[-1]  # type: ignore[index]

        self.target_layer: nn.Module = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        self._backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    @staticmethod
    def _disable_inplace_ops(model: nn.Module) -> None:
        """Disable `inplace=True` on SiLU/ReLU/Dropout layers so backward hooks never conflict."""
        for module in model.modules():
            if hasattr(module, "inplace") and isinstance(
                getattr(module, "inplace"), bool
            ):
                setattr(module, "inplace", False)

    def _save_activation(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Forward hook storing the activation tensor $A^k$."""
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> None:
        """Backward hook storing the gradient tensor $\\frac{\\partial y^c}{\\partial A^k}$."""
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove registered forward and backward hooks from `target_layer`."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def compute_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Compute the 2D Grad-CAM spatial heatmap in $[0, 1]$ from model weights.

        Args:
            input_tensor: Preprocessed input image tensor of shape `(1, 3, H, W)`.
            target_class: Optional class index $c \\in \\{0, 1, 2, 3\\}$ to explain.
                If `None`, uses the highest-scoring predicted class $\\arg\\max_c y^c$.

        Returns:
            Tuple of `(heatmap_2d, target_class_idx, target_class_probability)` where
            `heatmap_2d` is a `float32` NumPy array of shape `(H_in, W_in)` in `[0.0, 1.0]`.
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                f"Expected input_tensor of shape (1, 3, H, W), got {tuple(input_tensor.shape)}."
            )

        device = next(self.model.parameters()).device
        x = input_tensor.to(device).requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        # 1. Forward pass
        logits: torch.Tensor = self.model(x)  # Shape: (1, num_classes)
        probs: torch.Tensor = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        target_prob = float(probs[0, target_class].item())

        # 2. Backward pass wrt pre-softmax class score y^c
        score: torch.Tensor = logits[0, target_class]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks failed to capture forward activations or backward gradients."
            )

        # 3. Compute channel-wise weights \alpha_k^c via global average pooling over (H_feat, W_feat)
        # gradients shape: (1, C, H_feat, W_feat) -> weights shape: (1, C, 1, 1)
        weights: torch.Tensor = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 4. Weighted combination of forward feature maps followed by ReLU
        # activations shape: (1, C, H_feat, W_feat) -> cam shape: (1, 1, H_feat, W_feat)
        cam: torch.Tensor = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample spatial CAM to match input tensor dimensions (H_in, W_in)
        target_h, target_w = x.shape[2], x.shape[3]
        cam_upsampled = F.interpolate(
            cam,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        cam_np: np.ndarray = (
            cam_upsampled.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        )

        # Min-max normalize into [0.0, 1.0]
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        return cam_np, target_class, target_prob


def apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a normalized 2D heatmap in `[0, 1]` to a 3-channel RGB Jet colormap (`uint8`).

    Args:
        heatmap: 2D float array of shape `(H, W)` with values in `[0.0, 1.0]`.

    Returns:
        RGB `uint8` array of shape `(H, W, 3)`.
    """
    clipped = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
    r = np.clip(1.5 - np.abs(4.0 * clipped - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * clipped - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * clipped - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def overlay_heatmap_on_image(
    original_image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    output_size: Optional[Tuple[int, int]] = (224, 224),
) -> Image.Image:
    """Overlay a mathematical Grad-CAM heatmap onto the original RGB image.

    Args:
        original_image: Original input image as a PIL `Image.Image` or `uint8` RGB array.
        heatmap: 2D normalized Grad-CAM array in `[0.0, 1.0]`.
        alpha: Opacity weight of the Grad-CAM heatmap in `[0.0, 1.0]`.
        output_size: Optional `(width, height)` to resize the output overlay. If `None`,
            preserves the original image dimensions.

    Returns:
        Blended RGB `PIL.Image.Image`.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")

    if isinstance(original_image, np.ndarray):
        pil_orig = Image.fromarray(original_image.astype(np.uint8)).convert("RGB")
    else:
        pil_orig = original_image.convert("RGB")

    if output_size is not None:
        pil_orig = pil_orig.resize(output_size, Image.Resampling.BILINEAR)

    target_w, target_h = pil_orig.size

    # Resize heatmap to match original image dimensions if needed
    if heatmap.shape != (target_h, target_w):
        heatmap_img = Image.fromarray(
            np.clip(heatmap * 255.0, 0, 255).astype(np.uint8), mode="L"
        ).resize((target_w, target_h), Image.Resampling.BILINEAR)
        heatmap_resized = np.asarray(heatmap_img, dtype=np.float32) / 255.0
    else:
        heatmap_resized = heatmap

    colormap_rgb = apply_jet_colormap(heatmap_resized).astype(np.float32)
    base_rgb = np.asarray(pil_orig, dtype=np.float32)

    blended = (1.0 - alpha) * base_rgb + alpha * colormap_rgb
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended_uint8, mode="RGB")


def generate_gradcam_png_bytes(
    image: Image.Image,
    model: nn.Module,
    target_class: Optional[int] = None,
    alpha: float = 0.45,
) -> bytes:
    """Compute mathematical Grad-CAM for a PIL Image on `model` and return PNG bytes."""
    rgb_image = image.convert("RGB")
    input_tensor = get_val_test_transforms()(rgb_image).unsqueeze(0)

    gradcam = GradCAM(model=model)
    try:
        heatmap, _, _ = gradcam.compute_heatmap(
            input_tensor=input_tensor,
            target_class=target_class,
        )
    finally:
        gradcam.remove_hooks()

    blended_pil = overlay_heatmap_on_image(
        original_image=rgb_image,
        heatmap=heatmap,
        alpha=alpha,
        output_size=(224, 224),
    )

    buffer = io.BytesIO()
    blended_pil.save(buffer, format="PNG")
    return buffer.getvalue()


def load_gradcam_model(
    checkpoint_path: Union[str, Path] = Path("best_model.pth"),
    num_classes: int = 4,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Load the PyTorch EfficientNet-B0 model for mathematical Grad-CAM extraction."""
    ckpt = Path(checkpoint_path)
    if ckpt.is_file():
        return load_trained_model(str(ckpt), num_classes=num_classes, device=device)

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    model.to(device)
    model.eval()
    return model
