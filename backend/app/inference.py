"""ONNX Runtime wrapper class for SkinWise EfficientNet-B0 acne severity classifier."""

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = [
    "Mild (Level 0)",
    "Moderate (Level 1)",
    "Severe (Level 2)",
    "Very Severe (Level 3)",
]

INPUT_SIZE: tuple[int, int] = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AcneClassifier:
    """ONNX Runtime inference wrapper for the EfficientNet-B0 ACNE04 classifier.

    Usage:
        classifier = AcneClassifier("backend/app/best_model.onnx")
        result = classifier.predict(pil_image)
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            logger.warning("ONNX model not found at '%s'.", self.model_path)
            return
        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info("AcneClassifier loaded from '%s'.", self.model_path)
        except Exception as exc:
            logger.error("Failed to load ONNX model: %s", exc)
            self._session = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize to 224×224, normalize with ImageNet stats, return NCHW float32."""
        resized = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
        normalized = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(
            np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0),
            dtype=np.float32,
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        flat = logits.reshape(-1).astype(np.float64)
        exp = np.exp(flat - np.max(flat))
        return (exp / exp.sum()).astype(np.float64)

    def predict(self, image: Image.Image) -> Dict:
        """Run inference on a PIL Image and return structured prediction dict.

        Returns:
            dict with keys: predicted_class, predicted_class_index, confidence,
            latency_ms, probabilities.

        Raises:
            RuntimeError: if the model session is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"ONNX model is not loaded. Expected file at '{self.model_path}'."
            )

        tensor = self.preprocess(image)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: tensor})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = self._softmax(np.asarray(outputs[0]))
        pred_idx = int(np.argmax(probs))

        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "predicted_class_index": pred_idx,
            "confidence": round(float(probs[pred_idx]), 6),
            "latency_ms": round(latency_ms, 3),
            "probabilities": {
                name: round(float(p), 6)
                for name, p in zip(CLASS_NAMES, probs)
            },
        }


# ---------------------------------------------------------------------------
# Grad-CAM without PyTorch
# ---------------------------------------------------------------------------
# EfficientNet-B0 ends with: last conv block -> GlobalAvgPool -> Linear(1280 -> 4).
# For this head, the Grad-CAM gradient of class score y_c w.r.t. every position of
# feature map A_k is exactly W[c, k] / (H * W). So Grad-CAM only needs:
#   * the last conv feature maps   -> second output "features" of the ONNX model
#   * the classifier weights W     -> read from the ONNX file itself
# This gives the same heatmap as the autograd version in src/utils/gradcam.py,
# using NumPy only, so it runs inside the lightweight Docker image.

def load_classifier_weights(model_path: str | Path) -> Optional[np.ndarray]:
    """Read the final Linear layer weights (shape `(num_classes, channels)`) from an ONNX file.

    Returns `None` if the graph has no Gemm/MatMul layer producing the logits.
    """
    import onnx  # imported lazily: only needed once, when the weights are first loaded
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    logits_name = graph.output[0].name

    for node in graph.node:
        if logits_name not in node.output or node.op_type not in {"Gemm", "MatMul"}:
            continue
        weight_name = node.input[1]
        if weight_name not in initializers:
            return None
        weights = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
        trans_b = next((attr.i for attr in node.attribute if attr.name == "transB"), 0)
        # Gemm(transB=1) stores W as (classes, channels); MatMul/Gemm(transB=0) as (channels, classes)
        return weights if (node.op_type == "Gemm" and trans_b == 1) else weights.T
    return None


def compute_gradcam(
    features: np.ndarray,
    class_weights: np.ndarray,
    output_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap in `[0, 1]` from feature maps and one class's weights.

    Args:
        features: Last conv activations for ONE image, shape `(channels, h, w)`.
        class_weights: Classifier weights of the explained class, shape `(channels,)`.
        output_size: `(width, height)` of the returned heatmap.

    Returns:
        `float32` array of shape `(height, width)`.
    """
    channels, h, w = features.shape
    if class_weights.shape != (channels,):
        raise ValueError(f"Expected {channels} class weights, got shape {class_weights.shape}.")

    alpha = class_weights / float(h * w)                        # gradients, averaged over space
    cam = np.maximum(np.tensordot(alpha, features, axes=1), 0)  # ReLU(sum_k alpha_k * A_k)

    cam_img = Image.fromarray(cam.astype(np.float32), mode="F").resize(output_size, Image.Resampling.BILINEAR)
    cam = np.asarray(cam_img, dtype=np.float32)
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cam_min) / (cam_max - cam_min)


def apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a heatmap in `[0, 1]` to RGB `uint8` colours (blue = low, red = high)."""
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def render_gradcam_png(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    size: tuple[int, int] = INPUT_SIZE,
) -> bytes:
    """Blend the coloured heatmap over the image and encode the result as PNG bytes."""
    base = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    if heatmap.shape != (size[1], size[0]):
        heatmap = np.asarray(
            Image.fromarray(heatmap.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    colored = apply_jet_colormap(heatmap).astype(np.float32)
    blended = np.clip((1.0 - alpha) * base + alpha * colored, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(blended, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
