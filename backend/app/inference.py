"""ONNX Runtime wrapper class for SkinWise EfficientNet-B0 acne severity classifier."""

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
