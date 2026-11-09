"""Standalone evaluation script: runs ONNX model on the ACNE04 test split and exports metrics."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES: List[str] = [
    "Mild (Level 0)",
    "Moderate (Level 1)",
    "Severe (Level 2)",
    "Very Severe (Level 3)",
]

DIRECTORY_TO_CLASS: dict = {
    "acne0_1024": 0, "acne1_1024": 1, "acne2_1024": 2, "acne3_1024": 3,
    "0": 0, "1": 1, "2": 2, "3": 3,
    "mild": 0, "moderate": 1, "severe": 2, "very_severe": 3,
}


def load_test_images(data_dir: Path, test_ratio: float = 0.15) -> Tuple[List, List]:
    """Load the last `test_ratio` fraction of images per class as a held-out test set."""
    images, labels = [], []
    for folder_name, class_idx in DIRECTORY_TO_CLASS.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            continue
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        n_test = max(1, int(len(files) * test_ratio))
        for path in files[-n_test:]:
            images.append(path)
            labels.append(class_idx)
    logger.info("Test set: %d images across %d classes.", len(images), len(set(labels)))
    return images, labels


def preprocess(image_path: Path) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = Image.open(image_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    normalized = (arr - mean) / std
    return np.ascontiguousarray(np.expand_dims(np.transpose(normalized, (2, 0, 1)), 0), dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    flat = logits.reshape(-1).astype(np.float64)
    exp = np.exp(flat - np.max(flat))
    return exp / exp.sum()


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int = 4) -> dict:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true) if y_true else 0.0

    f1_per_class = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)

    macro_f1 = float(np.mean(f1_per_class))

    confusion = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "f1_per_class": {CLASS_NAMES[i]: round(f1_per_class[i], 4) for i in range(num_classes)},
        "confusion_matrix": confusion,
        "n_samples": len(y_true),
    }


def evaluate(onnx_path: str, data_dir: str, output: str = "eval_metrics.json") -> dict:
    import onnxruntime as ort

    logger.info("Loading ONNX model from '%s'.", onnx_path)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image_paths, y_true = load_test_images(Path(data_dir))
    if not image_paths:
        raise ValueError(f"No test images found in '{data_dir}'. Check dataset structure.")

    y_pred, latencies = [], []
    for i, path in enumerate(image_paths):
        tensor = preprocess(path)
        t0 = time.perf_counter()
        logits = session.run(None, {input_name: tensor})[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        probs = softmax(logits)
        y_pred.append(int(np.argmax(probs)))
        if (i + 1) % 20 == 0:
            logger.info("  Evaluated %d / %d images...", i + 1, len(image_paths))

    metrics = compute_metrics(y_true, y_pred)
    metrics["avg_latency_ms"] = round(float(np.mean(latencies)), 3)
    metrics["onnx_model"] = str(onnx_path)
    metrics["data_dir"] = str(data_dir)

    Path(output).write_text(json.dumps(metrics, indent=2))
    logger.info("Evaluation complete. Results saved to '%s'.", output)
    logger.info("  Accuracy  : %.2f%%", metrics["accuracy"] * 100)
    logger.info("  Macro-F1  : %.4f", metrics["macro_f1"])
    logger.info("  Avg Latency: %.2f ms", metrics["avg_latency_ms"])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SkinWise ONNX model on ACNE04 test split.")
    parser.add_argument("--onnx", default="backend/app/best_model.onnx", help="Path to ONNX model")
    parser.add_argument("--data-dir", default="acne_1024", help="Path to ACNE04 dataset root")
    parser.add_argument("--output", default="eval_metrics.json", help="Output JSON file for metrics")
    args = parser.parse_args()
    evaluate(onnx_path=args.onnx, data_dir=args.data_dir, output=args.output)
