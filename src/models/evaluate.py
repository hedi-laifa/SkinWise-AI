"""Standalone evaluation script: runs ONNX model on the ACNE04 test split and exports metrics.

The test split is generated with the same stratified split as training (seed=42, 70/15/15)
so there is zero overlap with training or validation data.
"""

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


def load_test_split(data_dir: Path, seed: int = 42) -> Tuple[List[Path], List[int]]:
    """Return the test split using the SAME stratified split as training.

    Uses src.data.dataset.stratified_split_samples with seed=42, 70/15/15 split,
    ensuring zero overlap between test and training/validation data.
    """
    try:
        from src.data.dataset import ACNE04Dataset, stratified_split_samples

        base = ACNE04Dataset(root_dir=data_dir, transform=None)
        _, _, test_samples = stratified_split_samples(
            samples=base.samples,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=seed,
        )
        image_paths = [p for p, _ in test_samples]
        labels = [lbl for _, lbl in test_samples]
        logger.info(
            "Test split (stratified, seed=%d): %d images across %d classes.",
            seed,
            len(labels),
            len(set(labels)),
        )
        return image_paths, labels

    except ImportError as exc:
        # No fallback on purpose: any other split would overlap with the training
        # images and silently inflate the test metrics (data leakage).
        raise ImportError(
            "Could not import src.data.dataset. Run this script from the project root "
            "(python -m src.models.evaluate ...) so the exact training split (seed=42) is used."
        ) from exc


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

    # Per-class breakdown for diagnosis
    per_class = {}
    for c in range(num_classes):
        support = sum(1 for t in y_true if t == c)
        predicted_as = sum(1 for p in y_pred if p == c)
        per_class[CLASS_NAMES[c]] = {
            "f1": round(f1_per_class[c], 4),
            "support": support,
            "predicted_as": predicted_as,
        }

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "f1_per_class": {CLASS_NAMES[i]: round(f1_per_class[i], 4) for i in range(num_classes)},
        "per_class_detail": per_class,
        "confusion_matrix": confusion,
        "n_samples": len(y_true),
    }


def evaluate(onnx_path: str, data_dir: str, output: str = "eval_metrics.json", seed: int = 42) -> dict:
    import onnxruntime as ort

    logger.info("Loading ONNX model from '%s'.", onnx_path)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image_paths, y_true = load_test_split(Path(data_dir), seed=seed)
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
    metrics["split"] = "stratified_seed42_70-15-15"

    Path(output).write_text(json.dumps(metrics, indent=2))
    logger.info("Evaluation complete. Results saved to '%s'.", output)
    logger.info("  Accuracy  : %.2f%%", metrics["accuracy"] * 100)
    logger.info("  Macro-F1  : %.4f", metrics["macro_f1"])
    logger.info("  Avg Latency: %.2f ms", metrics["avg_latency_ms"])

    # Diagnosis: warn if model is biased toward one class
    for name, detail in metrics["per_class_detail"].items():
        ratio = detail["predicted_as"] / max(1, detail["support"])
        if ratio > 2.0:
            logger.warning(
                "BIAS WARNING: Model predicts '%s' %.1fx more often than it appears in test set. "
                "Consider adding class weights to CrossEntropyLoss.",
                name, ratio,
            )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SkinWise ONNX model on ACNE04 test split.")
    parser.add_argument("--onnx", default="backend/app/best_model.onnx", help="Path to ONNX model")
    parser.add_argument("--data-dir", default="data/raw/acne_1024", help="Path to ACNE04 dataset root")
    parser.add_argument("--output", default="eval_metrics.json", help="Output JSON file for metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split (must match training)")
    args = parser.parse_args()
    evaluate(onnx_path=args.onnx, data_dir=args.data_dir, output=args.output, seed=args.seed)
