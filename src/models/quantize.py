"""Post-training weight optimization of the ONNX model: FP16 and INT8 (no retraining).

* FP16 : weights stored on 16 bits instead of 32 -> file about 2x smaller.
* INT8 : static quantization (QDQ format) of the Conv and Gemm layers. Activation ranges are
         calibrated on TRAINING images only (Percentile 99.99 %, robust to extreme values).
         EfficientNet is sensitive to 8-bit rounding: quantizing every layer drops validation
         accuracy from 80 % to 45 %. Keeping the 16 depthwise convolutions, the first conv and the
         final classifier in float32 brings it back to 79 % (setting chosen on the VALIDATION split).

Each version is evaluated on the same test split with `src.models.evaluate`, so the size /
latency / accuracy / Macro-F1 comparison is fair.

Usage:
    python -m src.models.quantize --onnx models/best_model.onnx --data-dir data/raw/acne_1024
"""

import argparse
import csv
import json
import logging
import random
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.transformers.float16 import convert_float_to_float16

from src.models.evaluate import evaluate, preprocess

logger = logging.getLogger(__name__)


class TrainImagesCalibrationReader(CalibrationDataReader):
    """Feeds preprocessed TRAINING images to ONNX Runtime to measure activation ranges."""

    def __init__(self, image_paths: List[Path], input_name: str) -> None:
        self.input_name = input_name
        self._iterator: Iterator[Path] = iter(image_paths)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        path = next(self._iterator, None)
        return None if path is None else {self.input_name: preprocess(path)}


def calibration_images(data_dir: Path, train_csv: Path, n_images: int, seed: int) -> List[Path]:
    """Pick `n_images` random images from the training split (never from val/test)."""
    with train_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    random.Random(seed).shuffle(rows)
    return [data_dir / row["image_path"] for row in rows[:n_images]]


def export_fp16(fp32_path: Path, fp16_path: Path) -> Path:
    """Convert weights to float16; inputs/outputs stay float32 so the API code is unchanged."""
    model_fp16 = convert_float_to_float16(onnx.load(str(fp32_path)), keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))
    return fp16_path


def layers_kept_in_float(model: onnx.ModelProto) -> List[str]:
    """Layers that lose too much accuracy in 8 bits: first conv, depthwise convs, final classifier."""
    convs = [node for node in model.graph.node if node.op_type == "Conv"]
    depthwise = [
        node.name for node in convs
        if any(attr.name == "group" and attr.i > 1 for attr in node.attribute)
    ]
    classifier = [node.name for node in model.graph.node if node.op_type in {"Gemm", "MatMul"}]
    return [convs[0].name] + depthwise + classifier


def export_int8(fp32_path: Path, int8_path: Path, calib_paths: List[Path]) -> Path:
    """Static INT8 quantization (per-channel weights, QDQ format) calibrated on training images."""
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(str(fp32_path), str(prepared))  # shape inference + graph optimisation
        prepared_model = onnx.load(str(prepared))
        input_name = prepared_model.graph.input[0].name
        excluded = layers_kept_in_float(prepared_model)
        logger.info("Keeping %d layers in float32 (first conv, depthwise convs, classifier).", len(excluded))
        quantize_static(
            model_input=str(prepared),
            model_output=str(int8_path),
            calibration_data_reader=TrainImagesCalibrationReader(calib_paths, input_name),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm"],
            nodes_to_exclude=excluded,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={"CalibPercentile": 99.99},
        )
    return int8_path


def run(fp32_path: Path, data_dir: Path, train_csv: Path, output_json: Path,
        n_calibration: int = 48, seed: int = 42) -> Dict[str, dict]:
    out_dir = fp32_path.parent
    fp16_path = export_fp16(fp32_path, out_dir / f"{fp32_path.stem}_fp16.onnx")
    logger.info("FP16 model written to '%s'.", fp16_path)

    calib = calibration_images(data_dir, train_csv, n_calibration, seed)
    int8_path = export_int8(fp32_path, out_dir / f"{fp32_path.stem}_int8.onnx", calib)
    logger.info("INT8 model written to '%s' (%d calibration images).", int8_path, len(calib))

    results: Dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, path in [("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)]:
            metrics = evaluate(str(path), str(data_dir), output=str(Path(tmp) / f"{name}.json"))
            results[name] = {
                "file": path.as_posix(),
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "avg_latency_ms": metrics["avg_latency_ms"],
            }

    base = results["fp32"]
    for row in results.values():
        row["size_vs_fp32"] = round(row["size_mb"] / base["size_mb"], 3)
        row["speedup_vs_fp32"] = round(base["avg_latency_ms"] / row["avg_latency_ms"], 2)
    output_json.write_text(json.dumps(results, indent=2))

    logger.info("%-6s %9s %9s %9s %12s", "model", "size MB", "accuracy", "macro_f1", "latency ms")
    for name, row in results.items():
        logger.info("%-6s %9.2f %9.4f %9.4f %12.2f", name, row["size_mb"], row["accuracy"],
                    row["macro_f1"], row["avg_latency_ms"])
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", force=True)
    parser = argparse.ArgumentParser(description="Create and compare FP16 / INT8 versions of the ONNX model.")
    parser.add_argument("--onnx", type=Path, default=Path("models/best_model.onnx"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/acne_1024"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("quantization_metrics.json"))
    # Percentile calibration keeps a histogram of every activation: 48 images fit in ~3 GB of RAM
    parser.add_argument("--n-calibration", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.onnx, args.data_dir, args.train_csv, args.output, args.n_calibration, args.seed)


if __name__ == "__main__":
    main()
