"""Unit tests for the inference helpers in backend/app/inference.py (no model file needed)."""

import io
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from backend.app.inference import (
    INPUT_SIZE,
    AcneClassifier,
    apply_jet_colormap,
    compute_gradcam,
    load_classifier_weights,
    render_gradcam_png,
)


def _tiny_onnx_classifier(path: Path, weights: np.ndarray) -> None:
    """Write a minimal ONNX graph: features (1,C) -> Gemm(transB=1) -> logits (1,K)."""
    n_classes, channels = weights.shape
    graph = helper.make_graph(
        nodes=[helper.make_node("Gemm", ["x", "W", "b"], ["logits"], transB=1)],
        name="tiny",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, channels])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_classes])],
        initializer=[
            numpy_helper.from_array(weights.astype(np.float32), "W"),
            numpy_helper.from_array(np.zeros(n_classes, np.float32), "b"),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path))


def test_preprocess_returns_normalized_nchw_tensor() -> None:
    """Preprocessing must give a float32 (1, 3, 224, 224) tensor, as the model expects."""
    classifier = AcneClassifier("does_not_exist.onnx")  # preprocessing works without a model
    tensor = classifier.preprocess(Image.new("RGB", (640, 480), color=(124, 116, 104)))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.abs(tensor).max() < 0.1  # ~ImageNet mean colour -> values close to 0


def test_softmax_is_a_probability_distribution() -> None:
    """Softmax output must be positive, sum to 1 and keep the order of the logits."""
    probs = AcneClassifier("does_not_exist.onnx")._softmax(np.array([[0.2, 1.8, -0.4, -1.1]]))
    assert np.all(probs > 0)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert int(np.argmax(probs)) == 1


def test_gradcam_highlights_the_active_region() -> None:
    """Grad-CAM must be hottest where the feature maps are active for the explained class."""
    features = np.zeros((3, 7, 7), dtype=np.float32)
    features[0, 6, 6] = 10.0                              # channel 0 active bottom-right
    heatmap = compute_gradcam(features, np.array([1.0, 0.0, 0.0]))
    assert heatmap.shape == (INPUT_SIZE[1], INPUT_SIZE[0])
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert heatmap[-1, -1] > 0.9 and heatmap[0, 0] < 0.1


def test_gradcam_ignores_negative_evidence() -> None:
    """ReLU step: regions that push AGAINST the class must not light up."""
    features = np.ones((1, 7, 7), dtype=np.float32)
    heatmap = compute_gradcam(features, np.array([-1.0]))  # only negative evidence
    assert np.all(heatmap == 0.0)


def test_load_classifier_weights_reads_final_layer(tmp_path: Path) -> None:
    """The (classes, channels) weight matrix must be read back from the ONNX file."""
    weights = np.arange(12, dtype=np.float32).reshape(4, 3)
    model_path = tmp_path / "tiny.onnx"
    _tiny_onnx_classifier(model_path, weights)
    np.testing.assert_array_equal(load_classifier_weights(model_path), weights)


def test_jet_colormap_goes_from_blue_to_red() -> None:
    """Low values must map to blue and high values to red."""
    colors = apply_jet_colormap(np.array([[0.0, 1.0]]))
    low, high = colors[0, 0], colors[0, 1]
    assert low[2] > low[0] and high[0] > high[2]


def test_render_gradcam_png_returns_png_of_model_size() -> None:
    """The overlay must be a valid 224x224 PNG image."""
    png = render_gradcam_png(Image.new("RGB", (500, 400)), np.zeros((7, 7), np.float32))
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert Image.open(io.BytesIO(png)).size == INPUT_SIZE
