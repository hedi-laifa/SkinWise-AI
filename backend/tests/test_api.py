"""Automated pytest suite for the SkinWise AI FastAPI + ONNX Runtime backend."""

import io
from typing import Any, List
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import CLASS_NAMES, MODEL_ARCHITECTURE, app


class _DummyOrtInput:
    """Lightweight stub representing an ONNX Runtime input node."""

    def __init__(self, name: str = "input") -> None:
        self.name = name


class _MockOrtSession:
    """Deterministic mock of `onnxruntime.InferenceSession` for unit testing."""

    def get_inputs(self) -> List[_DummyOrtInput]:
        return [_DummyOrtInput(name="input")]

    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> List[np.ndarray]:
        tensor = input_feed["input"]
        assert tensor.shape == (1, 3, 224, 224), f"Unexpected input tensor shape: {tensor.shape}"
        assert tensor.dtype == np.float32
        # Return deterministic 4-class logits favoring index 1 ("Moderate (Level 1)")
        logits = np.array([[0.2, 1.8, -0.4, -1.1]], dtype=np.float32)
        return [logits]


def _create_test_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (256, 256)) -> bytes:
    """Create a valid in-memory RGB image encoded as PNG or JPEG bytes."""
    image = Image.new("RGB", size, color=(210, 140, 130))
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def client() -> TestClient:
    """Provide a FastAPI TestClient with a deterministic ONNX session attached."""
    app.state.ort_session = _MockOrtSession()
    app.state.onnx_model_path = "backend/app/best_model.onnx"
    return TestClient(app)


def test_health_endpoint_returns_200(client: TestClient) -> None:
    """Verify GET /health returns HTTP 200 and reports model readiness."""
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert "onnx_model_path" in payload


def test_model_metadata_returns_expected_classes(client: TestClient) -> None:
    """Verify GET /model/metadata returns EfficientNet-B0 and the 4 ACNE04 classes."""
    response = client.get("/model/metadata")
    assert response.status_code == 200

    payload = response.json()
    assert payload["architecture"] == MODEL_ARCHITECTURE
    assert payload["num_classes"] == 4
    assert payload["classes"] == CLASS_NAMES
    assert payload["input_shape"] == [1, 3, 224, 224]


def test_predict_rejects_txt_file_upload(client: TestClient) -> None:
    """Verify POST /predict returns HTTP 400 when a .txt file is uploaded instead of an image."""
    txt_bytes = b"This is a plain text file, not a dermatological image."
    files = {"file": ("notes.txt", io.BytesIO(txt_bytes), "text/plain")}

    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_rejects_corrupt_image_bytes(client: TestClient) -> None:
    """Verify POST /predict returns HTTP 400 when corrupt bytes masquerade as a .jpg file."""
    corrupt_bytes = b"not-a-valid-jpeg-header-0000"
    files = {"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")}

    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Corrupt or unreadable image" in response.json()["detail"]


def test_predict_succeeds_with_valid_png(client: TestClient) -> None:
    """Verify POST /predict returns HTTP 200, predicted class, latency, and 4 probabilities."""
    png_bytes = _create_test_image_bytes(fmt="PNG")
    files = {"file": ("specimen.png", io.BytesIO(png_bytes), "image/png")}

    response = client.post("/predict", files=files)
    assert response.status_code == 200

    payload = response.json()
    assert payload["predicted_class"] == "Moderate (Level 1)"
    assert payload["predicted_class_index"] == 1
    assert isinstance(payload["latency_ms"], float)
    assert payload["latency_ms"] >= 0.0

    probs = payload["probabilities"]
    assert set(probs.keys()) == set(CLASS_NAMES)
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_predict_visualize_rejects_txt_file(client: TestClient) -> None:
    """Verify POST /predict/visualize returns HTTP 400 for invalid non-image files."""
    files = {"file": ("invalid.txt", io.BytesIO(b"hello world"), "text/plain")}
    response = client.post("/predict/visualize", files=files)
    assert response.status_code == 400


def test_predict_visualize_returns_png_heatmap(client: TestClient) -> None:
    """Verify POST /predict/visualize returns HTTP 200 with image/png media type."""
    jpg_bytes = _create_test_image_bytes(fmt="JPEG")
    files = {"file": ("sample.jpg", io.BytesIO(jpg_bytes), "image/jpeg")}

    response = client.post("/predict/visualize", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
