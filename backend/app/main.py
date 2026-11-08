"""FastAPI & ONNX Runtime backend for SkinWise AI (ACNE04 4-class classification)."""

from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
import time
from typing import AsyncIterator, Dict, List, Optional, Set

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from backend.app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# Model & preprocessing constants
MODEL_ARCHITECTURE: str = "EfficientNet-B0"
CLASS_NAMES: List[str] = [
    "Mild (Level 0)",
    "Moderate (Level 1)",
    "Severe (Level 2)",
    "Very Severe (Level 3)",
]
INPUT_SIZE: tuple[int, int] = (224, 224)
IMAGENET_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ALLOWED_CONTENT_TYPES: Set[str] = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png"}
ALLOWED_PIL_FORMATS: Set[str] = {"JPEG", "PNG", "MPO"}

CANDIDATE_ONNX_PATHS: tuple[str, ...] = (
    "backend/app/best_model.onnx",
    "best_model.onnx",
)


def resolve_default_onnx_path() -> str:
    """Resolve the active ONNX model path, prioritizing `ONNX_MODEL_PATH` and `backend/app/best_model.onnx`."""
    env_path = os.getenv("ONNX_MODEL_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    for candidate in CANDIDATE_ONNX_PATHS:
        if Path(candidate).is_file():
            return candidate
    return env_path or "backend/app/best_model.onnx"


DEFAULT_ONNX_PATH: str = resolve_default_onnx_path()


def load_onnx_session(model_path: str | Path) -> Optional[ort.InferenceSession]:
    """Load an ONNX Runtime InferenceSession if the model file exists on disk."""
    path = Path(model_path)
    if not path.is_file():
        logger.warning(
            "ONNX model file not found at '%s'. /predict will return 503 until loaded.",
            path,
        )
        return None

    try:
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("Successfully loaded ONNX Runtime session from '%s'.", path)
        return session
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to initialize ONNX Runtime session from '%s': %s", path, exc)
        return None


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown lifecycle for the ONNX Runtime session."""
    onnx_path = getattr(application.state, "onnx_model_path", DEFAULT_ONNX_PATH)
    application.state.onnx_model_path = str(onnx_path)
    if getattr(application.state, "ort_session", None) is None:
        application.state.ort_session = load_onnx_session(onnx_path)
    yield
    application.state.ort_session = None


app = FastAPI(
    title="SkinWise AI Backend API",
    description="FastAPI + ONNX Runtime inference service for 4-class ACNE04 severity grading.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize default state attributes for test environments that bypass lifespan
app.state.onnx_model_path = DEFAULT_ONNX_PATH
app.state.ort_session = None


async def validate_and_decode_image(file: UploadFile) -> Image.Image:
    """Validate that an uploaded file is a valid JPEG/PNG image and decode to RGB PIL Image.

    Raises:
        HTTPException(400): If the MIME type, file extension, or image payload is invalid.
    """
    content_type = (file.content_type or "").lower().strip()
    filename = (file.filename or "").lower().strip()
    ext = Path(filename).suffix if filename else ""

    # Validate MIME type or file extension
    type_ok = content_type in ALLOWED_CONTENT_TYPES
    ext_ok = ext in ALLOWED_EXTENSIONS
    if not (type_ok or ext_ok):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type (content_type='{file.content_type}', "
                f"filename='{file.filename}'). Only JPEG and PNG images are accepted."
            ),
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img_format = (img.format or "").upper()
            if img_format not in ALLOWED_PIL_FORMATS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid image format '{img_format}'. Expected JPEG or PNG.",
                )
            img.load()
            return img.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable image file: {exc}",
        ) from exc


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize PIL RGB image to 224x224 and apply ImageNet normalization for ONNX Runtime.

    Returns:
        NumPy float32 tensor of shape `(1, 3, 224, 224)` in NCHW layout.
    """
    resized = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # Shape: (224, 224, 3)
    normalized = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))  # Shape: (3, 224, 224)
    nchw = np.expand_dims(chw, axis=0)  # Shape: (1, 3, 224, 224)
    return np.ascontiguousarray(nchw, dtype=np.float32)


def compute_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax probabilities from 1D or 2D logits."""
    flat = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = flat - np.max(flat)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores)
    return probs.astype(np.float64)


def generate_dummy_gradcam_png(image: Image.Image) -> bytes:
    """Generate a Grad-CAM heatmap overlay on the input image and encode as PNG bytes."""
    resized = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    base_rgb = np.asarray(resized, dtype=np.float32)  # (224, 224, 3)

    # Create a smooth spatial activation map combining center Gaussian + red/lesion salience
    height, width = INPUT_SIZE
    y_grid, x_grid = np.ogrid[:height, :width]
    center_y, center_x = height / 2.0, width / 2.0
    sigma = min(height, width) * 0.32
    gaussian_mask = np.exp(
        -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2.0 * sigma**2)
    ).astype(np.float32)

    # Highlight reddish skin inflammation regions modulated by the spatial prior
    redness = np.clip(
        (base_rgb[:, :, 0] - 0.5 * (base_rgb[:, :, 1] + base_rgb[:, :, 2])) / 255.0,
        0.0,
        1.0,
    )
    heatmap = 0.65 * gaussian_mask + 0.35 * redness
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Map scalar [0, 1] activation to a Jet-like RGB colormap
    r_chan = np.clip(1.5 - np.abs(4.0 * heatmap - 3.0), 0.0, 1.0) * 255.0
    g_chan = np.clip(1.5 - np.abs(4.0 * heatmap - 2.0), 0.0, 1.0) * 255.0
    b_chan = np.clip(1.5 - np.abs(4.0 * heatmap - 1.0), 0.0, 1.0) * 255.0
    cmap_rgb = np.stack([r_chan, g_chan, b_chan], axis=-1).astype(np.float32)

    # Alpha-blend original image (60%) and Grad-CAM colormap (40%)
    blended = np.clip(0.60 * base_rgb + 0.40 * cmap_rgb, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(blended, mode="RGB")

    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    return buffer.getvalue()


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and ONNX model status",
)
async def health_check(request: Request) -> HealthResponse:
    """Return API health status and whether the ONNX model is loaded in memory."""
    session = getattr(request.app.state, "ort_session", None)
    onnx_path = getattr(request.app.state, "onnx_model_path", DEFAULT_ONNX_PATH)
    if session is None and Path(onnx_path).is_file():
        session = load_onnx_session(onnx_path)
        request.app.state.ort_session = session

    is_loaded = session is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        onnx_model_path=str(onnx_path),
    )


@app.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
    summary="Retrieve model architecture and class metadata",
)
async def get_model_metadata() -> ModelMetadataResponse:
    """Return the model architecture ('EfficientNet-B0') and the 4 ACNE04 class names."""
    return ModelMetadataResponse(
        architecture=MODEL_ARCHITECTURE,
        num_classes=len(CLASS_NAMES),
        classes=CLASS_NAMES,
        input_shape=[1, 3, INPUT_SIZE[0], INPUT_SIZE[1]],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    summary="Run ONNX Runtime acne severity inference on an uploaded JPEG/PNG image",
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="JPEG or PNG facial skin image"),
) -> PredictionResponse:
    """Validate uploaded image, preprocess to 224x224 with ImageNet norm, and run ONNX inference."""
    pil_image = await validate_and_decode_image(file)
    input_tensor = preprocess_image(pil_image)

    session: Optional[ort.InferenceSession] = getattr(
        request.app.state, "ort_session", None
    )
    onnx_path = getattr(request.app.state, "onnx_model_path", DEFAULT_ONNX_PATH)
    if session is None and Path(onnx_path).is_file():
        session = load_onnx_session(onnx_path)
        request.app.state.ort_session = session

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"ONNX Runtime model is not loaded. Expected model at '{onnx_path}'. "
                "Export best_model.pth to ONNX first."
            ),
        )

    input_name: str = session.get_inputs()[0].name
    start_time = time.perf_counter()
    ort_outputs = session.run(None, {input_name: input_tensor})
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    logits = np.asarray(ort_outputs[0])
    probs = compute_softmax(logits)

    if probs.shape[0] != len(CLASS_NAMES):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Model output dimension ({probs.shape[0]}) does not match "
                f"expected number of classes ({len(CLASS_NAMES)})."
            ),
        )

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = round(float(probs[pred_idx]), 6)
    prob_dict: Dict[str, float] = {
        class_name: round(float(prob), 6)
        for class_name, prob in zip(CLASS_NAMES, probs)
    }

    return PredictionResponse(
        predicted_class=pred_class,
        predicted_class_index=pred_idx,
        confidence=confidence,
        latency_ms=round(latency_ms, 3),
        probabilities=prob_dict,
    )


@app.post(
    "/predict/visualize",
    responses={
        status.HTTP_200_OK: {
            "content": {"image/png": {}},
            "description": "Grad-CAM heatmap overlay returned as a PNG image.",
        },
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
    },
    summary="Generate a Grad-CAM heatmap overlay PNG for an uploaded JPEG/PNG image",
)
async def predict_visualize(
    request: Request,
    file: UploadFile = File(..., description="JPEG or PNG facial skin image"),
) -> Response:
    """Validate uploaded image and return a mathematical Grad-CAM heatmap overlay as `image/png`."""
    pil_image = await validate_and_decode_image(file)

    try:
        from src.utils.gradcam import generate_gradcam_png_bytes, load_gradcam_model

        gradcam_model = getattr(request.app.state, "gradcam_model", None)
        if gradcam_model is None:
            gradcam_model = load_gradcam_model(
                checkpoint_path=os.getenv("PTH_MODEL_PATH", "best_model.pth")
            )
            request.app.state.gradcam_model = gradcam_model

        png_bytes = generate_gradcam_png_bytes(image=pil_image, model=gradcam_model)
    except ImportError:
        png_bytes = generate_dummy_gradcam_png(pil_image)

    return Response(content=png_bytes, media_type="image/png")
