"""Strict Pydantic v2 response schemas for the SkinWise AI FastAPI backend."""

from typing import Dict, List, Literal
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "onnx_model_path": "backend/app/model.onnx",
            }
        },
    )

    status: Literal["healthy", "degraded"] = Field(
        ...,
        description="Overall operational status of the API service.",
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ONNX Runtime InferenceSession is loaded and ready.",
    )
    onnx_model_path: str = Field(
        ...,
        description="Configured filesystem path to the ONNX model artifact.",
    )


class ModelMetadataResponse(BaseModel):
    """Response schema for GET /model/metadata."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "architecture": "EfficientNet-B0",
                "num_classes": 4,
                "classes": [
                    "Mild (Level 0)",
                    "Moderate (Level 1)",
                    "Severe (Level 2)",
                    "Very Severe (Level 3)",
                ],
                "input_shape": [1, 3, 224, 224],
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            }
        },
    )

    architecture: str = Field(
        default="EfficientNet-B0",
        description="Backbone neural network architecture name.",
    )
    num_classes: int = Field(
        default=4,
        ge=1,
        description="Total number of target acne severity classes.",
    )
    classes: List[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Ordered list of the 4 ACNE04 severity class names.",
    )
    input_shape: List[int] = Field(
        default=[1, 3, 224, 224],
        description="Expected NCHW input tensor shape for ONNX inference.",
    )


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predicted_class": "Moderate (Level 1)",
                "predicted_class_index": 1,
                "confidence": 0.7842,
                "latency_ms": 14.32,
                "probabilities": {
                    "Mild (Level 0)": 0.1021,
                    "Moderate (Level 1)": 0.7842,
                    "Severe (Level 2)": 0.0894,
                    "Very Severe (Level 3)": 0.0243,
                },
            }
        },
    )

    predicted_class: str = Field(
        ...,
        description="Human-readable name of the highest-probability severity class.",
    )
    predicted_class_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Integer index (0-3) of the predicted severity class.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the top predicted class.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="ONNX Runtime inference execution latency in milliseconds.",
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Mapping of each of the 4 class names to its softmax probability.",
    )


class ErrorResponse(BaseModel):
    """Standardized error response schema."""

    detail: str = Field(..., description="Human-readable error explanation.")
