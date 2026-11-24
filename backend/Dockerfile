# ==============================================================================
# SkinWise AI - Production backend image (FastAPI + ONNX Runtime, no PyTorch)
# ==============================================================================

# ---------------------------- Stage 1: Builder --------------------------------
# Installs the Python dependencies in an isolated virtual environment.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /build/requirements.txt

# ---------------------------- Stage 2: Runtime --------------------------------
# Only the virtual environment and the API code are copied: no compilers, no tests.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    ONNX_MODEL_PATH=/app/backend/app/best_model.onnx

WORKDIR /app

# libgomp1: OpenMP runtime used by ONNX Runtime | curl: container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
# The API code and the exported ONNX model (backend/app/best_model.onnx); tests are not copied
COPY backend/__init__.py /app/backend/__init__.py
COPY backend/app /app/backend/app

# Run as an unprivileged user instead of root
RUN useradd --create-home --uid 1000 appuser
USER 1000

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8000/health"]

# Production server: no --reload (that is a development option)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
