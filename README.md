# SkinWise AI — Automated Acne Severity Grading & MLOps Platform

**SkinWise AI** is an end-to-end deep learning and MLOps system for automated dermatological grading of facial acne severity on the **ACNE04** benchmark across the four clinical Hayashi severity grades:
- **Mild (Level 0)** (`acne0_1024`)
- **Moderate (Level 1)** (`acne1_1024`)
- **Severe (Level 2)** (`acne2_1024`)
- **Very Severe (Level 3)** (`acne3_1024`)

It pairs a leak-free **PyTorch + EfficientNet-B0** training pipeline (with **ResNet-18** baseline comparison, **MLflow** experiment tracking, **DVC** dataset versioning, **ONNX** export, and mathematical **Grad-CAM** explainability) with a decoupled **FastAPI + ONNX Runtime** backend and a **React 18 + Vite + Tailwind CSS** diagnostic frontend orchestrated via **Docker Compose**.

---

## Key Features

- **Leak-Free Data Pipeline (`src/data/`)**: Deterministic stratified train/val/test splitting (`70% / 15% / 15%`) performed strictly *before* applying augmentations, automatically excluding duplicate directories (`all_1024`, `small_1024`).
- **Transfer Learning & Baseline Comparison (`src/models/`)**:
  - **Primary Model (`train.py`)**: Fine-tunes an ImageNet-pretrained `EfficientNet-B0` backbone with `AdamW` and `CrossEntropyLoss`, selecting checkpoints by validation **Macro-F1** and exporting to **ONNX** (`best_model.onnx`).
  - **Baseline Model (`train_baseline.py`)**: Trains a `ResNet-18` baseline under identical splits and logs directly to the shared MLflow experiment for side-by-side comparison.
- **Explainable AI (`src/utils/gradcam.py`)**: Mathematical **Grad-CAM** activation mapping hooked into `model.features[-1]` to highlight inflammatory facial lesions driving the severity prediction.
- **High-Throughput ONNX Serving (`backend/`)**: Asynchronous FastAPI microservice powered by `onnxruntime` (`CPUExecutionProvider`) with strict Pydantic v2 schemas and image validation.
- **Interactive Diagnostic UI (`frontend/`)**: Responsive React 18 + Vite + Tailwind CSS v4 web interface featuring drag-and-drop image upload, 4-class confidence bars, inference latency metrics, and side-by-side Grad-CAM visualization.
- **Full MLOps & CI/CD**: Containerized multi-service deployment (`docker-compose.yml`) and automated linting (`flake8`) + API test suite (`pytest`) via GitHub Actions (`.github/workflows/ci.yml`).

---

## Project Structure

```text
SkinWise/
├── .github/workflows/
│   └── ci.yml                        # GitHub Actions CI (flake8 linting + pytest)
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI application & ONNX Runtime endpoints
│   │   ├── schemas.py                # Pydantic v2 request/response contracts
│   │   └── best_model.onnx           # Exported ONNX model graph for serving
│   ├── tests/
│   │   └── test_api.py               # Automated FastAPI unit & integration tests
│   ├── Dockerfile                    # Multi-stage Python 3.11 FastAPI container
│   └── requirements.txt              # Backend & testing Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx                   # React diagnostic dashboard & Grad-CAM viewer
│   │   ├── main.jsx                  # React DOM entry point
│   │   └── index.css                 # Tailwind CSS v4 entry stylesheet
│   ├── Dockerfile                    # Node 20 Alpine Vite container
│   ├── package.json                  # Frontend dependencies & scripts
│   └── vite.config.js                # Vite server & proxy configuration
├── report/
│   └── skinwise_technical_report.tex # LaTeX technical report
├── src/
│   ├── data/
│   │   ├── dataset.py                # ACNE04 dataset loader & leak-free stratified split
│   │   └── transforms.py             # Train & eval/inference torchvision transforms
│   ├── models/
│   │   ├── classifier.py             # EfficientNet-B0 4-class architecture builder
│   │   ├── train.py                  # Training loop, MLflow logging & ONNX export
│   │   └── train_baseline.py         # ResNet-18 baseline training script
│   └── utils/
│       └── gradcam.py                # Mathematical Grad-CAM implementation
├── .dockerignore
├── .flake8                           # Flake8 linter configuration
├── .gitignore
├── docker-compose.yml                # Full-stack container orchestration
└── README.md
```

---

## Quick Start (Docker Compose)

Launch both the FastAPI ONNX backend and the React/Vite frontend with hot-reloading using a single command:

```bash
docker-compose up --build
```

Once running, access the services at:
- **Frontend Diagnostic UI**: http://localhost:5173
- **FastAPI Interactive Swagger Docs**: http://localhost:8000/docs
- **Health Check Endpoint**: http://localhost:8000/health

To stop the containers:

```bash
docker-compose down
```

---

## Local Development & Training

### 1. Environment Setup

Create and activate a Python 3.11+ virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r backend/requirements.txt
pip install torch torchvision mlflow scikit-learn dvc
```

### 2. Data Versioning with DVC

The dataset expects the four ACNE04 class folders (`acne0_1024` through `acne3_1024`) inside `acne_1024/` or `data/raw/acne_1024/`. To version the dataset with DVC without committing raw images to Git:

```bash
dvc init
dvc add data/raw
git add data/raw.dvc .dvc/config .gitignore
```

### 3. Model Training & ONNX Export

Fine-tune **EfficientNet-B0** (automatically logs metrics/artifacts to `mlruns/` and exports `best_model.pth` and `backend/app/best_model.onnx`):

```bash
python -m src.models.train \
  --data-dir acne_1024 \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --output-path best_model.pth \
  --onnx-path backend/app/best_model.onnx
```

Train the **ResNet-18** baseline for comparative analysis in MLflow:

```bash
python -m src.models.train_baseline \
  --data-dir acne_1024 \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-4
```

Launch the **MLflow Tracking UI** to compare runs side-by-side:

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

### 4. Running Backend & Frontend Locally (Without Docker)

**Start the FastAPI Backend (Port 8000):**

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Start the React/Vite Frontend (Port 5173):**

```bash
cd frontend
npm install
npm run dev
```

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns service health (`healthy` or `degraded`) and whether the ONNX model session is loaded in memory. |
| `GET` | `/model/metadata` | Returns model architecture (`EfficientNet-B0`), input shape (`[1, 3, 224, 224]`), and the 4 ACNE04 severity class labels. |
| `POST` | `/predict` | Accepts a `multipart/form-data` image (`file`, JPEG or PNG), runs ONNX Runtime inference, and returns `predicted_class`, `predicted_class_index`, `confidence`, `latency_ms`, and per-class `probabilities`. |
| `POST` | `/predict/visualize` | Accepts a `multipart/form-data` image (`file`, JPEG or PNG) and returns a `224x224` Grad-CAM activation overlay as a binary `image/png` stream. |

### Example `curl` Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@acne_1024/acne0_1024/levle0_1.jpg;type=image/jpeg"
```

---

## Testing & Code Quality

Run the automated `flake8` linter and `pytest` unit test suite locally (matching the GitHub Actions CI pipeline):

```bash
# Lint backend and source modules
flake8 backend src --max-line-length=120 --extend-ignore=E203,W503

# Run FastAPI endpoint and validation tests
pytest backend/tests/test_api.py -v
```

---

## Technical Report

The LaTeX source for the accompanying academic paper is located at `report/skinwise_technical_report.tex`. Compile it with `pdflatex`:

```bash
cd report
pdflatex skinwise_technical_report.tex
```
