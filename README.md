# SkinWise AI — Automated Acne Severity Grading & MLOps Platform

**SkinWise AI** is an end-to-end deep learning and MLOps system for automated dermatological grading of facial acne severity on the **ACNE04** benchmark across the four clinical Hayashi severity grades:
- **Mild (Level 0)** (`acne0_1024`)
- **Moderate (Level 1)** (`acne1_1024`)
- **Severe (Level 2)** (`acne2_1024`)
- **Very Severe (Level 3)** (`acne3_1024`)

It pairs a leak-free **PyTorch + EfficientNet-B0** training pipeline (with **ResNet-18** baseline comparison, **MLflow** experiment tracking, **DVC** dataset versioning, **ONNX** export, and mathematical **Grad-CAM** explainability) with a decoupled **FastAPI + ONNX Runtime** backend and a **React 18 + Vite + Tailwind CSS** diagnostic frontend orchestrated via **Docker Compose**.

---

## Key Features

- **Leak-Free Data Pipeline (`src/data/`)**: Deterministic stratified train/val/test splitting (`70% / 15% / 15%`) performed strictly *before* applying augmentations, automatically excluding duplicate directories (`all_1024`, `small_1024`). The split is written to `data/processed/{train,val,test}.csv` by the DVC `prepare` stage.
- **Transfer Learning & Baseline Comparison (`src/models/`)**:
  - **Primary Model (`train.py`)**: Fine-tunes an ImageNet-pretrained `EfficientNet-B0` backbone with `AdamW` and `CrossEntropyLoss`, selecting checkpoints by validation **Macro-F1** and exporting to **ONNX** (`best_model.onnx`).
  - **Baseline Model (`train_baseline.py`)**: Trains a `ResNet-18` baseline under identical splits and logs directly to the shared MLflow experiment for side-by-side comparison.
- **Explainable AI (`src/utils/gradcam.py`)**: Mathematical **Grad-CAM** activation mapping hooked into `model.features[-1]` to highlight inflammatory facial lesions driving the severity prediction.
- **High-Throughput ONNX Serving (`backend/`)**: Asynchronous FastAPI microservice powered by `onnxruntime` (`CPUExecutionProvider`) with strict Pydantic v2 schemas and image validation.
- **Interactive Diagnostic UI (`frontend/`)**: Responsive React 18 + Vite + Tailwind CSS v4 web interface featuring drag-and-drop image upload, 4-class confidence bars, inference latency metrics, and side-by-side Grad-CAM visualization.
- **Full MLOps & CI/CD**: Containerized multi-service deployment (`docker-compose.yml`) and automated linting (`flake8`) + API test suite (`pytest`) via GitHub Actions (`.github/workflows/ci-tests.yml` + `docker-build.yml`).

---

## Project Structure

```text
SkinWise/
├── .github/workflows/
│   ├── ci-tests.yml                  # CI: flake8 linting + pytest
│   └── docker-build.yml              # CI: Docker image builds + compose smoke test
├── data/                             # Versioned with DVC, never committed to Git
│   ├── raw/
│   │   ├── acne_1024/                # ACNE04 images (acne0_1024 … acne3_1024)
│   │   └── acne_1024.dvc             # DVC pointer file (committed)
│   └── processed/                    # train.csv / val.csv / test.csv (DVC `prepare` stage)
├── models/                           # Model artifacts (DVC outputs, git-ignored)
│   ├── best_model.pth                # EfficientNet-B0 weights (trained on Kaggle GPU)
│   └── best_model.onnx               # ONNX export used for evaluation
├── src/                              # Deep Learning & MLOps source code
│   ├── data/
│   │   ├── dataset.py                # ACNE04 Dataset + leak-free stratified split
│   │   └── transforms.py             # Train / eval torchvision transforms
│   ├── models/
│   │   ├── architecture.py           # EfficientNet-B0 4-class network
│   │   ├── train.py                  # Training loop + MLflow tracking + ONNX export
│   │   ├── train_baseline.py         # ResNet-18 baseline
│   │   └── evaluate.py               # Test-set metrics on the ONNX model
│   └── utils/
│       └── gradcam.py                # Grad-CAM explainability
├── backend/                          # FastAPI REST API
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                   # FastAPI instance & routes
│   │   ├── schemas.py                # Pydantic validation models
│   │   ├── inference.py              # ONNX Runtime wrapper class
│   │   └── best_model.onnx           # Copy of the model served by the API
│   └── tests/
│       └── test_api.py               # Pytest suite (endpoints, mocks)
├── frontend/                         # React 18 + Vite + Tailwind
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.jsx                   # Page layout, state & API calls
│       ├── services/api.js           # fetch calls to FastAPI
│       ├── constants/severity.js     # Class names, colours, descriptions
│       └── components/
│           ├── Header.jsx            # Title + API health badge
│           ├── SeverityCard.jsx      # Grade reference cards
│           ├── Dropzone.jsx          # Drag & drop upload + Analyze button
│           ├── WebcamCapture.jsx     # Live camera capture
│           ├── ScanHistory.jsx       # Last 5 analyses
│           ├── AnalysisStates.jsx    # Empty / loading states
│           ├── PredictionBanner.jsx  # Predicted grade + confidence ring
│           ├── GradCamViewer.jsx     # Side-by-side / blended heatmap
│           └── ProbabilityBars.jsx   # Softmax probability chart
├── notebooks/
│   └── 01_eda.ipynb                  # Exploratory data analysis (read-only, outputs saved)
├── report/                           # LaTeX sources of the technical report
├── docker-compose.yml                # Local multi-container orchestration
├── dvc.yaml                          # DVC pipeline: prepare → train → export_onnx → evaluate
├── params.yaml                       # Training hyperparameters
├── eval_metrics.json                 # Test-set metrics (DVC metrics file)
├── .dvcignore / .dockerignore / .gitignore / .flake8
└── README.md
```

---

## Quick Start (Docker Compose)

Launch the production stack with a single command: the FastAPI + ONNX Runtime backend (no PyTorch, non-root user) and the React app built and served by **nginx**:

```bash
docker-compose up --build
```

Once running, access the services at:
- **Frontend Diagnostic UI**: http://localhost:5173
- **FastAPI Interactive Swagger Docs**: http://localhost:8000/docs
- **Health Check Endpoint**: http://localhost:8000/health

> The model file `backend/app/best_model.onnx` is not stored in Git: run `dvc pull` (or the training/export
> steps below) before building, otherwise `/health` reports `degraded` and `/predict` returns 503.

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
pip install torch torchvision mlflow scikit-learn dvc   # training tools
pip install pytest httpx flake8                          # test tools
```

### 2. Data Versioning with DVC

The ACNE04 images live in `data/raw/acne_1024/` and are tracked by DVC (Git only stores the small pointer `data/raw/acne_1024.dvc`).

```bash
dvc pull            # download the dataset (and model) from the DVC remote
dvc dag             # show the pipeline: prepare -> train -> export_onnx -> evaluate
dvc status          # check which stages are out of date
```

> The model was trained on a Kaggle GPU. Its outputs are recorded in `dvc.lock`
> (created with `dvc commit`, without retraining), so `dvc status` reports the pipeline as up to date.

**Shared DVC remote (DagsHub, free):** the dataset and models are stored on DagsHub so that
`dvc pull` works on any machine. One-time setup:

```bash
# 1. On https://dagshub.com/repo/connect, connect the GitHub repo hedi-laifa/SkinWise-AI
# 2. Add the remote (shared, committed to Git) and make it the default
dvc remote add -d origin https://dagshub.com/<dagshub-user>/SkinWise-AI.dvc
# 3. Credentials stay in .dvc/config.local (git-ignored, never committed)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-user>
dvc remote modify origin --local password <dagshub-token>   # DagsHub > Settings > Tokens
# 4. Upload the dataset and models
dvc push
```

### Exploratory Data Analysis

`notebooks/01_eda.ipynb` explores the dataset (class balance, image properties, colour bias, duplicate images, split check, effect of resizing). Its outputs are saved, so it can be read directly on GitHub. To re-run it:

```bash
pip install jupyter
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Data Split, Training, Evaluation

```bash
# Write the leak-free split (seed=42, 70/15/15) to data/processed/*.csv
python -m src.data.dataset

# Fine-tune EfficientNet-B0 (logs to MLflow, saves models/best_model.pth + .onnx)
python -m src.models.train --data-dir data/raw/acne_1024 --epochs 15 --batch-size 32

# Train the ResNet-18 baseline with the SAME settings as the EfficientNet-B0 run
# (same split, optimizer, epochs, augmentation) -> MLflow + models/baseline_resnet18.onnx
python -m src.models.train_baseline --data-dir data/raw/acne_1024 --epochs 15 \
  --no-color-augmentation --onnx-path models/baseline_resnet18.onnx
python -m src.models.evaluate --onnx models/baseline_resnet18.onnx \
  --data-dir data/raw/acne_1024 --output eval_metrics_baseline.json

# Evaluate the ONNX model on the test split -> eval_metrics.json
python -m src.models.evaluate --onnx models/best_model.onnx --data-dir data/raw/acne_1024
```

# Weight optimization (no retraining): FP16 + INT8 versions, compared on the test split
python -m src.models.quantize   # -> models/best_model_{fp16,int8}.onnx + quantization_metrics.json
```

| Model version | Size | Test accuracy | Test Macro-F1 |
| :--- | ---: | ---: | ---: |
| FP32 (served by the API) | 16.0 MB | 77.0 % | 0.729 |
| FP16 | 8.1 MB | 77.0 % | 0.729 |
| INT8 (static, depthwise convs kept in FP32) | 5.2 MB | 72.7 % | 0.689 |

**Baseline comparison** (same split, optimizer, epochs and augmentation; only the architecture changes):

| Model | Params | ONNX size | Test accuracy | Test Macro-F1 | F1 *Very Severe* | CPU latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 (baseline) | 11.2 M | 44.7 MB | 75.6 % | 0.670 | 0.455 | 20.5 ms |
| **EfficientNet-B0** | **4.0 M** | **16.0 MB** | **77.0 %** | **0.729** | **0.727** | **10.2 ms** |

Launch the **MLflow Tracking UI** to compare runs side-by-side:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

### 4. Development Mode (Without Docker, with hot-reload)

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
  -F "file=@data/raw/acne_1024/acne0_1024/levle0_1.jpg;type=image/jpeg"
```

---

## Testing & Code Quality

Run the same checks as the GitHub Actions CI pipeline:

```bash
# Python: lint + 16 tests (API endpoints and inference functions, incl. Grad-CAM)
flake8 backend src --max-line-length=120 --extend-ignore=E203,W503
pytest backend/tests -v

# Frontend: ESLint (zero warnings allowed) + production build
cd frontend
npm run lint
npm run build
```

---

## Technical Report

The final report is `rapport_technique.pdf` at the project root. Its LaTeX source is `report/rapport_technique.tex`, with all figures in `report/figures/`. To rebuild it, run `pdflatex rapport_technique.tex` twice from the `report/` folder, or upload the folder to Overleaf.
# SkinWise-AI
