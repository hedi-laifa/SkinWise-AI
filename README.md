# SkinWise AI

Acne severity grading from face photos. Project for the Advanced Deep Learning module (axis 1: CNN and transfer learning).

The model takes a photo and predicts one of the four grades used in the ACNE04 dataset: mild, moderate, severe or very severe. It also returns a Grad-CAM heatmap that shows which areas of the face the prediction is based on.

## Results

On the test split (209 images, never used for training or for picking the best epoch):

| Model | Accuracy | Macro-F1 | ONNX size | CPU latency |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 (baseline) | 75.6 % | 0.670 | 44.7 MB | 20.5 ms |
| EfficientNet-B0 | 77.0 % | 0.729 | 16.0 MB | 10.2 ms |

Both models were trained with the same split, optimizer, number of epochs and augmentation. EfficientNet-B0 is the one served by the API.

We also tried smaller versions of the model: FP16 (8.1 MB) gives the same scores, INT8 (5.2 MB) drops to 72.7 % accuracy and 0.689 Macro-F1.

## How it is organised

- `src/`: data split, training, evaluation, ONNX export, quantization and Grad-CAM.
- `backend/`: FastAPI service that runs the ONNX model with ONNX Runtime (no PyTorch needed).
- `frontend/`: React + Vite + Tailwind interface (drag and drop, example images, webcam, Grad-CAM view).
- `notebooks/01_eda.ipynb`: exploration of the dataset.
- `report/`: LaTeX source of the report. The compiled version is `rapport_technique.pdf`.
- DVC tracks the dataset and the models, MLflow tracks the training runs, GitHub Actions runs the tests and builds the Docker images.

```text
.github/workflows/   ci-tests.yml, docker-build.yml
data/raw/            acne_1024.dvc (images are in DVC, not in Git)
src/                 data/, models/, utils/
backend/             app/ (main.py, schemas.py, inference.py), tests/
frontend/            src/ (App.jsx, components/, services/api.js)
docker-compose.yml
dvc.yaml, params.yaml
```

## Run the app

```bash
docker-compose up --build
```

Then open:

- the app: http://localhost:5173
- the API docs: http://localhost:8000/docs

The model file `backend/app/best_model.onnx` is not stored in Git. Get it with `dvc pull`, or rebuild it with the training steps below. Without it the API still starts, but `/health` says `degraded` and `/predict` returns 503.

To run without Docker (with hot reload):

```bash
uvicorn backend.app.main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

## Train and evaluate

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r backend/requirements.txt
pip install torch torchvision mlflow scikit-learn dvc

dvc pull                        # dataset and models
python -m src.data.dataset      # writes the train/val/test split to data/processed/
python -m src.models.train --data-dir data/raw/acne_1024 --epochs 15 --batch-size 32
python -m src.models.evaluate --onnx models/best_model.onnx --data-dir data/raw/acne_1024
```

Baseline and quantization:

```bash
python -m src.models.train_baseline --data-dir data/raw/acne_1024 --epochs 15 \
  --no-color-augmentation --onnx-path models/baseline_resnet18.onnx
python -m src.models.evaluate --onnx models/baseline_resnet18.onnx \
  --data-dir data/raw/acne_1024 --output eval_metrics_baseline.json
python -m src.models.quantize
```

The whole pipeline is also described in `dvc.yaml` (`dvc dag` to see it, `dvc repro` to run it). Our model was trained on a Kaggle GPU, then its outputs were recorded with `dvc commit`.

To look at the training runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

## API

| Method | Route | What it does |
| --- | --- | --- |
| GET | `/health` | Server status, whether the model is loaded, GPU availability |
| GET | `/model/metadata` | Architecture, input shape and class names |
| POST | `/predict` | Takes a JPEG/PNG image, returns the grade, the probabilities and the latency |
| POST | `/predict/visualize` | Takes a JPEG/PNG image, returns the Grad-CAM overlay as a PNG |

```bash
curl -X POST http://localhost:8000/predict -F "file=@photo.jpg"
```

A file that is not a valid image gets a 400, and a request sent before the model is loaded gets a 503.

## Tests

Same checks as the CI:

```bash
pip install pytest httpx flake8
flake8 backend src --max-line-length=120 --extend-ignore=E203,W503
pytest backend/tests -v

cd frontend
npm run lint
npm run build
```

## Dataset

ACNE04 (Wu et al., ICCV 2019), 1,378 face images graded on the Hayashi scale: https://github.com/xpwu95/LDL

This is a student project, not a medical tool.
