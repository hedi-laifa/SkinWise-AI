"""Complete PyTorch training, MLflow tracking, and ONNX export pipeline for EfficientNet-B0 on ACNE04."""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Dict, List, Sequence, Tuple

import mlflow
import mlflow.pytorch
import onnx
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders
from src.models.classifier import build_efficientnet_b0

try:
    from sklearn.metrics import accuracy_score as sk_accuracy_score
    from sklearn.metrics import f1_score as sk_f1_score
except ImportError:
    sk_accuracy_score = None
    sk_f1_score = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and output paths for EfficientNet-B0 fine-tuning & ONNX export."""

    data_dir: Path = Path("acne_1024")
    num_classes: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    epochs: int = 15
    num_workers: int = 2
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone: bool = False
    seed: int = 42
    best_model_path: Path = Path("best_model.pth")
    best_onnx_path: Path = Path("backend/app/best_model.onnx")
    backend_onnx_path: Path = Path("backend/app/best_model.onnx")
    experiment_name: str = "SkinWise-ACNE04-EfficientNetB0"
    run_name: str = "Advanced-EfficientNetB0"
    tracking_uri: str = "mlruns"


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for PyTorch CPU and CUDA RNGs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute classification accuracy across predictions."""
    if not y_true:
        return 0.0
    if sk_accuracy_score is not None:
        return float(sk_accuracy_score(y_true, y_pred))
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return float(correct / len(y_true))


def compute_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 4,
) -> float:
    """Compute unweighted Macro-F1 score across all 4 ACNE04 classes."""
    if not y_true:
        return 0.0

    if sk_f1_score is not None:
        return float(
            sk_f1_score(
                y_true,
                y_pred,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )

    f1_per_class: List[float] = []
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        denom = (2 * tp) + fp + fn
        f1_per_class.append((2.0 * tp / denom) if denom > 0 else 0.0)

    return float(sum(f1_per_class) / num_classes)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_classes: int = 4,
) -> Tuple[float, float, float]:
    """Run one training epoch using AdamW and CrossEntropyLoss.

    Returns:
        Tuple of `(train_loss, train_accuracy, train_macro_f1)`.
    """
    model.train()
    running_loss: float = 0.0
    total_samples: int = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = running_loss / max(1, total_samples)
    accuracy = compute_accuracy(all_targets, all_preds)
    macro_f1 = compute_macro_f1(all_targets, all_preds, num_classes=num_classes)
    return avg_loss, accuracy, macro_f1


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 4,
) -> Tuple[float, float, float]:
    """Run one validation/evaluation epoch with gradients disabled.

    Returns:
        Tuple of `(val_loss, val_accuracy, val_macro_f1)`.
    """
    model.eval()
    running_loss: float = 0.0
    total_samples: int = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_targets.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = running_loss / max(1, total_samples)
    accuracy = compute_accuracy(all_targets, all_preds)
    macro_f1 = compute_macro_f1(all_targets, all_preds, num_classes=num_classes)
    return avg_loss, accuracy, macro_f1


def export_model_to_onnx(
    checkpoint_path: Path | str = Path("best_model.pth"),
    onnx_output_path: Path | str = Path("best_model.onnx"),
    backend_copy_path: Path | str | None = Path("backend/app/best_model.onnx"),
    num_classes: int = 4,
    opset_version: int = 17,
) -> Path:
    """Load the best `.pth` state dict and export a validated ONNX graph (`best_model.onnx`).

    Uses `torch.onnx.export` with dynamic batch axes so the FastAPI ONNX Runtime
    backend can directly consume `best_model.onnx`.

    Args:
        checkpoint_path: Path to the saved `best_model.pth` state dictionary.
        onnx_output_path: Output path for the exported ONNX graph (`best_model.onnx`).
        backend_copy_path: Optional path inside `backend/app/` to mirror the ONNX file
            for containerized FastAPI serving.
        num_classes: Number of output classes (default: 4).
        opset_version: ONNX opset version (default: 17).

    Returns:
        `Path` to the exported and verified `best_model.onnx` file.
    """
    ckpt_path = Path(checkpoint_path)
    out_path = Path(onnx_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Cannot export to ONNX: checkpoint file not found at '{ckpt_path}'."
        )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Validate the exported ONNX graph structure
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    logger.info("Exported and verified ONNX graph at '%s'.", out_path)

    # Mirror into backend/app/ so the FastAPI service & Docker build use the trained model
    if backend_copy_path is not None:
        backend_target = Path(backend_copy_path)
        backend_target.parent.mkdir(parents=True, exist_ok=True)
        if out_path.resolve() != backend_target.resolve():
            shutil.copy2(out_path, backend_target)
            logger.info("Copied '%s' -> '%s' for FastAPI serving.", out_path, backend_target)

    return out_path


def train_model(
    config: TrainingConfig,
    dataloaders: Dict[str, DataLoader[Tuple[torch.Tensor, int]]] | None = None,
    export_onnx_on_finish: bool = True,
) -> Tuple[nn.Module, float, Path]:
    """Fine-tune EfficientNet-B0 with MLflow tracking, save `best_model.pth`, and export `best_model.onnx`.

    Args:
        config: `TrainingConfig` containing hyperparameters and output file paths.
        dataloaders: Optional pre-constructed dict with `'train'` and `'val'` DataLoaders.
        export_onnx_on_finish: If True, exports `best_model.pth` to `best_model.onnx`
            at the end of training.

    Returns:
        Tuple of `(trained_model, best_val_macro_f1, best_model_pth_path)`.
    """
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting EfficientNet-B0 fine-tuning on device: %s", device)

    if dataloaders is None:
        dataloaders = create_dataloaders(
            root_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 1. Initialize EfficientNet-B0 with 4-class head
    model = build_efficientnet_b0(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout=config.dropout,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    # 2. Setup CrossEntropyLoss and AdamW optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_model_path = Path(config.best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Configure MLflow tracking
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    best_val_f1: float = -1.0
    best_val_acc: float = 0.0
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.set_tags(
            {
                "model_role": "advanced",
                "architecture_family": "EfficientNet",
                "benchmark": "ACNE04",
            }
        )
        # Log required hyperparameters
        mlflow.log_params(
            {
                "architecture": "efficientnet_b0",
                "model_tier": "advanced",
                "num_classes": config.num_classes,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "weight_decay": config.weight_decay,
                "optimizer": "AdamW",
                "loss_function": "CrossEntropyLoss",
                "pretrained": config.pretrained,
                "dropout": config.dropout,
                "freeze_backbone": config.freeze_backbone,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "seed": config.seed,
                "device": str(device),
            }
        )

        for epoch in range(1, config.epochs + 1):
            train_loss, train_acc, train_f1 = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_classes=config.num_classes,
            )

            val_loss, val_acc, val_f1 = evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=config.num_classes,
            )

            # Log training and validation metrics (including accuracy and Macro-F1)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_macro_f1": train_f1,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "macro_f1": val_f1,
                },
                step=epoch,
            )

            logger.info(
                "Epoch [%02d/%02d] | Train Loss: %.4f, Acc: %.4f, Macro-F1: %.4f | "
                "Val Loss: %.4f, Acc: %.4f, Macro-F1: %.4f",
                epoch,
                config.epochs,
                train_loss,
                train_acc,
                train_f1,
                val_loss,
                val_acc,
                val_f1,
            )

            # Save best model state dictionary based on validation Macro-F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_metrics(
                    {
                        "best_val_macro_f1": best_val_f1,
                        "best_val_accuracy": best_val_acc,
                    },
                    step=epoch,
                )
                logger.info(
                    "Saved new best model to '%s' (Val Macro-F1: %.4f, Val Acc: %.4f).",
                    best_model_path,
                    best_val_f1,
                    best_val_acc,
                )

        if best_model_path.is_file():
            mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")

            if export_onnx_on_finish:
                onnx_path = export_model_to_onnx(
                    checkpoint_path=best_model_path,
                    onnx_output_path=config.best_onnx_path,
                    backend_copy_path=config.backend_onnx_path,
                    num_classes=config.num_classes,
                )
                mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

    return model, best_val_f1, best_model_path


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse CLI arguments for training and ONNX export."""
    default_data_dir = (
        Path("acne_1024") if Path("acne_1024").exists() else Path("data/raw/acne_1024")
    )
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B0 on ACNE04, track with MLflow, and export best_model.onnx."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Path to the ACNE04 dataset directory.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("best_model.pth"),
        help="Output path for best PyTorch state dict (best_model.pth).",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("best_model.onnx"),
        help="Output path for exported ONNX model (best_model.onnx).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="SkinWise-ACNE04-EfficientNetB0",
        help="MLflow experiment name.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args(argv)
    return TrainingConfig(
        data_dir=args.data_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        best_model_path=args.output_path,
        best_onnx_path=args.onnx_path,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    """CLI entrypoint to run training and ONNX export."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_args()
    train_model(config)


if __name__ == "__main__":
    main()
