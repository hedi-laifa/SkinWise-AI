"""Baseline ResNet-18 training and MLflow tracking script for comparative analysis on ACNE04."""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from src.data.dataset import create_dataloaders
from src.models.train import (
    evaluate_one_epoch,
    set_seed,
    train_one_epoch,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Hyperparameters and configuration for the ResNet-18 baseline model."""

    data_dir: Path = Path("acne_1024")
    num_classes: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    epochs: int = 15
    num_workers: int = 2
    pretrained: bool = True
    freeze_backbone: bool = False
    seed: int = 42
    best_model_path: Path = Path("baseline_resnet18.pth")
    # Shares the exact same default MLflow experiment name as train.py for side-by-side UI comparison
    experiment_name: str = "SkinWise-ACNE04-EfficientNetB0"
    run_name: str = "Baseline-ResNet18"
    tracking_uri: str = "mlruns"


def build_resnet18_baseline(
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Load torchvision `resnet18` and replace the final `fc` head for 4-class ACNE04 grading.

    Args:
        num_classes: Number of target severity classes (default: 4).
        pretrained: Whether to initialize with ImageNet-1K pretrained weights.
        freeze_backbone: If True, freezes convolutional layers prior to `model.fc`.

    Returns:
        Configured `nn.Module` (ResNet-18) outputting `num_classes` logits.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model: nn.Module = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    return model


def train_baseline_model(
    config: BaselineTrainingConfig,
    dataloaders: Dict[str, DataLoader[Tuple[torch.Tensor, int]]] | None = None,
) -> Tuple[nn.Module, float, Path]:
    """Train the ResNet-18 baseline on identical splits and log comparable metrics to MLflow.

    Uses the exact same `seed`, `create_dataloaders` split, optimizer (`AdamW`),
    criterion (`CrossEntropyLoss`), and MLflow metric keys (`val_macro_f1`, `macro_f1`,
    `val_accuracy`, `accuracy`, `train_loss`, `val_loss`) as `src/models/train.py`
    so the baseline and EfficientNet-B0 runs can be compared side-by-side in the MLflow UI.

    Args:
        config: `BaselineTrainingConfig` containing hyperparameters and output paths.
        dataloaders: Optional pre-built dictionary of `'train'` and `'val'` DataLoaders.

    Returns:
        Tuple of `(trained_baseline_model, best_val_macro_f1, saved_checkpoint_path)`.
    """
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting ResNet-18 Baseline training on device: %s", device)

    # 1. Load identical stratified train/val/test splits using the same random seed
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

    # 2. Build ResNet-18 baseline model (4 classes)
    model = build_resnet18_baseline(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Setup identical CrossEntropyLoss and AdamW optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_model_path = Path(config.best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. Configure MLflow tracking under the shared experiment
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    best_val_f1: float = -1.0
    best_val_acc: float = 0.0

    with mlflow.start_run(run_name=config.run_name):
        mlflow.set_tags(
            {
                "model_role": "baseline",
                "architecture_family": "ResNet",
                "benchmark": "ACNE04",
            }
        )

        # Log hyperparameters matching train.py keys for side-by-side table comparison
        mlflow.log_params(
            {
                "architecture": "resnet18",
                "model_tier": "baseline",
                "num_classes": config.num_classes,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "weight_decay": config.weight_decay,
                "optimizer": "AdamW",
                "loss_function": "CrossEntropyLoss",
                "pretrained": config.pretrained,
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

            # Log identical metric names so MLflow overlays curves directly with EfficientNet-B0
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
                "[Baseline ResNet-18] Epoch [%02d/%02d] | Train Loss: %.4f, Acc: %.4f, "
                "Macro-F1: %.4f | Val Loss: %.4f, Acc: %.4f, Macro-F1: %.4f",
                epoch,
                config.epochs,
                train_loss,
                train_acc,
                train_f1,
                val_loss,
                val_acc,
                val_f1,
            )

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
                    "Saved new best ResNet-18 baseline to '%s' (Val Macro-F1: %.4f, Val Acc: %.4f).",
                    best_model_path,
                    best_val_f1,
                    best_val_acc,
                )

        if best_model_path.is_file():
            mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")

    return model, best_val_f1, best_model_path


def parse_args(argv: Sequence[str] | None = None) -> BaselineTrainingConfig:
    """Parse CLI arguments for the ResNet-18 baseline training script."""
    default_data_dir = (
        Path("acne_1024") if Path("acne_1024").exists() else Path("data/raw/acne_1024")
    )
    parser = argparse.ArgumentParser(
        description="Train a baseline ResNet-18 model on ACNE04 for comparative MLflow analysis."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Path to the ACNE04 dataset root directory.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("baseline_resnet18.pth"),
        help="Output path for the best ResNet-18 state dictionary.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="SkinWise-ACNE04-EfficientNetB0",
        help="MLflow experiment name (shared with train.py for side-by-side comparison).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="Baseline-ResNet18",
        help="MLflow run name displayed in the UI.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed matching train.py.")

    args = parser.parse_args(argv)
    return BaselineTrainingConfig(
        data_dir=args.data_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        best_model_path=args.output_path,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )


def main() -> None:
    """CLI entrypoint to train the ResNet-18 baseline model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_args()
    train_baseline_model(config)


if __name__ == "__main__":
    main()
