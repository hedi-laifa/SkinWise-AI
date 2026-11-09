"""Modular EfficientNet-B0 model definition for 4-class ACNE04 severity classification."""

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Instantiate an EfficientNet-B0 model and replace the final classifier head.

    Args:
        num_classes: Number of output classes (default: 4 for ACNE04).
        pretrained: If True, loads ImageNet-1K pre-trained weights.
        dropout: Dropout probability before the final linear layer.
        freeze_backbone: If True, freezes convolutional feature extractor weights.

    Returns:
        Configured `torch.nn.Module` (EfficientNet-B0) outputting `num_classes` logits.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model: nn.Module = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the final classifier head:
    # Default torchvision EfficientNet-B0 classifier is Sequential(Dropout, Linear(1280, 1000))
    in_features: int = model.classifier[1].in_features  # type: ignore[index, assignment]
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model


def load_trained_model(
    checkpoint_path: str,
    num_classes: int = 4,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load an EfficientNet-B0 model from a saved `.pth` state dict checkpoint.

    Args:
        checkpoint_path: Path to the saved `.pth` weights file.
        num_classes: Number of output classes.
        device: Target device ('cpu' or 'cuda').

    Returns:
        Model in evaluation mode (`model.eval()`) on the target device.
    """
    target_device = torch.device(device)
    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model
