"""Transform pipelines for ACNE04 training, validation, and testing."""

from typing import Tuple
from torchvision import transforms

IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: Tuple[int, int] = IMAGE_SIZE,
    horizontal_flip_prob: float = 0.5,
    rotation_degrees: float = 15.0,
) -> transforms.Compose:
    """Build the training transform pipeline with augmentation and ImageNet normalization.

    Args:
        image_size: Target spatial dimensions (height, width). Defaults to (224, 224).
        horizontal_flip_prob: Probability of random horizontal flip. Defaults to 0.5.
        rotation_degrees: Maximum absolute degrees for slight random rotation. Defaults to 15.0.

    Returns:
        A torchvision.transforms.Compose pipeline for training data.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_test_transforms(
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> transforms.Compose:
    """Build the validation/test transform pipeline with ZERO data augmentation.

    Args:
        image_size: Target spatial dimensions (height, width). Defaults to (224, 224).

    Returns:
        A deterministic torchvision.transforms.Compose pipeline for validation/testing.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_transforms(split: str = "train") -> transforms.Compose:
    """Return the appropriate transform pipeline for a dataset split.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        The corresponding torchvision.transforms.Compose pipeline.

    Raises:
        ValueError: If `split` is not one of 'train', 'val', or 'test'.
    """
    normalized_split = split.strip().lower()
    if normalized_split == "train":
        return get_train_transforms()
    if normalized_split in {"val", "valid", "validation", "test"}:
        return get_val_test_transforms()
    raise ValueError(
        f"Invalid split '{split}'. Expected one of: 'train', 'val', 'test'."
    )


# Pre-instantiated pipelines for convenient import
train_transforms: transforms.Compose = get_train_transforms()
val_test_transforms: transforms.Compose = get_val_test_transforms()
