"""Production PyTorch Dataset and DataLoader pipeline for the ACNE04 dataset (4 severity classes)."""

import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import get_train_transforms, get_val_test_transforms

logger = logging.getLogger(__name__)

# Supported image extensions (lowercase)
VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Canonical ACNE04 4-class human-readable labels
CLASS_IDX_TO_NAME: Dict[int, str] = {
    0: "Mild (Level 0)",
    1: "Moderate (Level 1)",
    2: "Severe (Level 2)",
    3: "Very Severe (Level 3)",
}

# Directory name mapping to 4 severity classes (0..3)
# Supports both raw ACNE04 directory names (acne0_1024..acne3_1024) and standard names
DIRECTORY_TO_CLASS_IDX: Dict[str, int] = {
    "acne0_1024": 0,
    "acne1_1024": 1,
    "acne2_1024": 2,
    "acne3_1024": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "level0": 0,
    "level1": 1,
    "level2": 2,
    "level3": 3,
    "class0": 0,
    "class1": 1,
    "class2": 2,
    "class3": 3,
    "mild": 0,
    "moderate": 1,
    "severe": 2,
    "very_severe": 3,
}

# Auxiliary folders in the raw ACNE04 distribution that contain duplicate copies of images
# Explicitly excluded to prevent data leakage across train/val/test splits
EXCLUDED_DIRECTORIES: Tuple[str, ...] = (
    "all_1024",
    "small_1024",
    "small_1024_renamed",
    "acne3_512_selection",
)

TransformCallable = Callable[[Image.Image], torch.Tensor]


class CorruptImageError(RuntimeError):
    """Raised when an image file is corrupt or cannot be decoded."""


class ACNE04Dataset(Dataset[Tuple[torch.Tensor, int]]):
    """PyTorch Dataset for ACNE04 4-class acne severity classification.

    Reads images from a class-organized directory structure and maps them to 4 integer
    classes (0, 1, 2, 3). Includes strict error handling for corrupt or unreadable images.
    """

    NUM_CLASSES: int = 4

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[TransformCallable] = None,
        samples: Optional[Sequence[Tuple[Path, int]]] = None,
        skip_corrupt_on_init: bool = False,
        max_corrupt_retries: int = 5,
    ) -> None:
        """Initialize the ACNE04Dataset.

        Args:
            root_dir: Path to the root dataset directory containing class subdirectories.
            transform: Optional torchvision transform pipeline applied to PIL RGB images.
            samples: Optional pre-collected sequence of (image_path, class_idx) pairs
                (used when creating leak-free train/val/test splits).
            skip_corrupt_on_init: If True, validates every image header during initialization
                and drops corrupt files upfront.
            max_corrupt_retries: Maximum fallback attempts in `__getitem__` if a corrupt
                image is encountered at runtime.

        Raises:
            FileNotFoundError: If `root_dir` does not exist when scanning directories.
            ValueError: If no valid images or expected 4 classes are found.
        """
        self.root_dir: Path = Path(root_dir)
        self.transform: Optional[TransformCallable] = transform
        self.max_corrupt_retries: int = max(1, max_corrupt_retries)
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = dict(CLASS_IDX_TO_NAME)

        if samples is not None:
            self.samples: List[Tuple[Path, int]] = list(samples)
        else:
            self.samples = self._discover_samples()

        if skip_corrupt_on_init:
            self.samples = self._filter_corrupt_samples(self.samples)

        if not self.samples:
            raise ValueError(
                f"No valid image samples found for ACNE04Dataset in '{self.root_dir}'."
            )

    def _discover_samples(self) -> List[Tuple[Path, int]]:
        """Scan `self.root_dir` and map subdirectories to the 4 ACNE04 classes."""
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory does not exist: '{self.root_dir}'"
            )
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"Dataset root path is not a directory: '{self.root_dir}'"
            )

        collected: List[Tuple[Path, int]] = []
        subdirs = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])

        for subdir in subdirs:
            dir_name_lower = subdir.name.lower()
            if dir_name_lower in EXCLUDED_DIRECTORIES:
                logger.debug(
                    "Skipping auxiliary duplicate directory '%s' to prevent data leakage.",
                    subdir.name,
                )
                continue

            if dir_name_lower not in DIRECTORY_TO_CLASS_IDX:
                logger.warning(
                    "Skipping unrecognized subdirectory '%s' in '%s'.",
                    subdir.name,
                    self.root_dir,
                )
                continue

            class_idx = DIRECTORY_TO_CLASS_IDX[dir_name_lower]
            if not (0 <= class_idx < self.NUM_CLASSES):
                raise ValueError(
                    f"Mapped class index {class_idx} for '{subdir.name}' is outside [0, {self.NUM_CLASSES - 1}]."
                )

            self.class_to_idx[subdir.name] = class_idx

            for file_path in sorted(subdir.iterdir()):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
                ):
                    collected.append((file_path, class_idx))

        discovered_classes = {label for _, label in collected}
        if len(discovered_classes) != self.NUM_CLASSES:
            logger.warning(
                "Expected %d classes in '%s', found %d classes: %s",
                self.NUM_CLASSES,
                self.root_dir,
                len(discovered_classes),
                sorted(discovered_classes),
            )

        return collected

    @staticmethod
    def _filter_corrupt_samples(
        samples: Sequence[Tuple[Path, int]],
    ) -> List[Tuple[Path, int]]:
        """Verify image files upfront and discard corrupt/unreadable files."""
        valid_samples: List[Tuple[Path, int]] = []
        for img_path, label in samples:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_samples.append((img_path, label))
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
                logger.warning(
                    "Excluding corrupt image during dataset initialization: '%s' (%s)",
                    img_path,
                    exc,
                )
        return valid_samples

    @staticmethod
    def _load_rgb_image(image_path: Path) -> Image.Image:
        """Safely load an image from disk and convert it to 3-channel RGB.

        Args:
            image_path: Path to the image file on disk.

        Returns:
            Decoded PIL.Image.Image in RGB mode.

        Raises:
            CorruptImageError: If the file is missing, truncated, or cannot be decoded.
        """
        try:
            with Image.open(image_path) as img:
                img.load()
                return img.convert("RGB")
        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
        ) as exc:
            raise CorruptImageError(
                f"Failed to load or decode image '{image_path}': {exc}"
            ) from exc

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load, transform, and return `(image_tensor, label)` for `index`.

        If a corrupt image is encountered at runtime, logs a warning and retries
        with a fallback index up to `self.max_corrupt_retries` times.

        Args:
            index: Sample index in `[0, len(self) - 1]`.

        Returns:
            Tuple of `(image_tensor, class_index)` where `image_tensor` has shape
            `(3, 224, 224)` and `class_index` is an integer in `{0, 1, 2, 3}`.

        Raises:
            IndexError: If `index` is out of range.
            CorruptImageError: If `max_corrupt_retries` consecutive samples fail to load.
        """
        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self.samples)}."
            )

        current_idx = index
        last_error: Optional[Exception] = None

        for attempt in range(self.max_corrupt_retries):
            image_path, label = self.samples[current_idx]
            try:
                pil_image = self._load_rgb_image(image_path)
                if self.transform is not None:
                    tensor_image = self.transform(pil_image)
                else:
                    tensor_image = get_val_test_transforms()(pil_image)
                return tensor_image, label
            except CorruptImageError as exc:
                last_error = exc
                logger.warning(
                    "Corrupt image at index %d ('%s'), attempt %d/%d: %s",
                    current_idx,
                    image_path,
                    attempt + 1,
                    self.max_corrupt_retries,
                    exc,
                )
                current_idx = (current_idx + 1) % len(self.samples)

        raise CorruptImageError(
            f"Exceeded maximum retries ({self.max_corrupt_retries}) loading images "
            f"starting from index {index}. Last error: {last_error}"
        )


def stratified_split_samples(
    samples: Sequence[Tuple[Path, int]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Perform a deterministic, stratified split of (path, label) pairs with zero leakage.

    Args:
        samples: Full sequence of `(image_path, class_idx)` pairs.
        train_ratio: Proportion of samples allocated to training.
        val_ratio: Proportion of samples allocated to validation.
        test_ratio: Proportion of samples allocated to testing.
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple of `(train_samples, val_samples, test_samples)`.

    Raises:
        ValueError: If split ratios do not sum to 1.0 or any ratio is non-positive.
    """
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must all be > 0.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}."
        )

    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for item in samples:
        by_class.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    for class_idx in sorted(by_class.keys()):
        class_items = list(by_class[class_idx])
        rng.shuffle(class_items)

        n_total = len(class_items)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # Ensure at least 1 sample in each split when possible
        n_train = max(1, min(n_train, n_total - 2)) if n_total >= 3 else n_train
        n_val = max(1, min(n_val, n_total - n_train - 1)) if n_total >= 3 else n_val

        train_samples.extend(class_items[:n_train])
        val_samples.extend(class_items[n_train : n_train + n_val])
        test_samples.extend(class_items[n_train + n_val :])

    return train_samples, val_samples, test_samples


def create_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 2,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    pin_memory: bool = True,
) -> Dict[str, DataLoader[Tuple[torch.Tensor, int]]]:
    """Create leak-free Train, Validation, and Test DataLoaders for ACNE04.

    Splits file paths BEFORE instantiating `ACNE04Dataset` so that:
    - No image path appears in more than one split.
    - `train` uses `get_train_transforms()` (augmentation + normalization).
    - `val` and `test` use `get_val_test_transforms()` (resize + normalization ONLY).

    Args:
        root_dir: Directory containing the ACNE04 class folders.
        batch_size: Mini-batch size for all loaders.
        num_workers: Number of subprocesses for data loading.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducible stratified splitting.
        pin_memory: Whether to pin CUDA memory in DataLoaders.

    Returns:
        Dictionary mapping `'train'`, `'val'`, and `'test'` to their DataLoaders.
    """
    base_dataset = ACNE04Dataset(root_dir=root_dir, transform=None)
    train_samples, val_samples, test_samples = stratified_split_samples(
        samples=base_dataset.samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_dataset = ACNE04Dataset(
        root_dir=root_dir,
        samples=train_samples,
        transform=get_train_transforms(),
    )
    val_dataset = ACNE04Dataset(
        root_dir=root_dir,
        samples=val_samples,
        transform=get_val_test_transforms(),
    )
    test_dataset = ACNE04Dataset(
        root_dir=root_dir,
        samples=test_samples,
        transform=get_val_test_transforms(),
    )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
