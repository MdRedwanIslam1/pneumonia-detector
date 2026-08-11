"""Build reusable TensorFlow datasets for training and evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.preprocess import build_data_augmentation, preprocess_example


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CLASS_TO_LABEL = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class DatasetBundle:
    """Keep the datasets and their supporting information together."""

    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    class_weights: dict[int, float]
    split_counts: dict[str, dict[str, int]]


def _find_class_images(class_dir: Path) -> list[Path]:
    """Return supported image files from exactly one class folder."""
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Expected class folder was not found: {class_dir}")

    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _collect_files(
    dataset_dir: Path,
    source_splits: tuple[str, ...],
) -> tuple[list[str], list[int]]:
    """Collect file paths and integer labels from selected source folders."""
    paths: list[str] = []
    labels: list[int] = []

    for split_name in source_splits:
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = dataset_dir / split_name / class_name
            class_images = _find_class_images(class_dir)
            paths.extend(str(path) for path in class_images)
            labels.extend([label] * len(class_images))

    if not paths:
        raise ValueError(f"No images were found below {dataset_dir}")

    return paths, labels


def collect_labeled_files(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    source_splits: tuple[str, ...] = ("test",),
) -> tuple[list[str], list[int]]:
    """Public helper returning deterministic file paths and integer labels."""
    return _collect_files(Path(dataset_dir).resolve(), source_splits)


def create_dataset_from_files(
    paths: list[str],
    labels: list[int],
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    seed: int = DEFAULT_SEED,
    apply_augmentation: bool = False,
) -> tf.data.Dataset:
    """Public helper for evaluation or inference on a known file list."""
    if len(paths) != len(labels):
        raise ValueError("paths and labels must contain the same number of items")
    return _make_tf_dataset(
        paths,
        labels,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        apply_augmentation=apply_augmentation,
    )


def _make_tf_dataset(
    paths: list[str],
    labels: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    apply_augmentation: bool,
) -> tf.data.Dataset:
    """Turn file paths into a batched, prefetched TensorFlow pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(preprocess_example, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if apply_augmentation:
        augmenter = build_data_augmentation()

        def augment_batch(
            images: tf.Tensor,
            batch_labels: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor]:
            augmented = augmenter(images, training=True)
            return tf.clip_by_value(augmented, 0.0, 1.0), batch_labels

        dataset = dataset.map(augment_batch, num_parallel_calls=AUTOTUNE)

    return dataset.prefetch(AUTOTUNE)


def _count_labels(labels: list[int]) -> dict[str, int]:
    """Return readable per-class counts for one split."""
    counts = Counter(labels)
    return {
        class_name: counts[label]
        for class_name, label in CLASS_TO_LABEL.items()
    }


def _compute_class_weights(labels: list[int]) -> dict[int, float]:
    """Give the smaller class more influence during model training."""
    label_array = np.asarray(labels, dtype=np.int32)
    class_labels = np.asarray(sorted(set(labels)), dtype=np.int32)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=label_array,
    )
    return {
        int(class_label): float(weight)
        for class_label, weight in zip(class_labels, weights)
    }


def create_datasets(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
    apply_training_augmentation: bool = True,
) -> DatasetBundle:
    """Create train, validation, and untouched test datasets.

    The original validation folder has only 16 images. We combine it with the
    original training folder and create a larger stratified validation split.
    The original test folder is never included in this process.
    """
    dataset_dir = Path(dataset_dir).resolve()
    development_paths, development_labels = _collect_files(
        dataset_dir,
        source_splits=("train", "val"),
    )
    test_paths, test_labels = _collect_files(
        dataset_dir,
        source_splits=("test",),
    )

    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        development_paths,
        development_labels,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=development_labels,
    )

    train_dataset = _make_tf_dataset(
        train_paths,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        apply_augmentation=apply_training_augmentation,
    )
    validation_dataset = _make_tf_dataset(
        validation_paths,
        validation_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        apply_augmentation=False,
    )
    test_dataset = _make_tf_dataset(
        test_paths,
        test_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        apply_augmentation=False,
    )

    split_counts = {
        "train": _count_labels(train_labels),
        "validation": _count_labels(validation_labels),
        "test": _count_labels(test_labels),
    }

    return DatasetBundle(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
        class_weights=_compute_class_weights(train_labels),
        split_counts=split_counts,
    )


def main() -> None:
    """Print the deterministic split summary as a quick command-line check."""
    bundle = create_datasets(apply_training_augmentation=False)
    print("Dataset split counts:")
    for split_name, counts in bundle.split_counts.items():
        print(f"  {split_name}: {counts} (total={sum(counts.values())})")
    print(f"Class weights: {bundle.class_weights}")


if __name__ == "__main__":
    main()
