"""Verify Phase 3 and save a visual preview of data augmentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_loader import CLASS_NAMES, PROJECT_ROOT, create_datasets
from src.preprocess import IMAGE_SIZE, NUM_CHANNELS, build_data_augmentation


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessing"


def save_augmentation_preview(
    images: tf.Tensor,
    labels: tf.Tensor,
    output_path: Path,
    sample_count: int = 5,
) -> None:
    """Save original images above their randomly augmented versions."""
    sample_count = min(sample_count, int(images.shape[0]))
    original_images = images[:sample_count]
    original_labels = labels[:sample_count]
    augmented_images = build_data_augmentation()(original_images, training=True)

    figure, axes = plt.subplots(2, sample_count, figsize=(3 * sample_count, 6))
    for index in range(sample_count):
        class_name = CLASS_NAMES[int(original_labels[index].numpy())]

        axes[0, index].imshow(original_images[index].numpy())
        axes[0, index].set_title(f"Original: {class_name}")
        axes[0, index].axis("off")

        axes[1, index].imshow(
            tf.clip_by_value(augmented_images[index], 0.0, 1.0).numpy()
        )
        axes[1, index].set_title("Augmented")
        axes[1, index].axis("off")

    figure.suptitle("Phase 3: conservative training augmentation", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Check dimensions, normalization, splits, weights, and augmentation."""
    bundle = create_datasets(apply_training_augmentation=False)
    images, labels = next(iter(bundle.train))

    expected_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS)
    actual_shape = tuple(images.shape[1:])
    pixel_min = float(tf.reduce_min(images).numpy())
    pixel_max = float(tf.reduce_max(images).numpy())

    assert actual_shape == expected_shape, (
        f"Expected image shape {expected_shape}, received {actual_shape}"
    )
    assert images.dtype == tf.float32, f"Expected float32, received {images.dtype}"
    assert 0.0 <= pixel_min <= pixel_max <= 1.0, (
        f"Expected pixels in [0, 1], received [{pixel_min}, {pixel_max}]"
    )

    preview_path = OUTPUT_DIR / "augmentation_preview.png"
    save_augmentation_preview(images, labels, preview_path)

    print("Phase 3 preprocessing verification passed.")
    print(f"Batch shape: {tuple(images.shape)}")
    print(f"Image dtype: {images.dtype.name}")
    print(f"Pixel range: {pixel_min:.4f} to {pixel_max:.4f}")
    print("\nDataset split counts:")
    for split_name, counts in bundle.split_counts.items():
        print(f"  {split_name}: {counts} (total={sum(counts.values())})")
    print(f"\nClass weights: {bundle.class_weights}")
    print(f"Augmentation preview saved to: {preview_path}")


if __name__ == "__main__":
    main()
