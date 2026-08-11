"""Explore and validate the chest X-ray dataset.

This script does not change any X-ray images. It creates an inventory CSV and
a sample-image grid so we can understand the data before training a model.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_exploration"
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def find_images(folder: Path) -> list[Path]:
    """Return supported image files below a folder in a stable order."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_image(path: Path) -> dict[str, object]:
    """Read one image and return metadata or a readable corruption error."""
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            return {
                "width": width,
                "height": height,
                "format": image.format or "UNKNOWN",
                "mode": image.mode,
                "corrupted": False,
                "error": "",
            }
    except (OSError, ValueError, UnidentifiedImageError) as error:
        return {
            "width": None,
            "height": None,
            "format": None,
            "mode": None,
            "corrupted": True,
            "error": str(error),
        }


def build_inventory(dataset_dir: Path) -> pd.DataFrame:
    """Inspect every expected split/class folder and build one metadata table."""
    records: list[dict[str, object]] = []

    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = dataset_dir / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected dataset folder was not found: {class_dir}"
                )

            for image_path in find_images(class_dir):
                records.append(
                    {
                        "path": str(image_path),
                        "split": split,
                        "class_name": class_name,
                        **inspect_image(image_path),
                    }
                )

    return pd.DataFrame.from_records(records)


def print_summary(inventory: pd.DataFrame) -> None:
    """Print counts and quality checks in a beginner-friendly format."""
    counts = (
        inventory.groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SPLITS, columns=CLASSES, fill_value=0)
    )
    counts["TOTAL"] = counts.sum(axis=1)
    counts.loc["TOTAL"] = counts.sum(axis=0)

    print("\nImage counts")
    print(counts.to_string())

    normal_count = int(counts.loc["TOTAL", "NORMAL"])
    pneumonia_count = int(counts.loc["TOTAL", "PNEUMONIA"])
    ratio = pneumonia_count / normal_count if normal_count else float("inf")
    print(
        f"\nOverall class ratio: {ratio:.2f} pneumonia images "
        "for every normal image."
    )
    print(
        "Why it matters: a model can favor the larger class and still appear "
        "accurate, so we will use class weights and medical metrics later."
    )

    corrupted = inventory[inventory["corrupted"]]
    print(f"\nCorrupted or unreadable files: {len(corrupted)}")
    if not corrupted.empty:
        print(corrupted[["path", "error"]].to_string(index=False))

    valid = inventory[~inventory["corrupted"]]
    size_counts = Counter(zip(valid["width"], valid["height"]))
    format_counts = Counter(valid["format"])
    mode_counts = Counter(valid["mode"])

    print(f"\nUnique image sizes: {len(size_counts)}")
    print("Most common sizes:")
    for (width, height), count in size_counts.most_common(10):
        print(f"  {int(width)}x{int(height)}: {count}")

    print(f"\nImage formats: {dict(format_counts)}")
    print(f"Image color modes: {dict(mode_counts)}")
    print(
        "Different image sizes are expected. In Phase 3, every image will be "
        "resized consistently before entering the neural network."
    )


def save_sample_grid(
    inventory: pd.DataFrame,
    output_path: Path,
    samples_per_class: int = 5,
    seed: int = 42,
) -> None:
    """Save a reproducible grid of training examples from both classes."""
    random_generator = random.Random(seed)
    rows: list[tuple[str, Path]] = []

    for class_name in CLASSES:
        candidates = [
            Path(path)
            for path in inventory.loc[
                (inventory["split"] == "train")
                & (inventory["class_name"] == class_name)
                & (~inventory["corrupted"]),
                "path",
            ]
        ]
        chosen = random_generator.sample(
            candidates, min(samples_per_class, len(candidates))
        )
        rows.extend((class_name, path) for path in chosen)

    figure, axes = plt.subplots(
        len(CLASSES),
        samples_per_class,
        figsize=(3 * samples_per_class, 6),
        squeeze=False,
    )

    for row_index, class_name in enumerate(CLASSES):
        class_samples = [path for label, path in rows if label == class_name]
        for column_index in range(samples_per_class):
            axis = axes[row_index][column_index]
            axis.axis("off")
            if column_index >= len(class_samples):
                continue

            image_path = class_samples[column_index]
            with Image.open(image_path) as image:
                axis.imshow(image, cmap="gray")
            axis.set_title(class_name)

    figure.suptitle("Chest X-ray samples from the training split", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Read optional command-line locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset folder (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Generated-report folder (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete Phase 2 dataset exploration."""
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Inspecting dataset: {dataset_dir}")
    inventory = build_inventory(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = output_dir / "image_inventory.csv"
    grid_path = output_dir / "sample_images.png"
    inventory.to_csv(inventory_path, index=False)
    save_sample_grid(inventory, grid_path)
    print_summary(inventory)

    print(f"\nInventory saved to: {inventory_path}")
    print(f"Sample grid saved to: {grid_path}")


if __name__ == "__main__":
    main()
