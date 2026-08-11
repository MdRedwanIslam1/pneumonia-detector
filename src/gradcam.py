"""Generate Grad-CAM explanations for DenseNet121 predictions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
ADVANCED_MODEL_PATH = PROJECT_ROOT / "models" / "densenet121_advanced_best.keras"
PHASE5_MODEL_PATH = PROJECT_ROOT / "models" / "densenet121_best.keras"
DEFAULT_MODEL_PATH = (
    ADVANCED_MODEL_PATH if ADVANCED_MODEL_PATH.is_file() else PHASE5_MODEL_PATH
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"


def compute_gradcam(
    model: keras.Model,
    image: tf.Tensor,
) -> tuple[np.ndarray, float, int]:
    """Return a heatmap for the model's predicted binary class.

    DenseNet's final feature maps retain rough spatial information. Grad-CAM
    weights those maps by how strongly each one changes the selected score.
    """
    if image.ndim == 3:
        image = tf.expand_dims(image, axis=0)

    normalization = model.get_layer("imagenet_normalization")
    base_model = model.get_layer("densenet121")
    pooling = model.get_layer("global_average_pool")
    dense = model.get_layer("decision_features")
    dropout = model.get_layer("dropout")
    output_layer = model.get_layer("pneumonia_probability")

    with tf.GradientTape() as tape:
        normalized = normalization(image)
        feature_maps = base_model(normalized, training=False)
        tape.watch(feature_maps)
        x = pooling(feature_maps)
        x = dense(x)
        x = dropout(x, training=False)
        probability = output_layer(x)[0, 0]
        predicted_label = tf.cast(probability >= 0.5, tf.int32)
        selected_score = tf.where(predicted_label == 1, probability, 1.0 - probability)

    gradients = tape.gradient(selected_score, feature_maps)
    if gradients is None:
        raise RuntimeError("Grad-CAM could not calculate feature-map gradients")

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0.0, heatmap / maximum, heatmap)

    return heatmap.numpy(), float(probability.numpy()), int(predicted_label.numpy())


def make_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    """Resize and blend a color heatmap over a normalized RGB image."""
    resized_heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        image.shape[:2],
        method="bilinear",
    ).numpy()[..., 0]
    colored_heatmap = plt.get_cmap("turbo")(resized_heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colored_heatmap, 0.0, 1.0)


def collect_validation_examples() -> dict[int, tf.Tensor]:
    """Collect one Normal and one Pneumonia image without touching the test set."""
    # Training-data helpers are imported only by this visualization script.
    # API inference can therefore use Grad-CAM without installing scikit-learn.
    from src.data_loader import create_datasets

    datasets = create_datasets(apply_training_augmentation=False)
    examples: dict[int, tf.Tensor] = {}

    for images, labels in datasets.validation:
        for image, label in zip(images, labels):
            label_value = int(label.numpy())
            if label_value not in examples:
                examples[label_value] = image
            if len(examples) == len(CLASS_NAMES):
                return examples

    raise RuntimeError("Could not find both classes in the validation dataset")


def save_gradcam_grid(
    model: keras.Model,
    examples: dict[int, tf.Tensor],
    output_path: Path,
) -> None:
    """Save original, heatmap, and overlay panels for both classes."""
    figure, axes = plt.subplots(len(CLASS_NAMES), 3, figsize=(11, 7))

    for row, actual_label in enumerate(range(len(CLASS_NAMES))):
        image_tensor = examples[actual_label]
        image = image_tensor.numpy()
        heatmap, probability, predicted_label = compute_gradcam(model, image_tensor)
        overlay = make_overlay(image, heatmap)
        confidence = probability if predicted_label == 1 else 1.0 - probability

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Actual: {CLASS_NAMES[actual_label]}")

        axes[row, 1].imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title("Grad-CAM heatmap")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"Predicted: {CLASS_NAMES[predicted_label]}\nConfidence: {confidence:.1%}"
        )

        for column in range(3):
            axes[row, column].axis("off")

    figure.suptitle(
        "Grad-CAM shows model influence, not a medical diagnosis",
        fontsize=15,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Read an optional model checkpoint path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    """Load the selected model and create two validation explanations."""
    args = parse_args()
    model_path = args.model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    examples = collect_validation_examples()
    output_path = OUTPUT_DIR / "gradcam_examples.png"
    save_gradcam_grid(model, examples, output_path)
    print(f"Grad-CAM examples saved to: {output_path}")
    print("Reminder: Grad-CAM is explanatory evidence, not clinical validation.")


if __name__ == "__main__":
    main()
