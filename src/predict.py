"""Reusable single-image prediction helpers and command-line interface."""

from __future__ import annotations

import argparse
import base64
import io
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from src.gradcam import compute_gradcam, make_overlay
from src.preprocess import IMAGE_SIZE, NUM_CHANNELS, decode_and_preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADVANCED_MODEL_PATH = PROJECT_ROOT / "models" / "densenet121_advanced_best.keras"
PHASE5_MODEL_PATH = PROJECT_ROOT / "models" / "densenet121_best.keras"
DEFAULT_MODEL_PATH = (
    ADVANCED_MODEL_PATH if ADVANCED_MODEL_PATH.is_file() else PHASE5_MODEL_PATH
)
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
DEFAULT_THRESHOLD = 0.5


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot become a valid model input."""


def load_prediction_model(model_path: Path = DEFAULT_MODEL_PATH) -> keras.Model:
    """Load a trained model without restoring training-only configuration."""
    resolved_path = model_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {resolved_path}")

    model = keras.models.load_model(resolved_path, compile=False)

    # One empty pass prepares TensorFlow before the first real API request.
    model(
        tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS)),
        training=False,
    )
    return model


def prepare_image_bytes(image_bytes: bytes) -> tf.Tensor:
    """Validate and convert uploaded bytes into one model-ready batch."""
    if not image_bytes:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            image_format = uploaded_image.format
            uploaded_image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError(
            "The file could not be decoded as a supported image."
        ) from error

    if image_format not in {"JPEG", "PNG"}:
        raise InvalidImageError("The image's actual format must be JPEG or PNG.")

    try:
        image = decode_and_preprocess_image(tf.convert_to_tensor(image_bytes))
    except (tf.errors.InvalidArgumentError, ValueError) as error:
        raise InvalidImageError(
            "The file could not be decoded as a supported image."
        ) from error

    if image.shape != (IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS):
        raise InvalidImageError("The decoded image has an unexpected shape.")
    if not bool(tf.reduce_all(tf.math.is_finite(image)).numpy()):
        raise InvalidImageError("The decoded image contains invalid pixel values.")

    return tf.expand_dims(image, axis=0)


def predict_image(
    model: keras.Model,
    image_batch: tf.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | str]:
    """Return the class and probabilities for one preprocessed image."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("The decision threshold must be between 0 and 1.")

    output = model(image_batch, training=False)
    probability = float(tf.reshape(output, (-1,))[0].numpy())
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("The model returned an invalid probability.")

    predicted_label = int(probability >= threshold)
    confidence = probability if predicted_label == 1 else 1.0 - probability
    return {
        "predicted_class": CLASS_NAMES[predicted_label],
        "confidence": confidence,
        "pneumonia_probability": probability,
        "threshold": threshold,
    }


def create_gradcam_data_url(model: keras.Model, image_batch: tf.Tensor) -> str:
    """Create a browser-ready base64 PNG containing the Grad-CAM overlay."""
    heatmap, _, _ = compute_gradcam(model, image_batch[0])
    image = image_batch[0].numpy()
    overlay = make_overlay(image, heatmap)
    overlay_uint8 = np.rint(overlay * 255.0).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(overlay_uint8, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_args() -> argparse.Namespace:
    """Read a local image path and optional inference settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    """Run one prediction from the terminal."""
    args = parse_args()
    image_path = args.image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image was not found: {image_path}")

    model = load_prediction_model(args.model_path)
    image_batch = prepare_image_bytes(image_path.read_bytes())
    result = predict_image(model, image_batch, args.threshold)

    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Pneumonia probability: {result['pneumonia_probability']:.2%}")
    print("Educational use only. This is not a medical diagnosis.")


if __name__ == "__main__":
    main()
