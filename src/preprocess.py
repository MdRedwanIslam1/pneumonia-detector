"""Image preprocessing and augmentation for chest X-rays.

Preprocessing makes every image consistent before it enters a neural network.
Augmentation creates gentle, random variations of training images so the model
learns robust visual patterns instead of memorizing exact files.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3


def decode_and_preprocess_image(image_bytes: tf.Tensor) -> tf.Tensor:
    """Decode image bytes and return the model's normalized image tensor.

    TensorFlow stores ordinary image pixels as integers from 0 to 255. Dividing
    by 255 converts them to floating-point values from 0 to 1, which is easier
    for a neural network to learn from.
    """
    image = tf.io.decode_image(
        image_bytes,
        channels=NUM_CHANNELS,
        expand_animations=False,
    )
    image.set_shape((None, None, NUM_CHANNELS))
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_and_preprocess_image(image_path: tf.Tensor) -> tf.Tensor:
    """Read an image file and apply the shared preprocessing pipeline."""
    image_bytes = tf.io.read_file(image_path)
    return decode_and_preprocess_image(image_bytes)


def preprocess_example(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Convert one file path and label into a model-ready training example."""
    image = load_and_preprocess_image(image_path)
    label = tf.cast(label, tf.float32)
    return image, label


def build_data_augmentation() -> keras.Sequential:
    """Create conservative random transformations for training images only.

    A horizontal flip can be reasonable for this coarse binary task because we
    are not predicting left-versus-right disease location. Vertical flips are
    excluded because upside-down anatomy is not medically realistic.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.03, fill_mode="reflect"),
            keras.layers.RandomZoom(
                height_factor=(-0.08, 0.08),
                width_factor=(-0.08, 0.08),
                fill_mode="reflect",
            ),
            keras.layers.RandomBrightness(0.08, value_range=(0.0, 1.0)),
        ],
        name="training_augmentation",
    )
