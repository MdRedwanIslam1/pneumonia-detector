"""Neural-network architectures for the pneumonia detector."""

from __future__ import annotations

from tensorflow import keras

from src.preprocess import IMAGE_SIZE, NUM_CHANNELS


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (*IMAGE_SIZE, NUM_CHANNELS),
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile a small CNN trained entirely from scratch.

    Each convolution block learns increasingly complex visual patterns. Global
    average pooling summarizes each learned feature map without creating the
    very large parameter count that a Flatten layer would produce.
    """
    inputs = keras.Input(shape=input_shape, name="xray_image")
    x = inputs

    for block_number, filters in enumerate((32, 64, 128, 256), start=1):
        x = keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name=f"block_{block_number}_conv",
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=2,
            name=f"block_{block_number}_pool",
        )(x)

    x = keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = keras.layers.Dense(128, activation="relu", name="decision_features")(x)
    x = keras.layers.Dropout(0.40, name="dropout")(x)
    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_probability",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def compile_binary_classifier(
    model: keras.Model,
    learning_rate: float,
) -> None:
    """Compile a binary classifier with metrics shared by both model types."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def build_densenet121_transfer_model(
    input_shape: tuple[int, int, int] = (*IMAGE_SIZE, NUM_CHANNELS),
    learning_rate: float = 1e-3,
    weights: str | None = "imagenet",
) -> tuple[keras.Model, keras.Model]:
    """Build a DenseNet121 feature extractor with a new binary head.

    The input pipeline supplies pixels in the range 0-1. ImageNet-trained
    DenseNet expects each RGB channel to be standardized with ImageNet's mean
    and standard deviation, so that conversion is part of the model.
    """
    inputs = keras.Input(shape=input_shape, name="xray_image")
    x = keras.layers.Normalization(
        mean=(0.485, 0.456, 0.406),
        variance=(0.229**2, 0.224**2, 0.225**2),
        name="imagenet_normalization",
    )(inputs)

    base_model = keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base_model.trainable = False

    # training=False keeps pretrained BatchNormalization statistics stable.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = keras.layers.Dense(128, activation="relu", name="decision_features")(x)
    x = keras.layers.Dropout(0.40, name="dropout")(x)
    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_probability",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="densenet121_transfer")
    compile_binary_classifier(model, learning_rate=learning_rate)
    return model, base_model


def enable_densenet_fine_tuning(
    base_model: keras.Model,
    trainable_top_layers: int = 40,
) -> int:
    """Unfreeze only the top DenseNet layers while keeping BatchNorm frozen."""
    if trainable_top_layers <= 0:
        raise ValueError("trainable_top_layers must be greater than zero")

    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - trainable_top_layers)

    for index, layer in enumerate(base_model.layers):
        should_train = index >= freeze_until
        if isinstance(layer, keras.layers.BatchNormalization):
            should_train = False
        layer.trainable = should_train

    return sum(1 for layer in base_model.layers if layer.trainable)


def main() -> None:
    """Print the baseline architecture and parameter count."""
    model = build_baseline_cnn()
    model.summary()


if __name__ == "__main__":
    main()
