"""Refine the selected DenseNet model with focal loss and safer callbacks."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data_loader import DEFAULT_SEED, PROJECT_ROOT, create_datasets


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "densenet121_best.keras"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "advanced_training"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_EPOCHS = 12
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_GAMMA = 2.0
DEFAULT_ALPHA = 0.25


class LearningRateLogger(keras.callbacks.Callback):
    """Add the current learning rate to every epoch's history."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is not None:
            logs["learning_rate"] = float(
                keras.backend.get_value(self.model.optimizer.learning_rate)
            )


def configure_tensorflow() -> list[tf.config.PhysicalDevice]:
    """Report GPUs and let TensorFlow allocate GPU memory gradually."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return gpus


def compile_with_focal_loss(
    model: keras.Model,
    learning_rate: float,
    gamma: float,
    alpha: float,
) -> None:
    """Compile with focal loss and explicit class balancing.

    Pneumonia is label 1 and receives alpha=0.25. Normal is label 0 and
    receives 1-alpha=0.75, roughly matching its smaller representation.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryFocalCrossentropy(
            gamma=gamma,
            apply_class_balancing=True,
            alpha=alpha,
        ),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def compile_for_comparison(model: keras.Model) -> None:
    """Use ordinary BCE so old and new checkpoints are compared fairly."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def evaluate(model: keras.Model, dataset: tf.data.Dataset) -> dict[str, float]:
    """Return ordinary-BCE validation metrics as JSON-friendly numbers."""
    values = model.evaluate(dataset, return_dict=True, verbose=1)
    return {name: float(value) for name, value in values.items()}


def plot_history(history: pd.DataFrame, output_path) -> None:
    """Plot accuracy, focal loss, and learning-rate changes."""
    epochs = history.index + 1
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(epochs, history["accuracy"], marker="o", label="Training")
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    axes[0].set_title("Advanced accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], marker="o", label="Training")
    axes[1].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[1].set_title("Focal loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(epochs, history["learning_rate"], marker="o")
    axes[2].set_title("Learning-rate schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Read advanced-training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Train an advanced candidate, compare it fairly, and retain the winner."""
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    gpus = configure_tensorflow()
    print(f"GPUs detected: {gpus}")

    model_path = PROJECT_ROOT / args.model_path if not os.path.isabs(args.model_path) else args.model_path
    model_path = os.fspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Selected Phase 5 model was not found: {model_path}")

    datasets = create_datasets(
        batch_size=args.batch_size,
        seed=args.seed,
        apply_training_augmentation=True,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = MODEL_DIR / "densenet121_advanced_candidate.keras"
    selected_path = MODEL_DIR / "densenet121_advanced_best.keras"

    original_model = keras.models.load_model(model_path, compile=False)
    compile_for_comparison(original_model)
    original_metrics = evaluate(original_model, datasets.validation)

    candidate_model = keras.models.load_model(model_path, compile=False)
    candidate_model.get_layer("densenet121").trainable = False
    compile_with_focal_loss(
        candidate_model,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        alpha=args.alpha,
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            candidate_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
        LearningRateLogger(),
    ]

    print("\nTraining the focal-loss candidate. Class weights are handled by the loss.")
    history = candidate_model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=args.max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    candidate_model = keras.models.load_model(candidate_path, compile=False)
    compile_for_comparison(candidate_model)
    candidate_metrics = evaluate(candidate_model, datasets.validation)

    if candidate_metrics["loss"] < original_metrics["loss"]:
        selected_source = "advanced_focal"
        selected_model = candidate_model
        selected_metrics = candidate_metrics
    else:
        selected_source = "phase5_frozen"
        selected_model = original_model
        selected_metrics = original_metrics
    selected_model.save(selected_path)

    history_frame = pd.DataFrame(history.history)
    history_path = OUTPUT_DIR / "history.csv"
    curves_path = OUTPUT_DIR / "training_curves.png"
    history_frame.to_csv(history_path, index_label="epoch")
    plot_history(history_frame, curves_path)

    summary = {
        "model": "densenet121_advanced",
        "focal_gamma": args.gamma,
        "focal_alpha_for_pneumonia": args.alpha,
        "focal_weight_for_normal": 1.0 - args.alpha,
        "max_epochs": args.max_epochs,
        "epochs_completed": len(history_frame),
        "original_validation_metrics": original_metrics,
        "candidate_validation_metrics": candidate_metrics,
        "selected_source": selected_source,
        "selected_validation_metrics": selected_metrics,
    }
    summary_path = OUTPUT_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAdvanced training complete.")
    print(f"Original validation metrics: {original_metrics}")
    print(f"Focal candidate validation metrics: {candidate_metrics}")
    print(f"Selected source: {selected_source}")
    print(f"Selected validation metrics: {selected_metrics}")
    print(f"Selected model: {selected_path}")
    print(f"Training curves: {curves_path}")


if __name__ == "__main__":
    main()
