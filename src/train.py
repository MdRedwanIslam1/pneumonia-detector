"""Train the Phase 4 baseline CNN and save its learning curves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data_loader import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    PROJECT_ROOT,
    create_datasets,
)
from src.model import build_baseline_cnn


DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_cnn"


def configure_tensorflow() -> list[tf.config.PhysicalDevice]:
    """Report GPUs and avoid reserving all GPU memory at startup."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Memory growth cannot be changed after a GPU has been initialized.
            pass
    return gpus


def plot_training_curves(history: keras.callbacks.History, output_path: Path) -> None:
    """Save training-versus-validation accuracy and loss charts."""
    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, metrics["accuracy"], marker="o", label="Training")
    axes[0].plot(epochs, metrics["val_accuracy"], marker="o", label="Validation")
    axes[0].set_title("Baseline CNN accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, metrics["loss"], marker="o", label="Training")
    axes[1].plot(epochs, metrics["val_loss"], marker="o", label="Validation")
    axes[1].set_title("Baseline CNN loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary cross-entropy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Read beginner-friendly training options from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Run the complete baseline training workflow."""
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    gpus = configure_tensorflow()

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs detected: {gpus}")
    if not gpus:
        print("WARNING: No GPU detected. Training will use the CPU and be slower.")

    print("\nPreparing datasets...")
    datasets = create_datasets(
        batch_size=args.batch_size,
        seed=args.seed,
        apply_training_augmentation=True,
    )
    for split_name, counts in datasets.split_counts.items():
        print(f"  {split_name}: {counts} (total={sum(counts.values())})")
    print(f"Class weights: {datasets.class_weights}")

    model = build_baseline_cnn(learning_rate=args.learning_rate)
    print("\nBaseline architecture:")
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_DIR / "baseline_cnn_best.keras"
    history_path = OUTPUT_DIR / "history.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(history_path),
    ]

    print("\nStarting baseline training...")
    history = model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=args.epochs,
        class_weight=datasets.class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    curves_path = OUTPUT_DIR / "training_curves.png"
    plot_training_curves(history, curves_path)

    best_model = keras.models.load_model(best_model_path)
    validation_metrics = best_model.evaluate(
        datasets.validation,
        return_dict=True,
        verbose=1,
    )
    validation_metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in validation_metrics.items()
    }
    best_epoch = min(
        range(len(history.history["val_loss"])),
        key=history.history["val_loss"].__getitem__,
    ) + 1

    summary = {
        "model": "baseline_cnn",
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "best_validation_metrics": validation_metrics,
        "class_weights": datasets.class_weights,
        "split_counts": datasets.split_counts,
    }
    summary_path = OUTPUT_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Re-save the CSV through pandas so numeric columns have a clean table format.
    pd.DataFrame(history.history).to_csv(history_path, index_label="epoch")

    print("\nBaseline training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation metrics: {validation_metrics}")
    print(f"Best model: {best_model_path}")
    print(f"Training history: {history_path}")
    print(f"Training curves: {curves_path}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()
