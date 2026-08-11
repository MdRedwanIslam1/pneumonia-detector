"""Train and fine-tune the Phase 5 DenseNet121 transfer model."""

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

from src.data_loader import DEFAULT_SEED, PROJECT_ROOT, create_datasets
from src.model import (
    build_densenet121_transfer_model,
    compile_binary_classifier,
    enable_densenet_fine_tuning,
)


DEFAULT_BATCH_SIZE = 16
DEFAULT_FROZEN_EPOCHS = 5
DEFAULT_FINE_TUNE_EPOCHS = 5
DEFAULT_FROZEN_LEARNING_RATE = 1e-3
DEFAULT_FINE_TUNE_LEARNING_RATE = 1e-5
DEFAULT_TRAINABLE_TOP_LAYERS = 40
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "densenet121"
BASELINE_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "baseline_cnn" / "run_summary.json"


def configure_tensorflow() -> list[tf.config.PhysicalDevice]:
    """Report GPUs and allow TensorFlow to grow GPU memory as needed."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return gpus


def make_callbacks(checkpoint_path: Path) -> list[keras.callbacks.Callback]:
    """Save the lowest-validation-loss checkpoint for one training stage."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
    ]


def history_frame(
    history: keras.callbacks.History,
    stage: str,
    epoch_offset: int,
) -> pd.DataFrame:
    """Convert one Keras history into a labeled table."""
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", range(epoch_offset + 1, epoch_offset + len(frame) + 1))
    frame.insert(1, "stage", stage)
    return frame


def plot_training_curves(history: pd.DataFrame, stage_boundary: int, output_path: Path) -> None:
    """Plot both transfer-learning stages on shared accuracy and loss charts."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["epoch"], history["accuracy"], marker="o", label="Training")
    axes[0].plot(
        history["epoch"],
        history["val_accuracy"],
        marker="o",
        label="Validation",
    )
    axes[0].set_title("DenseNet121 accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["loss"], marker="o", label="Training")
    axes[1].plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation",
    )
    axes[1].set_title("DenseNet121 loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary cross-entropy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for axis in axes:
        axis.axvline(
            stage_boundary + 0.5,
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label="Fine-tuning starts",
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate(model: keras.Model, validation_dataset: tf.data.Dataset) -> dict[str, float]:
    """Return JSON-friendly validation metrics."""
    metrics = model.evaluate(validation_dataset, return_dict=True, verbose=1)
    return {name: float(value) for name, value in metrics.items()}


def load_baseline_metrics() -> dict[str, float] | None:
    """Load the Phase 4 comparison point when it is available."""
    if not BASELINE_SUMMARY_PATH.is_file():
        return None
    summary = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
    return summary.get("best_validation_metrics")


def parse_args() -> argparse.Namespace:
    """Read the two-stage training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--frozen-epochs", type=int, default=DEFAULT_FROZEN_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS)
    parser.add_argument(
        "--frozen-learning-rate",
        type=float,
        default=DEFAULT_FROZEN_LEARNING_RATE,
    )
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=DEFAULT_FINE_TUNE_LEARNING_RATE,
    )
    parser.add_argument(
        "--trainable-top-layers",
        type=int,
        default=DEFAULT_TRAINABLE_TOP_LAYERS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    """Run frozen-head training, fine-tuning, comparison, and saving."""
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    gpus = configure_tensorflow()

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs detected: {gpus}")
    if not gpus:
        print("WARNING: No GPU detected. DenseNet training will be very slow.")

    print("\nPreparing datasets...")
    datasets = create_datasets(
        batch_size=args.batch_size,
        seed=args.seed,
        apply_training_augmentation=True,
    )
    print(f"Class weights: {datasets.class_weights}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_checkpoint = MODEL_DIR / "densenet121_frozen_best.keras"
    fine_tuned_checkpoint = MODEL_DIR / "densenet121_finetuned_best.keras"
    selected_model_path = MODEL_DIR / "densenet121_best.keras"

    print("\nBuilding DenseNet121 and downloading ImageNet weights if needed...")
    model, _ = build_densenet121_transfer_model(
        learning_rate=args.frozen_learning_rate,
        weights="imagenet",
    )
    model.summary()

    print("\nStage 1: training the new classification head with DenseNet frozen.")
    frozen_history = model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=args.frozen_epochs,
        class_weight=datasets.class_weights,
        callbacks=make_callbacks(frozen_checkpoint),
        verbose=1,
    )

    frozen_model = keras.models.load_model(frozen_checkpoint)
    frozen_metrics = evaluate(frozen_model, datasets.validation)

    print("\nStage 2: fine-tuning the top DenseNet layers.")
    model = keras.models.load_model(frozen_checkpoint)
    base_model = model.get_layer("densenet121")
    trainable_layer_count = enable_densenet_fine_tuning(
        base_model,
        trainable_top_layers=args.trainable_top_layers,
    )
    compile_binary_classifier(model, learning_rate=args.fine_tune_learning_rate)
    print(
        f"Trainable DenseNet layers: {trainable_layer_count} "
        f"of {len(base_model.layers)}"
    )

    fine_tune_history = model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=args.fine_tune_epochs,
        class_weight=datasets.class_weights,
        callbacks=make_callbacks(fine_tuned_checkpoint),
        verbose=1,
    )

    fine_tuned_model = keras.models.load_model(fine_tuned_checkpoint)
    fine_tuned_metrics = evaluate(fine_tuned_model, datasets.validation)

    if fine_tuned_metrics["loss"] <= frozen_metrics["loss"]:
        selected_stage = "fine_tuned"
        selected_model = fine_tuned_model
        selected_metrics = fine_tuned_metrics
    else:
        selected_stage = "frozen"
        selected_model = frozen_model
        selected_metrics = frozen_metrics
    selected_model.save(selected_model_path)

    frozen_frame = history_frame(frozen_history, "frozen", epoch_offset=0)
    fine_tune_frame = history_frame(
        fine_tune_history,
        "fine_tune",
        epoch_offset=len(frozen_frame),
    )
    combined_history = pd.concat([frozen_frame, fine_tune_frame], ignore_index=True)
    history_path = OUTPUT_DIR / "history.csv"
    curves_path = OUTPUT_DIR / "training_curves.png"
    combined_history.to_csv(history_path, index=False)
    plot_training_curves(
        combined_history,
        stage_boundary=len(frozen_frame),
        output_path=curves_path,
    )

    baseline_metrics = load_baseline_metrics()
    accuracy_improvement = None
    if baseline_metrics is not None:
        accuracy_improvement = (
            selected_metrics["accuracy"] - float(baseline_metrics["accuracy"])
        )

    summary = {
        "model": "densenet121_transfer",
        "batch_size": args.batch_size,
        "frozen_epochs": args.frozen_epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "trainable_top_layers_requested": args.trainable_top_layers,
        "trainable_densenet_layers": trainable_layer_count,
        "frozen_validation_metrics": frozen_metrics,
        "fine_tuned_validation_metrics": fine_tuned_metrics,
        "selected_stage": selected_stage,
        "best_validation_metrics": selected_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "validation_accuracy_improvement": accuracy_improvement,
        "class_weights": datasets.class_weights,
        "split_counts": datasets.split_counts,
    }
    summary_path = OUTPUT_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDenseNet121 transfer learning complete.")
    print(f"Frozen-stage validation metrics: {frozen_metrics}")
    print(f"Fine-tuned validation metrics: {fine_tuned_metrics}")
    print(f"Selected stage: {selected_stage}")
    print(f"Selected validation metrics: {selected_metrics}")
    if accuracy_improvement is not None:
        print(
            "Validation accuracy change versus baseline: "
            f"{accuracy_improvement * 100:+.2f} percentage points"
        )
    print(f"Selected model: {selected_model_path}")
    print(f"Training curves: {curves_path}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()
