"""Run rigorous Phase 7 evaluation on the untouched test set."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

from src.data_loader import (
    CLASS_NAMES,
    DEFAULT_DATASET_DIR,
    PROJECT_ROOT,
    collect_labeled_files,
    create_dataset_from_files,
)


ADVANCED_MODEL_PATH = PROJECT_ROOT / "models" / "densenet121_advanced_best.keras"
PHASE5_MODEL_PATH = PROJECT_ROOT / "models" / "densenet121_best.keras"
DEFAULT_MODEL_PATH = (
    ADVANCED_MODEL_PATH if ADVANCED_MODEL_PATH.is_file() else PHASE5_MODEL_PATH
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_BATCH_SIZE = 16
DEFAULT_THRESHOLD = 0.5


def configure_tensorflow() -> list[tf.config.PhysicalDevice]:
    """Report GPUs and avoid reserving all GPU memory immediately."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return gpus


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Calculate medical classification metrics at one fixed threshold."""
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics: dict[str, float | int] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_sensitivity": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "test_samples": int(len(labels)),
    }
    return metrics, predictions


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    """Save a labeled confusion-matrix heatmap."""
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_title("Test-set confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    auc_value: float,
    output_path: Path,
) -> None:
    """Save the ROC curve across all possible probability thresholds."""
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"DenseNet121 (AUC = {auc_value:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Random")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Test-set ROC curve")
    axis.set_xlabel("False-positive rate (1 - specificity)")
    axis.set_ylabel("True-positive rate (sensitivity)")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def build_prediction_table(
    paths: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Create one auditable row for every test image."""
    confidence = np.where(predictions == 1, probabilities, 1.0 - probabilities)
    return pd.DataFrame(
        {
            "path": paths,
            "actual_label": labels,
            "actual_class": [CLASS_NAMES[label] for label in labels],
            "pneumonia_probability": probabilities,
            "predicted_label": predictions,
            "predicted_class": [CLASS_NAMES[label] for label in predictions],
            "confidence": confidence,
            "correct": labels == predictions,
        }
    )


def select_misclassifications(
    predictions: pd.DataFrame,
    examples_per_error_type: int = 6,
) -> pd.DataFrame:
    """Select high-confidence false negatives and false positives for review."""
    false_negatives = predictions[
        (predictions["actual_label"] == 1)
        & (predictions["predicted_label"] == 0)
    ].nsmallest(examples_per_error_type, "pneumonia_probability")

    false_positives = predictions[
        (predictions["actual_label"] == 0)
        & (predictions["predicted_label"] == 1)
    ].nlargest(examples_per_error_type, "pneumonia_probability")

    return pd.concat([false_negatives, false_positives], ignore_index=True)


def save_misclassification_grid(
    examples: pd.DataFrame,
    output_path: Path,
    columns: int = 3,
) -> None:
    """Save selected mistakes for qualitative review."""
    if examples.empty:
        return

    rows = math.ceil(len(examples) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1)

    for axis, (_, example) in zip(axes, examples.iterrows()):
        with Image.open(example["path"]) as image:
            axis.imshow(image, cmap="gray")
        axis.set_title(
            f"Actual: {example['actual_class']}\n"
            f"Predicted: {example['predicted_class']}\n"
            f"P(Pneumonia): {example['pneumonia_probability']:.1%}"
        )
        axis.axis("off")

    for axis in axes[len(examples) :]:
        axis.axis("off")

    figure.suptitle(
        "Selected test-set mistakes: false negatives first, then false positives",
        fontsize=15,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Read model, batch-size, and fixed-threshold options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    """Predict the untouched test set and save all Phase 7 evidence."""
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    model_path = args.model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {model_path}")

    gpus = configure_tensorflow()
    print(f"GPUs detected: {gpus}")
    print(f"Loading final selected model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    test_paths, test_labels = collect_labeled_files(
        DEFAULT_DATASET_DIR,
        source_splits=("test",),
    )
    test_dataset = create_dataset_from_files(
        test_paths,
        test_labels,
        batch_size=args.batch_size,
        shuffle=False,
        apply_augmentation=False,
    )
    image_dataset = test_dataset.map(
        lambda images, labels: images,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    probabilities = model.predict(image_dataset, verbose=1).reshape(-1)
    labels = np.asarray(test_labels, dtype=np.int32)
    if len(probabilities) != len(labels):
        raise RuntimeError("Prediction count does not match the test-set size")

    metrics, predictions = calculate_metrics(
        labels,
        probabilities,
        threshold=args.threshold,
    )
    prediction_table = build_prediction_table(
        test_paths,
        labels,
        probabilities,
        predictions,
    )
    mistakes = select_misclassifications(prediction_table)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics.json"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    mistakes_path = OUTPUT_DIR / "selected_misclassifications.csv"
    confusion_path = OUTPUT_DIR / "confusion_matrix.png"
    roc_path = OUTPUT_DIR / "roc_curve.png"
    mistake_grid_path = OUTPUT_DIR / "misclassified_examples.png"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    prediction_table.to_csv(predictions_path, index=False)
    mistakes.to_csv(mistakes_path, index=False)
    save_confusion_matrix(labels, predictions, confusion_path)
    save_roc_curve(labels, probabilities, float(metrics["roc_auc"]), roc_path)
    save_misclassification_grid(mistakes, mistake_grid_path)

    print("\nPhase 7 test evaluation complete.")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.6f}")
        else:
            print(f"  {metric_name}: {value}")
    print(f"\nMetrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"ROC curve: {roc_path}")
    print(f"Misclassification grid: {mistake_grid_path}")


if __name__ == "__main__":
    main()
