"""Focused tests for the Phase 12 experiment importer."""

from pathlib import Path

import pytest

from src.track_experiments import (
    RUN_DEFINITIONS,
    extract_run_data,
    sqlite_tracking_uri,
    summary_fingerprint,
    tracked_artifacts,
)


def definition(run_name: str):
    """Return one named run definition used by the real importer."""
    return next(item for item in RUN_DEFINITIONS if item.run_name == run_name)


def test_baseline_summary_extracts_comparable_values() -> None:
    payload = {
        "model": "baseline_cnn",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs_requested": 10,
        "best_epoch": 9,
        "class_weights": {"0": 1.9, "1": 0.67},
        "best_validation_metrics": {
            "accuracy": 0.88,
            "loss": 0.23,
            "precision": 0.98,
            "recall": 0.86,
        },
    }

    parameters, metrics = extract_run_data(
        definition("phase4-baseline-cnn"), payload
    )

    assert parameters["batch_size"] == 32
    assert parameters["class_weight_normal"] == 1.9
    assert metrics == {
        "accuracy": 0.88,
        "loss": 0.23,
        "precision": 0.98,
        "recall": 0.86,
    }


def test_test_sensitivity_is_normalized_to_recall() -> None:
    payload = {
        "threshold": 0.5,
        "test_samples": 624,
        "accuracy": 0.79,
        "precision": 0.75,
        "recall_sensitivity": 0.99,
        "specificity": 0.46,
        "f1_score": 0.85,
        "roc_auc": 0.95,
        "true_negatives": 108,
        "false_positives": 126,
        "false_negatives": 3,
        "true_positives": 387,
    }

    parameters, metrics = extract_run_data(
        definition("phase7-held-out-test"), payload
    )

    assert parameters["model"] == "densenet121_advanced_best"
    assert metrics["recall"] == pytest.approx(0.99)
    assert "recall_sensitivity" not in metrics


def test_summary_fingerprint_changes_with_results() -> None:
    run = definition("phase4-baseline-cnn")

    first = summary_fingerprint(run, b'{"accuracy": 0.8}')
    second = summary_fingerprint(run, b'{"accuracy": 0.9}')

    assert first != second
    assert len(first) == 64


def test_only_small_result_artifacts_are_selected(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "curves.png").write_bytes(b"png")
    (tmp_path / "model.keras").write_bytes(b"large model")

    selected = [path.name for path in tracked_artifacts(tmp_path)]

    assert selected == ["curves.png", "metrics.json"]


def test_sqlite_uri_uses_the_requested_database(tmp_path: Path) -> None:
    database = tmp_path / "tracking" / "mlflow.db"

    uri = sqlite_tracking_uri(database)

    assert uri.startswith("sqlite:///")
    assert uri.endswith("/tracking/mlflow.db")
