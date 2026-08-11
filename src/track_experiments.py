"""Import completed project results into a local MLflow experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_NAME = "pneumonia-detection"
TRACKED_ARTIFACT_SUFFIXES = {".csv", ".json", ".png"}


@dataclass(frozen=True)
class RunDefinition:
    """Describe how one completed phase maps into an MLflow run."""

    run_name: str
    phase: str
    summary_path: str
    metrics_path: tuple[str, ...]
    metric_aliases: tuple[tuple[str, str], ...]
    parameter_paths: tuple[tuple[str, tuple[str, ...]], ...]
    tags: tuple[tuple[str, str], ...]


RUN_DEFINITIONS = (
    RunDefinition(
        run_name="phase4-baseline-cnn",
        phase="4",
        summary_path="outputs/baseline_cnn/run_summary.json",
        metrics_path=("best_validation_metrics",),
        metric_aliases=(
            ("accuracy", "accuracy"),
            ("loss", "loss"),
            ("precision", "precision"),
            ("recall", "recall"),
        ),
        parameter_paths=(
            ("model", ("model",)),
            ("batch_size", ("batch_size",)),
            ("learning_rate", ("learning_rate",)),
            ("epochs_requested", ("epochs_requested",)),
            ("best_epoch", ("best_epoch",)),
            ("class_weight_normal", ("class_weights", "0")),
            ("class_weight_pneumonia", ("class_weights", "1")),
        ),
        tags=(("dataset_split", "validation"), ("model_family", "custom_cnn")),
    ),
    RunDefinition(
        run_name="phase5-densenet121-transfer",
        phase="5",
        summary_path="outputs/densenet121/run_summary.json",
        metrics_path=("best_validation_metrics",),
        metric_aliases=(
            ("accuracy", "accuracy"),
            ("loss", "loss"),
            ("precision", "precision"),
            ("recall", "recall"),
        ),
        parameter_paths=(
            ("model", ("model",)),
            ("batch_size", ("batch_size",)),
            ("frozen_epochs", ("frozen_epochs",)),
            ("fine_tune_epochs", ("fine_tune_epochs",)),
            ("trainable_densenet_layers", ("trainable_densenet_layers",)),
            ("selected_stage", ("selected_stage",)),
        ),
        tags=(("dataset_split", "validation"), ("model_family", "densenet121")),
    ),
    RunDefinition(
        run_name="phase6-advanced-candidate",
        phase="6",
        summary_path="outputs/advanced_training/run_summary.json",
        metrics_path=("selected_validation_metrics",),
        metric_aliases=(
            ("accuracy", "accuracy"),
            ("loss", "loss"),
            ("precision", "precision"),
            ("recall", "recall"),
        ),
        parameter_paths=(
            ("model", ("model",)),
            ("focal_gamma", ("focal_gamma",)),
            ("focal_alpha_pneumonia", ("focal_alpha_for_pneumonia",)),
            ("max_epochs", ("max_epochs",)),
            ("epochs_completed", ("epochs_completed",)),
            ("selected_source", ("selected_source",)),
        ),
        tags=(("dataset_split", "validation"), ("model_family", "densenet121")),
    ),
    RunDefinition(
        run_name="phase7-held-out-test",
        phase="7",
        summary_path="outputs/evaluation/metrics.json",
        metrics_path=(),
        metric_aliases=(
            ("accuracy", "accuracy"),
            ("precision", "precision"),
            ("recall_sensitivity", "recall"),
            ("specificity", "specificity"),
            ("f1_score", "f1_score"),
            ("roc_auc", "roc_auc"),
            ("true_negatives", "true_negatives"),
            ("false_positives", "false_positives"),
            ("false_negatives", "false_negatives"),
            ("true_positives", "true_positives"),
        ),
        parameter_paths=(
            ("model", ()),
            ("threshold", ("threshold",)),
            ("test_samples", ("test_samples",)),
        ),
        tags=(("dataset_split", "test"), ("model_family", "densenet121")),
    ),
)


def read_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Read a value from nested dictionaries using a tuple of keys."""
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def extract_run_data(
    definition: RunDefinition,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Extract comparable parameters and metrics from one phase summary."""
    parameters: dict[str, Any] = {}
    for parameter_name, path in definition.parameter_paths:
        if not path and parameter_name == "model":
            parameters[parameter_name] = "densenet121_advanced_best"
            continue
        parameters[parameter_name] = read_nested(payload, path)

    metric_payload = read_nested(payload, definition.metrics_path)
    if not isinstance(metric_payload, dict):
        raise TypeError(f"Metrics for {definition.run_name} must be a dictionary.")

    metrics = {
        target_name: float(metric_payload[source_name])
        for source_name, target_name in definition.metric_aliases
    }
    return parameters, metrics


def summary_fingerprint(definition: RunDefinition, summary_bytes: bytes) -> str:
    """Return a stable identifier used to avoid duplicate imports."""
    digest = hashlib.sha256()
    digest.update(definition.run_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(summary_bytes)
    return digest.hexdigest()


def sqlite_tracking_uri(database_path: Path) -> str:
    """Build a cross-platform SQLite URI for MLflow."""
    return f"sqlite:///{database_path.resolve().as_posix()}"


def current_git_commit(project_root: Path) -> str:
    """Read the current commit when Git is available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def ensure_experiment(client: Any, name: str, artifact_directory: Path) -> str:
    """Create the MLflow experiment once and return its identifier."""
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=artifact_directory.as_uri())


def tracked_artifacts(output_directory: Path) -> list[Path]:
    """Return small result files while deliberately excluding model weights."""
    if not output_directory.is_dir():
        return []
    return sorted(
        path
        for path in output_directory.iterdir()
        if path.is_file() and path.suffix.lower() in TRACKED_ARTIFACT_SUFFIXES
    )


def import_completed_runs(
    project_root: Path,
    database_path: Path,
    experiment_name: str,
    force: bool = False,
) -> list[dict[str, str]]:
    """Import available phase summaries and return their resulting statuses."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise RuntimeError(
            "MLflow is not installed. Run: pip install -r requirements-mlops.txt"
        ) from error

    database_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_directory = database_path.parent / "artifacts"
    mlflow.set_tracking_uri(sqlite_tracking_uri(database_path))
    client = MlflowClient()
    experiment_id = ensure_experiment(client, experiment_name, artifact_directory)
    git_commit = current_git_commit(project_root)
    results: list[dict[str, str]] = []

    for definition in RUN_DEFINITIONS:
        summary_path = project_root / definition.summary_path
        if not summary_path.is_file():
            results.append({"run": definition.run_name, "status": "missing"})
            continue

        summary_bytes = summary_path.read_bytes()
        fingerprint = summary_fingerprint(definition, summary_bytes)
        existing = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.import_fingerprint = '{fingerprint}'",
            max_results=1,
        )
        if existing and not force:
            existing_run_id = existing[0].info.run_id
            client.set_tag(existing_run_id, "source_commit", git_commit)
            client.set_tag(existing_run_id, "mlflow.source.git.commit", git_commit)
            results.append(
                {
                    "run": definition.run_name,
                    "status": "already_imported",
                    "run_id": existing_run_id,
                }
            )
            continue

        payload = json.loads(summary_bytes)
        parameters, metrics = extract_run_data(definition, payload)
        tags = {
            "project_phase": definition.phase,
            "import_fingerprint": fingerprint,
            "source_commit": git_commit,
            "mlflow.source.git.commit": git_commit,
            **dict(definition.tags),
        }

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=definition.run_name,
            tags=tags,
        ) as run:
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            for artifact_path in tracked_artifacts(summary_path.parent):
                mlflow.log_artifact(os.fspath(artifact_path), artifact_path="run_outputs")

            results.append(
                {
                    "run": definition.run_name,
                    "status": "imported",
                    "run_id": run.info.run_id,
                }
            )

    return results


def parse_args() -> argparse.Namespace:
    """Read local tracking options from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Import duplicate runs even when the summaries have not changed.",
    )
    return parser.parse_args()


def main() -> None:
    """Import existing runs and print the command for opening the dashboard."""
    args = parse_args()
    project_root = args.project_root.resolve()
    database_path = (
        args.database.resolve()
        if args.database is not None
        else project_root / "mlruns" / "mlflow.db"
    )
    results = import_completed_runs(
        project_root=project_root,
        database_path=database_path,
        experiment_name=args.experiment_name,
        force=args.force,
    )

    print("\nMLflow experiment import complete.")
    for result in results:
        run_id = f" ({result['run_id']})" if "run_id" in result else ""
        print(f"  {result['run']}: {result['status']}{run_id}")
    print(f"Database: {database_path}")
    print("Dashboard: bash scripts/start_mlflow.sh")
    print("Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
