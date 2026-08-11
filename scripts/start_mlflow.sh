#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

# Reuse the environment and GPU-library setup used by every training phase.
source "${PROJECT_ROOT}/scripts/activate_wsl.sh" >/dev/null

cd "${PROJECT_ROOT}"
exec mlflow server \
    --backend-store-uri "sqlite:///${PROJECT_ROOT}/mlruns/mlflow.db" \
    --default-artifact-root "${PROJECT_ROOT}/mlruns/artifacts" \
    --host 0.0.0.0 \
    --port "${MLFLOW_PORT}"
