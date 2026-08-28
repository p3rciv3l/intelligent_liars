#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
bash scripts/bootstrap_truth_editing_prerequisite_worker.sh
python docker/truth-editing/validate_runtime.py
python - <<'PY'
import importlib.metadata

expected = {
    "boto3": "1.40.18",
    "botocore": "1.40.18",
    "optuna": "4.9.0",
    "transformers": "4.57.1",
    "wandb": "0.29.0",
}
for package, version in expected.items():
    actual = importlib.metadata.version(package)
    if actual != version:
        raise SystemExit(f"{package} version differs: {actual} != {version}")
PY
