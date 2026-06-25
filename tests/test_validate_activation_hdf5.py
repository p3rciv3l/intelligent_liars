"""Tests for the activation HDF5 validation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "validate_activation_hdf5.py"
)
SPEC = importlib.util.spec_from_file_location("validate_activation_hdf5", SCRIPT_PATH)
if SPEC is None:
    raise RuntimeError(f"could not load validator script at {SCRIPT_PATH}")
validate_module = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError(f"validator script has no loader: {SCRIPT_PATH}")
SPEC.loader.exec_module(validate_module)


def _write_hdf5(
    path: Path,
    *,
    duplicate_key: bool = False,
    missing_metadata_dataset: str | None = None,
    hidden_dim_by_layer_task: dict[tuple[str, str], int] | None = None,
    non_finite: bool = False,
) -> None:
    tasks = ("task_a", "task_b")
    task_rows = {"task_a": 3, "task_b": 2}
    example_splits_by_task = {
        "task_a": np.array([0, 2, 3], dtype=np.int64),
        "task_b": np.array([0, 1, 2], dtype=np.int64),
    }
    hidden_dim_by_layer_task = hidden_dim_by_layer_task or {}
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = validate_module.EXPECTED_FORMAT
        metadata = h5.create_group("metadata")
        for task_name in tasks:
            rows = task_rows[task_name]
            task_group = metadata.create_group(task_name)
            example_splits = example_splits_by_task[task_name]
            example_count = len(example_splits) - 1
            source_indices = np.repeat(
                np.arange(example_count, dtype=np.int64),
                np.diff(example_splits),
            )
            output_indices = source_indices + 10
            example_source_indices = np.arange(example_count, dtype=np.int64)
            example_output_indices = example_source_indices + 10
            if duplicate_key and task_name == "task_a":
                example_source_indices[1] = example_source_indices[0]
                example_output_indices[1] = example_output_indices[0]
            labels = np.array([-1, 0, 1], dtype=np.int8)[:rows]
            if task_name == "task_b":
                labels = np.array([0, 1], dtype=np.int8)
            datasets: dict[str, Any] = {
                "source_indices": source_indices,
                "output_indices": output_indices,
                "labels": labels,
                "example_splits": example_splits,
                "example_indices": source_indices,
                "token_positions": np.arange(rows, dtype=np.int64) + 200,
                "logit_positions": np.array([], dtype=np.int64),
                "example_labels": np.array([-1, 1], dtype=np.int8)[:example_count],
                "example_source_indices": example_source_indices,
                "example_output_indices": example_output_indices,
            }
            for dataset_name, data in datasets.items():
                if dataset_name == missing_metadata_dataset and task_name == "task_a":
                    continue
                task_group.create_dataset(dataset_name, data=data)

        layers = h5.create_group("layers")
        for layer_name in ("layer_0", "layer_1"):
            layer_group = layers.create_group(layer_name)
            for task_name in tasks:
                rows = task_rows[task_name]
                hidden_dim = hidden_dim_by_layer_task.get((layer_name, task_name), 4)
                data = np.ones((rows, hidden_dim), dtype=np.float32)
                if non_finite and layer_name == "layer_0" and task_name == "task_a":
                    data[0, 0] = np.nan
                layer_group.create_dataset(task_name, data=data)


def _validate(path: Path, **kwargs: Any) -> dict[str, Any]:
    return validate_module.validate_activation_hdf5(
        path,
        expected_tasks=["task_a", "task_b"],
        expected_task_counts={"task_a": 3, "task_b": 2},
        expected_example_counts={"task_a": 2, "task_b": 2},
        expected_layer_count=2,
        expected_hidden_dim=4,
        **kwargs,
    )


def test_validate_activation_hdf5_success(tmp_path: Path) -> None:
    path = tmp_path / "activations.h5"
    _write_hdf5(path)

    result = _validate(path)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["hidden_dim"] == 4
    assert result["task_rows"] == {"task_a": 3, "task_b": 2}
    assert result["example_counts"] == {"task_a": 2, "task_b": 2}
    assert result["binary_usable_counts"] == {"task_a": 2, "task_b": 2}


def test_validate_activation_hdf5_duplicate_key_failure(tmp_path: Path) -> None:
    path = tmp_path / "activations.h5"
    _write_hdf5(path, duplicate_key=True)

    result = _validate(path)

    assert result["ok"] is False
    assert any("duplicate" in error for error in result["errors"])


def test_validate_activation_hdf5_missing_metadata_failure(tmp_path: Path) -> None:
    path = tmp_path / "activations.h5"
    _write_hdf5(path, missing_metadata_dataset="labels")

    result = _validate(path)

    assert result["ok"] is False
    assert any("missing metadata datasets" in error for error in result["errors"])


def test_validate_activation_hdf5_hidden_dim_mismatch_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "activations.h5"
    _write_hdf5(path, hidden_dim_by_layer_task={("layer_1", "task_b"): 5})

    result = _validate(path)

    assert result["ok"] is False
    assert any("hidden dim" in error for error in result["errors"])


def test_validate_activation_hdf5_non_finite_sample_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "activations.h5"
    _write_hdf5(path, non_finite=True)

    result = _validate(path, finite_check="sample")

    assert result["ok"] is False
    assert any("non-finite" in error for error in result["errors"])


def test_validate_activation_hdf5_non_finite_full_failure(tmp_path: Path) -> None:
    path = tmp_path / "activations.h5"
    _write_hdf5(path, non_finite=True)

    result = _validate(path, finite_check="full")

    assert result["ok"] is False
    assert any("non-finite" in error for error in result["errors"])
