from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


def _load_merge_plan_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "plan_activation_hdf5_merge.py"
    spec = importlib.util.spec_from_file_location("plan_activation_hdf5_merge_for_tests", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_hdf5(path: Path, tasks: list[str]) -> None:
    with h5py.File(path, "w") as h5:
        metadata = h5.create_group("metadata")
        for task in tasks:
            metadata.create_group(task)


def _write_valid_activation_hdf5(
    path: Path,
    tasks: list[str],
    *,
    layer_count: int = 2,
    hidden_dim: int = 4,
) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "qwen_answer_token_activations_v2"
        metadata = h5.create_group("metadata")
        for task in tasks:
            task_group = metadata.create_group(task)
            task_group.create_dataset("source_indices", data=np.asarray([0, 1], dtype=np.int64))
            task_group.create_dataset("output_indices", data=np.asarray([0, 0], dtype=np.int64))
            task_group.create_dataset("labels", data=np.asarray([0, 1], dtype=np.int8))
            task_group.create_dataset("example_splits", data=np.asarray([0, 1, 2], dtype=np.int64))
            task_group.create_dataset("example_indices", data=np.asarray([0, 1], dtype=np.int64))
            task_group.create_dataset("token_positions", data=np.asarray([3, 4], dtype=np.int64))
            task_group.create_dataset("logit_positions", data=np.asarray([2, 3], dtype=np.int64))
            task_group.create_dataset("example_labels", data=np.asarray([0, 1], dtype=np.int8))
            task_group.create_dataset("example_source_indices", data=np.asarray([0, 1], dtype=np.int64))
            task_group.create_dataset("example_output_indices", data=np.asarray([0, 0], dtype=np.int64))
            for layer in range(layer_count):
                layer_group = h5.require_group(f"layer_{layer}")
                layer_group.create_dataset(
                    task,
                    data=np.ones((2, hidden_dim), dtype=np.float16),
                )


def test_plan_merge_accepts_expected_distinct_tasks(tmp_path):
    merge_plan = _load_merge_plan_module()
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_hdf5(first, ["task_a", "task_b"])
    _write_hdf5(second, ["task_c"])

    report = merge_plan.plan_merge([first, second], expected_tasks=["task_a", "task_b", "task_c"])

    assert report["ok"] is True
    assert report["task_count"] == 3
    assert report["duplicate_tasks"] == {}


def test_plan_merge_rejects_duplicate_tasks(tmp_path):
    merge_plan = _load_merge_plan_module()
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_hdf5(first, ["task_a"])
    _write_hdf5(second, ["task_a"])

    report = merge_plan.plan_merge([first, second])

    assert report["ok"] is False
    assert report["duplicate_tasks"]["task_a"] == [str(first), str(second)]


def test_plan_merge_rejects_missing_expected_tasks(tmp_path):
    merge_plan = _load_merge_plan_module()
    first = tmp_path / "first.h5"
    _write_hdf5(first, ["task_a"])

    report = merge_plan.plan_merge([first], expected_tasks=["task_a", "task_b"])

    assert report["ok"] is False
    assert "missing expected tasks: ['task_b']" in report["errors"]


def test_plan_merge_can_validate_input_structure(tmp_path):
    merge_plan = _load_merge_plan_module()
    first = tmp_path / "first.h5"
    _write_valid_activation_hdf5(first, ["task_a"], layer_count=2, hidden_dim=4)

    report = merge_plan.plan_merge(
        [first],
        expected_tasks=["task_a"],
        validate_inputs=True,
        expected_layer_count=2,
        expected_hidden_dim=4,
    )

    assert report["ok"] is True
    assert report["input_validations"][str(first)]["ok"] is True
    assert report["input_validations"][str(first)]["hidden_dim"] == 4


def test_plan_merge_rejects_input_validation_failure(tmp_path):
    merge_plan = _load_merge_plan_module()
    first = tmp_path / "first.h5"
    _write_valid_activation_hdf5(first, ["task_a"], layer_count=2, hidden_dim=3)

    report = merge_plan.plan_merge(
        [first],
        expected_tasks=["task_a"],
        validate_inputs=True,
        expected_layer_count=2,
        expected_hidden_dim=4,
    )

    assert report["ok"] is False
    assert report["input_validations"][str(first)]["ok"] is False
    assert any("input validation failed" in error for error in report["errors"])
