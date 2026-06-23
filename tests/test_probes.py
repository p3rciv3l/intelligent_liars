from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from intelligent_liars.probes import (
    DIRECTION_SIGN_CONVENTION,
    PROBE_RESULT_FORMAT,
    train_probe_directions,
)


def _write_probe_fixture(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    example_splits = np.arange(0, 18, 2, dtype=np.int64)

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "qwen_answer_token_activations_v2"
        metadata_group = handle.require_group("metadata")
        for task, task_offset in {
            "roleplaying__plain": 0.0,
            "sandbagging_v2__wmdp_mmlu": 0.4,
        }.items():
            metadata = metadata_group.create_group(task)
            metadata.create_dataset("example_labels", data=labels)
            metadata.create_dataset("example_splits", data=example_splits)
            metadata.create_dataset("example_token_counts", data=np.full(8, 2, dtype=np.int64))
            metadata.create_dataset("example_source_indices", data=np.arange(8, dtype=np.int64))
            metadata.create_dataset("example_output_indices", data=np.zeros(8, dtype=np.int64))

            base_features = np.asarray(
                [
                    [-2.2, -1.9, 0.0],
                    [-2.0, -2.1, 0.1],
                    [-1.8, -2.2, -0.1],
                    [-2.1, -1.8, 0.0],
                    [1.8, 2.0, 0.0],
                    [2.2, 1.9, 0.1],
                    [2.0, 2.2, -0.1],
                    [2.1, 1.8, 0.0],
                ],
                dtype=np.float32,
            )
            for layer, layer_scale in {0: 1.0, 1: 1.5}.items():
                layer_group = handle.require_group(f"layer_{layer}")
                token_rows = np.repeat((base_features + task_offset) * layer_scale, repeats=2, axis=0)
                layer_group.create_dataset(task, data=token_rows.astype(np.float16))


def test_train_probe_directions_splits_and_reports_by_example(tmp_path):
    input_path = tmp_path / "activations.h5"
    output_path = tmp_path / "probe-results.json"
    _write_probe_fixture(input_path)

    summary = train_probe_directions(
        input_path=input_path,
        output_path=output_path,
        layers="0-1",
        test_size=0.5,
        random_seed=3,
    )

    assert summary.tasks == ("roleplaying__plain", "sandbagging_v2__wmdp_mmlu")
    assert summary.layers == (0, 1)
    assert summary.within_task_results == 4
    assert summary.cross_task_results == 4
    assert summary.direction_results == 4

    payload = json.loads(output_path.read_text())
    assert payload["format"] == PROBE_RESULT_FORMAT
    assert payload["pooling"] == "mean_answer_tokens_per_example"
    assert payload["split_unit"] == "example"
    assert payload["direction_sign_convention"] == DIRECTION_SIGN_CONVENTION
    assert set(payload["tasks"]) == {"roleplaying__plain", "sandbagging_v2__wmdp_mmlu"}

    for result in payload["within_task"]:
        assert result["train_examples"] == 4
        assert result["test_examples"] == 4
        assert result["train_label_counts"] == {"honest": 2, "deceptive": 2}
        assert result["test_label_counts"] == {"honest": 2, "deceptive": 2}
        assert result["accuracy"] == 1.0
        assert result["balanced_accuracy"] == 1.0
        assert result["auc"] == 1.0
        assert result["direction_norm"] > 0
        assert result["direction_sign_convention"] == DIRECTION_SIGN_CONVENTION
        assert "direction_vector" not in result

    for result in payload["cross_task"]:
        assert result["train_task"] != result["test_task"]
        assert result["train_examples"] == 4
        assert result["test_examples"] == 8
        assert result["test_label_counts"] == {"honest": 4, "deceptive": 4}
        assert result["balanced_accuracy"] == 1.0
        assert "direction_vector" not in result

    for direction in payload["directions"]:
        assert direction["kind"] == "final_direction"
        assert direction["trained_on"] == "all_examples"
        assert direction["train_examples"] == 8
        assert direction["train_label_counts"] == {"honest": 4, "deceptive": 4}
        assert direction["feature_count"] == 3
        assert len(direction["direction_vector"]) == 3
        assert direction["direction_norm"] > 0
        assert direction["direction_sign_convention"] == DIRECTION_SIGN_CONVENTION


def test_train_probe_directions_rejects_nonzero_example_splits(tmp_path):
    h5py = pytest.importorskip("h5py")
    input_path = tmp_path / "activations.h5"
    output_path = tmp_path / "probe-results.json"
    _write_probe_fixture(input_path)

    with h5py.File(input_path, "r+") as handle:
        handle["metadata/roleplaying__plain/example_splits"][0] = 1

    with pytest.raises(ValueError, match="example_splits must start at 0"):
        train_probe_directions(
            input_path=input_path,
            output_path=output_path,
            layers="0",
        )
