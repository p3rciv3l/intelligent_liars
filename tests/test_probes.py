from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import intelligent_liars.probes as probes
from intelligent_liars.probes import (
    DIRECTION_SIGN_CONVENTION,
    PROBE_PREFLIGHT_FORMAT,
    PROBE_RESULT_FORMAT,
    preflight_probe_training,
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


def _write_probe_preflight_fixture(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    labels_by_task = {
        "trainable_task": np.asarray([0] * 40 + [1] * 40, dtype=np.int8),
        "eval_only_task": np.asarray([0] * 12 + [1] * 8, dtype=np.int8),
        "insider_trading__upscale": np.asarray([0] * 3 + [1] * 3, dtype=np.int8),
        "single_class_task": np.asarray([0] * 10, dtype=np.int8),
    }

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "qwen_answer_token_activations_v2"
        metadata_group = handle.require_group("metadata")
        for task, labels in labels_by_task.items():
            example_count = int(labels.shape[0])
            example_splits = np.arange(example_count + 1, dtype=np.int64)
            metadata = metadata_group.create_group(task)
            metadata.create_dataset("example_labels", data=labels)
            metadata.create_dataset("example_splits", data=example_splits)
            metadata.create_dataset("example_token_counts", data=np.ones(example_count, dtype=np.int64))
            metadata.create_dataset("example_source_indices", data=np.arange(example_count, dtype=np.int64))
            metadata.create_dataset("example_output_indices", data=np.zeros(example_count, dtype=np.int64))

            data = np.zeros((example_count, 4), dtype=np.float16)
            for layer in range(36):
                layer_group = handle.require_group(f"layer_{layer}")
                layer_group.create_dataset(task, data=data)


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


def test_train_probe_directions_reuses_split_classifiers_for_cross_task_eval(tmp_path, monkeypatch):
    input_path = tmp_path / "activations.h5"
    output_path = tmp_path / "probe-results.json"
    _write_probe_fixture(input_path)

    fit_calls = 0
    real_fit_probe_classifier = probes._fit_probe_classifier

    def counting_fit_probe_classifier(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return real_fit_probe_classifier(*args, **kwargs)

    monkeypatch.setattr(probes, "_fit_probe_classifier", counting_fit_probe_classifier)

    summary = train_probe_directions(
        input_path=input_path,
        output_path=output_path,
        layers="0-1",
        test_size=0.5,
        random_seed=3,
    )

    assert summary.within_task_results == 4
    assert summary.cross_task_results == 4
    assert summary.direction_results == 4
    assert fit_calls == 8


def test_train_probe_directions_can_train_general_domain_probe(tmp_path):
    input_path = tmp_path / "activations.h5"
    output_path = tmp_path / "probe-results.json"
    _write_probe_fixture(input_path)

    summary = train_probe_directions(
        input_path=input_path,
        output_path=output_path,
        layers="0",
        test_size=0.5,
        random_seed=3,
        train_general_domain_probe=True,
        general_task_class_cap=2,
    )

    assert summary.general_domain_results == 2
    assert summary.general_domain_direction_results == 1

    payload = json.loads(output_path.read_text())
    assert payload["settings"]["train_general_domain_probe"] is True
    assert payload["settings"]["general_task_class_cap"] == 2
    general = payload["general_domain"]
    assert general["training_policy"]["task_class_cap"] == 2
    assert len(general["evaluations"]) == 2
    assert len(general["directions"]) == 1

    for result in general["evaluations"]:
        assert result["kind"] == "general_domain"
        assert result["train_task"] == "general_domain"
        assert result["train_examples"] == 8
        assert result["train_label_counts"] == {"honest": 4, "deceptive": 4}
        assert result["test_examples"] == 4
        assert result["test_label_counts"] == {"honest": 2, "deceptive": 2}
        assert result["balanced_accuracy"] == 1.0
        assert result["train_task_label_counts"] == {
            "roleplaying__plain": {"honest": 2, "deceptive": 2},
            "sandbagging_v2__wmdp_mmlu": {"honest": 2, "deceptive": 2},
        }

    direction = general["directions"][0]
    assert direction["kind"] == "final_direction"
    assert direction["task"] == "general_domain"
    assert direction["trained_on"] == "balanced_capped_all_selected_task_examples"
    assert direction["train_examples"] == 8
    assert direction["train_label_counts"] == {"honest": 4, "deceptive": 4}
    assert len(direction["direction_vector"]) == 3
    assert direction["train_task_label_counts"] == {
        "roleplaying__plain": {"honest": 2, "deceptive": 2},
        "sandbagging_v2__wmdp_mmlu": {"honest": 2, "deceptive": 2},
    }


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


def test_preflight_probe_training_classifies_tasks_and_writes_report(tmp_path):
    input_path = tmp_path / "activations.h5"
    output_path = tmp_path / "probe-preflight.json"
    _write_probe_preflight_fixture(input_path)

    summary = preflight_probe_training(
        input_path=input_path,
        output_path=output_path,
        test_size=0.25,
        random_seed=3,
    )

    assert summary.output_path == output_path.resolve()
    assert summary.trainable_tasks == ("trainable_task",)
    assert summary.eval_only_tasks == ("eval_only_task",)
    assert summary.blocked_tasks == ("insider_trading__upscale", "single_class_task")

    payload = json.loads(output_path.read_text())
    assert payload["format"] == PROBE_PREFLIGHT_FORMAT
    assert payload["phase"] == "preflight_only_no_training"
    assert "directions" not in payload
    assert "within_task" not in payload
    assert "cross_task" not in payload

    assert payload["activation_store"]["tasks"] == 4
    assert payload["activation_store"]["layers"] == 36
    assert payload["activation_store"]["hidden_dim"] == 4
    assert payload["activation_store"]["dtype"] == "float16"
    assert payload["activation_store"]["total_examples"] == 116
    assert payload["activation_store"]["total_token_rows"] == 116

    assert payload["policy"]["metrics"]["primary"] == ["balanced_accuracy", "roc_auc"]
    assert payload["policy"]["layer_policy"]["selected_layers"] == [3, 7, 11, 15, 19, 23, 27, 31, 35]
    assert payload["policy"]["layer_policy"]["selected_layers_arg"] == "3,7,11,15,19,23,27,31,35"

    tasks = {task["task"]: task for task in payload["tasks"]}
    assert tasks["trainable_task"]["status"] == "trainable"
    assert tasks["trainable_task"]["estimated_split"]["test_label_counts"] == {"honest": 10, "deceptive": 10}
    assert tasks["eval_only_task"]["status"] == "eval_only"
    assert "not large enough" in payload["policy"]["status_policy"]["eval_only"]
    assert "training sources need at least 20" in tasks["eval_only_task"]["reason"]
    assert tasks["insider_trading__upscale"]["status"] == "blocked"
    assert "minimum for a stable target/test set is 5" in tasks["insider_trading__upscale"]["reason"]
    assert tasks["single_class_task"]["status"] == "blocked"
    assert "missing one of the two required classes" in tasks["single_class_task"]["reason"]

    command = payload["recommendations"]["future_train_command_not_executed"]
    assert "--layers 3,7,11,15,19,23,27,31,35" in command
    assert "--task trainable_task" in command
    assert "eval_only_task" not in command
    assert "insider_trading__upscale" not in command

    estimate = payload["io_memory_estimate"]
    assert estimate["raw_activation_bytes_per_layer"] == 116 * 4 * 2
    assert estimate["raw_activation_bytes_selected_layers"] == 116 * 4 * 2 * 9
    assert estimate["raw_activation_bytes_all_layers"] == 116 * 4 * 2 * 36
    assert estimate["pooled_feature_matrix_bytes_per_layer_float32"] == 116 * 4 * 4


def test_preflight_probe_training_uses_copyable_layer_list_without_parser_refactor(tmp_path):
    input_path = tmp_path / "activations.h5"
    _write_probe_fixture(input_path)

    summary = preflight_probe_training(
        input_path=input_path,
        min_train_examples_per_class=2,
        min_eval_examples_per_class=2,
        min_test_examples_per_class=1,
    )

    assert summary.report["policy"]["layer_policy"]["selected_layers"] == [0, 1]
    assert summary.report["policy"]["layer_policy"]["selected_layers_arg"] == "0,1"
    assert "--layers 0,1" in summary.report["recommendations"]["future_train_command_not_executed"]


def test_preflight_probe_training_blocks_one_per_class_before_split(tmp_path):
    h5py = pytest.importorskip("h5py")
    input_path = tmp_path / "tiny-activations.h5"
    labels = np.asarray([0, 1], dtype=np.int8)

    with h5py.File(input_path, "w") as handle:
        handle.attrs["format"] = "qwen_answer_token_activations_v2"
        metadata = handle.require_group("metadata").create_group("tiny_task")
        metadata.create_dataset("example_labels", data=labels)
        metadata.create_dataset("example_splits", data=np.asarray([0, 1, 2], dtype=np.int64))
        metadata.create_dataset("example_token_counts", data=np.ones(2, dtype=np.int64))
        metadata.create_dataset("example_source_indices", data=np.arange(2, dtype=np.int64))
        metadata.create_dataset("example_output_indices", data=np.zeros(2, dtype=np.int64))
        for layer in range(2):
            handle.require_group(f"layer_{layer}").create_dataset("tiny_task", data=np.zeros((2, 3), dtype=np.float16))

    summary = preflight_probe_training(
        input_path=input_path,
        min_train_examples_per_class=1,
        min_eval_examples_per_class=1,
        min_test_examples_per_class=1,
    )

    task = summary.report["tasks"][0]
    assert summary.blocked_tasks == ("tiny_task",)
    assert task["status"] == "blocked"
    assert "at least 2 per class" in task["reason"]
