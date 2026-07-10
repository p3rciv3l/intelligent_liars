from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from intelligent_liars.probe_gpu_sensitivity import (
    GPU_SENSITIVITY_DIRECTION_SIGN_CONVENTION,
    GPU_SENSITIVITY_RESULT_FORMAT,
    run_probe_gpu_sensitivity,
)


def _write_pooled_cache_fixture(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    features_by_task = {
        "claims__fictional": np.asarray(
            [
                [-2.2, -1.8, 0.1],
                [-1.9, -2.1, -0.1],
                [-2.0, -2.2, 0.0],
                [-1.8, -1.9, 0.1],
                [2.1, 1.8, -0.1],
                [1.9, 2.2, 0.1],
                [2.2, 2.0, 0.0],
                [1.8, 1.9, -0.1],
            ],
            dtype=np.float32,
        ),
        "internal_state__fictional": np.asarray(
            [
                [-1.7, -2.0, 0.2],
                [-2.0, -1.7, -0.2],
                [-1.8, -2.1, 0.0],
                [-2.1, -1.8, 0.1],
                [1.7, 2.0, -0.2],
                [2.0, 1.7, 0.2],
                [1.8, 2.1, 0.0],
                [2.1, 1.8, -0.1],
            ],
            dtype=np.float32,
        ),
    }

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "qwen_answer_token_pooled_features_v1"
        handle.attrs["pooling"] = "mean_answer_tokens_per_example"
        handle.attrs["hidden_dim"] = 3
        metadata = handle.require_group("metadata")
        layer = handle.require_group("layer_19")
        for task, features in features_by_task.items():
            task_metadata = metadata.create_group(task)
            task_metadata.create_dataset("example_labels", data=labels)
            task_metadata.create_dataset(
                "example_splits", data=np.arange(labels.size + 1, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "example_token_counts", data=np.ones(labels.size, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "example_source_indices", data=np.arange(labels.size, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "example_output_indices", data=np.zeros(labels.size, dtype=np.int64)
            )
            layer.create_dataset(task, data=features)


def _stable_payload(payload: dict[str, object]) -> dict[str, object]:
    stable = json.loads(json.dumps(payload))
    stable.pop("created_at")
    for field in ("training", "final_direction"):
        stable[field]["convergence"].pop("fit_seconds")
        stable[field]["convergence"].pop("train_examples_per_second")
    return stable


def test_cpu_sensitivity_probe_is_deterministic_on_tiny_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "pooled-cache.h5"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    _write_pooled_cache_fixture(cache_path)

    first = run_probe_gpu_sensitivity(
        cache_path=cache_path,
        output_path=first_output,
        layer=19,
        regularization_c=1.0,
        random_seed=7,
        tasks=["claims__fictional", "internal_state__fictional"],
        device="cpu",
        max_steps=100,
        tolerance=1e-7,
        test_size=0.5,
        general_task_class_cap=2,
    )
    run_probe_gpu_sensitivity(
        cache_path=cache_path,
        output_path=second_output,
        layer=19,
        regularization_c=1.0,
        random_seed=7,
        tasks=["claims__fictional", "internal_state__fictional"],
        device="cpu",
        max_steps=100,
        tolerance=1e-7,
        test_size=0.5,
        general_task_class_cap=2,
    )

    assert first.output_path == first_output.resolve()
    assert first.device == "cpu"
    first_payload = json.loads(first_output.read_text())
    second_payload = json.loads(second_output.read_text())
    assert _stable_payload(first_payload) == _stable_payload(second_payload)
    assert first_payload["format"] == GPU_SENSITIVITY_RESULT_FORMAT
    assert first_payload["purpose"] == "sensitivity_throughput_comparison_only"
    assert first_payload["canonical_evidence"] is False
    assert first_payload["replacement_for_sklearn_liblinear"] is False
    assert first_payload["direction_sign_convention"] == (
        GPU_SENSITIVITY_DIRECTION_SIGN_CONVENTION
    )
    assert first_payload["settings"]["device_requested"] == "cpu"
    assert first_payload["settings"]["device_used"] == "cpu"
    assert first_payload["training"]["train_examples"] == 8
    assert first_payload["training"]["task_label_counts"] == {
        "claims__fictional": {"honest": 2, "deceptive": 2},
        "internal_state__fictional": {"honest": 2, "deceptive": 2},
    }
    assert len(first_payload["training"]["direction_vector"]) == 3
    assert first_payload["final_direction"]["trained_on"] == (
        "balanced_capped_all_selected_task_examples"
    )
    assert len(first_payload["final_direction"]["direction_vector"]) == 3
    assert first_payload["training"]["direction_vector"][0] > 0
    assert first_payload["training"]["direction_vector"][1] > 0
    assert {result["task"] for result in first_payload["evaluations"]} == {
        "claims__fictional",
        "internal_state__fictional",
    }
    assert all(
        result["balanced_accuracy"] == 1.0 for result in first_payload["evaluations"]
    )
    assert all(result["auc"] == 1.0 for result in first_payload["evaluations"])


def test_cli_runs_cpu_sensitivity_probe_with_repeatable_tasks(tmp_path: Path) -> None:
    cache_path = tmp_path / "pooled-cache.h5"
    output_path = tmp_path / "result.json"
    _write_pooled_cache_fixture(cache_path)
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_probe_gpu_sensitivity.py"),
            "--cache",
            str(cache_path),
            "--output",
            str(output_path),
            "--layer",
            "19",
            "--c",
            "1.0",
            "--seed",
            "7",
            "--task",
            "claims__fictional",
            "--task",
            "internal_state__fictional",
            "--device",
            "cpu",
            "--max-steps",
            "100",
            "--tolerance",
            "1e-7",
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text())
    assert payload["settings"]["tasks"] == [
        "claims__fictional",
        "internal_state__fictional",
    ]
    assert payload["settings"]["device_used"] == "cpu"
    assert "sensitivity-only probe" in completed.stdout


def test_sensitivity_probe_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    cache_path = tmp_path / "pooled-cache.h5"
    output_path = tmp_path / "result.json"
    _write_pooled_cache_fixture(cache_path)
    output_path.write_text("existing result\n")

    with pytest.raises(FileExistsError, match="GPU sensitivity result already exists"):
        run_probe_gpu_sensitivity(
            cache_path=cache_path,
            output_path=output_path,
            layer=19,
            tasks=["claims__fictional"],
            device="cpu",
            max_steps=10,
        )

    assert output_path.read_text() == "existing result\n"
    assert not list(tmp_path.glob(".result.json.tmp-*"))


def test_mps_request_does_not_fall_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "pooled-cache.h5"
    output_path = tmp_path / "result.json"
    _write_pooled_cache_fixture(cache_path)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="Pass device='cpu' explicitly"):
        run_probe_gpu_sensitivity(
            cache_path=cache_path,
            output_path=output_path,
            layer=19,
            tasks=["claims__fictional"],
            device="mps",
            max_steps=10,
        )

    assert not output_path.exists()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple MPS is unavailable on this machine",
)
def test_mps_sensitivity_probe_runs_on_tiny_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "pooled-cache.h5"
    output_path = tmp_path / "mps-result.json"
    _write_pooled_cache_fixture(cache_path)

    summary = run_probe_gpu_sensitivity(
        cache_path=cache_path,
        output_path=output_path,
        layer=19,
        tasks=["claims__fictional"],
        device="mps",
        max_steps=100,
        tolerance=1e-6,
        test_size=0.5,
        general_task_class_cap=2,
    )

    payload = json.loads(output_path.read_text())
    assert summary.device == "mps"
    assert payload["settings"]["device_used"] == "mps"
    assert payload["training"]["convergence"]["steps"] <= 100
    assert payload["final_direction"]["convergence"]["steps"] <= 100
    assert np.isfinite(payload["training"]["convergence"]["final_loss"])
    assert payload["training"]["convergence"]["train_examples_per_second"] > 0.0
    assert payload["evaluations"][0]["balanced_accuracy"] == 1.0
