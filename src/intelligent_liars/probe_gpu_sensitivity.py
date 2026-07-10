from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from intelligent_liars.probes import POOLED_FEATURE_CACHE_FORMAT


HONEST_LABEL = 0
DECEPTIVE_LABEL = 1
GPU_SENSITIVITY_RESULT_FORMAT = "qwen_answer_token_probe_gpu_sensitivity_v1"
GPU_SENSITIVITY_DIRECTION_SIGN_CONVENTION = (
    "pytorch_logistic_weight_positive_points_honest_to_deceptive"
)


@dataclass(frozen=True)
class ProbeGpuSensitivitySummary:
    output_path: Path
    cache_path: Path
    tasks: tuple[str, ...]
    layer: int
    device: str
    evaluations: int


@dataclass(frozen=True)
class _TaskData:
    labels: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class _TaskSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class _TrainingSample:
    features: np.ndarray
    labels: np.ndarray
    task_label_counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class _FittedProbe:
    direction: np.ndarray
    intercept: float
    convergence: dict[str, Any]


def run_probe_gpu_sensitivity(
    *,
    cache_path: Path,
    output_path: Path,
    layer: int,
    regularization_c: float = 1.0,
    random_seed: int = 0,
    tasks: Sequence[str] | None = None,
    device: str = "mps",
    max_steps: int = 1000,
    tolerance: float = 1e-7,
    test_size: float = 0.25,
    general_task_class_cap: int | None = 1000,
) -> ProbeGpuSensitivitySummary:
    """Fit one non-canonical PyTorch general-domain probe from a pooled cache.

    This entrypoint exists only for Apple-MPS sensitivity and throughput
    comparisons. Canonical probe evidence remains the sklearn/liblinear path.
    """

    _validate_settings(
        regularization_c=regularization_c,
        max_steps=max_steps,
        tolerance=tolerance,
        test_size=test_size,
        general_task_class_cap=general_task_class_cap,
    )
    cache_path = cache_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"GPU sensitivity result already exists: {output_path}")
    torch_device = _resolve_device(device)

    task_data = _load_task_layer_data(
        cache_path=cache_path,
        layer=layer,
        tasks=tasks,
    )
    selected_tasks = tuple(task_data)
    splits = {
        task: _make_example_split(
            data.labels,
            test_size=test_size,
            random_seed=random_seed,
            task=task,
        )
        for task, data in task_data.items()
    }
    sample = _make_general_domain_sample(
        task_data=task_data,
        splits=splits,
        tasks=selected_tasks,
        task_class_cap=general_task_class_cap,
        random_seed=random_seed + int(layer) * 1009,
    )
    evaluation_fitted = _fit_torch_logistic_regression(
        sample.features,
        sample.labels,
        device=torch_device,
        regularization_c=regularization_c,
        max_steps=max_steps,
        tolerance=tolerance,
        random_seed=random_seed,
    )

    final_sample = _make_general_domain_sample(
        task_data=task_data,
        splits={
            task: _TaskSplit(
                train_indices=np.arange(data.labels.shape[0], dtype=np.int64),
                test_indices=np.empty(0, dtype=np.int64),
            )
            for task, data in task_data.items()
        },
        tasks=selected_tasks,
        task_class_cap=general_task_class_cap,
        random_seed=random_seed + int(layer) * 1009 + 503,
    )
    final_fitted = _fit_torch_logistic_regression(
        final_sample.features,
        final_sample.labels,
        device=torch_device,
        regularization_c=regularization_c,
        max_steps=max_steps,
        tolerance=tolerance,
        random_seed=random_seed,
    )

    evaluations = [
        _evaluate_task(
            task=task,
            data=task_data[task],
            split=splits[task],
            direction=evaluation_fitted.direction,
            intercept=evaluation_fitted.intercept,
            layer=layer,
        )
        for task in selected_tasks
    ]
    payload = {
        "format": GPU_SENSITIVITY_RESULT_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "sensitivity_throughput_comparison_only",
        "canonical_evidence": False,
        "replacement_for_sklearn_liblinear": False,
        "cache_path": str(cache_path),
        "pooling": "mean_answer_tokens_per_example",
        "split_unit": "example",
        "label_convention": "HONEST=0, DECEPTIVE=1",
        "direction_sign_convention": GPU_SENSITIVITY_DIRECTION_SIGN_CONVENTION,
        "settings": {
            "layer": int(layer),
            "tasks": list(selected_tasks),
            "test_size": float(test_size),
            "random_seed": int(random_seed),
            "regularization_c": float(regularization_c),
            "general_task_class_cap": general_task_class_cap,
            "device_requested": device,
            "device_used": torch_device.type,
            "dtype": "float32",
            "max_steps": int(max_steps),
            "tolerance": float(tolerance),
            "optimizer": "torch.optim.LBFGS",
            "class_weight": "balanced",
            "general_domain_train": (
                "pooled selected task train splits, capped per task and label"
            ),
        },
        "training": {
            "train_examples": int(sample.labels.shape[0]),
            "train_label_counts": _label_counts(sample.labels),
            "task_label_counts": sample.task_label_counts,
            "objective": (
                "mean class-balanced binary cross entropy plus "
                "L2(direction)/(2*C*n); intercept unregularized"
            ),
            "direction_vector": evaluation_fitted.direction.astype(np.float64).tolist(),
            "direction_norm": float(np.linalg.norm(evaluation_fitted.direction)),
            "direction_sign_convention": GPU_SENSITIVITY_DIRECTION_SIGN_CONVENTION,
            "intercept": evaluation_fitted.intercept,
            "convergence": evaluation_fitted.convergence,
        },
        "final_direction": {
            "trained_on": "balanced_capped_all_selected_task_examples",
            "train_examples": int(final_sample.labels.shape[0]),
            "train_label_counts": _label_counts(final_sample.labels),
            "task_label_counts": final_sample.task_label_counts,
            "direction_vector": final_fitted.direction.astype(np.float64).tolist(),
            "direction_norm": float(np.linalg.norm(final_fitted.direction)),
            "direction_sign_convention": GPU_SENSITIVITY_DIRECTION_SIGN_CONVENTION,
            "intercept": final_fitted.intercept,
            "convergence": final_fitted.convergence,
        },
        "evaluations": evaluations,
    }
    _write_json_without_overwrite(output_path, payload)
    return ProbeGpuSensitivitySummary(
        output_path=output_path,
        cache_path=cache_path,
        tasks=selected_tasks,
        layer=int(layer),
        device=torch_device.type,
        evaluations=len(evaluations),
    )


def _validate_settings(
    *,
    regularization_c: float,
    max_steps: int,
    tolerance: float,
    test_size: float,
    general_task_class_cap: int | None,
) -> None:
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive.")
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if general_task_class_cap is not None and general_task_class_cap < 1:
        raise ValueError("general_task_class_cap must be positive when provided.")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "mps":
        raise ValueError("device must be 'mps' or 'cpu'.")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Pass device='cpu' explicitly to use the CPU fallback."
        )
    return torch.device("mps")


def _load_task_layer_data(
    *,
    cache_path: Path,
    layer: int,
    tasks: Sequence[str] | None,
) -> dict[str, _TaskData]:
    import h5py

    with h5py.File(cache_path, "r") as handle:
        if handle.attrs.get("format") != POOLED_FEATURE_CACHE_FORMAT:
            raise ValueError(
                f"Unsupported pooled feature cache format in {cache_path}: "
                f"{handle.attrs.get('format')!r}"
            )
        if handle.attrs.get("pooling") != "mean_answer_tokens_per_example":
            raise ValueError(
                f"Unsupported pooled feature cache pooling in {cache_path}."
            )
        if "metadata" not in handle:
            raise ValueError(
                f"Pooled feature cache is missing metadata group: {cache_path}"
            )

        available = tuple(sorted(str(task) for task in handle["metadata"].keys()))
        if not available:
            raise ValueError("Pooled feature cache contains no tasks.")
        selected = available if not tasks else tuple(tasks)
        missing = [task for task in selected if task not in available]
        if missing:
            raise ValueError(
                f"Pooled feature cache missing requested task(s): {missing}. "
                f"Available tasks: {list(available)}"
            )

        result: dict[str, _TaskData] = {}
        for task in selected:
            feature_path = f"layer_{int(layer)}/{task}"
            if feature_path not in handle:
                raise ValueError(f"Pooled feature cache is missing {feature_path}.")
            labels = np.asarray(
                handle[f"metadata/{task}/example_labels"][:], dtype=np.int64
            )
            features = np.asarray(handle[feature_path][:], dtype=np.float32)
            if features.ndim != 2:
                raise ValueError(f"{feature_path} must be a 2D feature dataset.")
            if features.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{feature_path} row count {features.shape[0]} does not match "
                    f"example_labels count {labels.shape[0]}."
                )
            unsupported = sorted(
                set(int(label) for label in labels) - {HONEST_LABEL, DECEPTIVE_LABEL}
            )
            if unsupported:
                raise ValueError(
                    f"Task {task!r} contains non-binary labels: {unsupported}"
                )
            result[task] = _TaskData(labels=labels, features=features)

    hidden_dims = {int(data.features.shape[1]) for data in result.values()}
    if len(hidden_dims) != 1:
        raise ValueError(
            f"Hidden dim mismatch for layer {int(layer)}: {sorted(hidden_dims)}"
        )
    return result


def _make_example_split(
    labels: np.ndarray,
    *,
    test_size: float,
    random_seed: int,
    task: str,
) -> _TaskSplit:
    classes, counts = np.unique(labels, return_counts=True)
    if set(classes.tolist()) != {HONEST_LABEL, DECEPTIVE_LABEL}:
        raise ValueError(
            f"Task {task!r} must contain both honest and deceptive examples."
        )
    if int(counts.min()) < 2:
        raise ValueError(
            f"Task {task!r} needs at least two examples per class for train/test splitting."
        )

    example_count = int(labels.shape[0])
    class_count = int(classes.shape[0])
    requested_test_count = max(class_count, math.ceil(example_count * test_size))
    max_test_count = example_count - class_count
    test_count = min(requested_test_count, max_test_count)
    if test_count < class_count:
        raise ValueError(
            f"Task {task!r} is too small for an example-level stratified split."
        )

    indices = np.arange(example_count)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_count,
        random_state=random_seed,
        stratify=labels,
    )
    return _TaskSplit(
        train_indices=np.asarray(sorted(train_indices.tolist()), dtype=np.int64),
        test_indices=np.asarray(sorted(test_indices.tolist()), dtype=np.int64),
    )


def _make_general_domain_sample(
    *,
    task_data: dict[str, _TaskData],
    splits: dict[str, _TaskSplit],
    tasks: Sequence[str],
    task_class_cap: int | None,
    random_seed: int,
) -> _TrainingSample:
    rng = np.random.default_rng(random_seed)
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    task_label_counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        data = task_data[task]
        base_indices = splits[task].train_indices
        selected_for_task: list[np.ndarray] = []
        task_label_counts[task] = {"honest": 0, "deceptive": 0}
        for label, name in (
            (HONEST_LABEL, "honest"),
            (DECEPTIVE_LABEL, "deceptive"),
        ):
            label_indices = base_indices[data.labels[base_indices] == label]
            if task_class_cap is not None and label_indices.shape[0] > task_class_cap:
                label_indices = np.sort(
                    rng.choice(label_indices, size=task_class_cap, replace=False)
                )
            selected_for_task.append(label_indices)
            task_label_counts[task][name] = int(label_indices.shape[0])
        selected_indices = np.concatenate(selected_for_task)
        if selected_indices.size:
            feature_parts.append(data.features[selected_indices])
            label_parts.append(data.labels[selected_indices])

    if not feature_parts:
        raise ValueError("No examples selected for general-domain probe training.")
    features = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    if set(np.unique(labels).tolist()) != {HONEST_LABEL, DECEPTIVE_LABEL}:
        raise ValueError(
            "General-domain probe sample must contain both honest and deceptive examples."
        )
    return _TrainingSample(
        features=features,
        labels=labels,
        task_label_counts=task_label_counts,
    )


def _fit_torch_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    regularization_c: float,
    max_steps: int,
    tolerance: float,
    random_seed: int,
) -> _FittedProbe:
    torch.manual_seed(random_seed)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    direction = torch.zeros(
        x.shape[1], dtype=torch.float32, device=device, requires_grad=True
    )
    intercept = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)

    counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = labels.shape[0] / (2.0 * counts)
    sample_weights = torch.as_tensor(
        class_weights[labels], dtype=torch.float32, device=device
    )
    optimizer = torch.optim.LBFGS(
        [direction, intercept],
        max_iter=max_steps,
        tolerance_grad=tolerance,
        tolerance_change=tolerance,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def objective() -> torch.Tensor:
        logits = x.mv(direction) + intercept
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        data_loss = torch.sum(losses * sample_weights) / labels.shape[0]
        l2_penalty = torch.sum(direction.square()) / (
            2.0 * regularization_c * labels.shape[0]
        )
        return data_loss + l2_penalty

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        closure_evaluations += 1
        return loss

    started = time.perf_counter()
    optimizer.step(closure)
    fit_seconds = time.perf_counter() - started
    optimizer.zero_grad()
    final_loss_tensor = objective()
    final_loss_tensor.backward()
    assert direction.grad is not None
    assert intercept.grad is not None
    gradient_max_abs = max(
        float(direction.grad.detach().abs().max().cpu().item()),
        float(intercept.grad.detach().abs().cpu().item()),
    )
    state = optimizer.state.get(direction, {})
    steps = int(state.get("n_iter", 0))
    if gradient_max_abs <= tolerance:
        stop_reason = "gradient_tolerance"
        converged = True
    elif steps < max_steps:
        stop_reason = "optimizer_stopped_before_max_steps"
        converged = False
    else:
        stop_reason = "max_steps"
        converged = False

    return _FittedProbe(
        direction=direction.detach().cpu().numpy().astype(np.float32, copy=True),
        intercept=float(intercept.detach().cpu().item()),
        convergence={
            "converged": converged,
            "stop_reason": stop_reason,
            "steps": steps,
            "closure_evaluations": closure_evaluations,
            "max_steps": int(max_steps),
            "tolerance": float(tolerance),
            "final_loss": float(final_loss_tensor.detach().cpu().item()),
            "gradient_max_abs": gradient_max_abs,
            "fit_seconds": fit_seconds,
            "train_examples_per_second": (
                float(labels.shape[0] / fit_seconds) if fit_seconds > 0.0 else None
            ),
        },
    )


def _evaluate_task(
    *,
    task: str,
    data: _TaskData,
    split: _TaskSplit,
    direction: np.ndarray,
    intercept: float,
    layer: int,
) -> dict[str, Any]:
    test_labels = data.labels[split.test_indices]
    scores = data.features[split.test_indices] @ direction + intercept
    predictions = (scores >= 0.0).astype(np.int64)
    auc = (
        None
        if np.unique(test_labels).shape[0] < 2
        else float(roc_auc_score(test_labels, scores))
    )
    return {
        "kind": "general_domain",
        "task": task,
        "train_task": "general_domain",
        "test_task": task,
        "layer": int(layer),
        "test_examples": int(test_labels.shape[0]),
        "test_label_counts": _label_counts(test_labels),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "auc": auc,
    }


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "honest": int(np.sum(labels == HONEST_LABEL)),
        "deceptive": int(np.sum(labels == DECEPTIVE_LABEL)),
    }


def _write_json_without_overwrite(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        try:
            os.link(temp_path, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"GPU sensitivity result already exists: {output_path}"
            ) from error
        temp_path.unlink()
    finally:
        if temp_path.exists():
            temp_path.unlink()
