from __future__ import annotations

import json
import math
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


HONEST_LABEL = 0
DECEPTIVE_LABEL = 1
PROBE_RESULT_FORMAT = "qwen_answer_token_probe_results_v2"
PROBE_PREFLIGHT_FORMAT = "qwen_answer_token_probe_preflight_v1"
DIRECTION_SIGN_CONVENTION = "sklearn_logistic_coef_positive_points_honest_to_deceptive"
PROBE_PREFLIGHT_SPARSE_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35)
TRAINABLE_TASK_STATUS = "trainable"
EVAL_ONLY_TASK_STATUS = "eval_only"
BLOCKED_TASK_STATUS = "blocked"


@dataclass(frozen=True)
class ProbeTrainingSummary:
    output_path: Path
    input_path: Path
    tasks: tuple[str, ...]
    layers: tuple[int, ...]
    within_task_results: int
    cross_task_results: int
    direction_results: int


@dataclass(frozen=True)
class ProbePreflightSummary:
    output_path: Path | None
    input_path: Path
    report: dict[str, Any]

    @property
    def trainable_tasks(self) -> tuple[str, ...]:
        return tuple(self.report["recommendations"]["trainable_tasks"])

    @property
    def eval_only_tasks(self) -> tuple[str, ...]:
        return tuple(self.report["recommendations"]["eval_only_tasks"])

    @property
    def blocked_tasks(self) -> tuple[str, ...]:
        return tuple(self.report["recommendations"]["blocked_tasks"])


@dataclass(frozen=True)
class _TaskMetadata:
    task: str
    labels: np.ndarray
    example_splits: np.ndarray
    example_source_indices: np.ndarray
    example_output_indices: np.ndarray

    @property
    def example_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def token_rows(self) -> int:
        return int(self.example_splits[-1])


@dataclass(frozen=True)
class _ProbePreflightTaskMetadata:
    task: str
    labels: np.ndarray
    example_splits: np.ndarray

    @property
    def example_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def token_rows(self) -> int:
        return int(self.example_splits[-1])


@dataclass(frozen=True)
class _TaskSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


def preflight_probe_training(
    *,
    input_path: Path,
    output_path: Path | None = None,
    tasks: Sequence[str] | None = None,
    test_size: float = 0.25,
    random_seed: int = 0,
    min_train_examples_per_class: int = 20,
    min_eval_examples_per_class: int = 5,
    min_test_examples_per_class: int = 5,
) -> ProbePreflightSummary:
    """Write a metadata-only readiness report for later probe training.

    This function is deliberately not a training entrypoint. It reads HDF5
    metadata and activation dataset shapes/dtypes, but never materializes
    activation tensors or fits a probe.
    """

    import h5py

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if min_train_examples_per_class < 1:
        raise ValueError("min_train_examples_per_class must be positive.")
    if min_eval_examples_per_class < 1:
        raise ValueError("min_eval_examples_per_class must be positive.")
    if min_test_examples_per_class < 1:
        raise ValueError("min_test_examples_per_class must be positive.")

    input_path = input_path.resolve()
    resolved_output_path = output_path.resolve() if output_path is not None else None

    with h5py.File(input_path, "r") as handle:
        _validate_activation_hdf5(handle, input_path=input_path)
        selected_tasks = _select_tasks(handle, tasks)
        common_layers = _common_layers_for_tasks(handle, selected_tasks)
        recommended_layers = _recommend_sparse_probe_layers(common_layers)
        task_metadata = {
            task: _read_preflight_task_metadata(handle, task)
            for task in selected_tasks
        }
        task_reports = [
            _probe_preflight_task_report(
                metadata,
                test_size=test_size,
                random_seed=random_seed,
                min_train_examples_per_class=min_train_examples_per_class,
                min_eval_examples_per_class=min_eval_examples_per_class,
                min_test_examples_per_class=min_test_examples_per_class,
            )
            for metadata in task_metadata.values()
        ]
        activation_shape = _activation_shape_summary(
            handle,
            tasks=selected_tasks,
            layers=common_layers,
        )

    trainable_tasks = tuple(report["task"] for report in task_reports if report["status"] == TRAINABLE_TASK_STATUS)
    eval_only_tasks = tuple(report["task"] for report in task_reports if report["status"] == EVAL_ONLY_TASK_STATUS)
    blocked_tasks = tuple(report["task"] for report in task_reports if report["status"] == BLOCKED_TASK_STATUS)
    total_examples = sum(int(report["examples"]) for report in task_reports)
    total_token_rows = sum(int(report["token_rows"]) for report in task_reports)
    selected_layers_arg = ",".join(str(layer) for layer in recommended_layers)

    output = {
        "format": PROBE_PREFLIGHT_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "phase": "preflight_only_no_training",
        "input_path": str(input_path),
        "artifact": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "size_gib": _bytes_to_gib(input_path.stat().st_size),
        },
        "activation_store": {
            "tasks": len(selected_tasks),
            "layers": len(common_layers),
            "layer_indices": list(common_layers),
            "hidden_dim": activation_shape["hidden_dim"],
            "dtype": activation_shape["dtype"],
            "total_examples": total_examples,
            "total_token_rows": total_token_rows,
        },
        "policy": {
            "split_unit": "example",
            "split_policy": "per-task stratified held-out split",
            "test_size": test_size,
            "random_seed": random_seed,
            "status_policy": {
                TRAINABLE_TASK_STATUS: (
                    "Both labels are present, each class meets the training threshold, "
                    "and the planned held-out split keeps enough examples per class."
                ),
                EVAL_ONLY_TASK_STATUS: (
                    "Both labels are present and the task is usable as a target/test set, "
                    "but it is not large enough to be a training source."
                ),
                BLOCKED_TASK_STATUS: (
                    "The task is too small, missing a class, or otherwise scientifically misleading "
                    "for the next probe phase."
                ),
            },
            "thresholds": {
                "min_train_examples_per_class": min_train_examples_per_class,
                "min_eval_examples_per_class": min_eval_examples_per_class,
                "min_test_examples_per_class": min_test_examples_per_class,
            },
            "metrics": {
                "primary": ["balanced_accuracy", "roc_auc"],
                "secondary": ["accuracy"],
                "reason": "Several tasks are imbalanced, so raw accuracy is not enough.",
            },
            "layer_policy": {
                "initial_pass": "sparse decoder-layer pilot",
                "selected_layers": list(recommended_layers),
                "selected_layers_arg": selected_layers_arg,
                "all_layers_available": list(common_layers),
            },
        },
        "recommendations": {
            "trainable_tasks": list(trainable_tasks),
            "eval_only_tasks": list(eval_only_tasks),
            "blocked_tasks": list(blocked_tasks),
            "selected_layers": list(recommended_layers),
            "selected_layers_arg": selected_layers_arg,
            "future_train_command_not_executed": _future_train_probe_command(
                input_path=input_path,
                trainable_tasks=trainable_tasks,
                layers_arg=selected_layers_arg,
                test_size=test_size,
                random_seed=random_seed,
            ),
            "caveats": [
                "This report is metadata-only and did not train probes.",
                "Eval-only tasks should be used as target/test sets, not as training sources.",
                "Blocked tasks should stay out of the next probe training run.",
            ],
        },
        "io_memory_estimate": _probe_io_memory_estimate(
            file_size_bytes=input_path.stat().st_size,
            total_examples=total_examples,
            total_token_rows=total_token_rows,
            hidden_dim=int(activation_shape["hidden_dim"]),
            dtype_itemsize=int(activation_shape["dtype_itemsize"]),
            selected_layer_count=len(recommended_layers),
            all_layer_count=len(common_layers),
        ),
        "tasks": task_reports,
    }

    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    return ProbePreflightSummary(
        output_path=resolved_output_path,
        input_path=input_path,
        report=output,
    )


def train_probe_directions(
    *,
    input_path: Path,
    output_path: Path,
    tasks: Sequence[str] | None = None,
    layers: str = "all",
    test_size: float = 0.25,
    random_seed: int = 0,
    max_iter: int = 1000,
    regularization_c: float = 1.0,
) -> ProbeTrainingSummary:
    """Train simple linear probes over mean-pooled answer-token activations.

    The split is made over examples before any model fitting. Token rows are
    pooled per example using `metadata/<task>/example_splits`, avoiding
    train/test leakage between tokens from the same answer.
    """

    import h5py

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    with h5py.File(input_path, "r") as handle:
        _validate_activation_hdf5(handle, input_path=input_path)
        selected_tasks = _select_tasks(handle, tasks)
        selected_layers = _select_layers(handle, selected_tasks, layers)
        task_metadata = {
            task: _read_task_metadata(handle, task)
            for task in selected_tasks
        }
        task_splits = {
            task: _make_example_split(
                metadata.labels,
                test_size=test_size,
                random_seed=random_seed,
                task=task,
            )
            for task, metadata in task_metadata.items()
        }
        task_summaries = {
            task: _task_summary(metadata)
            for task, metadata in task_metadata.items()
        }

        within_task_results: list[dict[str, Any]] = []
        cross_task_results: list[dict[str, Any]] = []
        direction_results: list[dict[str, Any]] = []
        for layer in selected_layers:
            pooled_by_task = {
                task: _mean_pool_layer(handle[f"layer_{layer}/{task}"], task_metadata[task].example_splits)
                for task in selected_tasks
            }
            for task in selected_tasks:
                metadata = task_metadata[task]
                split = task_splits[task]
                result = _fit_and_evaluate_probe(
                    x_train=pooled_by_task[task][split.train_indices],
                    y_train=metadata.labels[split.train_indices],
                    x_test=pooled_by_task[task][split.test_indices],
                    y_test=metadata.labels[split.test_indices],
                    layer=layer,
                    train_task=task,
                    test_task=task,
                    result_kind="within_task",
                    random_seed=random_seed,
                    max_iter=max_iter,
                    regularization_c=regularization_c,
                )
                within_task_results.append(result)

            for train_task in selected_tasks:
                source_metadata = task_metadata[train_task]
                source_split = task_splits[train_task]
                for test_task in selected_tasks:
                    if test_task == train_task:
                        continue
                    target_metadata = task_metadata[test_task]
                    result = _fit_and_evaluate_probe(
                        x_train=pooled_by_task[train_task][source_split.train_indices],
                        y_train=source_metadata.labels[source_split.train_indices],
                        x_test=pooled_by_task[test_task],
                        y_test=target_metadata.labels,
                        layer=layer,
                        train_task=train_task,
                        test_task=test_task,
                        result_kind="cross_task",
                        random_seed=random_seed,
                        max_iter=max_iter,
                        regularization_c=regularization_c,
                    )
                    cross_task_results.append(result)

            for task in selected_tasks:
                metadata = task_metadata[task]
                direction_results.append(
                    _fit_final_direction(
                        x_train=pooled_by_task[task],
                        y_train=metadata.labels,
                        layer=layer,
                        task=task,
                        random_seed=random_seed,
                        max_iter=max_iter,
                        regularization_c=regularization_c,
                    )
                )

    output = {
        "format": PROBE_RESULT_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "pooling": "mean_answer_tokens_per_example",
        "split_unit": "example",
        "label_convention": "HONEST=0, DECEPTIVE=1",
        "direction_sign_convention": DIRECTION_SIGN_CONVENTION,
        "settings": {
            "layers": list(selected_layers),
            "tasks": list(selected_tasks),
            "test_size": test_size,
            "random_seed": random_seed,
            "max_iter": max_iter,
            "regularization_c": regularization_c,
            "model": "sklearn.linear_model.LogisticRegression",
            "solver": "liblinear",
            "class_weight": "balanced",
            "cross_task_eval": "source_train_split_to_all_target_examples",
        },
        "directions": direction_results,
        "tasks": task_summaries,
        "within_task": within_task_results,
        "cross_task": cross_task_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return ProbeTrainingSummary(
        output_path=output_path,
        input_path=input_path,
        tasks=selected_tasks,
        layers=selected_layers,
        within_task_results=len(within_task_results),
        cross_task_results=len(cross_task_results),
        direction_results=len(direction_results),
    )


def _validate_activation_hdf5(handle: Any, *, input_path: Path) -> None:
    if handle.attrs.get("format") != "qwen_answer_token_activations_v2":
        raise ValueError(f"Unsupported activation HDF5 format in {input_path}: {handle.attrs.get('format')!r}")
    if "metadata" not in handle:
        raise ValueError(f"Activation HDF5 is missing metadata group: {input_path}")


def _select_tasks(handle: Any, tasks: Sequence[str] | None) -> tuple[str, ...]:
    available = tuple(sorted(str(task) for task in handle["metadata"].keys()))
    if not available:
        raise ValueError("Activation HDF5 contains no metadata tasks.")
    if not tasks:
        return available
    selected = tuple(tasks)
    missing = [task for task in selected if task not in available]
    if missing:
        raise ValueError(f"Activation HDF5 missing requested task(s): {missing}. Available tasks: {list(available)}")
    return selected


def _select_layers(handle: Any, tasks: Sequence[str], layer_spec: str) -> tuple[int, ...]:
    available_by_task: dict[str, set[int]] = {}
    for task in tasks:
        available_by_task[task] = {
            int(group_name.removeprefix("layer_"))
            for group_name in handle.keys()
            if group_name.startswith("layer_") and task in handle[group_name]
        }
    common_layers = set.intersection(*(layers for layers in available_by_task.values()))
    if not common_layers:
        raise ValueError(f"No common layers found for tasks: {list(tasks)}")
    available_layers = tuple(sorted(common_layers))
    if layer_spec.strip().lower() == "all":
        return available_layers

    selected: set[int] = set()
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("No layers were parsed.")
    missing = sorted(selected - common_layers)
    if missing:
        raise ValueError(f"Requested layer(s) not available for all selected tasks: {missing}")
    return tuple(sorted(selected))


def _common_layers_for_tasks(handle: Any, tasks: Sequence[str]) -> tuple[int, ...]:
    available_by_task: dict[str, set[int]] = {}
    for task in tasks:
        available_by_task[task] = {
            int(group_name.removeprefix("layer_"))
            for group_name in handle.keys()
            if group_name.startswith("layer_") and task in handle[group_name]
        }
    common_layers = set.intersection(*(layers for layers in available_by_task.values()))
    if not common_layers:
        raise ValueError(f"No common layers found for tasks: {list(tasks)}")
    return tuple(sorted(common_layers))


def _recommend_sparse_probe_layers(available_layers: Sequence[int]) -> tuple[int, ...]:
    available = tuple(sorted(int(layer) for layer in available_layers))
    canonical = tuple(layer for layer in PROBE_PREFLIGHT_SPARSE_LAYERS if layer in available)
    return canonical or available


def _read_task_metadata(handle: Any, task: str) -> _TaskMetadata:
    metadata = handle[f"metadata/{task}"]
    labels = np.asarray(metadata["example_labels"][:], dtype=np.int64)
    example_splits = np.asarray(metadata["example_splits"][:], dtype=np.int64)
    if example_splits.shape[0] != labels.shape[0] + 1:
        raise ValueError(
            f"metadata/{task}/example_splits length {example_splits.shape[0]} does not match "
            f"example_labels length {labels.shape[0]}."
        )
    if example_splits.shape[0] == 0 or int(example_splits[0]) != 0:
        raise ValueError(f"metadata/{task}/example_splits must start at 0.")
    if np.any(np.diff(example_splits) <= 0):
        raise ValueError(f"metadata/{task}/example_splits must be strictly increasing.")
    if "example_token_counts" in metadata:
        token_counts = np.asarray(metadata["example_token_counts"][:], dtype=np.int64)
        split_counts = np.diff(example_splits)
        if token_counts.shape != labels.shape or not np.array_equal(token_counts, split_counts):
            raise ValueError(f"metadata/{task}/example_token_counts does not match example_splits.")
    unsupported = sorted(set(int(label) for label in labels) - {HONEST_LABEL, DECEPTIVE_LABEL})
    if unsupported:
        raise ValueError(f"Task {task!r} contains non-binary labels in extracted activations: {unsupported}")
    if labels.size == 0:
        raise ValueError(f"Task {task!r} has no examples.")
    return _TaskMetadata(
        task=task,
        labels=labels,
        example_splits=example_splits,
        example_source_indices=np.asarray(metadata["example_source_indices"][:], dtype=np.int64),
        example_output_indices=np.asarray(metadata["example_output_indices"][:], dtype=np.int64),
    )


def _read_preflight_task_metadata(handle: Any, task: str) -> _ProbePreflightTaskMetadata:
    metadata = handle[f"metadata/{task}"]
    labels = np.asarray(metadata["example_labels"][:], dtype=np.int64)
    example_splits = np.asarray(metadata["example_splits"][:], dtype=np.int64)
    if example_splits.shape[0] != labels.shape[0] + 1:
        raise ValueError(
            f"metadata/{task}/example_splits length {example_splits.shape[0]} does not match "
            f"example_labels length {labels.shape[0]}."
        )
    if example_splits.shape[0] == 0 or int(example_splits[0]) != 0:
        raise ValueError(f"metadata/{task}/example_splits must start at 0.")
    if np.any(np.diff(example_splits) <= 0):
        raise ValueError(f"metadata/{task}/example_splits must be strictly increasing.")
    if "example_token_counts" in metadata:
        token_counts = np.asarray(metadata["example_token_counts"][:], dtype=np.int64)
        split_counts = np.diff(example_splits)
        if token_counts.shape != labels.shape or not np.array_equal(token_counts, split_counts):
            raise ValueError(f"metadata/{task}/example_token_counts does not match example_splits.")
    if labels.size == 0:
        raise ValueError(f"Task {task!r} has no examples.")
    return _ProbePreflightTaskMetadata(
        task=task,
        labels=labels,
        example_splits=example_splits,
    )


def _probe_preflight_task_report(
    metadata: _ProbePreflightTaskMetadata,
    *,
    test_size: float,
    random_seed: int,
    min_train_examples_per_class: int,
    min_eval_examples_per_class: int,
    min_test_examples_per_class: int,
) -> dict[str, Any]:
    label_counts = _label_counts(metadata.labels)
    unsupported = sorted(set(int(label) for label in metadata.labels) - {HONEST_LABEL, DECEPTIVE_LABEL})
    status, reason, estimated_split = _probe_preflight_status(
        labels=metadata.labels,
        task=metadata.task,
        unsupported_labels=unsupported,
        test_size=test_size,
        random_seed=random_seed,
        label_counts=label_counts,
        min_train_examples_per_class=min_train_examples_per_class,
        min_eval_examples_per_class=min_eval_examples_per_class,
        min_test_examples_per_class=min_test_examples_per_class,
    )
    min_class_examples = min(label_counts["honest"], label_counts["deceptive"])
    max_class_examples = max(label_counts["honest"], label_counts["deceptive"])
    return {
        "task": metadata.task,
        "status": status,
        "reason": reason,
        "examples": metadata.example_count,
        "token_rows": metadata.token_rows,
        "label_counts": label_counts,
        "unsupported_labels": unsupported,
        "min_class_examples": int(min_class_examples),
        "max_class_examples": int(max_class_examples),
        "class_balance_ratio": float(min_class_examples / max_class_examples) if max_class_examples else 0.0,
        "estimated_split": estimated_split,
    }


def _probe_preflight_status(
    *,
    labels: np.ndarray,
    task: str,
    unsupported_labels: Sequence[int],
    test_size: float,
    random_seed: int,
    label_counts: dict[str, int],
    min_train_examples_per_class: int,
    min_eval_examples_per_class: int,
    min_test_examples_per_class: int,
) -> tuple[str, str, dict[str, Any] | None]:
    if unsupported_labels:
        return (
            BLOCKED_TASK_STATUS,
            f"contains non-binary labels {list(unsupported_labels)}; probe training expects honest/deceptive labels only",
            None,
        )
    if label_counts["honest"] == 0 or label_counts["deceptive"] == 0:
        return (
            BLOCKED_TASK_STATUS,
            "missing one of the two required classes; balanced probe metrics would be misleading",
            None,
        )

    min_class_examples = min(label_counts["honest"], label_counts["deceptive"])
    if min_class_examples < min_eval_examples_per_class:
        return (
            BLOCKED_TASK_STATUS,
            (
                f"only {min_class_examples} examples in the smaller class; "
                f"minimum for a stable target/test set is {min_eval_examples_per_class}"
            ),
            None,
        )
    if min_class_examples < 2:
        return (
            BLOCKED_TASK_STATUS,
            (
                f"only {min_class_examples} examples in the smaller class; "
                "at least 2 per class are required for a stratified held-out split"
            ),
            None,
        )

    split = _make_example_split(labels, test_size=test_size, random_seed=random_seed, task=task)
    split_report = {
        "train_examples": int(split.train_indices.shape[0]),
        "test_examples": int(split.test_indices.shape[0]),
        "train_label_counts": _label_counts(labels[split.train_indices]),
        "test_label_counts": _label_counts(labels[split.test_indices]),
    }
    test_label_counts = split_report["test_label_counts"]
    min_test_examples = min(test_label_counts["honest"], test_label_counts["deceptive"])
    if min_class_examples < min_train_examples_per_class:
        return (
            EVAL_ONLY_TASK_STATUS,
            (
                f"has both labels but only {min_class_examples} examples in the smaller class; "
                f"training sources need at least {min_train_examples_per_class} per class"
            ),
            split_report,
        )
    if min_test_examples < min_test_examples_per_class:
        return (
            EVAL_ONLY_TASK_STATUS,
            (
                f"planned held-out split leaves only {min_test_examples} examples in the smaller test class; "
                f"minimum is {min_test_examples_per_class}"
            ),
            split_report,
        )
    return (
        TRAINABLE_TASK_STATUS,
        "meets per-class and held-out split thresholds for within-task probe training",
        split_report,
    )


def _activation_shape_summary(
    handle: Any,
    *,
    tasks: Sequence[str],
    layers: Sequence[int],
) -> dict[str, Any]:
    hidden_dims: set[int] = set()
    dtypes: set[str] = set()
    dtype_itemsizes: set[int] = set()
    for layer in layers:
        for task in tasks:
            dataset = handle[f"layer_{layer}/{task}"]
            if len(dataset.shape) != 2:
                raise ValueError(f"layer_{layer}/{task} must be a 2D activation dataset.")
            hidden_dims.add(int(dataset.shape[1]))
            dtype = np.dtype(dataset.dtype)
            dtypes.add(str(dtype))
            dtype_itemsizes.add(int(dtype.itemsize))
    if len(hidden_dims) != 1:
        raise ValueError(f"Hidden dim is inconsistent across selected layers/tasks: {sorted(hidden_dims)}")
    if len(dtypes) != 1 or len(dtype_itemsizes) != 1:
        raise ValueError(f"Activation dtype is inconsistent across selected layers/tasks: {sorted(dtypes)}")
    return {
        "hidden_dim": next(iter(hidden_dims)),
        "dtype": next(iter(dtypes)),
        "dtype_itemsize": next(iter(dtype_itemsizes)),
    }


def _probe_io_memory_estimate(
    *,
    file_size_bytes: int,
    total_examples: int,
    total_token_rows: int,
    hidden_dim: int,
    dtype_itemsize: int,
    selected_layer_count: int,
    all_layer_count: int,
) -> dict[str, Any]:
    raw_one_layer = total_token_rows * hidden_dim * dtype_itemsize
    pooled_one_layer_float32 = total_examples * hidden_dim * np.dtype("float32").itemsize
    selected_layers = raw_one_layer * selected_layer_count
    all_layers = raw_one_layer * all_layer_count
    return {
        "hdf5_file_bytes": int(file_size_bytes),
        "hdf5_file_gib": _bytes_to_gib(file_size_bytes),
        "raw_activation_bytes_per_layer": int(raw_one_layer),
        "raw_activation_gib_per_layer": _bytes_to_gib(raw_one_layer),
        "raw_activation_bytes_selected_layers": int(selected_layers),
        "raw_activation_gib_selected_layers": _bytes_to_gib(selected_layers),
        "raw_activation_bytes_all_layers": int(all_layers),
        "raw_activation_gib_all_layers": _bytes_to_gib(all_layers),
        "pooled_feature_matrix_bytes_per_layer_float32": int(pooled_one_layer_float32),
        "pooled_feature_matrix_gib_per_layer_float32": _bytes_to_gib(pooled_one_layer_float32),
        "selected_layer_count": int(selected_layer_count),
        "all_layer_count": int(all_layer_count),
        "recommendation": "Pool and fit one layer at a time; do not materialize all layer tensors together.",
    }


def _future_train_probe_command(
    *,
    input_path: Path,
    trainable_tasks: Sequence[str],
    layers_arg: str,
    test_size: float,
    random_seed: int,
) -> str | None:
    if not trainable_tasks:
        return None
    parts = [
        "PYTHONPATH=src",
        ".venv/bin/python",
        "-m",
        "intelligent_liars",
        "train-probes",
        "--input",
        str(input_path),
        "--output",
        "results/probes/preflight_sparse_probe_results.json",
        "--layers",
        layers_arg,
        "--test-size",
        str(test_size),
        "--random-seed",
        str(random_seed),
    ]
    for task in trainable_tasks:
        parts.extend(["--task", task])
    return " ".join(shlex.quote(part) for part in parts)


def _bytes_to_gib(value: int) -> float:
    return round(value / (1024**3), 4)


def _make_example_split(
    labels: np.ndarray,
    *,
    test_size: float,
    random_seed: int,
    task: str,
) -> _TaskSplit:
    classes, counts = np.unique(labels, return_counts=True)
    if set(classes.tolist()) != {HONEST_LABEL, DECEPTIVE_LABEL}:
        raise ValueError(f"Task {task!r} must contain both honest and deceptive examples.")
    if int(counts.min()) < 2:
        raise ValueError(f"Task {task!r} needs at least two examples per class for train/test splitting.")

    example_count = int(labels.shape[0])
    class_count = int(classes.shape[0])
    requested_test_count = max(class_count, math.ceil(example_count * test_size))
    max_test_count = example_count - class_count
    test_count = min(requested_test_count, max_test_count)
    if test_count < class_count:
        raise ValueError(f"Task {task!r} is too small for an example-level stratified split.")

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


def _task_summary(metadata: _TaskMetadata) -> dict[str, Any]:
    label_counts = _label_counts(metadata.labels)
    return {
        "examples": metadata.example_count,
        "token_rows": metadata.token_rows,
        "honest": label_counts["honest"],
        "deceptive": label_counts["deceptive"],
        "source_index_min": int(metadata.example_source_indices.min()),
        "source_index_max": int(metadata.example_source_indices.max()),
    }


def _mean_pool_layer(dataset: Any, example_splits: np.ndarray) -> np.ndarray:
    if int(example_splits[-1]) != int(dataset.shape[0]):
        raise ValueError(
            f"Layer dataset row count {dataset.shape[0]} does not match metadata split end {example_splits[-1]}."
        )
    data = dataset.astype("float32")[:]
    counts = np.diff(example_splits).astype(np.float32)
    if np.any(counts <= 0):
        raise ValueError("Example splits must define at least one token row per example.")
    sums = np.add.reduceat(data, example_splits[:-1], axis=0)
    return sums / counts[:, None]


def _fit_and_evaluate_probe(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    layer: int,
    train_task: str,
    test_task: str,
    result_kind: str,
    random_seed: int,
    max_iter: int,
    regularization_c: float,
) -> dict[str, Any]:
    classifier = _fit_probe_classifier(
        x_train,
        y_train,
        random_seed=random_seed,
        max_iter=max_iter,
        regularization_c=regularization_c,
    )
    predictions = classifier.predict(x_test)
    scores = classifier.decision_function(x_test)
    auc = _auc_or_none(y_test, scores)
    coef = classifier.coef_[0].astype(np.float64)
    return {
        "kind": result_kind,
        "task": test_task,
        "train_task": train_task,
        "test_task": test_task,
        "layer": int(layer),
        "train_examples": int(y_train.shape[0]),
        "test_examples": int(y_test.shape[0]),
        "train_label_counts": _label_counts(y_train),
        "test_label_counts": _label_counts(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "auc": auc,
        "direction_norm": float(np.linalg.norm(coef)),
        "direction_sign_convention": DIRECTION_SIGN_CONVENTION,
        "intercept": float(classifier.intercept_[0]),
    }


def _fit_final_direction(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    layer: int,
    task: str,
    random_seed: int,
    max_iter: int,
    regularization_c: float,
) -> dict[str, Any]:
    classifier = _fit_probe_classifier(
        x_train,
        y_train,
        random_seed=random_seed,
        max_iter=max_iter,
        regularization_c=regularization_c,
    )
    coef = classifier.coef_[0].astype(np.float64)
    return {
        "kind": "final_direction",
        "task": task,
        "layer": int(layer),
        "trained_on": "all_examples",
        "train_examples": int(y_train.shape[0]),
        "train_label_counts": _label_counts(y_train),
        "feature_count": int(coef.shape[0]),
        "direction_vector": coef.tolist(),
        "direction_norm": float(np.linalg.norm(coef)),
        "direction_sign_convention": DIRECTION_SIGN_CONVENTION,
        "intercept": float(classifier.intercept_[0]),
    }


def _fit_probe_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_seed: int,
    max_iter: int,
    regularization_c: float,
) -> LogisticRegression:
    classifier = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_seed,
        solver="liblinear",
    )
    classifier.fit(x_train, y_train)
    return classifier


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "honest": int(np.sum(labels == HONEST_LABEL)),
        "deceptive": int(np.sum(labels == DECEPTIVE_LABEL)),
    }


def _auc_or_none(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(y_true).shape[0] < 2:
        return None
    return float(roc_auc_score(y_true, scores))
