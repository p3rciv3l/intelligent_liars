from __future__ import annotations

import json
import math
import os
import shlex
import uuid
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
POOLED_FEATURE_CACHE_FORMAT = "qwen_answer_token_pooled_features_v1"
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
    general_domain_results: int = 0
    general_domain_direction_results: int = 0


@dataclass(frozen=True)
class PooledFeatureCacheSummary:
    output_path: Path
    input_path: Path
    tasks: tuple[str, ...]
    layers: tuple[int, ...]
    hidden_dim: int
    feature_datasets: int


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


@dataclass(frozen=True)
class _BalancedProbeSample:
    features: np.ndarray
    labels: np.ndarray
    task_label_counts: dict[str, dict[str, int]]


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
    evaluation_tasks: Sequence[str] | None = None,
    layers: str = "all",
    test_size: float = 0.25,
    random_seed: int = 0,
    max_iter: int = 1000,
    regularization_c: float = 1.0,
    train_general_domain_probe: bool = False,
    general_task_class_cap: int | None = 1000,
    overwrite: bool = False,
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
    if general_task_class_cap is not None and general_task_class_cap < 1:
        raise ValueError("general_task_class_cap must be positive when provided.")

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    _require_available_probe_output(output_path, overwrite=overwrite)

    with h5py.File(input_path, "r") as handle:
        source = _RawFeatureSource(handle, input_path=input_path)
        output, summary = _train_probe_directions_from_source(
            source=source,
            input_path=input_path,
            output_path=output_path,
            tasks=tasks,
            evaluation_tasks=evaluation_tasks,
            layers=layers,
            test_size=test_size,
            random_seed=random_seed,
            max_iter=max_iter,
            regularization_c=regularization_c,
            train_general_domain_probe=train_general_domain_probe,
            general_task_class_cap=general_task_class_cap,
            input_kind="activation_hdf5",
        )

    _write_probe_result(output_path, output, overwrite=overwrite)
    return summary


def train_probe_directions_from_cache(
    *,
    cache_path: Path,
    output_path: Path,
    tasks: Sequence[str] | None = None,
    evaluation_tasks: Sequence[str] | None = None,
    layers: str = "all",
    source_path: Path | None = None,
    test_size: float = 0.25,
    random_seed: int = 0,
    max_iter: int = 1000,
    regularization_c: float = 1.0,
    train_general_domain_probe: bool = False,
    general_task_class_cap: int | None = 1000,
    overwrite: bool = False,
) -> ProbeTrainingSummary:
    """Train simple linear probes from a pooled-feature cache."""

    import h5py

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    if general_task_class_cap is not None and general_task_class_cap < 1:
        raise ValueError("general_task_class_cap must be positive when provided.")

    cache_path = cache_path.resolve()
    output_path = output_path.resolve()
    _require_available_probe_output(output_path, overwrite=overwrite)
    resolved_source_path = source_path.resolve() if source_path is not None else None

    with h5py.File(cache_path, "r") as handle:
        source = _CachedFeatureSource(
            handle,
            cache_path=cache_path,
            expected_source_path=resolved_source_path,
        )
        output, summary = _train_probe_directions_from_source(
            source=source,
            input_path=cache_path,
            output_path=output_path,
            tasks=tasks,
            evaluation_tasks=evaluation_tasks,
            layers=layers,
            test_size=test_size,
            random_seed=random_seed,
            max_iter=max_iter,
            regularization_c=regularization_c,
            train_general_domain_probe=train_general_domain_probe,
            general_task_class_cap=general_task_class_cap,
            input_kind="pooled_feature_cache",
        )

    _write_probe_result(output_path, output, overwrite=overwrite)
    return summary


def build_pooled_feature_cache(
    *,
    input_path: Path,
    output_path: Path,
    tasks: Sequence[str] | None = None,
    layers: str = "all",
    dtype: str = "float32",
    compression: str | None = "lzf",
    overwrite: bool = False,
) -> PooledFeatureCacheSummary:
    """Build an HDF5 cache of mean-pooled per-example feature matrices."""

    import h5py

    if np.dtype(dtype) != np.dtype("float32"):
        raise ValueError("Pooled feature cache currently supports dtype='float32' only.")

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Pooled feature cache already exists: {output_path}. Pass overwrite=True to replace it.")
    temp_output_path = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")

    try:
        with h5py.File(input_path, "r") as source_handle:
            source = _RawFeatureSource(source_handle, input_path=input_path)
            selected_tasks = source.select_tasks(tasks)
            selected_layers = source.select_layers(selected_tasks, layers)
            task_metadata = {
                task: source.read_task_metadata(task)
                for task in selected_tasks
            }
            hidden_dim = _activation_shape_summary(
                source_handle,
                tasks=selected_tasks,
                layers=selected_layers,
            )["hidden_dim"]
            source_stat = input_path.stat()
            with h5py.File(temp_output_path, "w") as cache_handle:
                cache_handle.attrs["format"] = POOLED_FEATURE_CACHE_FORMAT
                cache_handle.attrs["created_at"] = datetime.now(UTC).isoformat()
                cache_handle.attrs["pooling"] = "mean_answer_tokens_per_example"
                cache_handle.attrs["dtype"] = str(np.dtype(dtype))
                cache_handle.attrs["hidden_dim"] = int(hidden_dim)
                cache_handle.attrs["selected_layers"] = np.asarray(selected_layers, dtype=np.int64)
                cache_handle.attrs["selected_tasks_json"] = json.dumps(list(selected_tasks), sort_keys=True)
                cache_handle.attrs["label_convention"] = "HONEST=0, DECEPTIVE=1"
                cache_handle.attrs["source_path"] = str(input_path)
                cache_handle.attrs["source_size_bytes"] = int(source_stat.st_size)
                cache_handle.attrs["source_mtime_ns"] = int(source_stat.st_mtime_ns)
                cache_handle.attrs["source_format"] = str(source_handle.attrs.get("format", ""))

                metadata_group = cache_handle.require_group("metadata")
                for task, metadata in task_metadata.items():
                    target = metadata_group.create_group(task)
                    target.create_dataset("example_labels", data=metadata.labels.astype(np.int8))
                    target.create_dataset("example_splits", data=metadata.example_splits.astype(np.int64))
                    target.create_dataset("example_token_counts", data=np.diff(metadata.example_splits).astype(np.int64))
                    target.create_dataset("example_source_indices", data=metadata.example_source_indices.astype(np.int64))
                    target.create_dataset("example_output_indices", data=metadata.example_output_indices.astype(np.int64))

                for layer in selected_layers:
                    layer_group = cache_handle.require_group(f"layer_{layer}")
                    for task in selected_tasks:
                        pooled = source.pooled_features(layer, task).astype(dtype, copy=False)
                        layer_group.create_dataset(task, data=pooled, compression=compression)
        os.replace(temp_output_path, output_path)
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink()

    return PooledFeatureCacheSummary(
        output_path=output_path,
        input_path=input_path,
        tasks=selected_tasks,
        layers=selected_layers,
        hidden_dim=int(hidden_dim),
        feature_datasets=len(selected_tasks) * len(selected_layers),
    )


def _train_probe_directions_from_source(
    *,
    source: "_FeatureSource",
    input_path: Path,
    output_path: Path,
    tasks: Sequence[str] | None,
    evaluation_tasks: Sequence[str] | None,
    layers: str,
    test_size: float,
    random_seed: int,
    max_iter: int,
    regularization_c: float,
    train_general_domain_probe: bool,
    general_task_class_cap: int | None,
    input_kind: str,
) -> tuple[dict[str, Any], ProbeTrainingSummary]:
    selected_tasks = source.select_tasks(tasks)
    selected_evaluation_tasks = (
        selected_tasks
        if evaluation_tasks is None
        else source.select_tasks(evaluation_tasks)
    )
    all_selected_tasks = tuple(dict.fromkeys((*selected_tasks, *selected_evaluation_tasks)))
    selected_layers = source.select_layers(all_selected_tasks, layers)
    task_metadata = {
        task: source.read_task_metadata(task)
        for task in all_selected_tasks
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
    general_domain_results: list[dict[str, Any]] = []
    general_domain_direction_results: list[dict[str, Any]] = []
    for layer in selected_layers:
        pooled_by_task = {
            task: source.pooled_features(layer, task)
            for task in all_selected_tasks
        }
        _validate_pooled_feature_shapes(
            pooled_by_task=pooled_by_task,
            task_metadata=task_metadata,
            layer=layer,
        )
        split_classifiers: dict[str, LogisticRegression] = {}
        for task in selected_tasks:
            metadata = task_metadata[task]
            split = task_splits[task]
            train_features = pooled_by_task[task][split.train_indices]
            train_labels = metadata.labels[split.train_indices]
            classifier = _fit_probe_classifier(
                train_features,
                train_labels,
                random_seed=random_seed,
                max_iter=max_iter,
                regularization_c=regularization_c,
            )
            split_classifiers[task] = classifier
            result = _evaluate_fitted_probe(
                classifier=classifier,
                y_train=train_labels,
                x_test=pooled_by_task[task][split.test_indices],
                y_test=metadata.labels[split.test_indices],
                layer=layer,
                train_task=task,
                test_task=task,
                result_kind="within_task",
            )
            within_task_results.append(result)

        for train_task in selected_tasks:
            source_metadata = task_metadata[train_task]
            source_split = task_splits[train_task]
            source_train_labels = source_metadata.labels[source_split.train_indices]
            classifier = split_classifiers[train_task]
            for test_task in selected_evaluation_tasks:
                if test_task == train_task:
                    continue
                target_metadata = task_metadata[test_task]
                result = _evaluate_fitted_probe(
                    classifier=classifier,
                    y_train=source_train_labels,
                    x_test=pooled_by_task[test_task],
                    y_test=target_metadata.labels,
                    layer=layer,
                    train_task=train_task,
                    test_task=test_task,
                    result_kind="cross_task",
                )
                cross_task_results.append(result)

        if train_general_domain_probe:
            general_train_sample = _make_balanced_probe_sample(
                pooled_by_task=pooled_by_task,
                task_metadata=task_metadata,
                indices_by_task={
                    task: task_splits[task].train_indices
                    for task in selected_tasks
                },
                tasks=selected_tasks,
                task_class_cap=general_task_class_cap,
                random_seed=random_seed + int(layer) * 1009,
            )
            general_classifier = _fit_probe_classifier(
                general_train_sample.features,
                general_train_sample.labels,
                random_seed=random_seed,
                max_iter=max_iter,
                regularization_c=regularization_c,
            )
            for test_task in selected_evaluation_tasks:
                target_metadata = task_metadata[test_task]
                target_split = task_splits[test_task]
                general_domain_results.append(
                    _evaluate_fitted_probe(
                        classifier=general_classifier,
                        y_train=general_train_sample.labels,
                        x_test=pooled_by_task[test_task][target_split.test_indices],
                        y_test=target_metadata.labels[target_split.test_indices],
                        layer=layer,
                        train_task="general_domain",
                        test_task=test_task,
                        result_kind="general_domain",
                        train_task_label_counts=general_train_sample.task_label_counts,
                    )
                )

            general_final_sample = _make_balanced_probe_sample(
                pooled_by_task=pooled_by_task,
                task_metadata=task_metadata,
                indices_by_task={
                    task: np.arange(task_metadata[task].example_count, dtype=np.int64)
                    for task in selected_tasks
                },
                tasks=selected_tasks,
                task_class_cap=general_task_class_cap,
                random_seed=random_seed + int(layer) * 1009 + 503,
            )
            general_domain_direction_results.append(
                _fit_final_direction(
                    x_train=general_final_sample.features,
                    y_train=general_final_sample.labels,
                    layer=layer,
                    task="general_domain",
                    trained_on="balanced_capped_all_selected_task_examples",
                    random_seed=random_seed,
                    max_iter=max_iter,
                    regularization_c=regularization_c,
                    train_task_label_counts=general_final_sample.task_label_counts,
                )
            )

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
        "input_kind": input_kind,
        "pooling": "mean_answer_tokens_per_example",
        "split_unit": "example",
        "label_convention": "HONEST=0, DECEPTIVE=1",
        "direction_sign_convention": DIRECTION_SIGN_CONVENTION,
        "settings": {
            "layers": list(selected_layers),
            "tasks": list(selected_tasks),
            "evaluation_tasks": list(selected_evaluation_tasks),
            "test_size": test_size,
            "random_seed": random_seed,
            "max_iter": max_iter,
            "regularization_c": regularization_c,
            "train_general_domain_probe": train_general_domain_probe,
            "general_task_class_cap": general_task_class_cap,
            "model": "sklearn.linear_model.LogisticRegression",
            "solver": "liblinear",
            "class_weight": "balanced",
            "cross_task_eval": "source_train_split_to_all_target_examples",
            "general_domain_train": "pooled selected task train splits, capped per task and label",
        },
        "directions": direction_results,
        "tasks": task_summaries,
        "within_task": within_task_results,
        "cross_task": cross_task_results,
    }
    if train_general_domain_probe:
        output["general_domain"] = {
            "evaluations": general_domain_results,
            "directions": general_domain_direction_results,
            "training_policy": {
                "task_class_cap": general_task_class_cap,
                "split_source": "per-task train split for evaluation; all examples for final direction",
                "sampling": "deterministic without replacement within each task/label bucket",
            },
        }
    summary = ProbeTrainingSummary(
        output_path=output_path,
        input_path=input_path,
        tasks=selected_tasks,
        layers=selected_layers,
        within_task_results=len(within_task_results),
        cross_task_results=len(cross_task_results),
        direction_results=len(direction_results),
        general_domain_results=len(general_domain_results),
        general_domain_direction_results=len(general_domain_direction_results),
    )
    return output, summary


def _require_available_probe_output(output_path: Path, *, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Probe result already exists: {output_path}. Pass overwrite=True to replace it."
        )


def _write_probe_result(
    output_path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp_output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if overwrite:
            os.replace(temp_output_path, output_path)
        else:
            try:
                os.link(temp_output_path, output_path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Probe result already exists: {output_path}. Pass overwrite=True to replace it."
                ) from error
            temp_output_path.unlink()
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink()


class _FeatureSource:
    def select_tasks(self, tasks: Sequence[str] | None) -> tuple[str, ...]:
        raise NotImplementedError

    def select_layers(self, tasks: Sequence[str], layer_spec: str) -> tuple[int, ...]:
        raise NotImplementedError

    def read_task_metadata(self, task: str) -> _TaskMetadata:
        raise NotImplementedError

    def pooled_features(self, layer: int, task: str) -> np.ndarray:
        raise NotImplementedError


class _RawFeatureSource(_FeatureSource):
    def __init__(self, handle: Any, *, input_path: Path) -> None:
        _validate_activation_hdf5(handle, input_path=input_path)
        self._handle = handle

    def select_tasks(self, tasks: Sequence[str] | None) -> tuple[str, ...]:
        return _select_tasks(self._handle, tasks)

    def select_layers(self, tasks: Sequence[str], layer_spec: str) -> tuple[int, ...]:
        return _select_layers(self._handle, tasks, layer_spec)

    def read_task_metadata(self, task: str) -> _TaskMetadata:
        return _read_task_metadata(self._handle, task)

    def pooled_features(self, layer: int, task: str) -> np.ndarray:
        metadata = self.read_task_metadata(task)
        return _mean_pool_layer(
            self._handle[f"layer_{layer}/{task}"],
            metadata.example_splits,
        )


class _CachedFeatureSource(_FeatureSource):
    def __init__(
        self,
        handle: Any,
        *,
        cache_path: Path,
        expected_source_path: Path | None = None,
    ) -> None:
        _validate_pooled_feature_cache(
            handle,
            cache_path=cache_path,
            expected_source_path=expected_source_path,
        )
        self._handle = handle

    def select_tasks(self, tasks: Sequence[str] | None) -> tuple[str, ...]:
        return _select_tasks(self._handle, tasks)

    def select_layers(self, tasks: Sequence[str], layer_spec: str) -> tuple[int, ...]:
        return _select_layers(self._handle, tasks, layer_spec)

    def read_task_metadata(self, task: str) -> _TaskMetadata:
        return _read_task_metadata(self._handle, task)

    def pooled_features(self, layer: int, task: str) -> np.ndarray:
        return np.asarray(self._handle[f"layer_{layer}/{task}"][:], dtype=np.float32)


def _validate_pooled_feature_cache(
    handle: Any,
    *,
    cache_path: Path,
    expected_source_path: Path | None,
) -> None:
    if handle.attrs.get("format") != POOLED_FEATURE_CACHE_FORMAT:
        raise ValueError(f"Unsupported pooled feature cache format in {cache_path}: {handle.attrs.get('format')!r}")
    if handle.attrs.get("pooling") != "mean_answer_tokens_per_example":
        raise ValueError(f"Unsupported pooled feature cache pooling in {cache_path}: {handle.attrs.get('pooling')!r}")
    if "metadata" not in handle:
        raise ValueError(f"Pooled feature cache is missing metadata group: {cache_path}")
    if expected_source_path is not None:
        cached_source = Path(str(handle.attrs.get("source_path", ""))).expanduser()
        if cached_source != expected_source_path:
            raise ValueError(
                f"Pooled feature cache source mismatch: cache source is {cached_source}, "
                f"expected {expected_source_path}."
            )
        if expected_source_path.exists():
            stat = expected_source_path.stat()
            cached_size = int(handle.attrs.get("source_size_bytes", -1))
            cached_mtime_ns = int(handle.attrs.get("source_mtime_ns", -1))
            if cached_size != int(stat.st_size) or cached_mtime_ns != int(stat.st_mtime_ns):
                raise ValueError(
                    "Pooled feature cache source identity mismatch: "
                    f"cached size/mtime=({cached_size}, {cached_mtime_ns}), "
                    f"current size/mtime=({stat.st_size}, {stat.st_mtime_ns})."
                )


def _feature_source_shape_summary(
    source: _FeatureSource,
    *,
    tasks: Sequence[str],
    layers: Sequence[int],
) -> dict[str, Any]:
    hidden_dims: set[int] = set()
    dtypes: set[str] = set()
    for layer in layers:
        for task in tasks:
            features = source.pooled_features(layer, task)
            if features.ndim != 2:
                raise ValueError(f"layer_{layer}/{task} must be a 2D feature dataset.")
            hidden_dims.add(int(features.shape[1]))
            dtypes.add(str(np.dtype(features.dtype)))
    if len(hidden_dims) != 1:
        raise ValueError(f"Hidden dim is inconsistent across selected layers/tasks: {sorted(hidden_dims)}")
    if len(dtypes) != 1:
        raise ValueError(f"Feature dtype is inconsistent across selected layers/tasks: {sorted(dtypes)}")
    return {
        "hidden_dim": next(iter(hidden_dims)),
        "dtype": next(iter(dtypes)),
    }


def _validate_pooled_feature_shapes(
    *,
    pooled_by_task: dict[str, np.ndarray],
    task_metadata: dict[str, _TaskMetadata],
    layer: int,
) -> None:
    hidden_dims: set[int] = set()
    for task, features in pooled_by_task.items():
        if features.ndim != 2:
            raise ValueError(f"layer_{layer}/{task} pooled features must be 2D.")
        expected_rows = task_metadata[task].example_count
        if int(features.shape[0]) != expected_rows:
            raise ValueError(
                f"layer_{layer}/{task} pooled feature row count {features.shape[0]} "
                f"does not match metadata example count {expected_rows}."
            )
        hidden_dims.add(int(features.shape[1]))
    if len(hidden_dims) != 1:
        raise ValueError(f"Hidden dim mismatch for layer {layer}: {sorted(hidden_dims)}")


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
    test_label_counts = _label_counts(labels[split.test_indices])
    split_report = {
        "train_examples": int(split.train_indices.shape[0]),
        "test_examples": int(split.test_indices.shape[0]),
        "train_label_counts": _label_counts(labels[split.train_indices]),
        "test_label_counts": test_label_counts,
    }
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


def _make_balanced_probe_sample(
    *,
    pooled_by_task: dict[str, np.ndarray],
    task_metadata: dict[str, _TaskMetadata],
    indices_by_task: dict[str, np.ndarray],
    tasks: Sequence[str],
    task_class_cap: int | None,
    random_seed: int,
) -> _BalancedProbeSample:
    rng = np.random.default_rng(random_seed)
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    task_label_counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        labels = task_metadata[task].labels
        base_indices = np.asarray(indices_by_task[task], dtype=np.int64)
        selected_for_task: list[np.ndarray] = []
        task_label_counts[task] = {"honest": 0, "deceptive": 0}
        for label, label_name in ((HONEST_LABEL, "honest"), (DECEPTIVE_LABEL, "deceptive")):
            label_indices = base_indices[labels[base_indices] == label]
            if task_class_cap is not None and label_indices.shape[0] > task_class_cap:
                label_indices = np.sort(rng.choice(label_indices, size=task_class_cap, replace=False))
            selected_for_task.append(label_indices)
            task_label_counts[task][label_name] = int(label_indices.shape[0])
        selected_indices = np.concatenate(selected_for_task)
        if selected_indices.size == 0:
            continue
        feature_parts.append(pooled_by_task[task][selected_indices])
        label_parts.append(labels[selected_indices])
    if not feature_parts:
        raise ValueError("No examples selected for general-domain probe training.")

    features = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    if set(np.unique(labels).tolist()) != {HONEST_LABEL, DECEPTIVE_LABEL}:
        raise ValueError("General-domain probe sample must contain both honest and deceptive examples.")
    return _BalancedProbeSample(
        features=features,
        labels=labels,
        task_label_counts=task_label_counts,
    )


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
    return _evaluate_fitted_probe(
        classifier=classifier,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        layer=layer,
        train_task=train_task,
        test_task=test_task,
        result_kind=result_kind,
    )


def _evaluate_fitted_probe(
    *,
    classifier: LogisticRegression,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    layer: int,
    train_task: str,
    test_task: str,
    result_kind: str,
    train_task_label_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    predictions = classifier.predict(x_test)
    scores = classifier.decision_function(x_test)
    auc = _auc_or_none(y_test, scores)
    coef = classifier.coef_[0].astype(np.float64)
    result = {
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
    if train_task_label_counts is not None:
        result["train_task_label_counts"] = train_task_label_counts
    return result


def _fit_final_direction(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    layer: int,
    task: str,
    trained_on: str = "all_examples",
    random_seed: int,
    max_iter: int,
    regularization_c: float,
    train_task_label_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    classifier = _fit_probe_classifier(
        x_train,
        y_train,
        random_seed=random_seed,
        max_iter=max_iter,
        regularization_c=regularization_c,
    )
    coef = classifier.coef_[0].astype(np.float64)
    result = {
        "kind": "final_direction",
        "task": task,
        "layer": int(layer),
        "trained_on": trained_on,
        "train_examples": int(y_train.shape[0]),
        "train_label_counts": _label_counts(y_train),
        "feature_count": int(coef.shape[0]),
        "direction_vector": coef.tolist(),
        "direction_norm": float(np.linalg.norm(coef)),
        "direction_sign_convention": DIRECTION_SIGN_CONVENTION,
        "intercept": float(classifier.intercept_[0]),
    }
    if train_task_label_counts is not None:
        result["train_task_label_counts"] = train_task_label_counts
    return result


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
