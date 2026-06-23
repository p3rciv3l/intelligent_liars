from __future__ import annotations

import json
import math
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
DIRECTION_SIGN_CONVENTION = "sklearn_logistic_coef_positive_points_honest_to_deceptive"


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
class _TaskSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


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
