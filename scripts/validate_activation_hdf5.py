#!/usr/bin/env python3
"""Validate answer-token activation HDF5 artifacts before downstream use."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXPECTED_FORMAT = "qwen_answer_token_activations_v2"
REQUIRED_METADATA_DATASETS = (
    "source_indices",
    "output_indices",
    "labels",
    "example_splits",
    "example_indices",
    "token_positions",
    "logit_positions",
    "example_labels",
)
TOKEN_ROW_METADATA_DATASETS = (
    "source_indices",
    "output_indices",
    "labels",
    "example_indices",
    "token_positions",
)
EXAMPLE_ROW_METADATA_DATASETS = (
    "example_labels",
    "example_source_indices",
    "example_output_indices",
)
ALLOWED_LABELS = {-1, 0, 1}
LAYER_NAME_RE = re.compile(r"^(?:layer[_-]?)?\d+$")


def validate_activation_hdf5(
    path: str | Path,
    *,
    expected_tasks: list[str] | None = None,
    expected_task_count: int | None = None,
    expected_task_counts: dict[str, int] | None = None,
    expected_example_counts: dict[str, int] | None = None,
    expected_layer_count: int | None = None,
    expected_hidden_dim: int | None = None,
    finite_check: str = "sample",
    finite_sample_rows: int = 32,
    sha256: bool = False,
    expected_format: str = EXPECTED_FORMAT,
) -> dict[str, Any]:
    """Validate an activation HDF5 file and return a JSON-serializable report."""
    artifact_path = Path(path)
    errors: list[str] = []
    result: dict[str, Any] = {
        "ok": False,
        "path": str(artifact_path),
        "expected_format": expected_format,
        "root_attrs": {},
        "tasks": [],
        "task_count": 0,
        "task_rows": {},
        "example_counts": {},
        "layers": [],
        "layer_count": 0,
        "hidden_dim": None,
        "label_counts": {},
        "binary_usable_counts": {},
        "finite_check": {
            "mode": finite_check,
            "datasets_checked": 0,
            "sample_rows_per_dataset": finite_sample_rows,
            "non_finite": [],
        },
        "sha256": None,
        "errors": errors,
    }

    if finite_check not in {"none", "sample", "full"}:
        errors.append(
            f"finite_check must be one of none, sample, full; got {finite_check!r}"
        )
        return result

    if not artifact_path.exists():
        errors.append("file does not exist")
        return result
    if not artifact_path.is_file():
        errors.append("path is not a file")
        return result

    try:
        with h5py.File(artifact_path, "r") as h5:
            result["root_attrs"] = {
                key: _jsonable_attr(value) for key, value in h5.attrs.items()
            }
            if not _attrs_contain_expected_format(h5.attrs, expected_format):
                errors.append(
                    "root attrs do not include expected format "
                    f"{expected_format!r}"
                )

            metadata_group = h5.get("metadata")
            if not isinstance(metadata_group, h5py.Group):
                errors.append("metadata group is missing")
                result["ok"] = False
                return _finish_result(result, artifact_path, sha256)

            task_groups = _discover_task_metadata_groups(metadata_group)
            task_names = sorted(task_groups)
            result["tasks"] = task_names
            result["task_count"] = len(task_names)
            if not task_names:
                errors.append("task list is empty")

            declared_tasks = _read_declared_task_list(metadata_group)
            if declared_tasks is not None and sorted(declared_tasks) != task_names:
                errors.append(
                    "declared task list does not match metadata task groups: "
                    f"declared={sorted(declared_tasks)!r}, groups={task_names!r}"
                )

            _validate_expected_tasks(
                task_names=task_names,
                expected_tasks=expected_tasks,
                expected_task_count=expected_task_count,
                errors=errors,
            )

            task_row_counts: dict[str, int] = {}
            example_counts: dict[str, int] = {}
            for task_name in task_names:
                task_group = task_groups[task_name]
                row_count, example_count = _validate_task_metadata(
                    task_name=task_name,
                    task_group=task_group,
                    errors=errors,
                    label_counts=result["label_counts"],
                    binary_usable_counts=result["binary_usable_counts"],
                )
                if row_count is not None:
                    task_row_counts[task_name] = row_count
                if example_count is not None:
                    example_counts[task_name] = example_count

            result["task_rows"] = task_row_counts
            result["example_counts"] = example_counts
            expected_task_counts = expected_task_counts or {}
            for task_name, expected_count in sorted(expected_task_counts.items()):
                actual_count = task_row_counts.get(task_name)
                if actual_count is None:
                    errors.append(
                        f"expected row count for unknown task {task_name!r}"
                    )
                elif actual_count != expected_count:
                    errors.append(
                        f"task {task_name!r} has {actual_count} rows; "
                        f"expected {expected_count}"
                    )
            expected_example_counts = expected_example_counts or {}
            for task_name, expected_count in sorted(expected_example_counts.items()):
                actual_count = example_counts.get(task_name)
                if actual_count is None:
                    errors.append(
                        f"expected example count for unknown task {task_name!r}"
                    )
                elif actual_count != expected_count:
                    errors.append(
                        f"task {task_name!r} has {actual_count} examples; "
                        f"expected {expected_count}"
                    )

            layer_task_datasets = _discover_activation_datasets(h5, task_names)
            layer_names = sorted(layer_task_datasets, key=_layer_sort_key)
            result["layers"] = layer_names
            result["layer_count"] = len(layer_names)
            if not layer_names:
                errors.append("no activation layer groups found")
            if (
                expected_layer_count is not None
                and len(layer_names) != expected_layer_count
            ):
                errors.append(
                    f"found {len(layer_names)} layers; "
                    f"expected {expected_layer_count}"
                )

            hidden_dims: set[int] = set()
            for layer_name in layer_names:
                datasets_by_task = layer_task_datasets[layer_name]
                layer_tasks = sorted(datasets_by_task)
                if layer_tasks != task_names:
                    missing = sorted(set(task_names) - set(layer_tasks))
                    extra = sorted(set(layer_tasks) - set(task_names))
                    if missing:
                        errors.append(
                            f"layer {layer_name!r} is missing tasks {missing!r}"
                        )
                    if extra:
                        errors.append(
                            f"layer {layer_name!r} has extra tasks {extra!r}"
                        )

                for task_name, dataset in sorted(datasets_by_task.items()):
                    _validate_activation_dataset(
                        layer_name=layer_name,
                        task_name=task_name,
                        dataset=dataset,
                        expected_rows=task_row_counts.get(task_name),
                        hidden_dims=hidden_dims,
                        finite_check=finite_check,
                        finite_sample_rows=finite_sample_rows,
                        finite_result=result["finite_check"],
                        errors=errors,
                    )

            if len(hidden_dims) == 1:
                result["hidden_dim"] = next(iter(hidden_dims))
            elif len(hidden_dims) > 1:
                errors.append(
                    "hidden dim is inconsistent across activation datasets: "
                    f"{sorted(hidden_dims)!r}"
                )

            if (
                expected_hidden_dim is not None
                and result["hidden_dim"] != expected_hidden_dim
            ):
                errors.append(
                    f"hidden dim is {result['hidden_dim']}; "
                    f"expected {expected_hidden_dim}"
                )

    except OSError as exc:
        errors.append(f"file is not a readable HDF5 file: {exc}")
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        errors.append(f"unexpected validation error: {exc}")

    return _finish_result(result, artifact_path, sha256)


def _finish_result(
    result: dict[str, Any],
    artifact_path: Path,
    sha256: bool,
) -> dict[str, Any]:
    if sha256 and artifact_path.exists() and artifact_path.is_file():
        result["sha256"] = _sha256_file(artifact_path)
    result["ok"] = not result["errors"]
    return result


def _jsonable_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _jsonable_attr(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable_attr(item) for item in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def _attrs_contain_expected_format(attrs: h5py.AttributeManager, expected: str) -> bool:
    return any(_value_contains_string(value, expected) for value in attrs.values())


def _value_contains_string(value: Any, expected: str) -> bool:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace") == expected
    if isinstance(value, str):
        return value == expected
    if isinstance(value, np.generic):
        return _value_contains_string(value.item(), expected)
    if isinstance(value, np.ndarray):
        return any(_value_contains_string(item, expected) for item in value.flat)
    return False


def _discover_task_metadata_groups(
    metadata_group: h5py.Group,
) -> dict[str, h5py.Group]:
    if isinstance(metadata_group.get("tasks"), h5py.Group):
        container = metadata_group["tasks"]
    else:
        container = metadata_group

    task_groups: dict[str, h5py.Group] = {}
    for name, item in container.items():
        if not isinstance(item, h5py.Group):
            continue
        if any(dataset_name in item for dataset_name in REQUIRED_METADATA_DATASETS):
            task_groups[name] = item
    return task_groups


def _read_declared_task_list(metadata_group: h5py.Group) -> list[str] | None:
    for name in ("task_names", "task_list", "tasks"):
        item = metadata_group.get(name)
        if isinstance(item, h5py.Dataset):
            value = item[()]
            if np.isscalar(value) or isinstance(value, bytes):
                return [_decode_scalar(value)]
            return [_decode_scalar(element) for element in value]
    return None


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _decode_scalar(value.item())
    return str(value)


def _validate_expected_tasks(
    *,
    task_names: list[str],
    expected_tasks: list[str] | None,
    expected_task_count: int | None,
    errors: list[str],
) -> None:
    if expected_task_count is not None and len(task_names) != expected_task_count:
        errors.append(
            f"found {len(task_names)} tasks; expected {expected_task_count}"
        )
    if expected_tasks is not None and sorted(expected_tasks) != task_names:
        errors.append(
            f"task list {task_names!r} does not match expected "
            f"{sorted(expected_tasks)!r}"
        )


def _validate_task_metadata(
    *,
    task_name: str,
    task_group: h5py.Group,
    errors: list[str],
    label_counts: dict[str, dict[str, int]],
    binary_usable_counts: dict[str, int],
) -> tuple[int | None, int | None]:
    missing = [
        dataset_name
        for dataset_name in REQUIRED_METADATA_DATASETS
        if not isinstance(task_group.get(dataset_name), h5py.Dataset)
    ]
    if missing:
        errors.append(
            f"task {task_name!r} is missing metadata datasets {missing!r}"
        )
        return None, None

    lengths: dict[str, int] = {}
    for dataset_name in REQUIRED_METADATA_DATASETS:
        dataset = task_group[dataset_name]
        if dataset.ndim == 0:
            errors.append(
                f"task {task_name!r} metadata {dataset_name!r} is scalar"
            )
            continue
        lengths[dataset_name] = dataset.shape[0]

    if "labels" not in lengths:
        return None, None
    row_count = lengths["labels"]

    token_lengths = {
        dataset_name: lengths[dataset_name]
        for dataset_name in TOKEN_ROW_METADATA_DATASETS
        if dataset_name in lengths
    }
    bad_token_lengths = {
        dataset_name: length
        for dataset_name, length in token_lengths.items()
        if length != row_count
    }
    if bad_token_lengths:
        errors.append(
            f"task {task_name!r} token metadata row counts do not align with "
            f"labels={row_count}: {bad_token_lengths!r}"
        )
        return None, None

    logit_positions_len = lengths.get("logit_positions")
    if logit_positions_len not in (None, 0, row_count):
        errors.append(
            f"task {task_name!r} metadata 'logit_positions' has "
            f"{logit_positions_len} rows; expected 0 or {row_count}"
        )
        return None, None

    example_count = _validate_example_splits(
        task_name=task_name,
        task_group=task_group,
        token_row_count=row_count,
        errors=errors,
    )
    if example_count is None:
        return None, None

    for dataset_name in EXAMPLE_ROW_METADATA_DATASETS:
        dataset = task_group.get(dataset_name)
        if dataset is None:
            continue
        if dataset.shape[0] != example_count:
            errors.append(
                f"task {task_name!r} metadata {dataset_name!r} has "
                f"{dataset.shape[0]} rows; expected {example_count} examples"
            )
            return None, None

    _validate_duplicate_keys(task_name, task_group, example_count, errors)
    _validate_label_dataset(
        task_name=task_name,
        dataset_name="labels",
        dataset=task_group["labels"],
        errors=errors,
        label_counts=label_counts,
        binary_usable_counts=binary_usable_counts,
    )
    _validate_label_dataset(
        task_name=task_name,
        dataset_name="example_labels",
        dataset=task_group["example_labels"],
        errors=errors,
        label_counts=None,
        binary_usable_counts=None,
    )

    return row_count, example_count


def _validate_example_splits(
    *,
    task_name: str,
    task_group: h5py.Group,
    token_row_count: int,
    errors: list[str],
) -> int | None:
    try:
        splits = np.asarray(task_group["example_splits"][()]).astype(np.int64)
    except (TypeError, ValueError) as exc:
        errors.append(
            f"task {task_name!r} metadata 'example_splits' is not integer-like: {exc}"
        )
        return None
    if splits.ndim != 1:
        errors.append(f"task {task_name!r} metadata 'example_splits' is not 1D")
        return None
    if len(splits) == 0:
        errors.append(f"task {task_name!r} metadata 'example_splits' is empty")
        return None
    if int(splits[0]) != 0:
        errors.append(
            f"task {task_name!r} metadata 'example_splits' starts at "
            f"{int(splits[0])}; expected 0"
        )
        return None
    if int(splits[-1]) != token_row_count:
        errors.append(
            f"task {task_name!r} metadata 'example_splits' ends at "
            f"{int(splits[-1])}; expected token rows {token_row_count}"
        )
        return None
    if np.any(np.diff(splits) < 0):
        errors.append(
            f"task {task_name!r} metadata 'example_splits' is not nondecreasing"
        )
        return None
    return len(splits) - 1


def _validate_duplicate_keys(
    task_name: str,
    task_group: h5py.Group,
    example_count: int,
    errors: list[str],
) -> None:
    if (
        "example_source_indices" in task_group
        and "example_output_indices" in task_group
    ):
        source_indices = task_group["example_source_indices"][()]
        output_indices = task_group["example_output_indices"][()]
    elif task_group["source_indices"].shape[0] == example_count:
        source_indices = task_group["source_indices"][()]
        output_indices = task_group["output_indices"][()]
    else:
        errors.append(
            f"task {task_name!r} cannot validate duplicate example keys without "
            "'example_source_indices' and 'example_output_indices'"
        )
        return
    seen: set[tuple[str, Any, Any]] = set()
    duplicates: list[tuple[Any, Any]] = []
    for source_index, output_index in zip(source_indices, output_indices, strict=True):
        key = (
            task_name,
            _hashable_scalar(source_index),
            _hashable_scalar(output_index),
        )
        if key in seen:
            duplicates.append((key[1], key[2]))
            if len(duplicates) >= 5:
                break
        seen.add(key)

    if duplicates:
        errors.append(
            f"task {task_name!r} has duplicate "
            "(task, source_index, output_index) keys; "
            f"first duplicates={duplicates!r}"
        )


def _hashable_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _hashable_scalar(value.item())
    if isinstance(value, np.ndarray):
        return tuple(_hashable_scalar(item) for item in value.tolist())
    return value


def _validate_label_dataset(
    *,
    task_name: str,
    dataset_name: str,
    dataset: h5py.Dataset,
    errors: list[str],
    label_counts: dict[str, dict[str, int]] | None,
    binary_usable_counts: dict[str, int] | None,
) -> None:
    try:
        labels = np.asarray(dataset[()])
        int_labels = labels.astype(np.int64)
    except (TypeError, ValueError) as exc:
        errors.append(
            f"task {task_name!r} metadata {dataset_name!r} is not integer-like: "
            f"{exc}"
        )
        return

    if not np.array_equal(labels, int_labels):
        errors.append(
            f"task {task_name!r} metadata {dataset_name!r} contains "
            "non-integer labels"
        )
        return

    values, counts = np.unique(int_labels, return_counts=True)
    counts_by_label = {
        str(int(value)): int(count) for value, count in zip(values, counts)
    }
    invalid = sorted(int(value) for value in values if int(value) not in ALLOWED_LABELS)
    if invalid:
        errors.append(
            f"task {task_name!r} metadata {dataset_name!r} has invalid "
            f"labels {invalid!r}; allowed={sorted(ALLOWED_LABELS)!r}"
        )

    if label_counts is not None:
        label_counts[task_name] = counts_by_label
    if binary_usable_counts is not None:
        binary_usable_counts[task_name] = int(
            counts_by_label.get("0", 0) + counts_by_label.get("1", 0)
        )


def _discover_activation_datasets(
    h5: h5py.File,
    task_names: list[str],
) -> dict[str, dict[str, h5py.Dataset]]:
    for container_name in ("layers", "activations"):
        item = h5.get(container_name)
        if isinstance(item, h5py.Group):
            discovered = _discover_layer_first(item, task_names)
            if discovered:
                return discovered
            discovered = _discover_task_first(item, task_names)
            if discovered:
                return discovered

    root_layer_groups = {
        name: item
        for name, item in h5.items()
        if isinstance(item, h5py.Group) and _looks_like_layer_name(name)
    }
    return _layer_group_mapping(root_layer_groups)


def _discover_layer_first(
    container: h5py.Group,
    task_names: list[str],
) -> dict[str, dict[str, h5py.Dataset]]:
    layer_groups = {
        name: item
        for name, item in container.items()
        if isinstance(item, h5py.Group) and _looks_like_layer_name(name)
    }
    discovered = _layer_group_mapping(layer_groups)
    if discovered:
        return discovered

    candidate_groups = {
        name: item
        for name, item in container.items()
        if isinstance(item, h5py.Group)
    }
    discovered = _layer_group_mapping(candidate_groups)
    if task_names and discovered:
        matching_layers = {
            layer_name: datasets
            for layer_name, datasets in discovered.items()
            if set(datasets).intersection(task_names)
        }
        if matching_layers:
            return matching_layers
    return {}


def _layer_group_mapping(
    layer_groups: dict[str, h5py.Group],
) -> dict[str, dict[str, h5py.Dataset]]:
    discovered: dict[str, dict[str, h5py.Dataset]] = {}
    for layer_name, layer_group in layer_groups.items():
        datasets = {
            task_name: dataset
            for task_name, dataset in layer_group.items()
            if isinstance(dataset, h5py.Dataset)
        }
        if datasets:
            discovered[layer_name] = datasets
    return discovered


def _discover_task_first(
    container: h5py.Group,
    task_names: list[str],
) -> dict[str, dict[str, h5py.Dataset]]:
    discovered: dict[str, dict[str, h5py.Dataset]] = {}
    for task_name in task_names:
        task_group = container.get(task_name)
        if not isinstance(task_group, h5py.Group):
            continue
        for layer_name, dataset in task_group.items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            if not _looks_like_layer_name(layer_name):
                continue
            discovered.setdefault(layer_name, {})[task_name] = dataset
    return discovered


def _looks_like_layer_name(name: str) -> bool:
    return bool(LAYER_NAME_RE.match(name))


def _layer_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"\d+", name)
    if match:
        return (int(match.group()), name)
    return (sys.maxsize, name)


def _validate_activation_dataset(
    *,
    layer_name: str,
    task_name: str,
    dataset: h5py.Dataset,
    expected_rows: int | None,
    hidden_dims: set[int],
    finite_check: str,
    finite_sample_rows: int,
    finite_result: dict[str, Any],
    errors: list[str],
) -> None:
    dataset_path = dataset.name
    if dataset.ndim != 2:
        errors.append(
            f"activation dataset {dataset_path!r} must be 2D; "
            f"shape={dataset.shape!r}"
        )
        return
    if not np.issubdtype(dataset.dtype, np.number):
        errors.append(
            f"activation dataset {dataset_path!r} must be numeric; "
            f"dtype={dataset.dtype}"
        )
        return

    rows, hidden_dim = dataset.shape
    hidden_dims.add(int(hidden_dim))
    if expected_rows is not None and rows != expected_rows:
        errors.append(
            f"activation dataset {dataset_path!r} has {rows} rows, but "
            f"task {task_name!r} metadata has {expected_rows} rows"
        )

    if finite_check == "none":
        return

    finite_result["datasets_checked"] += 1
    if finite_check == "full":
        data = dataset[()]
    else:
        row_indices = _sample_indices(rows, finite_sample_rows)
        if not row_indices:
            data = np.empty((0, hidden_dim), dtype=dataset.dtype)
        else:
            data = dataset[row_indices, :]

    if not np.isfinite(data).all():
        finite_result["non_finite"].append(
            {
                "layer": layer_name,
                "task": task_name,
                "dataset": dataset_path,
            }
        )
        errors.append(
            f"activation dataset {dataset_path!r} contains non-finite values "
            f"under finite_check={finite_check!r}"
        )


def _sample_indices(row_count: int, sample_rows: int) -> list[int]:
    if row_count <= 0 or sample_rows <= 0:
        return []
    if row_count <= sample_rows:
        return list(range(row_count))
    indices = np.linspace(0, row_count - 1, num=sample_rows, dtype=np.int64)
    return sorted(set(int(index) for index in indices))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items


def _parse_task_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        for part in parts:
            if "=" not in part:
                raise argparse.ArgumentTypeError(
                    "expected task counts in TASK=COUNT form"
                )
            task_name, count_text = part.split("=", 1)
            task_name = task_name.strip()
            if not task_name:
                raise argparse.ArgumentTypeError("task name cannot be empty")
            try:
                counts[task_name] = int(count_text)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid count for task {task_name!r}: {count_text!r}"
                ) from exc
    return counts


def _format_human_summary(result: dict[str, Any]) -> str:
    status = "OK" if result["ok"] else "FAILED"
    lines = [
        f"activation HDF5 validation: {status}",
        f"path: {result['path']}",
        f"tasks: {result['task_count']} {result['tasks']}",
        f"layers: {result['layer_count']} {result['layers']}",
        f"hidden_dim: {result['hidden_dim']}",
        f"task_rows: {result['task_rows']}",
        f"example_counts: {result['example_counts']}",
        f"label_counts: {result['label_counts']}",
        f"binary_usable_counts: {result['binary_usable_counts']}",
        (
            "finite_check: "
            f"{result['finite_check']['mode']} "
            f"checked={result['finite_check']['datasets_checked']} "
            f"non_finite={len(result['finite_check']['non_finite'])}"
        ),
    ]
    if result.get("sha256"):
        lines.append(f"sha256: {result['sha256']}")
    if result["errors"]:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in result["errors"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate activation HDF5 artifacts before DVC/probe use."
    )
    parser.add_argument("path", help="Path to the activation HDF5 artifact")
    parser.add_argument(
        "--expected-format",
        default=EXPECTED_FORMAT,
        help=f"Expected root attr format value (default: {EXPECTED_FORMAT})",
    )
    parser.add_argument(
        "--expected-tasks",
        help="Comma-separated task names expected in metadata",
    )
    parser.add_argument(
        "--expected-task-count",
        type=int,
        help="Expected number of tasks",
    )
    parser.add_argument(
        "--expected-task-counts",
        action="append",
        default=[],
        help="Expected rows per task, as TASK=COUNT pairs; repeat or comma-separate",
    )
    parser.add_argument(
        "--expected-task-rows",
        action="append",
        default=[],
        help="Alias for --expected-task-counts",
    )
    parser.add_argument(
        "--expected-example-counts",
        action="append",
        default=[],
        help="Expected examples per task, as TASK=COUNT pairs; repeat or comma-separate",
    )
    parser.add_argument(
        "--expected-examples",
        action="append",
        default=[],
        help="Alias for --expected-example-counts",
    )
    parser.add_argument(
        "--expected-layer-count",
        type=int,
        help="Expected number of layer groups",
    )
    parser.add_argument(
        "--expected-hidden-dim",
        type=int,
        help="Expected activation hidden dimension",
    )
    parser.add_argument(
        "--finite-check",
        choices=("none", "sample", "full"),
        default="sample",
        help="How aggressively to check activations for finite values",
    )
    parser.add_argument(
        "--finite-sample-rows",
        type=int,
        default=32,
        help="Rows per layer/task dataset checked when --finite-check=sample",
    )
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Include a sha256 digest of the HDF5 file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the activation HDF5 validator CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        expected_task_counts = _parse_task_counts(
            args.expected_task_counts + args.expected_task_rows
        )
        expected_example_counts = _parse_task_counts(
            args.expected_example_counts + args.expected_examples
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    result = validate_activation_hdf5(
        args.path,
        expected_tasks=_parse_csv(args.expected_tasks),
        expected_task_count=args.expected_task_count,
        expected_task_counts=expected_task_counts,
        expected_example_counts=expected_example_counts,
        expected_layer_count=args.expected_layer_count,
        expected_hidden_dim=args.expected_hidden_dim,
        finite_check=args.finite_check,
        finite_sample_rows=args.finite_sample_rows,
        sha256=args.sha256,
        expected_format=args.expected_format,
    )

    sys.stderr.write(_format_human_summary(result) + "\n")
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
