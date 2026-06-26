from __future__ import annotations

import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from intelligent_liars.activation_types import (
    ActivationExample,
    ActivationExtractionSettings,
    ActivationMergeSummary,
)
from intelligent_liars.rollouts import qwen_model_slug
from intelligent_liars.run_control import (
    acquire_lock,
    command_line,
    install_signal_cleanup,
    lock_payload,
    new_run_id,
)


def write_activation_hdf5(
    *,
    output_path: Path,
    task: str,
    layers: Sequence[int],
    activations_by_layer: Mapping[int, Sequence[Any]],
    logits_by_batch: Sequence[Any] | None = None,
    source_indices: Sequence[int],
    output_indices: Sequence[int],
    labels: Sequence[int],
    example_indices: Sequence[int] = (),
    token_positions: Sequence[int] = (),
    logit_positions: Sequence[int] = (),
    example_splits: Sequence[int],
    example_source_indices: Sequence[int] = (),
    example_output_indices: Sequence[int] = (),
    example_labels: Sequence[int] = (),
    example_token_counts: Sequence[int] = (),
    detected_answer_texts: Sequence[str] = (),
    decoded_answer_texts: Sequence[str] = (),
    char_spans_json: Sequence[str] = (),
    messages_json: Sequence[str] = (),
    rendered_texts: Sequence[str] = (),
    source_datasets: Sequence[str] = (),
    raw_labels_json: Sequence[str] = (),
    label_schemas: Sequence[str] = (),
    example_metadata_json: Sequence[str] = (),
    model_id: str,
    backend_name: str = "transformers-hooks",
    surface_names: Mapping[int, str] | None = None,
    preserved_input_keys: Sequence[str] = (),
    dataset_id: str | None = None,
    processor_id: str | None = None,
    task_metadata_attrs: Mapping[str, Any] | None = None,
    compression: Literal["gzip", "lzf", "none"] = "lzf",
    overwrite: bool,
    resume: bool = False,
    storage_dtype: Literal["float16", "float32"] = "float16",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = len(labels)
    if len(source_indices) != row_count or len(output_indices) != row_count:
        raise ValueError(
            "source_indices, output_indices, and labels must have the same token-row count."
        )
    if example_splits[-1] != row_count:
        raise ValueError("example_splits[-1] must equal the number of token rows.")
    if example_indices and len(example_indices) != row_count:
        raise ValueError("example_indices must have one value per token row.")
    if token_positions and len(token_positions) != row_count:
        raise ValueError("token_positions must have one value per token row.")

    surface_names = surface_names or {}
    shard_path = _write_activation_shard(
        output_path=output_path,
        task=task,
        layers=layers,
        activations_by_layer=activations_by_layer,
        logits_by_batch=logits_by_batch or (),
        source_indices=source_indices,
        output_indices=output_indices,
        labels=labels,
        example_indices=example_indices,
        token_positions=token_positions,
        logit_positions=logit_positions,
        example_splits=example_splits,
        example_source_indices=example_source_indices,
        example_output_indices=example_output_indices,
        example_labels=example_labels,
        example_token_counts=example_token_counts,
        detected_answer_texts=detected_answer_texts,
        decoded_answer_texts=decoded_answer_texts,
        char_spans_json=char_spans_json,
        messages_json=messages_json,
        rendered_texts=rendered_texts,
        source_datasets=source_datasets,
        raw_labels_json=raw_labels_json,
        label_schemas=label_schemas,
        example_metadata_json=example_metadata_json,
        model_id=model_id,
        backend_name=backend_name,
        surface_names=surface_names,
        preserved_input_keys=preserved_input_keys,
        dataset_id=dataset_id,
        processor_id=processor_id,
        task_metadata_attrs=task_metadata_attrs or {},
        compression=compression,
        storage_dtype=storage_dtype,
    )
    try:
        _merge_activation_shard(
            output_path=output_path,
            shard_path=shard_path,
            task=task,
            layers=layers,
            overwrite=overwrite,
            resume=resume,
        )
    finally:
        shard_path.unlink(missing_ok=True)


def _write_activation_shard(
    *,
    output_path: Path,
    task: str,
    layers: Sequence[int],
    activations_by_layer: Mapping[int, Sequence[Any]],
    logits_by_batch: Sequence[Any],
    source_indices: Sequence[int],
    output_indices: Sequence[int],
    labels: Sequence[int],
    example_indices: Sequence[int],
    token_positions: Sequence[int],
    logit_positions: Sequence[int],
    example_splits: Sequence[int],
    example_source_indices: Sequence[int],
    example_output_indices: Sequence[int],
    example_labels: Sequence[int],
    example_token_counts: Sequence[int],
    detected_answer_texts: Sequence[str],
    decoded_answer_texts: Sequence[str],
    char_spans_json: Sequence[str],
    messages_json: Sequence[str],
    rendered_texts: Sequence[str],
    source_datasets: Sequence[str],
    raw_labels_json: Sequence[str],
    label_schemas: Sequence[str],
    example_metadata_json: Sequence[str],
    model_id: str,
    backend_name: str,
    surface_names: Mapping[int, str],
    preserved_input_keys: Sequence[str],
    dataset_id: str | None,
    processor_id: str | None,
    task_metadata_attrs: Mapping[str, Any],
    compression: Literal["gzip", "lzf", "none"],
    storage_dtype: Literal["float16", "float32"],
) -> Path:
    import h5py
    import numpy as np

    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.{_safe_hdf5_name(task)}.",
        suffix=".shard.h5",
        dir=output_path.parent,
        delete=False,
    )
    tmp_file.close()
    shard_path = Path(tmp_file.name)

    try:
        with h5py.File(shard_path, "w") as handle:
            handle.attrs["format"] = "qwen_answer_token_activations_v2"
            handle.attrs["model_id"] = model_id
            handle.attrs["processor_id"] = processor_id or model_id
            handle.attrs["backend"] = backend_name
            handle.attrs["created_at"] = datetime.now(UTC).isoformat()
            handle.attrs["versions_json"] = json.dumps(
                _version_metadata(), sort_keys=True
            )
            handle.attrs["source_output_path"] = str(output_path)

            metadata = handle.require_group("metadata")
            task_metadata = metadata.create_group(task)
            task_metadata.attrs["dataset_id"] = dataset_id or ""
            task_metadata.attrs["label_convention"] = "HONEST=0, DECEPTIVE=1, SKIP=-1"
            task_metadata.attrs["aggregation"] = "token_rows/no_pooling"
            task_metadata.attrs["layer_convention"] = (
                "Qwen3-VL decoder block output; embeddings are not layer_0"
            )
            task_metadata.attrs["backend"] = backend_name
            task_metadata.attrs["model_id"] = model_id
            task_metadata.attrs["processor_id"] = processor_id or model_id
            task_metadata.attrs["preserved_input_keys_json"] = json.dumps(
                list(preserved_input_keys), sort_keys=True
            )
            task_metadata.attrs["layers_json"] = json.dumps(
                [int(layer) for layer in layers]
            )
            task_metadata.attrs["surface_names_json"] = json.dumps(
                {
                    str(layer): surface_names.get(
                        layer, f"decoder_layer_{layer}_answer_tokens"
                    )
                    for layer in layers
                },
                sort_keys=True,
            )
            task_metadata.attrs["dataset_metadata_json"] = json.dumps(
                _jsonable_metadata(task_metadata_attrs),
                sort_keys=True,
            )
            for attr_name, attr_value in task_metadata_attrs.items():
                if isinstance(attr_value, bool | int | float | str):
                    task_metadata.attrs[attr_name] = attr_value
            task_metadata.attrs["activation_storage_dtype"] = storage_dtype

            task_metadata.create_dataset(
                "source_indices", data=np.asarray(source_indices, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "output_indices", data=np.asarray(output_indices, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "labels", data=np.asarray(labels, dtype=np.int8)
            )
            task_metadata.create_dataset(
                "example_splits", data=np.asarray(example_splits, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "example_indices", data=np.asarray(example_indices, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "token_positions", data=np.asarray(token_positions, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "logit_positions", data=np.asarray(logit_positions, dtype=np.int64)
            )
            task_metadata.create_dataset(
                "example_source_indices",
                data=np.asarray(example_source_indices, dtype=np.int64),
            )
            task_metadata.create_dataset(
                "example_output_indices",
                data=np.asarray(example_output_indices, dtype=np.int64),
            )
            task_metadata.create_dataset(
                "example_labels", data=np.asarray(example_labels, dtype=np.int8)
            )
            task_metadata.create_dataset(
                "example_token_counts",
                data=np.asarray(example_token_counts, dtype=np.int64),
            )
            string_dtype = h5py.string_dtype(encoding="utf-8")
            task_metadata.create_dataset(
                "detected_answer_texts",
                data=np.asarray(list(detected_answer_texts), dtype=object),
                dtype=string_dtype,
            )
            task_metadata.create_dataset(
                "decoded_answer_texts",
                data=np.asarray(list(decoded_answer_texts), dtype=object),
                dtype=string_dtype,
            )
            task_metadata.create_dataset(
                "char_spans_json",
                data=np.asarray(list(char_spans_json), dtype=object),
                dtype=string_dtype,
            )
            for dataset_name, values in {
                "messages_json": messages_json,
                "rendered_texts": rendered_texts,
                "source_datasets": source_datasets,
                "raw_labels_json": raw_labels_json,
                "label_schemas": label_schemas,
                "example_metadata_json": example_metadata_json,
            }.items():
                task_metadata.create_dataset(
                    dataset_name,
                    data=np.asarray(list(values), dtype=object),
                    dtype=string_dtype,
                )

            token_rows = len(labels)
            for layer in sorted(layers):
                tensors = list(activations_by_layer[layer])
                data = (
                    _tensor_sequence_to_numpy(tensors, storage_dtype=storage_dtype)
                    if tensors
                    else np.empty((0, 0), dtype=np.dtype(storage_dtype))
                )
                if data.shape[0] != token_rows:
                    raise ValueError(
                        f"layer_{layer} has {data.shape[0]} rows, expected {token_rows} from metadata."
                    )
                group = handle.require_group(f"layer_{layer}")
                dataset = group.create_dataset(
                    task,
                    data=data,
                    compression=_normalize_compression_mode(compression),
                )
                dataset.attrs["surface"] = surface_names.get(
                    layer, f"decoder_layer_{layer}_answer_tokens"
                )

            if logits_by_batch:
                logits = _tensor_sequence_to_numpy(
                    logits_by_batch, storage_dtype=storage_dtype
                )
                logits_group = handle.require_group("logits")
                logits_dataset = logits_group.create_dataset(
                    task,
                    data=logits,
                    compression=_normalize_compression_mode(compression),
                )
                logits_dataset.attrs["position_convention"] = (
                    "next-token logits at answer token position minus 1"
                )
    except Exception:
        shard_path.unlink(missing_ok=True)
        raise

    return shard_path


def _tensor_sequence_to_numpy(
    tensors: Sequence[Any],
    *,
    storage_dtype: Literal["float16", "float32"],
) -> Any:
    import numpy as np
    import torch

    if storage_dtype == "float16":
        torch_dtype = torch.float16
        numpy_dtype = np.float16
    elif storage_dtype == "float32":
        torch_dtype = torch.float32
        numpy_dtype = np.float32
    else:
        raise ValueError(f"Unsupported activation storage dtype: {storage_dtype!r}")
    return (
        torch.cat(list(tensors), dim=0)
        .detach()
        .cpu()
        .to(dtype=torch_dtype)
        .numpy()
        .astype(numpy_dtype, copy=False)
    )


def _normalize_compression_mode(
    compression: Literal["gzip", "lzf", "none"],
) -> str | None:
    """Map compression tokens to h5py-compatible values."""
    if compression == "none":
        return None
    return compression


def merge_activation_hdf5_shards(
    shard_paths: Sequence[Path],
    *,
    output_path: Path,
    overwrite: bool = False,
    compression: Literal["gzip", "lzf", "none"] = "lzf",
    merge_strategy: Literal["auto", "concat", "copy-disjoint"] = "auto",
    expected_queue_plan_id: str | None = None,
    require_queue_plan_id: bool = False,
    force_stale_merge_lock: bool = False,
) -> ActivationMergeSummary:
    import h5py
    import numpy as np

    shard_paths = tuple(Path(path) for path in shard_paths)
    if not shard_paths:
        raise ValueError("At least one activation shard is required.")
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Activation shard(s) not found: {missing}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use overwrite=True to replace it."
        )
    if merge_strategy not in {"auto", "concat", "copy-disjoint"}:
        raise ValueError(f"Unsupported activation merge strategy: {merge_strategy!r}")
    queue_plan_ids = _verify_activation_shard_queue_plan_ids(
        shard_paths,
        expected_queue_plan_id=expected_queue_plan_id,
        require_queue_plan_id=require_queue_plan_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merge_lock = acquire_lock(
        output_path.with_name(f"{output_path.name}.merge.lock"),
        lock_payload(
            run_id=new_run_id(),
            queue_plan_id=expected_queue_plan_id or ",".join(queue_plan_ids) or None,
            command=command_line(),
            kind="activation-merge",
            extra={"output_path": str(output_path)},
        ),
        force_stale_lock=force_stale_merge_lock,
    )
    signal_cleanup = install_signal_cleanup(merge_lock)
    try:
        if merge_strategy in {
            "auto",
            "copy-disjoint",
        } and _activation_shards_have_disjoint_tasks(shard_paths):
            return _copy_disjoint_activation_hdf5_tasks(
                shard_paths,
                output_path=output_path,
                overwrite=overwrite,
            )
        if merge_strategy == "copy-disjoint":
            raise FileExistsError(
                "Cannot copy-disjoint merge activation HDF5 files because at least one task is duplicated."
            )

        tmp_file = tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.merge.",
            suffix=".h5",
            dir=output_path.parent,
            delete=False,
        )
        tmp_file.close()
        tmp_path = Path(tmp_file.name)

        examples_by_task: dict[str, int] = {}
        token_rows_by_task: dict[str, int] = {}
        try:
            with (
                h5py.File(shard_paths[0], "r") as first,
                h5py.File(tmp_path, "w") as target,
            ):
                _copy_root_attrs(target, first)
                target.attrs["created_at"] = datetime.now(UTC).isoformat()
                target.attrs["merged_at"] = datetime.now(UTC).isoformat()
                target.attrs["merged_shard_count"] = len(shard_paths)
                target.attrs["source_output_path"] = str(output_path)
                target.attrs["merged_compression"] = compression
                target.attrs["merged_strategy"] = "concat"
                target.attrs["merged_queue_plan_ids_json"] = json.dumps(
                    queue_plan_ids, sort_keys=True
                )

                for shard_path in shard_paths[1:]:
                    with h5py.File(shard_path, "r") as shard:
                        _verify_root_attrs_compatible(first, shard)

                tasks = _tasks_in_activation_shards(shard_paths)
                for task in tasks:
                    merged = _collect_merged_activation_task(shard_paths, task=task)
                    examples_by_task[task] = len(merged["example_source_indices"])
                    token_rows_by_task[task] = len(merged["labels"])

                    metadata = target.require_group("metadata")
                    task_metadata = metadata.create_group(task)
                    for attr_name, attr_value in merged["attrs"].items():
                        task_metadata.attrs[attr_name] = attr_value
                    task_metadata.attrs["merged_at"] = target.attrs["merged_at"]
                    task_metadata.attrs["merged_shard_count"] = int(
                        merged["shard_count"]
                    )
                    task_metadata.attrs["merged_source_paths_json"] = json.dumps(
                        [str(path) for path in merged["source_paths"]],
                        sort_keys=True,
                    )
                    task_metadata.attrs["merged_example_count"] = examples_by_task[task]
                    task_metadata.attrs["merged_token_rows"] = token_rows_by_task[task]

                    for dataset_name in _TOKEN_METADATA_DATASETS:
                        task_metadata.create_dataset(
                            dataset_name, data=np.asarray(merged[dataset_name])
                        )
                    for dataset_name in _EXAMPLE_INT_METADATA_DATASETS:
                        task_metadata.create_dataset(
                            dataset_name, data=np.asarray(merged[dataset_name])
                        )
                    string_dtype = h5py.string_dtype(encoding="utf-8")
                    for dataset_name in _EXAMPLE_STRING_METADATA_DATASETS:
                        task_metadata.create_dataset(
                            dataset_name,
                            data=np.asarray(merged[dataset_name], dtype=object),
                            dtype=string_dtype,
                        )

                    for layer, payload in sorted(merged["layers"].items()):
                        layer_group = target.require_group(f"layer_{layer}")
                        dataset = layer_group.create_dataset(
                            task,
                            data=payload["data"],
                            compression=_normalize_compression_mode(compression),
                        )
                        for attr_name, attr_value in payload["attrs"].items():
                            dataset.attrs[attr_name] = attr_value

                    if merged["logits"] is not None:
                        logits_group = target.require_group("logits")
                        logits_dataset = logits_group.create_dataset(
                            task,
                            data=merged["logits"]["data"],
                            compression=_normalize_compression_mode(compression),
                        )
                        for attr_name, attr_value in merged["logits"]["attrs"].items():
                            logits_dataset.attrs[attr_name] = attr_value

            output_path.unlink(missing_ok=True)
            tmp_path.replace(output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return ActivationMergeSummary(
            output_path=output_path,
            shard_paths=shard_paths,
            tasks=tuple(sorted(examples_by_task)),
            examples_by_task=examples_by_task,
            token_rows_by_task=token_rows_by_task,
        )
    finally:
        signal_cleanup.restore()
        merge_lock.release()


def _activation_shards_have_disjoint_tasks(shard_paths: Sequence[Path]) -> bool:
    task_owners: dict[str, Path] = {}
    for shard_path in shard_paths:
        for task in _tasks_in_activation_shard(shard_path):
            if task in task_owners:
                return False
            task_owners[task] = shard_path
    return bool(task_owners)


def _activation_shard_queue_plan_id(shard_path: Path) -> str | None:
    import h5py

    with h5py.File(shard_path, "r") as shard:
        value = shard.attrs.get("queue_plan_id")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str) and value:
            return value
    return None


def _verify_activation_shard_queue_plan_ids(
    shard_paths: Sequence[Path],
    *,
    expected_queue_plan_id: str | None,
    require_queue_plan_id: bool,
) -> tuple[str, ...]:
    queue_plan_ids: list[str] = []
    for shard_path in shard_paths:
        queue_plan_id = _activation_shard_queue_plan_id(shard_path)
        if queue_plan_id is None:
            if require_queue_plan_id:
                raise FileExistsError(
                    f"Activation shard is missing queue_plan_id: {shard_path}"
                )
            continue
        if (
            expected_queue_plan_id is not None
            and queue_plan_id != expected_queue_plan_id
        ):
            raise FileExistsError(
                f"Activation shard queue_plan_id differs for {shard_path}: "
                f"{queue_plan_id} != {expected_queue_plan_id}"
            )
        queue_plan_ids.append(queue_plan_id)
    unique_ids = tuple(sorted(set(queue_plan_ids)))
    if expected_queue_plan_id is not None and not unique_ids:
        raise FileExistsError(
            f"No activation shard has expected queue_plan_id={expected_queue_plan_id}."
        )
    return unique_ids


def _tasks_in_activation_shard(shard_path: Path) -> tuple[str, ...]:
    import h5py

    with h5py.File(shard_path, "r") as shard:
        return _tasks_in_activation_handle(shard)


def _tasks_in_activation_handle(handle: Any) -> tuple[str, ...]:
    import h5py

    metadata = handle.get("metadata")
    if metadata is None:
        return ()
    return tuple(
        sorted(
            str(name) for name, item in metadata.items() if isinstance(item, h5py.Group)
        )
    )


def _copy_disjoint_activation_hdf5_tasks(
    shard_paths: Sequence[Path],
    *,
    output_path: Path,
    overwrite: bool,
) -> ActivationMergeSummary:
    import h5py

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.copy-disjoint.",
        suffix=".h5",
        dir=output_path.parent,
        delete=False,
    )
    tmp_file.close()
    tmp_path = Path(tmp_file.name)

    examples_by_task: dict[str, int] = {}
    token_rows_by_task: dict[str, int] = {}
    copied_tasks: list[str] = []
    try:
        with (
            h5py.File(shard_paths[0], "r") as first,
            h5py.File(tmp_path, "w") as target,
        ):
            _copy_root_attrs(target, first)
            merged_at = datetime.now(UTC).isoformat()
            target.attrs["created_at"] = merged_at
            target.attrs["merged_at"] = merged_at
            target.attrs["merged_shard_count"] = len(shard_paths)
            target.attrs["source_output_path"] = str(output_path)
            target.attrs["merged_compression"] = "preserved"
            target.attrs["merged_strategy"] = "copy-disjoint"
            target.attrs["merged_queue_plan_ids_json"] = json.dumps(
                _verify_activation_shard_queue_plan_ids(
                    shard_paths,
                    expected_queue_plan_id=None,
                    require_queue_plan_id=False,
                ),
                sort_keys=True,
            )

            for shard_path in shard_paths:
                with h5py.File(shard_path, "r") as shard:
                    _verify_root_attrs_compatible(first, shard)
                    for task in _tasks_in_activation_handle(shard):
                        if task in copied_tasks:
                            raise FileExistsError(
                                f"Cannot copy-disjoint merge duplicate task {task!r}."
                            )
                        _copy_disjoint_activation_task(
                            target=target,
                            shard=shard,
                            shard_path=shard_path,
                            task=task,
                            merged_at=merged_at,
                            examples_by_task=examples_by_task,
                            token_rows_by_task=token_rows_by_task,
                        )
                        copied_tasks.append(task)

        output_path.unlink(missing_ok=True)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return ActivationMergeSummary(
        output_path=output_path,
        shard_paths=tuple(shard_paths),
        tasks=tuple(sorted(copied_tasks)),
        examples_by_task=examples_by_task,
        token_rows_by_task=token_rows_by_task,
    )


def _copy_disjoint_activation_task(
    *,
    target: Any,
    shard: Any,
    shard_path: Path,
    task: str,
    merged_at: str,
    examples_by_task: dict[str, int],
    token_rows_by_task: dict[str, int],
) -> None:
    metadata = target.require_group("metadata")
    if task in metadata:
        raise FileExistsError(f"Cannot copy-disjoint merge duplicate metadata/{task}.")
    shard.copy(f"metadata/{task}", metadata, name=task)
    task_metadata = metadata[task]
    token_rows = int(task_metadata["labels"].shape[0])
    examples = int(task_metadata["example_labels"].shape[0])
    token_rows_by_task[task] = token_rows
    examples_by_task[task] = examples
    task_metadata.attrs["merged_at"] = merged_at
    task_metadata.attrs["merged_shard_count"] = 1
    task_metadata.attrs["merged_source_paths_json"] = json.dumps(
        [str(shard_path)], sort_keys=True
    )
    task_metadata.attrs["merged_example_count"] = examples
    task_metadata.attrs["merged_token_rows"] = token_rows

    for layer in _layer_indices_for_task(shard, task_metadata, task=task):
        source_path = f"layer_{layer}/{task}"
        if source_path not in shard:
            raise KeyError(f"Missing activation dataset in shard: {source_path}")
        layer_group = target.require_group(f"layer_{layer}")
        if task in layer_group:
            raise FileExistsError(
                f"Cannot copy-disjoint merge duplicate layer_{layer}/{task}."
            )
        shard.copy(source_path, layer_group, name=task)

    logits_path = f"logits/{task}"
    if logits_path in shard:
        logits_group = target.require_group("logits")
        if task in logits_group:
            raise FileExistsError(
                f"Cannot copy-disjoint merge duplicate logits/{task}."
            )
        shard.copy(logits_path, logits_group, name=task)


_TOKEN_METADATA_DATASETS = (
    "source_indices",
    "output_indices",
    "labels",
    "example_splits",
    "example_indices",
    "token_positions",
    "logit_positions",
)
_EXAMPLE_INT_METADATA_DATASETS = (
    "example_source_indices",
    "example_output_indices",
    "example_labels",
    "example_token_counts",
)
_EXAMPLE_STRING_METADATA_DATASETS = (
    "detected_answer_texts",
    "decoded_answer_texts",
    "char_spans_json",
    "messages_json",
    "rendered_texts",
    "source_datasets",
    "raw_labels_json",
    "label_schemas",
    "example_metadata_json",
)


def _tasks_in_activation_shards(shard_paths: Sequence[Path]) -> tuple[str, ...]:
    tasks: set[str] = set()
    for shard_path in shard_paths:
        tasks.update(_tasks_in_activation_shard(shard_path))
    if not tasks:
        raise ValueError("No metadata task groups found in activation shards.")
    return tuple(sorted(tasks))


def _collect_merged_activation_task(
    shard_paths: Sequence[Path], *, task: str
) -> dict[str, Any]:
    import h5py
    import numpy as np

    token_values: dict[str, list[Any]] = {name: [] for name in _TOKEN_METADATA_DATASETS}
    example_values: dict[str, list[Any]] = {
        name: []
        for name in (
            *_EXAMPLE_INT_METADATA_DATASETS,
            *_EXAMPLE_STRING_METADATA_DATASETS,
        )
    }
    example_splits = [0]
    layers: dict[int, dict[str, Any]] = {}
    logits_payloads: list[Any] = []
    logits_attrs: dict[str, Any] | None = None
    first_attrs: dict[str, Any] | None = None
    seen_examples: set[tuple[int, int]] = set()
    source_paths: list[Path] = []

    for shard_path in shard_paths:
        with h5py.File(shard_path, "r") as shard:
            if f"metadata/{task}" not in shard:
                continue
            source_paths.append(shard_path)
            meta = shard[f"metadata/{task}"]
            if first_attrs is None:
                first_attrs = dict(meta.attrs.items())
            else:
                _verify_merge_task_attrs_compatible(
                    first_attrs, dict(meta.attrs.items()), task=task
                )

            splits = meta["example_splits"][...].astype(np.int64)
            source_indices = meta["example_source_indices"][...].astype(np.int64)
            output_indices = meta["example_output_indices"][...].astype(np.int64)
            layers_in_shard = _layer_indices_for_task(shard, meta, task=task)
            for layer in layers_in_shard:
                dataset = shard[f"layer_{layer}/{task}"]
                payload = layers.setdefault(
                    layer, {"pieces": [], "attrs": dict(dataset.attrs.items())}
                )
                if payload["attrs"].get("surface") != dataset.attrs.get("surface"):
                    raise FileExistsError(
                        f"Cannot merge {task}: layer_{layer} surface attr differs."
                    )

            if f"logits/{task}" in shard:
                logits_dataset = shard[f"logits/{task}"]
                logits_attrs = logits_attrs or dict(logits_dataset.attrs.items())
                if logits_attrs != dict(logits_dataset.attrs.items()):
                    raise FileExistsError(f"Cannot merge {task}: logits attrs differ.")
                logits_payloads.append(logits_dataset[...])

            for example_idx, (source_index, output_index) in enumerate(
                zip(source_indices, output_indices, strict=True)
            ):
                key = (int(source_index), int(output_index))
                if key in seen_examples:
                    raise FileExistsError(
                        f"Cannot merge {task}: duplicate example source_index={source_index} output_index={output_index}."
                    )
                seen_examples.add(key)
                start = int(splits[example_idx])
                end = int(splits[example_idx + 1])
                token_count = end - start

                for dataset_name in (
                    "source_indices",
                    "output_indices",
                    "labels",
                    "token_positions",
                ):
                    token_values[dataset_name].extend(
                        meta[dataset_name][start:end].tolist()
                    )
                if "logit_positions" in meta:
                    logit_positions = meta["logit_positions"][...]
                    if len(logit_positions) == 0:
                        pass
                    elif len(logit_positions) == len(meta["labels"]):
                        token_values["logit_positions"].extend(
                            logit_positions[start:end].tolist()
                        )
                    else:
                        raise ValueError(
                            f"Cannot merge {task}: logit_positions has {len(logit_positions)} rows "
                            f"but labels has {len(meta['labels'])}."
                        )
                token_values["example_indices"].extend(
                    [len(example_splits) - 1] * token_count
                )
                example_splits.append(example_splits[-1] + token_count)

                for dataset_name in _EXAMPLE_INT_METADATA_DATASETS:
                    example_values[dataset_name].append(
                        int(meta[dataset_name][example_idx])
                    )
                for dataset_name in _EXAMPLE_STRING_METADATA_DATASETS:
                    example_values[dataset_name].append(
                        _decode_hdf5_scalar(meta[dataset_name][example_idx])
                    )
                for layer in layers_in_shard:
                    layers[layer]["pieces"].append(
                        shard[f"layer_{layer}/{task}"][start:end]
                    )

    if first_attrs is None:
        raise ValueError(f"No shards contained task {task!r}.")

    token_values["example_splits"] = example_splits
    merged_layers = {
        layer: {
            "data": np.concatenate(payload["pieces"], axis=0)
            if payload["pieces"]
            else np.empty((0, 0)),
            "attrs": payload["attrs"],
        }
        for layer, payload in layers.items()
    }
    for layer, payload in merged_layers.items():
        if payload["data"].shape[0] != len(token_values["labels"]):
            raise ValueError(
                f"Merged layer_{layer}/{task} has {payload['data'].shape[0]} rows, "
                f"expected {len(token_values['labels'])}."
            )

    merged: dict[str, Any] = {
        **token_values,
        **example_values,
        "attrs": first_attrs,
        "layers": merged_layers,
        "logits": None,
        "source_paths": tuple(source_paths),
        "shard_count": len(source_paths),
    }
    if logits_payloads:
        merged["logits"] = {
            "data": np.concatenate(logits_payloads, axis=0),
            "attrs": logits_attrs or {},
        }
    return merged


def _verify_merge_task_attrs_compatible(
    existing: Mapping[str, Any], incoming: Mapping[str, Any], *, task: str
) -> None:
    for attr_name in (
        "dataset_id",
        "model_id",
        "processor_id",
        "backend",
        "aggregation",
        "activation_storage_dtype",
    ):
        if existing.get(attr_name) != incoming.get(attr_name):
            raise FileExistsError(
                f"Cannot merge {task}: metadata attr {attr_name!r} differs."
            )


def _layer_indices_for_task(
    handle: Any, metadata: Any, *, task: str
) -> tuple[int, ...]:
    raw_layers = metadata.attrs.get("layers_json")
    layers = tuple(int(layer) for layer in json.loads(raw_layers)) if raw_layers else ()
    if layers:
        return layers
    found = []
    for group_name in handle.keys():
        if group_name.startswith("layer_") and task in handle[group_name]:
            found.append(int(group_name.removeprefix("layer_")))
    return tuple(sorted(found))


def _decode_hdf5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _merge_activation_shard(
    *,
    output_path: Path,
    shard_path: Path,
    task: str,
    layers: Sequence[int],
    overwrite: bool,
    resume: bool,
) -> None:
    import h5py

    with h5py.File(shard_path, "r") as shard, h5py.File(output_path, "a") as target:
        if target.keys() and not overwrite:
            _verify_root_attrs_compatible(target, shard)
        if overwrite:
            _delete_task_outputs(target, task)

        metadata = target.require_group("metadata")
        if task in metadata:
            if overwrite:
                del metadata[task]
            elif resume:
                _verify_metadata_compatible(
                    metadata[task], shard[f"metadata/{task}"], task=task
                )
                _merge_resume_metadata_attrs(metadata[task], shard[f"metadata/{task}"])
            else:
                raise FileExistsError(
                    f"{output_path} already contains metadata/{task}."
                )
        if task not in metadata:
            shard.copy(f"metadata/{task}", metadata, name=task)

        for layer in sorted(layers):
            target_group = target.require_group(f"layer_{layer}")
            shard_dataset_path = f"layer_{layer}/{task}"
            if task in target_group:
                if overwrite:
                    del target_group[task]
                elif resume:
                    _verify_dataset_shape_compatible(
                        target_group[task],
                        shard[shard_dataset_path],
                        dataset_name=f"layer_{layer}/{task}",
                    )
                    continue
                else:
                    raise FileExistsError(
                        f"{output_path} already contains layer_{layer}/{task}."
                    )
            shard.copy(shard_dataset_path, target_group, name=task)

        if "logits" in shard and task in shard["logits"]:
            logits_group = target.require_group("logits")
            if task in logits_group:
                if overwrite:
                    del logits_group[task]
                elif resume:
                    _verify_dataset_shape_compatible(
                        logits_group[task],
                        shard[f"logits/{task}"],
                        dataset_name=f"logits/{task}",
                    )
                    return
                else:
                    raise FileExistsError(
                        f"{output_path} already contains logits/{task}."
                    )
            shard.copy(f"logits/{task}", logits_group, name=task)

        _copy_root_attrs(target, shard)


def _delete_task_outputs(handle: Any, task: str) -> None:
    metadata = handle.get("metadata")
    if metadata is not None and task in metadata:
        del metadata[task]

    for group_name in list(handle.keys()):
        if group_name.startswith("layer_") and task in handle[group_name]:
            del handle[group_name][task]

    logits = handle.get("logits")
    if logits is not None and task in logits:
        del logits[task]


def _verify_metadata_compatible(existing: Any, incoming: Any, *, task: str) -> None:
    for attr_name in (
        "dataset_id",
        "model_id",
        "processor_id",
        "backend",
        "preserved_input_keys_json",
        "dataset_metadata_json",
        "aggregation",
    ):
        if existing.attrs.get(attr_name) != incoming.attrs.get(attr_name):
            raise FileExistsError(
                f"Cannot resume {task}: metadata attr {attr_name!r} differs."
            )

    _verify_layer_metadata_compatible(existing, incoming, task=task)

    for dataset_name in (
        "source_indices",
        "output_indices",
        "labels",
        "example_splits",
        "example_indices",
        "token_positions",
        "logit_positions",
        "example_source_indices",
        "example_output_indices",
        "example_labels",
        "example_token_counts",
        "detected_answer_texts",
        "decoded_answer_texts",
        "char_spans_json",
        "messages_json",
        "rendered_texts",
        "source_datasets",
        "raw_labels_json",
        "label_schemas",
        "example_metadata_json",
    ):
        if dataset_name not in existing or dataset_name not in incoming:
            raise FileExistsError(
                f"Cannot resume {task}: missing metadata/{task}/{dataset_name}."
            )
        if existing[dataset_name].shape != incoming[dataset_name].shape:
            raise FileExistsError(
                f"Cannot resume {task}: metadata/{dataset_name} shape mismatch "
                f"{existing[dataset_name].shape} != {incoming[dataset_name].shape}."
            )
        if (existing[dataset_name][...] != incoming[dataset_name][...]).any():
            raise FileExistsError(
                f"Cannot resume {task}: metadata/{dataset_name} differs."
            )


def _verify_layer_metadata_compatible(
    existing: Any, incoming: Any, *, task: str
) -> None:
    existing_layers = set(json.loads(existing.attrs.get("layers_json", "[]")))
    incoming_layers = set(json.loads(incoming.attrs.get("layers_json", "[]")))
    existing_surfaces = json.loads(existing.attrs.get("surface_names_json", "{}"))
    incoming_surfaces = json.loads(incoming.attrs.get("surface_names_json", "{}"))
    for layer in existing_layers & incoming_layers:
        key = str(layer)
        if existing_surfaces.get(key) != incoming_surfaces.get(key):
            raise FileExistsError(
                f"Cannot resume {task}: surface for layer {layer} differs."
            )


def _merge_resume_metadata_attrs(existing: Any, incoming: Any) -> None:
    existing_layers = set(json.loads(existing.attrs.get("layers_json", "[]")))
    incoming_layers = set(json.loads(incoming.attrs.get("layers_json", "[]")))
    merged_layers = sorted(int(layer) for layer in existing_layers | incoming_layers)
    existing_surfaces = json.loads(existing.attrs.get("surface_names_json", "{}"))
    incoming_surfaces = json.loads(incoming.attrs.get("surface_names_json", "{}"))
    merged_surfaces = {**existing_surfaces, **incoming_surfaces}
    existing.attrs["layers_json"] = json.dumps(merged_layers)
    existing.attrs["surface_names_json"] = json.dumps(merged_surfaces, sort_keys=True)


def _verify_dataset_shape_compatible(
    existing: Any, incoming: Any, *, dataset_name: str
) -> None:
    if existing.shape != incoming.shape:
        raise FileExistsError(
            f"Cannot resume {dataset_name}: existing shape {existing.shape} != incoming shape {incoming.shape}."
        )
    if existing.attrs.get("surface") != incoming.attrs.get("surface"):
        raise FileExistsError(f"Cannot resume {dataset_name}: surface attr differs.")
    if (existing[...] != incoming[...]).any():
        raise FileExistsError(
            f"Cannot resume {dataset_name}: existing data differs from regenerated data."
        )


def _verify_root_attrs_compatible(existing: Any, incoming: Any) -> None:
    for attr_name in ("format", "model_id", "processor_id", "backend", "versions_json"):
        if existing.attrs.get(attr_name) != incoming.attrs.get(attr_name):
            raise FileExistsError(
                f"Cannot merge activation shard: root attr {attr_name!r} differs."
            )


def _copy_root_attrs(target: Any, shard: Any) -> None:
    for attr_name in (
        "model_id",
        "processor_id",
        "format",
        "backend",
        "versions_json",
        "created_at",
        "source_output_path",
    ):
        target.attrs[attr_name] = shard.attrs[attr_name]


def _extraction_settings_attrs(
    settings: ActivationExtractionSettings,
) -> dict[str, Any]:
    return {
        "extraction_layers_json": json.dumps(list(settings.layers)),
        "extraction_batch_size": settings.batch_size,
        "extraction_start": settings.start,
        "extraction_limit": "" if settings.limit is None else settings.limit,
        "extraction_verify_masks": settings.verify_masks,
        "extraction_max_length": ""
        if settings.max_length is None
        else settings.max_length,
        "extraction_capture_logits": settings.capture_logits,
        "extraction_resume": settings.resume,
        "extraction_compression": settings.compression,
        "extraction_storage_dtype": settings.storage_dtype,
    }


def _task_metadata_attrs(examples: Sequence[ActivationExample]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    source_kinds = sorted(
        {
            str(example.metadata["sycophancy_source_kind"])
            for example in examples
            if example.metadata and example.metadata.get("sycophancy_source_kind")
        }
    )
    generated_models = sorted(
        {
            str(example.metadata["generated_model"])
            for example in examples
            if example.metadata and example.metadata.get("generated_model")
        }
    )
    if source_kinds:
        attrs["sycophancy_source_kinds_json"] = json.dumps(source_kinds, sort_keys=True)
        if len(source_kinds) == 1:
            attrs["sycophancy_source_kind"] = source_kinds[0]
    if generated_models:
        attrs["sycophancy_generated_models_json"] = json.dumps(
            generated_models, sort_keys=True
        )
        if len(generated_models) == 1:
            attrs["sycophancy_generated_model"] = generated_models[0]
    return attrs


def _jsonable_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in metadata.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Mapping):
        return _jsonable_metadata(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable_value(item) for item in value]
    return str(value)


def _version_metadata() -> dict[str, str]:
    import importlib.metadata
    import platform

    packages = ("torch", "transformers", "qwen-vl-utils", "nnsight", "h5py", "numpy")
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _safe_hdf5_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "task"


def default_activation_output_path(model_id: str) -> Path:
    return (
        Path("artifacts")
        / "activations"
        / f"extracted_feats_all_layers_{qwen_model_slug(model_id)}.h5"
    )
