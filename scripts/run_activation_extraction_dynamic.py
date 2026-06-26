#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from intelligent_liars.models import DEFAULT_MODEL_ID, qwen_model_load_description
from intelligent_liars.run_control import (
    acquire_lock,
    command_line,
    current_git_commit,
    file_sha256,
    install_signal_cleanup,
    lock_payload,
    new_run_id,
    stable_sha256,
)


DEFAULT_GENERATED_MODEL = "qwen3-vl-8b-thinking"
PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExampleKey:
    source_index: int
    output_index: int


@dataclass(frozen=True)
class Unit:
    chunk_id: str
    task: str
    rollout_path: str
    examples: tuple[ExampleKey, ...]
    estimated_chars: int
    attempt: int = 0
    source_type: str = "rollout"
    generated_model: str = DEFAULT_GENERATED_MODEL
    source_sha256: str | None = None
    example_manifest_sha256: str | None = None
    dynamic_run_id: str = ""
    queue_plan_id: str = ""

    @property
    def key(self) -> str:
        return f"{self.chunk_id}__attempt-{self.attempt:02d}"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def serialise_unit(unit: Unit) -> dict[str, Any]:
    payload = asdict(unit)
    payload["examples"] = [asdict(example) for example in unit.examples]
    return payload


def deserialise_unit(payload: dict[str, Any]) -> Unit:
    return Unit(
        chunk_id=str(payload["chunk_id"]),
        task=str(payload["task"]),
        rollout_path=str(payload["rollout_path"]),
        examples=tuple(ExampleKey(**example) for example in payload["examples"]),
        estimated_chars=int(payload["estimated_chars"]),
        attempt=int(payload.get("attempt", 0)),
        source_type=str(payload.get("source_type", "rollout")),
        generated_model=str(payload.get("generated_model", DEFAULT_GENERATED_MODEL)),
        source_sha256=payload.get("source_sha256"),
        example_manifest_sha256=payload.get("example_manifest_sha256"),
        dynamic_run_id=str(payload.get("dynamic_run_id", "")),
        queue_plan_id=str(payload.get("queue_plan_id", "")),
    )


def ensure_dirs(run_dir: Path) -> None:
    for name in ("pending", "running", "done", "failed", "completed", "outputs", "logs", "heartbeats"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def create_initial_queue(
    args: argparse.Namespace,
    *,
    overwrite_queue: bool,
    run_id: str = "",
    queue_plan_id: str = "",
    units: list[Unit] | None = None,
) -> list[Unit]:
    run_dir = Path(args.run_dir).resolve()
    ensure_dirs(run_dir)
    if overwrite_queue:
        for subdir in ("pending", "running", "done", "failed", "completed", "outputs"):
            for path in (run_dir / subdir).glob("*"):
                path.unlink()
    units = units or plan_units_from_args(args)
    existing = queue_unit_ids(run_dir)
    for queue_idx, unit in enumerate(units):
        stamped = stamp_unit_plan(unit, run_id=run_id, queue_plan_id=queue_plan_id)
        if stamped.key in existing:
            continue
        write_json(run_dir / "pending" / f"{queue_idx:05d}-{stamped.key}.json", serialise_unit(stamped))
        existing.add(stamped.key)
    return units


def plan_units_from_args(args: argparse.Namespace) -> list[Unit]:
    return plan_units(
        project_root=Path(args.project_root).resolve(),
        rollout_paths=[Path(path) for path in args.path or []],
        tasks=list(args.task or []),
        generated_model=args.generated_model,
        chunk_chars=args.chunk_chars,
        max_examples_per_chunk=args.max_examples_per_chunk,
        limit=args.limit,
    )


def plan_units(
    *,
    project_root: Path,
    rollout_paths: list[Path],
    tasks: list[str],
    generated_model: str,
    chunk_chars: int,
    max_examples_per_chunk: int,
    limit: int | None,
) -> list[Unit]:
    from intelligent_liars.activations import ActivationDataset

    units: list[Unit] = []
    sources: list[tuple[str, str, Any]] = []
    for rollout_path in rollout_paths:
        resolved = rollout_path if rollout_path.is_absolute() else project_root / rollout_path
        dataset = ActivationDataset.from_rollout(resolved)
        source_id = str(resolved.relative_to(project_root) if resolved.is_relative_to(project_root) else resolved)
        sources.append(("rollout", source_id, dataset))
    for task in tasks:
        dataset = ActivationDataset.from_named_task(
            task,
            project_root=project_root,
            generated_model=generated_model,
        )
        sources.append(("named_task", task, dataset))
    if not sources:
        raise ValueError("At least one --path rollout file or --task named dataset is required.")

    for source_type, source_id, dataset in sources:
        examples = list(dataset.labeled_for_probe())
        if limit is not None:
            examples = examples[:limit]
        pending: list[Any] = []
        pending_chars = 0

        def flush() -> None:
            nonlocal pending, pending_chars
            if not pending:
                return
            chunk_id = f"{len(units):05d}-{dataset.task}"
            units.append(
                Unit(
                    chunk_id=chunk_id,
                    task=dataset.task,
                    rollout_path=source_id,
                    examples=tuple(ExampleKey(example.source_index, example.output_index) for example in pending),
                    estimated_chars=pending_chars,
                    source_type=source_type,
                    generated_model=generated_model,
                    source_sha256=_dataset_source_sha256(dataset),
                    example_manifest_sha256=_example_manifest_sha256(pending),
                )
            )
            pending = []
            pending_chars = 0

        for example in sorted(examples, key=estimated_example_chars, reverse=True):
            estimated = estimated_example_chars(example)
            if pending and (pending_chars + estimated > chunk_chars or len(pending) >= max_examples_per_chunk):
                flush()
            pending.append(example)
            pending_chars += estimated
            if pending_chars >= chunk_chars or len(pending) >= max_examples_per_chunk:
                flush()
        flush()
    return sorted(units, key=lambda unit: unit.estimated_chars, reverse=True)


def stamp_unit_plan(unit: Unit, *, run_id: str, queue_plan_id: str) -> Unit:
    return Unit(
        chunk_id=unit.chunk_id,
        task=unit.task,
        rollout_path=unit.rollout_path,
        examples=unit.examples,
        estimated_chars=unit.estimated_chars,
        attempt=unit.attempt,
        source_type=unit.source_type,
        generated_model=unit.generated_model,
        source_sha256=unit.source_sha256,
        example_manifest_sha256=unit.example_manifest_sha256,
        dynamic_run_id=run_id,
        queue_plan_id=queue_plan_id,
    )


def _dataset_source_sha256(dataset: Any) -> str | None:
    source_path = getattr(dataset, "source_path", None)
    if source_path is None:
        return None
    path = Path(source_path)
    if not path.exists() or not path.is_file():
        return None
    return file_sha256(path)


def _example_manifest_sha256(examples: list[Any]) -> str:
    payload = []
    for example in examples:
        payload.append(
            {
                "source_index": int(example.source_index),
                "output_index": int(example.output_index),
                "label": int(example.label),
                "detected_text_sha256": _text_sha256(example.detected_text),
                "messages_sha256": stable_sha256(example.messages),
                "source_dataset": example.source_dataset,
                "raw_label": None if example.raw_label is None else str(example.raw_label),
                "label_schema": getattr(example.label_schema, "value", str(example.label_schema)),
            }
        )
    return stable_sha256(payload)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def estimated_example_chars(example: Any) -> int:
    message_chars = 0
    for message in example.messages:
        message_chars += content_chars(message.get("content", ""))
    return max(1, message_chars, len(example.detected_text))


def content_chars(content: Any) -> int:
    if isinstance(content, list):
        return sum(content_chars(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return len(str(content.get("text", "")))
        return 0
    return len(str(content))


def unit_plan_payload(unit: Unit) -> dict[str, Any]:
    return {
        "chunk_id": unit.chunk_id,
        "task": unit.task,
        "rollout_path": unit.rollout_path,
        "examples": [asdict(example) for example in unit.examples],
        "estimated_chars": unit.estimated_chars,
        "attempt": unit.attempt,
        "source_type": unit.source_type,
        "generated_model": unit.generated_model,
        "source_sha256": unit.source_sha256,
        "example_manifest_sha256": unit.example_manifest_sha256,
    }


def build_plan_manifest(args: argparse.Namespace, units: list[Unit]) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "activation_extraction_dynamic",
        "model": {
            "model_id": DEFAULT_MODEL_ID,
            "load": qwen_model_load_description(),
            "generated_model": args.generated_model,
        },
        "sources": {
            "rollout_paths": [str(path) for path in args.path or []],
            "named_tasks": list(args.task or []),
            "limit": args.limit,
        },
        "chunking": {
            "chunk_chars": args.chunk_chars,
            "max_examples_per_chunk": args.max_examples_per_chunk,
        },
        "extraction": {
            "layers": args.layers,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "verify_masks": not args.no_verify_masks,
            "capture_logits": args.capture_logits,
            "storage_dtype": args.storage_dtype,
            "compression": args.compression,
        },
        "units": [unit_plan_payload(unit) for unit in units],
    }


def prepare_queue_plan(args: argparse.Namespace, units: list[Unit], *, overwrite_queue: bool) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    manifest = build_plan_manifest(args, units)
    queue_plan_id = stable_sha256(manifest)
    plan_path = run_dir / "queue_plan.json"
    if plan_path.exists() and not overwrite_queue:
        existing = read_json(plan_path)
        if existing.get("queue_plan_id") != queue_plan_id:
            raise SystemExit(
                f"Existing queue plan differs for {run_dir}: "
                f"{existing.get('queue_plan_id')} != {queue_plan_id}. "
                "Use a fresh run directory or --overwrite-queue intentionally."
            )
        return existing
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": new_run_id(),
        "queue_plan_id": queue_plan_id,
        "created_at": now(),
        "git_commit": current_git_commit(Path(args.project_root).resolve()),
        "command": command_line(),
        "manifest": manifest,
    }


def write_queue_plan(run_dir: Path, plan: dict[str, Any]) -> None:
    write_json(run_dir / "queue_plan.json", plan)


def load_queue_plan(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "queue_plan.json")


def planned_unit_keys_from_plan(plan: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for item in plan.get("manifest", {}).get("units", []):
        chunk_id = str(item["chunk_id"])
        attempt = int(item.get("attempt", 0))
        keys.add(f"{chunk_id}__attempt-{attempt:02d}")
    return keys


def queue_unit_ids(run_dir: Path) -> set[str]:
    existing: set[str] = set()
    for name in ("pending", "running", "done", "failed"):
        for path in (run_dir / name).glob("*.json"):
            try:
                existing.add(deserialise_unit(read_json(path)).key)
            except Exception:
                existing.add(path.stem)
    for path in (run_dir / "outputs").glob("*.h5"):
        unit_key = _output_shard_unit_key(path)
        if unit_key is not None:
            existing.add(unit_key)
    return existing


def run_dir_has_queue_state(run_dir: Path) -> bool:
    for name, pattern in (
        ("pending", "*.json"),
        ("running", "*.json"),
        ("done", "*.json"),
        ("failed", "*.json"),
        ("outputs", "*.h5"),
    ):
        if any((run_dir / name).glob(pattern)):
            return True
    return False


def claim_unit(run_dir: Path) -> tuple[Path, Unit] | None:
    for path in sorted((run_dir / "pending").glob("*.json")):
        try:
            unit = deserialise_unit(read_json(path))
        except Exception:
            continue
        dest = run_dir / "running" / path.name
        try:
            os.replace(path, dest)
            return dest, unit
        except OSError:
            continue
    return None


def requeue_running_units(run_dir: Path) -> tuple[int, int]:
    """Return stale running units to the pending queue before restart."""
    requeued = 0
    promoted = 0
    for running_path in sorted((run_dir / "running").glob("*.json")):
        try:
            unit = deserialise_unit(read_json(running_path))
        except Exception:
            running_path.unlink(missing_ok=True)
            continue
        output_path = run_dir / "outputs" / f"{unit.key}.h5"
        if _shard_is_complete(output_path, expected_unit=unit):
            done_path = run_dir / "done" / running_path.name
            if not done_path.exists():
                os.replace(running_path, done_path)
            else:
                running_path.unlink()
            promoted += 1
            continue

        pending_path = run_dir / "pending" / running_path.name
        if pending_path.exists():
            running_path.unlink()
            continue

        os.replace(running_path, pending_path)
        requeued += 1

    return requeued, promoted


def repair_stale_done_units(run_dir: Path) -> tuple[int, int]:
    """Move done units without a valid shard output back to pending.

    Returns:
        A tuple of (requeued_units, removed_duplicate_units).
    """
    done_dir = run_dir / "done"
    pending_dir = run_dir / "pending"
    requeued = 0
    removed = 0
    seen_unit_ids: set[str] = set()
    for done_path in sorted(done_dir.glob("*.json")):
        try:
            unit = deserialise_unit(read_json(done_path))
        except Exception:
            done_path.unlink(missing_ok=True)
            removed += 1
            continue

        unit_id = unit.key
        if unit_id in seen_unit_ids:
            done_path.unlink(missing_ok=True)
            removed += 1
            continue
        seen_unit_ids.add(unit_id)

        output_path = run_dir / "outputs" / f"{unit_id}.h5"
        if not _shard_is_complete(output_path, expected_unit=unit):
            done_path.unlink(missing_ok=True)
            pending_path = pending_dir / f"{unit_id}.json"
            if not pending_path.exists():
                write_json(pending_path, serialise_unit(unit))
                requeued += 1
    return requeued, removed


def _shard_is_complete(path: Path, *, expected_unit: Unit | None = None, expected_queue_plan_id: str | None = None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            if expected_unit is not None and expected_unit.queue_plan_id:
                if str(handle.attrs.get("dynamic_unit_key", "")) != expected_unit.key:
                    return False
                if expected_unit.queue_plan_id and str(handle.attrs.get("queue_plan_id", "")) != expected_unit.queue_plan_id:
                    return False
            if expected_queue_plan_id is not None:
                if str(handle.attrs.get("queue_plan_id", "")) != expected_queue_plan_id:
                    return False
            metadata = handle.get("metadata")
            if not isinstance(metadata, h5py.Group) or len(metadata.keys()) == 0:
                return False

            if (layer_container := handle.get("layers")) is not None:
                if isinstance(layer_container, h5py.Group) and _has_layer_dataset(layer_container):
                    return True
            return _has_layer_dataset(handle)
    except Exception:
        return False


def _output_shard_unit_key(path: Path, *, expected_queue_plan_id: str | None = None) -> str | None:
    if not _shard_is_complete(path, expected_queue_plan_id=expected_queue_plan_id):
        return None
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            unit_key = handle.attrs.get("dynamic_unit_key")
            if isinstance(unit_key, bytes):
                unit_key = unit_key.decode("utf-8")
            if isinstance(unit_key, str) and unit_key:
                return unit_key
    except Exception:
        return None
    if expected_queue_plan_id is not None:
        return None
    return path.stem


def _has_layer_dataset(group: Any) -> bool:
    import h5py

    for name, item in group.items():
        if not name.startswith("layer_") or not isinstance(item, h5py.Group):
            continue
        if any(isinstance(child, h5py.Dataset) for child in item.values()):
            return True
    return False


def count_state(run_dir: Path) -> dict[str, int]:
    state = {
        name: len(list((run_dir / name).glob("*.json")))
        for name in ("pending", "running", "done", "failed", "completed")
    }
    state["outputs"] = len(list((run_dir / "outputs").glob("*.h5")))
    return state


def audit_run_state(
    run_dir: Path,
    *,
    planned_unit_keys: set[str],
    queue_plan_id: str | None,
) -> dict[str, Any]:
    ensure_dirs(run_dir)
    membership: dict[str, list[str]] = {unit_key: [] for unit_key in planned_unit_keys}
    malformed_queue_files: list[str] = []
    unexpected_units: list[str] = []
    duplicate_units: dict[str, list[str]] = {}
    invalid_output_files: list[str] = []
    duplicate_output_keys: dict[str, list[str]] = {}
    valid_output_units: set[str] = set()
    output_markers_by_key: dict[str, list[str]] = {}

    for state_name in ("pending", "running", "done", "failed"):
        for path in sorted((run_dir / state_name).glob("*.json")):
            try:
                unit = deserialise_unit(read_json(path))
            except Exception:
                malformed_queue_files.append(str(path))
                continue
            marker = f"{state_name}:{path.name}"
            if queue_plan_id is not None and unit.queue_plan_id != queue_plan_id:
                unexpected_units.append(marker)
                continue
            if unit.key not in membership:
                unexpected_units.append(marker)
                continue
            membership[unit.key].append(marker)

    for path in sorted((run_dir / "outputs").glob("*.h5")):
        unit_key = _output_shard_unit_key(path, expected_queue_plan_id=queue_plan_id)
        if unit_key is None:
            invalid_output_files.append(str(path))
            continue
        marker = f"outputs:{path.name}"
        if unit_key not in membership:
            unexpected_units.append(marker)
            continue
        membership[unit_key].append(marker)
        valid_output_units.add(unit_key)
        output_markers_by_key.setdefault(unit_key, []).append(marker)

    for unit_key, markers in sorted(membership.items()):
        state_markers = [marker for marker in markers if not marker.startswith("outputs:")]
        if len(state_markers) > 1:
            duplicate_units[unit_key] = state_markers
    for unit_key, markers in sorted(output_markers_by_key.items()):
        if len(markers) > 1:
            duplicate_output_keys[unit_key] = markers

    missing_units = [unit_key for unit_key, markers in sorted(membership.items()) if not markers]
    done_without_output = [
        unit_key
        for unit_key, markers in sorted(membership.items())
        if any(marker.startswith("done:") for marker in markers)
        and unit_key not in valid_output_units
    ]
    output_without_done = [
        unit_key
        for unit_key, markers in sorted(membership.items())
        if unit_key in valid_output_units
        and not any(marker.startswith("done:") for marker in markers)
    ]
    issues = {
        "malformed_queue_files": malformed_queue_files,
        "invalid_output_files": invalid_output_files,
        "unexpected_units": unexpected_units,
        "missing_units": missing_units,
        "duplicate_units": duplicate_units,
        "done_without_output": done_without_output,
        "output_without_done": output_without_done,
        "duplicate_output_keys": duplicate_output_keys,
    }
    return {
        "ok": not any(bool(value) for value in issues.values()),
        "planned": len(planned_unit_keys),
        "state": count_state(run_dir),
        "valid_outputs": len(valid_output_units),
        "issues": issues,
    }


def build_chunk_dataset(project_root: Path, unit: Unit):
    from intelligent_liars.activations import ActivationDataset

    if unit.source_type == "named_task":
        dataset = ActivationDataset.from_named_task(
            unit.task,
            project_root=project_root,
            generated_model=unit.generated_model,
        )
        resolved = dataset.source_path
    elif unit.source_type == "rollout":
        rollout_path = Path(unit.rollout_path)
        resolved = rollout_path if rollout_path.is_absolute() else project_root / rollout_path
        dataset = ActivationDataset.from_rollout(resolved, task=unit.task)
    else:
        raise ValueError(f"Unsupported unit source_type: {unit.source_type!r}")
    by_key = {
        (example.source_index, example.output_index): example
        for example in dataset.labeled_for_probe()
    }
    examples = []
    for key in unit.examples:
        try:
            examples.append(by_key[(key.source_index, key.output_index)])
        except KeyError as exc:
            raise KeyError(
                f"Chunk {unit.chunk_id} references missing example "
                f"source_index={key.source_index} output_index={key.output_index}."
            ) from exc
    return ActivationDataset(
        task=unit.task,
        examples=tuple(examples),
        source_path=resolved,
        dataset_id=dataset.dataset_id or unit.rollout_path,
    )


def stamp_activation_shard(path: Path, unit: Unit) -> None:
    import h5py

    with h5py.File(path, "a") as handle:
        handle.attrs["dynamic_run_id"] = unit.dynamic_run_id
        handle.attrs["queue_plan_id"] = unit.queue_plan_id
        handle.attrs["dynamic_unit_key"] = unit.key
        handle.attrs["dynamic_chunk_id"] = unit.chunk_id
        handle.attrs["dynamic_source_type"] = unit.source_type
        handle.attrs["dynamic_source_id"] = unit.rollout_path
        if unit.source_sha256:
            handle.attrs["dynamic_source_sha256"] = unit.source_sha256
        if unit.example_manifest_sha256:
            handle.attrs["dynamic_example_manifest_sha256"] = unit.example_manifest_sha256
        metadata = handle.get("metadata")
        if isinstance(metadata, h5py.Group):
            for task_metadata in metadata.values():
                if not isinstance(task_metadata, h5py.Group):
                    continue
                task_metadata.attrs["dynamic_run_id"] = unit.dynamic_run_id
                task_metadata.attrs["queue_plan_id"] = unit.queue_plan_id
                task_metadata.attrs["dynamic_unit_key"] = unit.key
                task_metadata.attrs["dynamic_chunk_id"] = unit.chunk_id


def run_worker(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    gpu = str(args.gpu)
    slot_id = str(args.slot_id or f"gpu-{gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("PYTHONPATH", "src")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("INTELLIGENT_LIARS_PROGRESS", "1")

    from intelligent_liars.activation_backends import TransformersHookBackend
    from intelligent_liars.activations import (
        ActivationExtractionSettings,
        extract_dataset_activations,
        parse_layer_spec,
    )
    from intelligent_liars.models import load_model_and_processor, model_config_from_env

    bundle = load_model_and_processor(model_config_from_env())
    backend = TransformersHookBackend(bundle)
    layer_indices = parse_layer_spec(args.layers, backend.decoder_layer_count())
    heartbeat = run_dir / "heartbeats" / f"worker-{slot_id}.json"
    processed = 0

    while not (run_dir / "STOP").exists():
        write_json(heartbeat, {"time": now(), "gpu": gpu, "slot": slot_id, "state": "waiting", "processed": processed})
        claimed = claim_unit(run_dir)
        if claimed is None:
            time.sleep(args.idle_sleep)
            continue
        unit_path, unit = claimed
        output_path = run_dir / "outputs" / f"{unit.key}.h5"
        staged_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
        started = time.time()
        write_json(
            heartbeat,
            {
                "time": now(),
                "gpu": gpu,
                "slot": slot_id,
                "state": "running",
                "chunk_id": unit.chunk_id,
                "task": unit.task,
                "examples": len(unit.examples),
                "processed": processed,
            },
        )
        try:
            dataset = build_chunk_dataset(project_root, unit)
            settings = ActivationExtractionSettings(
                layers=layer_indices,
                batch_size=args.batch_size,
                verify_masks=not args.no_verify_masks,
                max_length=args.max_length,
                capture_logits=args.capture_logits,
                storage_dtype=args.storage_dtype,
                compression=args.compression,
            )
            summary = extract_dataset_activations(
                bundle=bundle,
                dataset=dataset,
                output_path=staged_output_path,
                settings=settings,
                overwrite=True,
                backend=backend,
            )
            stamp_activation_shard(staged_output_path, unit)
            os.replace(staged_output_path, output_path)
            elapsed = time.time() - started
            write_json(
                run_dir / "completed" / f"{unit.key}.json",
                {
                    "time": now(),
                    "gpu": gpu,
                    "slot": slot_id,
                    "elapsed_seconds": round(elapsed, 3),
                    "unit": serialise_unit(unit),
                    "summary": {
                        "task": summary.task,
                        "examples_seen": summary.examples_seen,
                        "examples_extracted": summary.examples_extracted,
                        "masked_tokens": summary.masked_tokens,
                        "layers": list(summary.layers),
                        "output_path": str(summary.output_path),
                    },
                },
            )
            os.replace(unit_path, run_dir / "done" / unit_path.name)
            processed += 1
            print(
                f"WORKER_DONE time={now()} gpu={gpu} slot={slot_id} chunk={unit.chunk_id} "
                f"task={unit.task} examples={len(unit.examples)} elapsed={elapsed:.1f}s output={output_path}",
                flush=True,
            )
        except Exception as exc:
            staged_output_path.unlink(missing_ok=True)
            write_json(
                run_dir / "failed" / unit_path.name,
                {
                    "time": now(),
                    "gpu": gpu,
                    "slot": slot_id,
                    "error": repr(exc),
                    "unit": serialise_unit(unit),
                },
            )
            unit_path.unlink(missing_ok=True)
            print(
                f"WORKER_FAILED time={now()} gpu={gpu} slot={slot_id} chunk={unit.chunk_id} "
                f"task={unit.task} error={exc!r}",
                flush=True,
            )
    write_json(heartbeat, {"time": now(), "gpu": gpu, "slot": slot_id, "state": "stopped", "processed": processed})
    return 0


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc!r}"


def run_supervisor(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    ensure_dirs(run_dir)
    planned_units = plan_units_from_args(args)
    if not args.overwrite_queue and not (run_dir / "queue_plan.json").exists() and run_dir_has_queue_state(run_dir):
        raise SystemExit(
            f"Existing queue/output state in {run_dir} has no queue_plan.json. "
            "Use --overwrite-queue or a fresh run directory."
        )
    queue_plan = prepare_queue_plan(args, planned_units, overwrite_queue=args.overwrite_queue)
    run_id = str(queue_plan["run_id"])
    queue_plan_id = str(queue_plan["queue_plan_id"])
    planned_unit_keys = {unit.key for unit in planned_units}
    lock = acquire_lock(
        run_dir / "run.lock",
        lock_payload(
            run_id=run_id,
            queue_plan_id=queue_plan_id,
            command=command_line(),
            kind="activation-supervisor",
        ),
        force_stale_lock=args.force_stale_lock,
    )
    signal_cleanup = install_signal_cleanup(lock)
    try:
        write_queue_plan(run_dir, queue_plan)
        if args.overwrite_queue:
            requeued, promoted = 0, 0
            repaired_done, removed_done = 0, 0
        else:
            requeued, promoted = requeue_running_units(run_dir)
            repaired_done, removed_done = repair_stale_done_units(run_dir)
        created_units = create_initial_queue(
            args,
            overwrite_queue=args.overwrite_queue,
            run_id=run_id,
            queue_plan_id=queue_plan_id,
            units=planned_units,
        )
        status = audit_run_state(
            run_dir,
            planned_unit_keys=planned_unit_keys,
            queue_plan_id=queue_plan_id,
        )
        pending_files = sorted((run_dir / "pending").glob("*.json"))
        if not args.plan_only and not args.allow_empty_queue:
            if not pending_files and not list((run_dir / "running").glob("*.json")) and status["valid_outputs"] < status["planned"]:
                print("SUPERVISOR_GUARD pending=0 running=0 but output coverage is incomplete. Inspect status.")
                return 0
        write_json(
            run_dir / "run_metadata.json",
            {
                "time": now(),
                "project_root": str(project_root),
                "run_dir": str(run_dir),
                "paths": args.path,
                "tasks": args.task,
                "generated_model": args.generated_model,
                "gpus": parse_csv(args.gpus),
                "layers": args.layers,
                "batch_size": args.batch_size,
                "chunk_chars": args.chunk_chars,
                "max_examples_per_chunk": args.max_examples_per_chunk,
                "planned_units": len(created_units),
                "run_id": run_id,
                "queue_plan_id": queue_plan_id,
                "storage_dtype": args.storage_dtype,
                "capture_logits": args.capture_logits,
                "compression": args.compression,
                "requeued_running_units": requeued,
                "promoted_running_units": promoted,
                "repaired_done_units": repaired_done,
                "removed_done_units": removed_done,
                "status": status,
            },
        )
        if args.plan_only:
            units = list(created_units)
            queue_by_task = dict(Counter(unit.task for unit in units))
            queue_by_source_type = dict(Counter(unit.source_type for unit in units))
            print(f"PLAN_QUEUE_PLAN_ID={queue_plan_id}")
            print(f"PLAN_PENDING_UNITS={len(units)}")
            print(f"PLAN_BY_TASK={json.dumps(queue_by_task, sort_keys=True)}")
            print(f"PLAN_BY_SOURCE_TYPE={json.dumps(queue_by_source_type, sort_keys=True)}")
            total_examples = sum(len(unit.examples) for unit in units)
            print(f"PLAN_EXAMPLES_TOTAL={total_examples}")
            for unit in sorted(units, key=lambda unit: (unit.task, unit.chunk_id))[:10]:
                print(
                    f"PLAN_UNIT task={unit.task} "
                    f"source={unit.source_type} "
                    f"estimated_chars={unit.estimated_chars} "
                    f"examples={len(unit.examples)}",
                )
            return 0

        env_base = os.environ.copy()
        env_base.setdefault("PYTHONPATH", "src")
        env_base.setdefault("HF_HUB_DISABLE_XET", "1")
        env_base.setdefault("INTELLIGENT_LIARS_PROGRESS", "1")
        workers: list[subprocess.Popen[str]] = []
        for idx, gpu in enumerate(parse_csv(args.gpus)):
            slot_id = f"slot-{idx:02d}-gpu-{gpu}"
            worker_log = run_dir / "logs" / f"worker-{slot_id}.log"
            worker_log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--project-root",
                str(project_root),
                "--run-dir",
                str(run_dir),
                "--gpu",
                gpu,
                "--slot-id",
                slot_id,
                "--layers",
                args.layers,
                "--batch-size",
                str(args.batch_size),
                "--storage-dtype",
                args.storage_dtype,
                "--generated-model",
                args.generated_model,
                "--idle-sleep",
                str(args.idle_sleep),
            ]
            if args.max_length is not None:
                cmd.extend(["--max-length", str(args.max_length)])
            if args.no_verify_masks:
                cmd.append("--no-verify-masks")
            if args.capture_logits:
                cmd.append("--capture-logits")
            cmd.extend(["--compression", args.compression])
            handle = worker_log.open("a")
            workers.append(subprocess.Popen(cmd, cwd=project_root, env=env_base, stdout=handle, stderr=subprocess.STDOUT, text=True))

        print(
            f"SUPERVISOR_START time={now()} run_dir={run_dir} workers={len(workers)} "
            f"queue_plan_id={queue_plan_id}",
            flush=True,
        )
        try:
            while True:
                status = audit_run_state(
                    run_dir,
                    planned_unit_keys=planned_unit_keys,
                    queue_plan_id=queue_plan_id,
                )
                state = status["state"]
                pending = state["pending"]
                running = state["running"]
                failed = state["failed"]
                alive = sum(worker.poll() is None for worker in workers)
                print(
                    f"STATUS time={now()} alive={alive} "
                    f"planned={status['planned']} valid_outputs={status['valid_outputs']} "
                    f"state={json.dumps(state, sort_keys=True)} "
                    f"issues={json.dumps(status['issues'], sort_keys=True)}",
                    flush=True,
                )
                print(gpu_snapshot(), flush=True)
                if failed:
                    raise SystemExit("Extraction worker failed. Inspect run_dir/failed and worker logs.")
                if pending == 0 and running == 0:
                    if status["valid_outputs"] != status["planned"] or not status["ok"]:
                        raise SystemExit(
                            "Queue markers are empty but validated output coverage is incomplete or mismatched. "
                            f"status={json.dumps(status, sort_keys=True)}"
                        )
                    (run_dir / "STOP").write_text(now() + "\n")
                    time.sleep(5)
                    for worker in workers:
                        if worker.poll() is None:
                            worker.terminate()
                    merge_outputs(args, expected_queue_plan_id=queue_plan_id)
                    print(f"SUPERVISOR_DONE time={now()}", flush=True)
                    return 0
                if alive == 0:
                    raise SystemExit("All extraction workers exited before queue completion.")
                time.sleep(args.poll_seconds)
        finally:
            (run_dir / "STOP").write_text(now() + "\n")
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
    finally:
        signal_cleanup.restore()
        lock.release()


def merge_outputs(args: argparse.Namespace, *, expected_queue_plan_id: str | None = None) -> None:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    shard_paths = [
        path
        for path in sorted(run_dir.glob("outputs/*.h5"))
        if _output_shard_unit_key(path, expected_queue_plan_id=expected_queue_plan_id) is not None
    ]
    if not shard_paths:
        raise SystemExit(f"No activation shards found under {run_dir / 'outputs'}.")
    output_path = Path(args.output)
    output_path = output_path if output_path.is_absolute() else project_root / output_path
    from intelligent_liars.activations import merge_activation_hdf5_shards

    summary = merge_activation_hdf5_shards(
        shard_paths,
        output_path=output_path,
        overwrite=args.overwrite_output,
        compression=args.compression,
        expected_queue_plan_id=expected_queue_plan_id,
        require_queue_plan_id=expected_queue_plan_id is not None,
    )
    print(
        f"MERGE_DONE time={now()} output={summary.output_path} "
        f"tasks={json.dumps(summary.examples_by_task, sort_keys=True)}",
        flush=True,
    )


def run_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    plan = load_queue_plan(run_dir)
    report = audit_run_state(
        run_dir,
        planned_unit_keys=planned_unit_keys_from_plan(plan),
        queue_plan_id=str(plan["queue_plan_id"]),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic multi-process activation extraction runner.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", default=".")
    common.add_argument("--run-dir", required=True)
    common.add_argument("--layers", default="all")
    common.add_argument("--batch-size", type=int, default=1)
    common.add_argument("--max-length", type=int)
    common.add_argument("--no-verify-masks", action="store_true")
    common.add_argument("--capture-logits", action="store_true")
    common.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    common.add_argument("--compression", choices=("gzip", "lzf", "none"), default="lzf")
    common.add_argument("--generated-model", default=DEFAULT_GENERATED_MODEL)

    supervisor = sub.add_parser("supervisor", parents=[common])
    supervisor.add_argument("--path", action="append")
    supervisor.add_argument("--task", action="append")
    supervisor.add_argument("--output", required=True)
    supervisor.add_argument("--gpus", default="0")
    supervisor.add_argument("--chunk-chars", type=int, default=12000)
    supervisor.add_argument("--max-examples-per-chunk", type=int, default=16)
    supervisor.add_argument("--limit", type=int)
    supervisor.add_argument("--poll-seconds", type=float, default=30.0)
    supervisor.add_argument("--idle-sleep", type=float, default=5.0)
    supervisor.add_argument("--overwrite-queue", action="store_true")
    supervisor.add_argument("--overwrite-output", action="store_true")
    supervisor.add_argument("--plan-only", action="store_true")
    supervisor.add_argument(
        "--force-stale-lock",
        action="store_true",
        help="Break run.lock only when it is on this host and the recorded PID is dead.",
    )
    supervisor.add_argument(
        "--allow-empty-queue",
        action="store_true",
        help="Allow the supervisor to proceed when pending/running queues are empty.",
    )

    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--gpu", required=True)
    worker.add_argument("--slot-id")
    worker.add_argument("--idle-sleep", type=float, default=5.0)

    status = sub.add_parser("status")
    status.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "supervisor":
        return run_supervisor(args)
    if args.cmd == "worker":
        return run_worker(args)
    if args.cmd == "status":
        return run_status(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
