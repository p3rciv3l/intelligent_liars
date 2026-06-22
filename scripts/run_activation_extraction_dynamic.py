#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    )


def ensure_dirs(run_dir: Path) -> None:
    for name in ("pending", "running", "done", "failed", "completed", "outputs", "logs", "heartbeats"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def create_initial_queue(args: argparse.Namespace, *, overwrite_queue: bool) -> list[Unit]:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    ensure_dirs(run_dir)
    if overwrite_queue:
        for subdir in ("pending", "running", "done", "failed", "completed"):
            for path in (run_dir / subdir).glob("*.json"):
                path.unlink()
    existing = []
    for subdir in ("pending", "running", "done"):
        existing.extend((run_dir / subdir).glob("*.json"))
    if existing:
        return []

    units = plan_units(
        project_root=project_root,
        rollout_paths=[Path(path) for path in args.path],
        chunk_chars=args.chunk_chars,
        max_examples_per_chunk=args.max_examples_per_chunk,
        limit=args.limit,
    )
    for queue_idx, unit in enumerate(units):
        write_json(run_dir / "pending" / f"{queue_idx:05d}-{unit.key}.json", serialise_unit(unit))
    return units


def plan_units(
    *,
    project_root: Path,
    rollout_paths: list[Path],
    chunk_chars: int,
    max_examples_per_chunk: int,
    limit: int | None,
) -> list[Unit]:
    from intelligent_liars.activations import ActivationDataset

    units: list[Unit] = []
    for rollout_path in rollout_paths:
        resolved = rollout_path if rollout_path.is_absolute() else project_root / rollout_path
        dataset = ActivationDataset.from_rollout(resolved)
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
                    rollout_path=str(resolved.relative_to(project_root) if resolved.is_relative_to(project_root) else resolved),
                    examples=tuple(ExampleKey(example.source_index, example.output_index) for example in pending),
                    estimated_chars=pending_chars,
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


def build_chunk_dataset(project_root: Path, unit: Unit):
    from intelligent_liars.activations import ActivationDataset

    rollout_path = Path(unit.rollout_path)
    resolved = rollout_path if rollout_path.is_absolute() else project_root / rollout_path
    dataset = ActivationDataset.from_rollout(resolved, task=unit.task)
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
        dataset_id=str(resolved),
    )


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
            )
            summary = extract_dataset_activations(
                bundle=bundle,
                dataset=dataset,
                output_path=output_path,
                settings=settings,
                overwrite=True,
                backend=backend,
            )
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
    created_units = create_initial_queue(args, overwrite_queue=args.overwrite_queue)
    pending_files = sorted((run_dir / "pending").glob("*.json"))
    write_json(
        run_dir / "run_metadata.json",
        {
            "time": now(),
            "project_root": str(project_root),
            "run_dir": str(run_dir),
            "paths": args.path,
            "gpus": parse_csv(args.gpus),
            "layers": args.layers,
            "batch_size": args.batch_size,
            "chunk_chars": args.chunk_chars,
            "max_examples_per_chunk": args.max_examples_per_chunk,
            "planned_units": len(created_units) or len(pending_files),
            "storage_dtype": args.storage_dtype,
            "capture_logits": args.capture_logits,
        },
    )
    if args.plan_only:
        for path in pending_files:
            print(path.read_text().strip())
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
            "--idle-sleep",
            str(args.idle_sleep),
        ]
        if args.max_length is not None:
            cmd.extend(["--max-length", str(args.max_length)])
        if args.no_verify_masks:
            cmd.append("--no-verify-masks")
        if args.capture_logits:
            cmd.append("--capture-logits")
        handle = worker_log.open("a")
        workers.append(subprocess.Popen(cmd, cwd=project_root, env=env_base, stdout=handle, stderr=subprocess.STDOUT, text=True))

    print(f"SUPERVISOR_START time={now()} run_dir={run_dir} workers={len(workers)}", flush=True)
    try:
        while True:
            pending = len(list((run_dir / "pending").glob("*.json")))
            running = len(list((run_dir / "running").glob("*.json")))
            done = len(list((run_dir / "done").glob("*.json")))
            failed = len(list((run_dir / "failed").glob("*.json")))
            alive = sum(worker.poll() is None for worker in workers)
            print(
                f"STATUS time={now()} alive={alive} pending={pending} running={running} done={done} failed={failed}",
                flush=True,
            )
            print(gpu_snapshot(), flush=True)
            if failed:
                raise SystemExit("Extraction worker failed. Inspect run_dir/failed and worker logs.")
            if pending == 0 and running == 0:
                (run_dir / "STOP").write_text(now() + "\n")
                time.sleep(5)
                for worker in workers:
                    if worker.poll() is None:
                        worker.terminate()
                merge_outputs(args)
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


def merge_outputs(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    shard_paths = sorted(run_dir.glob("outputs/*.h5"))
    if not shard_paths:
        raise SystemExit(f"No activation shards found under {run_dir / 'outputs'}.")
    output_path = Path(args.output)
    output_path = output_path if output_path.is_absolute() else project_root / output_path
    from intelligent_liars.activations import merge_activation_hdf5_shards

    summary = merge_activation_hdf5_shards(shard_paths, output_path=output_path, overwrite=args.overwrite_output)
    print(
        f"MERGE_DONE time={now()} output={summary.output_path} "
        f"tasks={json.dumps(summary.examples_by_task, sort_keys=True)}",
        flush=True,
    )


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

    supervisor = sub.add_parser("supervisor", parents=[common])
    supervisor.add_argument("--path", action="append", required=True)
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

    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--gpu", required=True)
    worker.add_argument("--slot-id")
    worker.add_argument("--idle-sleep", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "supervisor":
        return run_supervisor(args)
    if args.cmd == "worker":
        return run_worker(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
