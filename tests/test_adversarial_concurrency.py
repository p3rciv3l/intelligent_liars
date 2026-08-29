"""Deterministic failure-injection tests for queue and checkpoint boundaries."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from argparse import Namespace
from pathlib import Path

import pytest

from intelligent_liars.durable_checkpoints import (
    advance_latest_checkpoint,
    create_checkpoint_generation,
    resolve_latest_checkpoint,
)
from intelligent_liars.dynamic_queue import claim_unit, requeue_running_units


IDENTITY = {"run_id": "adversarial", "plan_sha256": "a" * 64}


def _source(root: Path, value: bytes) -> Path:
    source = root / f"source-{value.decode()}"
    source.mkdir()
    (source / "weights.bin").write_bytes(value)
    return source


def _unit(value: str) -> dict[str, str]:
    return {"unit_id": value, "queue_plan_id": "plan-1"}


def _decode(payload: dict[str, str]) -> dict[str, str]:
    return payload


def _running_path(run_dir: Path, pending: Path, _unit: dict[str, str]) -> Path:
    return run_dir / "running" / pending.name


def test_two_workers_cannot_reserve_one_pending_unit(tmp_path: Path):
    (tmp_path / "pending").mkdir()
    (tmp_path / "running").mkdir()
    pending = tmp_path / "pending" / "unit.json"
    pending.write_text(json.dumps(_unit("only-once")))

    barrier = threading.Barrier(2)
    results: list[tuple[Path, dict[str, str]] | None] = []

    def worker() -> None:
        barrier.wait()
        results.append(
            claim_unit(
                tmp_path,
                deserialise_unit=lambda value: _decode(value),
                running_path=_running_path,
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(result is not None for result in results) == 1
    assert not pending.exists()
    assert sorted(path.name for path in (tmp_path / "running").glob("*.json")) == [
        "unit.json"
    ]


def test_worker_crash_requeues_running_unit_without_duplicate_pending(tmp_path: Path):
    for name in ("pending", "running", "done", "outputs"):
        (tmp_path / name).mkdir()
    running = tmp_path / "running" / "unit.json"
    running.write_text(json.dumps(_unit("crashed")))

    result = requeue_running_units(
        tmp_path,
        deserialise_unit=_decode,
        serialise_unit=lambda value: value,
        output_path=lambda root, _value: root / "outputs" / "crashed.out",
        output_is_valid=lambda _path, _value: False,
        pending_path=lambda root, _running, _value: root / "pending" / "unit.json",
        done_path=lambda root, _running, _value: root / "done" / "unit.json",
    )

    assert result == (1, 0)
    assert not running.exists()
    assert json.loads((tmp_path / "pending" / "unit.json").read_text()) == _unit("crashed")

    # A second recovery pass must not create a second reservation.
    assert requeue_running_units(
        tmp_path,
        deserialise_unit=_decode,
        serialise_unit=lambda value: value,
        output_path=lambda root, _value: root / "outputs" / "crashed.out",
        output_is_valid=lambda _path, _value: False,
        pending_path=lambda root, _running, _value: root / "pending" / "unit.json",
        done_path=lambda root, _running, _value: root / "done" / "unit.json",
    ) == (0, 0)


def test_checkpoint_pointer_lock_survives_concurrent_advances(tmp_path: Path):
    root = tmp_path / "checkpoints"
    generations = [
        create_checkpoint_generation(
            root,
            identity=IDENTITY,
            generation_id=f"generation-{index}",
            source_dir=_source(tmp_path, str(index).encode()),
        )
        for index in (1, 2)
    ]
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def advance(generation) -> None:
        try:
            barrier.wait()
            advance_latest_checkpoint(
                root,
                generation.generation_id,
                identity=IDENTITY,
                durable_verifier=lambda _candidate: True,
            )
        except BaseException as error:  # make thread failures visible to pytest
            errors.append(error)

    threads = [threading.Thread(target=advance, args=(generation,)) for generation in generations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    latest = resolve_latest_checkpoint(root, expected_identity=IDENTITY)
    assert latest.generation_id in {generation.generation_id for generation in generations}
    pointer = json.loads((root / "latest.json").read_text())
    assert pointer["generation_id"] == latest.generation_id
    assert len(pointer["retained_generation_ids"]) == 1 or pointer["previous_generation_id"] in {
        generation.generation_id for generation in generations
    }


def test_stalled_provider_times_out_without_controller_ack(monkeypatch, tmp_path: Path):
    import importlib.util

    script = Path(__file__).parents[1] / "scripts" / "run_step5_checkpoint_controller.py"
    spec = importlib.util.spec_from_file_location("checkpoint_controller_adversarial", script)
    assert spec and spec.loader
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)

    request = {
        "generation_id": "generation-stalled",
        "request_sha256": "b" * 64,
        "archive_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "size_bytes": 1,
    }

    class NoSuchKey(Exception):
        pass

    class Client:
        exceptions = SimpleNamespace(NoSuchKey=NoSuchKey)

        def head_object(self, **_kwargs):
            raise NoSuchKey()

        def generate_presigned_url(self, *_args, **_kwargs):
            return "https://provider.invalid/put"

    class Exchange:
        def __init__(self):
            self.secrets: dict[str, object] = {}

        def read(self, role: str, _generation: str):
            return request if role == "requests" else None

        def delete(self, *_args):
            return None

        def write_secret(self, _generation, value):
            self.secrets["value"] = value

    clock = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(controller, "time", Namespace(monotonic=lambda: next(clock), sleep=lambda _seconds: None))
    monkeypatch.setattr(controller, "validate_upload_request", lambda value: value)
    exchange = Exchange()
    args = Namespace(
        bucket="bucket", prefix="prefix", url_expiry_seconds=1, poll_seconds=0.1
    )

    with pytest.raises(TimeoutError, match="generation-specific PUT"):
        controller.process_request(exchange, Client(), args, "generation-stalled")
    assert "value" in exchange.secrets
