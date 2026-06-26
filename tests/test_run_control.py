from __future__ import annotations

import json
import os

import pytest

from intelligent_liars import run_control


def test_stable_sha256_ignores_mapping_order():
    left = {"b": [2, 1], "a": {"x": True}}
    right = {"a": {"x": True}, "b": [2, 1]}

    assert run_control.stable_sha256(left) == run_control.stable_sha256(right)


def test_acquire_lock_refuses_existing_live_lock(tmp_path):
    lock_path = tmp_path / "run.lock"
    payload = run_control.lock_payload(run_id="run-1", queue_plan_id="plan-1")
    lock = run_control.acquire_lock(lock_path, payload)

    with pytest.raises(run_control.LockHeldError, match="Lock exists"):
        run_control.acquire_lock(
            lock_path,
            run_control.lock_payload(run_id="run-2", queue_plan_id="plan-2"),
        )

    lock.release()
    assert not lock_path.exists()


def test_force_stale_lock_breaks_same_host_dead_pid(tmp_path):
    lock_path = tmp_path / "run.lock"
    stale_payload = {
        "host": run_control.current_host(),
        "owner_pid": 999_999_999,
        "started_at": "2026-06-25T00:00:00+00:00",
        "run_id": "old-run",
    }
    lock_path.write_text(json.dumps(stale_payload))

    lock = run_control.acquire_lock(
        lock_path,
        run_control.lock_payload(run_id="new-run", queue_plan_id="plan"),
        force_stale_lock=True,
    )

    assert json.loads(lock_path.read_text())["run_id"] == "new-run"
    lock.release()


def test_force_stale_lock_refuses_cross_host_lock(tmp_path):
    lock_path = tmp_path / "run.lock"
    stale_payload = {
        "host": "other-host",
        "owner_pid": 999_999_999,
        "started_at": "2026-06-25T00:00:00+00:00",
        "run_id": "old-run",
    }
    lock_path.write_text(json.dumps(stale_payload))

    with pytest.raises(run_control.LockHeldError, match="cross-host lock"):
        run_control.acquire_lock(
            lock_path,
            run_control.lock_payload(run_id="new-run", queue_plan_id="plan"),
            force_stale_lock=True,
        )

    assert json.loads(lock_path.read_text())["host"] == "other-host"


def test_release_does_not_remove_replaced_lock(tmp_path):
    lock_path = tmp_path / "run.lock"
    lock = run_control.acquire_lock(
        lock_path,
        run_control.lock_payload(run_id="old-run", queue_plan_id="plan"),
    )
    lock_path.write_text(
        json.dumps(
            {
                "host": run_control.current_host(),
                "owner_pid": os.getpid(),
                "started_at": "later",
                "run_id": "new-run",
            }
        )
    )

    lock.release()

    assert json.loads(lock_path.read_text())["run_id"] == "new-run"
