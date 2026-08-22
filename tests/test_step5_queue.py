from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from intelligent_liars.step5_queue import plan_step5_queue


ARMS = [
    {"name": "tiny13"},
    {"name": "tiny64"},
    {"name": "lora1"},
]


def _state(tasks: dict[str, dict] | None = None) -> dict:
    task_records = tasks or {}
    project_instances = [
        {
            "instance_id": record["instance_id"],
            "instance_status": record["instance_status"],
        }
        for record in task_records.values()
        if record.get("instance_id")
        and record.get("instance_status")
        in {"created", "loading", "running", "stopped", "paused"}
    ]
    return {
        "format": "tinylora_step5_queue_state_v1",
        "tasks": task_records,
        "project_instances": project_instances,
    }


def _checkpoint() -> dict[str, object]:
    return {
        "uri": "s3://example/checkpoints/step-25.pt",
        "sha256": "a" * 64,
        "verified": True,
        "step": 25,
    }


def test_empty_queue_launches_at_most_three_independent_workers():
    planned = plan_step5_queue(
        arms=ARMS,
        seeds=[7, 8],
        state=_state(),
        max_concurrency=3,
    )
    assert [action["task_id"] for action in planned["actions"]] == [
        "tiny13.seed-7",
        "tiny64.seed-7",
        "lora1.seed-7",
    ]
    assert {action["action"] for action in planned["actions"]} == {
        "launch_new_instance"
    }
    assert planned["summary"]["planned_active_workers"] == 3


def test_planning_is_idempotent_and_has_no_wall_clock_fields():
    first = plan_step5_queue(
        arms=ARMS,
        seeds=[7],
        state=_state(),
        max_concurrency=2,
    )
    second = plan_step5_queue(
        arms=ARMS,
        seeds=[7],
        state=_state(),
        max_concurrency=2,
    )
    assert first == second
    assert "created_at" not in json.dumps(first)


def test_fourth_worker_is_rejected_even_when_requested():
    with pytest.raises(ValueError, match="at most 3"):
        plan_step5_queue(
            arms=ARMS,
            seeds=[7, 8],
            state=_state(),
            max_concurrency=4,
        )


def test_authoritative_project_inventory_is_required():
    with pytest.raises(ValueError, match="authoritative project_instances"):
        plan_step5_queue(
            arms=ARMS,
            seeds=[7],
            state={"format": "tinylora_step5_queue_state_v1", "tasks": {}},
            max_concurrency=3,
        )


def test_running_workers_reduce_available_queue_slots():
    tasks = {
        "tiny13.seed-7": {
            "status": "running",
            "attempt": 1,
            "instance_id": "instance-a",
            "instance_status": "running",
        },
        "tiny13.seed-8": {
            "status": "running",
            "attempt": 1,
            "instance_id": "instance-b",
            "instance_status": "running",
        },
    }
    planned = plan_step5_queue(
        arms=ARMS,
        seeds=[7, 8],
        state=_state(tasks),
        max_concurrency=3,
    )
    assert len(planned["actions"]) == 1
    assert planned["actions"][0]["task_id"] == "tiny64.seed-7"


def test_software_failure_requires_diagnosis_on_same_instance():
    tasks = {
        "tiny13.seed-7": {
            "status": "failed",
            "attempt": 1,
            "instance_id": "instance-a",
            "instance_status": "stopped",
            "failure": {"class": "software", "diagnosis": "pending"},
            "checkpoint": _checkpoint(),
        }
    }
    planned = plan_step5_queue(
        arms=ARMS[:1],
        seeds=[7],
        state=_state(tasks),
        max_concurrency=3,
    )
    assert planned["actions"] == [
        {
            "action": "diagnose_same_instance",
            "attempt": 1,
            "instance_id": "instance-a",
            "reason": "software_failure_requires_same_host_diagnosis",
            "task_id": "tiny13.seed-7",
        }
    ]


def test_resolved_software_failure_resumes_verified_checkpoint_same_instance():
    tasks = {
        "tiny13.seed-7": {
            "status": "failed",
            "attempt": 1,
            "instance_id": "instance-a",
            "instance_status": "stopped",
            "failure": {"class": "software", "diagnosis": "resolved"},
            "checkpoint": _checkpoint(),
        }
    }
    action = plan_step5_queue(
        arms=ARMS[:1],
        seeds=[7],
        state=_state(tasks),
        max_concurrency=3,
    )["actions"][0]
    assert action["action"] == "resume_same_instance"
    assert action["instance_id"] == "instance-a"
    assert action["attempt"] == 2
    assert action["checkpoint"]["verified"] is True


def test_unverified_checkpoint_is_never_used_for_retry():
    tasks = {
        "tiny13.seed-7": {
            "status": "failed",
            "attempt": 1,
            "instance_id": "instance-a",
            "instance_status": "stopped",
            "failure": {"class": "software", "diagnosis": "resolved"},
            "checkpoint": {
                "uri": "s3://example/unverified.pt",
                "sha256": "b" * 64,
                "verified": False,
            },
        }
    }
    action = plan_step5_queue(
        arms=ARMS[:1],
        seeds=[7],
        state=_state(tasks),
        max_concurrency=3,
    )["actions"][0]
    assert action["action"] == "retry_same_instance_from_start"
    assert "checkpoint" not in action


def test_resolved_software_failure_on_running_instance_does_not_consume_extra_slot():
    tasks = {
        "tiny13.seed-7": {
            "status": "failed",
            "attempt": 1,
            "instance_id": "instance-a",
            "instance_status": "running",
            "failure": {"class": "software", "diagnosis": "resolved"},
            "checkpoint": _checkpoint(),
        }
    }
    planned = plan_step5_queue(
        arms=ARMS[:1],
        seeds=[7],
        state=_state(tasks),
        max_concurrency=1,
    )
    assert planned["actions"][0]["action"] == "retry_same_instance"
    assert planned["summary"]["planned_active_workers"] == 1


def test_confirmed_host_loss_permits_replacement_and_checkpoint_resume():
    tasks = {
        "tiny13.seed-7": {
            "status": "failed",
            "attempt": 2,
            "instance_id": "lost-instance",
            "instance_status": "lost",
            "failure": {"class": "host_loss", "diagnosis": "confirmed"},
            "checkpoint": _checkpoint(),
        }
    }
    action = plan_step5_queue(
        arms=ARMS[:1],
        seeds=[7],
        state=_state(tasks),
        max_concurrency=3,
    )["actions"][0]
    assert action["action"] == "launch_replacement_instance"
    assert action["replaces_instance_id"] == "lost-instance"
    assert action["attempt"] == 3
    assert action["checkpoint"]["uri"].startswith("s3://")


def test_host_loss_cannot_replace_a_stopped_recoverable_instance():
    tasks = {
        "tiny13.seed-7": {
            "status": "failed",
            "attempt": 2,
            "instance_id": "recoverable-instance",
            "instance_status": "stopped",
            "failure": {"class": "host_loss", "diagnosis": "confirmed"},
        }
    }
    with pytest.raises(ValueError, match="requires an unavailable instance"):
        plan_step5_queue(
            arms=ARMS[:1],
            seeds=[7],
            state=_state(tasks),
            max_concurrency=3,
        )


def test_pending_task_reuses_stopped_instance_before_renting():
    tasks = {
        "tiny13.seed-7": {
            "status": "pending",
            "attempt": 0,
            "instance_id": "instance-a",
            "instance_status": "stopped",
        }
    }
    planned = plan_step5_queue(
        arms=ARMS[:1],
        seeds=[7],
        state=_state(tasks),
        max_concurrency=3,
    )
    assert planned["actions"] == [
        {
            "action": "resume_existing_instance",
            "attempt": 1,
            "instance_id": "instance-a",
            "reason": "existing_worker_must_be_reused_before_rental",
            "task_id": "tiny13.seed-7",
        }
    ]


def test_pending_task_cannot_resume_unavailable_instance():
    tasks = {
        "tiny13.seed-7": {
            "status": "pending",
            "attempt": 0,
            "instance_id": "unavailable-instance",
            "instance_status": "unavailable",
        }
    }
    with pytest.raises(ValueError, match="classify confirmed host loss"):
        plan_step5_queue(
            arms=ARMS[:1],
            seeds=[7],
            state=_state(tasks),
            max_concurrency=3,
        )


def test_software_failure_cannot_retry_unavailable_instance():
    tasks = {
        "tiny13.seed-7": {
            "status": "failed",
            "attempt": 1,
            "instance_id": "unavailable-instance",
            "instance_status": "unavailable",
            "failure": {"class": "software", "diagnosis": "resolved"},
        }
    }
    with pytest.raises(ValueError, match="must retain its original instance"):
        plan_step5_queue(
            arms=ARMS[:1],
            seeds=[7],
            state=_state(tasks),
            max_concurrency=3,
        )


def test_paused_project_instances_count_against_three_worker_cap():
    tasks = {
        f"tiny13.seed-{seed}": {
            "status": "failed",
            "attempt": 1,
            "instance_id": f"instance-{seed}",
            "instance_status": "stopped",
            "failure": {"class": "software", "diagnosis": "pending"},
        }
        for seed in (1, 2, 3)
    }
    planned = plan_step5_queue(
        arms=ARMS,
        seeds=[1, 2, 3, 4],
        state=_state(tasks),
        max_concurrency=3,
    )
    assert all(action["action"] == "diagnose_same_instance" for action in planned["actions"])
    assert not any(action["action"] == "launch_new_instance" for action in planned["actions"])
    assert planned["summary"]["project_worker_inventory"] == 3


def test_cli_is_dry_run_and_writes_deterministic_plan(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "tinylora_step5_plan_v1",
                "arms": ARMS,
                "schedule": {"max_concurrent_single_gpu_workers": 3},
                "large_run_enabled": False,
                "paid_execution_enabled": False,
            }
        )
    )
    state = tmp_path / "state.json"
    state.write_text(json.dumps(_state()))
    output = tmp_path / "queue.json"
    script = Path(__file__).parents[1] / "scripts" / "plan_tinylora_step5_fleet.py"
    command = [
        sys.executable,
        str(script),
        "--step5-plan",
        str(manifest),
        "--state",
        str(state),
        "--output",
        str(output),
        "--seeds",
        "7,8",
        "--max-concurrency",
        "2",
    ]
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    first = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    first_bytes = output.read_bytes()
    second = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert output.read_bytes() == first_bytes
    assert json.loads(first.stdout) == json.loads(second.stdout)
    assert json.loads(first.stdout)["dry_run"] is True
