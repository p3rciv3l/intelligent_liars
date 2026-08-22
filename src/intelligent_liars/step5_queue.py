"""Deterministic, dry-run scheduling for the bounded Step 5 GPU screen.

This module plans work.  It deliberately has no cloud-provider client and cannot
rent, start, stop, or destroy a machine.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


QUEUE_PLAN_FORMAT = "tinylora_step5_queue_plan_v1"
QUEUE_STATE_FORMAT = "tinylora_step5_queue_state_v1"
HARD_PROJECT_WORKER_CAP = 3

_INSTANCE_INVENTORY_STATUSES = {"created", "loading", "running", "stopped", "paused"}
_ACTIVE_INSTANCE_STATUSES = {"created", "loading", "running"}
_TASK_STATUSES = {"pending", "running", "succeeded", "failed"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _plan_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _task_id(arm: str, seed: int) -> str:
    return f"{arm}.seed-{seed}"


def _normalize_arms(arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(arm) for arm in arms]
    names = [str(arm.get("name", "")) for arm in normalized]
    if not normalized or any(not name for name in names):
        raise ValueError("Every Step 5 arm requires a nonempty name")
    if len(names) != len(set(names)):
        raise ValueError("Step 5 arm names must be unique")
    return normalized


def _normalize_seeds(seeds: Iterable[int]) -> list[int]:
    normalized = [int(seed) for seed in seeds]
    if not normalized:
        raise ValueError("At least one Step 5 seed is required")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Step 5 seeds must be unique")
    if any(seed < 0 for seed in normalized):
        raise ValueError("Step 5 seeds must be nonnegative")
    return normalized


def _verified_checkpoint(task_state: Mapping[str, Any]) -> dict[str, Any] | None:
    checkpoint = task_state.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("verified"):
        return None
    uri = checkpoint.get("uri")
    digest = checkpoint.get("sha256")
    if not isinstance(uri, str) or not uri:
        raise ValueError("A verified checkpoint requires a durable URI")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("A verified checkpoint requires a lowercase SHA-256")
    return dict(checkpoint)


def _instance_inventory(state: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    inventory: set[str] = set()
    active: set[str] = set()
    records: list[Mapping[str, Any]] = []
    tasks = state.get("tasks", {})
    if isinstance(tasks, Mapping):
        records.extend(record for record in tasks.values() if isinstance(record, Mapping))
    project_instances = state.get("project_instances", [])
    if isinstance(project_instances, list):
        records.extend(
            record for record in project_instances if isinstance(record, Mapping)
        )
    for record in records:
        instance_id = record.get("instance_id")
        instance_status = record.get("instance_status")
        if not isinstance(instance_id, str) or not instance_id:
            continue
        if instance_status in _INSTANCE_INVENTORY_STATUSES:
            inventory.add(instance_id)
        if instance_status in _ACTIVE_INSTANCE_STATUSES:
            active.add(instance_id)
    return inventory, active


def _validate_state(
    state: Mapping[str, Any],
    *,
    known_task_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if state.get("format") != QUEUE_STATE_FORMAT:
        raise ValueError(f"Queue state must use format {QUEUE_STATE_FORMAT}")
    tasks = state.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("Queue state tasks must be an object")
    unknown = sorted(set(map(str, tasks)) - known_task_ids)
    if unknown:
        raise ValueError(f"Queue state contains unknown tasks: {unknown}")
    normalized: dict[str, dict[str, Any]] = {}
    instance_owners: dict[str, str] = {}
    for task_id, raw in tasks.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Queue task state must be an object: {task_id}")
        record = dict(raw)
        status = record.get("status")
        if status not in _TASK_STATUSES:
            raise ValueError(f"Unsupported queue task status for {task_id}: {status!r}")
        attempt = record.get("attempt", 0 if status == "pending" else 1)
        if not isinstance(attempt, int) or attempt < 0:
            raise ValueError(f"Queue attempt must be a nonnegative integer: {task_id}")
        record["attempt"] = attempt
        instance_id = record.get("instance_id")
        instance_status = record.get("instance_status")
        if instance_id is not None:
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(f"Invalid instance id for {task_id}")
            if instance_status is None:
                raise ValueError(f"Instance status is required for {task_id}")
            owner = instance_owners.setdefault(instance_id, str(task_id))
            if owner != task_id:
                raise ValueError(f"Instance {instance_id} is assigned to multiple tasks")
        if status == "running" and instance_status not in _ACTIVE_INSTANCE_STATUSES:
            raise ValueError(f"Running task {task_id} requires an active instance")
        if status == "failed":
            failure = record.get("failure")
            if not isinstance(failure, Mapping):
                raise ValueError(f"Failed task {task_id} requires failure classification")
            failure_class = failure.get("class")
            if failure_class not in {"software", "host_loss"}:
                raise ValueError(f"Failed task {task_id} has unsupported failure class")
            if failure_class == "software":
                if not instance_id or instance_status in {"lost", "destroyed", "terminated"}:
                    raise ValueError(
                        f"Software failure {task_id} must retain its original instance"
                    )
                if failure.get("diagnosis") not in {"pending", "resolved"}:
                    raise ValueError(
                        f"Software failure {task_id} requires pending/resolved diagnosis"
                    )
            elif failure.get("diagnosis") not in {"pending", "confirmed"}:
                raise ValueError(
                    f"Host loss {task_id} requires pending/confirmed diagnosis"
                )
        _verified_checkpoint(record)
        normalized[str(task_id)] = record
    return normalized


def plan_step5_queue(
    *,
    arms: Sequence[Mapping[str, Any]],
    seeds: Iterable[int],
    state: Mapping[str, Any],
    max_concurrency: int,
) -> dict[str, Any]:
    """Plan the next bounded queue transition without changing external state."""
    if not 1 <= max_concurrency <= HARD_PROJECT_WORKER_CAP:
        raise ValueError(
            f"Step 5 allows at most {HARD_PROJECT_WORKER_CAP} concurrent workers"
        )
    normalized_arms = _normalize_arms(arms)
    normalized_seeds = _normalize_seeds(seeds)
    jobs: list[dict[str, Any]] = [
        {
            "task_id": _task_id(str(arm["name"]), seed),
            "arm": str(arm["name"]),
            "seed": seed,
        }
        for seed in normalized_seeds
        for arm in normalized_arms
    ]
    task_states = _validate_state(
        state,
        known_task_ids={str(job["task_id"]) for job in jobs},
    )
    inventory, active_instances = _instance_inventory(state)
    if len(inventory) > HARD_PROJECT_WORKER_CAP:
        raise ValueError(
            f"Project already has {len(inventory)} workers; refusing a fourth worker"
        )
    if len(active_instances) > max_concurrency:
        raise ValueError(
            "Existing active Step 5 workers exceed configured max concurrency"
        )

    actions: list[dict[str, Any]] = []
    capacity_candidates: list[
        tuple[dict[str, Any], dict[str, Any], bool, bool]
    ] = []
    pending_candidates: list[dict[str, Any]] = []
    for job in jobs:
        task_id = str(job["task_id"])
        task = task_states.get(task_id, {"status": "pending", "attempt": 0})
        status = task["status"]
        if status in {"running", "succeeded"}:
            continue
        if status == "pending":
            pending_candidates.append(job)
            continue
        failure = task["failure"]
        failure_class = failure["class"]
        if failure_class == "software" and failure["diagnosis"] == "pending":
            actions.append(
                {
                    "action": "diagnose_same_instance",
                    "attempt": task["attempt"],
                    "instance_id": task["instance_id"],
                    "reason": "software_failure_requires_same_host_diagnosis",
                    "task_id": task_id,
                }
            )
            continue
        if failure_class == "host_loss" and failure["diagnosis"] == "pending":
            actions.append(
                {
                    "action": "confirm_host_loss",
                    "attempt": task["attempt"],
                    "instance_id": task.get("instance_id"),
                    "reason": "replacement_forbidden_until_host_loss_is_confirmed",
                    "task_id": task_id,
                }
            )
            continue
        checkpoint = _verified_checkpoint(task)
        if failure_class == "software":
            instance_is_active = task["instance_status"] in _ACTIVE_INSTANCE_STATUSES
            action = {
                "action": (
                    "retry_same_instance"
                    if instance_is_active
                    else (
                        "resume_same_instance"
                        if checkpoint is not None
                        else "retry_same_instance_from_start"
                    )
                ),
                "attempt": task["attempt"] + 1,
                "instance_id": task["instance_id"],
                "reason": "software_failure_reuses_diagnosed_instance",
                "task_id": task_id,
            }
            if checkpoint is not None:
                action["checkpoint"] = checkpoint
            capacity_candidates.append((job, action, False, not instance_is_active))
        else:
            action = {
                "action": "launch_replacement_instance",
                "attempt": task["attempt"] + 1,
                "reason": "confirmed_host_loss_permits_replacement",
                "replaces_instance_id": task.get("instance_id"),
                "task_id": task_id,
            }
            if checkpoint is not None:
                action["checkpoint"] = checkpoint
            capacity_candidates.append((job, action, True, True))

    for job in pending_candidates:
        capacity_candidates.append(
            (
                job,
                {
                    "action": "launch_new_instance",
                    "attempt": 1,
                    "reason": "independent_single_gpu_task_is_pending",
                    "task_id": job["task_id"],
                },
                True,
                True,
            )
        )

    planned_active = len(active_instances)
    planned_inventory = len(inventory)
    for _job, action, adds_instance, adds_active in capacity_candidates:
        if adds_active and planned_active >= max_concurrency:
            continue
        if adds_instance and planned_inventory >= HARD_PROJECT_WORKER_CAP:
            continue
        actions.append(action)
        if adds_active:
            planned_active += 1
        if adds_instance:
            planned_inventory += 1

    basis = {
        "format": QUEUE_PLAN_FORMAT,
        "dry_run": True,
        "hard_project_worker_cap": HARD_PROJECT_WORKER_CAP,
        "max_concurrency": max_concurrency,
        "arms": normalized_arms,
        "seeds": normalized_seeds,
        "jobs": jobs,
        "observed_state": dict(state),
        "actions": actions,
        "summary": {
            "jobs_total": len(jobs),
            "jobs_succeeded": sum(
                task.get("status") == "succeeded" for task in task_states.values()
            ),
            "jobs_running": sum(
                task.get("status") == "running" for task in task_states.values()
            ),
            "project_worker_inventory": len(inventory),
            "active_workers": len(active_instances),
            "planned_active_workers": planned_active,
            "planned_project_worker_inventory": planned_inventory,
        },
    }
    return {**basis, "plan_id": _plan_id(basis)}
