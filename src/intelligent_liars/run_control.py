from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import socket
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4


class LockHeldError(RuntimeError):
    """Raised when a live or unsafe lock prevents a run-control operation."""


class ManifestValidationError(ValueError):
    """Raised when an OSWorld run manifest is not frozen and reproducible."""


class LedgerValidationError(ValueError):
    """Raised when an append-only ledger event is invalid."""


class TerminalState(StrEnum):
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    REFUSAL = "refusal"
    INVALID_ACTION = "invalid_action"
    MODEL_ERROR = "model_error"
    EVALUATOR_ERROR = "evaluator_error"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class BudgetDecision(StrEnum):
    CONTINUE = "continue"
    STOP_NEW_LEASES = "stop-new-leases"
    HARD_STOP = "hard-stop"


MODEL_RUN_HARD_STOP_USD = Decimal("70")
PROMOTION_PROJECTED_TOTAL_LIMIT_USD = Decimal("60")
COMBINED_RUN_ENVELOPE_USD = Decimal("140")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id() -> str:
    return str(uuid4())


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def task_grid_sha256(task_ids: Sequence[str]) -> str:
    """Hash the ordered task ID list as canonical JSON."""
    return stable_sha256(list(task_ids))


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{key} must be an object")
    return value


def _require_string(payload: Mapping[str, Any], key: str, path: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{path}.{key} must be a non-empty string")
    return value


def _require_hash(value: str, path: str, lengths: set[int]) -> None:
    if len(value) not in lengths or any(character not in "0123456789abcdefABCDEF" for character in value):
        expected = " or ".join(str(length) for length in sorted(lengths))
        raise ManifestValidationError(f"{path} must be a {expected}-character hexadecimal hash")


@dataclass(frozen=True)
class FrozenRunManifest:
    payload: Mapping[str, Any]
    manifest_hash: str
    run_id: str

    @property
    def task_ids(self) -> tuple[str, ...]:
        task_grid = _require_mapping(self.payload, "task_grid")
        return tuple(task_grid["task_ids"])

    @property
    def step_cap(self) -> int:
        run = _require_mapping(self.payload, "run")
        return int(run["step_cap"])

    @property
    def results_key(self) -> str:
        return f"osworld/{self.run_id}"


def validate_run_manifest(payload: Mapping[str, Any]) -> FrozenRunManifest:
    if payload.get("schema_version") != 1:
        raise ManifestValidationError("schema_version must be 1")

    repository = _require_mapping(payload, "repository")
    repository_commit = _require_string(repository, "commit", "repository")
    _require_hash(repository_commit, "repository.commit", {40})
    if repository.get("dirty") is not False:
        raise ManifestValidationError("repository.dirty must be false")

    osworld = _require_mapping(payload, "osworld")
    osworld_commit = _require_string(osworld, "commit", "osworld")
    _require_hash(osworld_commit, "osworld.commit", {40})

    task_grid = _require_mapping(payload, "task_grid")
    task_grid_hash = _require_string(task_grid, "sha256", "task_grid")
    _require_hash(task_grid_hash, "task_grid.sha256", {64})
    task_ids = task_grid.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
    ):
        raise ManifestValidationError("task_grid.task_ids must be a non-empty string list")
    if len(task_ids) != len(set(task_ids)):
        raise ManifestValidationError("task_grid.task_ids must not contain duplicates")
    if task_grid_hash.lower() != task_grid_sha256(task_ids):
        raise ManifestValidationError(
            "task_grid.sha256 must equal SHA-256(canonical_json(task_grid.task_ids))"
        )

    model = _require_mapping(payload, "model")
    _require_string(model, "id", "model")
    model_revision = _require_string(model, "revision", "model")
    _require_hash(model_revision, "model.revision", {40, 64})

    run = _require_mapping(payload, "run")
    step_cap = run.get("step_cap")
    if not isinstance(step_cap, int) or isinstance(step_cap, bool) or step_cap <= 0:
        raise ManifestValidationError("run.step_cap must be a positive integer")

    normalized = json.loads(canonical_json(payload))
    manifest_hash = stable_sha256(normalized)
    return FrozenRunManifest(
        payload=normalized,
        manifest_hash=manifest_hash,
        run_id=f"osworld-{manifest_hash[:20]}",
    )


def load_run_manifest(path: Path) -> FrozenRunManifest:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"Cannot load manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestValidationError("manifest root must be an object")
    return validate_run_manifest(payload)


def initialize_run_directory(results_root: Path, manifest: FrozenRunManifest) -> Path:
    run_directory = results_root / manifest.run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest.payload, indent=2, sort_keys=True) + "\n"
    )
    return run_directory


def validate_higher_step_run_policy(
    fifty_step: FrozenRunManifest,
    hundred_step: FrozenRunManifest,
) -> None:
    if fifty_step.step_cap != 50 or hundred_step.step_cap != 100:
        raise ManifestValidationError("higher-step policy requires separate 50-step and 100-step manifests")
    fifty_comparable = deepcopy(fifty_step.payload)
    hundred_comparable = deepcopy(hundred_step.payload)
    del fifty_comparable["run"]["step_cap"]
    del hundred_comparable["run"]["step_cap"]
    if fifty_comparable != hundred_comparable:
        raise ManifestValidationError(
            "50-step and 100-step manifests may differ only in run.step_cap"
        )
    if fifty_step.run_id == hundred_step.run_id or fifty_step.results_key == hundred_step.results_key:
        raise ManifestValidationError("50-step and 100-step runs must have separate result keys")


def _append_json_line(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short append to {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class TaskAttempt:
    run_id: str
    task_id: str
    attempt: int
    terminal_state: TerminalState
    started_at: str
    finished_at: str
    artifact_checksums: Mapping[str, str]
    detail: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "terminal_state", TerminalState(self.terminal_state))
        except ValueError as exc:
            raise LedgerValidationError("unknown terminal state") from exc

    def as_dict(self) -> dict[str, Any]:
        event = {
            "event": "task_attempt",
            "run_id": self.run_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "terminal_state": self.terminal_state.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifact_checksums": dict(self.artifact_checksums),
        }
        if self.detail is not None:
            event["detail"] = self.detail
        return event


class AttemptLedger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id

    def append(self, attempt: TaskAttempt) -> None:
        if attempt.run_id != self.run_id:
            raise LedgerValidationError("attempt run_id does not match ledger run_id")
        if not attempt.task_id or attempt.attempt < 1:
            raise LedgerValidationError("task_id must be set and attempt must be positive")
        if not attempt.started_at or not attempt.finished_at:
            raise LedgerValidationError("attempt timestamps must be set")
        for name, checksum in attempt.artifact_checksums.items():
            if not name or len(checksum) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in checksum
            ):
                raise LedgerValidationError("artifact checksums must be named SHA-256 hashes")
        if any(
            event.get("task_id") == attempt.task_id
            and event.get("attempt") == attempt.attempt
            for event in self.events()
        ):
            raise LedgerValidationError("task attempt already exists")
        _append_json_line(self.path, attempt.as_dict())

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            try:
                event = json.loads(line)
                TerminalState(event["terminal_state"])
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise LedgerValidationError(
                    f"invalid attempt ledger event at line {line_number}"
                ) from exc
            events.append(event)
        return events


def _decimal(value: Decimal | str | int, field: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"{field} must be a decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


@dataclass(frozen=True)
class CostSample:
    provider: str
    resource: str
    accrued_cost_usd: Decimal
    sampled_at: str

    def __post_init__(self) -> None:
        if not self.provider or not self.resource or not self.sampled_at:
            raise ValueError("cost sample provider, resource, and sampled_at are required")
        object.__setattr__(
            self,
            "accrued_cost_usd",
            _decimal(self.accrued_cost_usd, "accrued_cost_usd"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "event": "cost_sample",
            "provider": self.provider,
            "resource": self.resource,
            "accrued_cost_usd": str(self.accrued_cost_usd),
            "sampled_at": self.sampled_at,
        }


class CostLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, sample: CostSample) -> None:
        _append_json_line(self.path, sample.as_dict())

    def samples(self) -> list[CostSample]:
        if not self.path.exists():
            return []
        samples = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            try:
                event = json.loads(line)
                if event.get("event") != "cost_sample":
                    raise ValueError
                samples.append(
                    CostSample(
                        provider=event["provider"],
                        resource=event["resource"],
                        accrued_cost_usd=Decimal(event["accrued_cost_usd"]),
                        sampled_at=event["sampled_at"],
                    )
                )
            except (KeyError, ValueError, json.JSONDecodeError, InvalidOperation) as exc:
                raise LedgerValidationError(
                    f"invalid cost ledger event at line {line_number}"
                ) from exc
        return samples

    def total(self) -> Decimal:
        return sum(self.totals_by_resource().values(), Decimal("0"))

    def totals_by_resource(self) -> dict[tuple[str, str], Decimal]:
        totals: dict[tuple[str, str], Decimal] = {}
        for sample in self.samples():
            key = (sample.provider, sample.resource)
            totals[key] = sample.accrued_cost_usd
        return totals


@dataclass(frozen=True)
class BudgetPolicy:
    stop_new_leases_usd: Decimal
    hard_stop_usd: Decimal

    def __post_init__(self) -> None:
        stop_new = _decimal(self.stop_new_leases_usd, "stop_new_leases_usd")
        hard_stop = _decimal(self.hard_stop_usd, "hard_stop_usd")
        if stop_new > hard_stop:
            raise ValueError("stop_new_leases_usd must not exceed hard_stop_usd")
        object.__setattr__(self, "stop_new_leases_usd", stop_new)
        object.__setattr__(self, "hard_stop_usd", hard_stop)

    def decide(
        self,
        accrued_cost_usd: Decimal,
        authorized_active_cost_usd: Decimal = Decimal("0"),
    ) -> BudgetDecision:
        accrued = _decimal(accrued_cost_usd, "accrued_cost_usd")
        active = _decimal(authorized_active_cost_usd, "authorized_active_cost_usd")
        authorized_total = accrued + active
        if authorized_total >= self.hard_stop_usd:
            return BudgetDecision.HARD_STOP
        if authorized_total >= self.stop_new_leases_usd:
            return BudgetDecision.STOP_NEW_LEASES
        return BudgetDecision.CONTINUE


@dataclass(frozen=True)
class EvaluationBudgetPolicy:
    model_run_hard_stop_usd: Decimal = MODEL_RUN_HARD_STOP_USD
    promotion_projected_total_limit_usd: Decimal = PROMOTION_PROJECTED_TOTAL_LIMIT_USD
    combined_run_envelope_usd: Decimal = COMBINED_RUN_ENVELOPE_USD

    def __post_init__(self) -> None:
        model_limit = _decimal(self.model_run_hard_stop_usd, "model_run_hard_stop_usd")
        promotion_limit = _decimal(
            self.promotion_projected_total_limit_usd,
            "promotion_projected_total_limit_usd",
        )
        combined_limit = _decimal(
            self.combined_run_envelope_usd,
            "combined_run_envelope_usd",
        )
        if model_limit > MODEL_RUN_HARD_STOP_USD:
            raise ValueError("model run hard stop cannot exceed $70")
        if promotion_limit > PROMOTION_PROJECTED_TOTAL_LIMIT_USD:
            raise ValueError("100-step promotion limit cannot exceed $60")
        if combined_limit > COMBINED_RUN_ENVELOPE_USD:
            raise ValueError("combined run envelope cannot exceed $140")
        object.__setattr__(self, "model_run_hard_stop_usd", model_limit)
        object.__setattr__(self, "promotion_projected_total_limit_usd", promotion_limit)
        object.__setattr__(self, "combined_run_envelope_usd", combined_limit)

    def may_promote_to_100_steps(self, projected_total_usd: Decimal) -> bool:
        projected = _decimal(projected_total_usd, "projected_total_usd")
        return projected < self.promotion_projected_total_limit_usd

    def combined_decision(
        self,
        baseline_authorized_usd: Decimal,
        intervention_authorized_usd: Decimal,
    ) -> BudgetDecision:
        combined = _decimal(
            baseline_authorized_usd,
            "baseline_authorized_usd",
        ) + _decimal(intervention_authorized_usd, "intervention_authorized_usd")
        if combined >= self.combined_run_envelope_usd:
            return BudgetDecision.HARD_STOP
        return BudgetDecision.CONTINUE


def desired_worker_count(
    remaining_tasks: int,
    maximum_workers: int,
    tasks_per_worker: int = 1,
) -> int:
    if remaining_tasks < 0 or maximum_workers < 0 or tasks_per_worker <= 0:
        raise ValueError("worker counts must be non-negative and tasks_per_worker must be positive")
    needed = (remaining_tasks + tasks_per_worker - 1) // tasks_per_worker
    return min(maximum_workers, needed)


@dataclass(frozen=True)
class LaunchResource:
    provider: str
    resource: str
    instance_count: int
    ttl_minutes: int
    estimated_hourly_rate_usd: Decimal

    def __post_init__(self) -> None:
        if not self.provider or not self.resource:
            raise ValueError("launch resource provider and resource are required")
        if self.instance_count < 0 or self.ttl_minutes <= 0:
            raise ValueError("instance_count must be non-negative and ttl_minutes must be positive")
        object.__setattr__(
            self,
            "estimated_hourly_rate_usd",
            _decimal(self.estimated_hourly_rate_usd, "estimated_hourly_rate_usd"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "resource": self.resource,
            "instance_count": self.instance_count,
            "ttl_minutes": self.ttl_minutes,
            "estimated_hourly_rate_usd": str(self.estimated_hourly_rate_usd),
        }


@dataclass(frozen=True)
class LaunchProposal:
    run_id: str
    dry_run: bool
    resources: tuple[LaunchResource, ...]
    projected_total_usd: Decimal
    maximum_authorized_spend_usd: Decimal
    teardown_checklist: tuple[str, ...]

    @property
    def estimated_hourly_rate_usd(self) -> Decimal:
        return sum(
            (
                resource.estimated_hourly_rate_usd * resource.instance_count
                for resource in self.resources
            ),
            Decimal("0"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dry_run": self.dry_run,
            "resources": [resource.as_dict() for resource in self.resources],
            "estimated_hourly_rate_usd": str(self.estimated_hourly_rate_usd),
            "projected_total_usd": str(self.projected_total_usd),
            "maximum_authorized_spend_usd": str(self.maximum_authorized_spend_usd),
            "teardown_checklist": list(self.teardown_checklist),
        }


def create_launch_proposal(
    manifest: FrozenRunManifest,
    resources: Sequence[LaunchResource],
    maximum_authorized_spend_usd: Decimal,
    teardown_checklist: Sequence[str],
    projected_total_usd: Decimal | None = None,
) -> LaunchProposal:
    maximum_spend = _decimal(maximum_authorized_spend_usd, "maximum_authorized_spend_usd")
    projected_total = _decimal(
        projected_total_usd if projected_total_usd is not None else maximum_spend,
        "projected_total_usd",
    )
    if maximum_spend > MODEL_RUN_HARD_STOP_USD:
        raise ValueError("maximum_authorized_spend_usd cannot exceed the $70 model-run hard stop")
    if (
        manifest.step_cap == 100
        and projected_total >= PROMOTION_PROJECTED_TOTAL_LIMIT_USD
    ):
        raise ValueError("100-step projected_total_usd must be strictly below $60")
    if not resources:
        raise ValueError("at least one launch resource is required")
    if not teardown_checklist or any(not item for item in teardown_checklist):
        raise ValueError("teardown_checklist must contain explicit actions")
    return LaunchProposal(
        run_id=manifest.run_id,
        dry_run=True,
        resources=tuple(resources),
        projected_total_usd=projected_total,
        maximum_authorized_spend_usd=maximum_spend,
        teardown_checklist=tuple(teardown_checklist),
    )


class CloudActions(Protocol):
    def launch(self, proposal: LaunchProposal) -> None: ...


def dispatch_launch(
    proposal: LaunchProposal,
    cloud_actions: CloudActions,
    *,
    dry_run: bool = True,
) -> None:
    if dry_run:
        return
    cloud_actions.launch(proposal)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_line(argv: Sequence[str] | None = None) -> str:
    return " ".join(shlex.quote(part) for part in (argv or sys.argv))


def current_git_commit(project_root: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def current_host() -> str:
    return socket.gethostname()


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class FileLock:
    path: Path
    payload: Mapping[str, Any]
    acquired: bool = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = json.loads(self.path.read_text())
        except Exception:
            self.acquired = False
            return
        if (
            existing.get("owner_pid") == self.payload.get("owner_pid")
            and existing.get("host") == self.payload.get("host")
            and existing.get("started_at") == self.payload.get("started_at")
            and existing.get("run_id") == self.payload.get("run_id")
        ):
            self.path.unlink(missing_ok=True)
        self.acquired = False


@dataclass
class SignalCleanup:
    previous_handlers: Mapping[int, Any]

    def restore(self) -> None:
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)


def lock_payload(
    *,
    run_id: str,
    queue_plan_id: str | None,
    command: str | None = None,
    kind: str = "run",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "owner_pid": os.getpid(),
        "host": current_host(),
        "started_at": now_iso(),
        "command": command or command_line(),
        "run_id": run_id,
    }
    if queue_plan_id is not None:
        payload["queue_plan_id"] = queue_plan_id
    if extra:
        payload.update(dict(extra))
    return payload


def acquire_lock(
    path: Path,
    payload: Mapping[str, Any],
    *,
    force_stale_lock: bool = False,
) -> FileLock:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _create_lock(path, payload)
    except FileExistsError:
        if not force_stale_lock:
            raise LockHeldError(_lock_error_message(path)) from None
        _break_same_host_stale_lock(path)
        return _create_lock(path, payload)


def _create_lock(path: Path, payload: Mapping[str, Any]) -> FileLock:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    return FileLock(path=path, payload=dict(payload))


def _lock_error_message(path: Path) -> str:
    try:
        existing = json.loads(path.read_text())
    except Exception:
        return f"Lock exists and is not valid JSON: {path}"
    return (
        f"Lock exists: {path} "
        f"host={existing.get('host')} pid={existing.get('owner_pid')} "
        f"run_id={existing.get('run_id')} started_at={existing.get('started_at')}"
    )


def _break_same_host_stale_lock(path: Path) -> None:
    try:
        existing = json.loads(path.read_text())
    except Exception as exc:
        raise LockHeldError(f"Cannot prove malformed lock is stale: {path}") from exc

    lock_host = existing.get("host")
    owner_pid = existing.get("owner_pid")
    if lock_host != current_host():
        raise LockHeldError(
            f"Refusing to break cross-host lock {path}: host={lock_host!r} current_host={current_host()!r}"
        )
    if not isinstance(owner_pid, int):
        raise LockHeldError(f"Cannot prove lock owner pid is stale: {path}")
    if pid_is_alive(owner_pid):
        raise LockHeldError(f"Refusing to break live lock {path}: pid={owner_pid}")
    path.unlink()


def install_signal_cleanup(lock: FileLock) -> SignalCleanup:
    previous_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }

    def _handler(signum: int, _frame: FrameType | None) -> None:
        lock.release()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    return SignalCleanup(previous_handlers=previous_handlers)
