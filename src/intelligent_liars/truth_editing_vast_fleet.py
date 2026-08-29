"""Synchronous multi-instance execution behind the truth-editing batch seam.

Optuna remains exclusively in the controller process.  This module receives an
already-suggested barrier, evaluates its trials on independent bounded workers,
durably receipts every outcome, and returns only after the entire barrier is
complete.  Workers never suggest or observe trials.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import base64
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .truth_editing_batch_execution import BatchEvaluationRequest
from .truth_editing_failure_policy import PaidJudgeCircuitOpen
from .truth_editing_live_judge import FROZEN_JUDGE_CONFIG_SHA256
from .truth_editing_gpu_telemetry import GpuTelemetryCollector
from .truth_editing_study import EvaluationResult
from .truth_editing_vast_prerequisites import Offer
from .truth_editing_vast_prerequisites import EphemeralWorkloadSecret
from .truth_editing_vast_production import (
    ProductionVastConfig,
    execute_production_lifecycle,
    production_lifecycle_plan,
)


FLEET_FORMAT = "truth_editing_vast_fleet_v2"
ADAPTIVE_FLEET_FORMAT = "truth_editing_vast_fleet_v3"
FLEET_BUDGET_FORMAT = "truth_editing_vast_fleet_budget_v1"
LEGACY_TRIAL_RECEIPT_FORMAT = "truth_editing_vast_fleet_trial_receipt_v1"
TRIAL_RECEIPT_FORMAT = "truth_editing_vast_fleet_trial_receipt_v2"
STOP_RECEIPT_FORMAT = "truth_editing_vast_fleet_stop_receipt_v1"
RECEIPT_DURABLE_EVENT_FORMAT = "truth_editing_vast_fleet_receipt_durable_event_v1"
PHASE_BOUNDARIES = {"discovery": 80, "expanded": 160, "finalist": 200}
_HEX = frozenset("0123456789abcdef")
_EXECUTION_MODE = "persistent_single_host_eight_gpu"
_INFRASTRUCTURE_COSTS = (
    "gpu_compute",
    "storage",
    "network_download",
    "network_upload",
)
_EXPECTED_JUDGE_BUDGET_MAPPING = {
    "format": "truth_editing_production_judge_budget_config_v1",
    "all_in_maximum_spend_usd": "50",
    "non_judge_reserved_spend_usd": "45",
    "maximum_judge_spend_usd": "5",
    "per_call_reservation_usd": "0.025",
    "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
}
_SAFE_TRIAL_TELEMETRY_FIELDS = frozenset(
    {
        "evaluation_seconds",
        "generated_tokens",
        "generated_tokens_per_second",
        "cuda_peak_allocated_bytes",
        "judge_calls",
        "judge_failures",
        "judge_latency_seconds",
        "judge_cost_usd",
    }
)


class FleetError(RuntimeError):
    """A fleet action would violate frozen identity or lifecycle semantics."""


class FleetCircuitOpen(FleetError, PaidJudgeCircuitOpen):
    """A paid judge/circuit failure that must abort the whole study immediately."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FleetError("fleet value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


EXPECTED_PRODUCTION_JUDGE_BUDGET_SHA256 = _sha(_EXPECTED_JUDGE_BUDGET_MAPPING)


def _exact(raw: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(raw) != fields:
        raise FleetError(f"{label} fields changed")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FleetError(f"{label} must be an object")
    result = dict(value)
    _canonical(result)
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise FleetError(f"{label} must be a nonempty trimmed string")
    return value


def _integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FleetError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FleetError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or (
        maximum is not None and result > maximum
    ):
        raise FleetError(f"{label} is outside its allowed bound")
    return result


def _money(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise FleetError(f"{label} must be finite nonnegative money text")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise FleetError(f"{label} must be finite nonnegative money text") from error
    if not isinstance(value, str) or not result.is_finite() or result < 0:
        raise FleetError(f"{label} must be finite nonnegative money text")
    canonical = format(result, "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    if value != canonical:
        raise FleetError(f"{label} must use canonical money text")
    return result


def _digest(value: Any, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in _HEX for char in result):
        raise FleetError(f"{label} must be a lowercase SHA-256")
    return result


def _relative_json_path(value: Any, label: str) -> str:
    result = _text(value, label)
    relative = PurePosixPath(result)
    if (
        relative.is_absolute()
        or relative.suffix != ".json"
        or relative.name in {"", ".", ".."}
        or ".." in relative.parts
    ):
        raise FleetError(f"{label} must be a safe relative JSON path")
    return result


@dataclass(frozen=True)
class FleetConfig:
    format: str
    fleet_id: str
    worker_count: int
    all_in_maximum_spend_usd: Decimal
    maximum_infrastructure_spend_usd: Decimal
    maximum_judge_spend_usd: Decimal
    production_judge_budget_config_sha256: str
    budget_identity_sha256: str
    maximum_host_lease_seconds: int
    maximum_fetch_gib: float
    production_config_path: str
    production_config_sha256: str
    bundle_sha256: str | None
    receipt_directory: Path
    batch_size: int
    adaptive_capacity_policy_path: str | None
    adaptive_capacity_policy_sha256: str | None
    study_config_path: str | None
    study_config_sha256: str | None
    study_identity_sha256: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FleetConfig":
        raw = _mapping(value, "fleet config")
        format_name = raw.get("format")
        legacy = format_name == FLEET_FORMAT
        if legacy:
            expected_fields = {
                "format",
                "fleet_id",
                "phase_boundaries",
                "execution_mode",
                "worker_count",
                "budget",
                "production_config",
                "bundle_sha256",
                "receipt_directory",
                "capability_test_access",
            }
        elif format_name == ADAPTIVE_FLEET_FORMAT:
            expected_fields = {
                "format",
                "fleet_id",
                "adaptive_capacity_policy",
                "study",
                "execution_topology",
                "budget",
                "production_config",
                "receipt_directory",
                "capability_test_access",
            }
        else:
            raise FleetError("fleet config format changed")
        _exact(raw, expected_fields, "fleet config")

        if legacy:
            if raw["phase_boundaries"] != PHASE_BOUNDARIES:
                raise FleetError("phase boundaries must remain exactly 80/160/200")
            execution_mode = raw["execution_mode"]
            worker_count = _integer(raw["worker_count"], "worker_count")
            batch_size = worker_count
            capacity_policy_path = None
            capacity_policy_sha256 = None
            study_config_path = None
            study_config_sha256 = None
            study_identity_sha256 = None
            bundle_sha256 = _digest(raw["bundle_sha256"], "bundle_sha256")
        else:
            topology = _mapping(raw["execution_topology"], "execution_topology")
            _exact(topology, {"mode", "worker_count", "batch_size"}, "execution_topology")
            execution_mode = topology["mode"]
            worker_count = _integer(
                topology["worker_count"], "execution_topology.worker_count"
            )
            batch_size = _integer(
                topology["batch_size"], "execution_topology.batch_size"
            )
            capacity_policy = _mapping(
                raw["adaptive_capacity_policy"], "adaptive_capacity_policy"
            )
            _exact(capacity_policy, {"path", "sha256"}, "adaptive_capacity_policy")
            capacity_policy_path = _relative_json_path(
                capacity_policy["path"], "adaptive_capacity_policy.path"
            )
            capacity_policy_sha256 = _digest(
                capacity_policy["sha256"], "adaptive_capacity_policy.sha256"
            )
            study = _mapping(raw["study"], "study")
            _exact(
                study,
                {"path", "config_sha256", "identity_sha256"},
                "study",
            )
            study_config_path = _relative_json_path(study["path"], "study.path")
            study_config_sha256 = _digest(
                study["config_sha256"], "study.config_sha256"
            )
            study_identity_sha256 = _digest(
                study["identity_sha256"], "study.identity_sha256"
            )
            # The final archive contains this fleet config, so binding its byte
            # hash here would be an impossible self-reference. Transport
            # archive integrity belongs to the outer Vast lifecycle receipt.
            bundle_sha256 = None
        if execution_mode != _EXECUTION_MODE:
            raise FleetError("production fleet must use one persistent eight-GPU host")
        if raw["capability_test_access"] is not False:
            raise FleetError("production fleet must have no capability-test access")
        budget = _mapping(raw["budget"], "budget")
        _exact(
            budget,
            {
                "format",
                "all_in_maximum_spend_usd",
                "maximum_infrastructure_spend_usd",
                "maximum_judge_spend_usd",
                "production_judge_budget_config_sha256",
                "included_infrastructure_costs",
                "maximum_host_lease_seconds",
                "maximum_fetch_gib",
                "identity_sha256",
            },
            "budget",
        )
        if budget["format"] != FLEET_BUDGET_FORMAT:
            raise FleetError("fleet budget format changed")
        production = _mapping(raw["production_config"], "production_config")
        _exact(production, {"path", "sha256"}, "production_config")
        production_path = _relative_json_path(
            production["path"], "production_config.path"
        )
        receipt = PurePosixPath(_text(raw["receipt_directory"], "receipt_directory"))
        if ".." in receipt.parts:
            raise FleetError("receipt_directory must not traverse parents")
        if worker_count != 8 or batch_size != 8:
            raise FleetError(
                "persistent production fleet requires exactly eight workers and batch size eight"
            )
        all_in = _money(
            budget["all_in_maximum_spend_usd"],
            "budget.all_in_maximum_spend_usd",
        )
        infrastructure = _money(
            budget["maximum_infrastructure_spend_usd"],
            "budget.maximum_infrastructure_spend_usd",
        )
        judge = _money(
            budget["maximum_judge_spend_usd"],
            "budget.maximum_judge_spend_usd",
        )
        if infrastructure != Decimal("45"):
            raise FleetError("maximum infrastructure spend must remain exactly 45 USD")
        if judge != Decimal("5"):
            raise FleetError("maximum judge spend must remain exactly 5 USD")
        if all_in != Decimal("50") or infrastructure + judge != all_in:
            raise FleetError("all-in budget must remain the exact 45 + 5 = 50 USD split")
        if budget["included_infrastructure_costs"] != list(_INFRASTRUCTURE_COSTS):
            raise FleetError("infrastructure ceiling must include GPU, storage, and network costs")
        judge_budget_sha256 = _digest(
            budget["production_judge_budget_config_sha256"],
            "budget.production_judge_budget_config_sha256",
        )
        if judge_budget_sha256 != EXPECTED_PRODUCTION_JUDGE_BUDGET_SHA256:
            raise FleetError("production judge budget identity differs")
        claimed_budget_identity = _digest(
            budget["identity_sha256"], "budget.identity_sha256"
        )
        unsigned_budget = dict(budget)
        unsigned_budget.pop("identity_sha256")
        if claimed_budget_identity != _sha(unsigned_budget):
            raise FleetError("fleet budget identity differs")
        return cls(
            format=str(format_name),
            fleet_id=_text(raw["fleet_id"], "fleet_id"),
            worker_count=worker_count,
            all_in_maximum_spend_usd=all_in,
            maximum_infrastructure_spend_usd=infrastructure,
            maximum_judge_spend_usd=judge,
            production_judge_budget_config_sha256=judge_budget_sha256,
            budget_identity_sha256=claimed_budget_identity,
            maximum_host_lease_seconds=_integer(
                budget["maximum_host_lease_seconds"],
                "budget.maximum_host_lease_seconds",
                60,
            ),
            maximum_fetch_gib=_number(
                budget["maximum_fetch_gib"],
                "budget.maximum_fetch_gib",
                maximum=1.0,
            ),
            production_config_path=production_path,
            production_config_sha256=_digest(
                production["sha256"], "production_config.sha256"
            ),
            bundle_sha256=bundle_sha256,
            receipt_directory=Path(str(receipt)),
            batch_size=batch_size,
            adaptive_capacity_policy_path=capacity_policy_path,
            adaptive_capacity_policy_sha256=capacity_policy_sha256,
            study_config_path=study_config_path,
            study_config_sha256=study_config_sha256,
            study_identity_sha256=study_identity_sha256,
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        common = {
            "format": self.format,
            "fleet_id": self.fleet_id,
        }
        if self.format == FLEET_FORMAT:
            execution = {
                "phase_boundaries": dict(PHASE_BOUNDARIES),
                "execution_mode": _EXECUTION_MODE,
                "worker_count": self.worker_count,
            }
        else:
            if None in {
                self.adaptive_capacity_policy_path,
                self.adaptive_capacity_policy_sha256,
                self.study_config_path,
                self.study_config_sha256,
                self.study_identity_sha256,
            }:
                raise FleetError("adaptive fleet identity is incomplete")
            execution = {
                "adaptive_capacity_policy": {
                    "path": self.adaptive_capacity_policy_path,
                    "sha256": self.adaptive_capacity_policy_sha256,
                },
                "study": {
                    "path": self.study_config_path,
                    "config_sha256": self.study_config_sha256,
                    "identity_sha256": self.study_identity_sha256,
                },
                "execution_topology": {
                    "mode": _EXECUTION_MODE,
                    "worker_count": self.worker_count,
                    "batch_size": self.batch_size,
                },
            }
        identity = {
            **common,
            **execution,
            "budget": {
                "format": FLEET_BUDGET_FORMAT,
                "all_in_maximum_spend_usd": str(self.all_in_maximum_spend_usd),
                "maximum_infrastructure_spend_usd": str(
                    self.maximum_infrastructure_spend_usd
                ),
                "maximum_judge_spend_usd": str(self.maximum_judge_spend_usd),
                "production_judge_budget_config_sha256": (
                    self.production_judge_budget_config_sha256
                ),
                "included_infrastructure_costs": list(_INFRASTRUCTURE_COSTS),
                "maximum_host_lease_seconds": self.maximum_host_lease_seconds,
                "maximum_fetch_gib": self.maximum_fetch_gib,
                "identity_sha256": self.budget_identity_sha256,
            },
            "production_config": {
                "path": self.production_config_path,
                "sha256": self.production_config_sha256,
            },
            "receipt_directory": str(self.receipt_directory),
            "capability_test_access": False,
        }
        if self.format == FLEET_FORMAT:
            identity["bundle_sha256"] = self.bundle_sha256
        return identity

    @property
    def identity_sha256(self) -> str:
        return _sha(self.identity)

    def stop_after_trials(self, phase: str) -> int:
        if self.format == ADAPTIVE_FLEET_FORMAT:
            raise FleetError(
                "adaptive fleet stopping is owned by the adaptive capacity policy"
            )
        try:
            return PHASE_BOUNDARIES[phase]
        except KeyError as error:
            raise FleetError("phase must be discovery, expanded, or finalist") from error


def verify_production_config_binding(
    config: FleetConfig,
    *,
    repo: Path,
    requested_path: Path | None = None,
) -> Path:
    """Resolve and hash-check the contract-selected production config.

    The filename carries no version authority. The strict fleet contract binds
    an arbitrary safe repository-relative JSON path to its exact content hash.
    """

    root = repo.resolve()
    contract_path = root / config.production_config_path
    if requested_path is not None and requested_path.resolve() != contract_path.resolve():
        raise FleetError("requested production config path differs from fleet contract")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise FleetError("contract production config is not a regular file")
    resolved = contract_path.resolve()
    if root != resolved.parent and root not in resolved.parents:
        raise FleetError("contract production config escapes repository")
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != config.production_config_sha256:
        raise FleetError("production config content hash differs from fleet contract")
    return contract_path


def _verify_repository_file_binding(
    *,
    repo: Path,
    relative_path: str,
    expected_sha256: str,
    requested_path: Path,
    label: str,
) -> Path:
    root = repo.resolve()
    contract_path = root / relative_path
    if requested_path.resolve() != contract_path.resolve():
        raise FleetError(f"requested {label} path differs from fleet contract")
    if contract_path.is_symlink() or not contract_path.is_file():
        raise FleetError(f"contract {label} is not a regular file")
    resolved = contract_path.resolve()
    if root != resolved.parent and root not in resolved.parents:
        raise FleetError(f"contract {label} escapes repository")
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != expected_sha256:
        raise FleetError(f"{label} content hash differs from fleet contract")
    return contract_path


def verify_adaptive_fleet_bindings(
    config: FleetConfig,
    *,
    repo: Path,
    requested_capacity_policy_path: Path,
    requested_study_config_path: Path,
    observed_study_identity_sha256: str,
) -> tuple[Path, Path]:
    """Verify every adaptive input selected outside the fleet config.

    This is the adaptive launch seam: callers cannot accidentally combine a
    valid fleet contract with a different capacity policy or Optuna study.
    """

    if config.format != ADAPTIVE_FLEET_FORMAT:
        raise FleetError("adaptive execution requires an adaptive fleet config")
    if (
        config.adaptive_capacity_policy_path is None
        or config.adaptive_capacity_policy_sha256 is None
        or config.study_config_path is None
        or config.study_config_sha256 is None
        or config.study_identity_sha256 is None
    ):
        raise FleetError("adaptive fleet identity is incomplete")
    if (
        _digest(observed_study_identity_sha256, "observed_study_identity_sha256")
        != config.study_identity_sha256
    ):
        raise FleetError("observed study identity differs from fleet contract")
    policy_path = _verify_repository_file_binding(
        repo=repo,
        relative_path=config.adaptive_capacity_policy_path,
        expected_sha256=config.adaptive_capacity_policy_sha256,
        requested_path=requested_capacity_policy_path,
        label="adaptive capacity policy",
    )
    study_path = _verify_repository_file_binding(
        repo=repo,
        relative_path=config.study_config_path,
        expected_sha256=config.study_config_sha256,
        requested_path=requested_study_config_path,
        label="study config",
    )
    return policy_path, study_path


class FleetTrialWorker(Protocol):
    """One independent one-GPU worker controlled by the local study process."""

    def evaluate(self, dispatch: Mapping[str, Any]) -> EvaluationResult: ...

    def close(self) -> None: ...


WorkerFactory = Callable[[int], FleetTrialWorker]
LifecycleExecute = Callable[..., Mapping[str, Any]]
TrialTelemetryCallback = Callable[[int, str, Mapping[str, float]], None]
TrialReceiptDurableCallback = Callable[[Mapping[str, Any]], None]


def _proposal_mapping(value: Any) -> Any:
    method = getattr(value, "to_dict", None)
    return method() if callable(method) else value


def _dispatch(
    config: FleetConfig, request: BatchEvaluationRequest[Any]
) -> dict[str, Any]:
    body = {
        "format": "truth_editing_vast_fleet_dispatch_v1",
        "fleet_config_sha256": config.identity_sha256,
        "production_config_sha256": config.production_config_sha256,
        "trial_id": request.trial_id,
        "ordinal": request.ordinal,
        "proposal": _proposal_mapping(request.proposal),
        "record_ids": list(request.record_ids),
        "objective_names": list(request.objective_names),
    }
    if config.bundle_sha256 is not None:
        body["bundle_sha256"] = config.bundle_sha256
    return {**body, "request_sha256": _sha(body)}


def _result_mapping(result: EvaluationResult) -> dict[str, Any]:
    return {
        "outcome_kind": result.outcome_kind,
        "metrics": dict(result.metrics),
        "detail": result.detail,
    }


def _parse_result(raw: Any) -> EvaluationResult:
    result = _mapping(raw, "trial result")
    _exact(result, {"outcome_kind", "metrics", "detail"}, "trial result")
    kind = result["outcome_kind"]
    metrics = _mapping(result["metrics"], "trial result metrics")
    detail = result["detail"]
    if kind == "operational_failure":
        if metrics or not isinstance(detail, str) or not detail:
            raise FleetError("operational trial result is malformed")
        return EvaluationResult.operational_failure(detail)
    if kind not in {"successful", "scientifically_infeasible"}:
        raise FleetError("trial outcome kind is invalid")
    if detail is not None and not isinstance(detail, str):
        raise FleetError("trial result detail is invalid")
    parsed_metrics: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise FleetError("trial metrics must be finite numbers")
        parsed_metrics[str(name)] = float(value)
    return EvaluationResult(str(kind), parsed_metrics, detail)  # type: ignore[arg-type]


def _safe_trial_telemetry(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for name in _SAFE_TRIAL_TELEMETRY_FIELDS:
        item = value.get(name)
        if item is None:
            continue
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0
        ):
            raise FleetError("worker telemetry is malformed")
        result[name] = float(item)
    return result


def _receipt_durable_event(
    *, path: Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the privacy-minimal event for one already-verified receipt."""

    return {
        "format": RECEIPT_DURABLE_EVENT_FORMAT,
        "fleet_config_sha256": receipt["fleet_config_sha256"],
        "trial_id": receipt["trial_id"],
        "ordinal": receipt["ordinal"],
        "request_sha256": receipt["request_sha256"],
        "receipt_path": str(path.resolve()),
        "receipt_sha256": receipt["receipt_sha256"],
    }


class FleetBatchEvaluator:
    """Batch-capable evaluator used directly by ``TruthEditingStudy.run``."""

    def __init__(
        self,
        config: FleetConfig,
        *,
        worker_factory: WorkerFactory,
        telemetry: GpuTelemetryCollector | None = None,
        trial_telemetry_callback: TrialTelemetryCallback | None = None,
        receipt_directory_override: Path | None = None,
        trial_receipt_durable_callback: TrialReceiptDurableCallback | None = None,
    ) -> None:
        if receipt_directory_override is not None:
            if (
                not isinstance(receipt_directory_override, Path)
                or not receipt_directory_override.is_absolute()
                or receipt_directory_override == Path(receipt_directory_override.anchor)
            ):
                raise FleetError("receipt directory override must be an absolute non-root path")
            receipt_directory = receipt_directory_override
        else:
            receipt_directory = config.receipt_directory
        self.config = config
        self._receipt_directory = receipt_directory
        self._worker_factory = worker_factory
        self._workers: list[FleetTrialWorker] = []
        self._telemetry = telemetry
        self._trial_telemetry_callback = trial_telemetry_callback
        self._trial_receipt_durable_callback = trial_receipt_durable_callback
        self._closed = False

    @property
    def receipt_directory(self) -> Path:
        """Actual mutable storage path, intentionally outside frozen identity."""

        return self._receipt_directory

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter": "truth_editing_vast_fleet_batch_v1",
            "fleet_config_sha256": self.config.identity_sha256,
            "controller_only_optuna_observation": True,
            "completion_order_affects_suggestions": False,
        }

    def __enter__(self) -> "FleetBatchEvaluator":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _ensure_workers(self, count: int) -> None:
        if self._closed:
            raise FleetError("fleet is already closed")
        while len(self._workers) < count:
            self._workers.append(self._worker_factory(len(self._workers)))

    def _receipt_path(self, trial_id: str) -> Path:
        if not trial_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in trial_id):
            raise FleetError("trial_id is unsafe for durable receipt storage")
        return self._receipt_directory / f"{trial_id}.json"

    def _load_receipt(
        self, path: Path, dispatch: Mapping[str, Any]
    ) -> EvaluationResult:
        if path.is_symlink() or not path.is_file():
            raise FleetError("trial receipt is not a regular file")
        try:
            raw = _mapping(json.loads(path.read_text()), "trial receipt")
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FleetError("trial receipt is unreadable") from error
        receipt_format = raw.get("format")
        fields = {
                "format",
                "fleet_config_sha256",
                "trial_id",
                "ordinal",
                "request_sha256",
                "worker_slot",
                "result",
                "receipt_sha256",
        }
        if receipt_format == TRIAL_RECEIPT_FORMAT:
            fields.add("telemetry")
        elif receipt_format != LEGACY_TRIAL_RECEIPT_FORMAT:
            raise FleetError("durable trial receipt format changed")
        _exact(raw, fields, "trial receipt")
        unsigned = dict(raw)
        claimed = unsigned.pop("receipt_sha256")
        if (
            claimed != _sha(unsigned)
            or raw["fleet_config_sha256"] != self.config.identity_sha256
            or raw["trial_id"] != dispatch["trial_id"]
            or raw["ordinal"] != dispatch["ordinal"]
            or raw["request_sha256"] != dispatch["request_sha256"]
        ):
            raise FleetError("durable trial receipt identity differs from request")
        worker_slot = _integer(raw["worker_slot"], "trial receipt worker_slot", 0)
        if worker_slot >= self.config.worker_count:
            raise FleetError("trial receipt worker slot is outside fleet topology")
        telemetry = (
            _safe_trial_telemetry(raw["telemetry"])
            if receipt_format == TRIAL_RECEIPT_FORMAT
            else {}
        )
        result = _parse_result(raw["result"])
        if self._trial_receipt_durable_callback is not None:
            self._trial_receipt_durable_callback(
                _receipt_durable_event(path=path, receipt=raw)
            )
        if self._trial_telemetry_callback is not None and receipt_format == TRIAL_RECEIPT_FORMAT:
            self._trial_telemetry_callback(
                worker_slot, str(dispatch["trial_id"]), telemetry
            )
        return result

    def _save_receipt(
        self,
        path: Path,
        dispatch: Mapping[str, Any],
        worker_slot: int,
        result: EvaluationResult,
        telemetry: Mapping[str, float],
    ) -> Mapping[str, Any]:
        unsigned = {
            "format": TRIAL_RECEIPT_FORMAT,
            "fleet_config_sha256": self.config.identity_sha256,
            "trial_id": dispatch["trial_id"],
            "ordinal": dispatch["ordinal"],
            "request_sha256": dispatch["request_sha256"],
            "worker_slot": worker_slot,
            "result": _result_mapping(result),
            "telemetry": dict(telemetry),
        }
        receipt = {**unsigned, "receipt_sha256": _sha(unsigned)}
        _atomic_json(path, receipt)
        return receipt

    def evaluate_batch(
        self, requests: Sequence[BatchEvaluationRequest[Any]]
    ) -> tuple[EvaluationResult, ...]:
        frozen = tuple(requests)
        if not frozen:
            return ()
        if len({item.trial_id for item in frozen}) != len(frozen) or len(
            {item.ordinal for item in frozen}
        ) != len(frozen):
            raise FleetError("batch contains duplicate trials")
        dispatches = tuple(_dispatch(self.config, request) for request in frozen)
        results: list[EvaluationResult | None] = [None] * len(frozen)
        pending: list[tuple[int, dict[str, Any], Path]] = []
        for index, dispatch in enumerate(dispatches):
            path = self._receipt_path(str(dispatch["trial_id"]))
            if path.exists() or path.is_symlink():
                results[index] = self._load_receipt(path, dispatch)
            else:
                pending.append((index, dispatch, path))
        if pending:
            worker_count = min(self.config.worker_count, len(pending))
            self._ensure_workers(worker_count)
            assignments = [pending[slot::worker_count] for slot in range(worker_count)]

            def run_slot(slot: int) -> list[tuple[int, EvaluationResult]]:
                completed: list[tuple[int, EvaluationResult]] = []
                for index, dispatch, path in assignments[slot]:
                    trial_id = str(dispatch["trial_id"])
                    safe_telemetry: dict[str, float] = {}
                    try:
                        if self._telemetry is not None:
                            self._telemetry.begin_trial(slot, trial_id)
                        outcome = self._workers[slot].evaluate(dispatch)
                        if not isinstance(outcome, EvaluationResult):
                            raise FleetError("worker returned a non-EvaluationResult value")
                        raw_telemetry = getattr(
                            self._workers[slot], "last_telemetry", {}
                        )
                        safe_telemetry = _safe_trial_telemetry(raw_telemetry)
                    except FleetCircuitOpen:
                        raise
                    except Exception as error:
                        outcome = EvaluationResult.operational_failure(
                            f"{type(error).__name__}: {error}"
                        )
                    finally:
                        if self._telemetry is not None:
                            try:
                                self._telemetry.end_trial(slot, trial_id)
                            except ValueError:
                                pass
                    receipt = self._save_receipt(
                        path, dispatch, slot, outcome, safe_telemetry
                    )
                    if self._trial_receipt_durable_callback is not None:
                        # Worker slots invoke this seam concurrently. The
                        # durable adapter owns any ordering it requires.
                        self._trial_receipt_durable_callback(
                            _receipt_durable_event(path=path, receipt=receipt)
                        )
                    if self._trial_telemetry_callback is not None:
                        self._trial_telemetry_callback(
                            slot, trial_id, safe_telemetry
                        )
                    completed.append((index, outcome))
                return completed

            executor = ThreadPoolExecutor(max_workers=worker_count)
            futures = [executor.submit(run_slot, slot) for slot in range(worker_count)]
            try:
                for future in futures:
                    for index, result in future.result():
                        results[index] = result
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        if any(result is None for result in results):
            raise FleetError("full batch barrier did not produce every trial result")
        return tuple(result for result in results if result is not None)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
        closed = 0
        for slot, worker in enumerate(self._workers):
            try:
                worker.close()
                closed += 1
            except BaseException as error:
                errors.append(f"slot {slot}: {type(error).__name__}: {error}")
        self._closed = True
        unsigned = {
            "format": STOP_RECEIPT_FORMAT,
            "fleet_config_sha256": self.config.identity_sha256,
            "created_workers": len(self._workers),
            "closed_workers": closed,
            "all_workers_closed": not errors and closed == len(self._workers),
            "cleanup_errors": errors,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(
            self._receipt_directory.parent / "fleet-stop-receipt.json",
            {**unsigned, "receipt_sha256": _sha(unsigned)},
        )
        if errors:
            raise FleetError("fleet cleanup failed: " + "; ".join(errors))


def _phase_for_ordinal(ordinal: int) -> str:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 0 <= ordinal < 200:
        raise FleetError("trial ordinal is outside the frozen 200-trial study")
    if ordinal < PHASE_BOUNDARIES["discovery"]:
        return "discovery"
    if ordinal < PHASE_BOUNDARIES["expanded"]:
        return "expanded"
    return "finalist"


def _bind_phase(command: tuple[str, ...], phase: str) -> tuple[str, ...]:
    result = list(command)
    matches = [index for index, value in enumerate(result[:-1]) if value == "--phase"]
    if len(matches) != 1:
        raise FleetError("production worker command must contain exactly one --phase")
    result[matches[0] + 1] = phase
    return tuple(result)


class VastLifecycleTrialWorker:
    """Concrete ephemeral Vast worker with unconditional fetch/destroy semantics.

    The existing production lifecycle owns offer revalidation, upload, hard
    deadline, bounded fetch, destruction, and zero-lineage verification.  The
    fleet adapter only binds one immutable request to that lifecycle.
    """

    def __init__(
        self,
        *,
        slot: int,
        fleet_config: FleetConfig,
        production_job: ProductionVastConfig,
        offer: Offer,
        repo: Path,
        bundle: Path,
        fetch_root: Path,
        metadata_root: Path,
        workload_secret: EphemeralWorkloadSecret | None,
        vastai: str = "vastai",
        ssh_identity: Path | None = None,
        lifecycle_execute: LifecycleExecute = execute_production_lifecycle,
    ) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise FleetError("worker slot must be a nonnegative integer")
        verify_production_config_binding(fleet_config, repo=repo)
        if (
            bundle.is_symlink()
            or not bundle.is_file()
            or hashlib.sha256(bundle.read_bytes()).hexdigest()
            != fleet_config.bundle_sha256
        ):
            raise FleetError("production bundle identity differs from fleet contract")
        base = production_job.base_job
        if Decimal(str(base.maximum_cost_usd)) > fleet_config.maximum_infrastructure_spend_usd:
            raise FleetError("production job cost exceeds the all-in infrastructure bound")
        if base.maximum_elapsed_seconds > fleet_config.maximum_host_lease_seconds:
            raise FleetError("production job time exceeds persistent host lease bound")
        if base.maximum_upload_gib > fleet_config.maximum_fetch_gib:
            raise FleetError("production job fetch exceeds fleet bound")
        if "trial-result.json" not in base.expected_outputs:
            raise FleetError("production job must fetch trial-result.json")
        if not any(
            part.endswith("run_truth_editing_vast_fleet_worker.py")
            for part in production_job.workload_command
        ):
            raise FleetError("production job must select the fleet worker entrypoint")
        if (
            production_job.production_config_path != fleet_config.production_config_path
            or production_job.production_config_sha256
            != fleet_config.production_config_sha256
        ):
            raise FleetError("production job config identity differs from fleet contract")
        self.slot = slot
        self.last_telemetry: Mapping[str, float] = {}
        self._fleet = fleet_config
        self._job = production_job
        self._offer = offer
        self._bundle = bundle
        self._fetch_root = fetch_root
        self._metadata_root = metadata_root
        self._vastai = vastai
        self._ssh_identity = ssh_identity
        self._execute = lifecycle_execute
        self._workload_secret = workload_secret

    def evaluate(self, dispatch: Mapping[str, Any]) -> EvaluationResult:
        payload = _mapping(dispatch, "fleet dispatch")
        trial_id = _text(payload.get("trial_id"), "fleet dispatch trial_id")
        ordinal = _integer(payload.get("ordinal"), "fleet dispatch ordinal", 0)
        request_sha = _digest(payload.get("request_sha256"), "fleet dispatch request_sha256")
        phase = _phase_for_ordinal(ordinal)
        encoded = base64.urlsafe_b64encode(_canonical(payload)).decode("ascii")
        workload = _bind_phase(self._job.workload_command, phase) + (
            "--fleet-request-base64",
            encoded,
            "--fleet-result",
            f"{self._job.base_job.remote_output_dir}/trial-result.json",
        )
        job = replace(self._job, phase=phase, workload_command=workload)
        job.validate()
        fetch_dir = self._fetch_root / trial_id
        metadata_path = self._metadata_root / f"{trial_id}.json"
        plan = production_lifecycle_plan(
            vastai=self._vastai,
            config=job,
            offer=self._offer,
            bundle=self._bundle,
            fetch_dir=fetch_dir,
            ssh_identity=self._ssh_identity,
        )
        try:
            if self._workload_secret is None:
                raise FleetError("paid production execution requires an ephemeral workload secret")
            lifecycle_receipt = self._execute(
                plan=plan,
                config=job,
                metadata_path=metadata_path,
                workload_secret=self._workload_secret,
            )
        except Exception as error:
            if "paid semantic judge failed closed; failure_receipt_sha256=" in str(error):
                raise FleetCircuitOpen("paid semantic judge circuit opened") from None
            raise
        if lifecycle_receipt.get("destroyed") is not True:
            raise FleetError("Vast lifecycle lacks destruction evidence")
        result_path = fetch_dir / "trial-result.json"
        if result_path.is_symlink() or not result_path.is_file():
            raise FleetError("fetched trial result is missing")
        try:
            raw = _mapping(json.loads(result_path.read_text()), "worker result")
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FleetError("fetched worker result is unreadable") from error
        _exact(
            raw,
            {"format", "request_sha256", "result", "self_sha256"},
            "worker result",
        )
        unsigned = dict(raw)
        claimed = unsigned.pop("self_sha256")
        if (
            raw["format"] != "truth_editing_vast_fleet_worker_result_v1"
            or raw["request_sha256"] != request_sha
            or claimed != _sha(unsigned)
        ):
            raise FleetError("fetched worker result identity differs from request")
        return _parse_result(raw["result"])

    def close(self) -> None:
        # Each evaluate call uses execute_production_lifecycle, whose finally
        # block destroys the exact instance and proves zero remaining lineage.
        return None


class SubprocessCudaWorker:
    """Long-lived single-GPU worker process for an eight-GPU persistent host."""

    def __init__(
        self,
        slot: int,
        command: Sequence[str],
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        if not 0 <= slot < 8 or not command:
            raise FleetError("persistent CUDA worker slot/command is invalid")
        # W&B is owned by the single coordinator process.  Trial workers need
        # OpenRouter for semantic evaluation, but must never receive W&B
        # credentials or inherit a second run identity.
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("WANDB_") and not name.startswith("AWS_")
        }
        environment["CUDA_VISIBLE_DEVICES"] = str(slot)
        self.slot = slot
        self.last_telemetry: Mapping[str, float] = {}
        self._process = popen_factory(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Inherit the controller's stderr so verbose model loading cannot
            # fill an unread pipe and deadlock a long-lived worker.
            stderr=None,
            text=True,
            bufsize=1,
            env=environment,
        )

    def evaluate(self, dispatch: Mapping[str, Any]) -> EvaluationResult:
        if self._process.poll() is not None:
            raise FleetError(f"persistent CUDA worker {self.slot} exited early")
        if self._process.stdin is None or self._process.stdout is None:
            raise FleetError("persistent CUDA worker pipes are unavailable")
        self._process.stdin.write(_canonical(dict(dispatch)).decode() + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise FleetError(f"persistent CUDA worker {self.slot} returned no result")
        raw = _mapping(json.loads(line), "persistent worker response")
        if raw.get("fatal") is True:
            _exact(
                raw,
                {"fatal", "request_sha256", "failure_receipt_sha256"},
                "persistent worker paid failure",
            )
            if raw["request_sha256"] != dispatch.get("request_sha256"):
                raise FleetError("persistent worker paid failure identity differs")
            failure_sha = _digest(
                raw["failure_receipt_sha256"],
                "persistent worker paid failure receipt",
            )
            self.last_telemetry = {}
            return EvaluationResult.operational_failure(
                "paid semantic judge failed closed; "
                f"failure_receipt_sha256={failure_sha}"
            )
        if set(raw) not in (
            {"request_sha256", "result"},
            {"request_sha256", "result", "telemetry"},
        ) or raw["request_sha256"] != dispatch.get("request_sha256"):
            raise FleetError("persistent worker response identity differs")
        telemetry = raw.get("telemetry", {})
        if not isinstance(telemetry, Mapping):
            raise FleetError("persistent worker telemetry is malformed")
        safe: dict[str, float] = {}
        for name in {
            "evaluation_seconds",
            "generated_tokens",
            "generated_tokens_per_second",
            "cuda_peak_allocated_bytes",
        }:
            value = telemetry.get(name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise FleetError("persistent worker telemetry is malformed")
            safe[name] = float(value)
        self.last_telemetry = safe
        return _parse_result(raw["result"])

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        if self._process.stdin is not None:
            try:
                self._process.stdin.write('{"command":"stop"}\n')
                self._process.stdin.flush()
            except (OSError, BrokenPipeError):
                pass
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "ADAPTIVE_FLEET_FORMAT",
    "FLEET_FORMAT",
    "FLEET_BUDGET_FORMAT",
    "EXPECTED_PRODUCTION_JUDGE_BUDGET_SHA256",
    "PHASE_BOUNDARIES",
    "RECEIPT_DURABLE_EVENT_FORMAT",
    "FleetBatchEvaluator",
    "FleetConfig",
    "FleetError",
    "FleetCircuitOpen",
    "FleetTrialWorker",
    "TrialReceiptDurableCallback",
    "VastLifecycleTrialWorker",
    "SubprocessCudaWorker",
    "verify_adaptive_fleet_bindings",
    "verify_production_config_binding",
]
