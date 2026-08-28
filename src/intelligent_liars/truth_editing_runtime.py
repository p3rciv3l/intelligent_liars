"""Offline planning contracts for the truth-editing optimization runtime.

This module is deliberately a planning seam, not an execution engine.  It
describes the production runtime identity, expands a deterministic tiered
schedule into synchronous batches, and estimates a conservative wall-clock
bound from a measured throughput receipt.  Nothing here loads a model,
starts Optuna, contacts a provider, or mutates cloud state.

The production contract is intentionally narrow:

* one persistent Transformers model per GPU worker;
* explicit ``cuda:0`` placement inside each worker (the worker's visible GPU
  is selected outside this module);
* BF16, FlashAttention-2, cache enabled, no quantization or speculation; and
* synchronous batches: the next batch is not suggested until the current one
  has completed.

The target checkpoint and all of those runtime choices are part of the
identity.  A changed identity is a new experiment, never a compatible
estimate or resume.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


RUNTIME_PLAN_FORMAT = "truth_editing_runtime_plan_v1"
RUNTIME_IDENTITY_FORMAT = "truth_editing_runtime_identity_v1"
RUNTIME_ESTIMATE_FORMAT = "truth_editing_runtime_estimate_v1"
CANARY_PLAN_FORMAT = "truth_editing_runtime_canary_v1"
MAX_WALL_CLOCK_SECONDS = 24 * 60 * 60

TARGET_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
TARGET_MODEL_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"


class RuntimePlanError(ValueError):
    """Raised when a runtime plan is not a strict, safe offline contract."""


class RuntimeBudgetError(RuntimePlanError):
    """Raised when the conservative p90 estimate exceeds the wall-clock cap."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimePlanError("runtime value is not canonical JSON") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise RuntimePlanError(f"{name} has unknown keys: {sorted(unknown)}")
    if missing:
        raise RuntimePlanError(f"{name} is missing keys: {sorted(missing)}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimePlanError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimePlanError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RuntimePlanError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class RuntimeIdentity:
    """The complete model/inference identity required by this study."""

    model_id: str = TARGET_MODEL_ID
    model_revision: str = TARGET_MODEL_REVISION
    inference_backend: str = "transformers"
    dtype: str = "bfloat16"
    attention_implementation: str = "flash_attention_2"
    quantization: str | None = None
    speculative_decoding: bool = False
    device_map: str = "cuda:0"
    local_files_only: bool = True
    use_cache: bool = True
    persistent_model_per_gpu: bool = True
    model_loads_per_worker: int = 1
    workers_per_gpu: int = 1

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": RUNTIME_IDENTITY_FORMAT,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "inference_backend": self.inference_backend,
            "dtype": self.dtype,
            "attention_implementation": self.attention_implementation,
            "quantization": self.quantization,
            "speculative_decoding": self.speculative_decoding,
            "device_map": self.device_map,
            "local_files_only": self.local_files_only,
            "use_cache": self.use_cache,
            "persistent_model_per_gpu": self.persistent_model_per_gpu,
            "model_loads_per_worker": self.model_loads_per_worker,
            "workers_per_gpu": self.workers_per_gpu,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeIdentity":
        if not isinstance(value, Mapping):
            raise RuntimePlanError("runtime_identity must be an object")
        expected = {
            "format",
            "model_id",
            "model_revision",
            "inference_backend",
            "dtype",
            "attention_implementation",
            "quantization",
            "speculative_decoding",
            "device_map",
            "local_files_only",
            "use_cache",
            "persistent_model_per_gpu",
            "model_loads_per_worker",
            "workers_per_gpu",
        }
        _require_keys(value, expected, "runtime_identity")
        if value["format"] != RUNTIME_IDENTITY_FORMAT:
            raise RuntimePlanError("runtime identity format is unsupported")
        strings = (
            "model_id",
            "model_revision",
            "inference_backend",
            "dtype",
            "attention_implementation",
            "device_map",
        )
        for name in strings:
            if not isinstance(value[name], str) or not value[name]:
                raise RuntimePlanError(f"runtime_identity.{name} must be a non-empty string")
        if value["quantization"] is not None and not isinstance(value["quantization"], str):
            raise RuntimePlanError("runtime_identity.quantization must be null or a string")
        for name in (
            "speculative_decoding",
            "local_files_only",
            "use_cache",
            "persistent_model_per_gpu",
        ):
            if not isinstance(value[name], bool):
                raise RuntimePlanError(f"runtime_identity.{name} must be boolean")
        return cls(
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            inference_backend=value["inference_backend"],
            dtype=value["dtype"],
            attention_implementation=value["attention_implementation"],
            quantization=value["quantization"],
            speculative_decoding=value["speculative_decoding"],
            device_map=value["device_map"],
            local_files_only=value["local_files_only"],
            use_cache=value["use_cache"],
            persistent_model_per_gpu=value["persistent_model_per_gpu"],
            model_loads_per_worker=_positive_int(
                value["model_loads_per_worker"], "runtime_identity.model_loads_per_worker"
            ),
            workers_per_gpu=_positive_int(
                value["workers_per_gpu"], "runtime_identity.workers_per_gpu"
            ),
        )


DEFAULT_RUNTIME_IDENTITY = RuntimeIdentity()


def validate_runtime_identity(
    actual: RuntimeIdentity, expected: RuntimeIdentity = DEFAULT_RUNTIME_IDENTITY
) -> None:
    """Fail closed unless ``actual`` is exactly the frozen expected identity."""

    if actual.to_mapping() != expected.to_mapping():
        changed = sorted(
            key
            for key in actual.to_mapping()
            if actual.to_mapping()[key] != expected.to_mapping()[key]
        )
        raise RuntimePlanError(
            "runtime identity drift: " + ", ".join(changed or ["unknown field"])
        )
    if actual.persistent_model_per_gpu is not True:
        raise RuntimePlanError("runtime identity requires one persistent model per GPU")
    if actual.model_loads_per_worker != 1 or actual.workers_per_gpu != 1:
        raise RuntimePlanError("runtime identity requires one model load and one worker per GPU")
    if actual.quantization is not None or actual.speculative_decoding is not False:
        raise RuntimePlanError("quantization and speculative decoding are disabled")


@dataclass(frozen=True)
class RuntimeTier:
    """One deterministic Optuna workload tier."""

    name: str
    trial_count: int
    evaluation_items: int
    replicate_count: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise RuntimePlanError("tier name must be a non-empty string")
        _positive_int(self.trial_count, f"tier {self.name}.trial_count")
        _positive_int(self.evaluation_items, f"tier {self.name}.evaluation_items")
        _positive_int(self.replicate_count, f"tier {self.name}.replicate_count")
        _positive_int(self.max_new_tokens, f"tier {self.name}.max_new_tokens")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeTier":
        if not isinstance(value, Mapping):
            raise RuntimePlanError("runtime tier must be an object")
        _require_keys(
            value,
            {"name", "trial_count", "evaluation_items", "replicate_count", "max_new_tokens"},
            "runtime tier",
        )
        return cls(
            name=value["name"],
            trial_count=value["trial_count"],
            evaluation_items=value["evaluation_items"],
            replicate_count=value["replicate_count"],
            max_new_tokens=value["max_new_tokens"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trial_count": self.trial_count,
            "evaluation_items": self.evaluation_items,
            "replicate_count": self.replicate_count,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class RuntimePlan:
    """Immutable, offline-only execution plan."""

    identity: RuntimeIdentity
    tiers: tuple[RuntimeTier, ...]
    batch_size: int
    gpu_count: int
    base_seed: int
    p90_cap_seconds: float = MAX_WALL_CLOCK_SECONDS
    format: str = RUNTIME_PLAN_FORMAT

    def __post_init__(self) -> None:
        if self.format != RUNTIME_PLAN_FORMAT:
            raise RuntimePlanError("runtime plan format is unsupported")
        validate_runtime_identity(self.identity)
        if not self.tiers:
            raise RuntimePlanError("runtime plan requires at least one tier")
        if len({tier.name for tier in self.tiers}) != len(self.tiers):
            raise RuntimePlanError("runtime tier names must be unique")
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.gpu_count, "gpu_count")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise RuntimePlanError("base_seed must be an integer")
        cap = _nonnegative_float(self.p90_cap_seconds, "p90_cap_seconds")
        if cap != MAX_WALL_CLOCK_SECONDS:
            raise RuntimePlanError("p90_cap_seconds must remain the frozen 24-hour cap")

    @property
    def total_trials(self) -> int:
        return sum(tier.trial_count for tier in self.tiers)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "runtime_identity": self.identity.to_mapping(),
            "tiers": [tier.to_mapping() for tier in self.tiers],
            "execution": {
                "batch_size": self.batch_size,
                "gpu_count": self.gpu_count,
                "base_seed": self.base_seed,
                "synchronous_batches": True,
                "p90_cap_seconds": self.p90_cap_seconds,
            },
            "offline_only": True,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimePlan":
        _require_keys(
            value,
            {"format", "runtime_identity", "tiers", "execution", "offline_only"},
            "runtime plan",
        )
        if value["format"] != RUNTIME_PLAN_FORMAT:
            raise RuntimePlanError("runtime plan format is unsupported")
        if value["offline_only"] is not True:
            raise RuntimePlanError("runtime plan must be offline_only")
        raw_tiers = value["tiers"]
        if not isinstance(raw_tiers, list) or not raw_tiers:
            raise RuntimePlanError("runtime plan tiers must be a non-empty list")
        execution = value["execution"]
        if not isinstance(execution, Mapping):
            raise RuntimePlanError("runtime plan execution must be an object")
        _require_keys(
            execution,
            {"batch_size", "gpu_count", "base_seed", "synchronous_batches", "p90_cap_seconds"},
            "runtime plan execution",
        )
        if execution["synchronous_batches"] is not True:
            raise RuntimePlanError("runtime plan requires synchronous batches")
        identity = value["runtime_identity"]
        if not isinstance(identity, Mapping):
            raise RuntimePlanError("runtime plan runtime_identity must be an object")
        return cls(
            format=value["format"],
            identity=RuntimeIdentity.from_mapping(identity),
            tiers=tuple(RuntimeTier.from_mapping(tier) for tier in raw_tiers),
            batch_size=execution["batch_size"],
            gpu_count=execution["gpu_count"],
            base_seed=execution["base_seed"],
            p90_cap_seconds=execution["p90_cap_seconds"],
        )


@dataclass(frozen=True)
class SynchronousBatch:
    """A batch of suggestions that must complete before the next batch."""

    batch_id: int
    tier_name: str
    trial_indices: tuple[int, ...]
    trial_seeds: tuple[int, ...]
    replicate_count: int


def build_synchronous_batches(plan: RuntimePlan) -> tuple[SynchronousBatch, ...]:
    """Expand tiers in order into deterministic, non-overlapping trial batches."""

    batches: list[SynchronousBatch] = []
    next_trial = 0
    for tier in plan.tiers:
        indices = tuple(range(next_trial, next_trial + tier.trial_count))
        next_trial += tier.trial_count
        for offset in range(0, tier.trial_count, plan.batch_size):
            batch_indices = indices[offset : offset + plan.batch_size]
            batches.append(
                SynchronousBatch(
                    batch_id=len(batches),
                    tier_name=tier.name,
                    trial_indices=batch_indices,
                    trial_seeds=tuple(
                        int.from_bytes(
                            hashlib.sha256(
                                f"{plan.base_seed}:{tier.name}:{trial_index}".encode(
                                    "utf-8"
                                )
                            ).digest()[:8],
                            "big",
                        )
                        % (2**31 - 1)
                        for trial_index in batch_indices
                    ),
                    replicate_count=tier.replicate_count,
                )
            )
    return tuple(batches)


@dataclass(frozen=True)
class ThroughputBenchmark:
    """A measured decode receipt used by the pure wall-clock estimator."""

    tokens_per_second: float
    p90_slowdown: float = 1.20
    model_load_seconds: float = 180.0
    batch_overhead_seconds: float = 2.0
    gpu_count: int = 8

    def __post_init__(self) -> None:
        if _nonnegative_float(self.tokens_per_second, "tokens_per_second") <= 0:
            raise RuntimePlanError("tokens_per_second must be positive")
        if _nonnegative_float(self.p90_slowdown, "p90_slowdown") <= 0:
            raise RuntimePlanError("p90_slowdown must be positive")
        _nonnegative_float(self.model_load_seconds, "model_load_seconds")
        _nonnegative_float(self.batch_overhead_seconds, "batch_overhead_seconds")
        _positive_int(self.gpu_count, "gpu_count")


@dataclass(frozen=True)
class RuntimeEstimate:
    format: str
    runtime_identity_sha256: str
    total_trials: int
    total_sequences: int
    total_tokens: int
    batch_count: int
    gpu_count: int
    median_seconds: float
    p90_seconds: float
    p90_cap_seconds: float

    @property
    def p90_hours(self) -> float:
        return self.p90_seconds / 3600.0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "total_trials": self.total_trials,
            "total_sequences": self.total_sequences,
            "total_tokens": self.total_tokens,
            "batch_count": self.batch_count,
            "gpu_count": self.gpu_count,
            "median_seconds": self.median_seconds,
            "p90_seconds": self.p90_seconds,
            "p90_cap_seconds": self.p90_cap_seconds,
        }


def estimate_runtime(
    plan: RuntimePlan,
    benchmark: ThroughputBenchmark,
    *,
    actual_runtime_identity: RuntimeIdentity | None = None,
) -> RuntimeEstimate:
    """Estimate and enforce the conservative p90 wall-clock budget.

    Throughput is aggregate decode throughput for one GPU.  Workers are
    persistent and identical, so decode work divides by the configured GPU
    count.  Model hydration is parallel worker startup and is counted once;
    every synchronous batch pays its measured scheduling overhead.
    """

    validate_runtime_identity(plan.identity)
    if actual_runtime_identity is not None:
        validate_runtime_identity(actual_runtime_identity, plan.identity)
    if benchmark.gpu_count != plan.gpu_count:
        raise RuntimePlanError(
            "benchmark GPU count does not match the runtime plan; refusing to estimate"
        )
    total_sequences = sum(
        tier.trial_count * tier.replicate_count * tier.evaluation_items
        for tier in plan.tiers
    )
    total_tokens = sum(
        tier.trial_count
        * tier.replicate_count
        * tier.evaluation_items
        * tier.max_new_tokens
        for tier in plan.tiers
    )
    batches = build_synchronous_batches(plan)
    median_seconds = (
        benchmark.model_load_seconds
        + total_tokens / benchmark.tokens_per_second / benchmark.gpu_count
        + len(batches) * benchmark.batch_overhead_seconds
    )
    p90_seconds = (
        benchmark.model_load_seconds
        + (
            total_tokens / benchmark.tokens_per_second / benchmark.gpu_count
            + len(batches) * benchmark.batch_overhead_seconds
        )
        * benchmark.p90_slowdown
    )
    if p90_seconds > plan.p90_cap_seconds:
        raise RuntimeBudgetError(
            f"estimated p90 runtime {p90_seconds / 3600.0:.2f}h exceeds the 24-hour cap"
        )
    return RuntimeEstimate(
        format=RUNTIME_ESTIMATE_FORMAT,
        runtime_identity_sha256=plan.identity.identity_sha256,
        total_trials=plan.total_trials,
        total_sequences=total_sequences,
        total_tokens=total_tokens,
        batch_count=len(batches),
        gpu_count=benchmark.gpu_count,
        median_seconds=median_seconds,
        p90_seconds=p90_seconds,
        p90_cap_seconds=plan.p90_cap_seconds,
    )


@dataclass(frozen=True)
class CanaryPlan:
    """A plan-only operational canary using the production runtime identity."""

    format: str
    runtime_identity: RuntimeIdentity
    sample_count: int
    batch_size: int
    max_new_tokens: int
    execution_mode: str
    network_access: bool
    cloud_mutation: bool


def build_canary_plan(plan: RuntimePlan) -> CanaryPlan:
    """Build a production-parity canary description without executing it."""

    validate_runtime_identity(plan.identity)
    return CanaryPlan(
        format=CANARY_PLAN_FORMAT,
        runtime_identity=plan.identity,
        sample_count=16,
        batch_size=plan.batch_size,
        max_new_tokens=32,
        execution_mode="plan_only",
        network_access=False,
        cloud_mutation=False,
    )


def load_runtime_plan(path: Path) -> RuntimePlan:
    """Load one strict local JSON plan; no remote reads are attempted."""

    if path.is_symlink() or not path.is_file():
        raise RuntimePlanError(f"runtime plan is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimePlanError(f"unable to read runtime plan: {path}") from error
    if not isinstance(value, Mapping):
        raise RuntimePlanError("runtime plan JSON must be an object")
    return RuntimePlan.from_mapping(value)


def plan_report(
    plan: RuntimePlan, benchmark: ThroughputBenchmark
) -> dict[str, Any]:
    """Return the CLI's deterministic, machine-readable offline report."""

    estimate = estimate_runtime(plan, benchmark)
    canary = build_canary_plan(plan)
    return {
        "format": RUNTIME_PLAN_FORMAT,
        "runtime_identity": plan.identity.to_mapping(),
        "runtime_identity_sha256": plan.identity.identity_sha256,
        "estimate": estimate.to_mapping(),
        "synchronous_batches": [
            {
                "batch_id": batch.batch_id,
                "tier_name": batch.tier_name,
                "trial_indices": list(batch.trial_indices),
                "trial_seeds": list(batch.trial_seeds),
                "replicate_count": batch.replicate_count,
            }
            for batch in build_synchronous_batches(plan)
        ],
        "canary": {
            "format": canary.format,
            "sample_count": canary.sample_count,
            "batch_size": canary.batch_size,
            "max_new_tokens": canary.max_new_tokens,
            "execution_mode": canary.execution_mode,
            "network_access": canary.network_access,
            "cloud_mutation": canary.cloud_mutation,
        },
    }
