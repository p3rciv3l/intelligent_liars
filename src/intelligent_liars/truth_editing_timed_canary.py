"""Timed production-parity canary for topology selection.

The canary is deliberately separate from the Vast lifecycle and the full Optuna
controller.  A caller supplies exactly one production observation; this module
validates the scientific and operational evidence, measures end-to-end time, and
writes one identity-bound throughput/cost receipt.  It never rents hardware or
calls a provider itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


CONFIG_FORMAT = "truth_editing_timed_canary_config_v1"
OBSERVATION_FORMAT = "truth_editing_timed_canary_observation_v2"
RECEIPT_FORMAT = "truth_editing_timed_canary_receipt_v2"
PERSISTENCE_STRATA = ("text", "vision", "recorded_computer_use")
FROZEN_JUDGE = {
    "deployment_alias": "glm-5.3-flash",
    "model": "z-ai/glm-5.3-flash",
    "provider_route": "z-ai/fp8",
    "allow_fallbacks": False,
    "response_healing": True,
}
_HEX = frozenset("0123456789abcdef")


class TimedCanaryError(RuntimeError):
    """The canary contract or its observed evidence failed closed."""


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
        raise TimedCanaryError("canary value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TimedCanaryError(f"{label} must be an object")
    result = dict(value)
    _canonical(result)
    return result


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    result = _object(value, label)
    if set(result) != fields:
        raise TimedCanaryError(f"{label} fields changed")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TimedCanaryError(f"{label} must be a nonempty trimmed string")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TimedCanaryError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimedCanaryError(f"{label} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise TimedCanaryError(f"{label} must be a finite number >= {minimum}")
    return result


def _digest(value: Any, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(character not in _HEX for character in result):
        raise TimedCanaryError(f"{label} must be a lowercase SHA-256")
    return result


@dataclass(frozen=True)
class TimedCanaryConfig:
    canary_id: str
    production_config_path: str
    production_config_sha256: str
    model_sha256: str
    trial_count: int
    batch_count: int
    maximum_wall_seconds: float
    minimum_generated_tokens: int
    maximum_judge_calls: int
    maximum_judge_failures: int
    maximum_judge_cost_usd: float
    maximum_persistence_kl: tuple[tuple[str, float], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TimedCanaryConfig":
        raw = _exact(
            value,
            {
                "format",
                "canary_id",
                "production_config",
                "model_sha256",
                "workload",
                "limits",
                "judge",
            },
            "timed canary config",
        )
        if raw["format"] != CONFIG_FORMAT:
            raise TimedCanaryError("timed canary config format changed")
        production = _exact(
            raw["production_config"], {"path", "sha256"}, "production_config"
        )
        production_path = _text(production["path"], "production_config.path")
        production_parts = PurePosixPath(production_path)
        if (
            production_parts.is_absolute()
            or ".." in production_parts.parts
            or not production_parts.parts
            or production_parts.parts[0] != "configs"
            or production_parts.suffix != ".json"
            or str(production_parts) != production_path
        ):
            raise TimedCanaryError(
                "production_config.path must be a safe repository-relative JSON path under configs"
            )
        workload = _exact(
            raw["workload"], {"trial_count", "batch_count"}, "workload"
        )
        trial_count = _integer(workload["trial_count"], "workload.trial_count", 1)
        batch_count = _integer(workload["batch_count"], "workload.batch_count", 1)
        if trial_count != 1 or batch_count != 1:
            raise TimedCanaryError("canary must run exactly one trial and one batch")
        judge = _exact(raw["judge"], set(FROZEN_JUDGE), "judge")
        if judge != FROZEN_JUDGE:
            raise TimedCanaryError("judge must preserve the frozen GLM-5.3 Flash route")
        limits = _exact(
            raw["limits"],
            {
                "maximum_wall_seconds",
                "minimum_generated_tokens",
                "maximum_judge_calls",
                "maximum_judge_failures",
                "maximum_judge_cost_usd",
                "maximum_persistence_kl",
            },
            "limits",
        )
        persistence = _exact(
            limits["maximum_persistence_kl"],
            set(PERSISTENCE_STRATA),
            "limits.maximum_persistence_kl",
        )
        maximum_calls = _integer(
            limits["maximum_judge_calls"], "limits.maximum_judge_calls", 1
        )
        maximum_failures = _integer(
            limits["maximum_judge_failures"], "limits.maximum_judge_failures", 1
        )
        if maximum_failures > maximum_calls:
            raise TimedCanaryError("judge failure limit exceeds call limit")
        return cls(
            canary_id=_text(raw["canary_id"], "canary_id"),
            production_config_path=production_path,
            production_config_sha256=_digest(
                production["sha256"], "production_config.sha256"
            ),
            model_sha256=_digest(raw["model_sha256"], "model_sha256"),
            trial_count=trial_count,
            batch_count=batch_count,
            maximum_wall_seconds=_number(
                limits["maximum_wall_seconds"],
                "limits.maximum_wall_seconds",
                minimum=1.0,
            ),
            minimum_generated_tokens=_integer(
                limits["minimum_generated_tokens"],
                "limits.minimum_generated_tokens",
                1,
            ),
            maximum_judge_calls=maximum_calls,
            maximum_judge_failures=maximum_failures,
            maximum_judge_cost_usd=_number(
                limits["maximum_judge_cost_usd"],
                "limits.maximum_judge_cost_usd",
            ),
            maximum_persistence_kl=tuple(
                (name, _number(persistence[name], f"maximum_persistence_kl.{name}"))
                for name in PERSISTENCE_STRATA
            ),
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "format": CONFIG_FORMAT,
            "canary_id": self.canary_id,
            "production_config": {
                "path": self.production_config_path,
                "sha256": self.production_config_sha256,
            },
            "model_sha256": self.model_sha256,
            "workload": {
                "trial_count": self.trial_count,
                "batch_count": self.batch_count,
            },
            "limits": {
                "maximum_wall_seconds": self.maximum_wall_seconds,
                "minimum_generated_tokens": self.minimum_generated_tokens,
                "maximum_judge_calls": self.maximum_judge_calls,
                "maximum_judge_failures": self.maximum_judge_failures,
                "maximum_judge_cost_usd": self.maximum_judge_cost_usd,
                "maximum_persistence_kl": dict(self.maximum_persistence_kl),
            },
            "judge": dict(FROZEN_JUDGE),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha(self.identity)


class JudgeCircuitBreaker:
    """Small in-process guard used by the live canary judge adapter.

    Callers must invoke ``before_call`` immediately before transport activity and
    ``record_call`` exactly once afterward.  The breaker opens on the configured
    failure, call, or price ceiling; it never retries on its own.
    """

    def __init__(self, config: TimedCanaryConfig) -> None:
        self._config = config
        self._attempted = 0
        self._succeeded = 0
        self._failed = 0
        self._cost = 0.0
        self._open = False
        self._call_in_flight = False

    def before_call(self) -> None:
        if self._open:
            raise TimedCanaryError("judge circuit breaker is open")
        if self._call_in_flight:
            raise TimedCanaryError("judge circuit breaker already has a call in flight")
        self._call_in_flight = True

    def record_call(self, *, success: bool, cost_usd: float) -> None:
        if not self._call_in_flight:
            raise TimedCanaryError("judge call was not admitted by the circuit breaker")
        if not isinstance(success, bool):
            raise TimedCanaryError("judge call success must be boolean")
        cost = _number(cost_usd, "judge call cost_usd")
        self._call_in_flight = False
        self._attempted += 1
        self._cost += cost
        if success:
            self._succeeded += 1
        else:
            self._failed += 1
        self._open = (
            self._attempted >= self._config.maximum_judge_calls
            or self._failed >= self._config.maximum_judge_failures
            or self._cost >= self._config.maximum_judge_cost_usd
        )

    def receipt(self) -> Mapping[str, Any]:
        if self._call_in_flight:
            raise TimedCanaryError("judge circuit breaker has an unreceipted call")
        return {
            "attempted_calls": self._attempted,
            "successful_calls": self._succeeded,
            "failed_calls": self._failed,
            "cost_usd": self._cost,
            "circuit_opened": self._open,
        }


def _validate_observation(
    raw_value: Mapping[str, Any], config: TimedCanaryConfig
) -> dict[str, Any]:
    raw = _exact(
        raw_value,
        {
            "format",
            "canary_id",
            "production_config_path",
            "production_config_sha256",
            "model_sha256",
            "actual_model_loaded",
            "batch_count",
            "generated_tokens",
            "generation_seconds",
            "judge",
            "persistence_kl",
            "trials",
        },
        "canary observation",
    )
    if raw["format"] != OBSERVATION_FORMAT:
        raise TimedCanaryError("canary observation format changed")
    if raw["canary_id"] != config.canary_id:
        raise TimedCanaryError("canary observation identity differs from config")
    if (
        raw["production_config_path"] != config.production_config_path
        or raw["production_config_sha256"] != config.production_config_sha256
    ):
        raise TimedCanaryError("canary production config identity differs")
    if raw["model_sha256"] != config.model_sha256 or raw["actual_model_loaded"] is not True:
        raise TimedCanaryError("canary did not prove the actual model was loaded")
    if raw["batch_count"] != 1:
        raise TimedCanaryError("canary must report exactly one batch")
    tokens = _integer(raw["generated_tokens"], "generated_tokens", 1)
    if tokens < config.minimum_generated_tokens:
        raise TimedCanaryError("canary generated too few tokens for a TPS measurement")
    _number(raw["generation_seconds"], "generation_seconds", minimum=1e-9)

    judge = _exact(
        raw["judge"],
        {
            "deployment_alias",
            "model",
            "provider_route",
            "fallback_used",
            "response_healing_used",
            "attempted_calls",
            "successful_calls",
            "failed_calls",
            "cost_usd",
            "elapsed_seconds",
            "circuit_opened",
        },
        "judge observation",
    )
    expected_judge = dict(FROZEN_JUDGE)
    if (
        judge["deployment_alias"] != expected_judge["deployment_alias"]
        or judge["model"] != expected_judge["model"]
        or judge["provider_route"] != expected_judge["provider_route"]
        or judge["fallback_used"] is not False
        or judge["response_healing_used"] is not True
    ):
        raise TimedCanaryError("judge observation differs from the frozen GLM-5.3 Flash route")
    attempted = _integer(judge["attempted_calls"], "judge.attempted_calls")
    succeeded = _integer(judge["successful_calls"], "judge.successful_calls")
    failed = _integer(judge["failed_calls"], "judge.failed_calls")
    judge_cost = _number(judge["cost_usd"], "judge.cost_usd")
    _number(judge["elapsed_seconds"], "judge.elapsed_seconds")
    circuit_opened = judge["circuit_opened"]
    if not isinstance(circuit_opened, bool):
        raise TimedCanaryError("judge.circuit_opened must be boolean")
    if (
        attempted != succeeded + failed
        or attempted == 0
        or attempted > config.maximum_judge_calls
        or failed > config.maximum_judge_failures
        or judge_cost > config.maximum_judge_cost_usd
        or (
            failed >= config.maximum_judge_failures
            or attempted >= config.maximum_judge_calls
            or judge_cost >= config.maximum_judge_cost_usd
        )
        != circuit_opened
    ):
        raise TimedCanaryError("judge circuit breaker evidence is invalid")

    observed_kl = _exact(
        raw["persistence_kl"], set(PERSISTENCE_STRATA), "persistence-KL strata"
    )
    for name, maximum in config.maximum_persistence_kl:
        if _number(observed_kl[name], f"persistence_kl.{name}") > maximum:
            raise TimedCanaryError(f"persistence-KL limit exceeded for {name}")

    trials = raw["trials"]
    if not isinstance(trials, list) or len(trials) != 1:
        raise TimedCanaryError("canary must report exactly one trial output")
    trial = _exact(
        trials[0],
        {"trial_id", "outcome_kind", "metrics", "output_sha256"},
        "trial output",
    )
    _text(trial["trial_id"], "trial_id")
    if trial["outcome_kind"] not in {"successful", "scientifically_infeasible"}:
        raise TimedCanaryError("canary trial did not produce a scientific result")
    metrics = _object(trial["metrics"], "trial metrics")
    if not metrics:
        raise TimedCanaryError("canary trial metrics must not be empty")
    for name, value in metrics.items():
        _text(name, "trial metric name")
        _number(value, f"trial metric {name}")
    _digest(trial["output_sha256"], "trial output SHA-256")
    return raw


def run_timed_canary(
    *,
    config: TimedCanaryConfig,
    workload: Callable[[], Mapping[str, Any]],
    gpu_hourly_usd: float,
    receipt_path: Path,
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    """Execute one injected workload and write a receipt only after every gate passes."""

    hourly = _number(gpu_hourly_usd, "gpu_hourly_usd", minimum=1e-12)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise TimedCanaryError("canary receipt path already exists")
    started = monotonic()
    observation = _validate_observation(workload(), config)
    finished = monotonic()
    elapsed = finished - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise TimedCanaryError("canary wall-clock measurement is invalid")
    if elapsed > config.maximum_wall_seconds:
        raise TimedCanaryError("canary exceeded its wall-clock limit")
    judge_elapsed = float(observation["judge"]["elapsed_seconds"])
    if judge_elapsed > elapsed:
        raise TimedCanaryError("judge elapsed time exceeds measured canary wall time")
    generation_seconds = float(observation["generation_seconds"])
    tokens = int(observation["generated_tokens"])
    trials_per_hour = 3600.0 / elapsed
    projected_hours = 200.0 / trials_per_hour
    unsigned = {
        "format": RECEIPT_FORMAT,
        "canary_config_sha256": config.identity_sha256,
        "production_config_path": config.production_config_path,
        "production_config_sha256": config.production_config_sha256,
        "model_sha256": config.model_sha256,
        "observation_sha256": _sha(observation),
        "trial_id": observation["trials"][0]["trial_id"],
        "trial_outcome_kind": observation["trials"][0]["outcome_kind"],
        "trial_output_sha256": observation["trials"][0]["output_sha256"],
        "generated_tokens": tokens,
        "generation_seconds": generation_seconds,
        "tokens_per_second": tokens / generation_seconds,
        "measured_wall_seconds": elapsed,
        "gpu_hourly_usd": hourly,
        "estimated_canary_cost_usd": hourly * elapsed / 3600.0,
        "single_worker_trials_per_hour": trials_per_hour,
        "single_worker_200_trial_hours": projected_hours,
        "single_worker_200_trial_cost_usd": projected_hours * hourly,
        "judge_calls": observation["judge"]["attempted_calls"],
        "judge_cost_usd": observation["judge"]["cost_usd"],
        "judge_elapsed_seconds": judge_elapsed,
        "persistence_kl": observation["persistence_kl"],
        "software_and_live_canary_passed": True,
    }
    receipt = {**unsigned, "receipt_sha256": _sha(unsigned)}
    _atomic_json(receipt_path, receipt)
    return receipt


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    temporary.replace(path)


__all__ = [
    "CONFIG_FORMAT",
    "OBSERVATION_FORMAT",
    "RECEIPT_FORMAT",
    "TimedCanaryConfig",
    "TimedCanaryError",
    "JudgeCircuitBreaker",
    "run_timed_canary",
]
