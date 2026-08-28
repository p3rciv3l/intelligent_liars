"""Repeat-calibrated capability-preservation KL ceilings.

This module owns the artifact seam between repeated base-vs-base preservation
measurements and the evaluator's immutable KL gates.  It performs no model
loading or inference.  Building and opening both replay the exact source
receipts, so a self-consistent but unsupported threshold file fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


CALIBRATION_FORMAT: Literal[
    "truth_editing_preservation_kl_calibration_v1"
] = "truth_editing_preservation_kl_calibration_v1"
_RECEIPT_FORMAT = "truth_editing_preservation_receipt_v1"
BASE_REPEAT_RECEIPT_FORMAT = "truth_editing_preservation_base_repeat_receipt_v1"
_TIERS = ("trial", "promoted", "finalist")
_STRATA = ("text", "vision", "recorded_computer_use")
_HEX = frozenset("0123456789abcdef")


class PreservationThresholdCalibrationError(ValueError):
    """Calibration evidence or artifact identity is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise PreservationThresholdCalibrationError(
            "value is not canonical JSON"
        ) from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreservationThresholdCalibrationError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PreservationThresholdCalibrationError(f"{name} must be an array")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise PreservationThresholdCalibrationError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PreservationThresholdCalibrationError(
            f"{name} must be a nonempty trimmed string"
        )
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PreservationThresholdCalibrationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreservationThresholdCalibrationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PreservationThresholdCalibrationError(
            f"{name} must be finite and non-negative"
        )
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreservationThresholdCalibrationError(f"{name} must be positive integer")
    return value


def _safe_relative(path: Path, name: str) -> Path:
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise PreservationThresholdCalibrationError(
            f"{name} must be a safe relative path"
        )
    return path


@dataclass(frozen=True)
class _ParsedReceipt:
    repeat_plan_sha256: str
    repeat_index: int
    collector_identity_sha256: str
    tier: str
    self_sha256: str
    spec_sha256: str
    edited_model_sha256: str
    strata: tuple[tuple[str, float], ...]


def _parse_inner_receipt(
    value: Any,
) -> tuple[str, str, str, str, tuple[tuple[str, float], ...]]:
    raw = _object(value, "preservation receipt")
    fields = {
        "format",
        "spec_sha256",
        "edited_model_sha256",
        "tier",
        "strata",
        "aggregate_kl",
        "vision_tower_byte_identical",
        "self_sha256",
    }
    _exact(raw, fields, "preservation receipt")
    if raw["format"] != _RECEIPT_FORMAT:
        raise PreservationThresholdCalibrationError(
            "unsupported preservation receipt format"
        )
    claimed = _sha(raw["self_sha256"], "preservation receipt.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if _hash(unsigned) != claimed:
        raise PreservationThresholdCalibrationError(
            "preservation receipt hash mismatch"
        )
    tier = _text(raw["tier"], "preservation receipt.tier")
    if tier not in _TIERS:
        raise PreservationThresholdCalibrationError("unknown preservation tier")
    if raw["vision_tower_byte_identical"] is not True:
        raise PreservationThresholdCalibrationError(
            "base repeat must preserve the vision tower"
        )
    strata: list[tuple[str, float]] = []
    weighted = 0.0
    tokens = 0
    for index, value in enumerate(_array(raw["strata"], "preservation receipt.strata")):
        item = _object(value, f"preservation receipt.strata[{index}]")
        _exact(
            item,
            {"stratum", "record_count", "assistant_token_count", "forward_kl"},
            f"preservation receipt.strata[{index}]",
        )
        stratum = _text(item["stratum"], "stratum")
        if stratum not in _STRATA:
            raise PreservationThresholdCalibrationError("unknown preservation stratum")
        _positive_integer(item["record_count"], "stratum.record_count")
        token_count = _positive_integer(
            item["assistant_token_count"], "stratum.assistant_token_count"
        )
        kl = _finite_nonnegative(item["forward_kl"], "stratum.forward_kl")
        strata.append((stratum, kl))
        weighted += token_count * kl
        tokens += token_count
    if tuple(name for name, _ in strata) != _STRATA:
        raise PreservationThresholdCalibrationError(
            "preservation receipt must contain ordered complete strata"
        )
    aggregate = _finite_nonnegative(raw["aggregate_kl"], "aggregate_kl")
    if not math.isclose(aggregate, weighted / tokens, rel_tol=1e-12, abs_tol=1e-12):
        raise PreservationThresholdCalibrationError(
            "preservation receipt aggregate KL differs"
        )
    return (
        claimed,
        _sha(raw["spec_sha256"], "spec_sha256"),
        _sha(raw["edited_model_sha256"], "edited_model_sha256"),
        tier,
        tuple(strata),
    )


def _parse_receipt(value: Any) -> _ParsedReceipt:
    raw = _object(value, "base repeat receipt")
    _exact(
        raw,
        {
            "format",
            "repeat_plan_sha256",
            "repeat_index",
            "base_model_sha256",
            "tier",
            "collector_identity_sha256",
            "preservation_receipt",
            "self_sha256",
        },
        "base repeat receipt",
    )
    if raw["format"] != BASE_REPEAT_RECEIPT_FORMAT:
        raise PreservationThresholdCalibrationError(
            "unsupported base repeat receipt format"
        )
    claimed = _sha(raw["self_sha256"], "base repeat receipt.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if _hash(unsigned) != claimed:
        raise PreservationThresholdCalibrationError("base repeat receipt hash mismatch")
    repeat_index = raw["repeat_index"]
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 0:
        raise PreservationThresholdCalibrationError(
            "base repeat receipt index must be non-negative integer"
        )
    tier = _text(raw["tier"], "base repeat receipt.tier")
    if tier not in _TIERS:
        raise PreservationThresholdCalibrationError("unknown preservation tier")
    inner_sha, spec_sha, edited_sha, inner_tier, strata = _parse_inner_receipt(
        raw["preservation_receipt"]
    )
    del inner_sha  # The outer self-hash already binds the complete inner receipt.
    base_sha = _sha(raw["base_model_sha256"], "base repeat base_model_sha256")
    if edited_sha != base_sha:
        raise PreservationThresholdCalibrationError(
            "base repeat inner receipt must score the frozen base model"
        )
    if inner_tier != tier:
        raise PreservationThresholdCalibrationError(
            "base repeat outer and inner tiers differ"
        )
    return _ParsedReceipt(
        repeat_plan_sha256=_sha(raw["repeat_plan_sha256"], "repeat_plan_sha256"),
        repeat_index=repeat_index,
        collector_identity_sha256=_sha(
            raw["collector_identity_sha256"], "collector_identity_sha256"
        ),
        tier=tier,
        self_sha256=claimed,
        spec_sha256=spec_sha,
        edited_model_sha256=base_sha,
        strata=strata,
    )


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    # Conservative nearest-rank empirical quantile: never interpolates below an
    # actually observed order statistic.
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _validate_method(
    minimum_repeats: Any,
    quantile: Any,
    absolute_margin: Any,
    relative_margin: Any,
) -> tuple[int, float, float, float]:
    repeats = _positive_integer(minimum_repeats, "minimum_repeats")
    if repeats < 5:
        raise PreservationThresholdCalibrationError(
            "minimum_repeats must be at least 5"
        )
    q = _finite_nonnegative(quantile, "quantile")
    if not 0.5 <= q < 1.0:
        raise PreservationThresholdCalibrationError(
            "quantile must be at least 0.5 and below 1"
        )
    absolute = _finite_nonnegative(absolute_margin, "absolute_margin")
    relative = _finite_nonnegative(relative_margin, "relative_margin")
    if absolute <= 0:
        raise PreservationThresholdCalibrationError(
            "absolute_margin must be positive"
        )
    return repeats, q, absolute, relative


def _derive_payload(
    *,
    calibration_id: str,
    base_model_sha256: str,
    sources: Sequence[tuple[Path, _ParsedReceipt]],
    minimum_repeats: int,
    quantile: float,
    absolute_margin: float,
    relative_margin: float,
) -> dict[str, Any]:
    if not sources:
        raise PreservationThresholdCalibrationError("source receipts must not be empty")
    spec_ids = {receipt.spec_sha256 for _, receipt in sources}
    if len(spec_ids) != 1:
        raise PreservationThresholdCalibrationError(
            "source receipts must share one preservation spec"
        )
    repeat_plans = {receipt.repeat_plan_sha256 for _, receipt in sources}
    if len(repeat_plans) != 1:
        raise PreservationThresholdCalibrationError(
            "source receipts must share one repeat plan"
        )
    source_hashes = [receipt.self_sha256 for _, receipt in sources]
    if len(set(source_hashes)) != len(source_hashes):
        raise PreservationThresholdCalibrationError(
            "source receipt identities must be unique"
        )
    if any(receipt.edited_model_sha256 != base_model_sha256 for _, receipt in sources):
        raise PreservationThresholdCalibrationError(
            "repeat receipts must score the frozen base model"
        )
    source_rows = [
        {
            "tier": receipt.tier,
            "repeat_index": receipt.repeat_index,
            "collector_identity_sha256": receipt.collector_identity_sha256,
            "path": path.as_posix(),
            "receipt_sha256": receipt.self_sha256,
        }
        for path, receipt in sources
    ]
    source_rows.sort(key=lambda row: (row["tier"], row["repeat_index"]))
    tiers: list[dict[str, Any]] = []
    for tier in _TIERS:
        tier_receipts = [receipt for _, receipt in sources if receipt.tier == tier]
        if len(tier_receipts) < minimum_repeats:
            raise PreservationThresholdCalibrationError(
                f"calibration requires at least {minimum_repeats} repeats for {tier}"
            )
        indices = sorted(receipt.repeat_index for receipt in tier_receipts)
        if indices != list(range(len(tier_receipts))):
            raise PreservationThresholdCalibrationError(
                f"repeat indices for {tier} must be contiguous from zero"
            )
        if len({receipt.collector_identity_sha256 for receipt in tier_receipts}) != 1:
            raise PreservationThresholdCalibrationError(
                f"base repeats for {tier} must share one collector identity"
            )
        rows: list[dict[str, Any]] = []
        for stratum in _STRATA:
            values = [dict(receipt.strata)[stratum] for receipt in tier_receipts]
            observed_max = max(values)
            observed_quantile = _quantile(values, quantile)
            ceiling = max(
                observed_max + absolute_margin,
                observed_quantile * (1.0 + relative_margin),
            )
            rows.append(
                {
                    "stratum": stratum,
                    "observed_max": observed_max,
                    "observed_quantile": observed_quantile,
                    "ceiling": ceiling,
                }
            )
        tiers.append(
            {"tier": tier, "repeat_count": len(tier_receipts), "strata": rows}
        )
    return {
        "format": CALIBRATION_FORMAT,
        "calibration_id": calibration_id,
        "base_model_sha256": base_model_sha256,
        "preservation_spec_sha256": next(iter(spec_ids)),
        "repeat_plan_sha256": next(iter(repeat_plans)),
        "method": {
            "minimum_repeats": minimum_repeats,
            "quantile": quantile,
            "absolute_margin": absolute_margin,
            "relative_margin": relative_margin,
            "ceiling_rule": "max(observed_max+absolute_margin,observed_quantile*(1+relative_margin))",
        },
        "source_receipts": source_rows,
        "tiers": tiers,
    }


@dataclass(frozen=True)
class PreservationThresholdCalibration:
    payload: Mapping[str, Any]

    @property
    def self_sha256(self) -> str:
        return str(self.payload["self_sha256"])

    @property
    def source_receipt_count(self) -> int:
        return len(self.payload["source_receipts"])

    def thresholds_for(self, tier: str) -> dict[str, float]:
        for row in self.payload["tiers"]:
            if row["tier"] == tier:
                return {
                    item["stratum"]: float(item["ceiling"])
                    for item in row["strata"]
                }
        raise PreservationThresholdCalibrationError(f"unknown tier {tier!r}")

    def collector_identities(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in self.payload["source_receipts"]:
            tier = str(row["tier"])
            identity = str(row["collector_identity_sha256"])
            previous = result.setdefault(tier, identity)
            if previous != identity:
                raise PreservationThresholdCalibrationError(
                    f"base repeats for {tier} have conflicting collector identities"
                )
        if set(result) != set(_TIERS):
            raise PreservationThresholdCalibrationError(
                "calibration collector identities do not cover every tier"
            )
        return result

    @classmethod
    def open(cls, path: Path | str) -> "PreservationThresholdCalibration":
        artifact_path = Path(path)
        if artifact_path.is_symlink():
            raise PreservationThresholdCalibrationError(
                "preservation threshold calibration must be a regular file"
            )
        try:
            raw_value = json.loads(artifact_path.read_text())
        except FileNotFoundError as error:
            raise PreservationThresholdCalibrationError(
                "preservation threshold calibration is missing"
            ) from error
        except json.JSONDecodeError as error:
            raise PreservationThresholdCalibrationError(
                "preservation threshold calibration is invalid JSON"
            ) from error
        raw = _object(raw_value, "preservation threshold calibration")
        fields = {
            "format",
            "calibration_id",
            "base_model_sha256",
            "preservation_spec_sha256",
            "repeat_plan_sha256",
            "method",
            "source_receipts",
            "tiers",
            "self_sha256",
        }
        _exact(raw, fields, "preservation threshold calibration")
        if raw["format"] != CALIBRATION_FORMAT:
            raise PreservationThresholdCalibrationError(
                "unsupported preservation threshold calibration format"
            )
        claimed = _sha(raw["self_sha256"], "calibration.self_sha256")
        unsigned = dict(raw)
        del unsigned["self_sha256"]
        if _hash(unsigned) != claimed:
            raise PreservationThresholdCalibrationError(
                "preservation threshold calibration hash mismatch"
            )
        method = _object(raw["method"], "calibration.method")
        _exact(
            method,
            {
                "minimum_repeats",
                "quantile",
                "absolute_margin",
                "relative_margin",
                "ceiling_rule",
            },
            "calibration.method",
        )
        repeats, quantile, absolute, relative = _validate_method(
            method["minimum_repeats"],
            method["quantile"],
            method["absolute_margin"],
            method["relative_margin"],
        )
        expected_rule = (
            "max(observed_max+absolute_margin,"
            "observed_quantile*(1+relative_margin))"
        )
        if method["ceiling_rule"] != expected_rule:
            raise PreservationThresholdCalibrationError("unknown ceiling rule")
        source_values = _array(raw["source_receipts"], "source_receipts")
        sources: list[tuple[Path, _ParsedReceipt]] = []
        for index, value in enumerate(source_values):
            row = _object(value, f"source_receipts[{index}]")
            _exact(
                row,
                {
                    "tier",
                    "repeat_index",
                    "collector_identity_sha256",
                    "path",
                    "receipt_sha256",
                },
                f"source_receipts[{index}]",
            )
            relative_path = _safe_relative(
                Path(_text(row["path"], "source receipt path")),
                "source receipt path",
            )
            source_path = artifact_path.parent / relative_path
            if source_path.is_symlink() or not source_path.is_file():
                if not source_path.exists():
                    raise PreservationThresholdCalibrationError(
                        f"source receipt is missing: {relative_path}"
                    )
                raise PreservationThresholdCalibrationError(
                    f"source receipt must be regular: {relative_path}"
                )
            try:
                source_path.resolve(strict=True).relative_to(
                    artifact_path.parent.resolve(strict=True)
                )
            except (OSError, ValueError) as error:
                raise PreservationThresholdCalibrationError(
                    f"source receipt escapes calibration directory: {relative_path}"
                ) from error
            try:
                receipt = _parse_receipt(json.loads(source_path.read_text()))
            except FileNotFoundError as error:
                raise PreservationThresholdCalibrationError(
                    f"source receipt is missing: {relative_path}"
                ) from error
            except json.JSONDecodeError as error:
                raise PreservationThresholdCalibrationError(
                    f"source receipt is invalid JSON: {relative_path}"
                ) from error
            if receipt.self_sha256 != _sha(row["receipt_sha256"], "source receipt SHA"):
                raise PreservationThresholdCalibrationError(
                    "source receipt identity differs from calibration"
                )
            if receipt.tier != row["tier"]:
                raise PreservationThresholdCalibrationError(
                    "source receipt tier differs from calibration"
                )
            if receipt.repeat_index != row["repeat_index"]:
                raise PreservationThresholdCalibrationError(
                    "source receipt repeat index differs from calibration"
                )
            if receipt.collector_identity_sha256 != _sha(
                row["collector_identity_sha256"], "source collector identity"
            ):
                raise PreservationThresholdCalibrationError(
                    "source collector identity differs from calibration"
                )
            sources.append((relative_path, receipt))
        replayed = _derive_payload(
            calibration_id=_text(raw["calibration_id"], "calibration_id"),
            base_model_sha256=_sha(raw["base_model_sha256"], "base_model_sha256"),
            sources=sources,
            minimum_repeats=repeats,
            quantile=quantile,
            absolute_margin=absolute,
            relative_margin=relative,
        )
        if replayed != unsigned:
            raise PreservationThresholdCalibrationError(
                "calibration does not replay from bound source receipts"
            )
        return cls(payload=dict(raw))


def build_preservation_threshold_calibration(
    destination: Path | str,
    *,
    calibration_id: str,
    base_model_sha256: str,
    receipt_paths: Sequence[Path | str],
    minimum_repeats: int,
    quantile: float,
    absolute_margin: float,
    relative_margin: float,
) -> PreservationThresholdCalibration:
    """Build and atomically publish a replay-verifiable calibration artifact."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    repeats, q, absolute, relative_margin_value = _validate_method(
        minimum_repeats, quantile, absolute_margin, relative_margin
    )
    sources: list[tuple[Path, _ParsedReceipt]] = []
    for source in receipt_paths:
        source_path = Path(source).resolve()
        try:
            relative_path = source_path.relative_to(output.parent.resolve())
        except ValueError as error:
            raise PreservationThresholdCalibrationError(
                "source receipts must be below the calibration directory"
            ) from error
        _safe_relative(relative_path, "source receipt path")
        try:
            receipt = _parse_receipt(json.loads(source_path.read_text()))
        except FileNotFoundError as error:
            raise PreservationThresholdCalibrationError(
                f"source receipt is missing: {relative_path}"
            ) from error
        except json.JSONDecodeError as error:
            raise PreservationThresholdCalibrationError(
                f"source receipt is invalid JSON: {relative_path}"
            ) from error
        sources.append((relative_path, receipt))
    unsigned = _derive_payload(
        calibration_id=_text(calibration_id, "calibration_id"),
        base_model_sha256=_sha(base_model_sha256, "base_model_sha256"),
        sources=sources,
        minimum_repeats=repeats,
        quantile=q,
        absolute_margin=absolute,
        relative_margin=relative_margin_value,
    )
    payload = {**unsigned, "self_sha256": _hash(unsigned)}
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    if output.exists():
        if output.read_bytes() != encoded:
            raise PreservationThresholdCalibrationError(
                "refusing to replace differing calibration artifact"
            )
    else:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise PreservationThresholdCalibrationError(
                    "refusing to replace differing calibration artifact"
                )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return PreservationThresholdCalibration.open(output)


__all__ = [
    "CALIBRATION_FORMAT",
    "PreservationThresholdCalibration",
    "PreservationThresholdCalibrationError",
    "build_preservation_threshold_calibration",
]
