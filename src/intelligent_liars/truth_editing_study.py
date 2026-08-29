"""Resumable optimizer-neutral study orchestration for persistent truth edits.

``TruthEditingStudy.run`` is the public seam.  Search and evaluation adapters
vary behind it; the durable journal, synchronous batch rule, hierarchical
search-space validation, tiered validation prefixes, and failure taxonomy do
not.  This module never loads a model and never exposes final-test records.

Routine search admits only qualified truth directions and persistent-weight
semantic drafts. Its deliberate broad stage also executes a matched persistent
orthogonal-basis control before concentration, without teaching TPE that the
control outcome came from its parent truth edit. Shuffled, activation-
restoration, re-ablation, false-trigger, and related controls remain finalist/
post-freeze evidence lanes. Their execution plans must bind a verified
``CompiledControlBasisReceipt.self_sha256`` to the parent compiled basis set;
an unbound control condition is never representable through this interface.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .heretic_truth_editing import OBJECTIVES
from .truth_editing_batch_execution import (
    BatchEvaluationRequest,
    BatchExecutionError,
    execute_ordered_batch,
)
from .truth_editing_contracts import DirectionBankManifest, DirectionEntry
from .truth_editing_broad_coverage import (
    BroadCoverageContract,
    KERNEL_CENTER_REGIONS,
    KERNEL_SHAPES,
    REFUSAL_SETTINGS,
    REFUSAL_STRENGTH_REGIONS,
    REFUSAL_WRITER_POLICIES,
    WRITER_CONFIGURATIONS,
    active_edit_arm,
    kernel_center_region,
    kernel_half_width_mode,
    kernel_shape,
    refusal_setting,
    refusal_strength_region,
    writer_configuration,
)


STUDY_CONFIG_FORMAT = "truth_editing_study_config_v1"
ADAPTIVE_STUDY_CONFIG_FORMAT = "truth_editing_study_config_v2"
ADAPTIVE_SEARCH_POLICY_FORMAT = "truth_editing_adaptive_search_policy_v1"
STUDY_JOURNAL_FORMAT = "truth_editing_study_journal_v1"
STUDY_REPORT_FORMAT = "truth_editing_study_report_v1"
# Stable semantic identity of the production orchestrator lineage first launched
# on 2026-08-29. Failure replay changes recovery behavior without changing the
# frozen search space, evaluator, or scientific meaning of a successful trial.
# Pinning this value preserves the checkpoint-48 lineage instead of coupling
# resumability to unrelated source-file byte changes.
STUDY_ORCHESTRATOR_SEMANTICS_SHA256 = (
    "5e6392c5975d04006e5a1ef32ba93896750a7ec8916ff85c23fcb4ec94a960b1"
)
_HEX = frozenset("0123456789abcdef")
_WRITER_POLICIES = ("attention", "mlp", "both")
_BASIS_METHODS = ("qr", "svd")
_STRENGTH_REGIONS = ("disabled", "projection", "reflection")
_NORMALIZATION_MODES = ("exact", "norm_preserving")
_DIRECTION_SCOPES = ("global", "per_layer")
_EDIT_ARMS = ("truth_only", "refusal_only", "joint")
_MATCHED_BASIS_CONTROLS = ("none", "orthogonal")
_REFUSAL_WRITER_POLICIES = ("attention", "mlp", "both")
_PROPOSAL_ORIGINS = ("coverage_anchor", "tpe_sampled")


class StudyError(ValueError):
    """A study cannot run without violating its frozen contract."""


class OperationalEvaluationError(RuntimeError):
    """A worker/runtime failure that must not become a scientific outcome."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise StudyError("study value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    observed = set(value)
    if observed != fields:
        raise StudyError(
            f"{name} fields differ; missing={sorted(fields - observed)}, "
            f"extra={sorted(observed - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StudyError(f"{name} must be a nonempty trimmed string")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StudyError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise StudyError(f"{name} must be a finite number")
    return result


def _hash(value: Any, name: str) -> str:
    value = _text(value, name)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise StudyError(f"{name} must be a lowercase SHA-256")
    return value


def _strength_region(value: float) -> str:
    if value == 0.0:
        return "disabled"
    if value <= 1.0:
        return "projection"
    return "reflection"


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StudyError(f"{name} must be an array")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if not result or len(set(result)) != len(result):
        raise StudyError(f"{name} must be a nonempty array of unique strings")
    return result


def _optional_strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise StudyError(f"{name} must be an array")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise StudyError(f"{name} must contain unique strings")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise StudyError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class WriterRegion:
    name: str
    layers: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "layers": list(self.layers)}


@dataclass(frozen=True)
class EvaluationTier:
    name: str
    record_limit: int
    through_trial: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BroadCoveragePolicy:
    required_before_concentration: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveSearchPolicy:
    minimum_trials: int
    maximum_trials: int
    search_elapsed_limit_seconds: int
    reserve_elapsed_seconds: int
    all_in_budget_usd: str
    maximum_infrastructure_spend_usd: str
    maximum_evaluation_spend_usd: str
    evaluation_budget_reserve_fraction: str
    evaluation_spend_reserve_usd: str
    broad_coverage: BroadCoveragePolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ADAPTIVE_SEARCH_POLICY_FORMAT,
            "minimum_trials": self.minimum_trials,
            "maximum_trials": self.maximum_trials,
            "search_elapsed_limit_seconds": self.search_elapsed_limit_seconds,
            "reserve_elapsed_seconds": self.reserve_elapsed_seconds,
            "all_in_budget_usd": self.all_in_budget_usd,
            "maximum_infrastructure_spend_usd": (
                self.maximum_infrastructure_spend_usd
            ),
            "maximum_evaluation_spend_usd": self.maximum_evaluation_spend_usd,
            "evaluation_budget_reserve_fraction": (
                self.evaluation_budget_reserve_fraction
            ),
            "evaluation_spend_reserve_usd": self.evaluation_spend_reserve_usd,
            "broad_coverage": self.broad_coverage.to_dict(),
        }


@dataclass(frozen=True)
class TruthEditingStudyConfig:
    format: str
    study_id: str
    sampler_seed: int
    batch_size: int
    max_trials: int
    max_directions_per_trial: int
    max_rank: int
    strength_min: float
    strength_max: float
    writer_regions: tuple[WriterRegion, ...]
    evaluation_tiers: tuple[EvaluationTier, ...]
    dataset_manifest_sha256: str
    validation_record_ids: tuple[str, ...]
    objective_names: tuple[str, ...]
    tpe_startup_trials: int
    tpe_ei_candidates: int
    tpe_multivariate: bool
    search_policy: AdaptiveSearchPolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "format": self.format,
            "study_id": self.study_id,
            "sampler_seed": self.sampler_seed,
            "batch_size": self.batch_size,
            "max_trials": self.max_trials,
            "max_directions_per_trial": self.max_directions_per_trial,
            "max_rank": self.max_rank,
            "strength_min": self.strength_min,
            "strength_max": self.strength_max,
            "writer_regions": [item.to_dict() for item in self.writer_regions],
            "evaluation_tiers": [item.to_dict() for item in self.evaluation_tiers],
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "validation_record_ids": list(self.validation_record_ids),
            "objective_names": list(self.objective_names),
            "tpe_startup_trials": self.tpe_startup_trials,
            "tpe_ei_candidates": self.tpe_ei_candidates,
            "tpe_multivariate": self.tpe_multivariate,
        }
        if self.search_policy is not None:
            result["search_policy"] = self.search_policy.to_dict()
        return result

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_dict())


def parse_truth_editing_study_config(value: Mapping[str, Any]) -> TruthEditingStudyConfig:
    if not isinstance(value, Mapping):
        raise StudyError("study config must be an object")
    common_fields = {
        "format", "study_id", "sampler_seed", "batch_size", "max_trials",
        "max_directions_per_trial", "max_rank", "strength_min", "strength_max",
        "writer_regions", "evaluation_tiers", "dataset_manifest_sha256",
        "validation_record_ids", "objective_names",
        "tpe_startup_trials", "tpe_ei_candidates", "tpe_multivariate",
    }
    config_format = value.get("format")
    if config_format == STUDY_CONFIG_FORMAT:
        fields = common_fields
    elif config_format == ADAPTIVE_STUDY_CONFIG_FORMAT:
        fields = common_fields | {"search_policy"}
    else:
        raise StudyError("study config format is unsupported")
    _exact(value, fields, "study config")
    regions_raw = value["writer_regions"]
    if isinstance(regions_raw, (str, bytes)) or not isinstance(regions_raw, Sequence):
        raise StudyError("writer_regions must be an array")
    regions: list[WriterRegion] = []
    for index, raw in enumerate(regions_raw):
        if not isinstance(raw, Mapping):
            raise StudyError(f"writer_regions[{index}] must be an object")
        _exact(raw, {"name", "layers"}, f"writer_regions[{index}]")
        layers_raw = raw["layers"]
        if isinstance(layers_raw, (str, bytes)) or not isinstance(layers_raw, Sequence):
            raise StudyError(f"writer_regions[{index}].layers must be an array")
        layers = tuple(_integer(item, f"writer_regions[{index}].layers", 0) for item in layers_raw)
        if not layers or len(set(layers)) != len(layers):
            raise StudyError(f"writer_regions[{index}].layers must be nonempty and unique")
        regions.append(WriterRegion(_text(raw["name"], "writer region name"), layers))
    if not regions or len({item.name for item in regions}) != len(regions):
        raise StudyError("writer region names must be nonempty and unique")

    tiers_raw = value["evaluation_tiers"]
    if isinstance(tiers_raw, (str, bytes)) or not isinstance(tiers_raw, Sequence):
        raise StudyError("evaluation_tiers must be an array")
    tiers: list[EvaluationTier] = []
    for index, raw in enumerate(tiers_raw):
        if not isinstance(raw, Mapping):
            raise StudyError(f"evaluation_tiers[{index}] must be an object")
        _exact(raw, {"name", "record_limit", "through_trial"}, f"evaluation_tiers[{index}]")
        tiers.append(EvaluationTier(
            _text(raw["name"], "evaluation tier name"),
            _integer(raw["record_limit"], "evaluation tier record_limit"),
            _integer(raw["through_trial"], "evaluation tier through_trial"),
        ))
    max_trials = _integer(value["max_trials"], "max_trials")
    if not tiers or tuple(item.through_trial for item in tiers) != tuple(
        sorted(item.through_trial for item in tiers)
    ) or tiers[-1].through_trial != max_trials:
        raise StudyError("evaluation tiers must be ordered and end at max_trials")
    validation_ids = _strings(value["validation_record_ids"], "validation_record_ids")
    if any(item.record_limit > len(validation_ids) for item in tiers):
        raise StudyError("evaluation tier record_limit exceeds validation dataset")
    objectives = _strings(value["objective_names"], "objective_names")
    if objectives != OBJECTIVES:
        raise StudyError("objective_names must be the frozen ordered semantic objectives")
    strength_min = _number(value["strength_min"], "strength_min")
    strength_max = _number(value["strength_max"], "strength_max")
    if strength_min != 0.0 or strength_max != 2.0:
        raise StudyError("the routine search strength range must be exactly [0, 2]")
    seed = value["sampler_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StudyError("sampler_seed must be an integer")
    startup_trials = _integer(value["tpe_startup_trials"], "tpe_startup_trials", 0)
    if startup_trials > max_trials:
        raise StudyError("tpe_startup_trials cannot exceed max_trials")
    ei_candidates = _integer(value["tpe_ei_candidates"], "tpe_ei_candidates")
    if ei_candidates != 128:
        raise StudyError("tpe_ei_candidates must preserve the documented value 128")
    if value["tpe_multivariate"] is not True:
        raise StudyError("tpe_multivariate must be true")
    search_policy: AdaptiveSearchPolicy | None = None
    if config_format == ADAPTIVE_STUDY_CONFIG_FORMAT:
        raw_policy = value["search_policy"]
        if not isinstance(raw_policy, Mapping):
            raise StudyError("search_policy must be an object")
        _exact(
            raw_policy,
            {
                "format",
                "minimum_trials",
                "maximum_trials",
                "search_elapsed_limit_seconds",
                "reserve_elapsed_seconds",
                "all_in_budget_usd",
                "maximum_infrastructure_spend_usd",
                "maximum_evaluation_spend_usd",
                "evaluation_budget_reserve_fraction",
                "evaluation_spend_reserve_usd",
                "broad_coverage",
            },
            "search_policy",
        )
        if raw_policy["format"] != ADAPTIVE_SEARCH_POLICY_FORMAT:
            raise StudyError("search_policy format is unsupported")
        broad = raw_policy["broad_coverage"]
        if not isinstance(broad, Mapping):
            raise StudyError("broad coverage must be an object")
        _exact(
            broad,
            {"required_before_concentration"},
            "broad coverage",
        )
        minimum_trials = _integer(
            raw_policy["minimum_trials"], "search_policy.minimum_trials"
        )
        maximum_trials = _integer(
            raw_policy["maximum_trials"], "search_policy.maximum_trials"
        )
        if minimum_trials != 200:
            raise StudyError("search_policy.minimum_trials must be exactly 200")
        if maximum_trials != 800 or max_trials != maximum_trials:
            raise StudyError(
                "search_policy.maximum_trials and max_trials must be exactly 800"
            )
        if _integer(value["batch_size"], "batch_size") != 8:
            raise StudyError("adaptive production batch_size must be exactly 8")
        if (
            _integer(
                raw_policy["search_elapsed_limit_seconds"],
                "search_policy.search_elapsed_limit_seconds",
            )
            != 21 * 3600
        ):
            raise StudyError("adaptive search time must be exactly 21 hours")
        if (
            _integer(
                raw_policy["reserve_elapsed_seconds"],
                "search_policy.reserve_elapsed_seconds",
            )
            != 3 * 3600
        ):
            raise StudyError("adaptive finalization reserve must be exactly 3 hours")
        if raw_policy["all_in_budget_usd"] != "50":
            raise StudyError("adaptive all-in budget must be exactly 50 USD")
        if raw_policy["maximum_infrastructure_spend_usd"] != "45":
            raise StudyError("adaptive infrastructure budget must be exactly 45 USD")
        if raw_policy["maximum_evaluation_spend_usd"] != "5":
            raise StudyError("adaptive evaluation budget must be exactly 5 USD")
        if raw_policy["evaluation_budget_reserve_fraction"] != "0.20":
            raise StudyError("adaptive evaluation budget reserve fraction must be 0.20")
        if raw_policy["evaluation_spend_reserve_usd"] != "1":
            raise StudyError("adaptive evaluation reserve must be exactly 1 USD")
        if broad.get("required_before_concentration") is not True:
            raise StudyError("broad coverage must be required before concentration")
        if tuple(item.through_trial for item in tiers) != (80, 200, 800):
            raise StudyError("adaptive evaluation tiers must end at 80/200/800")
        search_policy = AdaptiveSearchPolicy(
            minimum_trials=minimum_trials,
            maximum_trials=maximum_trials,
            search_elapsed_limit_seconds=21 * 3600,
            reserve_elapsed_seconds=3 * 3600,
            all_in_budget_usd="50",
            maximum_infrastructure_spend_usd="45",
            maximum_evaluation_spend_usd="5",
            evaluation_budget_reserve_fraction="0.20",
            evaluation_spend_reserve_usd="1",
            broad_coverage=BroadCoveragePolicy(
                required_before_concentration=True,
            ),
        )
    return TruthEditingStudyConfig(
        format=str(config_format),
        study_id=_text(value["study_id"], "study_id"), sampler_seed=seed,
        batch_size=_integer(value["batch_size"], "batch_size"), max_trials=max_trials,
        max_directions_per_trial=_integer(value["max_directions_per_trial"], "max_directions_per_trial"),
        max_rank=_integer(value["max_rank"], "max_rank"), strength_min=strength_min,
        strength_max=strength_max, writer_regions=tuple(regions), evaluation_tiers=tuple(tiers),
        dataset_manifest_sha256=_hash(value["dataset_manifest_sha256"], "dataset_manifest_sha256"),
        validation_record_ids=validation_ids, objective_names=objectives,
        tpe_startup_trials=startup_trials, tpe_ei_candidates=ei_candidates,
        tpe_multivariate=True,
        search_policy=search_policy,
    )


def load_truth_editing_study_config(path: Path | str) -> TruthEditingStudyConfig:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise StudyError(f"study config is not a regular file: {path}")
    try:
        raw = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StudyError("study config is unreadable") from error
    return parse_truth_editing_study_config(raw)


@dataclass(frozen=True)
class CoverageLedger:
    families: tuple[str, ...] = ()
    source_layers: tuple[int, ...] = ()
    writer_regions: tuple[str, ...] = ()
    writer_policies: tuple[str, ...] = ()
    basis_methods: tuple[str, ...] = ()
    strength_regions: tuple[str, ...] = ()
    basis_scopes: tuple[str, ...] = ()
    direction_scopes: tuple[str, ...] = ()
    normalization_modes: tuple[str, ...] = ()
    edit_arms: tuple[str, ...] = ()
    active_edit_arms: tuple[str, ...] = ()
    writer_configurations: tuple[str, ...] = ()
    refusal_settings: tuple[str, ...] = ()
    refusal_writer_policies: tuple[str, ...] = ()
    kernel_center_regions: tuple[str, ...] = ()
    kernel_half_width_modes: tuple[str, ...] = ()
    kernel_shapes: tuple[str, ...] = ()
    refusal_strength_regions: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchProposal:
    """Semantic recipe draft; execution must bind a compiled basis-set receipt."""

    direction_ids: tuple[str, ...]
    direction_family: str
    source_layer: int
    basis_method: Literal["qr", "svd"]
    requested_rank: int
    writer_region: str
    writer_layers: tuple[int, ...]
    writer_policy: Literal["attention", "mlp", "both"]
    strength: float
    backend_type: Literal["persistent_weight"] = "persistent_weight"
    basis_scope: Literal["general", "domain", "mixed"] | None = None
    selected_domains: tuple[str, ...] = ()
    truth_direction_scope: Literal["global", "per_layer"] = "global"
    normalization_mode: Literal["exact", "norm_preserving"] = "exact"
    edit_arm: Literal["truth_only", "refusal_only", "joint"] = "truth_only"
    attention_enabled: bool | None = None
    attention_kernel_center: float | None = None
    attention_kernel_half_width: float | None = None
    attention_edge_strength: float | None = None
    attention_peak_strength: float | None = None
    mlp_enabled: bool | None = None
    mlp_kernel_center: float | None = None
    mlp_kernel_half_width: float | None = None
    mlp_edge_strength: float | None = None
    mlp_peak_strength: float | None = None
    refusal_enabled: bool = False
    refusal_direction_scope: Literal["global", "per_layer"] = "global"
    refusal_source_layer: int | None = None
    refusal_strength: float = 0.0
    refusal_writer_policy: Literal["attention", "mlp", "both"] = "both"
    matched_basis_control: Literal["none", "orthogonal"] = "none"
    proposal_origin: Literal["coverage_anchor", "tpe_sampled"] = "coverage_anchor"

    def __post_init__(self) -> None:
        """Canonicalize the legacy coarse writer fields into explicit kernels.

        Existing callers can still construct ``writer_policy``/``strength``
        proposals. New search drivers always emit every explicit field. This is
        a compatibility bridge at construction, not a permissive JSON parser.
        """

        if not self.writer_layers:
            return
        center = float(self.writer_layers[len(self.writer_layers) // 2])
        half_width = float(max(abs(layer - center) for layer in self.writer_layers))
        attention_enabled = self.writer_policy in {"attention", "both"}
        mlp_enabled = self.writer_policy in {"mlp", "both"}
        defaults: dict[str, Any] = {
            "attention_enabled": attention_enabled,
            "attention_kernel_center": center,
            "attention_kernel_half_width": half_width,
            "attention_edge_strength": self.strength,
            "attention_peak_strength": self.strength,
            "mlp_enabled": mlp_enabled,
            "mlp_kernel_center": center,
            "mlp_kernel_half_width": half_width,
            "mlp_edge_strength": self.strength,
            "mlp_peak_strength": self.strength,
        }
        for name, fallback in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, fallback)
        if self.basis_scope is None:
            object.__setattr__(
                self, "basis_scope",
                "general" if self.direction_family == "general"
                else "mixed" if self.direction_family == "mixed" else "domain",
            )

    @staticmethod
    def _kernel_strength(
        layer: int, *, enabled: bool, center: float, half_width: float,
        edge: float, peak: float,
    ) -> float:
        if not enabled:
            return 0.0
        distance = abs(layer - center)
        if half_width == 0.0:
            return peak if distance == 0.0 else 0.0
        if distance > half_width:
            return 0.0
        return peak + distance / half_width * (edge - peak)

    def writer_strength_plan(self) -> dict[str, dict[int, float]]:
        """Compile explicit triangular kernels; callers never infer semantics."""

        assert self.attention_enabled is not None
        assert self.attention_kernel_center is not None
        assert self.attention_kernel_half_width is not None
        assert self.attention_edge_strength is not None
        assert self.attention_peak_strength is not None
        assert self.mlp_enabled is not None
        assert self.mlp_kernel_center is not None
        assert self.mlp_kernel_half_width is not None
        assert self.mlp_edge_strength is not None
        assert self.mlp_peak_strength is not None
        return {
            "attention_by_layer": {
                layer: self._kernel_strength(
                    layer, enabled=self.attention_enabled,
                    center=self.attention_kernel_center,
                    half_width=self.attention_kernel_half_width,
                    edge=self.attention_edge_strength,
                    peak=self.attention_peak_strength,
                )
                for layer in self.writer_layers
            },
            "mlp_by_layer": {
                layer: self._kernel_strength(
                    layer, enabled=self.mlp_enabled,
                    center=self.mlp_kernel_center,
                    half_width=self.mlp_kernel_half_width,
                    edge=self.mlp_edge_strength,
                    peak=self.mlp_peak_strength,
                )
                for layer in self.writer_layers
            },
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["direction_ids"] = list(self.direction_ids)
        result["writer_layers"] = list(self.writer_layers)
        result["selected_domains"] = list(self.selected_domains)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SearchProposal":
        _exact(value, {
            "direction_ids", "direction_family", "source_layer", "basis_method",
            "requested_rank", "writer_region", "writer_layers", "writer_policy",
            "strength", "backend_type", "basis_scope", "selected_domains",
            "truth_direction_scope", "normalization_mode", "edit_arm",
            "attention_enabled", "attention_kernel_center",
            "attention_kernel_half_width", "attention_edge_strength",
            "attention_peak_strength", "mlp_enabled", "mlp_kernel_center",
            "mlp_kernel_half_width", "mlp_edge_strength", "mlp_peak_strength",
            "refusal_enabled", "refusal_direction_scope", "refusal_source_layer",
            "refusal_strength", "refusal_writer_policy",
            "matched_basis_control",
            "proposal_origin",
        }, "proposal")
        if value["backend_type"] != "persistent_weight":
            raise StudyError("routine optimization permits persistent_weight recipes only")
        if value["basis_method"] not in _BASIS_METHODS or value["writer_policy"] not in _WRITER_POLICIES:
            raise StudyError("proposal search category is invalid")
        if value["basis_scope"] not in {"general", "domain", "mixed"}:
            raise StudyError("proposal basis_scope is invalid")
        if value["truth_direction_scope"] not in _DIRECTION_SCOPES:
            raise StudyError("proposal truth_direction_scope is invalid")
        if value["normalization_mode"] not in _NORMALIZATION_MODES:
            raise StudyError("proposal normalization_mode is invalid")
        if value["edit_arm"] not in _EDIT_ARMS:
            raise StudyError("proposal edit_arm is invalid")
        if value["refusal_direction_scope"] not in _DIRECTION_SCOPES:
            raise StudyError("proposal refusal_direction_scope is invalid")
        if value["refusal_writer_policy"] not in _REFUSAL_WRITER_POLICIES:
            raise StudyError("proposal refusal_writer_policy is invalid")
        if value["matched_basis_control"] not in _MATCHED_BASIS_CONTROLS:
            raise StudyError("proposal matched_basis_control is invalid")
        if value["proposal_origin"] not in _PROPOSAL_ORIGINS:
            raise StudyError("proposal proposal_origin is invalid")
        return cls(
            direction_ids=_strings(value["direction_ids"], "proposal.direction_ids"),
            direction_family=_text(value["direction_family"], "proposal.direction_family"),
            source_layer=_integer(value["source_layer"], "proposal.source_layer", 0),
            basis_method=value["basis_method"], requested_rank=_integer(value["requested_rank"], "requested_rank"),
            writer_region=_text(value["writer_region"], "writer_region"),
            writer_layers=tuple(_integer(item, "writer_layer", 0) for item in value["writer_layers"]),
            writer_policy=value["writer_policy"], strength=_number(value["strength"], "strength"),
            basis_scope=value["basis_scope"],
            selected_domains=_optional_strings(value["selected_domains"], "selected_domains"),
            truth_direction_scope=value["truth_direction_scope"],
            normalization_mode=value["normalization_mode"], edit_arm=value["edit_arm"],
            attention_enabled=_boolean(value["attention_enabled"], "attention_enabled"),
            attention_kernel_center=_number(value["attention_kernel_center"], "attention_kernel_center"),
            attention_kernel_half_width=_number(value["attention_kernel_half_width"], "attention_kernel_half_width"),
            attention_edge_strength=_number(value["attention_edge_strength"], "attention_edge_strength"),
            attention_peak_strength=_number(value["attention_peak_strength"], "attention_peak_strength"),
            mlp_enabled=_boolean(value["mlp_enabled"], "mlp_enabled"),
            mlp_kernel_center=_number(value["mlp_kernel_center"], "mlp_kernel_center"),
            mlp_kernel_half_width=_number(value["mlp_kernel_half_width"], "mlp_kernel_half_width"),
            mlp_edge_strength=_number(value["mlp_edge_strength"], "mlp_edge_strength"),
            mlp_peak_strength=_number(value["mlp_peak_strength"], "mlp_peak_strength"),
            refusal_enabled=_boolean(value["refusal_enabled"], "refusal_enabled"),
            refusal_direction_scope=value["refusal_direction_scope"],
            refusal_source_layer=(
                None if value["refusal_source_layer"] is None
                else _integer(value["refusal_source_layer"], "refusal_source_layer", 0)
            ),
            refusal_strength=_number(value["refusal_strength"], "refusal_strength"),
            refusal_writer_policy=value["refusal_writer_policy"],
            matched_basis_control=value["matched_basis_control"],
            proposal_origin=value["proposal_origin"],
        )


@dataclass(frozen=True)
class FinalistControlRequest:
    """A matched persistent-basis control; never an activation intervention."""

    parent_proposal_sha256: str
    control_kind: Literal["orthogonal", "shuffled"]
    direction_ids: tuple[str, ...]
    source_layer: int
    requested_rank: int
    writer_layers: tuple[int, ...]
    writer_strength_plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_proposal_sha256": self.parent_proposal_sha256,
            "control_kind": self.control_kind,
            "direction_ids": list(self.direction_ids),
            "source_layer": self.source_layer,
            "requested_rank": self.requested_rank,
            "writer_layers": list(self.writer_layers),
            "writer_strength_plan_sha256": self.writer_strength_plan_sha256,
        }


def _active_study_arm(proposal: SearchProposal) -> str | None:
    """Classify an effective persistent study arm, including matched controls."""

    base_arm = active_edit_arm(proposal)
    if proposal.matched_basis_control == "orthogonal":
        return "orthogonal_control" if base_arm == "truth_only" else None
    return base_arm


def _coverage_ledger(proposals: Sequence[SearchProposal]) -> CoverageLedger:
    """Build the exact broad-stage ledger from accepted proposals."""

    center_regions = {
        value for item in proposals
        if (value := kernel_center_region(item)) is not None
    }
    half_width_modes = {
        value for item in proposals
        if (value := kernel_half_width_mode(item)) is not None
    }
    shapes = {
        value for item in proposals if (value := kernel_shape(item)) is not None
    }
    return CoverageLedger(
        families=tuple(sorted({item.direction_family for item in proposals})),
        source_layers=tuple(sorted({item.source_layer for item in proposals})),
        writer_regions=tuple(sorted({item.writer_region for item in proposals})),
        writer_policies=tuple(sorted({item.writer_policy for item in proposals})),
        basis_methods=tuple(sorted({item.basis_method for item in proposals})),
        strength_regions=tuple(sorted({_strength_region(item.strength) for item in proposals})),
        basis_scopes=tuple(sorted({cast(str, item.basis_scope) for item in proposals})),
        direction_scopes=tuple(sorted({item.truth_direction_scope for item in proposals})),
        normalization_modes=tuple(sorted({item.normalization_mode for item in proposals})),
        edit_arms=tuple(sorted({item.edit_arm for item in proposals})),
        active_edit_arms=tuple(sorted({
            value for item in proposals
            if (value := _active_study_arm(item)) is not None
        })),
        writer_configurations=tuple(sorted({writer_configuration(item) for item in proposals})),
        refusal_settings=tuple(sorted({refusal_setting(item) for item in proposals})),
        refusal_writer_policies=tuple(sorted({
            item.refusal_writer_policy for item in proposals
            if refusal_setting(item) != "disabled"
        })),
        kernel_center_regions=tuple(sorted(center_regions)),
        kernel_half_width_modes=tuple(sorted(half_width_modes)),
        kernel_shapes=tuple(sorted(shapes)),
        refusal_strength_regions=tuple(sorted({
            refusal_strength_region(item) for item in proposals
        })),
    )


def schedule_finalist_basis_controls(
    proposals: Sequence[SearchProposal],
) -> tuple[FinalistControlRequest, ...]:
    """Schedule equal-rank/layer/strength orthogonal and shuffled controls.

    These requests intentionally carry no activation transform or token scope.
    Activation restoration/re-ablation controls remain in the separate causal
    evidence lane and cannot enter routine Optuna through this API.
    """

    requests: list[FinalistControlRequest] = []
    for proposal in proposals:
        if proposal.backend_type != "persistent_weight":
            raise StudyError("finalist controls require a persistent parent proposal")
        parent_sha256 = _sha(proposal.to_dict())
        strength_sha256 = _sha(proposal.writer_strength_plan())
        for kind in ("orthogonal", "shuffled"):
            requests.append(FinalistControlRequest(
                parent_sha256,
                cast(Literal["orthogonal", "shuffled"], kind),
                proposal.direction_ids,
                proposal.source_layer,
                proposal.requested_rank,
                proposal.writer_layers,
                strength_sha256,
            ))
    return tuple(requests)


@dataclass(frozen=True)
class SearchRequest:
    ordinal: int
    config: TruthEditingStudyConfig
    directions: tuple[DirectionEntry, ...]
    coverage: CoverageLedger


def _scope_candidates(
    directions: Sequence[DirectionEntry], source_layer: int, basis_scope: str,
    maximum: int,
) -> tuple[DirectionEntry, ...]:
    """Return a deterministic executable pool for one sparse semantic scope."""

    layer_items = tuple(
        item for item in directions if item.source_layer == source_layer
    )
    general = tuple(item for item in layer_items if item.family == "general")
    domain = tuple(item for item in layer_items if item.family != "general")
    if basis_scope == "general":
        return general
    if basis_scope == "domain":
        return domain
    if basis_scope == "mixed" and general and domain and maximum >= 2:
        return general + domain
    return ()


def _required_basis_scopes(
    directions: Sequence[DirectionEntry], maximum: int,
) -> set[str]:
    return {
        scope
        for layer in {item.source_layer for item in directions}
        for scope in ("general", "domain", "mixed")
        if _scope_candidates(directions, layer, scope, maximum)
    }


def _broad_coverage_contract(
    config: TruthEditingStudyConfig,
    directions: Sequence[DirectionEntry],
) -> BroadCoverageContract:
    families = {item.family for item in directions}
    scopes = _required_basis_scopes(directions, config.max_directions_per_trial)
    if "mixed" in scopes:
        families.add("mixed")
    return BroadCoverageContract(
        families=frozenset(families),
        source_layers=frozenset(item.source_layer for item in directions),
        writer_regions=frozenset(item.name for item in config.writer_regions),
        writer_policies=frozenset(_WRITER_POLICIES),
        basis_methods=frozenset(_BASIS_METHODS),
        strength_regions=frozenset(_STRENGTH_REGIONS),
        basis_scopes=frozenset(scopes),
        direction_scopes=frozenset(_DIRECTION_SCOPES),
        normalization_modes=frozenset(_NORMALIZATION_MODES),
        edit_arms=frozenset(_EDIT_ARMS),
        active_edit_arms=frozenset(
            ("truth_only", "refusal_only", "joint", "orthogonal_control")
        ),
    )


class SearchDriver(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...
    def prepare(
        self, config: TruthEditingStudyConfig, directions: tuple[DirectionEntry, ...],
        state_path: Path,
    ) -> None: ...
    def suggest(self, request: SearchRequest) -> SearchProposal: ...
    def observe(self, trials: Sequence["StudyTrial"]) -> None: ...
    def complete_history_replay(self) -> None: ...


class BatchAdmission(Protocol):
    """Operational gate consulted before a complete batch may dispatch."""

    def admit_batch(
        self, *, completed_trials: int, batch_size: int, coverage_complete: bool,
        batch_started: bool = False,
    ) -> bool: ...

    def commit_batch(
        self, *, completed_trials: int, coverage_complete: bool
    ) -> None: ...


@dataclass(frozen=True)
class CompletedBatchCommit:
    """Integrity-bound observation delivered only at a complete batch barrier.

    The study journal is authoritative.  This record gives an adaptive
    controller enough information to persist capacity observations, advance
    its scheduler, and publish a checkpoint without teaching the study about
    any of those adapters.
    """

    study_identity_sha256: str
    batch_ordinal: int
    batch_size: int
    completed_trials: int
    journal_sha256: str
    batch_sha256: str
    trials: tuple["StudyTrial", ...]
    coverage: "CoverageLedger"
    coverage_complete: bool
    _coverage_counts: tuple[tuple[str, int, int], ...]

    @property
    def coverage_summary(self) -> dict[str, tuple[int, int]]:
        """Six stable ``completed, required`` counters for live monitoring."""

        return {
            name: (completed, required)
            for name, completed, required in self._coverage_counts
        }

    def _body(self) -> dict[str, Any]:
        return {
            "study_identity_sha256": self.study_identity_sha256,
            "batch_ordinal": self.batch_ordinal,
            "batch_size": self.batch_size,
            "completed_trials": self.completed_trials,
            "journal_sha256": self.journal_sha256,
            "batch_sha256": self.batch_sha256,
            "trials": [item.to_dict() for item in self.trials],
            "coverage": asdict(self.coverage),
            "coverage_complete": self.coverage_complete,
            "coverage_summary": {
                name: [completed, required]
                for name, completed, required in self._coverage_counts
            },
        }

    @property
    def commit_sha256(self) -> str:
        return _sha(self._body())

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        return {**body, "commit_sha256": _sha(body)}


AfterCompleteBatch = Callable[[CompletedBatchCommit], None]


@dataclass(frozen=True)
class PreparedStudyContext:
    """Durable study state exposed before the first admission of a run call."""

    study_identity_sha256: str
    journal_sha256: str
    completed_trials: int
    coverage: "CoverageLedger"
    coverage_complete: bool
    _coverage_counts: tuple[tuple[str, int, int], ...]

    @property
    def coverage_summary(self) -> dict[str, tuple[int, int]]:
        return {
            name: (completed, required)
            for name, completed, required in self._coverage_counts
        }

    def _body(self) -> dict[str, Any]:
        return {
            "study_identity_sha256": self.study_identity_sha256,
            "journal_sha256": self.journal_sha256,
            "completed_trials": self.completed_trials,
            "coverage": asdict(self.coverage),
            "coverage_complete": self.coverage_complete,
            "coverage_summary": {
                name: [completed, required]
                for name, completed, required in self._coverage_counts
            },
        }

    @property
    def context_sha256(self) -> str:
        return _sha(self._body())

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        return {**body, "context_sha256": _sha(body)}


AfterPrepareBeforeFirstAdmission = Callable[[PreparedStudyContext], None]


def _complete_driver_history_replay(driver: SearchDriver) -> None:
    """Notify replay-aware drivers while preserving simple custom adapters."""

    complete = getattr(driver, "complete_history_replay", None)
    if callable(complete):
        complete()


class OfflineDeterministicSearchDriver:
    """Dependency-free deterministic adapter used for replay and tests."""

    def __init__(self, *, seed: int) -> None:
        self.seed = seed
        self.observed_batch_sizes: list[int] = []
        self._reserved: list[SearchProposal] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"adapter": "offline_deterministic_v1", "seed": self.seed}

    def prepare(
        self, config: TruthEditingStudyConfig, directions: tuple[DirectionEntry, ...],
        state_path: Path,
    ) -> None:
        del config, directions, state_path

    def suggest(self, request: SearchRequest) -> SearchProposal:
        config = request.config
        directions = request.directions
        families = tuple(dict.fromkeys(item.family for item in directions))
        layers = tuple(dict.fromkeys(item.source_layer for item in directions))
        regions = tuple(item.name for item in config.writer_regions)
        observed_families = set(request.coverage.families) | {p.direction_family for p in self._reserved}
        observed_layers = set(request.coverage.source_layers) | {p.source_layer for p in self._reserved}
        observed_regions = set(request.coverage.writer_regions) | {p.writer_region for p in self._reserved}
        observed_methods = set(request.coverage.basis_methods) | {p.basis_method for p in self._reserved}
        observed_strength_regions = set(request.coverage.strength_regions) | {
            _strength_region(p.strength) for p in self._reserved
        }
        observed_basis_scopes = set(request.coverage.basis_scopes) | {
            p.basis_scope for p in self._reserved
        }
        observed_direction_scopes = set(request.coverage.direction_scopes) | {
            p.truth_direction_scope for p in self._reserved
        }
        observed_normalization_modes = set(request.coverage.normalization_modes) | {
            p.normalization_mode for p in self._reserved
        }
        observed_edit_arms = set(request.coverage.edit_arms) | {
            p.edit_arm for p in self._reserved
        }
        observed_active_edit_arms = set(request.coverage.active_edit_arms) | {
            value for p in self._reserved
            if (value := _active_study_arm(p)) is not None
        }
        observed_writer_configurations = set(
            request.coverage.writer_configurations
        ) | {writer_configuration(p) for p in self._reserved}
        observed_refusal_settings = set(request.coverage.refusal_settings) | {
            refusal_setting(p) for p in self._reserved
        }
        observed_refusal_writer_policies = set(
            request.coverage.refusal_writer_policies
        ) | {
            p.refusal_writer_policy for p in self._reserved
            if refusal_setting(p) != "disabled"
        }
        observed_kernel_center_regions = set(
            request.coverage.kernel_center_regions
        ) | {
            value for p in self._reserved
            if (value := kernel_center_region(p)) is not None
        }
        observed_kernel_half_width_modes = set(
            request.coverage.kernel_half_width_modes
        ) | {
            value for p in self._reserved
            if (value := kernel_half_width_mode(p)) is not None
        }
        observed_kernel_shapes = set(request.coverage.kernel_shapes) | {
            value for p in self._reserved
            if (value := kernel_shape(p)) is not None
        }
        observed_refusal_strength_regions = set(
            request.coverage.refusal_strength_regions
        ) | {refusal_strength_region(p) for p in self._reserved}

        def choose_missing(values, observed, fallback_index):
            return next((item for item in values if item not in observed), values[fallback_index % len(values)])

        rng = random.Random(f"{self.seed}:{request.ordinal}")
        missing_families = set(families) - observed_families
        missing_layers = set(layers) - observed_layers
        required_scopes = _required_basis_scopes(
            directions, config.max_directions_per_trial
        )
        missing_scopes = required_scopes - observed_basis_scopes
        target_scope = next(iter(sorted(missing_scopes)), None)
        scores = {
            item.direction_id: int(item.family in missing_families)
            + int(item.source_layer in missing_layers)
            + int(
                target_scope is not None
                and bool(_scope_candidates(
                    directions, item.source_layer, target_scope,
                    config.max_directions_per_trial,
                ))
            )
            for item in directions
        }
        best_score = max(scores.values())
        primary_candidates = [
            item for item in directions if scores[item.direction_id] == best_score
        ]
        primary = primary_candidates[request.ordinal % len(primary_candidates)]
        source_layer = primary.source_layer
        executable_scopes = [
            scope for scope in ("general", "domain", "mixed")
            if _scope_candidates(directions, source_layer, scope, config.max_directions_per_trial)
        ]
        basis_scope = (
            target_scope if target_scope in executable_scopes
            else choose_missing(executable_scopes, observed_basis_scopes, request.ordinal)
        )
        direction_scope = choose_missing(
            _DIRECTION_SCOPES, observed_direction_scopes, request.ordinal
        )
        if direction_scope == "per_layer":
            region = next(
                item for item in config.writer_regions if source_layer in item.layers
            )
            region_name = region.name
        else:
            region_name = choose_missing(regions, observed_regions, request.ordinal)
            region = next(item for item in config.writer_regions if item.name == region_name)
        method = choose_missing(_BASIS_METHODS, observed_methods, request.ordinal)
        normalization_mode = choose_missing(
            _NORMALIZATION_MODES, observed_normalization_modes, request.ordinal
        )
        missing_writer_configurations = (
            set(WRITER_CONFIGURATIONS) - observed_writer_configurations
        )
        missing_refusal_settings = set(REFUSAL_SETTINGS) - observed_refusal_settings
        missing_refusal_writer_policies = (
            set(REFUSAL_WRITER_POLICIES) - observed_refusal_writer_policies
        )
        edit_arm = choose_missing(_EDIT_ARMS, observed_edit_arms, request.ordinal)
        if "disabled" in missing_writer_configurations:
            edit_arm = "refusal_only"
        elif edit_arm == "refusal_only" and missing_writer_configurations:
            edit_arm = "joint" if "joint" in (
                set(_EDIT_ARMS) - observed_edit_arms
            ) else "truth_only"
        if missing_refusal_settings - {"disabled"} or missing_refusal_writer_policies:
            if edit_arm == "truth_only":
                edit_arm = "joint" if "joint" in (
                    set(_EDIT_ARMS) - observed_edit_arms
                ) else "refusal_only"
        matched_basis_control = (
            "orthogonal"
            if "orthogonal_control" not in observed_active_edit_arms
            else "none"
        )
        if matched_basis_control == "orthogonal":
            # A matched orthogonal basis is a persistent truth-writer control.
            # It must never acquire a refusal contribution or activation scope.
            edit_arm = "truth_only"
        strength_region = choose_missing(
            _STRENGTH_REGIONS, observed_strength_regions, request.ordinal
        )
        candidates = list(
            _scope_candidates(
                directions, source_layer, basis_scope, config.max_directions_per_trial
            )
        )
        count = min(config.max_directions_per_trial, len(candidates))
        selected_entries: tuple[DirectionEntry, ...]
        if basis_scope == "mixed":
            general_items = tuple(item for item in candidates if item.family == "general")
            domain_items = tuple(item for item in candidates if item.family != "general")
            selected_entries = (
                general_items[request.ordinal % len(general_items)],
                domain_items[request.ordinal % len(domain_items)],
            )
        else:
            start = request.ordinal % len(candidates)
            selected_entries = tuple(
                candidates[(start + offset) % len(candidates)] for offset in range(count)
            )
        selected = tuple(item.direction_id for item in selected_entries)
        selected_families = {item.family for item in selected_entries}
        family = next(iter(selected_families)) if len(selected_families) == 1 else "mixed"
        selected_domains = tuple(sorted({
            domain for item in selected_entries for domain in item.domains
            if domain not in {"general", "all"}
        }))
        available_rank = sum(item.rank for item in selected_entries)
        requested_rank = 1 + rng.randrange(min(config.max_rank, available_rank))
        coverage_strength = {
            "disabled": 0.0,
            "projection": min(1.0, config.strength_max),
            "reflection": config.strength_max,
        }[strength_region]
        if not config.strength_min <= coverage_strength <= config.strength_max:
            strength = config.strength_min + rng.random() * (
                config.strength_max - config.strength_min
            )
        else:
            strength = coverage_strength
        if edit_arm == "refusal_only":
            truth_writer_configuration = "disabled"
        else:
            truth_writer_configuration = choose_missing(
                ("attention", "mlp", "both"),
                observed_writer_configurations,
                request.ordinal,
            )
        attention_enabled = truth_writer_configuration in {"attention", "both"}
        mlp_enabled = truth_writer_configuration in {"mlp", "both"}
        target_center_region = choose_missing(
            KERNEL_CENTER_REGIONS, observed_kernel_center_regions, request.ordinal
        )
        target_width_mode = choose_missing(
            ("local", "broad"),
            observed_kernel_half_width_modes,
            request.ordinal,
        )
        target_kernel_shape = choose_missing(
            KERNEL_SHAPES, observed_kernel_shapes, request.ordinal
        )
        if direction_scope == "per_layer":
            center = float(source_layer)
            half_width = 0.0
            writer_layers: tuple[int, ...] = (source_layer,)
        else:
            writer_layers = region.layers
            center_index = {
                "early": 0,
                "middle": len(region.layers) // 2,
                "late": len(region.layers) - 1,
            }[target_center_region]
            center = float(region.layers[center_index])
            half_width = (
                0.0 if target_width_mode == "local"
                else float(max(region.layers) - min(region.layers))
            )
        if truth_writer_configuration != "disabled" and strength == 0.0:
            strength = 1.0
        edge = (
            strength if target_kernel_shape == "flat"
            else round(strength * 0.5, 12)
        )
        refusal_enabled = edit_arm in {"refusal_only", "joint"}
        refusal_scope = (
            choose_missing(
                ("global", "per_layer"),
                observed_refusal_settings,
                request.ordinal,
            )
            if refusal_enabled else "global"
        )
        refusal_writer_policy = (
            choose_missing(
                _REFUSAL_WRITER_POLICIES,
                observed_refusal_writer_policies,
                request.ordinal,
            )
            if refusal_enabled else "both"
        )
        target_refusal_strength_region = choose_missing(
            REFUSAL_STRENGTH_REGIONS,
            observed_refusal_strength_regions,
            request.ordinal,
        )
        refusal_strength = {
            "disabled": 0.0,
            "projection": 1.0,
            "reflection": 2.0,
        }[target_refusal_strength_region] if refusal_enabled else 0.0
        if refusal_enabled and (
            missing_refusal_settings - {"disabled"}
            or missing_refusal_writer_policies
        ) and refusal_strength == 0.0:
            refusal_strength = 1.0
        refusal_source_layer = (
            source_layer if refusal_enabled and refusal_scope == "global" else None
        )
        legacy_policy = (
            "both" if truth_writer_configuration == "disabled"
            else truth_writer_configuration
        )
        proposal = SearchProposal(
            selected, family, source_layer, method, requested_rank, region.name,
            writer_layers, cast(Literal["attention", "mlp", "both"], legacy_policy),
            round(strength, 12),
            basis_scope=cast(Literal["general", "domain", "mixed"], basis_scope),
            selected_domains=selected_domains,
            truth_direction_scope=direction_scope,
            normalization_mode=normalization_mode, edit_arm=edit_arm,
            attention_enabled=attention_enabled,
            attention_kernel_center=center,
            attention_kernel_half_width=half_width,
            attention_edge_strength=edge if attention_enabled else 0.0,
            attention_peak_strength=strength if attention_enabled else 0.0,
            mlp_enabled=mlp_enabled, mlp_kernel_center=center,
            mlp_kernel_half_width=half_width,
            mlp_edge_strength=edge if mlp_enabled else 0.0,
            mlp_peak_strength=strength if mlp_enabled else 0.0,
            refusal_enabled=refusal_enabled,
            refusal_direction_scope=cast(Literal["global", "per_layer"], refusal_scope),
            refusal_source_layer=refusal_source_layer,
            refusal_strength=refusal_strength,
            refusal_writer_policy=refusal_writer_policy,
            matched_basis_control=matched_basis_control,
        )
        self._reserved.append(proposal)
        return proposal

    def observe(self, trials: Sequence["StudyTrial"]) -> None:
        self.observed_batch_sizes.append(len(trials))
        self._reserved.clear()

    def complete_history_replay(self) -> None:
        """Mark the boundary between journal replay and new suggestions."""


class OptunaSearchDriver(OfflineDeterministicSearchDriver):
    """Optional adapter; imports Optuna only when explicitly selected.

    Coverage suggestions remain deterministic.  After coverage, Optuna's
    multivariate TPE observations may replace the deterministic concentration
    policy without changing the study or journal interfaces.
    """

    def __init__(self, *, seed: int) -> None:
        try:
            import optuna  # type: ignore
        except ImportError as error:
            raise StudyError(
                "Optuna search requested but optuna is not installed; install the locked "
                "runtime dependency or use --search-driver offline"
            ) from error
        super().__init__(seed=seed)
        self._optuna = optuna
        self._optuna_version = optuna.__version__
        self._study = optuna.create_study(
            directions=["maximize"] * len(OBJECTIVES),
            sampler=optuna.samplers.TPESampler(
                seed=seed, multivariate=True, n_startup_trials=60,
                n_ei_candidates=128,
            ),
        )
        self._live_trials: dict[int, Any] = {}
        self._config: TruthEditingStudyConfig | None = None
        self._directions: tuple[DirectionEntry, ...] = ()
        self._observed_trials: list[StudyTrial] = []
        self._pending_proposals: dict[int, SearchProposal] = {}
        self._persistent_study: Any | None = None
        self._persisted_ordinals: set[int] = set()
        self._history_replay_is_complete = False
        self._unresolved_operational_failures: dict[str, SearchProposal] = {}
        self._failure_replay_queue: list[str] = []
        self._failure_replays_inflight: dict[int, str] = {}

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"adapter": "optuna_multivariate_tpe_v2", "seed": self.seed, "version": self._optuna_version}

    @property
    def persistent_study_name(self) -> str:
        """The durable Optuna study name, available after ``prepare``."""

        if self._persistent_study is None:
            raise StudyError("persistent Optuna study is unavailable before prepare")
        return str(self._persistent_study.study_name)

    def prepare(
        self, config: TruthEditingStudyConfig, directions: tuple[DirectionEntry, ...],
        state_path: Path,
    ) -> None:
        self._config = config
        self._directions = directions
        sampler = self._optuna.samplers.TPESampler(
            seed=self.seed,
            multivariate=config.tpe_multivariate,
            n_startup_trials=config.tpe_startup_trials,
            n_ei_candidates=config.tpe_ei_candidates,
        )
        self._study = self._optuna.create_study(
            directions=["maximize"] * len(OBJECTIVES), sampler=sampler,
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        reopen_existing = state_path.is_file() and state_path.stat().st_size > 0
        try:
            journal_backend = self._optuna.storages.journal.JournalFileBackend(
                str(state_path)
            )
            storage = self._optuna.storages.JournalStorage(journal_backend)
        except (AttributeError, TypeError) as error:
            raise StudyError(
                "installed Optuna lacks JournalStorage/JournalFileBackend support"
            ) from error
        direction_identity = _sha(
            [
                [item.direction_id, item.artifact.vector_sha256]
                for item in directions
            ]
        )
        study_name = (
            f"{config.study_id}-{config.identity_sha256[:12]}-"
            f"{direction_identity[:12]}"
        )
        try:
            self._persistent_study = (
                self._optuna.load_study(
                    study_name=study_name,
                    storage=storage,
                    sampler=sampler,
                )
                if reopen_existing
                else self._optuna.create_study(
                    study_name=study_name,
                    directions=["maximize"] * len(OBJECTIVES),
                    storage=storage,
                    sampler=sampler,
                )
            )
        except KeyError as error:
            raise StudyError(
                "existing Optuna journal does not contain the expected study"
            ) from error
        self._persisted_ordinals = {
            int(trial.user_attrs["study_ordinal"])
            for trial in self._persistent_study.get_trials(deepcopy=False)
            if "study_ordinal" in trial.user_attrs
        }

    @staticmethod
    def _coverage_complete(request: SearchRequest) -> bool:
        return _broad_coverage_contract(
            request.config, request.directions
        ).is_complete(request.coverage)

    @staticmethod
    def _conditional_parameter(base: str, *context: object) -> str:
        """Return the stable Optuna name for one conditional distribution.

        Optuna permanently associates a distribution with a parameter name.
        Search branches whose choices or bounds depend on an earlier choice must
        therefore use different names.  The proposal compiler and frozen-trial
        compiler both call this function so live sampling and journal replay use
        exactly the same namespace.
        """

        return "::".join((base, *(str(item) for item in context)))

    def suggest(self, request: SearchRequest) -> SearchProposal:
        if self._history_replay_is_complete:
            while self._failure_replay_queue:
                proposal_sha256 = self._failure_replay_queue.pop(0)
                proposal = self._unresolved_operational_failures.get(proposal_sha256)
                if proposal is None:
                    continue
                self._pending_proposals[request.ordinal] = proposal
                self._failure_replays_inflight[request.ordinal] = proposal_sha256
                self._reserved.append(proposal)
                return proposal
        if not self._coverage_complete(request):
            proposal = super().suggest(request)
            self._pending_proposals[request.ordinal] = proposal
            return proposal
        trial = self._study.ask()
        available_basis_scopes = tuple(sorted(_required_basis_scopes(
            request.directions, request.config.max_directions_per_trial
        )))
        basis_scope = trial.suggest_categorical("basis_scope", available_basis_scopes)
        compatible_layers = tuple(sorted({
            item.source_layer for item in request.directions
            if _scope_candidates(
                request.directions, item.source_layer, basis_scope,
                request.config.max_directions_per_trial,
            )
        }))
        source_layer_parameter = self._conditional_parameter(
            "source_layer", basis_scope
        )
        source_layer = trial.suggest_categorical(
            source_layer_parameter, compatible_layers,
        )
        eligible = _scope_candidates(
            request.directions, source_layer, basis_scope,
            request.config.max_directions_per_trial,
        )
        if basis_scope == "mixed":
            general = tuple(item for item in eligible if item.family == "general")
            domain = tuple(item for item in eligible if item.family != "general")
            general_index = trial.suggest_int(
                self._conditional_parameter(
                    "general_direction_index", basis_scope, source_layer
                ),
                0,
                len(general) - 1,
            )
            domain_index = trial.suggest_int(
                self._conditional_parameter(
                    "domain_direction_index", basis_scope, source_layer
                ),
                0,
                len(domain) - 1,
            )
            selected = (general[general_index], domain[domain_index])
        else:
            direction_count = trial.suggest_int(
                self._conditional_parameter(
                    "direction_count", basis_scope, source_layer
                ),
                1,
                min(request.config.max_directions_per_trial, len(eligible)),
            )
            direction_start = trial.suggest_int(
                self._conditional_parameter(
                    "direction_start", basis_scope, source_layer
                ),
                0,
                len(eligible) - 1,
            )
            selected = tuple(
                eligible[(direction_start + offset) % len(eligible)]
                for offset in range(direction_count)
            )
        selected_ids = tuple(item.direction_id for item in selected)
        families = {item.family for item in selected}
        family = next(iter(families)) if len(families) == 1 else "mixed"
        selected_domains = tuple(sorted({
            domain for item in selected for domain in item.domains
            if domain not in {"general", "all"}
        }))
        truth_direction_scope = trial.suggest_categorical(
            "truth_direction_scope", _DIRECTION_SCOPES
        )
        if truth_direction_scope == "per_layer":
            region = next(
                item for item in request.config.writer_regions
                if source_layer in item.layers
            )
            writer_layers: tuple[int, ...] = (source_layer,)
        else:
            region_name = trial.suggest_categorical(
                "writer_region", tuple(item.name for item in request.config.writer_regions)
            )
            region = next(
                item for item in request.config.writer_regions if item.name == region_name
            )
            writer_layers = tuple(region.layers)
        method = trial.suggest_categorical("basis_method", _BASIS_METHODS)
        available_rank = min(
            request.config.max_rank, sum(item.rank for item in selected)
        )
        requested_rank = trial.suggest_int(
            self._conditional_parameter("requested_rank", available_rank),
            1,
            available_rank,
        )
        normalization_mode = trial.suggest_categorical(
            "normalization_mode", _NORMALIZATION_MODES
        )
        edit_arm = trial.suggest_categorical("edit_arm", _EDIT_ARMS)
        truth_active = edit_arm != "refusal_only"
        attention_enabled = truth_active and trial.suggest_categorical(
            "attention_enabled", (False, True)
        )
        mlp_enabled = truth_active and trial.suggest_categorical(
            "mlp_enabled", (False, True)
        )

        def kernel(site: str, enabled: bool) -> tuple[float, float, float, float]:
            if truth_direction_scope == "per_layer":
                center = float(source_layer)
                half_width = 0.0
            else:
                center = trial.suggest_float(
                    self._conditional_parameter(
                        f"{site}_kernel_center", region.name
                    ),
                    min(writer_layers),
                    max(writer_layers),
                )
                half_width = trial.suggest_float(
                    self._conditional_parameter(
                        f"{site}_kernel_half_width", region.name
                    ),
                    0.0,
                    float(max(writer_layers) - min(writer_layers)),
                )
            if not enabled:
                return center, half_width, 0.0, 0.0
            peak = trial.suggest_float(f"{site}_peak_strength", 0.0, 2.0)
            edge_ratio = trial.suggest_float(f"{site}_edge_ratio", 0.0, 1.0)
            return center, half_width, edge_ratio * peak, peak

        attn_center, attn_width, attn_edge, attn_peak = kernel(
            "attention", attention_enabled
        )
        mlp_center, mlp_width, mlp_edge, mlp_peak = kernel("mlp", mlp_enabled)
        refusal_enabled = edit_arm in {"refusal_only", "joint"}
        if refusal_enabled:
            refusal_scope = trial.suggest_categorical(
                "refusal_direction_scope", _DIRECTION_SCOPES
            )
            refusal_source_layer = (
                trial.suggest_categorical(
                    "refusal_source_layer",
                    tuple(sorted({item.source_layer for item in request.directions})),
                )
                if refusal_scope == "global" else None
            )
            refusal_writer_policy = trial.suggest_categorical(
                "refusal_writer_policy", _REFUSAL_WRITER_POLICIES
            )
            refusal_strength = trial.suggest_float("refusal_strength", 0.0, 2.0)
        else:
            refusal_scope = "global"
            refusal_source_layer = None
            refusal_writer_policy = "both"
            refusal_strength = 0.0
        writer_policy = (
            "both" if attention_enabled and mlp_enabled
            else "attention" if attention_enabled
            else "mlp" if mlp_enabled else "both"
        )
        strength = max(attn_peak, mlp_peak)
        proposal = SearchProposal(
            selected_ids, family, source_layer, method, requested_rank,
            region.name, writer_layers,
            cast(Literal["attention", "mlp", "both"], writer_policy), strength,
            basis_scope=cast(Literal["general", "domain", "mixed"], basis_scope),
            selected_domains=selected_domains,
            truth_direction_scope=cast(Literal["global", "per_layer"], truth_direction_scope),
            normalization_mode=cast(Literal["exact", "norm_preserving"], normalization_mode),
            edit_arm=cast(Literal["truth_only", "refusal_only", "joint"], edit_arm),
            attention_enabled=attention_enabled,
            attention_kernel_center=attn_center,
            attention_kernel_half_width=attn_width,
            attention_edge_strength=attn_edge,
            attention_peak_strength=attn_peak,
            mlp_enabled=mlp_enabled, mlp_kernel_center=mlp_center,
            mlp_kernel_half_width=mlp_width, mlp_edge_strength=mlp_edge,
            mlp_peak_strength=mlp_peak, refusal_enabled=refusal_enabled,
            refusal_direction_scope=cast(Literal["global", "per_layer"], refusal_scope),
            refusal_source_layer=refusal_source_layer,
            refusal_strength=refusal_strength,
            refusal_writer_policy=cast(
                Literal["attention", "mlp", "both"], refusal_writer_policy
            ),
            proposal_origin="tpe_sampled",
        )
        self._live_trials[request.ordinal] = trial
        self._pending_proposals[request.ordinal] = proposal
        self._reserved.append(proposal)
        return proposal

    def _frozen_trial(self, trial: "StudyTrial") -> Any:
        """Reconstruct an observation when resuming from the durable journal."""

        proposal = trial.proposal
        if self._config is None or not self._directions:
            raise StudyError("Optuna driver was not prepared with the frozen search space")
        categorical = self._optuna.distributions.CategoricalDistribution
        integer = self._optuna.distributions.IntDistribution
        floating = self._optuna.distributions.FloatDistribution
        available_scopes = tuple(sorted(_required_basis_scopes(
            self._directions, self._config.max_directions_per_trial
        )))
        compatible_layers = tuple(sorted({
            item.source_layer for item in self._directions
            if _scope_candidates(
                self._directions, item.source_layer, cast(str, proposal.basis_scope),
                self._config.max_directions_per_trial,
            )
        }))
        eligible = _scope_candidates(
            self._directions, proposal.source_layer, cast(str, proposal.basis_scope),
            self._config.max_directions_per_trial,
        )
        source_layer_parameter = self._conditional_parameter(
            "source_layer", proposal.basis_scope
        )
        available_rank = min(self._config.max_rank, sum(
            item.rank for item in self._directions
            if item.direction_id in proposal.direction_ids
        ))
        requested_rank_parameter = self._conditional_parameter(
            "requested_rank", available_rank
        )
        params: dict[str, Any] = {
            "basis_scope": proposal.basis_scope,
            source_layer_parameter: proposal.source_layer,
            "basis_method": proposal.basis_method,
            requested_rank_parameter: proposal.requested_rank,
            "normalization_mode": proposal.normalization_mode,
            "edit_arm": proposal.edit_arm,
            "truth_direction_scope": proposal.truth_direction_scope,
        }
        distributions: dict[str, Any] = {
            "basis_scope": categorical(available_scopes),
            source_layer_parameter: categorical(compatible_layers),
            "basis_method": categorical(_BASIS_METHODS),
            requested_rank_parameter: integer(1, available_rank),
            "normalization_mode": categorical(_NORMALIZATION_MODES),
            "edit_arm": categorical(_EDIT_ARMS),
            "truth_direction_scope": categorical(_DIRECTION_SCOPES),
        }
        if proposal.basis_scope == "mixed":
            general = tuple(item for item in eligible if item.family == "general")
            domain = tuple(item for item in eligible if item.family != "general")
            general_parameter = self._conditional_parameter(
                "general_direction_index", proposal.basis_scope,
                proposal.source_layer,
            )
            domain_parameter = self._conditional_parameter(
                "domain_direction_index", proposal.basis_scope,
                proposal.source_layer,
            )
            params[general_parameter] = next(
                index for index, item in enumerate(general)
                if item.direction_id == proposal.direction_ids[0]
            )
            params[domain_parameter] = next(
                index for index, item in enumerate(domain)
                if item.direction_id == proposal.direction_ids[1]
            )
            distributions[general_parameter] = integer(0, len(general) - 1)
            distributions[domain_parameter] = integer(0, len(domain) - 1)
        else:
            count_parameter = self._conditional_parameter(
                "direction_count", proposal.basis_scope, proposal.source_layer
            )
            start_parameter = self._conditional_parameter(
                "direction_start", proposal.basis_scope, proposal.source_layer
            )
            params[count_parameter] = len(proposal.direction_ids)
            params[start_parameter] = next(
                index for index, item in enumerate(eligible)
                if item.direction_id == proposal.direction_ids[0]
            )
            distributions[count_parameter] = integer(
                1, min(self._config.max_directions_per_trial, len(eligible))
            )
            distributions[start_parameter] = integer(0, len(eligible) - 1)
        if proposal.truth_direction_scope == "global":
            params["writer_region"] = proposal.writer_region
            distributions["writer_region"] = categorical(
                tuple(item.name for item in self._config.writer_regions)
            )
        if proposal.edit_arm != "refusal_only":
            params["attention_enabled"] = proposal.attention_enabled
            params["mlp_enabled"] = proposal.mlp_enabled
            distributions["attention_enabled"] = categorical((False, True))
            distributions["mlp_enabled"] = categorical((False, True))
        for site in ("attention", "mlp"):
            enabled = cast(bool, getattr(proposal, f"{site}_enabled"))
            center = cast(float, getattr(proposal, f"{site}_kernel_center"))
            width = cast(float, getattr(proposal, f"{site}_kernel_half_width"))
            edge = cast(float, getattr(proposal, f"{site}_edge_strength"))
            peak = cast(float, getattr(proposal, f"{site}_peak_strength"))
            if proposal.truth_direction_scope == "global":
                center_parameter = self._conditional_parameter(
                    f"{site}_kernel_center", proposal.writer_region
                )
                width_parameter = self._conditional_parameter(
                    f"{site}_kernel_half_width", proposal.writer_region
                )
                params[center_parameter] = center
                params[width_parameter] = width
                distributions[center_parameter] = floating(
                    min(proposal.writer_layers), max(proposal.writer_layers)
                )
                distributions[width_parameter] = floating(
                    0.0, float(max(proposal.writer_layers) - min(proposal.writer_layers))
                )
            if enabled:
                params[f"{site}_peak_strength"] = peak
                params[f"{site}_edge_ratio"] = edge / peak if peak else 0.0
                distributions[f"{site}_peak_strength"] = floating(0.0, 2.0)
                distributions[f"{site}_edge_ratio"] = floating(0.0, 1.0)
        if proposal.refusal_enabled:
            params["refusal_direction_scope"] = proposal.refusal_direction_scope
            params["refusal_writer_policy"] = proposal.refusal_writer_policy
            distributions["refusal_direction_scope"] = categorical(_DIRECTION_SCOPES)
            distributions["refusal_writer_policy"] = categorical(
                _REFUSAL_WRITER_POLICIES
            )
            if proposal.refusal_direction_scope == "global":
                params["refusal_source_layer"] = proposal.refusal_source_layer
                distributions["refusal_source_layer"] = categorical(tuple(sorted({
                    item.source_layer for item in self._directions
                })))
            params["refusal_strength"] = proposal.refusal_strength
            distributions["refusal_strength"] = floating(0.0, 2.0)
        if trial.result.outcome_kind == "operational_failure":
            return self._optuna.trial.create_trial(
                params=params, distributions=distributions,
                state=self._optuna.trial.TrialState.FAIL,
                user_attrs={
                    "study_ordinal": trial.ordinal,
                    "proposal_sha256": _sha(trial.proposal.to_dict()),
                },
            )
        if trial.result.outcome_kind == "scientifically_infeasible":
            return self._optuna.trial.create_trial(
                params=params, distributions=distributions,
                state=self._optuna.trial.TrialState.PRUNED,
                user_attrs={
                    "study_ordinal": trial.ordinal,
                    "proposal_sha256": _sha(trial.proposal.to_dict()),
                },
            )
        return self._optuna.trial.create_trial(
            params=params, distributions=distributions,
            values=tuple(trial.result.metrics[name] for name in OBJECTIVES),
            user_attrs={
                "study_ordinal": trial.ordinal,
                "proposal_sha256": _sha(trial.proposal.to_dict()),
            },
        )

    def observe(self, trials: Sequence["StudyTrial"]) -> None:
        self.observed_batch_sizes.append(len(trials))
        valid_history = [
            item
            for item in self._observed_trials
            if item.result.outcome_kind != "operational_failure"
        ]
        coverage = _coverage_ledger(tuple(item.proposal for item in valid_history))
        # On resume, regenerate every suggestion through the same seeded sampler
        # and accumulated observations. This advances sampler state exactly as an
        # uninterrupted run and verifies that the journal proposal is not a lossy
        # reconstruction.
        if self._config is None or not self._directions:
            raise StudyError("Optuna driver was not prepared with the frozen search space")
        for item in trials:
            if item.ordinal not in self._pending_proposals:
                proposal_sha256 = _sha(item.proposal.to_dict())
                if (
                    not self._history_replay_is_complete
                    and proposal_sha256 in self._unresolved_operational_failures
                ):
                    # A saved incomplete batch may already contain retries that
                    # were proposed before the process stopped. Reconstructing
                    # such a retry must not advance either deterministic or TPE
                    # sampler state.
                    regenerated = item.proposal
                    self._pending_proposals[item.ordinal] = regenerated
                    self._failure_replays_inflight[item.ordinal] = proposal_sha256
                    self._reserved.append(regenerated)
                else:
                    regenerated = self.suggest(
                        SearchRequest(
                            item.ordinal, self._config, self._directions, coverage
                        )
                    )
                if regenerated.to_dict() != item.proposal.to_dict():
                    raise StudyError("Optuna resume suggestion differs from journal")
            elif self._pending_proposals[item.ordinal].to_dict() != item.proposal.to_dict():
                raise StudyError("Optuna observed proposal differs from pending suggestion")
        live_tpe_ordinals: set[int] = set()
        for item in trials:
            live = self._live_trials.pop(item.ordinal, None)
            if live is not None:
                live_tpe_ordinals.add(item.ordinal)
            if item.proposal.matched_basis_control != "none":
                if live is not None:
                    raise StudyError(
                        "matched basis controls cannot be sampled by concentrating TPE"
                    )
                # The study journal retains the successful control trial. It is
                # deliberately absent from TPE observations so its outcome is
                # not attributed to the parent truth-direction parameters.
                continue
            if live is None:
                if item.result.outcome_kind != "operational_failure":
                    self._study.add_trial(self._frozen_trial(item))
            elif item.result.outcome_kind == "operational_failure":
                self._study.tell(live, state=self._optuna.trial.TrialState.FAIL)
            elif item.result.outcome_kind == "scientifically_infeasible":
                self._study.tell(live, state=self._optuna.trial.TrialState.PRUNED)
            else:
                self._study.tell(
                    live, tuple(item.result.metrics[name] for name in OBJECTIVES)
                )
        self._observed_trials.extend(trials)
        if self._persistent_study is None:
            raise StudyError("Optuna persistent journal storage is not prepared")
        for item in trials:
            if (
                item.proposal.matched_basis_control == "none"
                and item.ordinal not in self._persisted_ordinals
                and (
                    item.result.outcome_kind != "operational_failure"
                    or item.ordinal in live_tpe_ordinals
                )
            ):
                self._persistent_study.add_trial(self._frozen_trial(item))
                self._persisted_ordinals.add(item.ordinal)
        for item in trials:
            proposal_sha256 = _sha(item.proposal.to_dict())
            self._failure_replays_inflight.pop(item.ordinal, None)
            if item.result.outcome_kind == "operational_failure":
                if proposal_sha256 not in self._unresolved_operational_failures:
                    self._unresolved_operational_failures[proposal_sha256] = item.proposal
                if (
                    self._history_replay_is_complete
                    and proposal_sha256 not in self._failure_replay_queue
                    and proposal_sha256 not in self._failure_replays_inflight.values()
                ):
                    self._failure_replay_queue.append(proposal_sha256)
            else:
                self._unresolved_operational_failures.pop(proposal_sha256, None)
                self._failure_replay_queue = [
                    queued_sha256
                    for queued_sha256 in self._failure_replay_queue
                    if queued_sha256 != proposal_sha256
                ]
            self._pending_proposals.pop(item.ordinal, None)
        self._reserved.clear()

    def complete_history_replay(self) -> None:
        """Enable exact FIFO retries after all saved proposals are reconstructed."""

        if self._history_replay_is_complete:
            return
        self._history_replay_is_complete = True
        self._failure_replay_queue.extend(
            proposal_sha256
            for proposal_sha256 in self._unresolved_operational_failures
            if proposal_sha256 not in self._failure_replay_queue
        )


OutcomeKind = Literal["successful", "scientifically_infeasible", "operational_failure"]


@dataclass(frozen=True)
class EvaluationResult:
    outcome_kind: OutcomeKind
    metrics: Mapping[str, float]
    detail: str | None = None

    @classmethod
    def successful(cls, metrics: Mapping[str, float]) -> "EvaluationResult":
        return cls("successful", dict(metrics))

    @classmethod
    def scientifically_infeasible(cls, metrics: Mapping[str, float], detail: str) -> "EvaluationResult":
        return cls("scientifically_infeasible", dict(metrics), detail)

    @classmethod
    def operational_failure(cls, detail: str) -> "EvaluationResult":
        return cls("operational_failure", {}, detail)


class Evaluator(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...
    def evaluate(
        self, proposal: SearchProposal, *, trial_id: str,
        record_ids: tuple[str, ...], objective_names: tuple[str, ...],
    ) -> EvaluationResult: ...


class OfflineSyntheticEvaluator:
    """Deterministic software-only evaluator; never behavioral evidence."""

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"adapter": "offline_synthetic_evaluator_v1"}

    def evaluate(self, proposal, *, trial_id, record_ids, objective_names):
        digest = hashlib.sha256(_canonical({"trial_id": trial_id, "proposal": proposal.to_dict()})).digest()
        metrics = {
            name: int.from_bytes(digest[index * 4:index * 4 + 4], "big") / (2**32 - 1)
            for index, name in enumerate(objective_names)
        }
        return EvaluationResult.successful(metrics)

    def evaluate_matched_basis_control(
        self, proposal: SearchProposal, *, trial_id: str,
        record_ids: tuple[str, ...], objective_names: tuple[str, ...],
        control_kind: Literal["orthogonal"], execution_identity_sha256: str,
    ) -> EvaluationResult:
        """Offline adapter for the persistent matched-control evaluation seam."""

        del control_kind, execution_identity_sha256
        return self.evaluate(
            proposal,
            trial_id=trial_id,
            record_ids=record_ids,
            objective_names=objective_names,
        )


@dataclass(frozen=True)
class StudyTrial:
    trial_id: str
    ordinal: int
    batch_ordinal: int
    tier_name: str
    evaluation_record_ids: tuple[str, ...]
    proposal: SearchProposal
    result: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id, "ordinal": self.ordinal,
            "batch_ordinal": self.batch_ordinal, "tier_name": self.tier_name,
            "evaluation_record_ids": list(self.evaluation_record_ids),
            "proposal": self.proposal.to_dict(),
            "result": {"outcome_kind": self.result.outcome_kind, "metrics": dict(self.result.metrics), "detail": self.result.detail},
        }


@dataclass(frozen=True)
class StudyReport:
    format: Literal["truth_editing_study_report_v1"]
    study_identity_sha256: str
    trials: tuple[StudyTrial, ...]
    coverage: CoverageLedger
    coverage_complete: bool

    @property
    def completed_trials(self) -> int: return len(self.trials)
    @property
    def operational_failures(self) -> int: return sum(t.result.outcome_kind == "operational_failure" for t in self.trials)
    @property
    def scientifically_infeasible_trials(self) -> int: return sum(t.result.outcome_kind == "scientifically_infeasible" for t in self.trials)
    @property
    def successful_trials(self) -> int: return sum(t.result.outcome_kind == "successful" for t in self.trials)
    @property
    def unresolved_operational_failures(self) -> int:
        resolved_proposals: set[str] = set()
        unresolved = 0
        for trial in reversed(self.trials):
            proposal_sha256 = _sha(trial.proposal.to_dict())
            if trial.result.outcome_kind == "operational_failure":
                if proposal_sha256 not in resolved_proposals:
                    unresolved += 1
            else:
                resolved_proposals.add(proposal_sha256)
        return unresolved
    @property
    def selection_ready(self) -> bool:
        """Whether this journal can safely feed scientific model selection."""

        return self.coverage_complete and self.unresolved_operational_failures == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "study_identity_sha256": self.study_identity_sha256,
            "completed_trials": self.completed_trials,
            "successful_trials": self.successful_trials,
            "scientifically_infeasible_trials": self.scientifically_infeasible_trials,
            "operational_failures": self.operational_failures,
            "coverage": asdict(self.coverage),
            "coverage_complete": self.coverage_complete,
            "selection_ready": self.selection_ready,
            "trials": [item.to_dict() for item in self.trials],
        }


class TruthEditingStudy:
    def __init__(self, config: TruthEditingStudyConfig, direction_bank: DirectionBankManifest) -> None:
        if not isinstance(config, TruthEditingStudyConfig) or not isinstance(direction_bank, DirectionBankManifest):
            raise StudyError("study requires parsed config and direction bank")
        directions = tuple(
            item
            for item in direction_bank.directions
            if item.kind == "truth" and item.qualification.status == "qualified"
        )
        if not directions:
            raise StudyError("study requires at least one qualified truth direction")
        invalid_writer_layers = sorted({layer for region in config.writer_regions for layer in region.layers if layer >= direction_bank.model.decoder_layer_count})
        if invalid_writer_layers:
            raise StudyError(f"writer layers exceed model range: {invalid_writer_layers}")
        writer_layers = [layer for region in config.writer_regions for layer in region.layers]
        if len(writer_layers) != len(set(writer_layers)):
            raise StudyError("writer regions must not overlap")
        if set(writer_layers) != set(range(direction_bank.model.decoder_layer_count)):
            raise StudyError("writer regions must cover every decoder layer exactly once")
        self.config = config
        self.direction_bank = direction_bank
        self.directions = directions

    def _identity(
        self, driver: SearchDriver, evaluator: Evaluator
    ) -> tuple[str, dict[str, Any]]:
        batch_module_path = Path(__file__).with_name("truth_editing_batch_execution.py")
        batch_module_sha256 = hashlib.sha256(batch_module_path.read_bytes()).hexdigest()
        broad_module_path = Path(__file__).with_name("truth_editing_broad_coverage.py")
        broad_module_sha256 = hashlib.sha256(broad_module_path.read_bytes()).hexdigest()
        body = {
            "config_sha256": self.config.identity_sha256,
            "direction_manifest_sha256": self.direction_bank.self_sha256,
            "dataset_manifest_sha256": self.config.dataset_manifest_sha256,
            "search_driver": dict(driver.identity),
            "evaluator": dict(evaluator.identity),
            "orchestrator_module_sha256": STUDY_ORCHESTRATOR_SEMANTICS_SHA256,
            "batch_scheduler_module_sha256": batch_module_sha256,
            "broad_coverage_module_sha256": broad_module_sha256,
        }
        return _sha(body), body

    def _tier(self, ordinal: int) -> EvaluationTier:
        trial_number = ordinal + 1
        return next(item for item in self.config.evaluation_tiers if trial_number <= item.through_trial)

    def _validate_proposal(self, proposal: SearchProposal) -> None:
        if proposal.backend_type != "persistent_weight":
            raise StudyError("routine optimization permits persistent_weight recipes only")
        if not proposal.direction_ids or len(proposal.direction_ids) > self.config.max_directions_per_trial:
            raise StudyError("proposal direction count is outside the frozen sparse search")
        if len(set(proposal.direction_ids)) != len(proposal.direction_ids):
            raise StudyError("proposal direction IDs must be unique")
        by_id = {item.direction_id: item for item in self.directions}
        if any(direction_id not in by_id for direction_id in proposal.direction_ids):
            raise StudyError("proposal contains an unqualified or non-truth direction")
        selected = tuple(by_id[item] for item in proposal.direction_ids)
        selected_families = {item.family for item in selected}
        expected_family = next(iter(selected_families)) if len(selected_families) == 1 else "mixed"
        if proposal.direction_family != expected_family:
            raise StudyError("proposal direction family does not match selected directions")
        has_general = any(item.family == "general" for item in selected)
        has_domain = any(item.family != "general" for item in selected)
        expected_scope = "mixed" if has_general and has_domain else (
            "general" if has_general else "domain"
        )
        if proposal.basis_scope != expected_scope:
            raise StudyError("proposal basis scope does not match selected directions")
        expected_domains = tuple(sorted({
            domain for item in selected for domain in item.domains
            if domain not in {"general", "all"}
        }))
        if proposal.selected_domains != expected_domains:
            raise StudyError("proposal selected_domains do not match selected directions")
        if {item.source_layer for item in selected} != {proposal.source_layer}:
            raise StudyError("proposal directions must share exactly one source layer")
        if proposal.basis_method not in _BASIS_METHODS:
            raise StudyError("proposal basis method is outside QR/SVD")
        if proposal.requested_rank > min(
            self.config.max_rank, sum(item.rank for item in selected)
        ):
            raise StudyError("proposal rank exceeds the selected qualified rank")
        regions = {item.name: item.layers for item in self.config.writer_regions}
        region_layers = regions.get(proposal.writer_region)
        if region_layers is None or not set(proposal.writer_layers).issubset(region_layers):
            raise StudyError("proposal writer layers are outside its frozen region")
        if not proposal.writer_layers or len(set(proposal.writer_layers)) != len(proposal.writer_layers):
            raise StudyError("proposal writer layers must be nonempty and unique")
        if proposal.writer_policy not in _WRITER_POLICIES:
            raise StudyError("proposal writer policy is invalid")
        if not self.config.strength_min <= proposal.strength <= self.config.strength_max:
            raise StudyError("proposal strength is outside the frozen [0, 2] range")
        if proposal.truth_direction_scope not in _DIRECTION_SCOPES:
            raise StudyError("proposal truth direction scope is invalid")
        if proposal.normalization_mode not in _NORMALIZATION_MODES:
            raise StudyError("proposal normalization mode is invalid")
        if proposal.edit_arm not in _EDIT_ARMS:
            raise StudyError("proposal edit arm is invalid")
        if proposal.proposal_origin not in _PROPOSAL_ORIGINS:
            raise StudyError("proposal origin is invalid")
        explicit = (
            proposal.attention_enabled,
            proposal.attention_kernel_center,
            proposal.attention_kernel_half_width,
            proposal.attention_edge_strength,
            proposal.attention_peak_strength,
            proposal.mlp_enabled,
            proposal.mlp_kernel_center,
            proposal.mlp_kernel_half_width,
            proposal.mlp_edge_strength,
            proposal.mlp_peak_strength,
        )
        if any(item is None for item in explicit):
            raise StudyError("proposal writer kernels must be explicit")
        assert proposal.attention_enabled is not None and proposal.mlp_enabled is not None
        kernel_values = (
            proposal.attention_kernel_center, proposal.attention_kernel_half_width,
            proposal.attention_edge_strength, proposal.attention_peak_strength,
            proposal.mlp_kernel_center, proposal.mlp_kernel_half_width,
            proposal.mlp_edge_strength, proposal.mlp_peak_strength,
        )
        if any(not math.isfinite(cast(float, item)) for item in kernel_values):
            raise StudyError("proposal writer kernel values must be finite")
        for site, enabled, center, half_width, edge, peak in (
            ("attention", proposal.attention_enabled, proposal.attention_kernel_center,
             proposal.attention_kernel_half_width, proposal.attention_edge_strength,
             proposal.attention_peak_strength),
            ("mlp", proposal.mlp_enabled, proposal.mlp_kernel_center,
             proposal.mlp_kernel_half_width, proposal.mlp_edge_strength,
             proposal.mlp_peak_strength),
        ):
            assert center is not None and half_width is not None
            assert edge is not None and peak is not None
            if center < min(proposal.writer_layers) or center > max(proposal.writer_layers):
                raise StudyError(f"proposal {site} kernel center is outside writer layers")
            if half_width < 0.0 or not (0.0 <= edge <= peak <= 2.0):
                raise StudyError(f"proposal {site} kernel is invalid")
            if not enabled and (edge != 0.0 or peak != 0.0):
                raise StudyError(f"disabled proposal {site} writer must have zero strength")
        expected_policy = (
            "both" if proposal.attention_enabled and proposal.mlp_enabled
            else "attention" if proposal.attention_enabled
            else "mlp" if proposal.mlp_enabled else "both"
        )
        if proposal.writer_policy != expected_policy:
            raise StudyError("proposal legacy writer policy disagrees with explicit flags")
        if proposal.truth_direction_scope == "per_layer":
            if proposal.writer_layers != (proposal.source_layer,):
                raise StudyError("per-layer truth scope requires one source-matched writer layer")
            for enabled, center, width in (
                (proposal.attention_enabled, proposal.attention_kernel_center,
                 proposal.attention_kernel_half_width),
                (proposal.mlp_enabled, proposal.mlp_kernel_center,
                 proposal.mlp_kernel_half_width),
            ):
                if enabled and (center != proposal.source_layer or width != 0.0):
                    raise StudyError("per-layer truth scope requires a single-layer kernel")
        if proposal.refusal_direction_scope not in _DIRECTION_SCOPES:
            raise StudyError("proposal refusal direction scope is invalid")
        if proposal.refusal_writer_policy not in _REFUSAL_WRITER_POLICIES:
            raise StudyError("proposal refusal writer policy is invalid")
        if not 0.0 <= proposal.refusal_strength <= 2.0:
            raise StudyError("proposal refusal strength is outside [0, 2]")
        if proposal.refusal_enabled != (proposal.edit_arm in {"refusal_only", "joint"}):
            raise StudyError("proposal refusal_enabled disagrees with edit arm")
        if not proposal.refusal_enabled:
            if proposal.refusal_strength != 0.0 or proposal.refusal_source_layer is not None:
                raise StudyError("disabled refusal contribution must have zero strength and no source layer")
        elif proposal.refusal_direction_scope == "per_layer":
            if proposal.refusal_source_layer is not None:
                raise StudyError("per-layer refusal scope must not carry an inert source layer")
        elif proposal.refusal_source_layer is None or not (
            0 <= proposal.refusal_source_layer < self.direction_bank.model.decoder_layer_count
        ):
            raise StudyError("global refusal scope requires an in-range source layer")
        if proposal.edit_arm == "refusal_only" and (
            proposal.attention_enabled or proposal.mlp_enabled
        ):
            raise StudyError("refusal-only arm must disable truth writer kernels")
        if proposal.matched_basis_control == "orthogonal" and (
            proposal.edit_arm != "truth_only"
            or proposal.refusal_enabled
            or writer_configuration(proposal) == "disabled"
        ):
            raise StudyError(
                "orthogonal matched control requires an active persistent truth-only edit"
            )

    def _coverage(self, trials: Sequence[StudyTrial]) -> CoverageLedger:
        valid = [item for item in trials if item.result.outcome_kind != "operational_failure"]
        return _coverage_ledger(tuple(item.proposal for item in valid))

    def _coverage_complete(self, coverage: CoverageLedger) -> bool:
        return _broad_coverage_contract(
            self.config, self.directions
        ).is_complete(coverage)

    def _coverage_summary(
        self, coverage: CoverageLedger
    ) -> tuple[tuple[str, int, int], ...]:
        contract = _broad_coverage_contract(self.config, self.directions)
        strength_seen = set(coverage.strength_regions) | set(
            coverage.refusal_strength_regions
        )
        strength_required = set(contract.strength_regions) | set(
            contract.refusal_strength_regions
        )
        return (
            ("direction_family", len(coverage.families), len(contract.families)),
            ("layer_region", len(coverage.writer_regions), len(contract.writer_regions)),
            (
                "intervention_arm",
                len(coverage.active_edit_arms),
                len(contract.active_edit_arms),
            ),
            (
                "attention_mlp_configuration",
                len(coverage.writer_configurations),
                len(contract.writer_configurations),
            ),
            (
                "refusal_setting",
                len(coverage.refusal_settings),
                len(contract.refusal_settings),
            ),
            ("strength_range", len(strength_seen), len(strength_required)),
        )

    def _completed_batch_commit(
        self,
        *,
        identity: str,
        journal_path: Path,
        batch: Mapping[str, Any],
        batch_trials: Sequence[StudyTrial],
        completed_trials: Sequence[StudyTrial],
    ) -> CompletedBatchCommit:
        if len(batch_trials) != len(batch["trials"]):
            raise StudyError("completed-batch callback requires every batch result")
        persisted = json.loads(journal_path.read_text())
        journal_sha256 = _hash(
            persisted.get("journal_sha256"), "journal.journal_sha256"
        )
        coverage = self._coverage(completed_trials)
        batch_body = {
            "ordinal": batch["ordinal"],
            "trials": [item.to_dict() for item in batch_trials],
        }
        return CompletedBatchCommit(
            study_identity_sha256=identity,
            batch_ordinal=int(batch["ordinal"]),
            batch_size=len(batch_trials),
            completed_trials=len(completed_trials),
            journal_sha256=journal_sha256,
            batch_sha256=_sha(batch_body),
            trials=tuple(batch_trials),
            coverage=coverage,
            coverage_complete=self._coverage_complete(coverage),
            _coverage_counts=self._coverage_summary(coverage),
        )

    def _prepared_study_context(
        self,
        *,
        identity: str,
        journal_path: Path,
        completed_trials: Sequence[StudyTrial],
    ) -> PreparedStudyContext:
        persisted = json.loads(journal_path.read_text())
        journal_sha256 = _hash(
            persisted.get("journal_sha256"), "journal.journal_sha256"
        )
        coverage = self._coverage(completed_trials)
        return PreparedStudyContext(
            study_identity_sha256=identity,
            journal_sha256=journal_sha256,
            completed_trials=len(completed_trials),
            coverage=coverage,
            coverage_complete=self._coverage_complete(coverage),
            _coverage_counts=self._coverage_summary(coverage),
        )

    def _validate_evaluation_result(self, result: EvaluationResult) -> None:
        if not isinstance(result, EvaluationResult):
            raise StudyError("evaluator returned a non-EvaluationResult value")
        if result.outcome_kind not in {
            "successful", "scientifically_infeasible", "operational_failure"
        }:
            raise StudyError("evaluator returned an invalid outcome kind")
        if result.outcome_kind == "operational_failure":
            if result.metrics or not result.detail:
                raise StudyError("operational failures require detail and no metrics")
        else:
            if set(result.metrics) != set(self.config.objective_names):
                raise StudyError("scientific outcomes require every objective metric")
            if any(not math.isfinite(float(value)) for value in result.metrics.values()):
                raise StudyError("objective metrics must be finite")

    def _save(self, path: Path, raw: Mapping[str, Any]) -> None:
        raw = dict(raw)
        raw.pop("journal_sha256", None)
        raw["journal_sha256"] = _sha(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n")
        temporary.replace(path)

    def _load_trials(self, raw: Mapping[str, Any]) -> list[StudyTrial]:
        result: list[StudyTrial] = []
        for batch in raw["batches"]:
            for entry in batch["trials"]:
                result_raw = entry.get("result")
                if result_raw is None:
                    continue
                metrics = result_raw["metrics"]
                evaluation = EvaluationResult(result_raw["outcome_kind"], metrics, result_raw["detail"])
                proposal = SearchProposal.from_dict(entry["proposal"])
                self._validate_proposal(proposal)
                result.append(StudyTrial(
                    entry["trial_id"], entry["ordinal"], batch["ordinal"], entry["tier_name"],
                    tuple(entry["evaluation_record_ids"]), proposal, evaluation,
                ))
        return sorted(result, key=lambda item: item.ordinal)

    def _validate_journal_structure(
        self, journal: Mapping[str, Any], identity_inputs: Mapping[str, Any]
    ) -> None:
        _exact(
            journal,
            {
                "format", "study_identity_sha256", "identity_inputs",
                "batches", "journal_sha256",
            },
            "journal",
        )
        if journal["identity_inputs"] != identity_inputs:
            raise StudyError("journal identity inputs do not match the requested study")
        batches = journal["batches"]
        if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
            raise StudyError("journal.batches must be an array")
        maximum_batches = math.ceil(self.config.max_trials / self.config.batch_size)
        if len(batches) > maximum_batches:
            raise StudyError("journal contains more batches than max_trials")
        for batch_ordinal, batch in enumerate(batches):
            if not isinstance(batch, Mapping):
                raise StudyError(f"journal.batches[{batch_ordinal}] must be an object")
            _exact(batch, {"ordinal", "trials"}, f"journal.batches[{batch_ordinal}]")
            if batch["ordinal"] != batch_ordinal:
                raise StudyError("journal batch ordinals must be contiguous")
            entries = batch["trials"]
            if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
                raise StudyError("journal batch trials must be an array")
            expected_size = min(
                self.config.batch_size,
                self.config.max_trials - batch_ordinal * self.config.batch_size,
            )
            if len(entries) != expected_size:
                raise StudyError("journal batch size is invalid")
            for offset, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise StudyError("journal trial must be an object")
                _exact(
                    entry,
                    {
                        "trial_id", "ordinal", "tier_name", "evaluation_record_ids",
                        "proposal", "result",
                    },
                    "journal trial",
                )
                expected_ordinal = batch_ordinal * self.config.batch_size + offset
                if entry["ordinal"] != expected_ordinal or entry["trial_id"] != f"trial-{expected_ordinal:04d}":
                    raise StudyError("journal trial identity or ordering is invalid")
                if entry["result"] is None:
                    continue
                result = entry["result"]
                if not isinstance(result, Mapping):
                    raise StudyError("journal trial result must be an object or null")
                _exact(result, {"outcome_kind", "metrics", "detail"}, "journal trial result")
                if result["outcome_kind"] not in {
                    "successful", "scientifically_infeasible", "operational_failure"
                }:
                    raise StudyError("journal trial outcome kind is invalid")
                metrics = result["metrics"]
                if not isinstance(metrics, Mapping):
                    raise StudyError("journal trial metrics must be an object")
                if result["outcome_kind"] == "operational_failure":
                    if metrics or not isinstance(result["detail"], str) or not result["detail"]:
                        raise StudyError("journal operational failure is malformed")
                elif set(metrics) != set(self.config.objective_names) or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in metrics.values()
                ):
                    raise StudyError("journal scientific result metrics are malformed")

    def run(
        self,
        *,
        driver: SearchDriver,
        evaluator: Evaluator,
        journal_path: Path | str,
        stop_after_trials: int | None = None,
        batch_admission: BatchAdmission | None = None,
        after_complete_batch: AfterCompleteBatch | None = None,
        after_prepare_before_first_admission: (
            AfterPrepareBeforeFirstAdmission | None
        ) = None,
    ) -> StudyReport:
        target_trials = self.config.max_trials
        if stop_after_trials is not None:
            if (
                isinstance(stop_after_trials, bool)
                or not isinstance(stop_after_trials, int)
                or stop_after_trials <= 0
                or stop_after_trials > self.config.max_trials
                or (
                    stop_after_trials != self.config.max_trials
                    and stop_after_trials % self.config.batch_size != 0
                )
            ):
                raise StudyError(
                    "stop_after_trials must be a completed batch barrier within max_trials"
                )
            target_trials = stop_after_trials
        path = Path(journal_path)
        driver.prepare(
            self.config,
            self.directions,
            path.with_name(path.name + ".optuna.log"),
        )
        identity, identity_inputs = self._identity(driver, evaluator)
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise StudyError("journal is not a regular file")
            try:
                journal = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise StudyError("journal is unreadable") from error
            if journal.get("format") != STUDY_JOURNAL_FORMAT or journal.get("study_identity_sha256") != identity:
                raise StudyError("journal identity does not match the requested study")
            claimed_journal_hash = journal.get("journal_sha256")
            unsigned_journal = dict(journal)
            unsigned_journal.pop("journal_sha256", None)
            if claimed_journal_hash != _sha(unsigned_journal):
                raise StudyError("journal content identity mismatch")
            self._validate_journal_structure(journal, identity_inputs)
        else:
            journal = {
                "format": STUDY_JOURNAL_FORMAT, "study_identity_sha256": identity,
                "identity_inputs": identity_inputs, "batches": [],
            }
            self._save(path, journal)
            journal = json.loads(path.read_text())

        completed = self._load_trials(journal)
        if any(item.ordinal >= target_trials for item in completed):
            raise StudyError("journal already exceeds the requested stop boundary")
        completed_by_ordinal = {item.ordinal: item for item in completed}
        observed_batch_ordinals: set[int] = set()
        delivered_batch_ordinals: set[int] = set()
        incomplete_history_batch_ordinal: int | None = None
        # Replay only whole batches into the search adapter. A partially completed
        # batch remains invisible until all of its suggestions have outcomes.
        for batch in journal["batches"]:
            batch_ids = {entry["ordinal"] for entry in batch["trials"]}
            observed = [item for item in completed if item.ordinal in batch_ids]
            if len(observed) == len(batch["trials"]):
                driver.observe(observed)
                observed_batch_ordinals.add(batch["ordinal"])
                # The presence of a later admitted batch proves this callback
                # completed previously: admission is ordered after delivery.
                if batch["ordinal"] < len(journal["batches"]) - 1:
                    delivered_batch_ordinals.add(batch["ordinal"])
            elif batch["ordinal"] != len(journal["batches"]) - 1:
                raise StudyError("only the final journal batch may be incomplete")
            else:
                incomplete_history_batch_ordinal = batch["ordinal"]

        history_replay_completed = incomplete_history_batch_ordinal is None
        if history_replay_completed:
            _complete_driver_history_replay(driver)

        if after_prepare_before_first_admission is not None:
            committed_trials = tuple(
                item
                for item in completed
                if item.batch_ordinal in observed_batch_ordinals
            )
            after_prepare_before_first_admission(
                self._prepared_study_context(
                    identity=identity,
                    journal_path=path,
                    completed_trials=committed_trials,
                )
            )

        batch_ordinal = 0
        while batch_ordinal * self.config.batch_size < target_trials:
            start = batch_ordinal * self.config.batch_size
            size = min(self.config.batch_size, target_trials - start)
            existing_batch = (
                journal["batches"][batch_ordinal]
                if batch_ordinal < len(journal["batches"])
                else None
            )
            existing_batch_is_incomplete = (
                existing_batch is not None
                and any(entry["result"] is None for entry in existing_batch["trials"])
            )
            if (
                batch_admission is not None
                and (existing_batch is None or existing_batch_is_incomplete)
            ):
                committed = [
                    item
                    for item in completed
                    if item.batch_ordinal in observed_batch_ordinals
                ]
                coverage = self._coverage(committed)
                if not batch_admission.admit_batch(
                    completed_trials=len(committed),
                    batch_size=size,
                    coverage_complete=self._coverage_complete(coverage),
                    batch_started=(
                        existing_batch is not None
                        and any(
                            entry["result"] is not None
                            for entry in existing_batch["trials"]
                        )
                    ),
                ):
                    break
            if batch_ordinal == len(journal["batches"]):
                coverage = self._coverage(completed)
                entries = []
                for offset in range(size):
                    ordinal = start + offset
                    tier = self._tier(ordinal)
                    proposal = driver.suggest(
                        SearchRequest(ordinal, self.config, self.directions, coverage)
                    )
                    self._validate_proposal(proposal)
                    entries.append({
                        "trial_id": f"trial-{ordinal:04d}", "ordinal": ordinal,
                        "tier_name": tier.name,
                        "evaluation_record_ids": list(
                            self.config.validation_record_ids[:tier.record_limit]
                        ),
                        "proposal": proposal.to_dict(), "result": None,
                    })
                journal["batches"].append({"ordinal": batch_ordinal, "trials": entries})
                self._save(path, journal)
            batch = journal["batches"][batch_ordinal]
            pending_entries = [entry for entry in batch["trials"] if entry["result"] is None]
            pending_requests = []
            for entry in pending_entries:
                proposal = SearchProposal.from_dict(entry["proposal"])
                self._validate_proposal(proposal)
                pending_requests.append(BatchEvaluationRequest(
                    trial_id=entry["trial_id"],
                    ordinal=entry["ordinal"],
                    proposal=proposal,
                    record_ids=tuple(entry["evaluation_record_ids"]),
                    objective_names=self.config.objective_names,
                ))

            def evaluate_one(request: BatchEvaluationRequest[SearchProposal]) -> EvaluationResult:
                try:
                    if request.proposal.matched_basis_control == "orthogonal":
                        execution_identity_sha256 = _sha({
                            "study_identity_sha256": identity,
                            "trial_id": request.trial_id,
                            "proposal": request.proposal.to_dict(),
                            "control_kind": "orthogonal",
                        })
                        evaluate_control = getattr(
                            evaluator, "evaluate_matched_basis_control", None
                        )
                        if callable(evaluate_control):
                            return evaluate_control(
                                request.proposal,
                                trial_id=request.trial_id,
                                record_ids=request.record_ids,
                                objective_names=request.objective_names,
                                control_kind="orthogonal",
                                execution_identity_sha256=execution_identity_sha256,
                            )
                        return evaluator.evaluate(
                            request.proposal,
                            trial_id=request.trial_id,
                            record_ids=request.record_ids,
                            objective_names=request.objective_names,
                            control_kind="orthogonal",  # type: ignore[call-arg]
                            finalization_execution_identity_sha256=(  # type: ignore[call-arg]
                                execution_identity_sha256
                            ),
                        )
                    return evaluator.evaluate(
                        request.proposal,
                        trial_id=request.trial_id,
                        record_ids=request.record_ids,
                        objective_names=request.objective_names,
                    )
                except OperationalEvaluationError as error:
                    return EvaluationResult.operational_failure(str(error))

            entries_by_ordinal = {entry["ordinal"]: entry for entry in pending_entries}

            def accept_result(
                request: BatchEvaluationRequest[SearchProposal], result: EvaluationResult
            ) -> None:
                self._validate_evaluation_result(result)
                entry = entries_by_ordinal[request.ordinal]
                entry["result"] = {
                    "outcome_kind": result.outcome_kind,
                    "metrics": dict(result.metrics),
                    "detail": result.detail,
                }
                self._save(path, journal)
                completed_by_ordinal[request.ordinal] = StudyTrial(
                    entry["trial_id"],
                    entry["ordinal"],
                    batch["ordinal"],
                    entry["tier_name"],
                    tuple(entry["evaluation_record_ids"]),
                    request.proposal,
                    result,
                )

            try:
                execute_ordered_batch(
                    evaluator,
                    pending_requests,
                    evaluate_one=evaluate_one,
                    accept_result=accept_result,
                )
            except BatchExecutionError as error:
                raise StudyError(str(error)) from error
            except OperationalEvaluationError as error:
                for request in pending_requests:
                    if entries_by_ordinal[request.ordinal]["result"] is None:
                        accept_result(
                            request,
                            EvaluationResult.operational_failure(str(error)),
                        )
            batch_trials = [completed_by_ordinal[entry["ordinal"]] for entry in batch["trials"]]
            if batch_ordinal not in observed_batch_ordinals:
                driver.observe(batch_trials)
                observed_batch_ordinals.add(batch_ordinal)
            if (
                not history_replay_completed
                and batch_ordinal == incomplete_history_batch_ordinal
            ):
                _complete_driver_history_replay(driver)
                history_replay_completed = True
            completed = sorted(completed_by_ordinal.values(), key=lambda item: item.ordinal)
            if batch_ordinal not in delivered_batch_ordinals:
                callback = after_complete_batch
                if callback is None and batch_admission is not None:
                    legacy_commit = getattr(batch_admission, "commit_batch", None)
                    if callable(legacy_commit):
                        def callback(commit: CompletedBatchCommit) -> None:
                            legacy_commit(
                                completed_trials=commit.completed_trials,
                                coverage_complete=commit.coverage_complete,
                            )
                if callback is not None:
                    callback(self._completed_batch_commit(
                        identity=identity,
                        journal_path=path,
                        batch=batch,
                        batch_trials=batch_trials,
                        completed_trials=completed,
                    ))
                delivered_batch_ordinals.add(batch_ordinal)
            batch_ordinal += 1

        trials = tuple(sorted(completed_by_ordinal.values(), key=lambda item: item.ordinal))
        coverage = self._coverage(trials)
        return StudyReport(
            cast(Literal["truth_editing_study_report_v1"], STUDY_REPORT_FORMAT),
            identity,
            trials,
            coverage,
            self._coverage_complete(coverage),
        )
