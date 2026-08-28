"""Offline integration seam between truth-editing recipes and Heretic.

This module plans work; it does not load a model, start Optuna, call a judge,
or launch a GPU.  The important boundary is that an optimization plan contains
only the validation partition.  A final-test plan has a separate constructor
and requires an explicit freeze receipt, so accidentally feeding the test set
to an optimizer is a schema error rather than a convention.

The plan deliberately keeps semantic objectives and probe measurements in
different fields.  Probe AUROC, direction score movement, and similar values
are useful diagnostics, but cannot become the optimization reward through a
typo or an opaque weighted score.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .truth_editing_contracts import (
    DirectionBankManifest,
    InterventionRecipe,
    TruthEditingContractError,
    canonical_sha256,
    parse_direction_bank_manifest,
    parse_intervention_recipe,
    validate_recipe_compatibility,
)


HERETIC_CONFIG_FORMAT = "heretic_truth_editing_integration_config_v1"
HERETIC_PLAN_FORMAT = "heretic_truth_editing_trial_plan_v1"
HERETIC_TRIAL_FORMAT = "heretic_truth_editing_trial_v1"
HERETIC_BATCH_FORMAT = "heretic_truth_editing_batch_v1"

Phase = Literal["optimization", "final_test"]
DatasetSplit = Literal["train", "validation", "test"]

OBJECTIVES: tuple[str, ...] = (
    "valid_false_report_rate_lcb",
    "truth_report_dissociation_lcb",
    "capability_preservation_lcb",
)
DEFAULT_DIAGNOSTICS: tuple[str, ...] = (
    "probe_auroc",
    "probe_score_delta",
    "direction_projection_norm",
)
_D1_RE = re.compile(r"(?<![a-z0-9])d1(?![a-z0-9])", re.IGNORECASE)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_OBJECTIVE_SET = frozenset(OBJECTIVES)
_FORBIDDEN_OBJECTIVE_TERMS = (
    "probe",
    "auroc",
    "direction",
    "activation",
    "keyword",
)


class HereticIntegrationError(ValueError):
    """The Heretic adapter cannot prove a safe, phase-correct plan."""


def _exact(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise HereticIntegrationError(
            f"{name} fields differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise HereticIntegrationError(f"{name} must be a nonempty trimmed string")
    return value


def _sha(value: Any, name: str) -> str:
    value = _text(value, name)
    if not _SHA_RE.fullmatch(value):
        raise HereticIntegrationError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HereticIntegrationError(f"{name} must be an integer >= {minimum}")
    return value


def _strings(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HereticIntegrationError(f"{name} must be an array of strings")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise HereticIntegrationError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise HereticIntegrationError(f"{name} entries must be unique")
    return result


def _reject_d1(value: Any, name: str) -> None:
    """Reject D1 identifiers without rejecting ordinary words containing d1."""

    if isinstance(value, str) and _D1_RE.search(value):
        raise HereticIntegrationError(f"D1 is excluded from Heretic planning ({name})")


@dataclass(frozen=True)
class HereticRuntimeConfig:
    """The first-study runtime contract, represented without importing torch."""

    engine: Literal["transformers"] = "transformers"
    dtype: Literal["bfloat16"] = "bfloat16"
    attention_implementation: Literal["flash_attention_2"] = "flash_attention_2"
    quantization: None = None
    speculative_decoding: bool = False
    device_map: Literal["cuda:0"] = "cuda:0"
    use_cache: bool = True
    model_loads_per_worker: int = 1

    def __post_init__(self) -> None:
        if self.engine != "transformers":
            raise HereticIntegrationError("Heretic runtime engine must be transformers")
        if self.dtype != "bfloat16":
            raise HereticIntegrationError("frozen runtime dtype must be bfloat16")
        if self.attention_implementation != "flash_attention_2":
            raise HereticIntegrationError("frozen runtime must use FlashAttention-2")
        if self.quantization is not None:
            raise HereticIntegrationError("quantization is not part of the frozen runtime")
        if not isinstance(self.speculative_decoding, bool) or self.speculative_decoding:
            raise HereticIntegrationError("speculative decoding must be disabled")
        if self.device_map != "cuda:0":
            raise HereticIntegrationError("the frozen runtime requires explicit cuda:0")
        if self.use_cache is not True:
            raise HereticIntegrationError("the frozen runtime requires use_cache=true")
        if self.model_loads_per_worker != 1:
            raise HereticIntegrationError("a worker may load the model exactly once")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "dtype": self.dtype,
            "attention_implementation": self.attention_implementation,
            "quantization": self.quantization,
            "speculative_decoding": self.speculative_decoding,
            "device_map": self.device_map,
            "use_cache": self.use_cache,
            "model_loads_per_worker": self.model_loads_per_worker,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HereticRuntimeConfig:
        expected = {
            "engine",
            "dtype",
            "attention_implementation",
            "quantization",
            "speculative_decoding",
            "device_map",
            "use_cache",
            "model_loads_per_worker",
        }
        _exact(value, expected, "runtime")
        return cls(
            engine=value["engine"],
            dtype=value["dtype"],
            attention_implementation=value["attention_implementation"],
            quantization=value["quantization"],
            speculative_decoding=value["speculative_decoding"],
            device_map=value["device_map"],
            use_cache=value["use_cache"],
            model_loads_per_worker=value["model_loads_per_worker"],
        )


@dataclass(frozen=True)
class HereticStudyConfig:
    """Strict synchronous study settings shared by every trial batch."""

    study_id: str
    sampler: Literal["multivariate_tpe"] = "multivariate_tpe"
    scheduler: Literal["synchronous"] = "synchronous"
    batch_size: int = 8
    max_trials: int = 200
    startup_trials: int = 60
    sampler_seed: int = 0
    wall_clock_budget_seconds: int = 86400
    evaluation_split: Literal["validation"] = "validation"
    objective_names: tuple[str, ...] = OBJECTIVES
    diagnostic_metric_names: tuple[str, ...] = DEFAULT_DIAGNOSTICS
    runtime: HereticRuntimeConfig = HereticRuntimeConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _text(self.study_id, "study_id"))
        _reject_d1(self.study_id, "study_id")
        if self.sampler != "multivariate_tpe":
            raise HereticIntegrationError("the Heretic adapter requires multivariate_tpe")
        if self.scheduler != "synchronous":
            raise HereticIntegrationError("Heretic batches must be synchronous")
        for name, value in (
            ("batch_size", self.batch_size),
            ("max_trials", self.max_trials),
            ("startup_trials", self.startup_trials),
        ):
            _integer(value, name, minimum=1)
        _integer(self.wall_clock_budget_seconds, "wall_clock_budget_seconds", minimum=1)
        if self.wall_clock_budget_seconds > 86400:
            raise HereticIntegrationError("wall-clock budget must be at most 24 hours")
        if self.startup_trials > self.max_trials:
            raise HereticIntegrationError("startup_trials cannot exceed max_trials")
        if self.evaluation_split != "validation":
            raise HereticIntegrationError("optimization objectives may use validation only")
        if not isinstance(self.sampler_seed, int) or isinstance(self.sampler_seed, bool):
            raise HereticIntegrationError("sampler_seed must be an integer")
        objectives = _strings(self.objective_names, "objective_names")
        diagnostics = _strings(self.diagnostic_metric_names, "diagnostic_metric_names")
        if any(name not in _OBJECTIVE_SET for name in objectives):
            raise HereticIntegrationError(
                "objective_names must be the declared semantic objectives; "
                "probe and direction metrics are diagnostics only"
            )
        if set(objectives) & set(diagnostics):
            raise HereticIntegrationError("objective and diagnostic names must be disjoint")
        for name in diagnostics:
            if any(term in name.casefold() for term in _FORBIDDEN_OBJECTIVE_TERMS):
                continue
            # Non-probe diagnostics are allowed only when they are explicitly
            # named; this branch intentionally does not broaden the objective set.
        object.__setattr__(self, "objective_names", objectives)
        object.__setattr__(self, "diagnostic_metric_names", diagnostics)
        if not isinstance(self.runtime, HereticRuntimeConfig):
            raise HereticIntegrationError("runtime must be a HereticRuntimeConfig")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": HERETIC_CONFIG_FORMAT,
            "study_id": self.study_id,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "batch_size": self.batch_size,
            "max_trials": self.max_trials,
            "startup_trials": self.startup_trials,
            "sampler_seed": self.sampler_seed,
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "evaluation_split": self.evaluation_split,
            "objective_names": list(self.objective_names),
            "diagnostic_metric_names": list(self.diagnostic_metric_names),
            "runtime": self.runtime.to_dict(),
        }


def parse_study_config(value: Mapping[str, Any]) -> HereticStudyConfig:
    expected = {
        "format",
        "study_id",
        "sampler",
        "scheduler",
        "batch_size",
        "max_trials",
        "startup_trials",
        "sampler_seed",
        "wall_clock_budget_seconds",
        "evaluation_split",
        "objective_names",
        "diagnostic_metric_names",
        "runtime",
    }
    _exact(value, expected, "config")
    if value["format"] != HERETIC_CONFIG_FORMAT:
        raise HereticIntegrationError("unsupported Heretic config format")
    runtime = value["runtime"]
    if not isinstance(runtime, Mapping):
        raise HereticIntegrationError("runtime must be an object")
    return HereticStudyConfig(
        study_id=value["study_id"],
        sampler=value["sampler"],
        scheduler=value["scheduler"],
        batch_size=value["batch_size"],
        max_trials=value["max_trials"],
        startup_trials=value["startup_trials"],
        sampler_seed=value["sampler_seed"],
        wall_clock_budget_seconds=value["wall_clock_budget_seconds"],
        evaluation_split=value["evaluation_split"],
        objective_names=tuple(value["objective_names"]),
        diagnostic_metric_names=tuple(value["diagnostic_metric_names"]),
        runtime=HereticRuntimeConfig.from_mapping(runtime),
    )


def load_study_config(path: Path | str) -> HereticStudyConfig:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise HereticIntegrationError(f"config is not a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HereticIntegrationError("config is unreadable") from error
    if not isinstance(value, Mapping):
        raise HereticIntegrationError("config must be an object")
    return parse_study_config(value)


@dataclass(frozen=True)
class DatasetSplitIndex:
    """A minimal immutable dataset view used by the offline planner.

    The planner receives the requested partition lazily from a full dataset
    object or from this index.  It never enumerates the test partition while
    building an optimization plan.
    """

    dataset_id: str
    manifest_sha256: str
    split_record_ids: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _text(self.dataset_id, "dataset_id"))
        _reject_d1(self.dataset_id, "dataset_id")
        object.__setattr__(self, "manifest_sha256", _sha(self.manifest_sha256, "manifest_sha256"))
        normalized: dict[str, tuple[str, ...]] = {}
        for split, values in self.split_record_ids.items():
            if split not in {"train", "validation", "test", "quarantine"}:
                raise HereticIntegrationError(f"unsupported dataset split: {split}")
            ids = _strings(values, f"split_record_ids.{split}", allow_empty=True)
            for record_id in ids:
                _reject_d1(record_id, f"{split}.record_id")
            normalized[split] = ids
        object.__setattr__(self, "split_record_ids", normalized)

    def ids_for(self, split: DatasetSplit) -> tuple[str, ...]:
        if split not in self.split_record_ids:
            raise HereticIntegrationError(f"dataset has no {split} partition")
        return self.split_record_ids[split]


@dataclass(frozen=True)
class HereticTrial:
    format: Literal["heretic_truth_editing_trial_v1"]
    trial_id: str
    ordinal: int
    recipe_id: str
    condition_kind: str
    backend_type: str
    evaluation_split: Literal["validation", "test"]
    record_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "trial_id": self.trial_id,
            "ordinal": self.ordinal,
            "recipe_id": self.recipe_id,
            "condition_kind": self.condition_kind,
            "backend_type": self.backend_type,
            "evaluation_split": self.evaluation_split,
            "record_ids": list(self.record_ids),
        }


@dataclass(frozen=True)
class HereticBatch:
    format: Literal["heretic_truth_editing_batch_v1"]
    batch_id: str
    ordinal: int
    trial_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "batch_id": self.batch_id,
            "ordinal": self.ordinal,
            "trial_ids": list(self.trial_ids),
        }


@dataclass(frozen=True)
class HereticTrialPlan:
    format: Literal["heretic_truth_editing_trial_plan_v1"]
    study_id: str
    phase: Phase
    scheduler: Literal["synchronous"]
    batch_size: int
    dataset_id: str
    dataset_manifest_sha256: str
    direction_manifest_sha256: str
    evaluation_split: Literal["validation", "test"]
    objective_names: tuple[str, ...]
    diagnostic_metric_names: tuple[str, ...]
    trials: tuple[HereticTrial, ...]
    batches: tuple[HereticBatch, ...]
    test_record_ids: tuple[str, ...]
    freeze_receipt_sha256: str | None
    self_sha256: str

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "study_id": self.study_id,
            "phase": self.phase,
            "scheduler": self.scheduler,
            "batch_size": self.batch_size,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "direction_manifest_sha256": self.direction_manifest_sha256,
            "evaluation_split": self.evaluation_split,
            "objective_names": list(self.objective_names),
            "diagnostic_metric_names": list(self.diagnostic_metric_names),
            "trials": [item.to_dict() for item in self.trials],
            "batches": [item.to_dict() for item in self.batches],
            "test_record_ids": list(self.test_record_ids),
            "freeze_receipt_sha256": self.freeze_receipt_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "self_sha256": self.self_sha256}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)


def _recipe_and_manifest(
    recipes: Sequence[InterventionRecipe | Mapping[str, Any]],
    direction_manifest: DirectionBankManifest | Mapping[str, Any],
) -> tuple[tuple[InterventionRecipe, ...], DirectionBankManifest]:
    try:
        manifest = (
            direction_manifest
            if isinstance(direction_manifest, DirectionBankManifest)
            else parse_direction_bank_manifest(direction_manifest)
        )
    except (TruthEditingContractError, TypeError, KeyError) as error:
        raise HereticIntegrationError(f"direction manifest compatibility failed: {error}") from error
    _reject_d1(manifest.manifest_id, "direction manifest")
    for direction_id in manifest.direction_ids:
        _reject_d1(direction_id, "direction ID")
    parsed: list[InterventionRecipe] = []
    seen: set[str] = set()
    for index, value in enumerate(recipes):
        try:
            recipe = value if isinstance(value, InterventionRecipe) else parse_intervention_recipe(value)
            validate_recipe_compatibility(recipe, manifest)
        except (TruthEditingContractError, TypeError, KeyError) as error:
            raise HereticIntegrationError(
                f"recipe {index} compatibility failed: {error}"
            ) from error
        _reject_d1(recipe.recipe_id, f"recipe {index}")
        if recipe.recipe_id in seen:
            raise HereticIntegrationError(f"recipe_id is duplicated: {recipe.recipe_id}")
        seen.add(recipe.recipe_id)
        parsed.append(recipe)
    if not parsed:
        raise HereticIntegrationError("at least one recipe is required")
    return tuple(parsed), manifest


def _record_id(value: Any, name: str, expected_split: str) -> str:
    if isinstance(value, Mapping):
        if "split" in value and value["split"] != expected_split:
            raise HereticIntegrationError(
                f"{name} declares split {value['split']!r}, expected {expected_split!r}"
            )
        value = value.get("record_id")
    elif hasattr(value, "record_id"):
        declared = getattr(value, "split", expected_split)
        if declared != expected_split:
            raise HereticIntegrationError(
                f"{name} declares split {declared!r}, expected {expected_split!r}"
            )
        value = getattr(value, "record_id")
    return _text(value, f"{name}.record_id")


def _partition_ids(dataset: Any, split: DatasetSplit) -> tuple[str, ...]:
    """Read exactly one partition from a dataset source."""

    if isinstance(dataset, DatasetSplitIndex):
        values: Iterable[Any] = dataset.ids_for(split)
    elif isinstance(dataset, Mapping):
        if split not in dataset:
            raise HereticIntegrationError(f"dataset has no {split} partition")
        values = dataset[split]
    elif hasattr(dataset, "iter_split"):
        try:
            values = dataset.iter_split(split)
        except Exception as error:
            raise HereticIntegrationError(f"cannot read {split} partition") from error
    else:
        raise HereticIntegrationError("dataset must expose one named split or iter_split")
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise HereticIntegrationError(f"dataset {split} partition must be iterable")
    result = tuple(_record_id(value, f"{split}[{index}]", split) for index, value in enumerate(values))
    if not result:
        raise HereticIntegrationError(f"dataset {split} partition must not be empty")
    if len(set(result)) != len(result):
        raise HereticIntegrationError(f"duplicate record_id in {split} partition")
    for record_id in result:
        _reject_d1(record_id, f"{split}.record_id")
    return result


def _dataset_identity(dataset: Any) -> tuple[str, str]:
    if isinstance(dataset, DatasetSplitIndex):
        return dataset.dataset_id, dataset.manifest_sha256
    if isinstance(dataset, Mapping):
        # Mapping inputs are useful for offline fixtures and simple adapters;
        # metadata is deliberately explicit rather than inferred from row
        # content.  ``_partition_ids`` still reads only the requested split.
        dataset_id = dataset.get("dataset_id")
        manifest_sha256 = dataset.get("manifest_sha256")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise HereticIntegrationError("dataset identity is required")
        if not isinstance(manifest_sha256, str) or not _SHA_RE.fullmatch(manifest_sha256):
            raise HereticIntegrationError("dataset manifest_sha256 is required")
        return dataset_id, manifest_sha256
    request = getattr(dataset, "request", None)
    manifest = getattr(dataset, "manifest", None)
    dataset_id = getattr(request, "dataset_id", None) or (manifest or {}).get("dataset_id")
    manifest_sha256 = (manifest or {}).get("manifest_sha256") if isinstance(manifest, Mapping) else None
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise HereticIntegrationError("dataset identity is required")
    if not isinstance(manifest_sha256, str) or not _SHA_RE.fullmatch(manifest_sha256):
        raise HereticIntegrationError("dataset manifest_sha256 is required")
    return dataset_id, manifest_sha256


def _direction_manifest_hash(manifest: DirectionBankManifest) -> str:
    return manifest.self_sha256


def build_trial_plan(
    recipes: Sequence[InterventionRecipe | Mapping[str, Any]],
    *,
    dataset: Any,
    direction_manifest: DirectionBankManifest | Mapping[str, Any],
    config: HereticStudyConfig,
    phase: Phase = "optimization",
    frozen: bool = False,
    freeze_receipt_sha256: str | None = None,
) -> HereticTrialPlan:
    """Build deterministic synchronous batches for one study phase.

    ``phase="optimization"`` can only read validation.  ``phase="final_test"``
    is a separate, non-optimizing export and requires ``frozen=True`` plus a
    content-addressed freeze receipt.
    """

    if not isinstance(config, HereticStudyConfig):
        raise HereticIntegrationError("config must be a HereticStudyConfig")
    if phase not in {"optimization", "final_test"}:
        raise HereticIntegrationError(f"unsupported plan phase: {phase}")
    if phase == "optimization":
        if frozen or freeze_receipt_sha256 is not None:
            raise HereticIntegrationError("optimization plans cannot carry a freeze receipt")
        split: DatasetSplit = "validation"
        objectives = config.objective_names
    else:
        if not frozen:
            raise HereticIntegrationError("final-test planning requires a frozen study")
        if freeze_receipt_sha256 is None:
            raise HereticIntegrationError("final-test planning requires a freeze receipt")
        _sha(freeze_receipt_sha256, "freeze_receipt_sha256")
        split = "test"
        objectives = ()

    parsed, manifest = _recipe_and_manifest(recipes, direction_manifest)
    dataset_id, dataset_hash = _dataset_identity(dataset)
    _reject_d1(dataset_id, "dataset_id")
    record_ids = _partition_ids(dataset, split)
    trials = tuple(
        HereticTrial(
            format=cast(Literal["heretic_truth_editing_trial_v1"], HERETIC_TRIAL_FORMAT),
            trial_id=f"trial-{index:04d}",
            ordinal=index,
            recipe_id=recipe.recipe_id,
            condition_kind=recipe.condition_kind,
            backend_type=recipe.backend.type,
            evaluation_split=cast(Literal["validation", "test"], split),
            record_ids=record_ids,
        )
        for index, recipe in enumerate(parsed)
    )
    batches = tuple(
        HereticBatch(
            format=cast(Literal["heretic_truth_editing_batch_v1"], HERETIC_BATCH_FORMAT),
            batch_id=f"batch-{index:04d}",
            ordinal=index,
            trial_ids=tuple(item.trial_id for item in trials[start : start + config.batch_size]),
        )
        for index, start in enumerate(range(0, len(trials), config.batch_size))
    )
    body = {
        "format": HERETIC_PLAN_FORMAT,
        "study_id": config.study_id,
        "phase": phase,
        "scheduler": config.scheduler,
        "batch_size": config.batch_size,
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": dataset_hash,
        "direction_manifest_sha256": _direction_manifest_hash(manifest),
        "evaluation_split": split,
        "objective_names": list(objectives),
        "diagnostic_metric_names": list(config.diagnostic_metric_names),
        "trials": [item.to_dict() for item in trials],
        "batches": [item.to_dict() for item in batches],
        "test_record_ids": list(record_ids) if phase == "final_test" else [],
        "freeze_receipt_sha256": freeze_receipt_sha256,
    }
    return HereticTrialPlan(
        format=cast(Literal["heretic_truth_editing_trial_plan_v1"], HERETIC_PLAN_FORMAT),
        study_id=config.study_id,
        phase=phase,
        scheduler="synchronous",
        batch_size=config.batch_size,
        dataset_id=dataset_id,
        dataset_manifest_sha256=dataset_hash,
        direction_manifest_sha256=manifest.self_sha256,
        evaluation_split=cast(Literal["validation", "test"], split),
        objective_names=tuple(objectives),
        diagnostic_metric_names=config.diagnostic_metric_names,
        trials=trials,
        batches=batches,
        test_record_ids=record_ids if phase == "final_test" else (),
        freeze_receipt_sha256=freeze_receipt_sha256,
        self_sha256=canonical_sha256(body),
    )


def _parse_trial(value: Any, index: int) -> HereticTrial:
    if not isinstance(value, Mapping):
        raise HereticIntegrationError(f"trials[{index}] must be an object")
    expected = {
        "format",
        "trial_id",
        "ordinal",
        "recipe_id",
        "condition_kind",
        "backend_type",
        "evaluation_split",
        "record_ids",
    }
    _exact(value, expected, f"trials[{index}]")
    if value["format"] != HERETIC_TRIAL_FORMAT:
        raise HereticIntegrationError("unsupported trial format")
    split = value["evaluation_split"]
    if split not in {"validation", "test"}:
        raise HereticIntegrationError("trial evaluation_split is invalid")
    return HereticTrial(
        format=cast(Literal["heretic_truth_editing_trial_v1"], HERETIC_TRIAL_FORMAT),
        trial_id=_text(value["trial_id"], f"trials[{index}].trial_id"),
        ordinal=_integer(value["ordinal"], f"trials[{index}].ordinal"),
        recipe_id=_text(value["recipe_id"], f"trials[{index}].recipe_id"),
        condition_kind=_text(value["condition_kind"], f"trials[{index}].condition_kind"),
        backend_type=_text(value["backend_type"], f"trials[{index}].backend_type"),
        evaluation_split=split,
        record_ids=_strings(value["record_ids"], f"trials[{index}].record_ids"),
    )


def _parse_batch(value: Any, index: int) -> HereticBatch:
    if not isinstance(value, Mapping):
        raise HereticIntegrationError(f"batches[{index}] must be an object")
    expected = {"format", "batch_id", "ordinal", "trial_ids"}
    _exact(value, expected, f"batches[{index}]")
    if value["format"] != HERETIC_BATCH_FORMAT:
        raise HereticIntegrationError("unsupported batch format")
    return HereticBatch(
        format=cast(Literal["heretic_truth_editing_batch_v1"], HERETIC_BATCH_FORMAT),
        batch_id=_text(value["batch_id"], f"batches[{index}].batch_id"),
        ordinal=_integer(value["ordinal"], f"batches[{index}].ordinal"),
        trial_ids=_strings(value["trial_ids"], f"batches[{index}].trial_ids"),
    )


def parse_trial_plan(value: Mapping[str, Any]) -> HereticTrialPlan:
    expected = {
        "format",
        "study_id",
        "phase",
        "scheduler",
        "batch_size",
        "dataset_id",
        "dataset_manifest_sha256",
        "direction_manifest_sha256",
        "evaluation_split",
        "objective_names",
        "diagnostic_metric_names",
        "trials",
        "batches",
        "test_record_ids",
        "freeze_receipt_sha256",
        "self_sha256",
    }
    if not isinstance(value, Mapping):
        raise HereticIntegrationError("plan must be an object")
    _exact(value, expected, "plan")
    if value["format"] != HERETIC_PLAN_FORMAT:
        raise HereticIntegrationError("unsupported Heretic plan format")
    phase = value["phase"]
    split = value["evaluation_split"]
    if phase not in {"optimization", "final_test"}:
        raise HereticIntegrationError("plan phase is invalid")
    expected_split = "validation" if phase == "optimization" else "test"
    if split != expected_split:
        raise HereticIntegrationError("plan phase and evaluation split differ")
    scheduler = value["scheduler"]
    if scheduler != "synchronous":
        raise HereticIntegrationError("only synchronous plans are supported")
    objectives = _strings(value["objective_names"], "plan.objective_names", allow_empty=True)
    diagnostics = _strings(value["diagnostic_metric_names"], "plan.diagnostic_metric_names")
    if any(item not in _OBJECTIVE_SET for item in objectives):
        raise HereticIntegrationError("plan contains a non-semantic optimization objective")
    if set(objectives) & set(diagnostics):
        raise HereticIntegrationError("plan objective/diagnostic names overlap")
    trial_values = value["trials"]
    batch_values = value["batches"]
    if isinstance(trial_values, (str, bytes)) or not isinstance(trial_values, Sequence):
        raise HereticIntegrationError("plan.trials must be an array")
    if isinstance(batch_values, (str, bytes)) or not isinstance(batch_values, Sequence):
        raise HereticIntegrationError("plan.batches must be an array")
    trials = tuple(_parse_trial(item, index) for index, item in enumerate(trial_values))
    batches = tuple(_parse_batch(item, index) for index, item in enumerate(batch_values))
    if not trials or not batches:
        raise HereticIntegrationError("plan must contain trials and batches")
    if any(item.evaluation_split != expected_split for item in trials):
        raise HereticIntegrationError("trial split is not allowed for this plan phase")
    if any(item.ordinal != index for index, item in enumerate(trials)):
        raise HereticIntegrationError("trial ordinals must be contiguous and ordered")
    if any(item.ordinal != index for index, item in enumerate(batches)):
        raise HereticIntegrationError("batch ordinals must be contiguous and ordered")
    batch_size = _integer(value["batch_size"], "plan.batch_size", minimum=1)
    if any(len(item.trial_ids) > batch_size for item in batches):
        raise HereticIntegrationError("a batch exceeds the declared batch_size")
    if len({item.trial_id for item in trials}) != len(trials):
        raise HereticIntegrationError("plan trial IDs are not unique")
    trial_ids = tuple(item.trial_id for item in trials)
    batch_trial_ids = tuple(item for batch in batches for item in batch.trial_ids)
    if batch_trial_ids != trial_ids:
        raise HereticIntegrationError("batches must cover trials exactly once and in order")
    test_ids = _strings(value["test_record_ids"], "plan.test_record_ids", allow_empty=True)
    if phase == "optimization" and (test_ids or value["freeze_receipt_sha256"] is not None or objectives == ()):
        raise HereticIntegrationError("optimization plan cannot expose final-test records")
    freeze = value["freeze_receipt_sha256"]
    if phase == "final_test":
        if not isinstance(freeze, str) or not _SHA_RE.fullmatch(freeze):
            raise HereticIntegrationError("final-test plan requires a valid freeze receipt")
        if not test_ids:
            raise HereticIntegrationError("final-test plan requires test record IDs")
        if objectives:
            raise HereticIntegrationError("final-test plans cannot contain optimizer objectives")
    for record_id in test_ids:
        _reject_d1(record_id, "plan.test_record_ids")
    for trial in trials:
        _reject_d1(trial.recipe_id, f"trial {trial.trial_id}.recipe_id")
        _reject_d1(trial.condition_kind, f"trial {trial.trial_id}.condition_kind")
        for record_id in trial.record_ids:
            _reject_d1(record_id, f"trial {trial.trial_id}.record_id")
        if phase == "final_test" and tuple(trial.record_ids) != tuple(test_ids):
            raise HereticIntegrationError("final-test trial records differ from test_record_ids")
        if phase == "optimization" and any(record_id in test_ids for record_id in trial.record_ids):
            raise HereticIntegrationError("optimization trial references a test record")
    dataset_id = _text(value["dataset_id"], "plan.dataset_id")
    _reject_d1(dataset_id, "plan.dataset_id")
    body = dict(value)
    claimed = body.pop("self_sha256")
    if not isinstance(claimed, str) or canonical_sha256(body) != claimed:
        raise HereticIntegrationError("plan self hash mismatch")
    return HereticTrialPlan(
        format=cast(Literal["heretic_truth_editing_trial_plan_v1"], HERETIC_PLAN_FORMAT),
        study_id=_text(value["study_id"], "plan.study_id"),
        phase=phase,
        scheduler=scheduler,
        batch_size=batch_size,
        dataset_id=dataset_id,
        dataset_manifest_sha256=_sha(value["dataset_manifest_sha256"], "plan.dataset_manifest_sha256"),
        direction_manifest_sha256=_sha(value["direction_manifest_sha256"], "plan.direction_manifest_sha256"),
        evaluation_split=split,
        objective_names=objectives,
        diagnostic_metric_names=diagnostics,
        trials=trials,
        batches=batches,
        test_record_ids=test_ids,
        freeze_receipt_sha256=freeze,
        self_sha256=claimed,
    )


__all__ = [
    "DEFAULT_DIAGNOSTICS",
    "HERETIC_CONFIG_FORMAT",
    "HERETIC_PLAN_FORMAT",
    "DatasetSplitIndex",
    "HereticBatch",
    "HereticIntegrationError",
    "HereticRuntimeConfig",
    "HereticStudyConfig",
    "HereticTrial",
    "HereticTrialPlan",
    "build_trial_plan",
    "load_study_config",
    "parse_study_config",
    "parse_trial_plan",
]
