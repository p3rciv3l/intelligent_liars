"""One composition root for an executable truth-editing study.

The engine deliberately owns orchestration, not research arithmetic.  Model
mutation, response generation, judging, preservation scoring, search, and
artifact storage are injected adapters.  This keeps the command-line surface
stable while those deep modules remain independently testable.

Two phase boundaries are enforced here:

* an optimizer can evaluate the ``validation`` split only; and
* the ``test`` split is available only after a completed search result has
  been frozen by the artifact adapter.

Adapters may use GPUs, providers, or S3 in production.  The engine itself does
none of those things and its tests use only in-memory implementations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from .heretic_truth_editing import OBJECTIVES
from .truth_editing_contracts import (
    DirectionBankManifest,
    InterventionRecipe,
    parse_direction_bank_manifest,
    parse_intervention_recipe,
    validate_recipe_compatibility,
)
from .truth_editing_dataset import TruthEditingDataset


ENGINE_CONFIG_FORMAT = "truth_editing_engine_config_v1"
ENGINE_EVALUATION_FORMAT = "truth_editing_engine_evaluation_v1"
ENGINE_SEARCH_FORMAT = "truth_editing_engine_search_result_v1"
ENGINE_RUN_FORMAT = "truth_editing_engine_run_receipt_v1"

Split = Literal["validation", "test"]


class TruthEditingEngineError(ValueError):
    """The study cannot be composed or its receipts cannot be trusted."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TruthEditingEngineError("adapter receipt is not canonical JSON") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TruthEditingEngineError(f"{label} must be an object")
    result = dict(value)
    _canonical_json(result)
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TruthEditingEngineError(f"{label} must be a nonempty trimmed string")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TruthEditingEngineError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TruthEditingEngineError(f"{label} is unreadable") from error
    return _mapping(value, label)


@dataclass(frozen=True)
class EngineConfig:
    """Paths and immutable identity for one composed study."""

    study_id: str
    dataset_manifest: Path
    direction_bank_manifest: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _text(self.study_id, "study_id"))
        object.__setattr__(self, "dataset_manifest", Path(self.dataset_manifest))
        object.__setattr__(self, "direction_bank_manifest", Path(self.direction_bank_manifest))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, relative_to: Path) -> EngineConfig:
        expected = {
            "format",
            "study_id",
            "dataset_manifest",
            "direction_bank_manifest",
        }
        if set(value) != expected:
            raise TruthEditingEngineError(
                "engine config fields differ; "
                f"missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        if value["format"] != ENGINE_CONFIG_FORMAT:
            raise TruthEditingEngineError("unsupported engine config format")

        def resolve(raw: Any, label: str) -> Path:
            text = _text(raw, label)
            path = Path(text)
            return path if path.is_absolute() else relative_to / path

        return cls(
            study_id=value["study_id"],
            dataset_manifest=resolve(value["dataset_manifest"], "dataset_manifest"),
            direction_bank_manifest=resolve(
                value["direction_bank_manifest"], "direction_bank_manifest"
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": ENGINE_CONFIG_FORMAT,
            "study_id": self.study_id,
            "dataset_manifest": str(self.dataset_manifest),
            "direction_bank_manifest": str(self.direction_bank_manifest),
        }


class RuntimeAdapter(Protocol):
    """Apply one recipe and generate split-bound model evidence."""

    def evaluate(
        self,
        recipe: InterventionRecipe,
        records: Sequence[Any],
        *,
        split: Split,
    ) -> Mapping[str, Any]: ...


class JudgeAdapter(Protocol):
    """Convert runtime evidence into declared semantic objectives."""

    def score(
        self,
        runtime_receipt: Mapping[str, Any],
        records: Sequence[Any],
        *,
        split: Split,
    ) -> Mapping[str, Any]: ...


class SearchAdapter(Protocol):
    """Search recipes using only the supplied validation objective."""

    def run(
        self,
        *,
        study_id: str,
        direction_bank: DirectionBankManifest,
        evaluate: Callable[[InterventionRecipe], EvaluationReceipt],
    ) -> SearchResult: ...


class ArtifactAdapter(Protocol):
    """Freeze the completed search and return a durable identity receipt."""

    def freeze(self, result: SearchResult) -> Mapping[str, Any]: ...


class DatasetView(Protocol):
    """The narrow dataset surface needed by the engine."""

    @property
    def manifest_sha256(self) -> str: ...

    def records_for_split(self, split: Split) -> Sequence[Any]: ...


@dataclass(frozen=True)
class _DatasetV1View:
    dataset: TruthEditingDataset

    @property
    def manifest_sha256(self) -> str:
        manifest = self.dataset.manifest
        if not isinstance(manifest, Mapping):
            raise TruthEditingEngineError("run requires a materialized dataset manifest")
        identity = manifest.get("manifest_sha256")
        if not isinstance(identity, str):
            raise TruthEditingEngineError("dataset manifest identity is invalid")
        return _verified_sha(identity, "dataset manifest identity")

    def records_for_split(self, split: Split) -> Sequence[Any]:
        return tuple(self.dataset.iter_split(split))


@dataclass(frozen=True)
class _DatasetV2View:
    dataset: Any

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.dataset.manifest.to_payload())

    def records_for_split(self, split: Split) -> Sequence[Any]:
        return tuple(row for row in self.dataset.records if row.get("split") == split)


def _verified_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TruthEditingEngineError(f"{label} must be a lowercase SHA-256")
    return value


def _open_dataset(path: Path) -> DatasetView:
    """Open the v1 or canonical-QA-v2 dataset without adapter guesswork."""

    manifest_path = path / "manifest.json" if path.is_dir() else path
    raw = _load_json_object(manifest_path, "dataset manifest")
    dataset_format = raw.get("format")
    if dataset_format == "truth_editing_dataset_v1":
        return _DatasetV1View(TruthEditingDataset.open(manifest_path))
    if dataset_format == "truth_editing_canonical_qa_manifest_v2":
        from .truth_editing_dataset_v2 import TruthEditingDatasetV2

        return _DatasetV2View(TruthEditingDatasetV2.open(manifest_path.parent))
    raise TruthEditingEngineError(f"unsupported dataset format: {dataset_format!r}")


@dataclass(frozen=True)
class EvaluationReceipt:
    recipe: InterventionRecipe
    split: Split
    objectives: Mapping[str, float]
    runtime_receipt: Mapping[str, Any]
    judge_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.split not in ("validation", "test"):
            raise TruthEditingEngineError("evaluation split must be validation or test")
        expected = set(OBJECTIVES)
        if set(self.objectives) != expected:
            raise TruthEditingEngineError(
                "judge objectives differ from the declared semantic objectives"
            )
        normalized: dict[str, float] = {}
        for name in OBJECTIVES:
            raw = self.objectives[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TruthEditingEngineError(f"objective {name} must be numeric")
            number = float(raw)
            if not math.isfinite(number):
                raise TruthEditingEngineError(f"objective {name} must be finite")
            normalized[name] = number
        object.__setattr__(self, "objectives", normalized)
        object.__setattr__(
            self, "runtime_receipt", _mapping(self.runtime_receipt, "runtime_receipt")
        )
        object.__setattr__(
            self, "judge_receipt", _mapping(self.judge_receipt, "judge_receipt")
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": ENGINE_EVALUATION_FORMAT,
            "split": self.split,
            "recipe": self.recipe.to_dict(),
            "objectives": dict(self.objectives),
            "runtime_receipt": dict(self.runtime_receipt),
            "judge_receipt": dict(self.judge_receipt),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_mapping())


@dataclass(frozen=True)
class SearchResult:
    """The search adapter's complete, auditable result."""

    best_recipe: InterventionRecipe
    best_evaluation: EvaluationReceipt
    trial_evaluation_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.best_evaluation.split != "validation":
            raise TruthEditingEngineError("search best evaluation must use validation")
        if self.best_recipe.to_dict() != self.best_evaluation.recipe.to_dict():
            raise TruthEditingEngineError("search best recipe/evaluation mismatch")
        identities = tuple(self.trial_evaluation_sha256)
        if not identities or self.best_evaluation.identity_sha256 not in identities:
            raise TruthEditingEngineError("search trials do not contain the best evaluation")
        if len(set(identities)) != len(identities):
            raise TruthEditingEngineError("search trial identities must be unique")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in identities
        ):
            raise TruthEditingEngineError("search trial identities must be lowercase SHA-256")
        object.__setattr__(self, "trial_evaluation_sha256", identities)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": ENGINE_SEARCH_FORMAT,
            "best_recipe": self.best_recipe.to_dict(),
            "best_evaluation": self.best_evaluation.to_mapping(),
            "trial_evaluation_sha256": list(self.trial_evaluation_sha256),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_mapping())


@dataclass(frozen=True)
class RunReceipt:
    study_id: str
    dataset_manifest_sha256: str
    direction_bank_manifest_sha256: str
    search_result: SearchResult
    artifact_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _text(self.study_id, "study_id"))
        object.__setattr__(
            self,
            "dataset_manifest_sha256",
            _verified_sha(self.dataset_manifest_sha256, "dataset_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "direction_bank_manifest_sha256",
            _verified_sha(
                self.direction_bank_manifest_sha256,
                "direction_bank_manifest_sha256",
            ),
        )
        object.__setattr__(
            self, "artifact_receipt", _mapping(self.artifact_receipt, "artifact_receipt")
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": ENGINE_RUN_FORMAT,
            "study_id": self.study_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "direction_bank_manifest_sha256": self.direction_bank_manifest_sha256,
            "search_result": self.search_result.to_mapping(),
            "artifact_receipt": dict(self.artifact_receipt),
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_mapping())


class TruthEditingEngine:
    """Stable one-command facade over runtime, judge, search, and storage."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        dataset: DatasetView,
        direction_bank: DirectionBankManifest,
        runtime: RuntimeAdapter,
        judge: JudgeAdapter,
        search: SearchAdapter,
        artifacts: ArtifactAdapter,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.direction_bank = direction_bank
        self.runtime = runtime
        self.judge = judge
        self.search = search
        self.artifacts = artifacts
        self._evaluations: dict[str, EvaluationReceipt] = {}
        self._completed_search: SearchResult | None = None
        self._artifact_receipt: dict[str, Any] | None = None

    @classmethod
    def open(
        cls,
        config_path: Path | str,
        *,
        runtime: RuntimeAdapter,
        judge: JudgeAdapter,
        search: SearchAdapter,
        artifacts: ArtifactAdapter,
    ) -> TruthEditingEngine:
        path = Path(config_path)
        config = EngineConfig.from_mapping(
            _load_json_object(path, "engine config"), relative_to=path.parent
        )
        try:
            dataset = _open_dataset(config.dataset_manifest)
            direction_payload = _load_json_object(
                config.direction_bank_manifest, "direction bank manifest"
            )
            direction_bank = parse_direction_bank_manifest(direction_payload)
        except Exception as error:
            if isinstance(error, TruthEditingEngineError):
                raise
            raise TruthEditingEngineError("engine inputs failed strict validation") from error
        return cls(
            config=config,
            dataset=dataset,
            direction_bank=direction_bank,
            runtime=runtime,
            judge=judge,
            search=search,
            artifacts=artifacts,
        )

    def evaluate(self, recipe: InterventionRecipe, *, split: Split = "validation") -> EvaluationReceipt:
        """Evaluate one recipe; test remains sealed until search is frozen."""

        if split not in ("validation", "test"):
            raise TruthEditingEngineError("evaluation split must be validation or test")
        if split == "test" and self._artifact_receipt is None:
            raise TruthEditingEngineError("test evaluation requires a frozen search artifact")
        try:
            validate_recipe_compatibility(recipe, self.direction_bank)
        except Exception as error:
            raise TruthEditingEngineError("recipe is incompatible with the direction bank") from error
        records = tuple(self.dataset.records_for_split(split))
        if not records:
            raise TruthEditingEngineError(f"dataset {split} split is empty")
        runtime_receipt = _mapping(
            self.runtime.evaluate(recipe, records, split=split), "runtime_receipt"
        )
        judge_payload = _mapping(
            self.judge.score(runtime_receipt, records, split=split), "judge_receipt"
        )
        objectives = judge_payload.get("objectives")
        if not isinstance(objectives, Mapping):
            raise TruthEditingEngineError("judge receipt must contain objectives")
        receipt = EvaluationReceipt(
            recipe=recipe,
            split=split,
            objectives=dict(objectives),
            runtime_receipt=runtime_receipt,
            judge_receipt=judge_payload,
        )
        self._evaluations[receipt.identity_sha256] = receipt
        return receipt

    def run(self) -> RunReceipt:
        """Run optimization, verify its receipts, and freeze the winner."""

        if self._completed_search is not None:
            raise TruthEditingEngineError("this engine instance has already run")

        def objective(recipe: InterventionRecipe) -> EvaluationReceipt:
            return self.evaluate(recipe, split="validation")

        result = self.search.run(
            study_id=self.config.study_id,
            direction_bank=self.direction_bank,
            evaluate=objective,
        )
        if not isinstance(result, SearchResult):
            raise TruthEditingEngineError("search adapter must return SearchResult")
        declared = set(result.trial_evaluation_sha256)
        observed = set(self._evaluations)
        if declared != observed:
            if declared - observed:
                raise TruthEditingEngineError(
                    "search result contains unevaluated trial identities"
                )
            raise TruthEditingEngineError(
                "search result omits validation evaluations from its receipt"
            )
        artifact_receipt = _mapping(self.artifacts.freeze(result), "artifact_receipt")
        if artifact_receipt.get("search_result_sha256") != result.identity_sha256:
            raise TruthEditingEngineError("artifact receipt does not bind the search result")
        self._completed_search = result
        self._artifact_receipt = artifact_receipt
        return RunReceipt(
            study_id=self.config.study_id,
            dataset_manifest_sha256=self.dataset.manifest_sha256,
            direction_bank_manifest_sha256=self.direction_bank.self_sha256,
            search_result=result,
            artifact_receipt=artifact_receipt,
        )

def parse_recipe_file(path: Path | str) -> InterventionRecipe:
    """Load one strict recipe for the CLI without weakening its schema."""

    return parse_intervention_recipe(_load_json_object(Path(path), "intervention recipe"))


__all__ = [
    "ArtifactAdapter",
    "DatasetView",
    "ENGINE_CONFIG_FORMAT",
    "ENGINE_EVALUATION_FORMAT",
    "ENGINE_RUN_FORMAT",
    "ENGINE_SEARCH_FORMAT",
    "EngineConfig",
    "EvaluationReceipt",
    "JudgeAdapter",
    "RunReceipt",
    "RuntimeAdapter",
    "SearchAdapter",
    "SearchResult",
    "TruthEditingEngine",
    "TruthEditingEngineError",
    "parse_recipe_file",
]
