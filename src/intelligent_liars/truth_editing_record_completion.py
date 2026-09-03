"""Durable, exact-identity completion of semantic judge records.

The evaluator owns aggregation; this module owns only the smaller persistence
boundary around one successfully validated semantic record. A file store keeps
the complete judge result and cache receipt, including request, usage, price,
and adapter identities. Per-record locks make lookup-and-produce one operation
so concurrent workers cannot issue the same paid work through this seam.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from intelligent_liars.truth_editing_judge_contracts import (
    AbsoluteJudgeResult,
    JudgeCacheReceipt,
)


STORE_CONTRACT_FORMAT = "truth_editing_semantic_record_completion_store_v1"
SCOPE_FORMAT = "truth_editing_semantic_record_completion_scope_v1"
COMPLETION_FORMAT = "truth_editing_semantic_record_completion_v1"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_TIERS = frozenset({"discovery", "expanded", "finalist"})


class RecordCompletionError(RuntimeError):
    """Durable semantic completion state is invalid or incompatible."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RecordCompletionError("record completion value is not canonical JSON") from error
    return (rendered + "\n").encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise RecordCompletionError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
    ):
        raise RecordCompletionError(f"{name} must be bounded nonempty trimmed text")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordCompletionError(f"{name} must be an object")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise RecordCompletionError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


@dataclass(frozen=True)
class RecordCompletionRequirement:
    """Exact runtime record identity whose free-form answer needs a judge."""

    record_id: str
    prompt_sha256: str
    raw_generation_sha256: str

    def __post_init__(self) -> None:
        _text(self.record_id, "record_id")
        _digest(self.prompt_sha256, "prompt_sha256")
        _digest(self.raw_generation_sha256, "raw_generation_sha256")

    @classmethod
    def parse(cls, value: Any) -> "RecordCompletionRequirement":
        raw = _mapping(value, "record completion requirement")
        _exact(
            raw,
            {"record_id", "prompt_sha256", "raw_generation_sha256"},
            "record completion requirement",
        )
        return cls(
            _text(raw["record_id"], "record_id"),
            _digest(raw["prompt_sha256"], "prompt_sha256"),
            _digest(raw["raw_generation_sha256"], "raw_generation_sha256"),
        )

    def to_mapping(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "prompt_sha256": self.prompt_sha256,
            "raw_generation_sha256": self.raw_generation_sha256,
        }


@dataclass(frozen=True)
class RecordCompletionScope:
    """Versioned immutable namespace for all required records in one replay."""

    evaluator_config_sha256: str
    dataset_manifest_sha256: str
    recipe_sha256: str
    edited_model_sha256: str
    output_bundle_sha256: str
    tier: str
    judge_config_sha256: str
    rubric_sha256: str
    judge_execution_identity_sha256: str | None
    completion_store_identity_sha256: str
    requirements: tuple[RecordCompletionRequirement, ...]
    content_sha256: str
    format: str = SCOPE_FORMAT

    @classmethod
    def create(
        cls,
        *,
        evaluator_config_sha256: str,
        dataset_manifest_sha256: str,
        recipe_sha256: str,
        edited_model_sha256: str,
        output_bundle_sha256: str,
        tier: str,
        judge_config_sha256: str,
        rubric_sha256: str,
        judge_execution_identity_sha256: str | None,
        completion_store_identity_sha256: str,
        requirements: Sequence[RecordCompletionRequirement],
    ) -> "RecordCompletionScope":
        unsigned = {
            "format": SCOPE_FORMAT,
            "evaluator_config_sha256": evaluator_config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "recipe_sha256": recipe_sha256,
            "edited_model_sha256": edited_model_sha256,
            "output_bundle_sha256": output_bundle_sha256,
            "tier": tier,
            "judge_config_sha256": judge_config_sha256,
            "rubric_sha256": rubric_sha256,
            "judge_execution_identity_sha256": judge_execution_identity_sha256,
            "completion_store_identity_sha256": completion_store_identity_sha256,
            "requirements": [item.to_mapping() for item in requirements],
        }
        return cls.parse({**unsigned, "content_sha256": _hash(unsigned)})

    @classmethod
    def parse(cls, value: Any) -> "RecordCompletionScope":
        raw = _mapping(value, "record completion scope")
        fields = {
            "format",
            "evaluator_config_sha256",
            "dataset_manifest_sha256",
            "recipe_sha256",
            "edited_model_sha256",
            "output_bundle_sha256",
            "tier",
            "judge_config_sha256",
            "rubric_sha256",
            "judge_execution_identity_sha256",
            "completion_store_identity_sha256",
            "requirements",
            "content_sha256",
        }
        _exact(raw, fields, "record completion scope")
        if raw["format"] != SCOPE_FORMAT:
            raise RecordCompletionError("unsupported record completion scope format")
        claimed = _digest(raw["content_sha256"], "scope content_sha256")
        unsigned = {key: value for key, value in raw.items() if key != "content_sha256"}
        if _hash(unsigned) != claimed:
            raise RecordCompletionError("record completion scope hash differs")
        raw_requirements = raw["requirements"]
        if not isinstance(raw_requirements, list):
            raise RecordCompletionError("scope requirements must be an array")
        requirements = tuple(
            RecordCompletionRequirement.parse(item) for item in raw_requirements
        )
        ids = tuple(item.record_id for item in requirements)
        if len(set(ids)) != len(ids):
            raise RecordCompletionError("scope record IDs must be unique")
        tier = _text(raw["tier"], "tier")
        if tier not in _TIERS:
            raise RecordCompletionError(f"unsupported evaluation tier {tier!r}")
        execution_identity = raw["judge_execution_identity_sha256"]
        if execution_identity is not None:
            execution_identity = _digest(
                execution_identity, "judge_execution_identity_sha256"
            )
        return cls(
            evaluator_config_sha256=_digest(
                raw["evaluator_config_sha256"], "evaluator_config_sha256"
            ),
            dataset_manifest_sha256=_digest(
                raw["dataset_manifest_sha256"], "dataset_manifest_sha256"
            ),
            recipe_sha256=_digest(raw["recipe_sha256"], "recipe_sha256"),
            edited_model_sha256=_digest(
                raw["edited_model_sha256"], "edited_model_sha256"
            ),
            output_bundle_sha256=_digest(
                raw["output_bundle_sha256"], "output_bundle_sha256"
            ),
            tier=tier,
            judge_config_sha256=_digest(
                raw["judge_config_sha256"], "judge_config_sha256"
            ),
            rubric_sha256=_digest(raw["rubric_sha256"], "rubric_sha256"),
            judge_execution_identity_sha256=execution_identity,
            completion_store_identity_sha256=_digest(
                raw["completion_store_identity_sha256"],
                "completion_store_identity_sha256",
            ),
            requirements=requirements,
            content_sha256=claimed,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "evaluator_config_sha256": self.evaluator_config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "recipe_sha256": self.recipe_sha256,
            "edited_model_sha256": self.edited_model_sha256,
            "output_bundle_sha256": self.output_bundle_sha256,
            "tier": self.tier,
            "judge_config_sha256": self.judge_config_sha256,
            "rubric_sha256": self.rubric_sha256,
            "judge_execution_identity_sha256": self.judge_execution_identity_sha256,
            "completion_store_identity_sha256": self.completion_store_identity_sha256,
            "requirements": [item.to_mapping() for item in self.requirements],
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class CompletedSemanticRecord:
    """One validated terminal semantic result and its complete paid-call receipt."""

    scope_sha256: str
    requirement: RecordCompletionRequirement
    result: AbsoluteJudgeResult
    cache_receipt: JudgeCacheReceipt
    content_sha256: str
    format: str = COMPLETION_FORMAT

    @classmethod
    def create(
        cls,
        *,
        scope: RecordCompletionScope,
        requirement: RecordCompletionRequirement,
        result: AbsoluteJudgeResult,
        cache_receipt: JudgeCacheReceipt,
        accepted_judge_adapter_code_sha256s: Sequence[str],
    ) -> "CompletedSemanticRecord":
        unsigned = {
            "format": COMPLETION_FORMAT,
            "scope_sha256": scope.content_sha256,
            "requirement": requirement.to_mapping(),
            "result": result.to_payload(),
            "cache_receipt": cache_receipt.to_payload(),
        }
        return cls.parse(
            {**unsigned, "content_sha256": _hash(unsigned)},
            scope=scope,
            expected_requirement=requirement,
            accepted_judge_adapter_code_sha256s=accepted_judge_adapter_code_sha256s,
        )

    @classmethod
    def parse(
        cls,
        value: Any,
        *,
        scope: RecordCompletionScope,
        expected_requirement: RecordCompletionRequirement,
        accepted_judge_adapter_code_sha256s: Sequence[str],
    ) -> "CompletedSemanticRecord":
        raw = _mapping(value, "semantic record completion")
        _exact(
            raw,
            {
                "format",
                "scope_sha256",
                "requirement",
                "result",
                "cache_receipt",
                "content_sha256",
            },
            "semantic record completion",
        )
        if raw["format"] != COMPLETION_FORMAT:
            raise RecordCompletionError("unsupported semantic record completion format")
        claimed = _digest(raw["content_sha256"], "completion content_sha256")
        unsigned = {key: value for key, value in raw.items() if key != "content_sha256"}
        if _hash(unsigned) != claimed:
            raise RecordCompletionError("semantic record completion hash differs")
        if raw["scope_sha256"] != scope.content_sha256:
            raise RecordCompletionError("semantic record completion scope differs")
        requirement = RecordCompletionRequirement.parse(raw["requirement"])
        if requirement != expected_requirement or requirement not in scope.requirements:
            raise RecordCompletionError("semantic record requirement identity differs")
        try:
            result = AbsoluteJudgeResult.parse(raw["result"])
            receipt = JudgeCacheReceipt.parse(raw["cache_receipt"], result=result)
        except Exception as error:
            raise RecordCompletionError(
                f"semantic record judge evidence is invalid: {error}"
            ) from error
        if (
            result.judge_config_sha256 != scope.judge_config_sha256
            or result.rubric_sha256 != scope.rubric_sha256
            or receipt.judge_config_sha256 != scope.judge_config_sha256
            or receipt.rubric_sha256 != scope.rubric_sha256
        ):
            raise RecordCompletionError("semantic record frozen judge identity differs")
        if receipt.response_sha256s != (requirement.raw_generation_sha256,):
            raise RecordCompletionError("semantic record raw response identity differs")
        if result.result is None or tuple(
            item.response_id for item in result.result.responses
        ) != (requirement.record_id,):
            raise RecordCompletionError("semantic record result ID differs")
        accepted_codes = tuple(
            _digest(item, "accepted adapter code SHA-256")
            for item in accepted_judge_adapter_code_sha256s
        )
        if receipt.code_sha256 not in accepted_codes:
            raise RecordCompletionError(
                "semantic record judge adapter code is incompatible"
            )
        return cls(
            scope_sha256=scope.content_sha256,
            requirement=requirement,
            result=result,
            cache_receipt=receipt,
            content_sha256=claimed,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "scope_sha256": self.scope_sha256,
            "requirement": self.requirement.to_mapping(),
            "result": self.result.to_payload(),
            "cache_receipt": self.cache_receipt.to_payload(),
            "content_sha256": self.content_sha256,
        }


CompletionProducer = Callable[[], tuple[AbsoluteJudgeResult, JudgeCacheReceipt]]


class SemanticRecordCompletionStore(Protocol):
    """Public read-or-produce seam used by the optimizer-independent evaluator."""

    @property
    def identity_sha256(self) -> str: ...

    def resolve(
        self,
        scope: RecordCompletionScope,
        requirement: RecordCompletionRequirement,
        producer: CompletionProducer,
    ) -> CompletedSemanticRecord: ...

    def completed(
        self, scope: RecordCompletionScope
    ) -> Mapping[str, CompletedSemanticRecord]: ...

    def missing_record_ids(self, scope: RecordCompletionScope) -> tuple[str, ...]: ...


class FileSemanticRecordCompletionStore:
    """Fsync-backed, first-writer-wins semantic completion repository."""

    def __init__(
        self,
        root: Path | str,
        *,
        accepted_judge_adapter_code_sha256s: Sequence[str],
    ) -> None:
        accepted = tuple(
            sorted(
                {
                    _digest(value, "accepted adapter code SHA-256")
                    for value in accepted_judge_adapter_code_sha256s
                }
            )
        )
        if not accepted:
            raise RecordCompletionError(
                "at least one accepted judge adapter code SHA-256 is required"
            )
        self.root = Path(root)
        self.scopes_root = self.root / "scopes"
        self.locks_root = self.root / "locks"
        self.accepted_judge_adapter_code_sha256s = accepted
        unsigned_contract = {
            "format": STORE_CONTRACT_FORMAT,
            "accepted_judge_adapter_code_sha256s": list(accepted),
        }
        self._identity_sha256 = _hash(unsigned_contract)
        self._contract = {
            **unsigned_contract,
            "content_sha256": self._identity_sha256,
        }
        self._initialize()

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def resolve(
        self,
        scope: RecordCompletionScope,
        requirement: RecordCompletionRequirement,
        producer: CompletionProducer,
    ) -> CompletedSemanticRecord:
        parsed_scope = self._validated_scope(scope)
        expected = self._expected_requirement(parsed_scope, requirement)
        lock_path = self.locks_root / (
            f"{parsed_scope.content_sha256}-{_hash(expected.record_id)}.lock"
        )
        self._reject_symlink(lock_path, "record lock")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._ensure_scope(parsed_scope)
            path = self._completion_path(parsed_scope, expected)
            if path.exists() or path.is_symlink():
                return self._read_completion(path, parsed_scope, expected)
            produced = producer()
            if not isinstance(produced, tuple) or len(produced) != 2:
                raise RecordCompletionError(
                    "semantic completion producer must return result and cache receipt"
                )
            result, receipt = produced
            if not isinstance(result, AbsoluteJudgeResult) or not isinstance(
                receipt, JudgeCacheReceipt
            ):
                raise RecordCompletionError(
                    "semantic completion producer returned incompatible evidence"
                )
            completion = CompletedSemanticRecord.create(
                scope=parsed_scope,
                requirement=expected,
                result=result,
                cache_receipt=receipt,
                accepted_judge_adapter_code_sha256s=(
                    self.accepted_judge_adapter_code_sha256s
                ),
            )
            self._atomic_first_write(path, _canonical_bytes(completion.to_mapping()))
            committed = self._read_completion(path, parsed_scope, expected)
            if committed != completion:
                raise RecordCompletionError("semantic completion commit differs")
            return committed
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def completed(
        self, scope: RecordCompletionScope
    ) -> dict[str, CompletedSemanticRecord]:
        parsed_scope = self._validated_scope(scope)
        self._ensure_scope(parsed_scope)
        records_root = self._records_root(parsed_scope)
        expected_paths = {
            self._completion_path(parsed_scope, requirement): requirement
            for requirement in parsed_scope.requirements
        }
        observed_paths = set(records_root.glob("*.json"))
        if observed_paths - set(expected_paths):
            raise RecordCompletionError("semantic completion inventory has extra rows")
        result: dict[str, CompletedSemanticRecord] = {}
        for path, requirement in expected_paths.items():
            if path.exists() or path.is_symlink():
                result[requirement.record_id] = self._read_completion(
                    path, parsed_scope, requirement
                )
        return result

    def missing_record_ids(self, scope: RecordCompletionScope) -> tuple[str, ...]:
        completed = self.completed(scope)
        return tuple(
            requirement.record_id
            for requirement in scope.requirements
            if requirement.record_id not in completed
        )

    def _validated_scope(self, scope: RecordCompletionScope) -> RecordCompletionScope:
        if not isinstance(scope, RecordCompletionScope):
            raise RecordCompletionError("record completion scope has the wrong type")
        parsed = RecordCompletionScope.parse(scope.to_mapping())
        if parsed.completion_store_identity_sha256 != self.identity_sha256:
            raise RecordCompletionError("record completion store identity differs")
        return parsed

    @staticmethod
    def _expected_requirement(
        scope: RecordCompletionScope,
        requirement: RecordCompletionRequirement,
    ) -> RecordCompletionRequirement:
        if not isinstance(requirement, RecordCompletionRequirement):
            raise RecordCompletionError("record completion requirement has the wrong type")
        matches = tuple(
            item for item in scope.requirements if item.record_id == requirement.record_id
        )
        if matches != (requirement,):
            raise RecordCompletionError("record completion requirement differs from scope")
        return requirement

    def _initialize(self) -> None:
        self._reject_symlink(self.root, "root")
        self._mkdir_durable(self.root, parents=True)
        if not self.root.is_dir():
            raise RecordCompletionError("record completion root must be a directory")
        for path, name in (
            (self.scopes_root, "scopes directory"),
            (self.locks_root, "locks directory"),
        ):
            self._reject_symlink(path, name)
            self._mkdir_durable(path)
            if not path.is_dir():
                raise RecordCompletionError(f"record completion {name} is invalid")
        contract_path = self.root / "contract.json"
        if not contract_path.exists() and not contract_path.is_symlink():
            self._atomic_first_write(contract_path, _canonical_bytes(self._contract))
        observed = self._read_json(contract_path, "store contract")
        if observed != self._contract:
            unsigned = dict(observed)
            claimed = unsigned.pop("content_sha256", None)
            observed_adapters = unsigned.get(
                "accepted_judge_adapter_code_sha256s"
            )
            valid_observed_adapters = (
                isinstance(observed_adapters, list)
                and bool(observed_adapters)
                and all(
                    isinstance(value, str)
                    and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                    for value in observed_adapters
                )
            )
            if (
                unsigned.get("format") != STORE_CONTRACT_FORMAT
                or claimed != _hash(unsigned)
                or not valid_observed_adapters
                or not set(observed_adapters).issubset(  # type: ignore[arg-type]
                    self.accepted_judge_adapter_code_sha256s
                )
            ):
                raise RecordCompletionError(
                    "record completion store contract differs"
                )
            # Compatibility may grow monotonically across an adapter upgrade.
            # Keep the root's original identity so existing scopes retain their
            # immutable binding; each completion still validates against the
            # current accepted adapter set.
            self._contract = observed
            self._identity_sha256 = str(claimed)

    def _ensure_scope(self, scope: RecordCompletionScope) -> None:
        scope_root = self.scopes_root / scope.content_sha256
        records_root = scope_root / "records"
        self._reject_symlink(scope_root, "scope directory")
        self._mkdir_durable(scope_root)
        self._reject_symlink(records_root, "records directory")
        self._mkdir_durable(records_root)
        scope_path = scope_root / "scope.json"
        if not scope_path.exists() and not scope_path.is_symlink():
            self._atomic_first_write(scope_path, _canonical_bytes(scope.to_mapping()))
        observed = RecordCompletionScope.parse(
            self._read_json(scope_path, "scope contract")
        )
        if observed != scope:
            raise RecordCompletionError("record completion scope contract differs")

    def _records_root(self, scope: RecordCompletionScope) -> Path:
        return self.scopes_root / scope.content_sha256 / "records"

    def _completion_path(
        self,
        scope: RecordCompletionScope,
        requirement: RecordCompletionRequirement,
    ) -> Path:
        return self._records_root(scope) / f"{_hash(requirement.record_id)}.json"

    def _read_completion(
        self,
        path: Path,
        scope: RecordCompletionScope,
        requirement: RecordCompletionRequirement,
    ) -> CompletedSemanticRecord:
        return CompletedSemanticRecord.parse(
            self._read_json(path, "semantic completion row"),
            scope=scope,
            expected_requirement=requirement,
            accepted_judge_adapter_code_sha256s=(
                self.accepted_judge_adapter_code_sha256s
            ),
        )

    def _read_json(self, path: Path, name: str) -> dict[str, Any]:
        self._reject_symlink(path, name)
        if not path.is_file():
            raise RecordCompletionError(f"record completion {name} must be regular")
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RecordCompletionError(
                f"record completion {name} is unreadable"
            ) from error
        if not isinstance(value, dict):
            raise RecordCompletionError(f"record completion {name} must be an object")
        return value

    @staticmethod
    def _reject_symlink(path: Path, name: str) -> None:
        if path.is_symlink():
            raise RecordCompletionError(f"record completion {name} must not be a symlink")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _mkdir_durable(cls, path: Path, *, parents: bool = False) -> None:
        missing: list[Path] = []
        cursor = path
        while not cursor.exists():
            missing.append(cursor)
            if not parents:
                break
            cursor = cursor.parent
        path.mkdir(parents=parents, exist_ok=True)
        # Persist every newly created directory entry from the highest missing
        # ancestor downward. File fsync alone cannot make these parent entries
        # durable across a power loss.
        for created in reversed(missing):
            cls._fsync_directory(created)
            cls._fsync_directory(created.parent)

    def _atomic_first_write(self, path: Path, data: bytes) -> None:
        self._reject_symlink(path, path.name)
        self._mkdir_durable(path.parent, parents=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = [
    "COMPLETION_FORMAT",
    "CompletedSemanticRecord",
    "FileSemanticRecordCompletionStore",
    "RecordCompletionError",
    "RecordCompletionRequirement",
    "RecordCompletionScope",
    "SCOPE_FORMAT",
    "STORE_CONTRACT_FORMAT",
    "SemanticRecordCompletionStore",
]
