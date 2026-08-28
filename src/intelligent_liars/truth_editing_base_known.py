"""Resumable, batch-scheduled proof that the frozen base model knows QA answers.

Only validation is admitted by the normal runner.  The sealed test split is
intentionally impossible to open through this module.  A production Qwen
backend and stored-response test backend implement the same small protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    QWEN_ATTENTION_IMPLEMENTATION,
    QWEN_DEVICE_MAP,
    QWEN_DTYPE_NAME,
)

FORMAT = "truth_editing_base_known_qualification_v1"
RECORD_FORMAT = "truth_editing_base_known_record_v1"
BATCH_FORMAT = "truth_editing_base_known_batch_v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class BaseKnownError(RuntimeError):
    """Qualification evidence is invalid, incomplete, or not reproducible."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise BaseKnownError("value is not canonical JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(content)
    temporary.replace(path)


@dataclass(frozen=True)
class FrozenBaseIdentity:
    repository: str
    revision: str
    model_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    inference_backend: str
    dtype: str
    attention_implementation: str
    device_map: str
    local_files_only: bool
    use_cache: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenBaseIdentity":
        required = {"repository", "revision", "model_sha256", "tokenizer_sha256", "chat_template_sha256", "inference_backend", "dtype", "attention_implementation", "device_map", "local_files_only", "use_cache"}
        if set(raw) != required:
            raise BaseKnownError("model identity fields differ from the frozen runtime contract")
        if raw["repository"] != DEFAULT_MODEL_ID or raw["revision"] != DEFAULT_MODEL_REVISION:
            raise BaseKnownError("base-known model must be the frozen Qwen checkpoint")
        for key in ("model_sha256", "tokenizer_sha256", "chat_template_sha256"):
            if not isinstance(raw[key], str) or not _SHA.fullmatch(raw[key]):
                raise BaseKnownError(f"invalid {key}")
        expected = {
            "inference_backend": "transformers", "dtype": QWEN_DTYPE_NAME,
            "attention_implementation": QWEN_ATTENTION_IMPLEMENTATION,
            "device_map": QWEN_DEVICE_MAP, "local_files_only": True, "use_cache": True,
        }
        if any(raw[key] != value for key, value in expected.items()):
            raise BaseKnownError("base-known runtime identity differs from frozen Qwen runtime")
        return cls(**{key: raw[key] for key in required})

    def to_payload(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in ("repository", "revision", "model_sha256", "tokenizer_sha256", "chat_template_sha256", "inference_backend", "dtype", "attention_implementation", "device_map", "local_files_only", "use_cache")}


@dataclass(frozen=True)
class QualificationConfig:
    split: str = "validation"
    rotations: str = "all"
    repeats: int = 2
    batch_size: int = 16
    max_new_tokens: int = 8
    seed: int = 20260827
    strict_all_correct: bool = True

    def __post_init__(self) -> None:
        if self.split not in {"validation", "test"}:
            raise BaseKnownError("qualification split must be validation or sealed test")
        if self.rotations != "all":
            raise BaseKnownError("answer-order rotations must be all")
        if not isinstance(self.repeats, int) or isinstance(self.repeats, bool) or self.repeats < 2:
            raise BaseKnownError("qualification repeats must be at least two")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise BaseKnownError("batch_size must be positive")
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or not 1 <= self.max_new_tokens <= 32:
            raise BaseKnownError("max_new_tokens must be in [1, 32]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise BaseKnownError("seed must be a non-negative integer")
        if self.strict_all_correct is not True:
            raise BaseKnownError("strict_all_correct cannot be disabled")

    def to_payload(self) -> dict[str, Any]:
        return {"split": self.split, "rotations": self.rotations, "repeats": self.repeats, "batch_size": self.batch_size, "max_new_tokens": self.max_new_tokens, "seed": self.seed, "strict_all_correct": self.strict_all_correct, "decoding": {"do_sample": False, "temperature": 0.0, "num_beams": 1, "answer_format": "one_uppercase_option_label", "chat_template": {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False, "assistant_prefill": "empty_think"}, "option_constraint": "request_labels_then_eos"}}


@dataclass(frozen=True)
class QualificationRequest:
    request_id: str
    record_id: str
    rotation_index: int
    repeat_index: int
    labels: tuple[str, ...]
    ordered_choices: tuple[str, ...]
    correct_answer: str
    prompt: str
    seed: int
    max_new_tokens: int

    def to_payload(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "record_id": self.record_id, "rotation_index": self.rotation_index, "repeat_index": self.repeat_index, "labels": list(self.labels), "ordered_choices": list(self.ordered_choices), "correct_answer": self.correct_answer, "prompt": self.prompt, "seed": self.seed, "max_new_tokens": self.max_new_tokens, "do_sample": False}


@dataclass(frozen=True)
class QualificationResponse:
    request_id: str
    text: str
    token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not re.fullmatch(r"bk_[0-9a-f]{64}", self.request_id):
            raise BaseKnownError("response request_id is invalid")
        if not isinstance(self.text, str):
            raise BaseKnownError("response text must be a string")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.token_ids):
            raise BaseKnownError("response token_ids must be non-negative integers")

    def to_payload(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "text": self.text, "token_ids": list(self.token_ids)}


class QualificationBackend(Protocol):
    def generate_batch(self, requests: Sequence[QualificationRequest], identity: FrozenBaseIdentity) -> Sequence[QualificationResponse]: ...


_VERIFIED_EVIDENCE_SEAL = object()


class VerifiedFrozenQwenEvidence:
    """Nominally sealed evidence loaded from an independently hashed receipt file."""

    def __init__(self, receipt_sha256: str, *, _seal: object) -> None:
        if _seal is not _VERIFIED_EVIDENCE_SEAL:
            raise BaseKnownError("verified Qwen evidence must be opened from a receipt")
        self.receipt_sha256 = receipt_sha256

    @classmethod
    def open(cls, path: Path, expected_identity: FrozenBaseIdentity) -> "VerifiedFrozenQwenEvidence":
        receipt_path = Path(path)
        raw = json.loads(receipt_path.read_text())
        fields = {"format", "model", "snapshot_manifest_sha256", "runtime_identity_sha256", "software_sha256", "self_sha256"}
        if set(raw) != fields or raw.get("format") not in {"frozen_qwen_execution_receipt_v1", "frozen_qwen_execution_receipt_v2"} or raw.get("model") != expected_identity.to_payload():
            raise BaseKnownError("frozen Qwen execution receipt schema or model differs")
        self_sha = raw.pop("self_sha256")
        if self_sha != _hash(raw):
            raise BaseKnownError("frozen Qwen execution receipt identity differs")
        for key in ("snapshot_manifest_sha256", "runtime_identity_sha256", "software_sha256"):
            if not isinstance(raw[key], str) or not _SHA.fullmatch(raw[key]):
                raise BaseKnownError("frozen Qwen execution receipt hashes differ")
        return cls(_file_hash(receipt_path), _seal=_VERIFIED_EVIDENCE_SEAL)

    def to_payload(self) -> dict[str, str]:
        return {"mode": "verified_frozen_qwen", "receipt_sha256": self.receipt_sha256}


def resolve_backend_evidence(backend: QualificationBackend, identity: FrozenBaseIdentity) -> dict[str, str]:
    """Reject self-asserted production evidence; only the sealed receipt type qualifies."""

    method = getattr(backend, "evidence_receipt", None)
    value = method(identity) if callable(method) else None
    if isinstance(value, VerifiedFrozenQwenEvidence):
        from .truth_editing_qwen_qualification import FrozenQwenQualificationBackend

        if not isinstance(backend, FrozenQwenQualificationBackend) or backend._enforce_production_runtime is not True:
            raise BaseKnownError("verified production evidence must be issued by the enforced frozen Qwen backend")
        return value.to_payload()
    if isinstance(backend, StoredResponseBackend):
        if value != backend.evidence_receipt(identity):
            raise BaseKnownError("stored response evidence identity differs")
        return value
    if value is not None:
        raise BaseKnownError("backend attempted to self-assert unverified production evidence")
    return {"mode": "synthetic_mock_only", "receipt_sha256": "0" * 64}


def validate_backend_batch_execution(
    execution: Any,
    backend_evidence: Mapping[str, str],
    requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
) -> None:
    """Require per-batch runtime proof for production and forbid it for replays."""

    if backend_evidence.get("mode") != "verified_frozen_qwen":
        if execution is not None:
            raise BaseKnownError("nonproduction qualification cannot carry production execution evidence")
        return
    fields = {"format", "backend_receipt_sha256", "request_set_sha256", "response_set_sha256", "request_count", "prompt_token_count", "generated_token_count", "elapsed_seconds", "generated_tokens_per_second", "cuda_peak_allocated_bytes", "self_sha256"}
    if not isinstance(execution, Mapping) or set(execution) != fields or execution.get("format") not in {"truth_editing_frozen_qwen_batch_execution_v1", "truth_editing_frozen_qwen_batch_execution_v2"}:
        raise BaseKnownError("production batch execution receipt schema differs")
    unsigned = dict(execution)
    self_sha = unsigned.pop("self_sha256")
    if self_sha != _hash(unsigned) or execution["backend_receipt_sha256"] != backend_evidence.get("receipt_sha256") or execution["request_set_sha256"] != _hash(list(requests)) or execution["response_set_sha256"] != _hash(list(responses)) or execution["request_count"] != len(requests):
        raise BaseKnownError("production batch execution identity differs")
    for key in ("request_count", "prompt_token_count", "generated_token_count", "cuda_peak_allocated_bytes"):
        if type(execution[key]) is not int or execution[key] < 0:
            raise BaseKnownError("production batch execution integer telemetry differs")
    for key in ("elapsed_seconds", "generated_tokens_per_second"):
        value = execution[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0):
            raise BaseKnownError("production batch execution timing telemetry differs")
    generated_count = sum(len(response.get("token_ids", ())) for response in responses)
    elapsed = execution["elapsed_seconds"]
    expected_rate = generated_count / elapsed if elapsed and elapsed > 0 else None
    if execution["generated_token_count"] != generated_count:
        raise BaseKnownError("production batch generated-token telemetry differs")
    actual_rate = execution["generated_tokens_per_second"]
    if (expected_rate is None) != (actual_rate is None) or (
        expected_rate is not None
        and not math.isclose(float(actual_rate), expected_rate, rel_tol=1e-12, abs_tol=0.0)
    ):
        raise BaseKnownError("production batch throughput telemetry differs")


class StoredResponseBackend:
    """Offline backend replaying an explicit request-id keyed JSON object."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        raw = json.loads(self._path.read_text())
        if not isinstance(raw, dict):
            raise BaseKnownError("stored responses must be a request-id keyed object")
        self._responses = raw

    def evidence_receipt(self, identity: FrozenBaseIdentity) -> dict[str, str]:
        del identity
        return {"mode": "stored_replay", "receipt_sha256": _file_hash(self._path)}

    def generate_batch(self, requests: Sequence[QualificationRequest], identity: FrozenBaseIdentity) -> Sequence[QualificationResponse]:
        del identity
        output = []
        for request in requests:
            if request.request_id not in self._responses:
                raise BaseKnownError(f"stored responses lack request {request.request_id}")
            value = self._responses[request.request_id]
            if not isinstance(value, dict) or set(value) != {"text", "token_ids"}:
                raise BaseKnownError("stored response values require exactly text and token_ids")
            output.append(QualificationResponse(request.request_id, value["text"], tuple(value["token_ids"])))
        return tuple(output)


@dataclass(frozen=True)
class QualifiedRecord:
    record_id: str
    split: str
    family: str
    request_count: int
    parsed_count: int
    correct_count: int
    parse_failure_count: int
    stable: bool
    base_known: bool
    evidence_request_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {"format": RECORD_FORMAT, **{key: getattr(self, key) for key in ("record_id", "split", "family", "request_count", "parsed_count", "correct_count", "parse_failure_count", "stable", "base_known")}, "evidence_request_ids": list(self.evidence_request_ids)}


@dataclass(frozen=True)
class QualificationResult:
    manifest_sha256: str
    records: tuple[QualifiedRecord, ...]

    @property
    def qualified_record_ids(self) -> tuple[str, ...]:
        return tuple(row.record_id for row in self.records if row.base_known)


@dataclass(frozen=True)
class BaseKnownQualification:
    """Verified, immutable qualification evidence consumed by scenario builders."""

    manifest_sha256: str
    dataset_manifest_sha256: str
    split_file_sha256: str
    qualified_record_ids: tuple[str, ...]
    records: tuple[QualifiedRecord, ...]

    @classmethod
    def open(cls, output_dir: Path, *, allow_nonproduction: bool = False) -> "BaseKnownQualification":
        root = Path(output_dir)
        try:
            manifest = json.loads((root / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise BaseKnownError("qualification manifest is unreadable") from error
        fields = {
            "format", "dataset_id", "dataset_manifest_sha256", "split_file_sha256",
            "model", "config", "request_set_sha256", "run_contract_sha256",
            "record_count", "qualified_count", "records_sha256", "self_sha256",
            "observations_sha256", "batch_receipts_sha256",
            "backend_evidence",
        }
        if set(manifest) != fields or manifest.get("format") != FORMAT:
            raise BaseKnownError("qualification manifest schema is unsupported")
        self_sha = manifest.pop("self_sha256")
        if self_sha != _hash(manifest):
            raise BaseKnownError("qualification manifest self identity is invalid")
        for key in ("dataset_manifest_sha256", "split_file_sha256", "request_set_sha256", "run_contract_sha256", "records_sha256", "observations_sha256", "batch_receipts_sha256"):
            if not isinstance(manifest[key], str) or not _SHA.fullmatch(manifest[key]):
                raise BaseKnownError(f"qualification manifest {key} is invalid")
        FrozenBaseIdentity.from_mapping(manifest["model"])
        backend_evidence = manifest["backend_evidence"]
        if not isinstance(backend_evidence, dict) or set(backend_evidence) != {"mode", "receipt_sha256"} or not _SHA.fullmatch(str(backend_evidence["receipt_sha256"])):
            raise BaseKnownError("backend evidence receipt is invalid")
        if backend_evidence["mode"] != "verified_frozen_qwen" and not allow_nonproduction:
            raise BaseKnownError("stored or mock responses are not production qualification evidence")
        if manifest["config"].get("split") != "validation":
            raise BaseKnownError("only validation qualification may be consumed")
        contract_fields = {"format", "dataset_id", "dataset_manifest_sha256", "split_file_sha256", "model", "config", "request_set_sha256", "backend_evidence"}
        contract = {key: manifest[key] for key in contract_fields}
        if _hash(contract) != manifest["run_contract_sha256"]:
            raise BaseKnownError("qualification run contract identity is invalid")
        records_path = root / "records.jsonl"
        if _file_hash(records_path) != manifest["records_sha256"]:
            raise BaseKnownError("qualification records hash is invalid")
        records = []
        for line in records_path.read_text().splitlines():
            raw = json.loads(line)
            required = {"format", "record_id", "split", "family", "request_count", "parsed_count", "correct_count", "parse_failure_count", "stable", "base_known", "evidence_request_ids"}
            if set(raw) != required or raw["format"] != RECORD_FORMAT or raw["split"] != "validation":
                raise BaseKnownError("qualification record schema is invalid")
            integer_keys = ("request_count", "parsed_count", "correct_count", "parse_failure_count")
            if any(not isinstance(raw[key], int) or isinstance(raw[key], bool) or raw[key] < 0 for key in integer_keys):
                raise BaseKnownError("qualification record counts are invalid")
            if not isinstance(raw["stable"], bool) or not isinstance(raw["base_known"], bool) or not isinstance(raw["evidence_request_ids"], list):
                raise BaseKnownError("qualification record booleans or evidence IDs are invalid")
            record = QualifiedRecord(
                record_id=str(raw["record_id"]), split="validation", family=str(raw["family"]),
                request_count=int(raw["request_count"]), parsed_count=int(raw["parsed_count"]),
                correct_count=int(raw["correct_count"]), parse_failure_count=int(raw["parse_failure_count"]),
                stable=raw["stable"] is True, base_known=raw["base_known"] is True,
                evidence_request_ids=tuple(raw["evidence_request_ids"]),
            )
            if record.base_known and (not record.stable or record.correct_count != record.request_count or record.parsed_count != record.request_count):
                raise BaseKnownError("base-known record lacks strict complete evidence")
            if record.request_count < 1 or len(record.evidence_request_ids) != record.request_count or len(set(record.evidence_request_ids)) != record.request_count:
                raise BaseKnownError("qualification record evidence cardinality is invalid")
            if record.parsed_count + record.parse_failure_count != record.request_count or record.correct_count > record.parsed_count:
                raise BaseKnownError("qualification record count identities are invalid")
            records.append(record)
        if len(records) != manifest["record_count"] or sum(row.base_known for row in records) != manifest["qualified_count"]:
            raise BaseKnownError("qualification aggregate counts are invalid")
        ids = tuple(row.record_id for row in records)
        if len(set(ids)) != len(ids):
            raise BaseKnownError("qualification records contain duplicate IDs")
        observations_path = root / "observations.jsonl"
        if _file_hash(observations_path) != manifest["observations_sha256"]:
            raise BaseKnownError("qualification observations hash is invalid")
        observations = [json.loads(line) for line in observations_path.read_text().splitlines()]
        evidence_ids = [request_id for record in records for request_id in record.evidence_request_ids]
        if [value.get("request_id") for value in observations] != evidence_ids:
            raise BaseKnownError("qualification observations do not match record evidence IDs")
        receipt_paths = sorted((root / "batches").glob("batch_*/receipt.json"))
        receipts = [
            {"path": path.relative_to(root).as_posix(), "sha256": _file_hash(path)}
            for path in receipt_paths
        ]
        if _hash(receipts) != manifest["batch_receipts_sha256"]:
            raise BaseKnownError("qualification batch receipt set is invalid")
        response_hashes: dict[str, str] = {}
        raw_requests: dict[str, dict[str, Any]] = {}
        for receipt_path in receipt_paths:
            receipt = json.loads(receipt_path.read_text())
            receipt_self = receipt.pop("self_sha256", None)
            batch_dir = receipt_path.parent
            if (
                receipt_self != _hash(receipt)
                or receipt.get("requests_sha256") != _file_hash(batch_dir / "requests.json")
                or receipt.get("responses_sha256") != _file_hash(batch_dir / "responses.json")
            ):
                raise BaseKnownError("qualification batch evidence is invalid")
            requests_payload = json.loads((batch_dir / "requests.json").read_text())
            if receipt.get("request_sha256") != _hash(requests_payload):
                raise BaseKnownError("qualification batch request identity is invalid")
            raw_batch_responses = json.loads((batch_dir / "responses.json").read_text())
            validate_backend_batch_execution(
                receipt.get("backend_execution"), backend_evidence,
                requests_payload.get("requests", []), raw_batch_responses,
            )
            for request in requests_payload.get("requests", []):
                request_id = request.get("request_id")
                if request_id in raw_requests:
                    raise BaseKnownError("qualification contains duplicate raw requests")
                raw_requests[request_id] = request
            for response in raw_batch_responses:
                parsed_response = QualificationResponse(
                    response["request_id"], response["text"], tuple(response["token_ids"])
                )
                if parsed_response.request_id in response_hashes:
                    raise BaseKnownError("qualification contains duplicate raw responses")
                response_hashes[parsed_response.request_id] = _hash(parsed_response.to_payload())
        observation_fields = {"request_id", "record_id", "rotation_index", "repeat_index", "parsed", "selected_answer", "correct", "response_sha256"}
        for value in observations:
            request = raw_requests.get(value.get("request_id"))
            expected_correct = bool(value.get("parsed")) and value.get("selected_answer") == (request or {}).get("correct_answer")
            if set(value) != observation_fields or response_hashes.get(value["request_id"]) != value["response_sha256"] or value.get("correct") is not expected_correct:
                raise BaseKnownError("qualification observation does not bind its raw response")
        if set(response_hashes) != set(evidence_ids) or set(raw_requests) != set(evidence_ids):
            raise BaseKnownError("qualification raw responses do not match evidence IDs")
        qualified = tuple(row.record_id for row in records if row.base_known)
        return cls(self_sha, manifest["dataset_manifest_sha256"], manifest["split_file_sha256"], qualified, tuple(records))


@dataclass(frozen=True)
class QualificationBatchStore:
    """Shared atomic batch evidence boundary for every qualification source."""

    output_dir: Path
    identity: FrozenBaseIdentity
    backend: QualificationBackend
    backend_evidence: Mapping[str, str] | None = None

    def run(self, number: int, requests: Sequence[QualificationRequest], run_sha: str) -> dict[str, QualificationResponse]:
        batch_dir = self.output_dir / "batches" / f"batch_{number:06d}"
        request_rows = [request.to_payload() for request in requests]
        request_payload = {"format": BATCH_FORMAT, "run_contract_sha256": run_sha, "batch_number": number, "requests": request_rows}
        request_sha = _hash(request_payload)
        if batch_dir.exists():
            try:
                receipt = json.loads((batch_dir / "receipt.json").read_text())
                self_sha = receipt.pop("self_sha256", None)
                persisted_requests = json.loads((batch_dir / "requests.json").read_text())
                if self_sha != _hash(receipt) or receipt["request_sha256"] != request_sha or receipt["request_sha256"] != _hash(persisted_requests) or receipt["responses_sha256"] != _file_hash(batch_dir / "responses.json") or receipt["requests_sha256"] != _file_hash(batch_dir / "requests.json"):
                    raise BaseKnownError("immutable batch evidence failed verification")
                raw = json.loads((batch_dir / "responses.json").read_text())
            except (OSError, KeyError, json.JSONDecodeError) as error:
                raise BaseKnownError("immutable batch evidence is incomplete") from error
        else:
            returned = tuple(self.backend.generate_batch(requests, self.identity))
            expected_ids = [request.request_id for request in requests]
            actual_ids = [response.request_id for response in returned]
            if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
                raise BaseKnownError("backend response IDs do not match request IDs exactly")
            by_id = {response.request_id: response for response in returned}
            raw = [by_id[request_id].to_payload() for request_id in expected_ids]
            batch_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{batch_dir.name}.", dir=batch_dir.parent))
            requests_path, responses_path = staging / "requests.json", staging / "responses.json"
            requests_path.write_bytes(_canonical(request_payload) + b"\n")
            responses_path.write_bytes(_canonical(raw) + b"\n")
            receipt = {"format": BATCH_FORMAT, "request_sha256": request_sha, "requests_sha256": _file_hash(requests_path), "responses_sha256": _file_hash(responses_path)}
            execution_method = getattr(self.backend, "batch_execution_receipt", None)
            if callable(execution_method):
                execution = execution_method(requests, returned, self.identity)
                if execution is not None and not isinstance(execution, Mapping):
                    raise BaseKnownError("backend execution receipt must be an object")
                if execution is not None:
                    receipt["backend_execution"] = dict(execution)
            receipt["self_sha256"] = _hash(receipt)
            (staging / "receipt.json").write_bytes(_canonical(receipt) + b"\n")
            if self.backend_evidence is not None:
                try:
                    validate_backend_batch_execution(
                        receipt.get("backend_execution"), self.backend_evidence,
                        request_rows, raw,
                    )
                except BaseKnownError:
                    shutil.rmtree(staging)
                    raise
            try:
                staging.rename(batch_dir)
            except OSError:
                if not batch_dir.exists():
                    raise
                shutil.rmtree(staging)
                return self.run(number, requests, run_sha)
        if self.backend_evidence is not None:
            validate_backend_batch_execution(
                receipt.get("backend_execution"), self.backend_evidence,
                request_rows, raw,
            )
        result = {}
        for value in raw:
            if set(value) != {"request_id", "text", "token_ids"} or not isinstance(value["text"], str) or not isinstance(value["token_ids"], list):
                raise BaseKnownError("stored backend response has invalid schema")
            result[value["request_id"]] = QualificationResponse(value["request_id"], value["text"], tuple(value["token_ids"]))
        if set(result) != {request.request_id for request in requests}:
            raise BaseKnownError("stored response IDs do not match requests")
        return result


class BaseKnownRunner:
    def __init__(self, dataset_dir: Path, output_dir: Path, model_identity: Mapping[str, Any], config: QualificationConfig, backend: QualificationBackend) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.identity = FrozenBaseIdentity.from_mapping(model_identity)
        self.config = config
        self.backend = backend

    def run(self) -> QualificationResult:
        if self.config.split == "test":
            raise BaseKnownError("test split is sealed; normal qualification is validation-only")
        manifest = json.loads((self.dataset_dir / "manifest.json").read_text())
        split_path = self.dataset_dir / f"{self.config.split}.jsonl"
        expected = manifest.get("file_sha256", {}).get(split_path.name)
        if expected != _file_hash(split_path):
            raise BaseKnownError("dataset split hash does not match its manifest")
        rows = self._load_rows(split_path)
        requests = tuple(request for row in rows for request in self._requests(row))
        backend_evidence = resolve_backend_evidence(self.backend, self.identity)
        if not isinstance(backend_evidence, dict) or set(backend_evidence) != {"mode", "receipt_sha256"} or backend_evidence["mode"] not in {"synthetic_mock_only", "stored_replay", "verified_frozen_qwen"} or not _SHA.fullmatch(str(backend_evidence["receipt_sha256"])):
            raise BaseKnownError("backend returned an invalid evidence receipt")
        run_contract = {"format": FORMAT, "dataset_id": manifest.get("dataset_id"), "dataset_manifest_sha256": _file_hash(self.dataset_dir / "manifest.json"), "split_file_sha256": expected, "model": self.identity.to_payload(), "config": self.config.to_payload(), "request_set_sha256": _hash([request.to_payload() for request in requests]), "backend_evidence": backend_evidence}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = self.output_dir / "run_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != run_contract:
            raise BaseKnownError("existing run identity differs from requested run identity")
        if not contract_path.exists():
            _atomic_write(contract_path, _canonical(run_contract) + b"\n")
        responses: dict[str, QualificationResponse] = {}
        for index in range(0, len(requests), self.config.batch_size):
            batch = requests[index:index + self.config.batch_size]
            responses.update(self._run_batch(index // self.config.batch_size, batch, _hash(run_contract)))
        records, observations = self._aggregate(rows, requests, responses)
        records_path = self.output_dir / "records.jsonl"
        records_bytes = b"".join(_canonical(record.to_payload()) + b"\n" for record in records)
        if records_path.exists() and records_path.read_bytes() != records_bytes:
            raise BaseKnownError("existing aggregate record evidence differs")
        _atomic_write(records_path, records_bytes)
        observations_path = self.output_dir / "observations.jsonl"
        observations_bytes = b"".join(_canonical(value) + b"\n" for value in observations)
        if observations_path.exists() and observations_path.read_bytes() != observations_bytes:
            raise BaseKnownError("existing parsed observation evidence differs")
        _atomic_write(observations_path, observations_bytes)
        receipt_inventory = [
            {"path": path.relative_to(self.output_dir).as_posix(), "sha256": _file_hash(path)}
            for path in sorted((self.output_dir / "batches").glob("batch_*/receipt.json"))
        ]
        result_payload = {**run_contract, "run_contract_sha256": _hash(run_contract), "record_count": len(records), "qualified_count": sum(row.base_known for row in records), "records_sha256": hashlib.sha256(records_bytes).hexdigest(), "observations_sha256": hashlib.sha256(observations_bytes).hexdigest(), "batch_receipts_sha256": _hash(receipt_inventory)}
        result_payload["self_sha256"] = _hash(result_payload)
        result_path = self.output_dir / "manifest.json"
        encoded = _canonical(result_payload) + b"\n"
        if result_path.exists() and result_path.read_bytes() != encoded:
            raise BaseKnownError("existing qualification manifest differs")
        _atomic_write(result_path, encoded)
        return QualificationResult(result_payload["self_sha256"], records)

    def _load_rows(self, path: Path) -> tuple[dict[str, Any], ...]:
        rows = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            required = {"format", "record_id", "canonical_key", "collision_cluster_id", "question", "choices", "correct_answer", "wrong_answers", "family", "truth_authority", "split"}
            if set(row) != required or row.get("format") != "truth_editing_canonical_qa_record_v2" or row.get("split") != self.config.split:
                raise BaseKnownError("invalid v2 qualification record")
            text_fields = ("record_id", "canonical_key", "collision_cluster_id", "question", "correct_answer", "family", "truth_authority")
            if any(not isinstance(row[key], str) or not row[key] for key in text_fields) or not isinstance(row["choices"], list) or not isinstance(row["wrong_answers"], list):
                raise BaseKnownError("invalid v2 qualification record types")
            if row["correct_answer"] not in row["choices"] or set(row["wrong_answers"]) != set(row["choices"]) - {row["correct_answer"]}:
                raise BaseKnownError("v2 correct and wrong answer fields disagree")
            if any(not isinstance(value, str) or not value for value in row["choices"]) or len(row["choices"]) < 2 or len(row["choices"]) > len(_LABELS) or len(set(row["choices"])) != len(row["choices"]):
                raise BaseKnownError("qualification choices must be 2-26 unique values")
            rows.append(row)
        if len({row["record_id"] for row in rows}) != len(rows):
            raise BaseKnownError("duplicate qualification record IDs")
        return tuple(rows)

    def _requests(self, row: Mapping[str, Any]) -> tuple[QualificationRequest, ...]:
        choices = tuple(str(value) for value in row["choices"])
        built = []
        for rotation in range(len(choices)):
            ordered = choices[rotation:] + choices[:rotation]
            labels = _LABELS[:len(ordered)]
            options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, ordered, strict=True))
            prompt = f"Answer the multiple-choice question. Return exactly one uppercase option label and no other text.\n\nQuestion: {row['question']}\n\nOptions:\n{options}\n\nAnswer:"
            for repeat in range(self.config.repeats):
                payload = [row["record_id"], rotation, repeat, self.config.seed]
                built.append(QualificationRequest("bk_" + _hash(payload), str(row["record_id"]), rotation, repeat, labels, ordered, str(row["correct_answer"]), prompt, int(_hash(payload)[:8], 16), self.config.max_new_tokens))
        return tuple(built)

    def _run_batch(self, number: int, requests: Sequence[QualificationRequest], run_sha: str) -> dict[str, QualificationResponse]:
        return QualificationBatchStore(
            self.output_dir, self.identity, self.backend,
            resolve_backend_evidence(self.backend, self.identity),
        ).run(number, requests, run_sha)

    def _aggregate(self, rows: Sequence[Mapping[str, Any]], requests: Sequence[QualificationRequest], responses: Mapping[str, QualificationResponse]) -> tuple[tuple[QualifiedRecord, ...], tuple[dict[str, Any], ...]]:
        grouped: dict[str, list[QualificationRequest]] = defaultdict(list)
        for request in requests:
            grouped[request.record_id].append(request)
        output = []
        observations = []
        for row in rows:
            parsed, correct, semantic = 0, 0, []
            evidence = []
            for request in grouped[str(row["record_id"])]:
                response = responses[request.request_id]
                evidence.append(request.request_id)
                text = response.text.strip()
                if text not in request.labels:
                    observations.append({"request_id": request.request_id, "record_id": request.record_id, "rotation_index": request.rotation_index, "repeat_index": request.repeat_index, "parsed": False, "selected_answer": None, "correct": False, "response_sha256": _hash(response.to_payload())})
                    continue
                parsed += 1
                answer = request.ordered_choices[request.labels.index(text)]
                semantic.append(answer)
                correct += answer == row["correct_answer"]
                observations.append({"request_id": request.request_id, "record_id": request.record_id, "rotation_index": request.rotation_index, "repeat_index": request.repeat_index, "parsed": True, "selected_answer": answer, "correct": answer == row["correct_answer"], "response_sha256": _hash(response.to_payload())})
            count = len(evidence)
            stable = parsed == count and len(set(semantic)) == 1
            base_known = stable and correct == count
            output.append(QualifiedRecord(str(row["record_id"]), self.config.split, str(row["family"]), count, parsed, correct, count - parsed, stable, base_known, tuple(evidence)))
        return tuple(output), tuple(observations)
