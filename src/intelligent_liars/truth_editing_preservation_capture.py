"""Frozen-base capability-preservation logits capture.

The module owns one deep seam: :func:`capture_preservation_baselines` turns a
strict input plan into immutable, materializer-compatible safetensors and
per-record receipts.  Model-specific rendering and inference sit behind one
batched adapter interface; the production adapter uses the project's verified
Qwen loader while tests can provide a deterministic offline adapter.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file, save_file

from .models import (
    DEFAULT_MODEL_CONTENT_SHA256,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_SNAPSHOT_MANIFEST_SHA256,
    QWEN_ATTENTION_IMPLEMENTATION,
    QWEN_DEVICE_MAP,
    ModelBundle,
    ModelLoadConfig,
    load_model_and_processor,
)
from .truth_editing_preservation_materialization import (
    LEGACY_COMPACT_CAPTURE_REPRESENTATION,
    PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT,
    PRESERVATION_COMPACT_CAPTURE_REPRESENTATION,
)
from .truth_editing_preservation_runtime import (
    FrozenPreservationInput,
    PreservationRuntimeError,
    _parse_media,
    _parse_messages,
    _validate_media_bindings,
)


PRESERVATION_BASELINE_CAPTURE_PLAN_FORMAT = (
    "truth_editing_preservation_baseline_capture_plan_v2"
)
PRESERVATION_BASELINE_CAPTURE_RUN_FORMAT = (
    "truth_editing_preservation_baseline_capture_run_v2"
)
QWEN_BASELINE_CAPTURE_RUNTIME_FORMAT = (
    "truth_editing_qwen_preservation_baseline_capture_runtime_v2"
)
_HEX = frozenset("0123456789abcdef")
_IDENTITY_FIELDS = {
    "base_model_sha256",
    "tokenizer_sha256",
    "processor_sha256",
    "chat_template_sha256",
    "inference_runtime_sha256",
}


class PreservationBaselineCaptureError(RuntimeError):
    """The capture plan, backend identity, or produced evidence is not exact."""


@dataclass(frozen=True)
class BaselineCaptureRecord:
    record_id: str
    input_sha256: str
    input_payload: FrozenPreservationInput
    required_action_token_id: int | None


@dataclass(frozen=True)
class BaselineCaptureOutput:
    record_id: str
    base_indices: torch.Tensor
    base_probabilities: torch.Tensor
    assistant_positions: torch.Tensor
    sequence_length: int


class PreservationBaselineCaptureBackend(Protocol):
    """Adapter seam for one identity-bound, batched base-model forward pass."""

    @property
    def identity(self) -> Mapping[str, str]: ...

    def capture_batch(
        self, records: Sequence[BaselineCaptureRecord]
    ) -> Sequence[BaselineCaptureOutput]: ...


def _canonical_topk_preservation_targets(
    base_logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    required_token_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact logits using the total order (logit descending, token ID ascending)."""
    if not torch.isfinite(base_logits).all():
        raise ValueError("base logits must be finite")
    if not torch.isfinite(torch.tensor(float(temperature))) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if top_k < 2 or top_k > base_logits.shape[-1]:
        raise ValueError("top_k must be between 2 and the vocabulary size")
    if required_token_ids is not None:
        required = required_token_ids.to(device=base_logits.device, dtype=torch.int64)
        if required.shape != base_logits.shape[:-1]:
            raise ValueError("required token IDs must match the logits token dimensions")
    else:
        required = None

    scaled = base_logits / temperature
    flat_logits = scaled.reshape(-1, scaled.shape[-1])
    flat_required = required.reshape(-1) if required is not None else None
    selected_rows: list[torch.Tensor] = []
    for row_index, row in enumerate(flat_logits):
        # Only the cutoff set needs a vocabulary-wide scan.  Candidates above the
        # cutoff are bounded by top_k and ties at the cutoff are resolved by the
        # ascending IDs returned by nonzero.
        cutoff = torch.topk(row, k=top_k, sorted=False).values.min()
        above = torch.nonzero(row > cutoff, as_tuple=False).flatten()
        at_cutoff = torch.nonzero(row == cutoff, as_tuple=False).flatten()
        needed = top_k - int(above.numel())
        if needed < 0 or int(at_cutoff.numel()) < needed:
            raise ValueError("could not resolve the deterministic top-k cutoff")
        chosen = torch.cat((above, at_cutoff[:needed]))

        if flat_required is not None:
            required_id = flat_required[row_index]
            if required_id < 0 or required_id >= row.shape[-1]:
                raise ValueError("required token ID is outside the vocabulary")
            if not torch.any(chosen == required_id):
                # Replace the least-preferred selected token, then restore the
                # canonical total order below.
                chosen = chosen.clone()
                chosen[-1] = required_id

        # Sorting IDs first makes stable logit sorting resolve every equal-logit
        # group by token ID, including ties wholly above the cutoff.
        chosen = torch.sort(chosen).values
        order = torch.argsort(row[chosen], descending=True, stable=True)
        selected_rows.append(chosen[order])

    indices = torch.stack(selected_rows).reshape(*scaled.shape[:-1], top_k)
    selected = torch.gather(scaled, dim=-1, index=indices).float()
    log_normalizer = torch.logsumexp(scaled, dim=-1, keepdim=True).float()
    selected_probabilities = torch.exp(selected - log_normalizer)
    other = (1.0 - selected_probabilities.sum(dim=-1, keepdim=True)).clamp_min(
        torch.finfo(selected_probabilities.dtype).tiny
    )
    probabilities = torch.cat((selected_probabilities, other), dim=-1)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    if not torch.isfinite(probabilities).all():
        raise ValueError("compacted probabilities must be finite")
    return indices, probabilities


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PreservationBaselineCaptureError("value is not canonical JSON") from error


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PreservationBaselineCaptureError(f"file is unreadable: {path}") from error
    return digest.hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreservationBaselineCaptureError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PreservationBaselineCaptureError(f"{name} must be an array")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise PreservationBaselineCaptureError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PreservationBaselineCaptureError(f"{name} must be nonempty trimmed text")
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PreservationBaselineCaptureError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PreservationBaselineCaptureError(f"{name} must be a positive integer")
    return value


def _optional_token_id(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreservationBaselineCaptureError(
            f"{name} must be null or a non-negative integer"
        )
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not torch.isfinite(torch.tensor(float(value)))
        or float(value) <= 0
    ):
        raise PreservationBaselineCaptureError(f"{name} must be finite and positive")
    return float(value)


def _load_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreservationBaselineCaptureError(f"{name} is not strict JSON") from error


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _source_file(root: Path, value: Any, name: str) -> Path:
    relative = Path(_text(value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise PreservationBaselineCaptureError(f"{name} must stay below the plan directory")
    candidate = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PreservationBaselineCaptureError(f"{name} is missing or unreadable") from error
    if candidate.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise PreservationBaselineCaptureError(
            f"{name} must be a regular non-symlink file below the plan directory"
        )
    return resolved


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    system = platform.system()
    if system == "Darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(
            os.fsencode(source), os.fsencode(destination), ctypes.c_uint(0x00000004)
        )
    elif system == "Linux" and hasattr(libc, "renameat2"):
        result = libc.renameat2(
            ctypes.c_int(-100),
            os.fsencode(source),
            ctypes.c_int(-100),
            os.fsencode(destination),
            ctypes.c_uint(1),
        )
    else:  # pragma: no cover - production targets are macOS/Linux
        raise PreservationBaselineCaptureError(
            "atomic no-replace directory publication is unsupported"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PreservationBaselineCaptureError(f"output already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _parse_input(path: Path, record_id: str, expected_sha256: str) -> FrozenPreservationInput:
    try:
        content = path.read_bytes()
        raw = _object(json.loads(content.decode("utf-8")), f"input {record_id}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreservationBaselineCaptureError(f"input {record_id} is not strict JSON") from error
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise PreservationBaselineCaptureError(f"input {record_id} content hash differs")
    _exact(raw, {"messages", "media"}, f"input {record_id}")
    try:
        messages = _parse_messages(raw["messages"], record_id)
        media = _parse_media(raw["media"], path.parent, record_id)
        _validate_media_bindings(messages, media, record_id)
    except PreservationRuntimeError as error:
        raise PreservationBaselineCaptureError(
            f"input {record_id} is not a valid frozen preservation input"
        ) from error
    if messages[-1]["role"] != "assistant":
        raise PreservationBaselineCaptureError(
            f"input {record_id} must end with a teacher-forced assistant response"
        )
    assistant_content = messages[-1]["content"]
    if not isinstance(assistant_content, str):
        if not assistant_content or any(block["type"] != "text" for block in assistant_content):
            raise PreservationBaselineCaptureError(
                f"input {record_id} assistant response must contain only text"
            )
    return FrozenPreservationInput(
        record_id=record_id,
        messages=messages,
        media=media,
        source_sha256=expected_sha256,
    )


def _parse_plan(path: Path) -> tuple[dict[str, Any], tuple[BaselineCaptureRecord, ...]]:
    raw = _object(_load_json(path, "preservation baseline capture plan"), "plan")
    _exact(
        raw,
        {
            "format",
            *_IDENTITY_FIELDS,
            "batch_size",
            "top_k",
            "temperature",
            "records",
        },
        "preservation baseline capture plan",
    )
    if raw["format"] != PRESERVATION_BASELINE_CAPTURE_PLAN_FORMAT:
        raise PreservationBaselineCaptureError("unsupported baseline capture plan")
    normalized: dict[str, Any] = {
        "format": PRESERVATION_BASELINE_CAPTURE_PLAN_FORMAT,
        **{field: _sha(raw[field], f"plan.{field}") for field in _IDENTITY_FIELDS},
        "batch_size": _positive_integer(raw["batch_size"], "plan.batch_size"),
        "top_k": raw["top_k"],
        "temperature": _positive_float(raw["temperature"], "plan.temperature"),
    }
    if raw["top_k"] != 64:
        raise PreservationBaselineCaptureError("plan.top_k must be exactly 64")
    records_raw = _array(raw["records"], "plan.records")
    if not records_raw:
        raise PreservationBaselineCaptureError("plan.records must not be empty")
    records: list[BaselineCaptureRecord] = []
    normalized_records: list[dict[str, str]] = []
    for index, value in enumerate(records_raw):
        item = _object(value, f"plan.records[{index}]")
        _exact(
            item,
            {"record_id", "input_path", "input_sha256", "required_action_token_id"},
            f"plan.records[{index}]",
        )
        record_id = _text(item["record_id"], f"plan.records[{index}].record_id")
        input_sha256 = _sha(item["input_sha256"], f"plan.records[{index}].input_sha256")
        source = _source_file(path.parent, item["input_path"], f"plan.records[{index}].input_path")
        records.append(
            BaselineCaptureRecord(
                record_id=record_id,
                input_sha256=input_sha256,
                input_payload=_parse_input(source, record_id, input_sha256),
                required_action_token_id=_optional_token_id(
                    item["required_action_token_id"],
                    f"plan.records[{index}].required_action_token_id",
                ),
            )
        )
        normalized_records.append(
            {
                "record_id": record_id,
                "input_path": str(Path(item["input_path"])),
                "input_sha256": input_sha256,
                "required_action_token_id": item["required_action_token_id"],
            }
        )
    ids = tuple(record.record_id for record in records)
    if len(set(ids)) != len(ids):
        raise PreservationBaselineCaptureError("plan record IDs must be unique")
    normalized["records"] = normalized_records
    return normalized, tuple(records)


def _validate_backend_identity(
    raw: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, str]:
    _exact(raw, _IDENTITY_FIELDS, "capture backend identity")
    normalized = {field: _sha(raw[field], f"backend.{field}") for field in _IDENTITY_FIELDS}
    if any(normalized[field] != expected[field] for field in _IDENTITY_FIELDS):
        raise PreservationBaselineCaptureError("capture backend identity differs from plan")
    return normalized


def _validate_output(
    output: BaselineCaptureOutput, expected_record_id: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(output, BaselineCaptureOutput):
        raise PreservationBaselineCaptureError("capture backend returned the wrong output type")
    if output.record_id != expected_record_id:
        raise PreservationBaselineCaptureError("capture output record order or identity differs")
    indices = output.base_indices
    probabilities = output.base_probabilities
    positions = output.assistant_positions
    sequence_length = output.sequence_length
    if (
        not isinstance(indices, torch.Tensor)
        or indices.ndim != 3
        or indices.shape[0] != 1
        or indices.shape[-1] != 64
        or not isinstance(probabilities, torch.Tensor)
        or probabilities.shape != (*indices.shape[:-1], 65)
        or not isinstance(positions, torch.Tensor)
        or positions.shape != indices.shape[:-1]
    ):
        raise PreservationBaselineCaptureError("compact capture tensor shapes are incompatible")
    if indices.shape[1] < 1:
        raise PreservationBaselineCaptureError("capture has no assistant next-token positions")
    if indices.dtype != torch.int64 or positions.dtype != torch.int64:
        raise PreservationBaselineCaptureError("compact capture indices must use int64")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 2:
        raise PreservationBaselineCaptureError("capture sequence length is invalid")
    if torch.any(indices < 0):
        raise PreservationBaselineCaptureError("compact capture token indices are invalid")
    if torch.any(positions < 0) or torch.any(positions >= sequence_length - 1):
        raise PreservationBaselineCaptureError("assistant positions are outside the sequence")
    if positions.shape[1] > 1 and torch.any(positions[:, 1:] <= positions[:, :-1]):
        raise PreservationBaselineCaptureError("assistant positions must be strictly increasing")
    if any(torch.unique(row).numel() != 64 for row in indices.reshape(-1, 64)):
        raise PreservationBaselineCaptureError("top-64 token indices must be unique")
    if (
        not torch.isfinite(probabilities).all()
        or torch.any(probabilities < 0)
        or not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        )
    ):
        raise PreservationBaselineCaptureError("compact capture probabilities are invalid")
    return (
        indices.detach().to(device="cpu").contiguous(),
        probabilities.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        positions.detach().to(device="cpu").contiguous(),
        torch.tensor([sequence_length], dtype=torch.int64),
    )


def _materialize_capture(
    *,
    plan: Mapping[str, Any],
    records: Sequence[BaselineCaptureRecord],
    staging: Path,
    backend: PreservationBaselineCaptureBackend,
) -> dict[str, Any]:
    identity = _validate_backend_identity(_object(backend.identity, "backend identity"), plan)
    artifact_hashes: dict[str, str] = {}
    captured_records: list[dict[str, Any]] = []
    batch_size = int(plan["batch_size"])
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        try:
            outputs = tuple(backend.capture_batch(batch))
        except PreservationBaselineCaptureError:
            raise
        except Exception as error:
            raise PreservationBaselineCaptureError(
                f"capture backend failed: {type(error).__name__}: {error}"
            ) from error
        if len(outputs) != len(batch):
            raise PreservationBaselineCaptureError("capture output record count differs")
        for local_index, (record, output) in enumerate(zip(batch, outputs, strict=True)):
            index = offset + local_index
            indices, probabilities, positions, sequence_length = _validate_output(
                output, record.record_id
            )
            tensor_relative = f"base-logits/{index:04d}.safetensors"
            tensor_path = staging / tensor_relative
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    "base_indices": indices,
                    "base_probabilities": probabilities,
                    "assistant_positions": positions,
                    "sequence_length": sequence_length,
                },
                tensor_path,
            )
            # Reopen the exact bytes before they become evidence.
            reopened = load_file(tensor_path)
            if set(reopened) != {
                "base_indices",
                "base_probabilities",
                "assistant_positions",
                "sequence_length",
            }:
                raise PreservationBaselineCaptureError("stored capture tensor inventory differs")
            tensor_sha256 = _hash_file(tensor_path)
            unsigned = {
                "format": PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT,
                "record_id": record.record_id,
                "base_logits_sha256": tensor_sha256,
                "input_sha256": record.input_sha256,
                "representation": PRESERVATION_COMPACT_CAPTURE_REPRESENTATION,
                "top_k": 64,
                "temperature": plan["temperature"],
                "sequence_length": int(sequence_length.item()),
                "assistant_position_count": int(positions.shape[1]),
                **identity,
            }
            # base_model_sha256 is already in identity; keep the materializer's
            # exact receipt shape and no backend-only fields.
            receipt = {**unsigned, "self_sha256": _hash_json(unsigned)}
            receipt_relative = f"capture-receipts/{index:04d}.json"
            receipt_path = staging / receipt_relative
            _write_json(receipt_path, receipt)
            artifact_hashes[tensor_relative] = tensor_sha256
            artifact_hashes[receipt_relative] = _hash_file(receipt_path)
            captured_records.append(
                {
                    "record_id": record.record_id,
                    "input_sha256": record.input_sha256,
                    "base_logits_path": tensor_relative,
                    "base_logits_sha256": tensor_sha256,
                    "base_logits_capture_receipt_path": receipt_relative,
                    "base_logits_capture_receipt_sha256": artifact_hashes[receipt_relative],
                }
            )
    unsigned_run = {
        "format": PRESERVATION_BASELINE_CAPTURE_RUN_FORMAT,
        "plan_sha256": _hash_json(plan),
        "record_count": len(records),
        "backend_identity": identity,
        "records": captured_records,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
    }
    run = {**unsigned_run, "self_sha256": _hash_json(unsigned_run)}
    _write_json(staging / "capture-run-receipt.json", run)
    return run


def capture_preservation_baselines(
    plan_path: Path | str,
    output_dir: Path | str,
    *,
    backend: PreservationBaselineCaptureBackend | None = None,
    model_config: ModelLoadConfig | None = None,
) -> dict[str, Any]:
    """Capture one immutable base-logits bundle, refusing existing outputs."""

    try:
        plan_path = Path(plan_path).resolve(strict=True)
    except OSError as error:
        raise PreservationBaselineCaptureError("baseline capture plan is missing") from error
    plan, records = _parse_plan(plan_path)
    if backend is None:
        config = model_config or ModelLoadConfig()
        backend = VerifiedQwenBaselineCaptureBackend(
            model_config=config,
            expected_tokenizer_sha256=plan["tokenizer_sha256"],
            expected_processor_sha256=plan["processor_sha256"],
            expected_chat_template_sha256=plan["chat_template_sha256"],
            temperature=plan["temperature"],
        )
    requested = Path(output_dir).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    output = requested.parent.resolve() / requested.name
    if os.path.lexists(output):
        raise PreservationBaselineCaptureError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        receipt = _materialize_capture(
            plan=plan, records=records, staging=staging, backend=backend
        )
        _rename_no_replace(staging, output)
        return receipt
    except PreservationBaselineCaptureError:
        shutil.rmtree(staging)
        raise
    except Exception as error:
        shutil.rmtree(staging)
        raise PreservationBaselineCaptureError(
            f"baseline capture failed: {type(error).__name__}: {error}"
        ) from error


def open_preservation_baseline_capture(root: Path | str) -> dict[str, Any]:
    """Strictly reopen and verify every captured artifact and run identity."""

    requested = Path(root)
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise PreservationBaselineCaptureError("baseline capture directory is missing") from error
    if requested.is_symlink() or not resolved.is_dir():
        raise PreservationBaselineCaptureError("baseline capture must be a regular directory")
    if any(path.is_symlink() for path in resolved.rglob("*")):
        raise PreservationBaselineCaptureError("baseline capture must not contain symlinks")
    raw = _object(_load_json(resolved / "capture-run-receipt.json", "capture run receipt"), "run receipt")
    _exact(
        raw,
        {
            "format",
            "plan_sha256",
            "record_count",
            "backend_identity",
            "records",
            "artifact_sha256",
            "self_sha256",
        },
        "capture run receipt",
    )
    if raw["format"] != PRESERVATION_BASELINE_CAPTURE_RUN_FORMAT:
        raise PreservationBaselineCaptureError("unsupported capture run receipt")
    claimed = _sha(raw["self_sha256"], "capture run receipt.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if _hash_json(unsigned) != claimed:
        raise PreservationBaselineCaptureError("capture run receipt hash mismatch")
    count = _positive_integer(raw["record_count"], "capture run receipt.record_count")
    records = _array(raw["records"], "capture run receipt.records")
    if len(records) != count:
        raise PreservationBaselineCaptureError("capture run record count differs")
    identity = _object(raw["backend_identity"], "capture run backend identity")
    _exact(identity, _IDENTITY_FIELDS, "capture run backend identity")
    for field in _IDENTITY_FIELDS:
        _sha(identity[field], f"capture run backend identity.{field}")
    artifacts = _object(raw["artifact_sha256"], "capture run artifact_sha256")
    expected_paths: set[str] = set()
    for raw_path, digest in artifacts.items():
        relative = Path(_text(raw_path, "capture artifact path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise PreservationBaselineCaptureError("capture artifact path escapes directory")
        expected_paths.add(str(relative))
        if _hash_file(resolved / relative) != _sha(digest, f"capture artifact {relative}"):
            raise PreservationBaselineCaptureError("capture artifact content hash differs")
    observed_paths = {
        str(path.relative_to(resolved))
        for path in resolved.rglob("*")
        if path.is_file() and path.name != "capture-run-receipt.json"
    }
    if observed_paths != expected_paths:
        raise PreservationBaselineCaptureError("capture artifact inventory differs")
    seen_ids: set[str] = set()
    for index, value in enumerate(records):
        record = _object(value, f"capture run records[{index}]")
        _exact(
            record,
            {
                "record_id",
                "input_sha256",
                "base_logits_path",
                "base_logits_sha256",
                "base_logits_capture_receipt_path",
                "base_logits_capture_receipt_sha256",
            },
            f"capture run records[{index}]",
        )
        record_id = _text(record["record_id"], f"capture run records[{index}].record_id")
        if record_id in seen_ids:
            raise PreservationBaselineCaptureError("capture run record IDs must be unique")
        seen_ids.add(record_id)
        tensor_relative = Path(_text(record["base_logits_path"], "capture tensor path"))
        receipt_relative = Path(
            _text(record["base_logits_capture_receipt_path"], "capture receipt path")
        )
        if (
            tensor_relative.is_absolute()
            or receipt_relative.is_absolute()
            or ".." in tensor_relative.parts
            or ".." in receipt_relative.parts
        ):
            raise PreservationBaselineCaptureError("capture record path escapes directory")
        tensor_sha = _sha(record["base_logits_sha256"], "capture tensor SHA")
        receipt_sha = _sha(
            record["base_logits_capture_receipt_sha256"], "capture receipt SHA"
        )
        if (
            artifacts.get(str(tensor_relative)) != tensor_sha
            or artifacts.get(str(receipt_relative)) != receipt_sha
        ):
            raise PreservationBaselineCaptureError("capture record artifact binding differs")
        receipt = _object(
            _load_json(resolved / receipt_relative, "compact capture receipt"),
            "compact capture receipt",
        )
        receipt_fields = {
            "format",
            "record_id",
            "base_logits_sha256",
            "input_sha256",
            "representation",
            "top_k",
            "temperature",
            "sequence_length",
            "assistant_position_count",
            *_IDENTITY_FIELDS,
            "self_sha256",
        }
        _exact(receipt, receipt_fields, "compact capture receipt")
        if receipt["format"] != PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT:
            raise PreservationBaselineCaptureError("unsupported compact capture receipt")
        receipt_claimed = _sha(receipt["self_sha256"], "compact capture receipt SHA")
        receipt_unsigned = dict(receipt)
        del receipt_unsigned["self_sha256"]
        if _hash_json(receipt_unsigned) != receipt_claimed:
            raise PreservationBaselineCaptureError("compact capture receipt hash mismatch")
        if receipt["representation"] not in {
            LEGACY_COMPACT_CAPTURE_REPRESENTATION,
            PRESERVATION_COMPACT_CAPTURE_REPRESENTATION,
        }:
            raise PreservationBaselineCaptureError("unsupported compact capture representation")
        if receipt["top_k"] != 64:
            raise PreservationBaselineCaptureError("compact capture top_k differs")
        _positive_float(receipt["temperature"], "compact capture receipt.temperature")
        _positive_integer(
            receipt["sequence_length"], "compact capture receipt.sequence_length"
        )
        _positive_integer(
            receipt["assistant_position_count"],
            "compact capture receipt.assistant_position_count",
        )
        expected_receipt = {
            "record_id": record_id,
            "input_sha256": record["input_sha256"],
            "base_logits_sha256": tensor_sha,
            **identity,
        }
        if any(receipt[field] != expected for field, expected in expected_receipt.items()):
            raise PreservationBaselineCaptureError("compact capture receipt binding differs")
        try:
            tensors = load_file(resolved / tensor_relative)
            compact = BaselineCaptureOutput(
                record_id=record_id,
                base_indices=tensors["base_indices"],
                base_probabilities=tensors["base_probabilities"],
                assistant_positions=tensors["assistant_positions"],
                sequence_length=int(tensors["sequence_length"].item()),
            )
        except (KeyError, RuntimeError, ValueError) as error:
            raise PreservationBaselineCaptureError(
                "compact capture tensor packet is unreadable"
            ) from error
        _, _, positions, sequence_length = _validate_output(compact, record_id)
        if (
            receipt["sequence_length"] != int(sequence_length.item())
            or receipt["assistant_position_count"] != int(positions.shape[1])
        ):
            raise PreservationBaselineCaptureError("compact capture receipt metadata differs")
    return dict(raw)


def qwen_preservation_capture_runtime_sha256(config: ModelLoadConfig) -> str:
    """Return the frozen runtime identity used in every capture receipt."""

    versions: dict[str, str] = {}
    for distribution in ("torch", "transformers", "safetensors", "qwen-vl-utils"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return _hash_json(
        {
            "format": QWEN_BASELINE_CAPTURE_RUNTIME_FORMAT,
            "model_id": config.model_name,
            "revision": config.revision,
            "snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256,
            "dtype": "torch.bfloat16",
            "device_map": config.device_map,
            "attention_implementation": config.attn_implementation,
            "local_files_only": config.local_files_only,
            "use_cache": config.use_cache,
            "forward_use_cache": False,
            "teacher_forcing": "full_conversation_final_assistant_suffix_v1",
            "batch_padding": "left_prefix_alignment_v1",
            "capture_representation": PRESERVATION_COMPACT_CAPTURE_REPRESENTATION,
            "top_k_tie_break": "logit_descending_then_token_id_ascending_v1",
            "top_k_selection_device": "cpu",
            "probability_storage_dtype": "torch.float32",
            "index_storage_dtype": "torch.int64",
            "versions": versions,
        }
    )


def _default_processor_identity(bundle: ModelBundle) -> str:
    tokenizer = bundle.tokenizer
    processor = bundle.processor
    return _hash_json(
        {
            "format": "truth_editing_qwen_processor_identity_v1",
            "snapshot_manifest_sha256": bundle.config.expected_snapshot_manifest_sha256,
            "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
            "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "padding_side": getattr(tokenizer, "padding_side", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
    )


def _default_vision_info(
    conversations: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[Any], list[Any]]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as error:  # pragma: no cover - production dependency
        raise PreservationBaselineCaptureError(
            "multimodal capture requires qwen_vl_utils"
        ) from error
    images: list[Any] = []
    videos: list[Any] = []
    for conversation in conversations:
        image_values, video_values = process_vision_info(list(conversation))
        if image_values:
            images.extend(image_values)
        if video_values:
            videos.extend(video_values)
    return images, videos


class VerifiedQwenBaselineCaptureBackend:
    """Batched teacher-forced capture over one exactly verified Qwen bundle."""

    def __init__(
        self,
        *,
        model_config: ModelLoadConfig | None = None,
        bundle_loader: Callable[[ModelLoadConfig], ModelBundle] = load_model_and_processor,
        expected_tokenizer_sha256: str,
        expected_processor_sha256: str,
        expected_chat_template_sha256: str,
        enforce_production_runtime: bool = True,
        vision_info_loader: Callable[
            [Sequence[Sequence[Mapping[str, Any]]]], tuple[list[Any], list[Any]]
        ] = _default_vision_info,
        processor_identity_resolver: Callable[[ModelBundle], str] = _default_processor_identity,
        inference_runtime_sha256: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        self._config = model_config or ModelLoadConfig()
        self._loader = bundle_loader
        self._expected_processor = _sha(expected_processor_sha256, "expected processor SHA")
        self._expected_template = _sha(expected_chat_template_sha256, "expected chat template SHA")
        self._vision_info = vision_info_loader
        self._processor_identity = processor_identity_resolver
        self._enforce_production = enforce_production_runtime
        self._temperature = _positive_float(temperature, "capture temperature")
        runtime_sha = inference_runtime_sha256 or qwen_preservation_capture_runtime_sha256(
            self._config
        )
        self._identity = {
            "base_model_sha256": _sha(
                self._config.expected_model_sha256, "configured model SHA"
            ),
            "tokenizer_sha256": _sha(expected_tokenizer_sha256, "expected tokenizer SHA"),
            "processor_sha256": self._expected_processor,
            "chat_template_sha256": self._expected_template,
            "inference_runtime_sha256": _sha(runtime_sha, "inference runtime SHA"),
        }
        if enforce_production_runtime and (
            self._config.expected_model_sha256 != DEFAULT_MODEL_CONTENT_SHA256
            or self._config.expected_snapshot_manifest_sha256
            != DEFAULT_SNAPSHOT_MANIFEST_SHA256
            or bundle_loader is not load_model_and_processor
        ):
            raise PreservationBaselineCaptureError(
                "production capture requires the exact verified Qwen loader and snapshot"
            )
        self._bundle: ModelBundle | None = None

    @property
    def identity(self) -> Mapping[str, str]:
        return dict(self._identity)

    def _bundle_once(self) -> ModelBundle:
        if self._bundle is None:
            bundle = self._loader(self._config)
            if bundle.model is None:
                raise PreservationBaselineCaptureError("Qwen capture loader returned no model")
            expected_snapshot = {
                "model_id": DEFAULT_MODEL_ID,
                "revision": DEFAULT_MODEL_REVISION,
                "model_sha256": self._config.expected_model_sha256,
                "snapshot_manifest_sha256": self._config.expected_snapshot_manifest_sha256,
            }
            if (
                bundle.model_id != DEFAULT_MODEL_ID
                or bundle.model_revision != DEFAULT_MODEL_REVISION
                or bundle.verified_snapshot != expected_snapshot
            ):
                raise PreservationBaselineCaptureError("loaded Qwen snapshot identity differs")
            template = getattr(bundle.processor, "chat_template", None) or getattr(
                bundle.tokenizer, "chat_template", None
            )
            if (
                not isinstance(template, str)
                or hashlib.sha256(template.encode()).hexdigest() != self._expected_template
            ):
                raise PreservationBaselineCaptureError("loaded Qwen chat template differs")
            measured_processor = _sha(
                self._processor_identity(bundle), "measured processor identity"
            )
            if measured_processor != self._expected_processor:
                raise PreservationBaselineCaptureError("loaded Qwen processor identity differs")
            if self._enforce_production:
                self._verify_tokenizer_manifest()
                self._verify_production_bundle(bundle)
            bundle.model.eval()
            self._bundle = bundle
        return self._bundle

    def _verify_tokenizer_manifest(self) -> None:
        manifest_path = self._config.snapshot_manifest_path
        if manifest_path is None:
            raise PreservationBaselineCaptureError(
                "production capture requires a snapshot manifest"
            )
        try:
            raw = _object(_load_json(Path(manifest_path), "Qwen snapshot manifest"), "manifest")
            files = _array(raw["files"], "Qwen snapshot manifest.files")
            by_path = {
                _text(_object(item, "snapshot file")["path"], "snapshot file.path"):
                _sha(_object(item, "snapshot file")["sha256"], "snapshot file.sha256")
                for item in files
            }
        except (KeyError, PreservationBaselineCaptureError) as error:
            raise PreservationBaselineCaptureError(
                "Qwen snapshot manifest file inventory is invalid"
            ) from error
        if by_path.get("tokenizer.json") != self._identity["tokenizer_sha256"]:
            raise PreservationBaselineCaptureError(
                "Qwen snapshot tokenizer identity differs from capture plan"
            )

    def _verify_production_bundle(self, bundle: ModelBundle) -> None:
        assert bundle.model is not None
        try:
            parameters = tuple(bundle.model.parameters())
        except (AttributeError, TypeError) as error:
            raise PreservationBaselineCaptureError("Qwen model exposes no parameters") from error
        if not parameters:
            raise PreservationBaselineCaptureError("Qwen model exposes no parameters")
        if {str(item.device) for item in parameters} != {QWEN_DEVICE_MAP}:
            raise PreservationBaselineCaptureError("Qwen capture parameters must be on cuda:0")
        floating = {item.dtype for item in parameters if torch.is_floating_point(item)}
        if floating != {torch.bfloat16}:
            raise PreservationBaselineCaptureError("Qwen capture parameters must be BF16")
        implementation = getattr(bundle.model.config, "_attn_implementation", None)
        if implementation != QWEN_ATTENTION_IMPLEMENTATION:
            raise PreservationBaselineCaptureError("Qwen capture requires FlashAttention 2")
        if getattr(bundle.model.config, "use_cache", None) is not True:
            raise PreservationBaselineCaptureError("Qwen capture model must retain use_cache=true")

    def capture_batch(
        self, records: Sequence[BaselineCaptureRecord]
    ) -> Sequence[BaselineCaptureOutput]:
        if not records:
            return ()
        bundle = self._bundle_once()
        assert bundle.model is not None
        conversations = [record.input_payload.resolved_messages() for record in records]
        full_texts: list[str] = []
        prefix_texts: list[str] = []
        marker = "<|im_start|>assistant\n"
        for conversation in conversations:
            rendered = bundle.processor.apply_chat_template(
                [dict(message) for message in conversation],
                tokenize=False,
                add_generation_prompt=False,
            )
            if not isinstance(rendered, str):
                raise PreservationBaselineCaptureError("Qwen chat template returned non-text")
            marker_index = rendered.rfind(marker)
            if marker_index < 0:
                raise PreservationBaselineCaptureError(
                    "Qwen rendered conversation lacks final assistant marker"
                )
            full_texts.append(rendered)
            prefix_texts.append(rendered[: marker_index + len(marker)])
        images, videos = self._vision_info(conversations)
        common: dict[str, Any] = {"padding": True, "return_tensors": "pt"}
        if images:
            common["images"] = images
        if videos:
            common["videos"] = videos
        full = bundle.processor(text=full_texts, **common)
        prefix = bundle.processor(text=prefix_texts, **common)
        if not isinstance(full, Mapping) or not isinstance(prefix, Mapping):
            raise PreservationBaselineCaptureError("Qwen processor returned non-mapping inputs")
        full_ids = full.get("input_ids")
        full_mask = full.get("attention_mask")
        prefix_ids = prefix.get("input_ids")
        prefix_mask = prefix.get("attention_mask")
        if any(
            not isinstance(value, torch.Tensor) or value.ndim != 2
            for value in (full_ids, full_mask, prefix_ids, prefix_mask)
        ):
            raise PreservationBaselineCaptureError(
                "Qwen processor must return rank-two IDs and attention masks"
            )
        assert isinstance(full_ids, torch.Tensor)
        assert isinstance(full_mask, torch.Tensor)
        assert isinstance(prefix_ids, torch.Tensor)
        assert isinstance(prefix_mask, torch.Tensor)
        if full_ids.shape != full_mask.shape or prefix_ids.shape != prefix_mask.shape:
            raise PreservationBaselineCaptureError("Qwen processor masks are misaligned")
        if full_ids.shape[0] != len(records) or prefix_ids.shape[0] != len(records):
            raise PreservationBaselineCaptureError("Qwen processor batch size differs")
        slices: list[tuple[int, int, int]] = []
        for index in range(len(records)):
            full_active = full_ids[index][full_mask[index].bool()]
            prefix_active = prefix_ids[index][prefix_mask[index].bool()]
            if (
                prefix_active.numel() >= full_active.numel()
                or not torch.equal(full_active[: prefix_active.numel()], prefix_active)
            ):
                raise PreservationBaselineCaptureError(
                    "Qwen teacher-forced assistant prefix is not token-aligned"
                )
            active_positions = torch.nonzero(full_mask[index], as_tuple=False).flatten()
            if active_positions.numel() != full_active.numel():
                raise PreservationBaselineCaptureError("Qwen attention mask is invalid")
            start, stop = int(active_positions[0]), int(active_positions[-1]) + 1
            slices.append((start, stop, int(prefix_active.numel())))
        device = next(bundle.model.parameters()).device
        moved = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in full.items()
        }
        with torch.inference_mode():
            raw_output = bundle.model(**moved, use_cache=False)
        logits = getattr(raw_output, "logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != full_ids.shape
            or not torch.isfinite(logits).all()
        ):
            raise PreservationBaselineCaptureError("Qwen capture logits are invalid")
        outputs: list[BaselineCaptureOutput] = []
        for index, record in enumerate(records):
            start, stop, prefix_length = slices[index]
            active_ids = full_ids[index, start:stop].detach().to(device="cpu")
            labels = torch.full_like(active_ids, -100, dtype=torch.int64)
            labels[prefix_length:] = active_ids[prefix_length:].to(torch.int64)
            shifted_labels = labels[1:]
            assistant_positions = torch.nonzero(
                shifted_labels != -100, as_tuple=False
            ).flatten()
            if assistant_positions.numel() == 0:
                raise PreservationBaselineCaptureError(
                    "Qwen capture has no assistant next-token positions"
                )
            # Resolve the artifact-defining total order on CPU.  This avoids
            # backend-specific CUDA top-k tie membership becoming receipt identity.
            selected_logits = logits[
                index : index + 1,
                start + assistant_positions.to(device=logits.device),
                :,
            ].detach().to(device="cpu", dtype=torch.float32)
            required = shifted_labels[assistant_positions].unsqueeze(0).to(device="cpu")
            if record.required_action_token_id is not None:
                if record.required_action_token_id >= selected_logits.shape[-1]:
                    raise PreservationBaselineCaptureError(
                        "required action token is outside the Qwen vocabulary"
                    )
                required = torch.full_like(required, record.required_action_token_id)
            try:
                base_indices, base_probabilities = _canonical_topk_preservation_targets(
                    selected_logits,
                    top_k=64,
                    temperature=self._temperature,
                    required_token_ids=required,
                )
            except Exception as error:
                raise PreservationBaselineCaptureError(
                    "Qwen compact top-64 capture failed"
                ) from error
            outputs.append(
                BaselineCaptureOutput(
                    record_id=record.record_id,
                    base_indices=base_indices.detach().to(device="cpu", dtype=torch.int64),
                    base_probabilities=base_probabilities.detach().to(
                        device="cpu", dtype=torch.float32
                    ),
                    assistant_positions=assistant_positions.unsqueeze(0).to(
                        device="cpu", dtype=torch.int64
                    ),
                    sequence_length=stop - start,
                )
            )
        return tuple(outputs)
