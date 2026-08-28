"""Offline materialization of frozen capability-preservation runtime packets.

The module owns one deep seam: :func:`materialize_preservation_runtime_packet`
turns an identity-bound plan, frozen input sidecars, media, and base-model
logits into the complete packet consumed by ``TrialPreservationCollector``.
It never loads a language model, opens a network connection, or uses a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import ctypes
import errno
import io
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from safetensors.torch import load as load_safetensors

from .truth_editing_preservation import (
    CachedPreservationBaseline,
    PreservationRecord,
    PreservationSpec,
    build_cached_baseline,
)
from .truth_editing_preservation_runtime import (
    PRESERVATION_RUNTIME_CONFIG_FORMAT,
    PreservationRuntimeConfig,
    TrialPreservationCollector,
)


PRESERVATION_MATERIALIZATION_PLAN_FORMAT = (
    "truth_editing_preservation_materialization_plan_v1"
)
PRESERVATION_MATERIALIZATION_RECEIPT_FORMAT = (
    "truth_editing_preservation_materialization_receipt_v1"
)
PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT = (
    "truth_editing_preservation_base_logits_capture_receipt_v2"
)
PRESERVATION_COMPACT_CAPTURE_REPRESENTATION = (
    "assistant_top64_plus_other_token_id_tiebreak_v2"
)
LEGACY_COMPACT_CAPTURE_REPRESENTATION = "assistant_top64_plus_other_v1"
_LEGACY_PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT = (
    "truth_editing_preservation_base_logits_capture_receipt_v1"
)
_TIERS = ("trial", "promoted", "finalist")
_HEX = frozenset("0123456789abcdef")


class PreservationMaterializationError(RuntimeError):
    """The frozen materialization plan or one of its inputs is not exact."""


def _runtime_config_name(tier: str) -> str:
    return f"truth_editing_preservation_runtime_{tier}_v1.json"


def parse_preservation_materialization_receipt(value: Any) -> dict[str, Any]:
    """Strictly validate and normalize a materialization receipt."""

    raw = _object(value, "preservation materialization receipt")
    _exact(
        raw,
        {
            "format",
            "plan_sha256",
            "spec_sha256",
            "record_count",
            "runtime_config_sha256",
            "artifact_sha256",
            "self_sha256",
        },
        "preservation materialization receipt",
    )
    if raw["format"] != PRESERVATION_MATERIALIZATION_RECEIPT_FORMAT:
        raise PreservationMaterializationError(
            "unsupported preservation materialization receipt"
        )
    claimed = _sha(raw["self_sha256"], "materialization receipt.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if _hash_json(unsigned) != claimed:
        raise PreservationMaterializationError("materialization receipt hash mismatch")
    count = raw["record_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise PreservationMaterializationError(
            "materialization receipt record_count must be positive"
        )
    config_hashes = _object(
        raw["runtime_config_sha256"], "materialization receipt.runtime_config_sha256"
    )
    _exact(config_hashes, set(_TIERS), "materialization receipt.runtime_config_sha256")
    normalized_configs = {
        tier: _sha(config_hashes[tier], f"materialization receipt config {tier}")
        for tier in _TIERS
    }
    artifact_hashes = _object(
        raw["artifact_sha256"], "materialization receipt.artifact_sha256"
    )
    if not artifact_hashes:
        raise PreservationMaterializationError(
            "materialization receipt artifact hashes must not be empty"
        )
    normalized_artifacts: dict[str, str] = {}
    for path_value, digest in artifact_hashes.items():
        path = Path(_text(path_value, "materialization receipt artifact path"))
        if path.is_absolute() or ".." in path.parts:
            raise PreservationMaterializationError(
                "materialization receipt artifact path escapes the packet"
            )
        normalized_artifacts[str(path)] = _sha(
            digest, f"materialization receipt artifact {path}"
        )
    return {
        "format": PRESERVATION_MATERIALIZATION_RECEIPT_FORMAT,
        "plan_sha256": _sha(raw["plan_sha256"], "materialization receipt.plan_sha256"),
        "spec_sha256": _sha(raw["spec_sha256"], "materialization receipt.spec_sha256"),
        "record_count": count,
        "runtime_config_sha256": normalized_configs,
        "artifact_sha256": normalized_artifacts,
        "self_sha256": claimed,
    }


def _parse_capture_receipt(value: Any) -> dict[str, Any]:
    raw = _object(value, "base logits capture receipt")
    legacy_fields = {
        "format",
        "record_id",
        "base_logits_sha256",
        "input_sha256",
        "base_model_sha256",
        "tokenizer_sha256",
        "processor_sha256",
        "chat_template_sha256",
        "inference_runtime_sha256",
        "self_sha256",
    }
    compact_fields = legacy_fields | {
        "representation",
        "top_k",
        "temperature",
        "sequence_length",
        "assistant_position_count",
    }
    is_legacy = raw.get("format") == _LEGACY_PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT
    fields = legacy_fields if is_legacy else compact_fields
    _exact(raw, fields, "base logits capture receipt")
    if raw["format"] not in {
        _LEGACY_PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT,
        PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT,
    }:
        raise PreservationMaterializationError("unsupported base logits capture receipt")
    claimed = _sha(raw["self_sha256"], "capture receipt.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if _hash_json(unsigned) != claimed:
        raise PreservationMaterializationError("base logits capture receipt hash mismatch")
    normalized: dict[str, Any] = {
        "format": raw["format"],
        "record_id": _text(raw["record_id"], "capture receipt.record_id"),
    }
    for field in legacy_fields - {"format", "record_id"}:
        normalized[field] = _sha(raw[field], f"capture receipt.{field}")
    if not is_legacy:
        if raw["representation"] not in {
            LEGACY_COMPACT_CAPTURE_REPRESENTATION,
            PRESERVATION_COMPACT_CAPTURE_REPRESENTATION,
        }:
            raise PreservationMaterializationError("unsupported compact capture representation")
        if raw["top_k"] != 64:
            raise PreservationMaterializationError("compact capture top_k must be exactly 64")
        temperature = raw["temperature"]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not torch.isfinite(torch.tensor(float(temperature)))
            or float(temperature) <= 0
        ):
            raise PreservationMaterializationError("compact capture temperature is invalid")
        for field in ("sequence_length", "assistant_position_count"):
            count = raw[field]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise PreservationMaterializationError(f"compact capture {field} is invalid")
        if raw["sequence_length"] < 2:
            raise PreservationMaterializationError("compact capture sequence is too short")
        normalized.update(
            representation=raw["representation"],
            top_k=64,
            temperature=float(temperature),
            sequence_length=raw["sequence_length"],
            assistant_position_count=raw["assistant_position_count"],
        )
    return normalized


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
        raise PreservationMaterializationError("value is not canonical JSON") from error


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PreservationMaterializationError(f"source file is unreadable: {path}") from error
    return digest.hexdigest()


def _read_source_bytes(path: Path, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise PreservationMaterializationError(f"{name} is unreadable") from error


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreservationMaterializationError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PreservationMaterializationError(f"{name} must be an array")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise PreservationMaterializationError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PreservationMaterializationError(f"{name} must be a nonempty trimmed string")
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PreservationMaterializationError(f"{name} must be a lowercase SHA-256")
    return value


def _source_file(root: Path, value: Any, name: str) -> Path:
    relative = Path(_text(value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise PreservationMaterializationError(f"{name} must stay below the plan directory")
    candidate = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PreservationMaterializationError(f"{name} is missing or unreadable") from error
    if candidate.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise PreservationMaterializationError(
            f"{name} must be a regular non-symlink file below the plan directory"
        )
    return resolved


def _load_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreservationMaterializationError(f"{name} is not strict JSON") from error


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing inode."""

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
    elif system == "Windows":
        os.rename(source, destination)
        return
    else:
        raise PreservationMaterializationError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PreservationMaterializationError(f"output already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def open_preservation_runtime_packet(root: Path | str) -> dict[str, Any]:
    """Verify every packet artifact and all three runtime configs from disk."""

    packet_root = Path(root)
    try:
        resolved_root = packet_root.resolve(strict=True)
    except OSError as error:
        raise PreservationMaterializationError("preservation packet is missing") from error
    if packet_root.is_symlink() or not resolved_root.is_dir():
        raise PreservationMaterializationError(
            "preservation packet must be a regular non-symlink directory"
        )
    receipt_path = resolved_root / "materialization-receipt.json"
    if receipt_path.is_symlink():
        raise PreservationMaterializationError(
            "preservation materialization receipt must not be a symlink"
        )
    receipt = parse_preservation_materialization_receipt(
        _load_json(receipt_path, "preservation materialization receipt")
    )
    if any(path.is_symlink() for path in resolved_root.rglob("*")):
        raise PreservationMaterializationError(
            "preservation packet must not contain symlinks"
        )
    expected_paths = set(receipt["artifact_sha256"])
    observed_paths = {
        str(path.relative_to(resolved_root))
        for path in resolved_root.rglob("*")
        if path.is_file() and path != receipt_path
    }
    if observed_paths != expected_paths:
        raise PreservationMaterializationError(
            "preservation packet artifact inventory differs from its receipt"
        )
    for relative, expected in receipt["artifact_sha256"].items():
        path = resolved_root / relative
        if path.is_symlink() or _hash_file(path) != expected:
            raise PreservationMaterializationError(
                f"preservation packet artifact content hash differs: {relative}"
            )
    for tier in _TIERS:
        config = PreservationRuntimeConfig.load(resolved_root / _runtime_config_name(tier))
        if config.config_sha256 != receipt["runtime_config_sha256"][tier]:
            raise PreservationMaterializationError(
                f"preservation runtime {tier} config identity differs"
            )
    return receipt


def _copy_input_packet(
    *, source_path: Path, expected_sha256: str, destination: Path, media_root: Path
) -> dict[str, Any]:
    source_bytes = _read_source_bytes(source_path, "preservation input")
    if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
        raise PreservationMaterializationError("preservation input content hash differs")
    try:
        source_value = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreservationMaterializationError(
            "preservation input is not strict JSON"
        ) from error
    raw = _object(source_value, "preservation input")
    _exact(raw, {"messages", "media"}, "preservation input")
    media_values = _array(raw["media"], "preservation input.media")
    normalized_media: list[dict[str, Any]] = []
    for index, value in enumerate(media_values):
        item = _object(value, f"preservation input.media[{index}]")
        _exact(
            item,
            {"media_id", "media_type", "path", "sha256"},
            f"preservation input.media[{index}]",
        )
        media_path = _source_file(
            source_path.parent, item["path"], f"preservation input.media[{index}].path"
        )
        expected_media_sha = _sha(
            item["sha256"], f"preservation input.media[{index}].sha256"
        )
        media_bytes = _read_source_bytes(media_path, "preservation media")
        if hashlib.sha256(media_bytes).hexdigest() != expected_media_sha:
            raise PreservationMaterializationError("preservation media content hash differs")
        media_type = _text(item["media_type"], "preservation media.media_type")
        _validate_media_bytes(media_bytes, media_path, media_type)
        suffix = media_path.suffix.lower()
        copied_name = f"{index:04d}{suffix}"
        copied_path = media_root / copied_name
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        copied_path.write_bytes(media_bytes)
        normalized_media.append(
            {
                "media_id": _text(item["media_id"], "preservation media.media_id"),
                "media_type": media_type,
                "path": str(copied_path.relative_to(destination.parent)),
                "sha256": expected_media_sha,
            }
        )
    normalized = {"messages": raw["messages"], "media": normalized_media}
    _canonical_bytes(normalized)
    _write_json(destination, normalized)
    return normalized


def _validate_media_bytes(content: bytes, path: Path, media_type: str) -> None:
    if len(content) > 512 * 1024 * 1024:
        raise PreservationMaterializationError("preservation media exceeds 512 MiB")
    if media_type == "image":
        try:
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise PreservationMaterializationError(
                "preservation image is not decodable"
            ) from error
        if width <= 0 or height <= 0 or width * height > 100_000_000:
            raise PreservationMaterializationError("preservation image dimensions are invalid")
        return
    if media_type == "video":
        if path.suffix.lower() != ".mp4" or len(content) < 12 or content[4:8] != b"ftyp":
            raise PreservationMaterializationError(
                "preservation video must be a bounded MP4 container"
            )
        return
    if media_type != "recorded_computer_use_trace":
        raise PreservationMaterializationError("preservation media type is unsupported")


class _ValidationOnlyBackend:
    identity = {"format": "truth_editing_preservation_validation_only_backend_v1"}

    def infer_edited_logits(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("materialization validation must not run model inference")

    def vision_tower_sha256(self, bundle: Any) -> str:
        raise AssertionError("materialization validation must not inspect a model")


def _materialize(plan_path: Path, staging: Path) -> dict[str, Any]:
    raw = _object(_load_json(plan_path, "preservation materialization plan"), "plan")
    _exact(
        raw,
        {
            "format",
            "spec_id",
            "base_model_sha256",
            "tokenizer_sha256",
            "processor_sha256",
            "vision_tower_sha256",
            "chat_template_sha256",
            "top_k",
            "temperature",
            "records",
            "tiers",
        },
        "preservation materialization plan",
    )
    if raw["format"] != PRESERVATION_MATERIALIZATION_PLAN_FORMAT:
        raise PreservationMaterializationError("unsupported preservation materialization plan")
    if raw["top_k"] != 64:
        raise PreservationMaterializationError("preservation top_k must be exactly 64")
    plan_root = plan_path.parent
    records_raw = _array(raw["records"], "plan.records")
    if not records_raw:
        raise PreservationMaterializationError("plan.records must not be empty")
    spec_records: list[dict[str, Any]] = []
    materialized: list[
        tuple[PreservationRecord, dict[str, Any], Path, dict[str, Any]]
    ] = []
    for index, value in enumerate(records_raw):
        item = _object(value, f"plan.records[{index}]")
        _exact(
            item,
            {
                "record_id",
                "stratum",
                "required_action_token_id",
                "input_path",
                "input_sha256",
                "base_logits_path",
                "base_logits_sha256",
                "base_logits_capture_receipt_path",
                "base_logits_capture_receipt_sha256",
            },
            f"plan.records[{index}]",
        )
        record_id = _text(item["record_id"], f"plan.records[{index}].record_id")
        input_source = _source_file(
            plan_root, item["input_path"], f"plan.records[{index}].input_path"
        )
        input_destination = staging / "inputs" / f"{index:04d}.json"
        input_payload = _copy_input_packet(
            source_path=input_source,
            expected_sha256=_sha(
                item["input_sha256"], f"plan.records[{index}].input_sha256"
            ),
            destination=input_destination,
            media_root=staging / "inputs" / "media" / f"{index:04d}",
        )
        record_payload = {
            "record_id": record_id,
            "stratum": item["stratum"],
            "prompt_sha256": _hash_json(input_payload),
            "direct_target": False,
            "required_action_token_id": item["required_action_token_id"],
        }
        record = PreservationRecord.from_dict(record_payload)
        logits_path = _source_file(
            plan_root,
            item["base_logits_path"],
            f"plan.records[{index}].base_logits_path",
        )
        logits_bytes = _read_source_bytes(logits_path, "base logits packet")
        base_logits_sha256 = _sha(
            item["base_logits_sha256"], f"plan.records[{index}].base_logits_sha256"
        )
        if hashlib.sha256(logits_bytes).hexdigest() != base_logits_sha256:
            raise PreservationMaterializationError("base logits content hash differs")
        capture_path = _source_file(
            plan_root,
            item["base_logits_capture_receipt_path"],
            f"plan.records[{index}].base_logits_capture_receipt_path",
        )
        capture_bytes = _read_source_bytes(capture_path, "base logits capture receipt")
        if hashlib.sha256(capture_bytes).hexdigest() != _sha(
            item["base_logits_capture_receipt_sha256"],
            f"plan.records[{index}].base_logits_capture_receipt_sha256",
        ):
            raise PreservationMaterializationError(
                "base logits capture receipt content hash differs"
            )
        try:
            capture = _parse_capture_receipt(json.loads(capture_bytes.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PreservationMaterializationError(
                "base logits capture receipt is not strict JSON"
            ) from error
        expected_capture = {
            "record_id": record_id,
            "base_logits_sha256": base_logits_sha256,
            "input_sha256": item["input_sha256"],
            "base_model_sha256": raw["base_model_sha256"],
            "tokenizer_sha256": raw["tokenizer_sha256"],
            "processor_sha256": raw["processor_sha256"],
            "chat_template_sha256": raw["chat_template_sha256"],
        }
        if any(capture[field] != expected for field, expected in expected_capture.items()):
            raise PreservationMaterializationError(
                "base logits capture receipt differs from the materialization plan"
            )
        _write_json(staging / "capture-receipts" / f"{index:04d}.json", capture)
        sealed_logits_path = staging / ".sealed-base-logits" / f"{index:04d}.safetensors"
        sealed_logits_path.parent.mkdir(parents=True, exist_ok=True)
        sealed_logits_path.write_bytes(logits_bytes)
        spec_records.append(record_payload)
        materialized.append((record, input_payload, sealed_logits_path, capture))

    spec_payload = {
        "format": "truth_editing_preservation_spec_v1",
        "spec_id": _text(raw["spec_id"], "plan.spec_id"),
        "base_model_sha256": _sha(raw["base_model_sha256"], "plan.base_model_sha256"),
        "tokenizer_sha256": _sha(raw["tokenizer_sha256"], "plan.tokenizer_sha256"),
        "processor_sha256": _sha(raw["processor_sha256"], "plan.processor_sha256"),
        "vision_tower_sha256": _sha(
            raw["vision_tower_sha256"], "plan.vision_tower_sha256"
        ),
        "top_k": 64,
        "temperature": raw["temperature"],
        "records": spec_records,
        "tiers": raw["tiers"],
    }
    try:
        spec = PreservationSpec.from_dict(spec_payload)
    except Exception as error:
        raise PreservationMaterializationError("preservation specification is invalid") from error
    _write_json(staging / "spec.json", spec.to_dict())

    baseline_locations: list[dict[str, Any]] = []
    for index, (record, input_payload, logits_path, capture) in enumerate(materialized):
        try:
            tensors = load_safetensors(
                _read_source_bytes(logits_path, "sealed base logits packet")
            )
        except Exception as error:
            raise PreservationMaterializationError("base logits packet is unreadable") from error
        if capture["format"] == _LEGACY_PRESERVATION_BASE_LOGITS_CAPTURE_RECEIPT_FORMAT:
            if set(tensors) != {"base_logits", "labels"}:
                raise PreservationMaterializationError(
                    "legacy base logits packet must contain exactly base_logits and labels"
                )
            logits = tensors["base_logits"]
            labels = tensors["labels"]
            if labels.dtype != torch.int64:
                raise PreservationMaterializationError("base logits labels must use int64")
            sorted_logits = torch.sort(logits, dim=-1, descending=True).values
            if torch.any(sorted_logits[..., 63] == sorted_logits[..., 64]):
                raise PreservationMaterializationError(
                    "base logits are tied at the deterministic top-k cutoff"
                )
            try:
                baseline = build_cached_baseline(spec, record, logits, labels)
            except Exception as error:
                raise PreservationMaterializationError(
                    "base logits packet is incompatible"
                ) from error
        else:
            expected_tensors = {
                "base_indices",
                "base_probabilities",
                "assistant_positions",
                "sequence_length",
            }
            if set(tensors) != expected_tensors:
                raise PreservationMaterializationError(
                    "compact capture tensor inventory differs"
                )
            indices = tensors["base_indices"]
            probabilities = tensors["base_probabilities"]
            positions = tensors["assistant_positions"]
            sequence_tensor = tensors["sequence_length"]
            if (
                indices.dtype != torch.int64
                or positions.dtype != torch.int64
                or sequence_tensor.dtype != torch.int64
                or indices.ndim != 3
                or indices.shape[0] != 1
                or indices.shape[-1] != 64
                or probabilities.shape != (*indices.shape[:-1], 65)
                or positions.shape != indices.shape[:-1]
                or sequence_tensor.shape != (1,)
            ):
                raise PreservationMaterializationError(
                    "compact capture tensor shapes or dtypes are invalid"
                )
            sequence_length = int(sequence_tensor.item())
            if (
                sequence_length != capture["sequence_length"]
                or indices.shape[1] != capture["assistant_position_count"]
                or capture["top_k"] != spec.top_k
                or capture["temperature"] != spec.temperature
            ):
                raise PreservationMaterializationError(
                    "compact capture metadata differs from materialization plan"
                )
            if (
                sequence_length < 2
                or torch.any(indices < 0)
                or torch.any(positions < 0)
                or torch.any(positions >= sequence_length - 1)
                or (positions.shape[1] > 1 and torch.any(positions[:, 1:] <= positions[:, :-1]))
                or any(torch.unique(row).numel() != 64 for row in indices.reshape(-1, 64))
                or not torch.isfinite(probabilities).all()
                or torch.any(probabilities < 0)
                or not torch.allclose(
                    probabilities.sum(dim=-1),
                    torch.ones_like(probabilities[..., 0]),
                    atol=1e-5,
                    rtol=1e-5,
                )
            ):
                raise PreservationMaterializationError("compact capture values are invalid")
            if record.required_action_token_id is not None and not torch.all(
                (indices == record.required_action_token_id).any(dim=-1)
            ):
                raise PreservationMaterializationError(
                    "compact computer-use capture omits required action token"
                )
            expanded_indices = torch.zeros(
                (1, sequence_length - 1, 64), dtype=torch.int64
            )
            expanded_probabilities = torch.zeros(
                (1, sequence_length - 1, 65), dtype=torch.float32
            )
            expanded_probabilities[..., -1] = 1.0
            assistant_mask = torch.zeros(
                (1, sequence_length - 1), dtype=torch.bool
            )
            expanded_indices[:, positions[0], :] = indices
            expanded_probabilities[:, positions[0], :] = probabilities.to(torch.float32)
            assistant_mask[:, positions[0]] = True
            unsigned_baseline = {
                "format": "truth_editing_preservation_cache_v1",
                "spec_sha256": spec.self_sha256,
                "base_model_sha256": spec.base_model_sha256,
                "tokenizer_sha256": spec.tokenizer_sha256,
                "processor_sha256": spec.processor_sha256,
                "record_id": record.record_id,
                "prompt_sha256": record.prompt_sha256,
                "stratum": record.stratum,
                "base_indices": expanded_indices.tolist(),
                "base_probabilities": expanded_probabilities.tolist(),
                "assistant_mask": assistant_mask.tolist(),
            }
            try:
                baseline = CachedPreservationBaseline.from_dict(
                    {
                        **unsigned_baseline,
                        "cache_sha256": _hash_json(unsigned_baseline),
                    }
                )
            except Exception as error:
                raise PreservationMaterializationError(
                    "compact capture is incompatible"
                ) from error
        baseline_path = staging / "baselines" / f"{index:04d}.json"
        _write_json(baseline_path, baseline.to_dict())
        baseline_locations.append(
            {
                "record_id": record.record_id,
                "path": str(baseline_path.relative_to(staging)),
                "cache_sha256": baseline.cache_sha256,
                "input_path": f"inputs/{index:04d}.json",
                "input_sha256": _hash_json(input_payload),
            }
        )

    shutil.rmtree(staging / ".sealed-base-logits")

    config_hashes: dict[str, str] = {}
    chat_template_sha256 = _sha(
        raw["chat_template_sha256"], "plan.chat_template_sha256"
    )
    for tier in _TIERS:
        config_payload = {
            "format": PRESERVATION_RUNTIME_CONFIG_FORMAT,
            "spec_path": "spec.json",
            "tier": tier,
            "chat_template_sha256": chat_template_sha256,
            "base_vision_tower_sha256": spec.vision_tower_sha256,
            "baselines": baseline_locations,
        }
        config_path = staging / _runtime_config_name(tier)
        _write_json(config_path, config_payload)
        config_hashes[tier] = PreservationRuntimeConfig.load(config_path).config_sha256
        TrialPreservationCollector.from_config(
            config_path, backend=_ValidationOnlyBackend()
        )

    artifact_hashes = {
        str(path.relative_to(staging)): _hash_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    unsigned_receipt = {
        "format": PRESERVATION_MATERIALIZATION_RECEIPT_FORMAT,
        "plan_sha256": _hash_json(raw),
        "spec_sha256": spec.self_sha256,
        "record_count": len(spec.records),
        "runtime_config_sha256": config_hashes,
        "artifact_sha256": artifact_hashes,
    }
    receipt = {**unsigned_receipt, "self_sha256": _hash_json(unsigned_receipt)}
    receipt = parse_preservation_materialization_receipt(receipt)
    _write_json(staging / "materialization-receipt.json", receipt)
    return receipt


def materialize_preservation_runtime_packet(
    plan_path: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    """Build and validate one complete packet, refusing every existing output.

    The returned receipt is location-independent. Publication is a single
    directory rename, so an invalid source cannot leave a partial destination.
    """

    plan = Path(plan_path)
    try:
        plan = plan.resolve(strict=True)
    except OSError as error:
        raise PreservationMaterializationError("materialization plan is missing") from error
    requested_output = Path(output_dir).expanduser()
    if not requested_output.is_absolute():
        requested_output = Path.cwd() / requested_output
    output = requested_output.parent.resolve() / requested_output.name
    if os.path.lexists(output):
        raise PreservationMaterializationError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        receipt = _materialize(plan, staging)
        _rename_no_replace(staging, output)
        return receipt
    except PreservationMaterializationError:
        shutil.rmtree(staging)
        raise
    except Exception as error:
        shutil.rmtree(staging)
        raise PreservationMaterializationError(
            f"preservation materialization failed: {type(error).__name__}: {error}"
        ) from error


__all__ = [
    "PRESERVATION_MATERIALIZATION_PLAN_FORMAT",
    "PRESERVATION_MATERIALIZATION_RECEIPT_FORMAT",
    "PreservationMaterializationError",
    "materialize_preservation_runtime_packet",
    "open_preservation_runtime_packet",
    "parse_preservation_materialization_receipt",
]
