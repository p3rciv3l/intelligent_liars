"""Resumable production extraction of frozen Qwen refusal directions.

Only construction prompts are forwarded.  Each completed batch persists per-layer
float64 sums, never example residuals, and publishes its receipt last.  A restart
therefore either reuses a fully verified batch contribution or recomputes it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .models import ModelLoadConfig, load_model_and_processor
from .truth_editing_refusal_directions import (
    BANK_FORMAT,
    LAYER_RECEIPT_FORMAT,
    RefusalDirectionBank,
    RefusalDirectionConfig,
    RefusalDirectionLayerReceipt,
    RefusalPromptManifest,
    RefusalPromptRow,
    build_refusal_extraction_plan,
    canonical_json_bytes,
    canonical_sha256,
    parse_refusal_direction_bank,
)


RUN_RECEIPT_FORMAT = "truth_editing_refusal_extraction_run_receipt_v1"
BATCH_RECEIPT_FORMAT = "truth_editing_refusal_extraction_batch_receipt_v1"
STORED_FORMAT = "truth_editing_stored_refusal_residuals_v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_REV = re.compile(r"^[0-9a-f]{40}$")
_PRODUCTION_TORCH = "2.5.1+cu124"


class RefusalExtractionError(RuntimeError):
    """Extraction cannot continue without violating a frozen invariant."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _signed(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(unsigned)
    result["self_sha256"] = canonical_sha256(result)
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float64), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)  # type: ignore[arg-type]
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class RuntimeIdentity:
    backend: str
    repository: str
    revision: str
    model_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    transformers_version: str
    torch_version: str
    dtype: str
    attention_implementation: str
    device: str
    decoder_layer_count: int
    hidden_width: int

    def __post_init__(self) -> None:
        if not self.backend or self.backend != self.backend.strip():
            raise RefusalExtractionError("runtime backend must be a trimmed string")
        if _REV.fullmatch(self.revision) is None:
            raise RefusalExtractionError("runtime revision is invalid")
        for label in ("model_sha256", "tokenizer_sha256", "chat_template_sha256"):
            if _SHA.fullmatch(str(getattr(self, label))) is None:
                raise RefusalExtractionError(f"runtime {label} is invalid")
        if self.decoder_layer_count < 1 or self.hidden_width < 1:
            raise RefusalExtractionError("runtime model shape is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def self_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def verify_for(self, config: RefusalDirectionConfig) -> None:
        expected = {
            "repository": config.model.repository,
            "revision": config.model.revision,
            "model_sha256": config.model.model_sha256,
            "tokenizer_sha256": config.model.tokenizer_sha256,
            "chat_template_sha256": config.model.chat_template_sha256,
            "transformers_version": config.extraction.transformers_version,
            "torch_version": _PRODUCTION_TORCH,
            "dtype": "torch.bfloat16",
            "attention_implementation": "flash_attention_2",
            "device": "cuda:0",
            "decoder_layer_count": config.model.decoder_layer_count,
            "hidden_width": config.model.hidden_width,
        }
        actual = self.to_dict()
        differences = [key for key, value in expected.items() if actual[key] != value]
        if differences:
            raise RefusalExtractionError(
                f"runtime identity differs from frozen extraction contract: {differences}"
            )


@dataclass(frozen=True)
class BackendBatchResult:
    residuals_by_layer: Mapping[int, np.ndarray]
    input_token_count: int
    elapsed_seconds: float


class ResidualBackend(Protocol):
    @property
    def identity(self) -> RuntimeIdentity: ...

    def extract(self, rows: Sequence[RefusalPromptRow]) -> BackendBatchResult: ...


@dataclass(frozen=True)
class RefusalExtractionRunReceipt:
    format: str
    config_sha256: str
    prompt_manifest_sha256: str
    plan_sha256: str
    runtime_identity: RuntimeIdentity
    runtime_identity_sha256: str
    batch_size: int
    batch_count: int
    batch_receipt_sha256s: tuple[str, ...]
    completed_prompt_count: int
    input_token_count: int
    elapsed_seconds: float
    prompt_throughput: float
    token_throughput: float
    direction_bank_sha256: str
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["batch_receipt_sha256s"] = list(self.batch_receipt_sha256s)
        return value


@dataclass(frozen=True)
class RefusalExtractionResult:
    bank: RefusalDirectionBank
    receipt: RefusalExtractionRunReceipt
    resumed_batch_count: int


def parse_run_receipt(value: Any) -> RefusalExtractionRunReceipt:
    if not isinstance(value, dict):
        raise RefusalExtractionError("run receipt must be an object")
    fields = {
        "format",
        "config_sha256",
        "prompt_manifest_sha256",
        "plan_sha256",
        "runtime_identity",
        "runtime_identity_sha256",
        "batch_size",
        "batch_count",
        "batch_receipt_sha256s",
        "completed_prompt_count",
        "input_token_count",
        "elapsed_seconds",
        "prompt_throughput",
        "token_throughput",
        "direction_bank_sha256",
        "self_sha256",
    }
    if set(value) != fields or value.get("format") != RUN_RECEIPT_FORMAT:
        raise RefusalExtractionError("run receipt fields or format differ")
    unsigned = dict(value)
    claimed = unsigned.pop("self_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(unsigned) != claimed:
        raise RefusalExtractionError("run receipt self hash mismatch")
    try:
        identity = RuntimeIdentity(**value["runtime_identity"])
        hashes = tuple(value["batch_receipt_sha256s"])
        numeric = tuple(
            float(value[key])
            for key in ("elapsed_seconds", "prompt_throughput", "token_throughput")
        )
    except (TypeError, ValueError, KeyError) as error:
        raise RefusalExtractionError("run receipt values are malformed") from error
    for digest in (
        value["config_sha256"],
        value["prompt_manifest_sha256"],
        value["plan_sha256"],
        value["runtime_identity_sha256"],
        value["direction_bank_sha256"],
        *hashes,
    ):
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
            raise RefusalExtractionError("run receipt contains an invalid SHA-256")
    if value["runtime_identity_sha256"] != identity.self_sha256:
        raise RefusalExtractionError("run receipt runtime identity hash mismatch")
    integers = tuple(
        value[key]
        for key in (
            "batch_size",
            "batch_count",
            "completed_prompt_count",
            "input_token_count",
        )
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in integers
    ):
        raise RefusalExtractionError("run receipt counts are invalid")
    if len(hashes) != value["batch_count"] or any(
        not math.isfinite(item) or item < 0 for item in numeric
    ):
        raise RefusalExtractionError("run receipt telemetry is invalid")
    return RefusalExtractionRunReceipt(
        RUN_RECEIPT_FORMAT,
        value["config_sha256"],
        value["prompt_manifest_sha256"],
        value["plan_sha256"],
        identity,
        identity.self_sha256,
        value["batch_size"],
        value["batch_count"],
        hashes,
        value["completed_prompt_count"],
        value["input_token_count"],
        numeric[0],
        numeric[1],
        numeric[2],
        value["direction_bank_sha256"],
        claimed,
    )


class RefusalExtractionRunner:
    """Extract one immutable refusal bank, resuming at verified batch boundaries."""

    def __init__(
        self,
        config: RefusalDirectionConfig,
        prompts: RefusalPromptManifest,
        backend: ResidualBackend,
        output_dir: str | Path,
        *,
        batch_size: int = 8,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise RefusalExtractionError("batch_size must be a positive integer")
        self.config = config
        self.prompts = prompts
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

    def run(self) -> RefusalExtractionResult:
        identity = self.backend.identity
        identity.verify_for(self.config)
        plan = build_refusal_extraction_plan(self.config, self.prompts)
        if not plan.ready:
            raise RefusalExtractionError(f"extraction plan is blocked: {plan.blockers}")
        verify_inputs = getattr(self.backend, "verify_inputs", None)
        if verify_inputs is not None:
            verify_inputs(self.config, self.prompts, plan.self_sha256)
        construction = tuple(
            row for row in self.prompts.rows if row.partition == "construction"
        )
        batches = tuple(
            batch
            for role in ("harmless", "harmful")
            for selected in (tuple(row for row in construction if row.role == role),)
            for index in range(0, len(selected), self.batch_size)
            for batch in (selected[index : index + self.batch_size],)
        )
        contributions = []
        resumed_batch_count = 0
        for index, rows in enumerate(batches):
            existing = self._load_batch(index, rows, identity, plan.self_sha256)
            if existing is None:
                result = self.backend.extract(rows)
                existing = self._save_batch(
                    index, rows, result, identity, plan.self_sha256
                )
            else:
                resumed_batch_count += 1
            contributions.append(existing)
        bank = self._finalize(contributions)
        batch_hashes = tuple(item["receipt"]["self_sha256"] for item in contributions)
        prompt_count = sum(
            int(item["receipt"]["prompt_count"]) for item in contributions
        )
        token_count = sum(
            int(item["receipt"]["input_token_count"]) for item in contributions
        )
        elapsed = sum(
            float(item["receipt"]["elapsed_seconds"]) for item in contributions
        )
        unsigned = {
            "format": RUN_RECEIPT_FORMAT,
            "config_sha256": self.config.self_sha256,
            "prompt_manifest_sha256": self.prompts.self_sha256,
            "plan_sha256": plan.self_sha256,
            "runtime_identity": identity.to_dict(),
            "runtime_identity_sha256": identity.self_sha256,
            "batch_size": self.batch_size,
            "batch_count": len(contributions),
            "batch_receipt_sha256s": list(batch_hashes),
            "completed_prompt_count": prompt_count,
            "input_token_count": token_count,
            "elapsed_seconds": elapsed,
            "prompt_throughput": prompt_count / elapsed,
            "token_throughput": token_count / elapsed,
            "direction_bank_sha256": bank.self_sha256,
        }
        receipt_raw = _signed(unsigned)
        receipt = parse_run_receipt(receipt_raw)
        _atomic_json(self.output_dir / "run_receipt.json", receipt_raw)
        return RefusalExtractionResult(bank, receipt, resumed_batch_count)

    def _expected_batch(
        self,
        index: int,
        rows: Sequence[RefusalPromptRow],
        identity: RuntimeIdentity,
        plan_sha256: str,
    ) -> dict[str, Any]:
        roles = {row.role for row in rows}
        if len(roles) != 1:
            raise RefusalExtractionError(
                "a batch may not mix harmless and harmful prompts"
            )
        return {
            "format": BATCH_RECEIPT_FORMAT,
            "batch_index": index,
            "config_sha256": self.config.self_sha256,
            "prompt_manifest_sha256": self.prompts.self_sha256,
            "plan_sha256": plan_sha256,
            "runtime_identity_sha256": identity.self_sha256,
            "role": next(iter(roles)),
            "prompt_ids": [row.prompt_id for row in rows],
            "formatted_prompt_sha256s": [row.formatted_prompt_sha256 for row in rows],
            "prompt_count": len(rows),
        }

    def _load_batch(
        self,
        index: int,
        rows: Sequence[RefusalPromptRow],
        identity: RuntimeIdentity,
        plan_sha256: str,
    ) -> dict[str, Any] | None:
        root = self.output_dir / "batches" / f"{index:04d}"
        receipt_path = root / "receipt.json"
        if not receipt_path.exists():
            return None
        try:
            receipt = json.loads(receipt_path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise RefusalExtractionError(
                "completed batch receipt is unreadable"
            ) from error
        expected = self._expected_batch(index, rows, identity, plan_sha256)
        fixed = set(expected) | {
            "input_token_count",
            "elapsed_seconds",
            "prompt_throughput",
            "token_throughput",
            "contribution_path",
            "contribution_file_sha256",
            "layer_sum_sha256s",
            "self_sha256",
        }
        if not isinstance(receipt, dict) or set(receipt) != fixed:
            raise RefusalExtractionError("completed batch receipt fields differ")
        if any(receipt[key] != value for key, value in expected.items()):
            raise RefusalExtractionError("completed batch receipt identity differs")
        unsigned = dict(receipt)
        claimed = unsigned.pop("self_sha256")
        if canonical_sha256(unsigned) != claimed:
            raise RefusalExtractionError("completed batch receipt self hash mismatch")
        contribution = root / "contribution.npz"
        if (
            receipt["contribution_path"] != f"batches/{index:04d}/contribution.npz"
            or not contribution.is_file()
            or _sha_file(contribution) != receipt["contribution_file_sha256"]
        ):
            raise RefusalExtractionError(
                "completed batch contribution file identity mismatch"
            )
        sums = self._read_sums(contribution, receipt["layer_sum_sha256s"])
        return {"receipt": receipt, "sums": sums}

    def _read_sums(
        self, path: Path, expected_hashes: Mapping[str, str]
    ) -> dict[int, np.ndarray]:
        names = {f"layer_{layer:02d}" for layer in self.config.extraction.layers}
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != names or set(expected_hashes) != names:
                    raise RefusalExtractionError(
                        "batch contribution does not cover every configured layer"
                    )
                result = {
                    layer: np.asarray(archive[f"layer_{layer:02d}"], dtype=np.float64)
                    for layer in self.config.extraction.layers
                }
        except (OSError, ValueError) as error:
            raise RefusalExtractionError("batch contribution is unreadable") from error
        for layer, value in result.items():
            name = f"layer_{layer:02d}"
            if (
                value.shape != (self.config.model.hidden_width,)
                or not np.all(np.isfinite(value))
                or _array_sha(value) != expected_hashes[name]
            ):
                raise RefusalExtractionError(
                    "batch contribution layer identity differs"
                )
        return result

    def _save_batch(
        self,
        index: int,
        rows: Sequence[RefusalPromptRow],
        result: BackendBatchResult,
        identity: RuntimeIdentity,
        plan_sha256: str,
    ) -> dict[str, Any]:
        expected_layers = set(self.config.extraction.layers)
        if set(result.residuals_by_layer) != expected_layers:
            raise RefusalExtractionError(
                "backend did not return every configured layer"
            )
        if (
            isinstance(result.input_token_count, bool)
            or not isinstance(result.input_token_count, int)
            or result.input_token_count < len(rows)
        ):
            raise RefusalExtractionError("backend input token count is invalid")
        if not math.isfinite(result.elapsed_seconds) or result.elapsed_seconds <= 0:
            raise RefusalExtractionError("backend elapsed time is invalid")
        sums: dict[int, np.ndarray] = {}
        for layer in self.config.extraction.layers:
            residuals = np.asarray(result.residuals_by_layer[layer], dtype=np.float64)
            if residuals.shape != (
                len(rows),
                self.config.model.hidden_width,
            ) or not np.all(np.isfinite(residuals)):
                raise RefusalExtractionError(
                    "backend residual shape or finiteness differs"
                )
            sums[layer] = residuals.sum(axis=0, dtype=np.float64)
        root = self.output_dir / "batches" / f"{index:04d}"
        contribution = root / "contribution.npz"
        named = {f"layer_{layer:02d}": value for layer, value in sums.items()}
        _atomic_npz(contribution, named)
        base = self._expected_batch(index, rows, identity, plan_sha256)
        elapsed = float(result.elapsed_seconds)
        base.update(
            {
                "input_token_count": result.input_token_count,
                "elapsed_seconds": elapsed,
                "prompt_throughput": len(rows) / elapsed,
                "token_throughput": result.input_token_count / elapsed,
                "contribution_path": f"batches/{index:04d}/contribution.npz",
                "contribution_file_sha256": _sha_file(contribution),
                "layer_sum_sha256s": {
                    name: _array_sha(value) for name, value in named.items()
                },
            }
        )
        receipt = _signed(base)
        _atomic_json(root / "receipt.json", receipt)
        return {"receipt": receipt, "sums": sums}

    def _finalize(
        self, contributions: Sequence[Mapping[str, Any]]
    ) -> RefusalDirectionBank:
        by_role: dict[str, list[Mapping[str, Any]]] = {"harmless": [], "harmful": []}
        for item in contributions:
            by_role[str(item["receipt"]["role"])].append(item)
        counts = {
            role: sum(int(item["receipt"]["prompt_count"]) for item in items)
            for role, items in by_role.items()
        }
        if not all(counts.values()):
            raise RefusalExtractionError("both construction roles require examples")
        layer_receipts = []
        for layer in self.config.extraction.layers:
            role_means = {}
            for role, items in by_role.items():
                role_sum = np.zeros(self.config.model.hidden_width, dtype=np.float64)
                for item in items:
                    role_sum += item["sums"][layer]
                role_means[role] = role_sum / counts[role]
            direction = role_means["harmful"] - role_means["harmless"]
            norm = float(np.linalg.norm(direction))
            if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
                raise RefusalExtractionError(
                    f"layer {layer} refusal difference has zero norm"
                )
            direction /= norm
            vector_path = self.output_dir / "vectors" / f"layer-{layer:02d}.npy"
            _atomic_npy(vector_path, direction)
            unsigned = {
                "format": LAYER_RECEIPT_FORMAT,
                "receipt_id": f"raw-refusal-layer-{layer:02d}",
                "source_layer": layer,
                "width": self.config.model.hidden_width,
                "construction_harmless_count": counts["harmless"],
                "construction_harmful_count": counts["harmful"],
                "harmless_mean_sha256": _array_sha(role_means["harmless"]),
                "harmful_mean_sha256": _array_sha(role_means["harmful"]),
                "vector_path": f"vectors/layer-{layer:02d}.npy",
                "vector_file_sha256": _sha_file(vector_path),
                "vector_sha256": _array_sha(direction),
                "finite": True,
                "unit_norm": bool(
                    math.isclose(
                        float(np.linalg.norm(direction)), 1.0, rel_tol=0, abs_tol=1e-12
                    )
                ),
            }
            raw = _signed(unsigned)
            layer_receipts.append(RefusalDirectionLayerReceipt(**raw))
            _atomic_json(self.output_dir / "receipts" / f"layer-{layer:02d}.json", raw)
        bank_unsigned = {
            "format": BANK_FORMAT,
            "bank_id": f"{self.config.config_id}-raw-refusal",
            "config_sha256": self.config.self_sha256,
            "prompt_manifest_sha256": self.prompts.self_sha256,
            "model_sha256": self.config.model.model_sha256,
            "chat_template_sha256": self.config.model.chat_template_sha256,
            "per_layer_receipts": [asdict(item) for item in layer_receipts],
            "global_source_receipt_ids": [item.receipt_id for item in layer_receipts],
        }
        bank_raw = _signed(bank_unsigned)
        bank = parse_refusal_direction_bank(bank_raw, self.config, self.prompts)
        _atomic_json(self.output_dir / "direction_bank.json", bank_raw)
        return bank


class StoredResidualBackend:
    """Replay exact stored residuals for deterministic tests and offline recovery."""

    def __init__(
        self,
        identity: RuntimeIdentity,
        records: Mapping[str, Mapping[str, Any]],
        *,
        config_sha256: str,
        prompt_manifest_sha256: str,
        plan_sha256: str,
    ) -> None:
        self.identity = identity
        self._records = dict(records)
        self._config_sha256 = config_sha256
        self._prompt_manifest_sha256 = prompt_manifest_sha256
        self._plan_sha256 = plan_sha256

    @classmethod
    def from_path(cls, path: str | Path) -> "StoredResidualBackend":
        try:
            raw = json.loads(Path(path).read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise RefusalExtractionError(
                "stored residual file is unreadable"
            ) from error
        if (
            not isinstance(raw, dict)
            or set(raw)
            != {
                "format",
                "config_sha256",
                "prompt_manifest_sha256",
                "plan_sha256",
                "runtime_identity",
                "records",
                "self_sha256",
            }
            or raw["format"] != STORED_FORMAT
        ):
            raise RefusalExtractionError("stored residual file fields or format differ")
        unsigned = dict(raw)
        claimed = unsigned.pop("self_sha256")
        if not isinstance(claimed, str) or canonical_sha256(unsigned) != claimed:
            raise RefusalExtractionError("stored residual file self hash mismatch")
        for key in ("config_sha256", "prompt_manifest_sha256", "plan_sha256"):
            if not isinstance(raw[key], str) or _SHA.fullmatch(raw[key]) is None:
                raise RefusalExtractionError(f"stored residual {key} is invalid")
        try:
            identity = RuntimeIdentity(**raw["runtime_identity"])
        except (TypeError, ValueError) as error:
            raise RefusalExtractionError(
                "stored runtime identity is malformed"
            ) from error
        if not isinstance(raw["records"], list):
            raise RefusalExtractionError("stored residual records must be a list")
        records: dict[str, Mapping[str, Any]] = {}
        for item in raw["records"]:
            if not isinstance(item, dict) or set(item) != {
                "prompt_id",
                "formatted_prompt_sha256",
                "input_token_count",
                "residuals_by_layer",
            }:
                raise RefusalExtractionError("stored residual record fields differ")
            prompt_id = item["prompt_id"]
            if not isinstance(prompt_id, str) or not prompt_id or prompt_id in records:
                raise RefusalExtractionError(
                    "stored residual prompt ID is invalid or duplicated"
                )
            if (
                not isinstance(item["formatted_prompt_sha256"], str)
                or _SHA.fullmatch(item["formatted_prompt_sha256"]) is None
            ):
                raise RefusalExtractionError(
                    "stored formatted prompt identity is invalid"
                )
            records[prompt_id] = item
        return cls(
            identity,
            records,
            config_sha256=raw["config_sha256"],
            prompt_manifest_sha256=raw["prompt_manifest_sha256"],
            plan_sha256=raw["plan_sha256"],
        )

    def verify_inputs(
        self,
        config: RefusalDirectionConfig,
        prompts: RefusalPromptManifest,
        plan_sha256: str,
    ) -> None:
        if (
            self._config_sha256 != config.self_sha256
            or self._prompt_manifest_sha256 != prompts.self_sha256
            or self._plan_sha256 != plan_sha256
        ):
            raise RefusalExtractionError("stored residual input identity differs")

    def extract(self, rows: Sequence[RefusalPromptRow]) -> BackendBatchResult:
        selected = []
        for row in rows:
            try:
                record = self._records[row.prompt_id]
            except KeyError as error:
                raise RefusalExtractionError(
                    f"stored residual missing prompt {row.prompt_id}"
                ) from error
            if record["formatted_prompt_sha256"] != row.formatted_prompt_sha256:
                raise RefusalExtractionError("stored formatted prompt identity differs")
            selected.append(record)
        layer_keys = set(selected[0]["residuals_by_layer"])
        if any(set(item["residuals_by_layer"]) != layer_keys for item in selected):
            raise RefusalExtractionError(
                "stored residual layer coverage differs between rows"
            )
        residuals = {
            int(layer): np.asarray(
                [item["residuals_by_layer"][layer] for item in selected],
                dtype=np.float64,
            )
            for layer in layer_keys
        }
        token_count = sum(int(item["input_token_count"]) for item in selected)
        return BackendBatchResult(residuals, token_count, max(len(rows) / 1000.0, 1e-6))


class TransformersQwenResidualBackend:
    """Exact CUDA/BF16/FA2 backend for the pinned Qwen snapshot."""

    def __init__(
        self,
        config: RefusalDirectionConfig,
        *,
        cache_dir: str | Path,
        snapshot_manifest_path: str | Path,
        bundle: Any | None = None,
        bundle_loader: Callable[[ModelLoadConfig], Any] = load_model_and_processor,
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.snapshot_manifest_path = Path(snapshot_manifest_path)
        self._bundle: Any | None = bundle
        self._bundle_loader = bundle_loader
        self._identity: RuntimeIdentity | None = None

    @staticmethod
    def validate_environment(
        *, cuda_available: bool, transformers_version: str, torch_version: str
    ) -> None:
        if not cuda_available:
            raise RefusalExtractionError("production refusal extraction requires CUDA")
        if transformers_version != "4.57.1":
            raise RefusalExtractionError(
                "production Transformers version differs from 4.57.1"
            )
        if torch_version != _PRODUCTION_TORCH:
            raise RefusalExtractionError(
                f"production torch version differs from {_PRODUCTION_TORCH}"
            )

    @property
    def identity(self) -> RuntimeIdentity:
        self._load()
        assert self._identity is not None
        return self._identity

    def _load(self) -> None:
        if self._identity is not None:
            return
        import torch
        import transformers

        self.validate_environment(
            cuda_available=torch.cuda.is_available(),
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
        )
        load_config = ModelLoadConfig(
            model_name=self.config.model.repository,
            revision=self.config.model.revision,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
            local_files_only=True,
            use_cache=True,
            cache_dir=str(self.cache_dir),
            snapshot_manifest_path=str(self.snapshot_manifest_path),
            expected_model_sha256=self.config.model.model_sha256,
        )
        bundle = self._bundle or self._bundle_loader(load_config)
        verified = bundle.verified_snapshot
        if (
            not isinstance(verified, Mapping)
            or verified.get("model_sha256") != self.config.model.model_sha256
        ):
            raise RefusalExtractionError(
                "loaded snapshot lacks exact verified model identity"
            )
        manifest = json.loads(self.snapshot_manifest_path.read_bytes())
        files = manifest.get("files") if isinstance(manifest, dict) else None
        by_path = {
            item.get("path"): item for item in files or [] if isinstance(item, dict)
        }
        tokenizer_entry = by_path.get("tokenizer.json")
        if (
            not isinstance(tokenizer_entry, dict)
            or tokenizer_entry.get("sha256") != self.config.model.tokenizer_sha256
        ):
            raise RefusalExtractionError("verified tokenizer.json identity differs")
        template = getattr(bundle.processor, "chat_template", None) or getattr(
            bundle.tokenizer, "chat_template", None
        )
        if (
            not isinstance(template, str)
            or hashlib.sha256(template.encode()).hexdigest()
            != self.config.model.chat_template_sha256
        ):
            raise RefusalExtractionError("loaded chat template identity differs")
        model = bundle.model
        attention_values = {
            getattr(candidate, "_attn_implementation", None)
            for candidate in (
                getattr(model, "config", None),
                getattr(getattr(model, "config", None), "text_config", None),
            )
            if candidate is not None
        }
        if "flash_attention_2" not in attention_values:
            raise RefusalExtractionError(
                "loaded model is not configured for FlashAttention 2"
            )
        parameters = tuple(model.parameters()) if model is not None else ()
        if not parameters or any(
            parameter.device.type != "cuda"
            or parameter.device.index not in {None, 0}
            or parameter.dtype is not torch.bfloat16
            for parameter in parameters
        ):
            raise RefusalExtractionError("loaded model is not CUDA BF16")
        self._bundle = bundle
        self._identity = RuntimeIdentity(
            backend="transformers_qwen_cuda_v1",
            repository=self.config.model.repository,
            revision=self.config.model.revision,
            model_sha256=self.config.model.model_sha256,
            tokenizer_sha256=self.config.model.tokenizer_sha256,
            chat_template_sha256=self.config.model.chat_template_sha256,
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            dtype="torch.bfloat16",
            attention_implementation="flash_attention_2",
            device="cuda:0",
            decoder_layer_count=self.config.model.decoder_layer_count,
            hidden_width=self.config.model.hidden_width,
        )

    def extract(self, rows: Sequence[RefusalPromptRow]) -> BackendBatchResult:
        self._load()
        import torch

        bundle = self._bundle
        assert bundle is not None and bundle.model is not None
        texts = [
            bundle.processor.apply_chat_template(
                [
                    {"role": "system", "content": self.config.extraction.system_prompt},
                    {"role": "user", "content": row.prompt_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in rows
        ]
        encoded = bundle.processor(text=texts, padding=True, return_tensors="pt")
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise RefusalExtractionError("processor returned no input_ids")
        device = next(bundle.model.parameters()).device
        moved = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        if "attention_mask" in moved:
            token_count = int(moved["attention_mask"].sum().item())
        else:
            token_count = int(moved["input_ids"].numel())
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = bundle.model.generate(
                **moved,
                max_new_tokens=1,
                do_sample=False,
                use_cache=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(hidden_states, tuple) or len(hidden_states) != 1:
            raise RefusalExtractionError(
                "generation did not return one hidden-state step"
            )
        step = hidden_states[0]
        if (
            not isinstance(step, tuple)
            or len(step) != self.config.model.decoder_layer_count + 1
        ):
            raise RefusalExtractionError(
                "generation hidden-state layer coverage differs"
            )
        # Transformers exposes the prefill pass as generation step zero.  Its
        # final unpadded position is the decoder-layer output that predicts the
        # first generated token: the Heretic v1 extraction location frozen by
        # ``decoder_layer_output_first_generated_token_v1``.
        residuals = {
            layer: step[layer + 1][:, -1, :]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            for layer in self.config.extraction.layers
        }
        return BackendBatchResult(residuals, token_count, elapsed)


__all__ = [
    "BackendBatchResult",
    "RefusalExtractionError",
    "RefusalExtractionResult",
    "RefusalExtractionRunReceipt",
    "RefusalExtractionRunner",
    "ResidualBackend",
    "RuntimeIdentity",
    "StoredResidualBackend",
    "TransformersQwenResidualBackend",
    "parse_run_receipt",
]
