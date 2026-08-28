"""Strict, offline provenance sidecars for historical activation artifacts.

This module records *what is known* about an activation HDF5 without fetching
the DVC object or reading the large HDF5.  A sidecar can therefore preserve
historical validation receipts while making uncertainty explicit.  Parsing is
fail closed: unknown fields, malformed identities, inconsistent sizes, and a
``proven`` claim without every required binding are rejected.

The sidecar is intentionally independent of h5py, DVC, torch, and model
loading.  The optional inventory builder only canonicalizes already-recorded
metadata; it never hydrates or hashes an artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast


ACTIVATION_PROVENANCE_FORMAT = "activation_provenance_sidecar_v1"
INVENTORY_FORMAT = "activation_provenance_inventory_v1"
SPLIT_RECEIPT_FORMAT = "truth_editing_split_receipt_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceContractError(ValueError):
    """A provenance sidecar cannot be verified exactly."""


def canonical_sha256(value: Any) -> str:
    """Hash one finite, deterministic JSON value."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProvenanceContractError("value is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceContractError(f"{name} must be an object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ProvenanceContractError(
            f"{name} fields differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProvenanceContractError(f"{name} must be a nonempty trimmed string")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProvenanceContractError(f"{name} must be a lowercase SHA-256")
    return value


def _revision(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ProvenanceContractError(
            f"{name} must be a lowercase 40-character Git revision"
        )
    return value


def _integer(value: Any, name: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProvenanceContractError(f"{name} must be a nonnegative integer")
    return value


def _boolean(value: Any, name: str, *, optional: bool = False) -> bool | None:
    if value is None and optional:
        return None
    if not isinstance(value, bool):
        raise ProvenanceContractError(f"{name} must be boolean")
    return value


def _enum(value: Any, choices: set[str], name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    result = _text(value, name)
    if result not in choices:
        raise ProvenanceContractError(f"{name} must be one of {sorted(choices)}")
    return result


def _string_tuple(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProvenanceContractError(f"{name} must be an array of strings")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise ProvenanceContractError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ProvenanceContractError(f"{name} entries must be unique")
    return result


def _string_int_map(value: Any, name: str, keys: tuple[str, ...]) -> dict[str, int | None]:
    raw = _mapping(value, name)
    if set(raw) != set(keys):
        raise ProvenanceContractError(f"{name} keys must exactly match tasks")
    result: dict[str, int | None] = {}
    for key in keys:
        result[key] = _integer(raw[key], f"{name}.{key}", optional=True)
    return result


def _self_hash(raw: Mapping[str, Any], name: str) -> str:
    claimed = _sha256(raw.get("self_sha256"), f"{name}.self_sha256")
    unsigned = dict(raw)
    unsigned.pop("self_sha256", None)
    if canonical_sha256(unsigned) != claimed:
        raise ProvenanceContractError(f"{name} self hash mismatch")
    return claimed


@dataclass(frozen=True)
class ActivationArtifact:
    path: str
    byte_size: int | None
    dvc_pointer_path: str | None
    dvc_hash_algorithm: Literal["md5", "sha256"] | None
    dvc_hash: str | None
    dvc_size: int | None
    dvc_pointer_sha256: str | None
    direct_sha256: str | None
    direct_hash_evidence: Literal[
        "local_read", "historical_validation_receipt", "not_checked"
    ]


@dataclass(frozen=True)
class HDF5Inventory:
    validator_format: str
    tasks: tuple[str, ...]
    task_rows: dict[str, int | None]
    example_counts: dict[str, int | None]
    layers: tuple[str, ...]
    hidden_dim: int | None
    storage_dtype: Literal["float16", "float32", "bfloat16"] | None
    finite_check: Literal["none", "sample", "full"]
    validator_revision: str | None


@dataclass(frozen=True)
class ModelIdentity:
    repository: str
    revision: str | None
    content_sha256: str | None


@dataclass(frozen=True)
class ProcessorIdentity:
    repository: str
    revision: str | None
    content_sha256: str | None
    tokenizer_sha256: str | None
    chat_template_sha256: str | None


@dataclass(frozen=True)
class RuntimeIdentity:
    python_version: str | None
    torch_version: str | None
    transformers_version: str | None
    backend: str | None
    dtype: str | None
    device: str | None
    attention_implementation: str | None
    quantization: str | None
    batch_size: int | None
    use_cache: bool | None


@dataclass(frozen=True)
class SourceDatasetIdentity:
    dataset_id: str
    revision: str | None
    manifest_sha256: str | None
    source_row_ids_sha256: str | None


@dataclass(frozen=True)
class SplitReceipt:
    format: Literal["truth_editing_split_receipt_v1"]
    status: Literal["verified", "unknown"]
    split_name: str | None
    split_policy: str | None
    assignment_seed: int | None
    dataset_manifest_sha256: str | None
    ordered_row_ids_sha256: str | None
    group_ids_sha256: str | None
    disjoint_from: tuple[str, ...]
    receipt_sha256: str | None


@dataclass(frozen=True)
class ActivationProvenance:
    format: Literal["activation_provenance_sidecar_v1"]
    sidecar_id: str
    artifact: ActivationArtifact
    hdf5_inventory: HDF5Inventory
    model: ModelIdentity
    processor: ProcessorIdentity
    runtime: RuntimeIdentity
    source_dataset: SourceDatasetIdentity
    split_receipt: SplitReceipt
    evidence_status: Literal["proven", "verified_metadata", "unknown"]
    self_sha256: str

    @property
    def identity_sha256(self) -> str:
        return self.self_sha256

    def to_payload(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))

    def to_dict(self) -> dict[str, Any]:
        return self.to_payload()


@dataclass(frozen=True)
class ActivationProvenanceInventory:
    format: Literal["activation_provenance_inventory_v1"]
    inventory_id: str
    entries: tuple[ActivationProvenance, ...]
    self_sha256: str

    def to_payload(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))

    def to_dict(self) -> dict[str, Any]:
        return self.to_payload()


def _parse_artifact(value: Any) -> ActivationArtifact:
    raw = _mapping(value, "artifact")
    _exact(
        raw,
        {
            "path", "byte_size", "dvc_pointer_path", "dvc_hash_algorithm",
            "dvc_hash", "dvc_size", "dvc_pointer_sha256", "direct_sha256",
            "direct_hash_evidence",
        },
        "artifact",
    )
    path = _text(raw["path"], "artifact.path")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise ProvenanceContractError("artifact.path must be a normalized relative path")
    size = _integer(raw["byte_size"], "artifact.byte_size", optional=True)
    dvc_pointer_path = _optional_text(raw["dvc_pointer_path"], "artifact.dvc_pointer_path")
    algorithm = _enum(
        raw["dvc_hash_algorithm"], {"md5", "sha256"},
        "artifact.dvc_hash_algorithm", optional=True,
    )
    dvc_hash = raw["dvc_hash"]
    if dvc_hash is not None:
        if algorithm == "md5":
            if not isinstance(dvc_hash, str) or not _MD5.fullmatch(dvc_hash):
                raise ProvenanceContractError("artifact.dvc_hash must be a lowercase MD5")
        elif algorithm == "sha256":
            _sha256(dvc_hash, "artifact.dvc_hash")
        else:
            raise ProvenanceContractError(
                "artifact.dvc_hash_algorithm is required when artifact.dvc_hash is known"
            )
    elif algorithm is not None:
        raise ProvenanceContractError(
            "artifact.dvc_hash is required when artifact.dvc_hash_algorithm is known"
        )
    dvc_size = _integer(raw["dvc_size"], "artifact.dvc_size", optional=True)
    if size is not None and dvc_size is not None and size != dvc_size:
        raise ProvenanceContractError("artifact byte_size and dvc_size differ")
    pointer_sha = _sha256(raw["dvc_pointer_sha256"], "artifact.dvc_pointer_sha256", optional=True)
    direct = _sha256(raw["direct_sha256"], "artifact.direct_sha256", optional=True)
    evidence = _enum(
        raw["direct_hash_evidence"],
        {"local_read", "historical_validation_receipt", "not_checked"},
        "artifact.direct_hash_evidence",
    )
    if direct is None and evidence != "not_checked":
        raise ProvenanceContractError(
            "artifact.direct_hash_evidence must be not_checked when direct_sha256 is unknown"
        )
    if direct is not None and evidence == "not_checked":
        raise ProvenanceContractError(
            "artifact.direct_hash_evidence must identify evidence for a known direct_sha256"
        )
    return ActivationArtifact(
        path=path,
        byte_size=size,
        dvc_pointer_path=dvc_pointer_path,
        dvc_hash_algorithm=algorithm,  # type: ignore[arg-type]
        dvc_hash=dvc_hash,
        dvc_size=dvc_size,
        dvc_pointer_sha256=pointer_sha,
        direct_sha256=direct,
        direct_hash_evidence=evidence,  # type: ignore[arg-type]
    )


def _parse_hdf5_inventory(value: Any) -> HDF5Inventory:
    raw = _mapping(value, "hdf5_inventory")
    _exact(
        raw,
        {
            "validator_format", "tasks", "task_rows", "example_counts", "layers",
            "hidden_dim", "storage_dtype", "finite_check", "validator_revision",
        },
        "hdf5_inventory",
    )
    tasks = _string_tuple(raw["tasks"], "hdf5_inventory.tasks")
    layers = _string_tuple(raw["layers"], "hdf5_inventory.layers")
    return HDF5Inventory(
        validator_format=_text(raw["validator_format"], "hdf5_inventory.validator_format"),
        tasks=tasks,
        task_rows=_string_int_map(raw["task_rows"], "hdf5_inventory.task_rows", tasks),
        example_counts=_string_int_map(
            raw["example_counts"], "hdf5_inventory.example_counts", tasks
        ),
        layers=layers,
        hidden_dim=_integer(raw["hidden_dim"], "hdf5_inventory.hidden_dim", optional=True),
        storage_dtype=_enum(
            raw["storage_dtype"], {"float16", "float32", "bfloat16"},
            "hdf5_inventory.storage_dtype", optional=True,
        ),  # type: ignore[arg-type]
        finite_check=_enum(
            raw["finite_check"], {"none", "sample", "full"},
            "hdf5_inventory.finite_check",
        ),  # type: ignore[arg-type]
        validator_revision=_revision(
            raw["validator_revision"], "hdf5_inventory.validator_revision", optional=True
        ),
    )


def _parse_model(value: Any, name: str) -> ModelIdentity:
    raw = _mapping(value, name)
    _exact(raw, {"repository", "revision", "content_sha256"}, name)
    return ModelIdentity(
        repository=_text(raw["repository"], f"{name}.repository"),
        revision=_revision(raw["revision"], f"{name}.revision", optional=True),
        content_sha256=_sha256(raw["content_sha256"], f"{name}.content_sha256", optional=True),
    )


def _parse_processor(value: Any) -> ProcessorIdentity:
    raw = _mapping(value, "processor")
    _exact(
        raw,
        {"repository", "revision", "content_sha256", "tokenizer_sha256", "chat_template_sha256"},
        "processor",
    )
    return ProcessorIdentity(
        repository=_text(raw["repository"], "processor.repository"),
        revision=_revision(raw["revision"], "processor.revision", optional=True),
        content_sha256=_sha256(raw["content_sha256"], "processor.content_sha256", optional=True),
        tokenizer_sha256=_sha256(
            raw["tokenizer_sha256"], "processor.tokenizer_sha256", optional=True
        ),
        chat_template_sha256=_sha256(
            raw["chat_template_sha256"], "processor.chat_template_sha256", optional=True
        ),
    )


def _parse_runtime(value: Any) -> RuntimeIdentity:
    raw = _mapping(value, "runtime")
    _exact(
        raw,
        {
            "python_version", "torch_version", "transformers_version", "backend",
            "dtype", "device", "attention_implementation", "quantization",
            "batch_size", "use_cache",
        },
        "runtime",
    )
    return RuntimeIdentity(
        python_version=_optional_text(raw["python_version"], "runtime.python_version"),
        torch_version=_optional_text(raw["torch_version"], "runtime.torch_version"),
        transformers_version=_optional_text(
            raw["transformers_version"], "runtime.transformers_version"
        ),
        backend=_optional_text(raw["backend"], "runtime.backend"),
        dtype=_optional_text(raw["dtype"], "runtime.dtype"),
        device=_optional_text(raw["device"], "runtime.device"),
        attention_implementation=_optional_text(
            raw["attention_implementation"], "runtime.attention_implementation"
        ),
        quantization=_optional_text(raw["quantization"], "runtime.quantization"),
        batch_size=_integer(raw["batch_size"], "runtime.batch_size", optional=True),
        use_cache=_boolean(raw["use_cache"], "runtime.use_cache", optional=True),
    )


def _parse_source(value: Any) -> SourceDatasetIdentity:
    raw = _mapping(value, "source_dataset")
    _exact(raw, {"dataset_id", "revision", "manifest_sha256", "source_row_ids_sha256"}, "source_dataset")
    return SourceDatasetIdentity(
        dataset_id=_text(raw["dataset_id"], "source_dataset.dataset_id"),
        revision=_optional_text(raw["revision"], "source_dataset.revision"),
        manifest_sha256=_sha256(
            raw["manifest_sha256"], "source_dataset.manifest_sha256", optional=True
        ),
        source_row_ids_sha256=_sha256(
            raw["source_row_ids_sha256"], "source_dataset.source_row_ids_sha256", optional=True
        ),
    )


def _parse_split(value: Any) -> SplitReceipt:
    raw = _mapping(value, "split_receipt")
    _exact(
        raw,
        {
            "format", "status", "split_name", "split_policy", "assignment_seed",
            "dataset_manifest_sha256", "ordered_row_ids_sha256", "group_ids_sha256",
            "disjoint_from", "receipt_sha256",
        },
        "split_receipt",
    )
    status = _enum(raw["status"], {"verified", "unknown"}, "split_receipt.status")
    if raw["format"] != SPLIT_RECEIPT_FORMAT:
        raise ProvenanceContractError("split_receipt.format is unsupported")
    result = SplitReceipt(
        format=cast(Literal["truth_editing_split_receipt_v1"], SPLIT_RECEIPT_FORMAT),
        status=status,  # type: ignore[arg-type]
        split_name=_optional_text(raw["split_name"], "split_receipt.split_name"),
        split_policy=_optional_text(raw["split_policy"], "split_receipt.split_policy"),
        assignment_seed=_integer(raw["assignment_seed"], "split_receipt.assignment_seed", optional=True),
        dataset_manifest_sha256=_sha256(
            raw["dataset_manifest_sha256"],
            "split_receipt.dataset_manifest_sha256",
            optional=True,
        ),
        ordered_row_ids_sha256=_sha256(
            raw["ordered_row_ids_sha256"],
            "split_receipt.ordered_row_ids_sha256",
            optional=True,
        ),
        group_ids_sha256=_sha256(
            raw["group_ids_sha256"], "split_receipt.group_ids_sha256", optional=True
        ),
        disjoint_from=_string_tuple(
            raw["disjoint_from"], "split_receipt.disjoint_from", allow_empty=True
        ),
        receipt_sha256=_sha256(raw["receipt_sha256"], "split_receipt.receipt_sha256", optional=True),
    )
    if result.status == "verified":
        required = (
            result.split_name,
            result.split_policy,
            result.dataset_manifest_sha256,
            result.ordered_row_ids_sha256,
            result.group_ids_sha256,
            result.receipt_sha256,
        )
        if any(item is None for item in required):
            raise ProvenanceContractError(
                "verified split receipt is missing a required binding"
            )
    return result


def _require_proven_bindings(
    artifact: ActivationArtifact,
    hdf5: HDF5Inventory,
    model: ModelIdentity,
    processor: ProcessorIdentity,
    runtime: RuntimeIdentity,
    source: SourceDatasetIdentity,
    split: SplitReceipt,
) -> None:
    missing: list[str] = []
    if artifact.byte_size is None:
        missing.append("artifact.byte_size")
    if artifact.dvc_pointer_path is None:
        missing.append("artifact.dvc_pointer_path")
    if artifact.dvc_hash is None or artifact.dvc_hash_algorithm is None:
        missing.append("artifact.dvc_hash")
    if artifact.dvc_size is None:
        missing.append("artifact.dvc_size")
    if artifact.dvc_pointer_sha256 is None:
        missing.append("artifact.dvc_pointer_sha256")
    if artifact.direct_sha256 is None:
        missing.append("artifact.direct_sha256")
    if hdf5.hidden_dim is None or hdf5.storage_dtype is None:
        missing.append("hdf5_inventory.complete_shape_metadata")
    if any(value is None for value in hdf5.task_rows.values()) or any(
        value is None for value in hdf5.example_counts.values()
    ):
        missing.append("hdf5_inventory.complete_row_counts")
    if hdf5.finite_check != "full":
        missing.append("hdf5_inventory.finite_check=full")
    if model.revision is None:
        missing.append("model.revision")
    if processor.revision is None:
        missing.append("processor.revision")
    runtime_values = (
        runtime.python_version,
        runtime.torch_version,
        runtime.transformers_version,
        runtime.backend,
        runtime.dtype,
        runtime.device,
        runtime.attention_implementation,
        runtime.quantization,
        runtime.batch_size,
        runtime.use_cache,
    )
    if any(item is None for item in runtime_values):
        missing.append("runtime.complete")
    if source.revision is None or source.manifest_sha256 is None or source.source_row_ids_sha256 is None:
        missing.append("source_dataset.complete")
    if split.status != "verified":
        missing.append("split_receipt.status=verified")
    if missing:
        raise ProvenanceContractError(
            "proven evidence requires complete bindings; missing " + ", ".join(missing)
        )


def parse_activation_provenance(value: Any) -> ActivationProvenance:
    raw = _mapping(value, "sidecar")
    _exact(
        raw,
        {
            "format", "sidecar_id", "artifact", "hdf5_inventory", "model", "processor",
            "runtime", "source_dataset", "split_receipt", "evidence_status", "self_sha256",
        },
        "sidecar",
    )
    if raw["format"] != ACTIVATION_PROVENANCE_FORMAT:
        raise ProvenanceContractError("sidecar.format is unsupported")
    artifact = _parse_artifact(raw["artifact"])
    hdf5 = _parse_hdf5_inventory(raw["hdf5_inventory"])
    model = _parse_model(raw["model"], "model")
    processor = _parse_processor(raw["processor"])
    runtime = _parse_runtime(raw["runtime"])
    source = _parse_source(raw["source_dataset"])
    split = _parse_split(raw["split_receipt"])
    status = _enum(
        raw["evidence_status"], {"proven", "verified_metadata", "unknown"},
        "sidecar.evidence_status",
    )
    if status == "proven":
        _require_proven_bindings(artifact, hdf5, model, processor, runtime, source, split)
    self_sha = _self_hash(raw, "sidecar")
    return ActivationProvenance(
        format=cast(Literal["activation_provenance_sidecar_v1"], ACTIVATION_PROVENANCE_FORMAT),
        sidecar_id=_text(raw["sidecar_id"], "sidecar.sidecar_id"),
        artifact=artifact,
        hdf5_inventory=hdf5,
        model=model,
        processor=processor,
        runtime=runtime,
        source_dataset=source,
        split_receipt=split,
        evidence_status=status,  # type: ignore[arg-type]
        self_sha256=self_sha,
    )


def parse_inventory(value: Any) -> ActivationProvenanceInventory:
    raw = _mapping(value, "inventory")
    _exact(raw, {"format", "inventory_id", "entries", "self_sha256"}, "inventory")
    if raw["format"] != INVENTORY_FORMAT:
        raise ProvenanceContractError("inventory.format is unsupported")
    entries_raw = raw["entries"]
    if isinstance(entries_raw, (str, bytes)) or not isinstance(entries_raw, Sequence) or not entries_raw:
        raise ProvenanceContractError("inventory.entries must be a nonempty array")
    entries = tuple(parse_activation_provenance(item) for item in entries_raw)
    ids = [entry.sidecar_id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ProvenanceContractError("inventory sidecar IDs must be unique")
    self_sha = _self_hash(raw, "inventory")
    return ActivationProvenanceInventory(
        format=cast(Literal["activation_provenance_inventory_v1"], INVENTORY_FORMAT),
        inventory_id=_text(raw["inventory_id"], "inventory.inventory_id"),
        entries=entries,
        self_sha256=self_sha,
    )


def validate_compatibility(
    sidecar: ActivationProvenance,
    *,
    model_repository: str | None = None,
    model_revision: str | None = None,
    processor_repository: str | None = None,
    processor_revision: str | None = None,
    source_dataset_id: str | None = None,
    source_dataset_revision: str | None = None,
    split_name: str | None = None,
    runtime_backend: str | None = None,
    runtime_dtype: str | None = None,
    require_proven: bool = False,
) -> None:
    """Fail closed when a sidecar cannot satisfy a requested identity.

    ``None`` requirements mean "do not constrain this dimension".  A known
    requirement never matches an explicit unknown sidecar value.
    """

    if require_proven and sidecar.evidence_status != "proven":
        raise ProvenanceContractError("compatibility requires proven evidence")

    checks = (
        (model_repository, sidecar.model.repository, "model repository"),
        (model_revision, sidecar.model.revision, "model revision"),
        (processor_repository, sidecar.processor.repository, "processor repository"),
        (processor_revision, sidecar.processor.revision, "processor revision"),
        (source_dataset_id, sidecar.source_dataset.dataset_id, "source dataset"),
        (source_dataset_revision, sidecar.source_dataset.revision, "source dataset revision"),
        (split_name, sidecar.split_receipt.split_name, "split"),
        (runtime_backend, sidecar.runtime.backend, "runtime backend"),
        (runtime_dtype, sidecar.runtime.dtype, "runtime dtype"),
    )
    for expected, observed, label in checks:
        if expected is None:
            continue
        if observed is None:
            raise ProvenanceContractError(f"{label} is unknown")
        if expected != observed:
            raise ProvenanceContractError(
                f"{label} mismatch: expected {expected!r}, got {observed!r}"
            )


def load_activation_provenance(path: str | Path) -> ActivationProvenance:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceContractError(f"cannot read sidecar: {path}") from error
    return parse_activation_provenance(value)


def load_inventory(path: str | Path) -> ActivationProvenanceInventory:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceContractError(f"cannot read inventory: {path}") from error
    return parse_inventory(value)


def write_canonical_json(value: Mapping[str, Any], path: str | Path) -> None:
    """Write a parsed payload without touching any referenced artifact."""

    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ACTIVATION_PROVENANCE_FORMAT",
    "INVENTORY_FORMAT",
    "SPLIT_RECEIPT_FORMAT",
    "ActivationArtifact",
    "ActivationProvenance",
    "ActivationProvenanceInventory",
    "HDF5Inventory",
    "ModelIdentity",
    "ProcessorIdentity",
    "ProvenanceContractError",
    "RuntimeIdentity",
    "SourceDatasetIdentity",
    "SplitReceipt",
    "canonical_sha256",
    "load_activation_provenance",
    "load_inventory",
    "parse_activation_provenance",
    "parse_inventory",
    "validate_compatibility",
    "write_canonical_json",
]
