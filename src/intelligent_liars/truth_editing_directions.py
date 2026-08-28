"""Build and consume deterministic truth-editing direction banks.

The module has two public seams. ``build_direction_bank`` inventories legacy
probe JSON without copying large vectors. ``DirectionBank`` verifies and loads
those vectors, then compiles the small orthonormal basis used by a weight edit.
No optimizer, model loader, DVC client, or GPU runtime is imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from intelligent_liars.truth_editing_contracts import (
    DirectionBankManifest,
    canonical_json_bytes,
    canonical_sha256,
    parse_direction_bank_manifest,
)


SOURCE_INVENTORY_FORMAT = "truth_editing_direction_source_inventory_v1"
_HEX = frozenset("0123456789abcdef")
_STATUSES = frozenset({"candidate", "qualified", "diagnostic_only"})


class DirectionBankError(ValueError):
    """A source vector, bank artifact, or requested basis failed verification."""


@dataclass(frozen=True)
class DirectionCoverage:
    total: int
    source_records: int
    duplicate_records: int
    by_family: dict[str, int]
    by_layer: dict[int, int]
    by_status: dict[str, int]
    by_domain: dict[str, int]
    cells: tuple[tuple[str, str, int, str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "source_records": self.source_records,
            "duplicate_records": self.duplicate_records,
            "by_family": dict(sorted(self.by_family.items())),
            "by_layer": {str(k): v for k, v in sorted(self.by_layer.items())},
            "by_status": dict(sorted(self.by_status.items())),
            "by_domain": dict(sorted(self.by_domain.items())),
            "cells": [
                {
                    "family": family,
                    "domain": domain,
                    "source_layer": layer,
                    "status": status,
                    "count": count,
                }
                for family, domain, layer, status, count in self.cells
            ],
        }


@dataclass(frozen=True)
class DirectionBankBuild:
    manifest: DirectionBankManifest
    coverage: DirectionCoverage


@dataclass(frozen=True)
class CompiledBasis:
    direction_ids: tuple[str, ...]
    method: Literal["qr", "svd", "orthogonal_control", "shuffled_control"]
    requested_rank: int
    matrix: np.ndarray
    basis_sha256: str
    seed: int | None = None

    @property
    def width(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def rank(self) -> int:
        return int(self.matrix.shape[1])


@dataclass(frozen=True)
class CompiledBasisSet:
    """Canonical per-layer basis receipt consumed by an execution plan."""

    manifest_sha256: str
    method: Literal["qr", "svd", "orthogonal_control", "shuffled_control"]
    requested_rank: int
    by_layer: tuple[tuple[int, CompiledBasis], ...]
    basis_set_sha256: str
    model_sha256: str | None = None
    source_by_destination: tuple[tuple[int, int], ...] = ()
    destination_basis_sha256s: tuple[tuple[int, str], ...] = ()

    def verify(self) -> None:
        layers = tuple(layer for layer, _ in self.by_layer)
        if not layers or layers != tuple(sorted(set(layers))):
            raise DirectionBankError("basis set layers must be nonempty sorted unique")
        for _, basis in self.by_layer:
            _verify_compiled_basis(basis)
        relocated = bool(
            self.model_sha256
            or self.source_by_destination
            or self.destination_basis_sha256s
        )
        if relocated:
            if self.model_sha256 is None:
                raise DirectionBankError("relocated basis set lacks model identity")
            _sha(self.model_sha256, "relocated basis set.model_sha256")
            expected_destinations = tuple(layer for layer, _ in self.by_layer)
            lineage_destinations = tuple(
                destination for destination, _ in self.source_by_destination
            )
            hash_destinations = tuple(
                destination for destination, _ in self.destination_basis_sha256s
            )
            if lineage_destinations != expected_destinations:
                raise DirectionBankError(
                    "relocated basis source lineage differs from destination layers"
                )
            if hash_destinations != expected_destinations:
                raise DirectionBankError(
                    "relocated destination hashes differ from destination layers"
                )
            source_layers = tuple(source for _, source in self.source_by_destination)
            if not source_layers or len(set(source_layers)) != 1:
                raise DirectionBankError(
                    "relocated basis set must bind exactly one source layer"
                )
            expected_hashes = _destination_basis_hashes(
                self.manifest_sha256,
                self.model_sha256,
                self.source_by_destination,
                self.by_layer,
            )
            if self.destination_basis_sha256s != expected_hashes:
                raise DirectionBankError("relocated destination basis identity mismatch")
        observed = _basis_set_hash(
            self.manifest_sha256,
            self.method,
            self.requested_rank,
            self.by_layer,
            model_sha256=self.model_sha256,
            source_by_destination=self.source_by_destination,
            destination_basis_sha256s=self.destination_basis_sha256s,
        )
        if observed != self.basis_set_sha256:
            raise DirectionBankError("basis set identity mismatch")


@dataclass(frozen=True)
class CompiledControlBasisReceipt:
    format: Literal["truth_editing_control_basis_receipt_v1"]
    kind: Literal["orthogonal_control", "shuffled_control"]
    parent_basis_set_sha256: str
    seed: int
    rank_norm_policy: Literal["equal_rank_equal_unit_column_norm"]
    by_layer: tuple[tuple[int, CompiledBasis], ...]
    self_sha256: str

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "kind": self.kind,
            "parent_basis_set_sha256": self.parent_basis_set_sha256,
            "seed": self.seed,
            "rank_norm_policy": self.rank_norm_policy,
            "layers": [
                {
                    "source_layer": layer,
                    "derived_seed": basis.seed,
                    "rank": basis.rank,
                    "width": basis.width,
                    "matrix_sha256": basis.basis_sha256,
                }
                for layer, basis in self.by_layer
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._unsigned_dict()
        result["self_sha256"] = self.self_sha256
        return result

    def verify(self, parent: CompiledBasisSet) -> None:
        parent.verify()
        if parent.basis_set_sha256 != self.parent_basis_set_sha256:
            raise DirectionBankError("control receipt parent basis-set mismatch")
        if self.kind not in {"orthogonal_control", "shuffled_control"}:
            raise DirectionBankError("control receipt kind is unsupported")
        if self.rank_norm_policy != "equal_rank_equal_unit_column_norm":
            raise DirectionBankError("control receipt rank/norm policy mismatch")
        parent_layers = tuple(layer for layer, _ in parent.by_layer)
        control_layers = tuple(layer for layer, _ in self.by_layer)
        if control_layers != parent_layers:
            raise DirectionBankError("control receipt layer binding mismatch")
        for (layer, control), (_, parent_basis) in zip(
            self.by_layer, parent.by_layer, strict=True
        ):
            expected_seed = _control_layer_seed(
                self.parent_basis_set_sha256, self.seed, layer, self.kind
            )
            if control.seed != expected_seed:
                raise DirectionBankError("control receipt derived seed mismatch")
            if control.rank != parent_basis.rank or control.width != parent_basis.width:
                raise DirectionBankError("control receipt rank/width mismatch")
            _verify_compiled_basis(control)
            if not np.allclose(
                parent_basis.matrix.T @ control.matrix,
                np.zeros((parent_basis.rank, control.rank)),
                rtol=0.0,
                atol=1e-10,
            ):
                raise DirectionBankError("control receipt is not parent-orthogonal")
        if canonical_sha256(self._unsigned_dict()) != self.self_sha256:
            raise DirectionBankError("control receipt self hash mismatch")


def _exact(raw: dict[str, Any], expected: set[str], name: str) -> None:
    if set(raw) != expected:
        raise DirectionBankError(
            f"{name} fields differ; missing={sorted(expected - set(raw))}, "
            f"extra={sorted(set(raw) - expected)}"
        )


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DirectionBankError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DirectionBankError(f"{name} must be a nonempty trimmed string")
    return value


def _sha(value: Any, name: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(char not in _HEX for char in value)
    ):
        raise DirectionBankError(
            f"{name} must be lowercase hexadecimal length {length}"
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_vector(value: Any, *, width: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype="<f8")
    except (TypeError, ValueError) as error:
        raise DirectionBankError(f"{name} must contain numeric values") from error
    if vector.shape != (width,):
        raise DirectionBankError(
            f"{name} width mismatch: expected {width}, observed shape {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise DirectionBankError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise DirectionBankError(f"{name} has zero or invalid norm")
    result = np.asarray(vector / norm, dtype="<f8")
    result.setflags(write=False)
    return result


def vector_sha256(vector: np.ndarray) -> str:
    canonical = np.ascontiguousarray(vector, dtype="<f8")
    prefix = canonical_json_bytes(
        {"format": "truth_editing_vector_f64le_v1", "shape": list(canonical.shape)}
    )
    return hashlib.sha256(prefix + b"\0" + canonical.tobytes(order="C")).hexdigest()


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "unnamed"


def _safe_source_path(root: Path, raw_path: Any, name: str) -> tuple[Path, str]:
    relative = Path(_text(raw_path, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise DirectionBankError(f"{name} must be a contained relative path")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise DirectionBankError(f"{name} must not be a symlink")
    resolved_root = root.resolve()
    resolved = unresolved.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise DirectionBankError(f"{name} escapes the configured root")
    if not resolved.is_file():
        raise DirectionBankError(f"{name} does not identify a regular file")
    return resolved, relative.as_posix()


def _increment(target: dict[Any, int], key: Any) -> None:
    target[key] = target.get(key, 0) + 1


def _parse_source_config(config_path: Path, root: Path) -> dict[str, Any]:
    try:
        raw = _object(json.loads(config_path.read_text(encoding="utf-8")), "config")
    except (OSError, json.JSONDecodeError) as error:
        raise DirectionBankError(
            f"cannot read direction source config: {error}"
        ) from error
    _exact(raw, {"format", "manifest_id", "model", "sources", "self_sha256"}, "config")
    if raw["format"] != SOURCE_INVENTORY_FORMAT:
        raise DirectionBankError("unsupported direction source config format")
    claimed = _sha(raw["self_sha256"], "config.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if canonical_sha256(unsigned) != claimed:
        raise DirectionBankError("direction source config self hash mismatch")
    model = _object(raw["model"], "config.model")
    _exact(
        model,
        {
            "repository",
            "revision",
            "model_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
            "decoder_layer_count",
            "hidden_width",
        },
        "config.model",
    )
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise DirectionBankError("config.sources must be a nonempty array")
    return raw


def _expand_source_specs(values: list[Any], root: Path) -> list[dict[str, Any]]:
    simple_fields = {
        "source_id",
        "path",
        "qualification_status",
        "include_domain_directions",
        "include_general_directions",
        "provenance",
        "leakage",
    }
    glob_fields = (simple_fields - {"path"}) | {
        "path_glob",
        "expected_path_count",
        "ordered_paths_sha256",
    }
    expanded: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        source = _object(value, f"config.sources[{index}]")
        if set(source) == simple_fields:
            expanded.append(source)
            continue
        _exact(source, glob_fields, f"config.sources[{index}]")
        pattern = _text(source["path_glob"], f"config.sources[{index}].path_glob")
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise DirectionBankError(
                "source path_glob must be a contained relative glob"
            )
        matched = sorted(
            path.relative_to(root).as_posix()
            for path in root.glob(pattern)
            if path.is_file()
        )
        count = source["expected_path_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise DirectionBankError("expected_path_count must be a positive integer")
        if len(matched) != count:
            raise DirectionBankError(
                f"source glob match count mismatch: expected {count}, observed {len(matched)}"
            )
        expected_hash = _sha(
            source["ordered_paths_sha256"], "source.ordered_paths_sha256"
        )
        observed_hash = canonical_sha256(matched)
        if observed_hash != expected_hash:
            raise DirectionBankError(
                "source glob ordered path identity mismatch: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        base_id = _text(source["source_id"], "source.source_id")
        for relative in matched:
            item = {
                key: copy_json(value)
                for key, value in source.items()
                if key
                not in {
                    "path_glob",
                    "expected_path_count",
                    "ordered_paths_sha256",
                }
            }
            item["source_id"] = f"{base_id}-{Path(relative).stem}"
            item["path"] = relative
            expanded.append(item)
    priority = {"qualified": 0, "candidate": 1, "diagnostic_only": 2}
    return sorted(
        expanded,
        key=lambda item: (
            priority.get(item["qualification_status"], 99),
            item["source_id"],
            item["path"],
        ),
    )


def build_direction_bank(
    config_path: Path | str, *, root: Path | str
) -> DirectionBankBuild:
    """Inventory configured probe JSON and return a strict manifest plus coverage."""

    root_path = Path(root)
    raw = _parse_source_config(Path(config_path), root_path)
    model = raw["model"]
    width = model["hidden_width"]
    layers = model["decoder_layer_count"]
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise DirectionBankError("config.model.hidden_width must be a positive integer")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise DirectionBankError("config.model.decoder_layer_count must be positive")

    entries: list[dict[str, Any]] = []
    coverage_family: dict[str, int] = {}
    coverage_layer: dict[int, int] = {}
    coverage_status: dict[str, int] = {}
    coverage_domain: dict[str, int] = {}
    source_record_count = 0
    duplicate_record_count = 0
    seen_vectors: dict[str, tuple[str, int, str, str, float]] = {}
    source_ids: set[str] = set()
    for source_index, source_value in enumerate(
        _expand_source_specs(raw["sources"], root_path)
    ):
        source = _object(source_value, f"config.sources[{source_index}]")
        _exact(
            source,
            {
                "source_id",
                "path",
                "qualification_status",
                "include_domain_directions",
                "include_general_directions",
                "provenance",
                "leakage",
            },
            f"config.sources[{source_index}]",
        )
        source_id = _text(
            source["source_id"], f"config.sources[{source_index}].source_id"
        )
        if source_id in source_ids:
            raise DirectionBankError(f"duplicate source_id {source_id!r}")
        source_ids.add(source_id)
        status = _text(source["qualification_status"], "qualification_status")
        if status not in _STATUSES:
            raise DirectionBankError(f"unsupported qualification status {status!r}")
        if status == "qualified":
            raise DirectionBankError(
                "source inventory cannot self-attest qualification; qualified directions "
                "must be admitted from a clean reconstruction receipt"
            )
        for flag in ("include_domain_directions", "include_general_directions"):
            if not isinstance(source[flag], bool):
                raise DirectionBankError(f"{flag} must be boolean")
        source_path, source_rel = _safe_source_path(
            root_path, source["path"], f"config.sources[{source_index}].path"
        )
        file_hash = _file_sha256(source_path)
        try:
            payload = _object(
                json.loads(source_path.read_text(encoding="utf-8")), source_rel
            )
        except (OSError, json.JSONDecodeError) as error:
            raise DirectionBankError(f"cannot parse {source_rel}: {error}") from error

        collections: list[tuple[str, list[Any]]] = []
        if source["include_domain_directions"]:
            directions = payload.get("directions")
            if not isinstance(directions, list):
                raise DirectionBankError(f"{source_rel}.directions must be an array")
            collections.append(("directions", directions))
        if source["include_general_directions"]:
            general = _object(
                payload.get("general_domain"), f"{source_rel}.general_domain"
            )
            directions = general.get("directions")
            if not isinstance(directions, list):
                raise DirectionBankError(
                    f"{source_rel}.general_domain.directions must be an array"
                )
            collections.append(("general_domain/directions", directions))

        provenance = _object(source["provenance"], "source.provenance")
        _exact(
            provenance,
            {
                "dataset",
                "dataset_revision",
                "split",
                "ordered_row_ids_sha256",
                "source_code_revision",
            },
            "source.provenance",
        )
        leakage = _object(source["leakage"], "source.leakage")
        _exact(
            leakage,
            {
                "evaluation_disjoint",
                "heldout_family_disjoint",
                "sealed_audit_accessed",
                "audit_receipt_sha256",
            },
            "source.leakage",
        )
        for collection_path, values in collections:
            family = (
                "general"
                if collection_path.startswith("general_domain")
                else "domain_specific"
            )
            for item_index, item_value in enumerate(values):
                item = _object(
                    item_value, f"{source_rel}#/{collection_path}/{item_index}"
                )
                layer = item.get("layer")
                if (
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or not 0 <= layer < layers
                ):
                    raise DirectionBankError(
                        f"source layer {layer!r} exceeds model layer range"
                    )
                if item.get("feature_count") != width:
                    raise DirectionBankError(
                        "source feature_count differs from model hidden_width"
                    )
                task = _text(item.get("task"), "source direction task")
                try:
                    raw_vector = np.asarray(item.get("direction_vector"), dtype="<f8")
                    raw_norm = float(np.linalg.norm(raw_vector))
                except (TypeError, ValueError) as error:
                    raise DirectionBankError(
                        "source direction_vector must contain numeric values"
                    ) from error
                vector = _unit_vector(
                    item.get("direction_vector"),
                    width=width,
                    name="source direction_vector",
                )
                vector_hash = vector_sha256(vector)
                pointer = f"/{collection_path}/{item_index}/direction_vector"
                direction_id = (
                    f"{_slug(source_id)}--{family.replace('_', '-')}--l{layer}--"
                    f"{_slug(task)}--{vector_hash[:12]}"
                )
                receipt = canonical_sha256(
                    {
                        "format": "truth_editing_direction_qualification_receipt_v1",
                        "status": status,
                        "source_file_sha256": file_hash,
                        "json_pointer": pointer,
                        "vector_sha256": vector_hash,
                    }
                )
                intercept = item.get("intercept")
                if isinstance(intercept, bool) or not isinstance(
                    intercept, (int, float)
                ):
                    raise DirectionBankError("source intercept must be numeric")
                declared_norm = item.get("direction_norm")
                if declared_norm is not None and (
                    isinstance(declared_norm, bool)
                    or not isinstance(declared_norm, (int, float))
                    or not math.isclose(float(declared_norm), raw_norm, rel_tol=1e-9)
                ):
                    raise DirectionBankError(
                        "source direction_norm does not match direction_vector"
                    )
                sign = _text(
                    item.get("direction_sign_convention"), "direction sign convention"
                )
                source_record_count += 1
                metadata_identity = (
                    family,
                    layer,
                    task,
                    sign,
                    float(intercept) / raw_norm,
                )
                previous_metadata = seen_vectors.get(vector_hash)
                if previous_metadata is not None:
                    if previous_metadata != metadata_identity:
                        raise DirectionBankError(
                            "identical direction vector has conflicting family/layer/domain/sign "
                            f"metadata: {previous_metadata!r} versus {metadata_identity!r}"
                        )
                    duplicate_record_count += 1
                    continue
                seen_vectors[vector_hash] = metadata_identity
                entries.append(
                    {
                        "direction_id": direction_id,
                        "kind": "truth",
                        "family": family,
                        "basis_variant": "raw",
                        "domains": [task],
                        "source_layer": layer,
                        "width": width,
                        "rank": 1,
                        "artifact": {
                            "path": f"{source_rel}#{pointer}",
                            "file_sha256": file_hash,
                            "vector_sha256": vector_hash,
                        },
                        "construction": {
                            "basis_method": "raw",
                            "pooling": _text(
                                payload.get("pooling", "last_token"), "pooling"
                            ),
                            "token_position": "first_generated_token",
                            "normalization": "unit_l2",
                            "sign_convention": sign,
                            "intercept": float(intercept) / raw_norm,
                        },
                        "control_provenance": None,
                        "provenance": copy_json(provenance),
                        "leakage": copy_json(leakage),
                        "qualification": {
                            "status": status,
                            "receipt_sha256": receipt,
                            "finite": True,
                            "unit_norm": True,
                            "qualified_rank": 1,
                        },
                    }
                )
                _increment(coverage_family, family)
                _increment(coverage_layer, layer)
                _increment(coverage_status, status)
                _increment(coverage_domain, task)

    entries.sort(key=lambda item: item["direction_id"])
    manifest_raw: dict[str, Any] = {
        "format": "truth_editing_direction_bank_manifest_v1",
        "manifest_id": _text(raw["manifest_id"], "config.manifest_id"),
        "model": copy_json(model),
        "directions": entries,
    }
    manifest_raw["self_sha256"] = canonical_sha256(manifest_raw)
    try:
        manifest = parse_direction_bank_manifest(manifest_raw)
    except ValueError as error:
        raise DirectionBankError(
            f"built direction manifest is invalid: {error}"
        ) from error
    cell_counts: dict[tuple[str, str, int, str], int] = {}
    for entry in entries:
        cell = (
            entry["family"],
            entry["domains"][0],
            entry["source_layer"],
            entry["qualification"]["status"],
        )
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    return DirectionBankBuild(
        manifest=manifest,
        coverage=DirectionCoverage(
            total=len(entries),
            source_records=source_record_count,
            duplicate_records=duplicate_record_count,
            by_family=coverage_family,
            by_layer=coverage_layer,
            by_status=coverage_status,
            by_domain=coverage_domain,
            cells=tuple((*cell, count) for cell, count in sorted(cell_counts.items())),
        ),
    )


def copy_json(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _json_pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise DirectionBankError("artifact JSON pointer must start with '/'")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not token.isdigit():
                raise DirectionBankError(
                    f"invalid array token {token!r} in artifact pointer"
                )
            index = int(token)
            if index >= len(current):
                raise DirectionBankError("artifact JSON pointer array index is absent")
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise DirectionBankError(
                    f"artifact JSON pointer key {token!r} is absent"
                )
            current = current[token]
        else:
            raise DirectionBankError("artifact JSON pointer traverses a scalar")
    return current


def _canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype="<f8", order="C", copy=True)
    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            result[:, column_index] *= -1.0
    result.setflags(write=False)
    return result


def _basis_hash(
    matrix: np.ndarray,
    *,
    method: str,
    direction_ids: tuple[str, ...],
    requested_rank: int,
    seed: int | None = None,
) -> str:
    canonical = np.ascontiguousarray(matrix, dtype="<f8")
    identity = canonical_json_bytes(
        {
            "format": "truth_editing_compiled_basis_f64le_v1",
            "method": method,
            "direction_ids": list(direction_ids),
            "requested_rank": requested_rank,
            "seed": seed,
            "shape": list(canonical.shape),
        }
    )
    return hashlib.sha256(identity + b"\0" + canonical.tobytes(order="C")).hexdigest()


def _verify_compiled_basis(basis: CompiledBasis) -> None:
    if basis.matrix.shape != (basis.width, basis.requested_rank):
        raise DirectionBankError("compiled basis shape/rank mismatch")
    if not np.allclose(
        basis.matrix.T @ basis.matrix, np.eye(basis.rank), rtol=0.0, atol=1e-10
    ):
        raise DirectionBankError("compiled basis is not orthonormal")
    if len(basis.direction_ids) == 1 and basis.method in {"qr", "svd"}:
        observed = vector_sha256(basis.matrix[:, 0])
    else:
        observed = _basis_hash(
            basis.matrix,
            method=basis.method,
            direction_ids=basis.direction_ids,
            requested_rank=basis.requested_rank,
            seed=basis.seed,
        )
    if observed != basis.basis_sha256:
        raise DirectionBankError("compiled basis identity mismatch")


def _basis_set_hash(
    manifest_sha256: str,
    method: str,
    requested_rank: int,
    by_layer: tuple[tuple[int, CompiledBasis], ...],
    *,
    model_sha256: str | None = None,
    source_by_destination: tuple[tuple[int, int], ...] = (),
    destination_basis_sha256s: tuple[tuple[int, str], ...] = (),
) -> str:
    if source_by_destination:
        return canonical_sha256(
            {
                "format": "truth_editing_relocated_basis_set_receipt_v1",
                "manifest_sha256": manifest_sha256,
                "model_sha256": model_sha256,
                "method": method,
                "requested_rank": requested_rank,
                "destinations": [
                    {
                        "destination_layer": destination,
                        "source_layer": source,
                        "destination_basis_sha256": destination_hash,
                        "basis_sha256": basis.basis_sha256,
                        "direction_ids": list(basis.direction_ids),
                        "width": basis.width,
                        "rank": basis.rank,
                    }
                    for ((destination, basis), (_, source), (_, destination_hash))
                    in zip(
                        by_layer,
                        source_by_destination,
                        destination_basis_sha256s,
                        strict=True,
                    )
                ],
            }
        )
    return canonical_sha256(
        {
            "format": "truth_editing_compiled_basis_set_receipt_v1",
            "manifest_sha256": manifest_sha256,
            "method": method,
            "requested_rank": requested_rank,
            "layers": [
                {
                    "source_layer": layer,
                    "basis_sha256": basis.basis_sha256,
                    "direction_ids": list(basis.direction_ids),
                    "width": basis.width,
                    "rank": basis.rank,
                }
                for layer, basis in by_layer
            ],
        }
    )


def _destination_basis_hashes(
    manifest_sha256: str,
    model_sha256: str,
    source_by_destination: tuple[tuple[int, int], ...],
    by_layer: tuple[tuple[int, CompiledBasis], ...],
) -> tuple[tuple[int, str], ...]:
    return tuple(
        (
            destination,
            canonical_sha256(
                {
                    "format": "truth_editing_destination_basis_binding_v1",
                    "manifest_sha256": manifest_sha256,
                    "model_sha256": model_sha256,
                    "source_layer": source,
                    "destination_layer": destination,
                    "basis_sha256": basis.basis_sha256,
                    "direction_ids": list(basis.direction_ids),
                    "width": basis.width,
                    "rank": basis.rank,
                }
            ),
        )
        for ((destination, basis), (_, source)) in zip(
            by_layer, source_by_destination, strict=True
        )
    )


def _ordered_gram_schmidt(vectors: np.ndarray, requested_rank: int) -> np.ndarray:
    columns: list[np.ndarray] = []
    tolerance = np.finfo(np.float64).eps * max(vectors.shape) * 64
    for index in range(vectors.shape[1]):
        residual = np.array(vectors[:, index], dtype=np.float64, copy=True)
        # Reorthogonalize to make the selected span robust for near-collinear inputs.
        for _ in range(2):
            for basis_column in columns:
                residual -= basis_column * float(np.dot(basis_column, residual))
        norm = float(np.linalg.norm(residual))
        if norm <= tolerance:
            continue
        columns.append(residual / norm)
        if len(columns) == requested_rank:
            break
    if len(columns) != requested_rank:
        raise DirectionBankError(
            f"requested_rank {requested_rank} exceeds deterministic numerical rank "
            f"{len(columns)}"
        )
    result = np.asarray(np.column_stack(columns), dtype="<f8")
    result.setflags(write=False)
    return result


class DirectionBank:
    """Verified provider for manifest directions and deterministic bases."""

    def __init__(self, manifest: DirectionBankManifest, root: Path) -> None:
        self.manifest = manifest
        self._root = root.resolve()
        self._by_id = {item.direction_id: item for item in manifest.directions}

    @classmethod
    def open(cls, manifest_path: Path | str, *, root: Path | str) -> DirectionBank:
        try:
            raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            manifest = parse_direction_bank_manifest(raw)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise DirectionBankError(
                f"cannot open direction bank manifest: {error}"
            ) from error
        return cls(manifest, Path(root))

    def load_vector(
        self, direction_id: str, *, require_qualified: bool = True
    ) -> np.ndarray:
        try:
            entry = self._by_id[direction_id]
        except KeyError as error:
            raise DirectionBankError(
                f"unknown direction_id {direction_id!r}"
            ) from error
        if require_qualified and entry.qualification.status not in {
            "qualified",
            "qualified_control",
        }:
            raise DirectionBankError(
                f"direction {direction_id!r} is not qualified for optimization"
            )
        try:
            relative_text, pointer = entry.artifact.path.split("#", 1)
        except ValueError as error:
            raise DirectionBankError(
                f"direction {direction_id!r} artifact path lacks a JSON pointer"
            ) from error
        source_path, _ = _safe_source_path(
            self._root, relative_text, f"direction {direction_id!r} artifact path"
        )
        observed_file_hash = _file_sha256(source_path)
        if observed_file_hash != entry.artifact.file_sha256:
            raise DirectionBankError(
                f"direction {direction_id!r} file hash mismatch: "
                f"expected {entry.artifact.file_sha256}, observed {observed_file_hash}"
            )
        if source_path.suffix == ".npy":
            tokens = pointer.split("/")
            if len(tokens) != 2 or tokens[0] != "row" or not tokens[1].isdigit():
                raise DirectionBankError(
                    "npy artifact locator must be exactly #row/<index>"
                )
            try:
                matrix = np.load(source_path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise DirectionBankError(
                    f"cannot parse npy direction artifact: {error}"
                ) from error
            if matrix.dtype.str not in {"<f8", "=f8"} or matrix.ndim != 2:
                raise DirectionBankError(
                    "npy direction artifact must be a two-dimensional "
                    "little-endian float64 matrix"
                )
            if matrix.shape[1] != entry.width:
                raise DirectionBankError("npy direction artifact width mismatch")
            row_index = int(tokens[1])
            if row_index >= matrix.shape[0]:
                raise DirectionBankError("npy direction artifact row index is absent")
            vector = np.asarray(matrix[row_index], dtype="<f8")
            if not np.isfinite(vector).all() or not math.isclose(
                float(np.linalg.norm(vector)), 1.0, rel_tol=0.0, abs_tol=1e-10
            ):
                raise DirectionBankError(
                    "npy direction artifact row must be finite and unit normalized"
                )
            vector.setflags(write=False)
        else:
            try:
                document = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise DirectionBankError(
                    f"cannot parse direction artifact: {error}"
                ) from error
            vector = _unit_vector(
                _json_pointer(document, pointer),
                width=entry.width,
                name=f"direction {direction_id!r}",
            )
        observed_vector_hash = vector_sha256(vector)
        if observed_vector_hash != entry.artifact.vector_sha256:
            raise DirectionBankError(
                f"direction {direction_id!r} vector hash mismatch: "
                f"expected {entry.artifact.vector_sha256}, observed {observed_vector_hash}"
            )
        return vector

    def compile_basis(
        self,
        direction_ids: tuple[str, ...] | list[str],
        *,
        method: Literal["qr", "svd"],
        requested_rank: int,
    ) -> CompiledBasis:
        ids = tuple(direction_ids)
        if not ids or len(set(ids)) != len(ids):
            raise DirectionBankError("direction_ids must be nonempty and unique")
        if method not in {"qr", "svd"}:
            raise DirectionBankError("basis method must be 'qr' or 'svd'")
        if isinstance(requested_rank, bool) or not isinstance(requested_rank, int):
            raise DirectionBankError("requested_rank must be an integer")
        if requested_rank <= 0 or requested_rank > len(ids):
            raise DirectionBankError(
                "requested_rank must be positive and no greater than selected directions"
            )
        entries = []
        for direction_id in ids:
            if direction_id not in self._by_id:
                raise DirectionBankError(f"unknown direction_id {direction_id!r}")
            entries.append(self._by_id[direction_id])
        if len({entry.source_layer for entry in entries}) != 1:
            raise DirectionBankError("basis directions must use the same source layer")
        if len({entry.width for entry in entries}) != 1:
            raise DirectionBankError("basis directions must use the same width")
        vectors = np.column_stack([self.load_vector(item) for item in ids])
        if len(ids) == 1:
            matrix = np.asarray(vectors, dtype="<f8")
            matrix.setflags(write=False)
        elif method == "qr":
            matrix = _ordered_gram_schmidt(vectors, requested_rank)
        else:
            numerical_rank = int(np.linalg.matrix_rank(vectors))
            if requested_rank > numerical_rank:
                raise DirectionBankError(
                    f"requested_rank {requested_rank} exceeds numerical rank {numerical_rank}"
                )
            u, _, _ = np.linalg.svd(vectors, full_matrices=False)
            matrix = _canonicalize_columns(u[:, :requested_rank])
        if len(ids) == 1 and requested_rank == 1:
            basis_hash = entries[0].artifact.vector_sha256
        else:
            basis_hash = _basis_hash(
                matrix,
                method=method,
                direction_ids=ids,
                requested_rank=requested_rank,
            )
        return CompiledBasis(ids, method, requested_rank, matrix, basis_hash)

    def compile_basis_set(
        self,
        direction_ids: tuple[str, ...] | list[str],
        *,
        method: Literal["qr", "svd"],
        requested_rank: int,
    ) -> CompiledBasisSet:
        ids = tuple(direction_ids)
        if not ids or len(set(ids)) != len(ids):
            raise DirectionBankError("direction_ids must be nonempty and unique")
        grouped: dict[int, list[str]] = {}
        for direction_id in ids:
            try:
                layer = self._by_id[direction_id].source_layer
            except KeyError as error:
                raise DirectionBankError(
                    f"unknown direction_id {direction_id!r}"
                ) from error
            grouped.setdefault(layer, []).append(direction_id)
        by_layer = tuple(
            (
                layer,
                self.compile_basis(
                    tuple(sorted(grouped[layer])),
                    method=method,
                    requested_rank=requested_rank,
                ),
            )
            for layer in sorted(grouped)
        )
        result = CompiledBasisSet(
            manifest_sha256=self.manifest.self_sha256,
            method=method,
            requested_rank=requested_rank,
            by_layer=by_layer,
            basis_set_sha256=_basis_set_hash(
                self.manifest.self_sha256, method, requested_rank, by_layer
            ),
        )
        result.verify()
        return result

    def compile_relocated_basis_set(
        self,
        direction_ids: tuple[str, ...] | list[str],
        *,
        destination_layers: tuple[int, ...] | list[int],
        method: Literal["qr", "svd"],
        requested_rank: int,
        expected_model_sha256: str | None = None,
    ) -> CompiledBasisSet:
        """Compile one source-layer basis for explicit destination writer layers."""

        ids = tuple(direction_ids)
        if not ids or len(set(ids)) != len(ids):
            raise DirectionBankError("direction_ids must be nonempty and unique")
        destinations = tuple(destination_layers)
        if not destinations:
            raise DirectionBankError("destination_layers must be nonempty")
        if any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in destinations
        ):
            raise DirectionBankError("destination_layers must contain integers")
        if destinations != tuple(sorted(set(destinations))):
            raise DirectionBankError(
                "destination_layers must be sorted unique; reordering is not canonical"
            )
        layer_count = self.manifest.model.decoder_layer_count
        if any(layer < 0 or layer >= layer_count for layer in destinations):
            raise DirectionBankError("destination layer is outside the model layer range")
        model_sha256 = self.manifest.model.model_sha256
        if expected_model_sha256 is not None and expected_model_sha256 != model_sha256:
            raise DirectionBankError("direction-bank model identity mismatch")

        entries = []
        for direction_id in ids:
            try:
                entries.append(self._by_id[direction_id])
            except KeyError as error:
                raise DirectionBankError(
                    f"unknown direction_id {direction_id!r}"
                ) from error
        source_layers = {entry.source_layer for entry in entries}
        if len(source_layers) != 1:
            raise DirectionBankError(
                "relocated basis directions must use the same source layer"
            )
        if any(entry.width != self.manifest.model.hidden_width for entry in entries):
            raise DirectionBankError(
                "direction width is incompatible with the bound model hidden width"
            )
        source_layer = next(iter(source_layers))
        basis = self.compile_basis(
            tuple(sorted(ids)), method=method, requested_rank=requested_rank
        )
        if basis.width != self.manifest.model.hidden_width:
            raise DirectionBankError(
                "compiled basis width is incompatible with the bound model"
            )
        by_layer = tuple((layer, basis) for layer in destinations)
        lineage = tuple((layer, source_layer) for layer in destinations)
        destination_hashes = _destination_basis_hashes(
            self.manifest.self_sha256,
            model_sha256,
            lineage,
            by_layer,
        )
        result = CompiledBasisSet(
            manifest_sha256=self.manifest.self_sha256,
            method=method,
            requested_rank=requested_rank,
            by_layer=by_layer,
            basis_set_sha256=_basis_set_hash(
                self.manifest.self_sha256,
                method,
                requested_rank,
                by_layer,
                model_sha256=model_sha256,
                source_by_destination=lineage,
                destination_basis_sha256s=destination_hashes,
            ),
            model_sha256=model_sha256,
            source_by_destination=lineage,
            destination_basis_sha256s=destination_hashes,
        )
        result.verify()
        return result


def _control_basis(
    parent: CompiledBasis,
    *,
    seed: int,
    method: Literal["orthogonal_control", "shuffled_control"],
) -> CompiledBasis:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DirectionBankError("control seed must be a nonnegative integer")
    if parent.width - parent.rank < parent.rank:
        raise DirectionBankError(
            "equal-rank orthogonal control requires width of at least twice parent rank"
        )
    rng = np.random.default_rng(seed)
    if method == "shuffled_control":
        candidates = parent.matrix[rng.permutation(parent.width), :]
        candidates = np.column_stack(
            [candidates, rng.standard_normal((parent.width, parent.rank))]
        )
    else:
        candidates = rng.standard_normal((parent.width, parent.rank * 2))
    projected = candidates - parent.matrix @ (parent.matrix.T @ candidates)
    matrix = _canonicalize_columns(_ordered_gram_schmidt(projected, parent.rank))
    identity_ids = (parent.basis_sha256,)
    return CompiledBasis(
        identity_ids,
        method,
        parent.rank,
        matrix,
        _basis_hash(
            matrix,
            method=method,
            direction_ids=identity_ids,
            requested_rank=parent.rank,
            seed=seed,
        ),
        seed,
    )


def _control_layer_seed(
    parent_basis_set_sha256: str,
    seed: int,
    layer: int,
    kind: str,
) -> int:
    digest = canonical_sha256(
        {
            "format": "truth_editing_control_layer_seed_v1",
            "parent_basis_set_sha256": parent_basis_set_sha256,
            "seed": seed,
            "source_layer": layer,
            "kind": kind,
        }
    )
    return int(digest[:16], 16) % (2**63)


def compile_control_basis_receipt(
    parent: CompiledBasisSet,
    *,
    kind: Literal["orthogonal_control", "shuffled_control"],
    seed: int,
) -> CompiledControlBasisReceipt:
    parent.verify()
    if kind not in {"orthogonal_control", "shuffled_control"}:
        raise DirectionBankError(
            "control kind must be orthogonal_control or shuffled_control"
        )
    by_layer: list[tuple[int, CompiledBasis]] = []
    for layer, parent_basis in parent.by_layer:
        layer_seed = _control_layer_seed(parent.basis_set_sha256, seed, layer, kind)
        if kind == "orthogonal_control":
            control = compile_equal_rank_orthogonal_control(
                parent_basis, seed=layer_seed
            )
        else:
            control = compile_shuffled_control(parent_basis, seed=layer_seed)
        by_layer.append((layer, control))
    unsigned = {
        "format": "truth_editing_control_basis_receipt_v1",
        "kind": kind,
        "parent_basis_set_sha256": parent.basis_set_sha256,
        "seed": seed,
        "rank_norm_policy": "equal_rank_equal_unit_column_norm",
        "layers": [
            {
                "source_layer": layer,
                "derived_seed": basis.seed,
                "rank": basis.rank,
                "width": basis.width,
                "matrix_sha256": basis.basis_sha256,
            }
            for layer, basis in by_layer
        ],
    }
    receipt = CompiledControlBasisReceipt(
        format="truth_editing_control_basis_receipt_v1",
        kind=kind,
        parent_basis_set_sha256=parent.basis_set_sha256,
        seed=seed,
        rank_norm_policy="equal_rank_equal_unit_column_norm",
        by_layer=tuple(by_layer),
        self_sha256=canonical_sha256(unsigned),
    )
    receipt.verify(parent)
    return receipt


def compile_control_basis_set(
    parent: CompiledBasisSet,
    *,
    kind: Literal["orthogonal_control", "shuffled_control"],
    seed: int,
    orthogonal_to: tuple[CompiledBasisSet, ...] = (),
) -> CompiledBasisSet:
    """Compile a runtime-compatible matched control from a relocated parent.

    The result keeps the parent's model, source-layer, destination-layer, rank,
    and unit-column-norm contract while replacing only its direction subspace.
    """

    if not orthogonal_to:
        control_by_layer = compile_control_basis_receipt(
            parent, kind=kind, seed=seed
        ).by_layer
    else:
        for exclusion in orthogonal_to:
            exclusion.verify()
            if tuple(layer for layer, _ in exclusion.by_layer) != tuple(
                layer for layer, _ in parent.by_layer
            ):
                raise DirectionBankError(
                    "control exclusion layers must match the parent"
                )
        compiled: list[tuple[int, CompiledBasis]] = []
        for position, (layer, parent_basis) in enumerate(parent.by_layer):
            layer_seed = _control_layer_seed(
                parent.basis_set_sha256, seed, layer, kind
            )
            exclusions = tuple(
                exclusion.by_layer[position][1] for exclusion in orthogonal_to
            )
            compiled.append(
                (
                    layer,
                    _control_basis_with_exclusions(
                        parent_basis,
                        exclusions=exclusions,
                        seed=layer_seed,
                        method=kind,
                    ),
                )
            )
        control_by_layer = tuple(compiled)
    destination_hashes = _destination_basis_hashes(
        parent.manifest_sha256,
        parent.model_sha256,
        parent.source_by_destination,
        control_by_layer,
    )
    result = CompiledBasisSet(
        manifest_sha256=parent.manifest_sha256,
        method=kind,
        requested_rank=parent.requested_rank,
        by_layer=control_by_layer,
        basis_set_sha256=_basis_set_hash(
            parent.manifest_sha256,
            kind,
            parent.requested_rank,
            control_by_layer,
            model_sha256=parent.model_sha256,
            source_by_destination=parent.source_by_destination,
            destination_basis_sha256s=destination_hashes,
        ),
        model_sha256=parent.model_sha256,
        source_by_destination=parent.source_by_destination,
        destination_basis_sha256s=destination_hashes,
    )
    result.verify()
    return result


def _control_basis_with_exclusions(
    parent: CompiledBasis,
    *,
    exclusions: tuple[CompiledBasis, ...],
    seed: int,
    method: Literal["orthogonal_control", "shuffled_control"],
) -> CompiledBasis:
    matrices = (parent.matrix,) + tuple(item.matrix for item in exclusions)
    if any(matrix.shape[0] != parent.width for matrix in matrices):
        raise DirectionBankError("control exclusions have incompatible widths")
    exclusion = np.concatenate(matrices, axis=1)
    exclusion_q = np.linalg.qr(exclusion, mode="reduced")[0]
    if parent.width - exclusion_q.shape[1] < parent.rank:
        raise DirectionBankError(
            "matched control complement is too small for equal rank"
        )
    rng = np.random.default_rng(seed)
    candidates = (
        parent.matrix[rng.permutation(parent.width), :]
        if method == "shuffled_control"
        else rng.standard_normal((parent.width, parent.rank))
    )
    candidates = np.column_stack(
        (candidates, rng.standard_normal((parent.width, parent.rank * 2)))
    )
    projected = candidates - exclusion_q @ (exclusion_q.T @ candidates)
    matrix = _canonicalize_columns(_ordered_gram_schmidt(projected, parent.rank))
    identity_ids = (
        parent.basis_sha256,
        *(item.basis_sha256 for item in exclusions),
    )
    return CompiledBasis(
        identity_ids,
        method,
        parent.rank,
        matrix,
        _basis_hash(
            matrix,
            method=method,
            direction_ids=identity_ids,
            requested_rank=parent.rank,
            seed=seed,
        ),
        seed,
    )


def parse_control_basis_receipt(
    value: Any, *, parent: CompiledBasisSet
) -> CompiledControlBasisReceipt:
    raw = _object(value, "control basis receipt")
    _exact(
        raw,
        {
            "format",
            "kind",
            "parent_basis_set_sha256",
            "seed",
            "rank_norm_policy",
            "layers",
            "self_sha256",
        },
        "control basis receipt",
    )
    claimed = _sha(raw["self_sha256"], "control basis receipt.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if canonical_sha256(unsigned) != claimed:
        raise DirectionBankError("control basis receipt self hash mismatch")
    if raw["format"] != "truth_editing_control_basis_receipt_v1":
        raise DirectionBankError("control basis receipt format is unsupported")
    if raw["parent_basis_set_sha256"] != parent.basis_set_sha256:
        raise DirectionBankError("control receipt parent basis-set mismatch")
    kind = raw["kind"]
    if kind not in {"orthogonal_control", "shuffled_control"}:
        raise DirectionBankError("control basis receipt kind is unsupported")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DirectionBankError("control basis receipt seed must be nonnegative")
    expected = compile_control_basis_receipt(parent, kind=kind, seed=seed)
    if expected.to_dict() != raw:
        raise DirectionBankError(
            "persisted control basis receipt differs from deterministic compilation"
        )
    return expected


def compile_equal_rank_orthogonal_control(
    parent: CompiledBasis, *, seed: int
) -> CompiledBasis:
    return _control_basis(parent, seed=seed, method="orthogonal_control")


def compile_shuffled_control(parent: CompiledBasis, *, seed: int) -> CompiledBasis:
    return _control_basis(parent, seed=seed, method="shuffled_control")


def build_reconstruction_workload(
    manifest: DirectionBankManifest,
    *,
    activation_input: dict[str, Any],
    construction_row_allowlist_sha256: str | None,
    output_root: str,
    maximum_external_spend_usd: float,
) -> dict[str, Any]:
    """Describe every unqualified domain-by-layer refit without reading evaluation data."""

    _exact(
        activation_input,
        {
            "path",
            "byte_size",
            "direct_sha256",
            "dvc_md5",
            "evidence_status",
        },
        "activation_input",
    )
    _text(activation_input["path"], "activation_input.path")
    _sha(activation_input["direct_sha256"], "activation_input.direct_sha256")
    _sha(activation_input["dvc_md5"], "activation_input.dvc_md5", length=32)
    if activation_input["evidence_status"] != "verified_metadata":
        raise DirectionBankError(
            "activation input must have verified_metadata evidence"
        )
    byte_size = activation_input["byte_size"]
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
        raise DirectionBankError("activation_input.byte_size must be positive")
    if construction_row_allowlist_sha256 is not None:
        _sha(
            construction_row_allowlist_sha256,
            "construction_row_allowlist_sha256",
        )
    if (
        isinstance(maximum_external_spend_usd, bool)
        or not isinstance(maximum_external_spend_usd, float)
        or not 0.0 <= maximum_external_spend_usd <= 15.0
    ):
        raise DirectionBankError("maximum external spend must be a float from 0 to 15")
    root = _text(output_root, "output_root")
    if root.startswith("/") or ".." in Path(root).parts:
        raise DirectionBankError("output_root must be a contained relative path")

    domains = sorted(
        {domain for entry in manifest.directions for domain in entry.domains}
    )
    by_cell: dict[tuple[str, int], list[Any]] = {}
    for entry in manifest.directions:
        for domain in entry.domains:
            by_cell.setdefault((domain, entry.source_layer), []).append(entry)
    cells: list[dict[str, Any]] = []
    ready = construction_row_allowlist_sha256 is not None
    for domain in domains:
        for layer in range(manifest.model.decoder_layer_count):
            existing = by_cell.get((domain, layer), [])
            qualified = sum(
                entry.qualification.status == "qualified" for entry in existing
            )
            status_counts: dict[str, int] = {}
            for entry in existing:
                status = entry.qualification.status
                status_counts[status] = status_counts.get(status, 0) + 1
            action = "reuse_qualified" if qualified else "refit"
            cells.append(
                {
                    "domain": domain,
                    "source_layer": layer,
                    "action": action,
                    "existing_status_counts": dict(sorted(status_counts.items())),
                    "execution_status": (
                        "ready"
                        if qualified or ready
                        else "blocked_missing_clean_construction_row_allowlist"
                    ),
                    "expected_output": (
                        f"{root}/layer-{layer:02d}/{_slug(domain)}.json"
                    ),
                }
            )
    workload: dict[str, Any] = {
        "format": "truth_editing_direction_reconstruction_workload_v1",
        "direction_manifest_sha256": manifest.self_sha256,
        "model_revision": manifest.model.revision,
        "activation_input": copy_json(activation_input),
        "data_access": {
            "allowed_partition": "direction_construction",
            "construction_row_allowlist_sha256": construction_row_allowlist_sha256,
            "forbidden_partitions": [
                "optimizer_validation",
                "final_test",
                "final_audit",
                "judge_calibration",
            ],
        },
        "fit": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "solver": "liblinear",
            "class_weight": "balanced",
            "regularization_c": 1.0,
            "max_iter": 1000,
            "random_seed": 0,
            "normalization": "unit_l2_with_intercept_rescaled",
        },
        "resources": {
            "gpu_required": False,
            "minimum_ram_bytes": 68719476736,
            "minimum_free_disk_bytes": 75161927680,
            "maximum_external_spend_usd": maximum_external_spend_usd,
            "runtime_estimate_status": "benchmark_required",
        },
        "target_cell_count": len(cells),
        "refit_cell_count": sum(cell["action"] == "refit" for cell in cells),
        "blocked_cell_count": sum(
            cell["execution_status"].startswith("blocked") for cell in cells
        ),
        "cells": cells,
    }
    workload["self_sha256"] = canonical_sha256(workload)
    return workload


def _verify_refit_receipt_identity(value: Any, name: str) -> None:
    try:
        raw = asdict(value)
    except TypeError as error:
        raise DirectionBankError(f"{name} must be a frozen refit receipt") from error
    claimed = raw.pop("self_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(raw) != claimed:
        raise DirectionBankError(f"{name} self hash mismatch")


def promote_reconstructed_direction_bank(
    base_manifest: DirectionBankManifest,
    refit_receipt: Any,
    *,
    expected_plan_sha256: str,
    root: Path | str,
    manifest_id: str,
) -> DirectionBankManifest:
    """Promote a complete refit receipt; source configs cannot create qualification."""

    _sha(expected_plan_sha256, "expected_plan_sha256")
    _text(manifest_id, "manifest_id")
    _verify_refit_receipt_identity(refit_receipt, "refit aggregate receipt")
    if (
        getattr(refit_receipt, "format", None)
        != "truth_editing_direction_refit_receipt_v1"
    ):
        raise DirectionBankError("refit aggregate receipt format is unsupported")
    if refit_receipt.plan_sha256 != expected_plan_sha256:
        raise DirectionBankError("refit aggregate receipt plan identity mismatch")
    expected_layers = tuple(range(base_manifest.model.decoder_layer_count))
    layers = tuple(item.source_layer for item in refit_receipt.layer_receipts)
    if layers != expected_layers:
        raise DirectionBankError(
            "refit aggregate receipt must cover every layer in order"
        )
    expected_domains = {
        domain for item in base_manifest.directions for domain in item.domains
    }
    if not expected_domains:
        raise DirectionBankError("base manifest has no direction domains")
    first_domains = tuple(
        item.domain for item in refit_receipt.layer_receipts[0].direction_receipts
    )
    if set(first_domains) != expected_domains or len(first_domains) != len(
        expected_domains
    ):
        raise DirectionBankError(
            "refit receipt domain coverage differs from base manifest"
        )
    expected_domain_hash = canonical_sha256(list(first_domains))
    expected_count = len(expected_layers) * len(first_domains)
    if refit_receipt.completed_direction_count != expected_count:
        raise DirectionBankError("refit aggregate completed direction count mismatch")

    entries: list[dict[str, Any]] = []
    common_allowlist: str | None = None
    common_group_manifest: str | None = None
    common_dataset_manifest: str | None = None
    common_activation: tuple[str, str, str] | None = None
    common_probe_config: str | None = None
    for layer_receipt in refit_receipt.layer_receipts:
        _verify_refit_receipt_identity(layer_receipt, "refit layer receipt")
        if layer_receipt.format != "truth_editing_direction_refit_layer_receipt_v1":
            raise DirectionBankError("refit layer receipt format is unsupported")
        if (
            layer_receipt.matrix_shape
            != (
                len(first_domains),
                base_manifest.model.hidden_width,
            )
            or layer_receipt.matrix_dtype != "<f8"
        ):
            raise DirectionBankError("refit layer matrix shape/dtype mismatch")
        if layer_receipt.ordered_domains_sha256 != expected_domain_hash:
            raise DirectionBankError("refit layer ordered-domain identity mismatch")
        if (
            tuple(item.domain for item in layer_receipt.direction_receipts)
            != first_domains
        ):
            raise DirectionBankError("refit layer direction order mismatch")
        for row_index, item in enumerate(layer_receipt.direction_receipts):
            _verify_refit_receipt_identity(item, "refit direction receipt")
            if item.format != "truth_editing_refit_direction_v1":
                raise DirectionBankError(
                    "refit direction receipt format is unsupported"
                )
            if (
                item.source_layer != layer_receipt.source_layer
                or item.artifact_row != row_index
                or item.artifact_path
                != f"{layer_receipt.artifact_path}#row/{row_index}"
                or item.artifact_file_sha256 != layer_receipt.artifact_file_sha256
                or item.shard_id != layer_receipt.shard_id
            ):
                raise DirectionBankError(
                    "refit direction/layer artifact binding mismatch"
                )
            if (
                item.width != base_manifest.model.hidden_width
                or item.rank != 1
                or item.qualified_rank != 1
                or not item.finite
                or not item.unit_norm
            ):
                raise DirectionBankError(
                    "refit direction failed width/rank/finite/norm checks"
                )
            if (
                item.model_repository != base_manifest.model.repository
                or item.model_revision != base_manifest.model.revision
                or item.model_sha256 != base_manifest.model.model_sha256
            ):
                raise DirectionBankError("refit direction model identity mismatch")
            if item.reconstruction_plan_sha256 != expected_plan_sha256:
                raise DirectionBankError("refit direction plan identity mismatch")
            if item.construction_selector != "direction_construction":
                raise DirectionBankError(
                    "refit direction construction selector mismatch"
                )
            _sha(item.activation_direct_sha256, "refit activation direct sha256")
            _sha(item.activation_dvc_md5, "refit activation dvc md5", length=32)
            _sha(item.activation_sidecar_sha256, "refit activation sidecar sha256")
            _sha(item.probe_config_sha256, "refit probe config sha256")
            for field_name in (
                "construction_row_allowlist_sha256",
                "ordered_row_ids_sha256",
                "construction_group_manifest_sha256",
                "dataset_manifest_sha256",
            ):
                _sha(getattr(item, field_name), f"refit direction {field_name}")
            if common_allowlist is None:
                common_allowlist = item.construction_row_allowlist_sha256
                common_group_manifest = item.construction_group_manifest_sha256
                common_dataset_manifest = item.dataset_manifest_sha256
                common_activation = (
                    item.activation_direct_sha256,
                    item.activation_dvc_md5,
                    item.activation_sidecar_sha256,
                )
                common_probe_config = item.probe_config_sha256
            if (
                item.construction_row_allowlist_sha256 != common_allowlist
                or item.construction_group_manifest_sha256 != common_group_manifest
                or item.dataset_manifest_sha256 != common_dataset_manifest
                or (
                    item.activation_direct_sha256,
                    item.activation_dvc_md5,
                    item.activation_sidecar_sha256,
                )
                != common_activation
                or item.probe_config_sha256 != common_probe_config
            ):
                raise DirectionBankError(
                    "refit directions disagree on construction provenance"
                )
            family = "general" if item.domain == "general_domain" else "domain_specific"
            entries.append(
                {
                    "direction_id": item.direction_id,
                    "kind": "truth",
                    "family": family,
                    "basis_variant": "raw",
                    "domains": [item.domain],
                    "source_layer": item.source_layer,
                    "width": item.width,
                    "rank": item.rank,
                    "artifact": {
                        "path": item.artifact_path,
                        "file_sha256": item.artifact_file_sha256,
                        "vector_sha256": item.vector_sha256,
                    },
                    "construction": {
                        "basis_method": "raw",
                        "pooling": "mean_answer_tokens_per_example",
                        "token_position": "first_generated_token",
                        "normalization": "unit_l2",
                        "sign_convention": item.sign_convention,
                        "intercept": item.rescaled_intercept,
                    },
                    "control_provenance": None,
                    "provenance": {
                        "dataset": f"direction-refit:{item.dataset_manifest_sha256}",
                        "dataset_revision": item.dataset_manifest_sha256,
                        "split": "direction_construction",
                        "ordered_row_ids_sha256": item.ordered_row_ids_sha256,
                        "source_code_revision": item.source_code_revision,
                    },
                    "leakage": {
                        "evaluation_disjoint": True,
                        "heldout_family_disjoint": True,
                        "sealed_audit_accessed": False,
                        "audit_receipt_sha256": refit_receipt.self_sha256,
                    },
                    "qualification": {
                        "status": "qualified",
                        "receipt_sha256": refit_receipt.self_sha256,
                        "finite": True,
                        "unit_norm": True,
                        "qualified_rank": item.qualified_rank,
                    },
                }
            )
    entries.sort(key=lambda item: item["direction_id"])
    raw: dict[str, Any] = {
        "format": "truth_editing_direction_bank_manifest_v1",
        "manifest_id": manifest_id,
        "model": json.loads(canonical_json_bytes(asdict(base_manifest.model))),
        "directions": entries,
    }
    raw["self_sha256"] = canonical_sha256(raw)
    try:
        promoted = parse_direction_bank_manifest(raw)
        provider = DirectionBank(promoted, Path(root))
        for entry in promoted.directions:
            provider.load_vector(entry.direction_id)
    except (ValueError, OSError) as error:
        raise DirectionBankError(
            f"promoted refit direction bank is invalid: {error}"
        ) from error
    return promoted


__all__ = [
    "CompiledBasis",
    "CompiledBasisSet",
    "CompiledControlBasisReceipt",
    "DirectionBank",
    "DirectionBankBuild",
    "DirectionBankError",
    "DirectionCoverage",
    "SOURCE_INVENTORY_FORMAT",
    "build_direction_bank",
    "build_reconstruction_workload",
    "compile_equal_rank_orthogonal_control",
    "compile_control_basis_receipt",
    "compile_control_basis_set",
    "compile_shuffled_control",
    "promote_reconstructed_direction_bank",
    "parse_control_basis_receipt",
    "vector_sha256",
]
