"""Fail-closed qualification contracts for independent Step 5 probe ensembles.

This module does not fit probes.  It only validates existing probe artifacts,
separates their declared provenance, constructs deterministic negative controls,
and produces content-addressed receipts for downstream jobs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REGISTRY_FORMAT = "intelligent_liars_step5_probe_registry_v1"
MANIFEST_FORMAT = "intelligent_liars_step5_probe_qualification_v1"
ENSEMBLES = ("regularizer", "evaluator")
_CANONICAL_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]*\Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _required_identifier_list(value: Any, *, field: str, probe_id: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Probe {probe_id!r} must declare nonempty {field}")
    identifiers = [
        _required_string(item, field=f"Probe {probe_id!r} {field} item")
        for item in value
    ]
    invalid = [
        identifier
        for identifier in identifiers
        if _CANONICAL_IDENTIFIER.fullmatch(identifier) is None
    ]
    if invalid:
        raise ValueError(
            f"Probe {probe_id!r} {field} must use canonical identifier syntax: {invalid}"
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Probe {probe_id!r} contains duplicate {field}")
    return sorted(identifiers)


def _required_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _direction_at_path(
    payload: Any, path: Sequence[str], *, probe_id: str
) -> list[float]:
    value = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(
                f"Probe {probe_id!r} artifact is missing direction path component {part!r}"
            )
        value = value[part]
    if not isinstance(value, list) or not value:
        raise ValueError(f"Probe {probe_id!r} must contain a direction vector list")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"Probe {probe_id!r} direction vector must be numeric")
    direction = [float(item) for item in value]
    norm = math.sqrt(math.fsum(item * item for item in direction))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(
            f"Probe {probe_id!r} must contain a nonzero finite direction vector"
        )
    return direction


def _orthogonal_controls(
    direction: Sequence[float], *, probe_id: str, count: int
) -> list[dict[str, Any]]:
    dimension = len(direction)
    if dimension < 2:
        raise ValueError(
            "Orthogonal controls require directions with at least two dimensions"
        )
    if count > dimension - 1:
        raise ValueError(
            f"Requested {count} orthogonal controls but direction dimension is {dimension}"
        )
    direction_norm_squared = math.fsum(item * item for item in direction)
    start = int(hashlib.sha256(probe_id.encode("utf-8")).hexdigest(), 16) % dimension
    candidates = [(start + offset) % dimension for offset in range(dimension)]
    basis: list[list[float]] = []
    for index in candidates:
        vector = [0.0] * dimension
        vector[index] = 1.0
        along_direction = (
            math.fsum(vector[i] * direction[i] for i in range(dimension))
            / direction_norm_squared
        )
        vector = [vector[i] - along_direction * direction[i] for i in range(dimension)]
        for previous in basis:
            projection = math.fsum(vector[i] * previous[i] for i in range(dimension))
            vector = [vector[i] - projection * previous[i] for i in range(dimension)]
        norm = math.sqrt(math.fsum(item * item for item in vector))
        if norm <= 1e-12:
            continue
        vector = [item / norm for item in vector]
        # Remove the final floating-point residue along the probe direction.
        residue = (
            math.fsum(vector[i] * direction[i] for i in range(dimension))
            / direction_norm_squared
        )
        vector = [vector[i] - residue * direction[i] for i in range(dimension)]
        norm = math.sqrt(math.fsum(item * item for item in vector))
        vector = [item / norm for item in vector]
        basis.append(vector)
        if len(basis) == count:
            break
    if len(basis) != count:
        raise ValueError(
            f"Could not construct {count} orthogonal controls for {probe_id!r}"
        )
    return [
        {
            "control_id": f"{probe_id}.orthogonal.{index}",
            "kind": "orthogonal",
            "construction": "deterministic_canonical_basis_gram_schmidt_v1",
            "vector": vector,
            "vector_sha256": _sha256_json(vector),
        }
        for index, vector in enumerate(basis)
    ]


def _compile_probe(
    raw: Any,
    *,
    artifact_root: Path,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Every probe registry entry must be an object")
    probe_id = _required_string(raw.get("probe_id"), field="probe_id")
    ensemble = _required_string(
        raw.get("ensemble"), field=f"Probe {probe_id!r} ensemble"
    )
    if ensemble not in ENSEMBLES:
        raise ValueError(f"Probe {probe_id!r} ensemble must be one of {ENSEMBLES}")
    for field in ("layer", "token_pooling", "direction_sign_convention"):
        if raw.get(field) != qualification[field]:
            raise ValueError(
                f"Probe {probe_id!r} {field} does not match qualification {field}"
            )
    artifact_value = _required_string(
        raw.get("artifact_path"), field=f"Probe {probe_id!r} artifact_path"
    )
    artifact_path = Path(artifact_value)
    resolved_path = (
        artifact_path if artifact_path.is_absolute() else artifact_root / artifact_path
    ).resolve()
    if not resolved_path.is_file():
        raise ValueError(f"Probe {probe_id!r} artifact does not exist: {resolved_path}")
    direction_path = raw.get("artifact_direction_path")
    if not isinstance(direction_path, list) or not direction_path:
        raise ValueError(
            f"Probe {probe_id!r} artifact_direction_path must be a nonempty list"
        )
    direction_path = [
        _required_string(item, field=f"Probe {probe_id!r} direction path item")
        for item in direction_path
    ]
    try:
        artifact_bytes = resolved_path.read_bytes()
        artifact_payload = json.loads(artifact_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Probe {probe_id!r} artifact is not readable JSON: {exc}"
        ) from exc
    direction = _direction_at_path(artifact_payload, direction_path, probe_id=probe_id)
    sign_flip = [-item for item in direction]
    controls = [
        {
            "control_id": f"{probe_id}.sign_flip",
            "kind": "sign_flip",
            "construction": "exact_vector_negation_v1",
            "vector": sign_flip,
            "vector_sha256": _sha256_json(sign_flip),
        },
        *_orthogonal_controls(
            direction,
            probe_id=probe_id,
            count=qualification["orthogonal_controls_per_probe"],
        ),
    ]
    return {
        "probe_id": probe_id,
        "ensemble": ensemble,
        "artifact_path": artifact_value,
        "artifact_sha256": _sha256_bytes(artifact_bytes),
        "artifact_direction_path": direction_path,
        "direction_dimension": len(direction),
        "direction_sha256": _sha256_json(direction),
        "layer": qualification["layer"],
        "token_pooling": qualification["token_pooling"],
        "direction_sign_convention": qualification["direction_sign_convention"],
        "source_group_ids": _required_identifier_list(
            raw.get("source_group_ids"), field="source_group_ids", probe_id=probe_id
        ),
        "example_ids": _required_identifier_list(
            raw.get("example_ids"), field="example_ids", probe_id=probe_id
        ),
        "controls": controls,
    }


def _split_receipt(
    name: str,
    probes: Sequence[Mapping[str, Any]],
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "ensemble": name,
        "layer": qualification["layer"],
        "token_pooling": qualification["token_pooling"],
        "direction_sign_convention": qualification["direction_sign_convention"],
        "probe_ids": [probe["probe_id"] for probe in probes],
        "artifact_sha256s": [probe["artifact_sha256"] for probe in probes],
        "direction_sha256s": [probe["direction_sha256"] for probe in probes],
        "source_group_ids": sorted(
            {identifier for probe in probes for identifier in probe["source_group_ids"]}
        ),
        "example_ids": sorted(
            {identifier for probe in probes for identifier in probe["example_ids"]}
        ),
        "controls": [
            {
                "probe_id": probe["probe_id"],
                "controls": [
                    {
                        "control_id": control["control_id"],
                        "kind": control["kind"],
                        "vector_sha256": control["vector_sha256"],
                    }
                    for control in probe["controls"]
                ],
            }
            for probe in probes
        ],
    }
    return {**body, "receipt_sha256": _sha256_json(body)}


def compile_probe_qualification(
    registry: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    """Compile an immutable, leakage-checked manifest from existing probe artifacts."""
    if not isinstance(registry, Mapping) or registry.get("format") != REGISTRY_FORMAT:
        raise ValueError(f"Registry format must be {REGISTRY_FORMAT!r}")
    raw_qualification = registry.get("qualification")
    if not isinstance(raw_qualification, Mapping):
        raise ValueError("Registry qualification must be an object")
    qualification = {
        "layer": _required_int(
            raw_qualification.get("layer"), field="qualification layer"
        ),
        "token_pooling": _required_string(
            raw_qualification.get("token_pooling"), field="qualification token_pooling"
        ),
        "direction_sign_convention": _required_string(
            raw_qualification.get("direction_sign_convention"),
            field="qualification direction_sign_convention",
        ),
        "orthogonal_controls_per_probe": _required_int(
            raw_qualification.get("orthogonal_controls_per_probe"),
            field="qualification orthogonal_controls_per_probe",
            minimum=1,
        ),
        "split_unit": "source_group_id",
        "fitting_performed": False,
    }
    raw_probes = registry.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError("Registry probes must be a nonempty list")
    probes = [
        _compile_probe(
            raw,
            artifact_root=Path(artifact_root).resolve(),
            qualification=qualification,
        )
        for raw in raw_probes
    ]
    probe_ids = [probe["probe_id"] for probe in probes]
    if len(set(probe_ids)) != len(probe_ids):
        raise ValueError("Probe IDs must be unique")
    artifact_hashes = [probe["artifact_sha256"] for probe in probes]
    if len(set(artifact_hashes)) != len(artifact_hashes):
        raise ValueError("Probe registry contains duplicate artifact content")
    direction_hashes = [probe["direction_sha256"] for probe in probes]
    if len(set(direction_hashes)) != len(direction_hashes):
        raise ValueError("Probe registry contains duplicate direction vectors")
    dimensions = {probe["direction_dimension"] for probe in probes}
    if len(dimensions) != 1:
        raise ValueError("All qualified probe directions must have the same dimension")

    ensembles = {
        name: sorted(
            (probe for probe in probes if probe["ensemble"] == name),
            key=lambda probe: probe["probe_id"],
        )
        for name in ENSEMBLES
    }
    for name, members in ensembles.items():
        if not members:
            raise ValueError(f"Qualified {name} ensemble must be nonempty")
    for field in ("source_group_ids", "example_ids"):
        regularizer_ids = {
            identifier
            for probe in ensembles["regularizer"]
            for identifier in probe[field]
        }
        evaluator_ids = {
            identifier
            for probe in ensembles["evaluator"]
            for identifier in probe[field]
        }
        overlap = sorted(regularizer_ids & evaluator_ids)
        if overlap:
            raise ValueError(f"cross-ensemble {field} overlap: {overlap}")

    split_receipts = {
        name: _split_receipt(name, ensembles[name], qualification) for name in ENSEMBLES
    }
    body = {
        "format": MANIFEST_FORMAT,
        "status": "qualified",
        "qualification": qualification,
        "ensembles": ensembles,
        "split_receipts": split_receipts,
    }
    return {**body, "qualification_receipt_sha256": _sha256_json(body)}


def validate_probe_qualification(
    manifest: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    """Revalidate manifest receipts and current artifact bytes without refitting."""
    issues: list[str] = []
    if not isinstance(manifest, Mapping):
        return {
            "format": "intelligent_liars_step5_probe_qualification_validation_v1",
            "valid": False,
            "issues": ["manifest root must be an object"],
            "qualification_receipt_sha256": None,
        }
    receipt = manifest.get("qualification_receipt_sha256")
    try:
        qualification = manifest["qualification"]
        ensembles = manifest["ensembles"]
        registry = {
            "format": REGISTRY_FORMAT,
            "qualification": {
                "layer": qualification["layer"],
                "token_pooling": qualification["token_pooling"],
                "direction_sign_convention": qualification["direction_sign_convention"],
                "orthogonal_controls_per_probe": qualification[
                    "orthogonal_controls_per_probe"
                ],
            },
            "probes": [
                {
                    key: probe[key]
                    for key in (
                        "probe_id",
                        "ensemble",
                        "artifact_path",
                        "artifact_direction_path",
                        "source_group_ids",
                        "example_ids",
                        "layer",
                        "token_pooling",
                        "direction_sign_convention",
                    )
                }
                for name in ENSEMBLES
                for probe in ensembles[name]
            ],
        }
        expected = compile_probe_qualification(registry, artifact_root=artifact_root)
        if manifest != expected:
            issues.append(
                "manifest does not match its artifacts and qualification contract"
            )
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"manifest qualification failed: {exc}")
    return {
        "format": "intelligent_liars_step5_probe_qualification_validation_v1",
        "valid": not issues,
        "issues": issues,
        "qualification_receipt_sha256": receipt,
    }


def write_probe_qualification(
    registry: Mapping[str, Any], *, artifact_root: Path, output_path: Path
) -> dict[str, Any]:
    """Compile and atomically publish a new qualification manifest."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Probe qualification already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry_copy = json.loads(json.dumps(registry, allow_nan=False))
    for probe in registry_copy.get("probes", []):
        raw_path = Path(str(probe.get("artifact_path", "")))
        resolved_path = (
            raw_path if raw_path.is_absolute() else Path(artifact_root) / raw_path
        ).resolve()
        probe["artifact_path"] = os.path.relpath(resolved_path, output_path.parent)
    manifest = compile_probe_qualification(
        registry_copy, artifact_root=output_path.parent
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Probe qualification already exists: {output_path}"
            ) from exc
        Path(temporary_name).unlink()
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return manifest
