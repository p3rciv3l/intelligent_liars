#!/usr/bin/env python3
"""Promote one complete direction refit into a qualified production bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_contracts import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
    parse_direction_bank_manifest,
)
from intelligent_liars.truth_editing_direction_refit import (  # noqa: E402
    parse_direction_refit_receipt,
)
from intelligent_liars.truth_editing_directions import (  # noqa: E402
    DirectionBankError,
    promote_reconstructed_direction_bank,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise DirectionBankError(f"{name} must be a regular non-symlink file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DirectionBankError(f"cannot read {name}: {error}") from error


def _verified_plan(path: Path) -> tuple[dict[str, Any], str]:
    value = _load_json(path, "refit plan")
    if not isinstance(value, dict):
        raise DirectionBankError("refit plan must be an object")
    if value.get("format") != "truth_editing_direction_refit_plan_v1":
        raise DirectionBankError("refit plan format is unsupported")
    claimed = value.get("self_sha256")
    unsigned = dict(value)
    unsigned.pop("self_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(unsigned) != claimed:
        raise DirectionBankError("refit plan self hash mismatch")
    return value, claimed


def _coverage_receipt(
    *,
    manifest: Any,
    base_manifest: Any,
    plan: dict[str, Any],
    receipt: Any,
    artifact_root: Path,
    base_manifest_path: Path,
    plan_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for item in manifest.directions:
        by_family[item.family] = by_family.get(item.family, 0) + 1
        layer = str(item.source_layer)
        by_layer[layer] = by_layer.get(layer, 0) + 1
        for domain in item.domains:
            by_domain[domain] = by_domain.get(domain, 0) + 1
    value: dict[str, Any] = {
        "format": "truth_editing_qualified_direction_coverage_v1",
        "manifest_sha256": manifest.self_sha256,
        "base_manifest": {
            "path": str(base_manifest_path),
            "file_sha256": _file_sha256(base_manifest_path),
            "self_sha256": base_manifest.self_sha256,
        },
        "refit_plan": {
            "path": str(plan_path),
            "file_sha256": _file_sha256(plan_path),
            "self_sha256": plan["self_sha256"],
            "config_sha256": plan.get("config_sha256"),
        },
        "refit_receipt": {
            "path": str(receipt_path),
            "file_sha256": _file_sha256(receipt_path),
            "self_sha256": receipt.self_sha256,
        },
        "artifact_root": str(artifact_root),
        "model": asdict(manifest.model),
        "coverage": {
            "total": len(manifest.directions),
            "qualified": sum(
                item.qualification.status == "qualified"
                for item in manifest.directions
            ),
            "by_family": dict(sorted(by_family.items())),
            "by_layer": dict(sorted(by_layer.items(), key=lambda item: int(item[0]))),
            "by_domain": dict(sorted(by_domain.items())),
        },
    }
    value["self_sha256"] = canonical_sha256(value)
    return value


def _precheck_output(path: Path, content: bytes) -> bool:
    """Return True if a write is needed; accept an exact existing identity."""

    if path.is_symlink():
        raise DirectionBankError(f"output must not be a symlink: {path}")
    if not path.exists():
        return True
    if not path.is_file():
        raise DirectionBankError(f"output is not a regular file: {path}")
    if path.read_bytes() == content:
        return False
    raise DirectionBankError(
        f"refusing to overwrite different existing output: {path}; "
        "choose an explicit unused output path"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def promote(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    if not artifact_root.is_dir():
        raise DirectionBankError("artifact root must be an existing directory")
    plan, plan_sha256 = _verified_plan(args.plan)
    base_manifest = parse_direction_bank_manifest(
        _load_json(args.base_manifest, "base direction-bank manifest")
    )
    receipt = parse_direction_refit_receipt(
        _load_json(args.refit_receipt, "direction refit receipt")
    )
    manifest = promote_reconstructed_direction_bank(
        base_manifest,
        receipt,
        expected_plan_sha256=plan_sha256,
        root=artifact_root,
        manifest_id=args.manifest_id,
    )
    coverage = _coverage_receipt(
        manifest=manifest,
        base_manifest=base_manifest,
        plan=plan,
        receipt=receipt,
        artifact_root=artifact_root,
        base_manifest_path=args.base_manifest,
        plan_path=args.plan,
        receipt_path=args.refit_receipt,
    )
    manifest_bytes = canonical_json_bytes(manifest.to_dict()) + b"\n"
    coverage_bytes = canonical_json_bytes(coverage) + b"\n"
    write_manifest = _precheck_output(args.manifest_output, manifest_bytes)
    write_coverage = _precheck_output(args.coverage_output, coverage_bytes)
    if write_manifest:
        _atomic_write(args.manifest_output, manifest_bytes)
    if write_coverage:
        _atomic_write(args.coverage_output, coverage_bytes)
    return {
        "manifest": str(args.manifest_output),
        "manifest_sha256": manifest.self_sha256,
        "coverage": str(args.coverage_output),
        "coverage_sha256": coverage["self_sha256"],
        "direction_count": len(manifest.directions),
        "qualified_count": coverage["coverage"]["qualified"],
        "refit_receipt_sha256": receipt.self_sha256,
        "plan_sha256": plan_sha256,
        "model_sha256": manifest.model.model_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "configs/truth_editing_direction_bank_v1.json",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/directions/refit-v1/plan.json",
    )
    parser.add_argument(
        "--refit-receipt",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts/directions/refit-v1/refit-receipt.json",
    )
    parser.add_argument("--artifact-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs/truth_editing_direction_bank_qualified_v1.json",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs/truth_editing_direction_coverage_qualified_v1.json",
    )
    parser.add_argument(
        "--manifest-id",
        default="qwen3-vl-8b-thinking-qualified-truth-directions-v1",
    )
    return parser


def main() -> int:
    try:
        result = promote(_parser().parse_args())
    except (DirectionBankError, ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
