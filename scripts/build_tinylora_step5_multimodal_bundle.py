#!/usr/bin/env python3
"""Inventory, stage, validate, or rebase the portable Step 5 PixMo bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from intelligent_liars.step5_multimodal_assets import (
    build_asset_manifest,
    create_deterministic_tar,
    rebase_image_references,
    stage_multimodal_bundle,
    validate_staged_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPORA = (
    PROJECT_ROOT / "corpora/tinylora_deception_action_v1/step5_v1/preservation_train.jsonl",
    PROJECT_ROOT
    / "corpora/tinylora_deception_action_v1/step5_v1/preservation_development_vision.jsonl",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="validate sources and write the manifest")
    inventory.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    inventory.add_argument("--corpus", type=Path, action="append", default=[])
    inventory.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build", help="atomically stage a portable bundle")
    build.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    build.add_argument("--corpus", type=Path, action="append", default=[])
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--archive", type=Path)

    validate = subparsers.add_parser("validate", help="strictly validate a staged bundle")
    validate.add_argument("--bundle-root", type=Path, required=True)

    rebase = subparsers.add_parser("rebase", help="write JSONL with verified absolute image paths")
    rebase.add_argument("--bundle-root", type=Path, required=True)
    rebase.add_argument("--input", type=Path, required=True)
    rebase.add_argument("--output", type=Path, required=True)
    return parser


def _corpora(values: Sequence[Path]) -> list[Path]:
    return list(values) if values else list(DEFAULT_CORPORA)


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inventory":
        manifest = build_asset_manifest(_corpora(args.corpus), project_root=args.project_root)
        _write_new(args.output, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps({"manifest": str(args.output), **manifest["totals"]}, sort_keys=True))
        return 0
    if args.command == "build":
        manifest = stage_multimodal_bundle(
            _corpora(args.corpus),
            project_root=args.project_root,
            destination=args.output_dir,
        )
        result: dict[str, Any] = {"bundle": str(args.output_dir), **manifest["totals"]}
        if args.archive is not None:
            result["archive"] = str(args.archive)
            result["archive_sha256"] = create_deterministic_tar(args.output_dir, args.archive)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "validate":
        manifest = validate_staged_bundle(args.bundle_root)
        print(json.dumps({"bundle": str(args.bundle_root), "valid": True, **manifest["totals"]}, sort_keys=True))
        return 0
    manifest = validate_staged_bundle(args.bundle_root)
    rebased = rebase_image_references(
        _read_jsonl(args.input),
        bundle_root=args.bundle_root,
        manifest=manifest,
    )
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rebased
    )
    _write_new(args.output, payload)
    print(json.dumps({"output": str(args.output), "records": len(rebased)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
