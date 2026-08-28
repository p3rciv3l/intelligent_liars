#!/usr/bin/env python3
"""Plan, explicitly download, or verify the pinned narrow Qwen model cache.

This tool never uploads to S3.  ``download`` requires an explicit execution flag;
``plan`` fetches metadata only and therefore does not download weight blobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Make the repository script usable before the editable package is installed on a
# fresh host.  This path is derived from the script itself, never from cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.model_cache import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    CacheValidationError,
    build_cache_manifest,
    build_snapshot_plan,
    canonical_json_bytes,
    completion_marker,
    legal_artifact_descriptors,
    materialize_huggingface_cache,
    verify_snapshot,
)


HUB_API_BASE = "https://huggingface.co/api/models"
MIN_FREE_SPACE_MARGIN_BYTES = 2 * 1024**3
DOWNLOAD_USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CacheValidationError(f"expected a JSON object: {path}")
    return value


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def materialize_legal_bundle(output_dir: Path) -> dict[str, Any]:
    """Write pinned legal and attribution objects outside the loader inventory."""

    written: list[dict[str, Any]] = []
    for descriptor in legal_artifact_descriptors():
        if "content_utf8" in descriptor:
            content = str(descriptor["content_utf8"]).encode()
        else:
            request = urllib.request.Request(
                str(descriptor["source_url"]),
                headers={"User-Agent": DOWNLOAD_USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                content = response.read()
        if len(content) != descriptor["bytes"]:
            raise CacheValidationError(
                f"legal artifact size mismatch: {descriptor['path']}"
            )
        if hashlib.sha256(content).hexdigest() != descriptor["sha256"]:
            raise CacheValidationError(
                f"legal artifact SHA-256 mismatch: {descriptor['path']}"
            )
        target = output_dir / str(descriptor["path"])
        _write_bytes(target, content)
        written.append(
            {
                "path": descriptor["path"],
                "bytes": descriptor["bytes"],
                "sha256": descriptor["sha256"],
            }
        )
    return {"format": "tinylora_model_legal_bundle_v1", "files": written}


def fetch_hub_model_info() -> dict[str, Any]:
    """Fetch file metadata for the exact commit without fetching file contents."""

    repo = urllib.parse.quote(MODEL_ID, safe="/")
    revision = urllib.parse.quote(MODEL_REVISION, safe="")
    url = f"{HUB_API_BASE}/{repo}/revision/{revision}?blobs=true"
    request = urllib.request.Request(url, headers={"User-Agent": DOWNLOAD_USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        value = json.load(response)
    if not isinstance(value, dict):
        raise CacheValidationError("Hub model-info response is not an object")
    return value


def _write_verified_outputs(
    *,
    snapshot_dir: Path,
    plan: dict[str, Any],
    manifest_output: Path,
    completion_output: Path | None,
    s3_bucket: str | None,
    s3_prefix: str,
) -> dict[str, Any]:
    verified = verify_snapshot(snapshot_dir, plan)
    manifest = build_cache_manifest(
        plan,
        verified,
        bucket=s3_bucket,
        base_prefix=s3_prefix,
    )
    _write_json(manifest_output, manifest)
    if completion_output is not None:
        if s3_bucket is None:
            raise CacheValidationError(
                "--completion-output requires --s3-bucket so its destination is defined"
            )
        _write_json(completion_output, completion_marker(manifest))
    return manifest


def _require_outputs_outside_snapshot(snapshot_dir: Path, outputs: list[Path]) -> None:
    root = snapshot_dir.resolve()
    for output in outputs:
        resolved = output.resolve()
        if resolved == root or root in resolved.parents:
            raise CacheValidationError(
                f"cache metadata must live outside the exact snapshot inventory: {output}"
            )


def _download_snapshot(snapshot_dir: Path, plan: dict[str, Any]) -> None:
    expected = int(plan["expected_download_bytes"])
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(snapshot_dir.parent).free
    required = expected + MIN_FREE_SPACE_MARGIN_BYTES
    if free < required:
        raise CacheValidationError(
            f"insufficient free disk: need at least {required} bytes, found {free}"
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=list(plan["allow_patterns"]),
        local_dir=snapshot_dir,
        user_agent=DOWNLOAD_USER_AGENT,
    )


def _add_verification_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--completion-output", type=Path)
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix", default="model-cache/v1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="fetch metadata only; no weight download")
    plan.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify an existing narrow snapshot")
    verify.add_argument("--plan", type=Path, required=True)
    _add_verification_arguments(verify)

    download = subparsers.add_parser(
        "download", help="explicitly fetch the narrow pinned snapshot, then verify it"
    )
    download.add_argument("--execute-download", action="store_true")
    download.add_argument("--plan-output", type=Path, required=True)
    _add_verification_arguments(download)

    hydrate = subparsers.add_parser(
        "hydrate-hf-cache",
        help="verify a standalone snapshot and hydrate ModelLoadConfig's cache layout",
    )
    hydrate.add_argument("--plan", type=Path, required=True)
    hydrate.add_argument("--snapshot-dir", type=Path, required=True)
    hydrate.add_argument("--cache-dir", type=Path, required=True)
    hydrate.add_argument("--report-output", type=Path, required=True)

    legal = subparsers.add_parser(
        "legal", help="materialize pinned legal/attribution objects beside the cache"
    )
    legal.add_argument("--output-dir", type=Path, required=True)
    legal.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            plan = build_snapshot_plan(fetch_hub_model_info())
            _write_json(args.output, plan)
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if args.command == "hydrate-hf-cache":
            _require_outputs_outside_snapshot(
                args.snapshot_dir, [args.report_output]
            )
            report = materialize_huggingface_cache(
                args.snapshot_dir, args.cache_dir, _read_json(args.plan)
            )
            _write_json(args.report_output, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "legal":
            _require_outputs_outside_snapshot(args.output_dir, [args.report_output])
            report = materialize_legal_bundle(args.output_dir)
            _write_json(args.report_output, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        metadata_outputs = [args.manifest_output]
        if args.completion_output is not None:
            metadata_outputs.append(args.completion_output)
        if args.command == "verify":
            _require_outputs_outside_snapshot(args.snapshot_dir, metadata_outputs)
            plan = _read_json(args.plan)
        else:
            if not args.execute_download:
                raise CacheValidationError(
                    "download is disabled unless --execute-download is supplied"
                )
            metadata_outputs.append(args.plan_output)
            _require_outputs_outside_snapshot(args.snapshot_dir, metadata_outputs)
            plan = build_snapshot_plan(fetch_hub_model_info())
            _write_json(args.plan_output, plan)
            _download_snapshot(args.snapshot_dir, plan)

        manifest = _write_verified_outputs(
            snapshot_dir=args.snapshot_dir,
            plan=plan,
            manifest_output=args.manifest_output,
            completion_output=args.completion_output,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (CacheValidationError, OSError, ValueError) as error:
        print(f"model-cache error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
