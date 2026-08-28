#!/usr/bin/env python3
"""Build and print, or explicitly execute, one bounded Vast prerequisite job."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from intelligent_liars.truth_editing_vast_prerequisites import (
    JobConfig,
    Offer,
    build_bundle,
    execute_lifecycle,
    file_sha256,
    lifecycle_plan,
    sha256,
)


DEFAULT_VASTAI = str(Path(__file__).resolve().with_name("vastai_openai_ua.py"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--offer-json", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fetch-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ssh-identity", type=Path)
    parser.add_argument(
        "--vastai",
        default=DEFAULT_VASTAI,
        help="Vast CLI wrapper that applies the required HTTP user agent",
    )
    parser.add_argument(
        "--maximum-bundle-bytes",
        type=int,
        default=128 * 1024**2,
        help="Explicit uncompressed allowlist cap; defaults to 128 MiB",
    )
    parser.add_argument(
        "--reuse-bundle-sha256",
        help="Reuse an existing immutable bundle only when this SHA-256 and its manifest verify",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmed-cost-approval", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = JobConfig.from_mapping(json.loads(args.config.read_text()))
    offer_raw = json.loads(args.offer_json.read_text())
    if isinstance(offer_raw, list):
        if len(offer_raw) != 1:
            raise SystemExit("offer JSON must identify exactly one offer")
        offer_raw = offer_raw[0]
    offer = Offer.from_mapping(offer_raw)
    maximum_bundle_bytes = getattr(args, "maximum_bundle_bytes", 128 * 1024**2)
    if maximum_bundle_bytes <= 0:
        raise SystemExit("--maximum-bundle-bytes must be positive")
    expected_reuse_sha = getattr(args, "reuse_bundle_sha256", None)
    if expected_reuse_sha is None:
        bundle = build_bundle(
            args.repo,
            config,
            args.bundle,
            maximum_bytes=maximum_bundle_bytes,
        )
    else:
        if file_sha256(args.bundle) != expected_reuse_sha:
            raise SystemExit("existing bundle SHA-256 differs from --reuse-bundle-sha256")
        with tarfile.open(args.bundle, mode="r:gz") as archive:
            members = archive.getmembers()
            if any(not member.isfile() for member in members):
                raise SystemExit("existing bundle contains a non-regular member")
            manifest_members = [member for member in members if member.name == "bundle-manifest.json"]
            if len(manifest_members) != 1:
                raise SystemExit("existing bundle has no unique manifest")
            stream = archive.extractfile(manifest_members[0])
            if stream is None:
                raise SystemExit("existing bundle manifest is unreadable")
            manifest = json.loads(stream.read())
        unsigned = {"format": manifest.get("format"), "files": manifest.get("files")}
        if manifest.get("self_sha256") != sha256(unsigned):
            raise SystemExit("existing bundle manifest identity differs")
        rows = manifest.get("files")
        if not isinstance(rows, list) or [row.get("path") for row in rows] != sorted(config.bundle_paths):
            raise SystemExit("existing bundle allowlist differs from job config")
        for row in rows:
            path = args.repo / row["path"]
            if not path.is_file() or path.stat().st_size != row.get("bytes") or file_sha256(path) != row.get("sha256"):
                raise SystemExit(f"existing bundle source differs: {row.get('path')}")
        bundle = {**manifest, "archive_sha256": expected_reuse_sha}
    plan = lifecycle_plan(
        vastai=args.vastai,
        config=config,
        offer=offer,
        bundle=args.bundle,
        fetch_dir=args.fetch_dir,
        ssh_identity=args.ssh_identity,
    )
    output = {"bundle": bundle, "lifecycle": plan, "mode": "dry_run"}
    if args.execute:
        if not args.confirmed_cost_approval:
            raise SystemExit("--execute requires --confirmed-cost-approval")
        output["receipt"] = execute_lifecycle(
            plan=plan, config=config, metadata_path=args.metadata
        )
        output["mode"] = "executed"
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
