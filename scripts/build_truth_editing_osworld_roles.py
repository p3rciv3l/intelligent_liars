#!/usr/bin/env python3
"""Build or verify the frozen OSWorld truth-editing role artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from intelligent_liars.truth_editing_osworld_roles import (
    DEFAULT_SELECTION_SEED,
    build_osworld_role_ledger,
    open_osworld_build_receipt,
    open_osworld_optuna_manifest,
    open_osworld_role_ledger,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path)
    parser.add_argument("--ledger-output-dir", type=Path)
    parser.add_argument("--optimizer-output-dir", type=Path)
    parser.add_argument("--template-git-root", type=Path)
    parser.add_argument("--osworld-checkout", type=Path)
    parser.add_argument("--source-git-ref")
    parser.add_argument("--source-git-commit")
    parser.add_argument("--source-git-blob")
    parser.add_argument("--expected-template-sha256")
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--verify-private", type=Path)
    parser.add_argument("--verify-optimizer", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_private is not None or args.verify_optimizer is not None:
        required_verify = {
            "--verify-private": args.verify_private,
            "--verify-optimizer": args.verify_optimizer,
            "--template": args.template,
            "--osworld-checkout": args.osworld_checkout,
            "--source-git-commit": args.source_git_commit,
            "--source-git-blob": args.source_git_blob,
            "--expected-template-sha256": args.expected_template_sha256,
        }
        missing = [flag for flag, value in required_verify.items() if value is None]
        if missing:
            raise SystemExit(f"missing required verification arguments: {', '.join(missing)}")
        receipt = open_osworld_build_receipt(
            args.verify_private / "build-receipt-v2.json",
            expected_source_git_commit=args.source_git_commit,
            expected_source_git_blob=args.source_git_blob,
            expected_template_sha256=args.expected_template_sha256,
        )
        ledger = open_osworld_role_ledger(
            args.verify_private / "osworld-role-ledger-v2.json",
            verified_receipt=receipt,
            template_path=args.template,
            osworld_checkout=args.osworld_checkout,
        )
        manifest = open_osworld_optuna_manifest(
            args.verify_optimizer / "osworld-optuna-manifest-v2.json",
            verified_ledger=ledger,
            verified_receipt=receipt,
        )
        print(json.dumps({"ledger_id": ledger["ledger_id"], "manifest_id": manifest["manifest_id"]}, sort_keys=True))
        return 0
    required = {
        "--template": args.template,
        "--ledger-output-dir": args.ledger_output_dir,
        "--optimizer-output-dir": args.optimizer_output_dir,
        "--template-git-root": args.template_git_root,
        "--osworld-checkout": args.osworld_checkout,
        "--source-git-ref": args.source_git_ref,
        "--source-git-commit": args.source_git_commit,
        "--source-git-blob": args.source_git_blob,
        "--expected-template-sha256": args.expected_template_sha256,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    receipt = build_osworld_role_ledger(
        template_path=args.template,
        ledger_output_dir=args.ledger_output_dir,
        optimizer_output_dir=args.optimizer_output_dir,
        template_git_root=args.template_git_root,
        osworld_checkout=args.osworld_checkout,
        source_git_ref=args.source_git_ref,
        source_git_commit=args.source_git_commit,
        source_git_blob=args.source_git_blob,
        expected_template_sha256=args.expected_template_sha256,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
