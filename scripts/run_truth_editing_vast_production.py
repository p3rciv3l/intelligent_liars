#!/usr/bin/env python3
"""Build, inspect, or explicitly execute one bounded truth-editing Vast phase."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from intelligent_liars.truth_editing_vast_prerequisites import Offer
from intelligent_liars.truth_editing_vast_prerequisites import (
    EphemeralWorkloadSecret,
    resolve_aws_cli_profile_credentials,
    resolve_aws_workload_credentials,
)
from intelligent_liars.truth_editing_vast_production import (
    ProductionVastConfig,
    build_production_bundle,
    execute_production_lifecycle,
    production_lifecycle_plan,
)


DEFAULT_VASTAI = str(Path(__file__).resolve().with_name("vastai_openai_ua.py"))


def main() -> int:
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmed-cost-approval", action="store_true")
    parser.add_argument("--persistent-eight-gpu-host", action="store_true")
    parser.add_argument(
        "--aws-profile",
        help="Explicit named profile from the local shared credentials file.",
    )
    parser.add_argument(
        "--aws-credentials-file",
        type=Path,
        help="Private shared credentials file; requires --aws-profile.",
    )
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument(
        "--aws-minimum-validity-seconds",
        type=int,
        help=(
            "Explicit lower bound for a refreshable AWS session. The default still "
            "requires credentials for the entire adaptive lease."
        ),
    )
    args = parser.parse_args()

    config = ProductionVastConfig.from_mapping(json.loads(args.config.read_text()))
    offer_raw = json.loads(args.offer_json.read_text())
    if isinstance(offer_raw, list):
        if len(offer_raw) != 1:
            raise SystemExit("offer JSON must identify exactly one ephemeral candidate")
        offer_raw = offer_raw[0]
    offer = (
        Offer.from_multi_gpu_mapping(offer_raw)
        if args.persistent_eight_gpu_host
        else Offer.from_mapping(offer_raw)
    )
    bundle = build_production_bundle(args.repo, config, args.bundle)
    plan = production_lifecycle_plan(
        vastai=args.vastai,
        config=config,
        offer=offer,
        bundle=args.bundle,
        fetch_dir=args.fetch_dir,
        ssh_identity=args.ssh_identity,
    )
    result: dict[str, object] = {"mode": "dry_run", "bundle": bundle, "lifecycle": plan}
    if args.execute:
        if not args.confirmed_cost_approval:
            raise SystemExit("--execute requires --confirmed-cost-approval")
        aws_credentials = None
        if config.phase == "adaptive":
            full_lease_validity = config.base_job.maximum_elapsed_seconds + 300
            required_validity = (
                full_lease_validity
                if args.aws_minimum_validity_seconds is None
                else args.aws_minimum_validity_seconds
            )
            if not 600 <= required_validity <= full_lease_validity:
                raise SystemExit(
                    "--aws-minimum-validity-seconds must be between 600 seconds "
                    "and the adaptive lease plus cleanup window"
                )
            if args.aws_profile is not None and args.aws_credentials_file is None:
                aws_credentials = resolve_aws_cli_profile_credentials(
                    profile_name=args.aws_profile,
                    minimum_validity_seconds=required_validity,
                    aws_cli=args.aws_cli,
                )
            else:
                aws_credentials = resolve_aws_workload_credentials(
                    environment=os.environ,
                    profile_name=args.aws_profile,
                    credentials_file=args.aws_credentials_file,
                    minimum_validity_seconds=required_validity,
                )
        elif args.aws_profile is not None or args.aws_credentials_file is not None:
            raise SystemExit(
                "AWS credential selection is only valid for adaptive execution"
            )
        secret = EphemeralWorkloadSecret.production(
            openrouter_value=os.environ.get("OPENROUTER_API_KEY", ""),
            wandb_value=os.environ.get("WANDB_API_KEY"),
            aws_credentials=aws_credentials,
        )
        secret_lines = secret.stdin_payload().splitlines()
        if config.phase == "timed_canary" and (
            len(secret_lines) != 2 or not secret_lines[1]
        ):
            raise SystemExit(
                "timed-canary execution requires a valid WANDB_API_KEY through SSH stdin"
            )
        result["receipt"] = execute_production_lifecycle(
            plan=plan,
            config=config,
            metadata_path=args.metadata,
            workload_secret=secret,
            minimum_aws_validity_seconds=(
                required_validity if config.phase == "adaptive" else None
            ),
        )
        result["mode"] = "executed"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
