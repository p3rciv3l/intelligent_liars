from __future__ import annotations

import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import typer

from intelligent_liars.cli_app import app
from intelligent_liars.osworld_runtime import (
    InfrastructureTarget,
    VastLaunchPolicy,
    offline_preflight,
)
from intelligent_liars.run_control import (
    LaunchResource,
    ManifestValidationError,
    create_launch_proposal,
    load_run_manifest,
)


def _load_proposal_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Cannot load proposal config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("proposal config root must be an object")
    return payload


@app.command("osworld-proposal")
def osworld_proposal(
    manifest_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", dir_okay=False),
) -> None:
    """Validate a frozen run and emit an offline launch proposal."""
    try:
        manifest = load_run_manifest(manifest_path)
        config = _load_proposal_config(config_path)
        resources = [
            LaunchResource(
                provider=item["provider"],
                resource=item["resource"],
                instance_count=item["instance_count"],
                ttl_minutes=item["ttl_minutes"],
                estimated_hourly_rate_usd=Decimal(str(item["estimated_hourly_rate_usd"])),
            )
            for item in config["resources"]
        ]
        proposal = create_launch_proposal(
            manifest=manifest,
            resources=resources,
            maximum_authorized_spend_usd=Decimal(
                str(config["maximum_authorized_spend_usd"])
            ),
            teardown_checklist=config["teardown_checklist"],
            projected_total_usd=Decimal(
                str(
                    config.get(
                        "projected_total_usd",
                        config["maximum_authorized_spend_usd"],
                    )
                )
            ),
        )
    except (KeyError, TypeError, ValueError, InvalidOperation, ManifestValidationError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    rendered = json.dumps(proposal.as_dict(), indent=2, sort_keys=True) + "\n"
    if output is None:
        typer.echo(rendered, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        typer.echo(f"Wrote dry-run proposal to {output}")


@app.command("osworld-preflight")
def osworld_preflight(
    config_path: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    """Report credential names and offline launch-gate status without cloud calls."""
    config = _load_proposal_config(config_path)
    try:
        target_config = config["target"]
        vast_config = config["vast"]
        report = offline_preflight(
            os.environ,
            InfrastructureTarget(
                desktop_provider=target_config.get("desktop_provider", "aws"),
                execution_mode=target_config.get(
                    "execution_mode",
                    "official-osworld-host-client",
                ),
                nested_kvm=target_config.get("nested_kvm", False),
            ),
            VastLaunchPolicy(
                offer_id=vast_config["offer_id"],
                estimated_hourly_rate_usd=Decimal(
                    str(vast_config["estimated_hourly_rate_usd"])
                ),
                price_approved=vast_config["price_approved"],
                gpu_count=vast_config.get("gpu_count", 1),
            ),
            aws_auth_mode=config.get("aws", {}).get(
                "auth_mode",
                "controller-iam-role",
            ),
        )
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))
