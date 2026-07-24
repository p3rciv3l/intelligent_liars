from __future__ import annotations

import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import typer

from intelligent_liars.cli_app import app
from intelligent_liars.osworld_runtime import (
    InfrastructureTarget,
    VastLaunchPolicy,
    offline_preflight,
)
from intelligent_liars.osworld_overlay import (
    materialize_run_manifest,
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


def _object_section(
    config: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
) -> Mapping[str, Any]:
    value = config.get(name) if required else config.get(name, {})
    if not isinstance(value, Mapping):
        raise typer.BadParameter(f"{name} must be an object")
    return value


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
            provider_budgets=config["provider_budgets"],
            evaluation_envelopes=config["evaluation_envelopes"],
            teardown_checklist=config["teardown_checklist"],
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
    target_config = _object_section(config, "target")
    vast_config = _object_section(config, "vast")
    aws_config = _object_section(config, "aws", required=False)
    try:
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
            aws_auth_mode=aws_config.get(
                "auth_mode",
                "controller-iam-role",
            ),
        )
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(report.as_dict(), indent=2, sort_keys=True))


@app.command("osworld-materialize")
def osworld_materialize(
    template_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
    project_root: Path = typer.Option(Path("."), "--project-root", file_okay=False),
) -> None:
    """Materialize a frozen manifest from a template and the clean current HEAD."""
    try:
        manifest = materialize_run_manifest(
            template_path=template_path,
            output_path=output_path,
            project_root=project_root,
        )
    except (OSError, ValueError, ManifestValidationError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "cloud_side_effects": False,
                "manifest": str(output_path.resolve()),
                "manifest_sha256": manifest.manifest_hash,
                "run_id": manifest.run_id,
                "repository_commit": manifest.payload["repository"]["commit"],
                "task_count": len(manifest.task_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
