from __future__ import annotations

import json

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from intelligent_liars.cli_app import app
from intelligent_liars import cli_osworld  # noqa: F401

from test_osworld_run_control import manifest_payload


def test_osworld_proposal_cli_emits_dry_run_json(tmp_path):
    manifest = tmp_path / "manifest.json"
    config = tmp_path / "proposal.json"
    manifest.write_text(json.dumps(manifest_payload()))
    config.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "provider": "aws",
                        "resource": "t3.xlarge client",
                        "instance_count": 4,
                        "ttl_minutes": 180,
                        "estimated_hourly_rate_usd": "0.2295",
                    }
                ],
                "maximum_authorized_spend_usd": "8.11",
                "teardown_checklist": ["terminate every client", "verify zero resources"],
            }
        )
    )

    result = CliRunner().invoke(
        app,
        ["osworld-proposal", str(manifest), "--config", str(config)],
    )

    assert result.exit_code == 0, result.output
    assert any(command.name == "osworld-proposal" for command in app.registered_commands)
    proposal = json.loads(result.output)
    assert proposal["dry_run"] is True
    assert proposal["resources"][0]["instance_count"] == 4
    assert proposal["maximum_authorized_spend_usd"] == "8.11"


def test_osworld_preflight_reports_only_env_names_and_status(tmp_path, monkeypatch):
    names = (
        "VAST_API_KEY",
        "VASTAI_PATH",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "QWEN_ENDPOINT_URL",
        "QWEN_ENDPOINT_API_KEY",
        "OSWORLD_ARTIFACT_DESTINATION",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "preflight.json"
    config.write_text(
        json.dumps(
            {
                "target": {
                    "desktop_provider": "aws",
                    "execution_mode": "official-osworld-host-client",
                    "nested_kvm": False,
                },
                "vast": {
                    "offer_id": "offer-reviewed-offline",
                    "estimated_hourly_rate_usd": "0.60",
                    "price_approved": False,
                    "gpu_count": 1,
                },
                "aws": {"auth_mode": "controller-iam-role"},
            }
        )
    )

    result = CliRunner().invoke(
        app,
        ["osworld-preflight", "--config", str(config)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["dry_run"] is True
    assert report["ready"] is False
    assert report["credential_values_included"] is False
    assert {item["name"] for item in report["environment"]} == set(names)
