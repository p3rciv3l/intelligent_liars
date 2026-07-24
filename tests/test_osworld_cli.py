from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from intelligent_liars.cli_app import app
from intelligent_liars import cli_osworld  # noqa: F401
from intelligent_liars.osworld_overlay import ExecutionBlockedError, load_approved_proposal
from intelligent_liars.run_control import load_run_manifest

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
                    },
                    {
                        "provider": "vast",
                        "resource": "gpu",
                        "instance_count": 1,
                        "ttl_minutes": 180,
                        "estimated_hourly_rate_usd": "0.60",
                    },
                ],
                "provider_budgets": {
                    "aws": {
                        "projected_spend_usd": "8.11",
                        "maximum_spend_usd": "75",
                        "committed_spend_usd": "0",
                        "authorized_active_cost_usd": "0",
                        "stop_new_leases_usd": "65",
                        "hard_stop_usd": "75",
                    },
                    "vast": {
                        "projected_spend_usd": "9.5",
                        "maximum_spend_usd": "20",
                        "committed_spend_usd": "0",
                        "authorized_active_cost_usd": "0",
                        "stop_new_leases_usd": "19",
                        "hard_stop_usd": "20",
                    },
                },
                "evaluation_envelopes": {
                    "baseline_envelope_usd": "69",
                    "intervention_envelope_usd": "70",
                },
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
    assert set(proposal["provider_budgets"]) == {"aws", "vast"}
    assert proposal["provider_budgets"]["aws"]["maximum_spend_usd"] == "75"
    assert proposal["provider_budgets"]["vast"]["maximum_spend_usd"] == "20"

    proposal["approved"] = True
    proposal["approval_id"] = "approval-cli-integration"
    proposal["approved_at"] = "2026-07-22T00:00:00Z"
    approved_path = tmp_path / "approved.json"
    approved_path.write_text(json.dumps(proposal))

    loaded = load_approved_proposal(
        approved_path,
        load_run_manifest(manifest),
    )
    assert loaded.run_id == proposal["run_id"]
    assert set(loaded.provider_budgets) == {"aws", "vast"}

    proposal["resources"][0]["ttl_minutes"] = 181
    approved_path.write_text(json.dumps(proposal))
    with pytest.raises(ExecutionBlockedError, match="content hash mismatch"):
        load_approved_proposal(
            approved_path,
            load_run_manifest(manifest),
        )


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
        "OSWORLD_CLIENT_PASSWORD",
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", []),
        ("target", "aws"),
        ("vast", []),
        ("vast", 1),
        ("aws", []),
        ("aws", "controller-iam-role"),
    ],
)
def test_osworld_preflight_rejects_non_object_sections(tmp_path, field, value):
    payload = {
        "target": {},
        "vast": {
            "offer_id": "offer-reviewed-offline",
            "estimated_hourly_rate_usd": "0.60",
            "price_approved": False,
        },
        "aws": {},
    }
    payload[field] = value
    config = tmp_path / "invalid-preflight.json"
    config.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        app,
        ["osworld-preflight", "--config", str(config)],
    )

    assert result.exit_code == 2
    assert f"{field} must be an object" in result.output


def test_env_example_declares_osworld_client_password():
    env_example = (Path(__file__).parents[1] / ".env.example").read_text()
    assert env_example.splitlines().count("OSWORLD_CLIENT_PASSWORD=") == 1
