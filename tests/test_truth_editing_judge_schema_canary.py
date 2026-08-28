from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_judge_schema_canary import (
    CANARY_ID,
    CANARY_MAXIMUM_SPEND_USD,
    RECHECK_ID,
    RECHECK_MAXIMUM_SPEND_USD,
    REQUIRED_SCHEMA_FAILURE_DETAILS,
    build_judge_schema_recheck,
    build_judge_schema_canary,
)
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    LiveJudgeError,
    OperationalJudgeFailure,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


ROOT = Path(__file__).parents[1]
LIVE_ROOT = ROOT / "artifacts/truth-editing/judge-calibration/live-revised-policy-v1-json-object"
CANARY_ROOT = ROOT / "artifacts/truth-editing/judge-calibration/schema-repair-canary-v1"


def _build() -> tuple[dict[str, object], dict[str, object]]:
    return build_judge_schema_canary(
        plan_path=LIVE_ROOT / "plan.json",
        live_report_path=LIVE_ROOT / "live-report.json",
        calibration_report_path=LIVE_ROOT / "calibration-report.json",
        cache_dir=LIVE_ROOT / "cache",
        attempt_dir=LIVE_ROOT / "attempts",
    )


def _build_recheck() -> tuple[dict[str, object], dict[str, object]]:
    return build_judge_schema_recheck(
        canary_plan_path=CANARY_ROOT / "plan.json",
        canary_mapping_path=CANARY_ROOT / "mapping-receipt.json",
    )


def test_one_case_recheck_refreshes_failed_cluster_without_changing_v1() -> None:
    v1_plan_bytes = (CANARY_ROOT / "plan.json").read_bytes()
    v1_mapping_bytes = (CANARY_ROOT / "mapping-receipt.json").read_bytes()
    plan, mapping = _build_recheck()

    assert plan["calibration_id"] == RECHECK_ID
    assert plan["maximum_spend_usd"] == RECHECK_MAXIMUM_SPEND_USD == 0.005
    assert plan["judge_config_sha256"] == FROZEN_JUDGE_CONFIG_SHA256
    assert plan["pairwise_relationships"] == []
    assert len(plan["absolute_bundles"]) == 1
    assert mapping["source_cluster_id"] == "schema_failure_01"
    assert mapping["source_canary_plan_sha256"] == json.loads(
        v1_plan_bytes
    )["content_sha256"]
    assert mapping["source_canary_mapping_sha256"] == json.loads(
        v1_mapping_bytes
    )["content_sha256"]
    assert mapping["source_bundle_id"] != mapping["recheck_bundle_id"]
    assert mapping["source_response_ids"] != mapping["recheck_response_ids"]
    assert mapping["source_prompt_bundle_sha256"] != mapping[
        "recheck_prompt_bundle_sha256"
    ]
    assert mapping["source_raw_request_sha256"] != mapping[
        "recheck_raw_request_sha256"
    ]
    assert (CANARY_ROOT / "plan.json").read_bytes() == v1_plan_bytes
    assert (CANARY_ROOT / "mapping-receipt.json").read_bytes() == v1_mapping_bytes


def test_one_case_recheck_dry_runs_without_network_or_files(tmp_path: Path) -> None:
    plan, _ = _build_recheck()
    report = run_live_judge_calibration(
        plan,
        cache_dir=tmp_path / "fresh-cache",
        attempt_dir=tmp_path / "fresh-attempts",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["status"] == "dry_run"
    assert report["planned_paid_calls"] == 1
    assert report["maximum_spend_usd"] == 0.005
    assert report["actual_paid_calls"] == 0
    assert report["actual_spend_usd"] == 0.0
    assert not (tmp_path / "fresh-cache").exists()
    assert not (tmp_path / "fresh-attempts").exists()


def test_builder_selects_one_fresh_blinded_case_per_observed_failure() -> None:
    plan, receipt = _build()

    assert plan["maximum_spend_usd"] == CANARY_MAXIMUM_SPEND_USD == 0.02
    assert plan["judge_config_sha256"] == FROZEN_JUDGE_CONFIG_SHA256
    assert plan["pairwise_relationships"] == []
    assert len(plan["absolute_bundles"]) == 8
    assert {row["failure_detail"] for row in receipt["mappings"]} == set(
        REQUIRED_SCHEMA_FAILURE_DETAILS
    )
    assert all(row["source_operation_id"].startswith("hc_bundle_") for row in receipt["mappings"])
    assert all(row["canary_bundle_id"].startswith("judge_schema_canary_v1_") for row in receipt["mappings"])
    assert all(
        old != new
        for row in receipt["mappings"]
        for old, new in zip(row["source_response_ids"], row["canary_response_ids"], strict=True)
    )
    rendered = json.dumps(plan)
    assert "failure_detail" not in rendered
    assert "source_operation_id" not in rendered
    assert "expected_failure" not in rendered


def test_canary_is_runner_compatible_at_two_cent_cap_without_live_calls(tmp_path: Path) -> None:
    plan, _ = _build()
    report = run_live_judge_calibration(
        plan,
        cache_dir=tmp_path / "fresh-cache",
        attempt_dir=tmp_path / "fresh-attempts",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["status"] == "dry_run"
    assert report["planned_paid_calls"] == 8
    assert report["maximum_spend_usd"] == 0.02
    assert not (tmp_path / "fresh-cache").exists()
    assert not (tmp_path / "fresh-attempts").exists()


def test_canary_reaches_offline_transport_through_paid_runner_boundary(tmp_path: Path) -> None:
    class RejectingOfflineTransport:
        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def complete(self, request: dict[str, object]) -> dict[str, object]:
            self.requests.append(request)
            raise LiveJudgeError("offline canary transport rejection")

    plan, _ = _build()
    transport = RejectingOfflineTransport()
    with pytest.raises(OperationalJudgeFailure, match="provider_rejected_request"):
        run_live_judge_calibration(
            plan,
            cache_dir=tmp_path / "fresh-cache",
            attempt_dir=tmp_path / "fresh-attempts",
            transport=transport,
        )
    assert len(transport.requests) == 1
    provider = transport.requests[0]["provider"]
    assert provider["order"] == provider["only"] == ["z-ai/fp8"]
    assert provider["quantizations"] == ["fp8"]
    assert provider["allow_fallbacks"] is False
    assert len(list((tmp_path / "fresh-attempts").glob("*/*/failed.json"))) == 1


def test_canary_has_new_request_identities_and_rejects_tampered_sources(tmp_path: Path) -> None:
    plan, receipt = _build()
    old_requests = {row["source_raw_request_sha256"] for row in receipt["mappings"]}
    new_requests = {row["canary_raw_request_sha256"] for row in receipt["mappings"]}
    old_prompts = {row["source_prompt_bundle_sha256"] for row in receipt["mappings"]}
    new_prompts = {row["canary_prompt_bundle_sha256"] for row in receipt["mappings"]}
    assert len(new_requests) == 8
    assert old_requests.isdisjoint(new_requests)
    assert len(new_prompts) == 8
    assert old_prompts.isdisjoint(new_prompts)

    bad_report = tmp_path / "calibration-report.json"
    payload = json.loads((LIVE_ROOT / "calibration-report.json").read_text())
    payload["operational"]["failure_details"][REQUIRED_SCHEMA_FAILURE_DETAILS[0]] += 1
    bad_report.write_text(json.dumps(payload))
    with pytest.raises(LiveJudgeError, match="identity"):
        build_judge_schema_canary(
            plan_path=LIVE_ROOT / "plan.json",
            live_report_path=LIVE_ROOT / "live-report.json",
            calibration_report_path=bad_report,
            cache_dir=LIVE_ROOT / "cache",
            attempt_dir=LIVE_ROOT / "attempts",
        )


def test_builder_cli_writes_plan_and_mapping_once(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "mapping.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/build_truth_editing_judge_schema_canary.py"),
        "--source-plan",
        str(LIVE_ROOT / "plan.json"),
        "--source-live-report",
        str(LIVE_ROOT / "live-report.json"),
        "--source-calibration-report",
        str(LIVE_ROOT / "calibration-report.json"),
        "--source-cache-dir",
        str(LIVE_ROOT / "cache"),
        "--source-attempt-dir",
        str(LIVE_ROOT / "attempts"),
        "--output-plan",
        str(plan_path),
        "--output-mapping",
        str(receipt_path),
    ]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert json.loads(plan_path.read_text())["calibration_id"] == CANARY_ID
    assert json.loads(receipt_path.read_text())["canary_plan_sha256"] == json.loads(
        plan_path.read_text()
    )["content_sha256"]
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert second.returncode != 0

    same_output = command[:-4] + [
        "--output-plan",
        str(tmp_path / "same.json"),
        "--output-mapping",
        str(tmp_path / "same.json"),
    ]
    same = subprocess.run(
        same_output, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert same.returncode != 0
    assert not (tmp_path / "same.json").exists()
