from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "validation"
    / "truth_editing_recovery_20260903"
)


def _read(name: str) -> dict[str, Any]:
    value = json.loads((ROOT / name).read_text())
    assert isinstance(value, dict)
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _assert_self_hash(value: dict[str, Any]) -> None:
    claimed = value["content_sha256"]
    assert _sha({key: item for key, item in value.items() if key != "content_sha256"}) == claimed


def test_judge_soak_receipt_recomputes_from_sanitized_inventory() -> None:
    receipt = _read("judge_soak_receipt.json")
    evidence = _read("judge_soak_sanitized_evidence.json")
    _assert_self_hash(receipt)
    _assert_self_hash(evidence)
    assert evidence["content_sha256"] == receipt["sanitized_evidence_content_sha256"]
    assert evidence["completed_attempt_count"] == receipt["confirmed_completed_provider_calls"]
    assert round(evidence["completed_attempt_spend_usd"], 8) == receipt[
        "confirmed_completed_provider_spend_usd"
    ]
    assert len(evidence["ambiguous_pending_request_sha256s"]) == receipt[
        "ambiguous_crash_left_requests"
    ]
    boundary = evidence["kill_boundary_derivation"]
    assert boundary["completed_attempts_at_or_before_boundary"] == receipt[
        "hard_kill_completed_call_frontier"
    ]
    assert boundary["unresolved_pending_attempts_at_boundary"] == receipt[
        "hard_kill_in_flight_request_count"
    ]
    assert boundary["unresolved_pending_attempts_at_boundary"] == receipt[
        "maximum_concurrency"
    ]
    operations = evidence["operation_inventory"]
    assert len(operations["completed_ids"]) == receipt["completed_operation_count"]
    assert len(operations["failed_ids"]) == receipt["failed_operation_count"]
    assert len(evidence["result_receipt_sha256s"]) == receipt["result_receipt_count"]
    assert len(evidence["failure_receipt_sha256s"]) == receipt["failure_receipt_count"]
    assert len(evidence["correction_lineage_sha256s"]) == receipt[
        "correction_lineage_count"
    ]
    assert evidence["replay_report_byte_identical"] is True
    assert evidence["report_file_sha256"] == evidence["replay_report_file_sha256"]


def test_local_mps_optuna_receipt_accounts_every_trial() -> None:
    receipt = _read("local_mps_optuna_canary_receipt.json")
    _assert_self_hash(receipt)
    allowed = {"successful", "scientifically_infeasible", "operational_failure"}
    assert set(receipt["target_outcomes"]) <= allowed
    assert sum(receipt["target_outcomes"].values()) == receipt["target_completed_trials"] == 8
    assert receipt["source_history_preserved"] is True
    assert receipt["rescore_request_count"] == 1
    assert receipt["ordinary_continuation_trial_count"] == 3
    assert receipt["runtime_device"] == "mps"
    assert receipt["large_run_scientific_equivalence"] is False


def test_live_partial_record_recovery_receipt_is_missing_only() -> None:
    receipt = _read("semantic_record_recovery_receipt.json")
    _assert_self_hash(receipt)
    assert receipt["forced_signal"] == "SIGKILL"
    assert receipt["pre_kill_completed_record_ids"] == ["direct-1"]
    assert receipt["post_restart_judge_invocation_record_ids"] == ["direct-2"]
    assert receipt["duplicate_judge_invocation_record_ids"] == []
    assert len(receipt["provider_request_identities"]) == 2
    assert len(
        {item["cache_key_sha256"] for item in receipt["provider_request_identities"]}
    ) == 2
    assert len(receipt["semantic_completion_file_sha256s"]) == 2


def test_parent_observed_crash_receipt_proves_signal_frontier_and_replay() -> None:
    receipt = _read("parent_observed_crash_recovery_receipt.json")
    _assert_self_hash(receipt)
    event = receipt["kill_event"]
    _assert_self_hash(event)
    assert event["signal"] == "SIGKILL"
    assert event["child_return_code"] == -9
    assert event["max_concurrency"] == 8
    assert len(event["completed_request_sha256s"]) >= 32
    assert len(event["pending_request_sha256s"]) == 8
    assert receipt["pending_ambiguous_after_restart"] == 8
    assert receipt["replay_added_completed_calls"] == 0
    assert receipt["replay_report_byte_identical"] is True
    assert receipt["attempt_inventory_after_restart_sha256"] == receipt[
        "attempt_inventory_after_replay_sha256"
    ]
    assert receipt["terminal_operation_count"] == 100
