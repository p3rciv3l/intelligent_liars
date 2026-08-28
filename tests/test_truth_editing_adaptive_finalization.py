from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_adaptive_finalization import (
    AdaptiveFinalizationError,
    run_adaptive_finalization,
    write_adaptive_finalization_handoff,
)
from intelligent_liars.truth_editing_finalist_checkpoint import (
    export_finalist_checkpoint,
)
from test_truth_editing_finalist_checkpoint import (
    _production_export_fixture,
    _report,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _inputs(tmp_path: Path, *, phase: str = "finalization_reserved"):
    selection, compiler, bundle = _production_export_fixture(tmp_path / "fixture")
    report_payload = _report([("trial-a", (0.9, 0.8, 0.95))])
    report_payload["trials"][0]["proposal"] = selection["finalists"][0]["proposal"]
    report = tmp_path / "study-report.json"
    report.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n")
    receipt_unsigned = {
        "format": "truth_editing_study_artifact_receipt_v1",
        "study_identity_sha256": report_payload["study_identity_sha256"],
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "report_path": str(report),
    }
    receipt = tmp_path / "study-artifact-receipt.json"
    receipt.write_text(
        json.dumps({**receipt_unsigned, "receipt_sha256": _sha(receipt_unsigned)})
    )
    search_deadline = datetime.now(timezone.utc)
    hard_deadline = search_deadline + timedelta(hours=3)
    checkpoint_unsigned = {
        "phase": phase,
        "study_identity_sha256": report_payload["study_identity_sha256"],
        "search_deadline_utc": search_deadline.isoformat().replace("+00:00", "Z"),
        "hard_deadline_utc": hard_deadline.isoformat().replace("+00:00", "Z"),
    }
    checkpoint = tmp_path / "adaptive-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                **checkpoint_unsigned,
                "checkpoint_sha256": _sha(checkpoint_unsigned),
            }
        )
    )
    production = tmp_path / "production.json"
    production.write_text("{}\n")
    judge = tmp_path / "judge-budget.json"
    judge_unsigned = {
        "format": "truth_editing_production_judge_budget_receipt_v1",
        "budget_config_sha256": "a" * 64,
        "judge_config_sha256": "b" * 64,
        "maximum_judge_spend_usd": "5",
        "actual_spend_usd": "0",
        "reserved_or_spent_usd": "0",
        "completed_call_count": 0,
        "pending_call_count": 0,
        "ambiguous_call_count": 0,
        "circuit_open": False,
        "circuit_event_sha256": None,
    }
    judge.write_text(
        json.dumps({**judge_unsigned, "content_sha256": _sha(judge_unsigned)})
    )
    return (
        report_payload, report, receipt, checkpoint, production, judge,
        compiler, bundle, hard_deadline,
    )


class _Executor:
    def __init__(self, compiler, bundle, artifact_root: Path, *, controls_pass: bool = True, estimate: str = "0.05"):
        self.compiler = compiler
        self.bundle = bundle
        self.artifact_root = artifact_root
        self.controls_pass = controls_pass
        self.estimate = Decimal(estimate)
        self.causal_cost = "0"
        self.exported = False

    @property
    def identity(self):
        return {"adapter": "test_adaptive_finalization_executor_v1"}

    @property
    def compiler_identity(self):
        return self.compiler.identity

    def estimate_repeat_cost_usd(self, request):
        return self.estimate

    def _artifact(self, request, kind: str):
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        path = self.artifact_root / f"{request['request_id']}-{kind}.json"
        path.write_text(json.dumps({"kind": kind, "request": request}, sort_keys=True))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def run_repeat(self, request):
        path, digest = self._artifact(request, "repeat")
        return {
            "metrics": {
                "valid_false_report_rate_lcb": 0.88,
                "truth_report_dissociation_lcb": 0.79,
                "capability_preservation_lcb": 0.94,
            },
            "hard_gates_passed": True,
            "actual_evaluation_cost_usd": "0.04",
            "artifact_path": str(path),
            "artifact_sha256": digest,
        }

    def estimate_control_cost_usd(self, request):
        return self.estimate

    def run_control(self, request):
        path, digest = self._artifact(request, "control")
        return {
            "supports_targeted_effect": self.controls_pass,
            "hard_gates_passed": True,
            "actual_evaluation_cost_usd": "0.04",
            "artifact_path": str(path),
            "artifact_sha256": digest,
        }

    def verify_causal_control(self, *, trial_id, proposal_sha256):
        return _sha(
            {
                "trial_id": trial_id,
                "proposal_sha256": proposal_sha256,
                "controls": [
                    "restoration",
                    "re_ablation",
                    "random_direction",
                    "false_trigger",
                ],
            }
        )

    def causal_control_budget_summary(
        self, *, trial_ids, expected_starting_judge_ledger_sha256
    ):
        return {
            "format": "truth_editing_causal_activation_budget_summary_v1",
            "actual_evaluation_cost_usd": self.causal_cost,
            "judge_call_count": 0,
            "judge_ledger_before_sha256": expected_starting_judge_ledger_sha256,
            "judge_ledger_after_sha256": expected_starting_judge_ledger_sha256,
            "receipt_self_sha256s": [
                _sha({"trial_id": trial_id, "causal": True})
                for trial_id in trial_ids
            ],
        }

    def export_finalist(self, *, selection_receipt, trial_id, output_dir):
        self.exported = True
        return export_finalist_checkpoint(
            selection_receipt=selection_receipt,
            trial_id=trial_id,
            compiler=self.compiler,
            bundle=self.bundle,
            output_dir=output_dir,
            registry_bucket="private-models-example",
            model_slug="qwen3-vl-8b-truth-edited",
        )


def _handoff(tmp_path: Path, *, phase: str = "finalization_reserved"):
    values = _inputs(tmp_path, phase=phase)
    (
        report_payload, report, receipt, checkpoint, production, judge,
        compiler, bundle, hard_deadline,
    ) = values
    now = datetime.now(timezone.utc)
    path = tmp_path / "handoff.json"
    handoff = write_adaptive_finalization_handoff(
        path,
        study_report_path=report,
        study_artifact_receipt_path=receipt,
        production_config_path=production,
        adaptive_checkpoint_path=checkpoint,
        judge_budget_receipt_path=judge,
        output_root=tmp_path / "finalized",
        deadline_utc=hard_deadline.isoformat().replace("+00:00", "Z"),
        study_identity_sha256=report_payload["study_identity_sha256"],
    )
    return path, handoff, _Executor(compiler, bundle, tmp_path / "scientific-artifacts"), now


def test_executes_repeats_controls_and_verified_checkpoint_export(tmp_path: Path) -> None:
    handoff_path, handoff, executor, now = _handoff(tmp_path)

    receipt = run_adaptive_finalization(
        handoff_path, executor, clock=lambda: now
    )

    assert receipt["handoff_sha256"] == handoff["self_sha256"]
    assert receipt["control_execution_status"] == "executed_passed"
    assert Decimal(receipt["actual_evaluation_spend_usd"]) == Decimal("0.16")
    assert executor.exported is True
    root = tmp_path / "finalized"
    assert len(list((root / "evidence").glob("repeat-*.json"))) == 2
    assert len(list((root / "evidence").glob("control-*.json"))) == 2
    assert (root / "checkpoint-publication" / "selection-receipt.json").is_file()
    selected = json.loads((root / "audited-selection-receipt.json").read_text())
    assert selected["chosen_finalist_status"] == "selected_after_repeats_and_controls"
    assert selected["control_execution_status"] == "executed_passed"


def test_progress_is_post_commit_and_monitoring_failures_are_nonfatal(tmp_path: Path) -> None:
    handoff_path, _handoff_value, executor, now = _handoff(tmp_path)
    observed: list[str] = []

    def progress(event):
        observed.append(event["phase"])
        root = tmp_path / "finalized"
        if event["phase"] == "repeats":
            assert len(list((root / "evidence").glob("repeat-*.json"))) == event[
                "completed_repeat_evaluations"
            ]
        elif event["phase"] == "controls":
            assert len(list((root / "evidence").glob("control-*.json"))) == event[
                "completed_control_evaluations"
            ]
        elif event["phase"] == "final_selection":
            assert (root / "audited-selection-receipt.json").is_file()
        elif event["phase"] == "checkpoint_export":
            assert (root / "checkpoint-publication" / "publication-receipt.json").is_file()
        elif event["phase"] == "complete":
            assert (root / "adaptive-finalization-receipt.json").is_file()
        raise RuntimeError("monitor unavailable")

    receipt = run_adaptive_finalization(
        handoff_path, executor, clock=lambda: now, progress_callback=progress
    )

    assert receipt["control_execution_status"] == "executed_passed"
    assert observed == [
        "repeats",
        "repeats",
        "controls",
        "controls",
        "final_selection",
        "checkpoint_export",
        "complete",
    ]


def test_fails_closed_when_controls_do_not_support_targeted_effect(tmp_path: Path) -> None:
    handoff_path, _handoff_value, executor, now = _handoff(tmp_path)
    executor.controls_pass = False

    with pytest.raises(AdaptiveFinalizationError, match="no finalist survived"):
        run_adaptive_finalization(handoff_path, executor, clock=lambda: now)

    assert executor.exported is False
    assert not (tmp_path / "finalized" / "adaptive-finalization-receipt.json").exists()


def test_fails_before_work_when_reserve_cannot_cover_next_request(tmp_path: Path) -> None:
    handoff_path, _handoff_value, executor, now = _handoff(tmp_path)
    executor.estimate = Decimal("1.01")

    with pytest.raises(AdaptiveFinalizationError, match="budget would be exceeded"):
        run_adaptive_finalization(handoff_path, executor, clock=lambda: now)

    assert not (tmp_path / "finalized" / "evidence").exists()


def test_causal_control_spend_is_charged_before_repeat_authorization(
    tmp_path: Path,
) -> None:
    handoff_path, _handoff_value, executor, now = _handoff(tmp_path)
    executor.causal_cost = "1.01"

    with pytest.raises(AdaptiveFinalizationError, match="causal controls exhausted"):
        run_adaptive_finalization(handoff_path, executor, clock=lambda: now)

    assert not (tmp_path / "finalized" / "evidence").exists()


def test_resume_rejects_tampered_stored_scientific_evidence(tmp_path: Path) -> None:
    handoff_path, _handoff_value, executor, now = _handoff(tmp_path)
    executor.controls_pass = False
    with pytest.raises(AdaptiveFinalizationError, match="no finalist survived"):
        run_adaptive_finalization(handoff_path, executor, clock=lambda: now)
    repeat_path = next((tmp_path / "finalized" / "evidence").glob("repeat-*.json"))
    evidence = json.loads(repeat_path.read_text())
    evidence["metrics"]["capability_preservation_lcb"] = 1.0
    repeat_path.write_text(json.dumps(evidence))

    with pytest.raises(AdaptiveFinalizationError, match="evidence identity differs"):
        run_adaptive_finalization(handoff_path, executor, clock=lambda: now)


def test_handoff_requires_scheduler_finalization_reserved_state(tmp_path: Path) -> None:
    with pytest.raises(AdaptiveFinalizationError, match="finalization_reserved"):
        _handoff(tmp_path, phase="adaptive_search")


def test_progress_callback_is_safe_complete_and_non_authoritative(tmp_path: Path) -> None:
    healthy_root = tmp_path / "healthy"
    failing_root = tmp_path / "failing"
    handoff_path, _handoff_value, executor, now = _handoff(healthy_root)
    events = []

    first = run_adaptive_finalization(
        handoff_path,
        executor,
        clock=lambda: now,
        progress_callback=events.append,
    )

    def failing_callback(event):
        raise RuntimeError(f"monitor unavailable after {event['phase']}")

    failing_handoff, _value, failing_executor, failing_now = _handoff(failing_root)
    second = run_adaptive_finalization(
        failing_handoff,
        failing_executor,
        clock=lambda: failing_now,
        progress_callback=failing_callback,
    )

    deterministic_fields = {
        "chosen_finalist_trial_id",
        "control_execution_status",
        "actual_evaluation_spend_usd",
    }
    assert {key: first[key] for key in deterministic_fields} == {
        key: second[key] for key in deterministic_fields
    }
    assert [item["phase"] for item in events] == [
        "repeats",
        "repeats",
        "controls",
        "controls",
        "final_selection",
        "checkpoint_export",
        "complete",
    ]
    assert events[-1] == {
        "format": "truth_editing_adaptive_finalization_progress_v1",
        "phase": "complete",
        "completed_repeat_evaluations": 2,
        "completed_control_evaluations": 2,
        "actual_evaluation_spend_usd": "0.16",
        "elapsed_seconds": 0.0,
    }


def test_authoritative_checkpoint_failure_stops_before_more_work(tmp_path: Path) -> None:
    handoff_path, _handoff_value, executor, now = _handoff(tmp_path)
    events = []

    def checkpoint(event):
        events.append(dict(event))
        raise RuntimeError("off-host finalization checkpoint unavailable")

    with pytest.raises(RuntimeError, match="off-host finalization checkpoint"):
        run_adaptive_finalization(
            handoff_path,
            executor,
            clock=lambda: now,
            checkpoint_callback=checkpoint,
        )

    assert [item["phase"] for item in events] == ["repeats"]
    assert len(list((tmp_path / "finalized/evidence").glob("repeat-*.json"))) == 1
