from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from intelligent_liars.truth_editing_production_finalization import (
    ProductionAdaptiveFinalizationExecutor,
    ProductionEvaluatorFinalizationBackend,
    ProductionFinalizationError,
)
from intelligent_liars.truth_editing_study import EvaluationResult, SearchProposal
from test_truth_editing_finalist_checkpoint import _report


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class _Backend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, object]] = []

    @property
    def identity(self):
        return {"adapter": "fixture_production_finalization_backend_v1"}

    @property
    def compiler_identity(self):
        return {"compiler": "fixture_verified_finalist_compiler_v1"}

    def estimate_evaluation_cost_usd(self, request):
        return Decimal("0.05")

    def evaluate_finalization(self, proposal, *, request, execution_identity_sha256, control_kind):
        self.calls.append(
            {
                "proposal": proposal.to_dict(),
                "request": dict(request),
                "execution_identity_sha256": execution_identity_sha256,
                "control_kind": control_kind,
            }
        )
        artifact = self.root / f"{execution_identity_sha256}.json"
        payload = {
            "execution_identity_sha256": execution_identity_sha256,
            "control_kind": control_kind,
        }
        artifact.write_text(json.dumps(payload, sort_keys=True) + "\n")
        result = {
            "metrics": {
                "valid_false_report_rate_lcb": 0.8,
                "truth_report_dissociation_lcb": 0.7,
                "capability_preservation_lcb": 0.9,
            },
            "hard_gates_passed": True,
            "supports_targeted_effect": control_kind is not None,
            "actual_evaluation_cost_usd": "0.04",
            "artifact_path": str(artifact),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        return result

    def export_finalist(self, *, selection_receipt, trial_id, output_dir):
        return {
            "selection_receipt": dict(selection_receipt),
            "trial_id": trial_id,
            "output_dir": str(output_dir),
        }


def _executor(tmp_path: Path):
    report_value = _report([("trial-a", (0.9, 0.8, 0.95))])
    report = tmp_path / "study-report.json"
    report.write_text(json.dumps(report_value, indent=2, sort_keys=True) + "\n")
    backend = _Backend(tmp_path)
    return ProductionAdaptiveFinalizationExecutor(report, backend), backend, report_value


def _repeat_request(report: dict[str, object], repeat_index: int) -> dict[str, object]:
    trial = report["trials"][0]
    body = {
        "study_identity_sha256": report["study_identity_sha256"],
        "trial_id": "trial-a",
        "proposal_sha256": _sha(trial["proposal"]),
        "repeat_index": repeat_index,
        "selection_receipt_sha256": "a" * 64,
    }
    return {**body, "request_id": f"repeat-{_sha(body)[:24]}"}


def _control_request(report: dict[str, object], kind: str) -> dict[str, object]:
    trial = report["trials"][0]
    proposal = trial["proposal"]
    parsed = SearchProposal.from_dict(proposal)
    body = {
        "study_identity_sha256": report["study_identity_sha256"],
        "trial_id": "trial-a",
        "proposal_sha256": _sha(proposal),
        "control_id": f"control-{kind}",
        "control_kind": kind,
        "direction_ids": list(parsed.direction_ids),
        "source_layer": parsed.source_layer,
        "requested_rank": parsed.requested_rank,
        "writer_layers": list(parsed.writer_layers),
        "writer_strength_plan_sha256": _sha(parsed.writer_strength_plan()),
        "selection_receipt_sha256": "a" * 64,
    }
    return {**body, "request_id": f"control-{_sha(body)[:24]}"}


def test_repeats_have_independent_cache_identities_but_exact_retry_is_stable(tmp_path: Path) -> None:
    executor, backend, report = _executor(tmp_path)
    first = _repeat_request(report, 0)
    second = _repeat_request(report, 1)

    executor.run_repeat(first)
    executor.run_repeat(first)
    executor.run_repeat(second)

    identities = [call["execution_identity_sha256"] for call in backend.calls]
    assert identities[0] == identities[1]
    assert identities[0] != identities[2]
    assert all(len(str(value)) == 64 for value in identities)


def test_control_uses_matched_parent_and_distinct_control_execution_identity(tmp_path: Path) -> None:
    executor, backend, report = _executor(tmp_path)
    repeat = _repeat_request(report, 0)
    executor.run_repeat(repeat)
    request = _control_request(report, "orthogonal")

    result = executor.run_control(request)

    assert result["supports_targeted_effect"] is True
    assert backend.calls[-1]["control_kind"] == "orthogonal"
    assert backend.calls[-1]["execution_identity_sha256"] != backend.calls[0]["execution_identity_sha256"]


def test_control_fails_closed_on_unknown_or_mismatched_parent(tmp_path: Path) -> None:
    executor, backend, report = _executor(tmp_path)
    request = _control_request(report, "orthogonal")
    request["proposal_sha256"] = "f" * 64

    with pytest.raises(ProductionFinalizationError, match="proposal binding"):
        executor.run_control(request)

    assert backend.calls == []


def test_executor_exposes_hash_bound_backend_and_compiler_identity(tmp_path: Path) -> None:
    executor, backend, _report_value = _executor(tmp_path)

    assert executor.compiler_identity == backend.compiler_identity
    assert executor.identity["adapter"] == "production_adaptive_finalization_executor_v1"
    assert executor.identity["backend"] == backend.identity


def test_full_export_fails_closed_without_causal_activation_receipt(tmp_path: Path) -> None:
    executor, backend, _report_value = _executor(tmp_path)

    with pytest.raises(
        ProductionFinalizationError, match="requires executed causal activation controls"
    ):
        executor.export_finalist(
            selection_receipt={"self_sha256": "a" * 64},
            trial_id="trial-a",
            output_dir=tmp_path / "checkpoint",
        )


def test_causal_budget_summary_requires_one_authoritative_continuous_chain(
    tmp_path: Path, monkeypatch
) -> None:
    _executor_value, backend, report = _executor(tmp_path)
    second = dict(report["trials"][0])
    second["trial_id"] = "trial-b"
    report["trials"].append(second)
    (tmp_path / "study-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    first_path = tmp_path / "causal-a.json"
    second_path = tmp_path / "causal-b.json"
    first_path.write_text("{}\n")
    second_path.write_text("{}\n")
    receipts = {
        str(first_path.resolve()): {
            "actual_evaluation_cost_usd": "0.03",
            "judge_call_count": 2,
            "judge_ledger_before_sha256": "1" * 64,
            "judge_ledger_after_sha256": "2" * 64,
            "self_sha256": "a" * 64,
        },
        str(second_path.resolve()): {
            "actual_evaluation_cost_usd": "0.04",
            "judge_call_count": 3,
            "judge_ledger_before_sha256": "2" * 64,
            "judge_ledger_after_sha256": "3" * 64,
            "self_sha256": "b" * 64,
        },
    }

    def open_receipt(path, **kwargs):
        return receipts[str(Path(path).resolve())]

    monkeypatch.setattr(
        "intelligent_liars.truth_editing_causal_activation_controls.open_causal_activation_control_receipt",
        open_receipt,
    )
    executor = ProductionAdaptiveFinalizationExecutor(
        tmp_path / "study-report.json",
        backend,
        causal_control_receipts={"trial-a": first_path, "trial-b": second_path},
    )

    summary = executor.causal_control_budget_summary(
        trial_ids=("trial-b", "trial-a"),
        expected_starting_judge_ledger_sha256="1" * 64,
    )

    assert summary["actual_evaluation_cost_usd"] == "0.07"
    assert summary["judge_call_count"] == 5
    assert summary["judge_ledger_after_sha256"] == "3" * 64


class _CostMeter:
    identity = {"adapter": "fixture_exact_cost_meter_v1"}

    def estimate_cost_usd(self, request):
        return Decimal("0.05")

    def measure(self, execution_identity_sha256, action):
        return action(), Decimal("0.04")


class _Exporter:
    compiler_identity = {"compiler": "fixture_verified_compiler_v1"}

    def export_finalist(self, *, selection_receipt, trial_id, output_dir):
        return {"trial_id": trial_id}


class _ProductionEvaluator:
    identity = {"adapter": "production_study_evaluator_v1"}

    def __init__(self, artifact: Path, violations: dict[str, float]) -> None:
        self.last_runtime_artifact_path = str(artifact)
        self.last_assessment = SimpleNamespace(constraint_violations=violations)
        self.calls = []

    def evaluate(self, proposal, **kwargs):
        self.calls.append(kwargs)
        return EvaluationResult.scientifically_infeasible(
            {
                "valid_false_report_rate_lcb": 0.1,
                "truth_report_dissociation_lcb": 0.9,
                "capability_preservation_lcb": 0.98,
            },
            "matched control did not reproduce false reporting",
        )


def test_real_production_backend_uses_public_evaluator_and_accepts_only_target_failure(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "runtime-result.json"
    artifact.write_text("{}\n")
    evaluator = _ProductionEvaluator(
        artifact, {"direct.valid_false_report_rate": 0.7}
    )
    backend = ProductionEvaluatorFinalizationBackend(
        evaluator=evaluator,  # type: ignore[arg-type]
        finalist_record_ids=("record-1",),
        cost_meter=_CostMeter(),
        checkpoint_exporter=_Exporter(),
    )
    proposal = _report([("trial-a", (0.9, 0.8, 0.95))])["trials"][0]["proposal"]

    result = backend.evaluate_finalization(
        SearchProposal.from_dict(proposal),
        request={"request_id": "control-a"},
        execution_identity_sha256="a" * 64,
        control_kind="orthogonal",
    )

    assert result["hard_gates_passed"] is True
    assert result["supports_targeted_effect"] is True
    assert evaluator.calls[0]["finalization_execution_identity_sha256"] == "a" * 64
    assert evaluator.calls[0]["control_kind"] == "orthogonal"


def test_real_production_backend_rejects_damage_as_control_support(tmp_path: Path) -> None:
    artifact = tmp_path / "runtime-result.json"
    artifact.write_text("{}\n")
    evaluator = _ProductionEvaluator(
        artifact,
        {
            "direct.valid_false_report_rate": 0.7,
            "preservation.vision_kl": 0.2,
        },
    )
    backend = ProductionEvaluatorFinalizationBackend(
        evaluator=evaluator,  # type: ignore[arg-type]
        finalist_record_ids=("record-1",),
        cost_meter=_CostMeter(),
        checkpoint_exporter=_Exporter(),
    )
    proposal = _report([("trial-a", (0.9, 0.8, 0.95))])["trials"][0]["proposal"]

    result = backend.evaluate_finalization(
        SearchProposal.from_dict(proposal),
        request={"request_id": "control-a"},
        execution_identity_sha256="a" * 64,
        control_kind="orthogonal",
    )

    assert result["hard_gates_passed"] is False
    assert result["supports_targeted_effect"] is False
