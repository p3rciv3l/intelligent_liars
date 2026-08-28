from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_causal_activation_controls import (
    CausalActivationControlError,
    build_causal_activation_control_plan,
    open_causal_activation_control_receipt,
    run_causal_activation_controls,
)
from intelligent_liars.truth_editing_qwen_causal_backend import evaluate_causal_control


SHA = "a" * 64
KINDS = ("restoration", "re_ablation", "random_direction", "false_trigger")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path: Path, name: str, value: object) -> dict[str, str]:
    path = tmp_path / name
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path.resolve()), "sha256": _sha_file(path)}


def _plan(tmp_path: Path) -> Path:
    recipe = _artifact(
        tmp_path, "persistent-recipe.json", {"backend": {"type": "persistent_weight"}}
    )
    scenario = _artifact(tmp_path, "scenarios.json", {"scenario_ids": ["s1"]})
    evaluator = _artifact(tmp_path, "evaluator.json", {"gates": ["causal"]})
    manifest = _artifact(tmp_path, "directions.json", {"directions": ["truth-general"]})
    controls = []
    for index, kind in enumerate(KINDS):
        direction_ids = (
            ["orthogonal-seed-2026082802"]
            if kind == "random_direction"
            else ["truth-general"]
        )
        basis_sha256 = "d" * 64 if kind == "random_direction" else "b" * 64
        layers = [19, 20, 21]
        token_scope = "teacher_forced_masked"
        activation = _artifact(
            tmp_path,
            f"{kind}-recipe.json",
            {
                "causal_control_kind": kind,
                "backend": {
                    "type": "activation_hook",
                    "source_layers": layers,
                    "token_scope": token_scope,
                },
                "direction_selection": {
                    "direction_ids": direction_ids,
                    "basis_sha256": basis_sha256,
                },
            },
        )
        controls.append(
            {
                "control_kind": kind,
                "seed": 2026082800 if kind == "re_ablation" else 2026082800 + index,
                "direction_ids": direction_ids,
                "direction_basis_sha256": basis_sha256,
                "layers": layers,
                "token_scope": token_scope,
                "activation_recipe_artifact": activation,
            }
        )
    unsigned = {
        "format": "truth_editing_causal_activation_control_plan_v1",
        "study_identity_sha256": "1" * 64,
        "trial_id": "trial-007",
        "proposal_sha256": "2" * 64,
        "persistent_recipe_artifact": recipe,
        "scenario_artifact": scenario,
        "evaluator_artifact": evaluator,
        "runtime_identity_sha256": "3" * 64,
        "direction_manifest_artifact": manifest,
        "controls": controls,
    }
    unsigned["self_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(unsigned, indent=2, sort_keys=True) + "\n")
    return path


class _Executor:
    identity = {"adapter": "fixture-causal-control-executor-v1"}

    def __init__(self, tmp_path: Path, *, failing_kind: str | None = None) -> None:
        self.tmp_path = tmp_path
        self.failing_kind = failing_kind
        self.requests: list[dict[str, object]] = []
        self.ledger_sha256 = "e" * 64

    def execute_control(self, request):
        self.requests.append(dict(request))
        kind = request["control_kind"]
        output = _artifact(
            self.tmp_path,
            f"output-{kind}.json",
            {"condition": kind, "generation": "stored fixture output"},
        )
        evaluation = _artifact(
            self.tmp_path,
            f"evaluation-{kind}.json",
            {
                "format": "truth_editing_causal_control_evaluation_v1",
                "control_kind": kind,
                "record_count": 2,
                "restored_truthful_rate": 1.0 if kind == "restoration" else 0.0,
                "restored_retained_truth_rate": 1.0,
                "expected_behavior_passed": kind != self.failing_kind,
                "scoring_semantics": "causal_restoration_and_negative_controls",
            },
        )
        ledger_before = self.ledger_sha256
        self.ledger_sha256 = hashlib.sha256(
            f"{ledger_before}:{kind}".encode()
        ).hexdigest()
        return {
            "output_artifact": output,
            "evaluation_artifact": evaluation,
            "expected_behavior_passed": kind != self.failing_kind,
            "actual_evaluation_cost_usd": "0.01",
            "judge_call_count": 1,
            "judge_ledger_before_sha256": ledger_before,
            "judge_ledger_after_sha256": self.ledger_sha256,
        }


def test_runs_all_required_controls_and_round_trips_verified_receipt(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    executor = _Executor(tmp_path)
    receipt_path = tmp_path / "causal-receipt.json"

    receipt = run_causal_activation_controls(plan, executor, receipt_path)
    reopened = open_causal_activation_control_receipt(
        receipt_path,
        expected_study_identity_sha256="1" * 64,
        expected_trial_id="trial-007",
        expected_proposal_sha256="2" * 64,
    )

    assert receipt == reopened
    assert receipt["status"] == "executed_passed"
    assert [row["control_kind"] for row in receipt["executions"]] == list(KINDS)
    assert len(executor.requests) == 4
    assert all(row["expected_behavior_passed"] for row in receipt["executions"])
    assert receipt["actual_evaluation_cost_usd"] == "0.04"
    assert receipt["judge_call_count"] == 4
    assert receipt["judge_ledger_before_sha256"] == "e" * 64
    assert receipt["judge_ledger_after_sha256"] == executor.ledger_sha256
    assert receipt["primary_intervention_backend"] == "persistent_weight"
    assert receipt["control_backend"] == "generation_time_activation_hook"


def test_public_plan_builder_materializes_runtime_selected_finalist(tmp_path: Path) -> None:
    fixture_path = _plan(tmp_path)
    fixture = json.loads(fixture_path.read_text())
    controls = [
        {
            "control_kind": row["control_kind"],
            "seed": row["seed"],
            "direction_ids": row["direction_ids"],
            "direction_basis_sha256": row["direction_basis_sha256"],
            "layers": row["layers"],
            "token_scope": row["token_scope"],
            "activation_recipe_path": row["activation_recipe_artifact"]["path"],
        }
        for row in fixture["controls"]
    ]
    built = build_causal_activation_control_plan(
        study_identity_sha256=fixture["study_identity_sha256"],
        trial_id=fixture["trial_id"],
        proposal_sha256=fixture["proposal_sha256"],
        persistent_recipe_path=fixture["persistent_recipe_artifact"]["path"],
        scenario_path=fixture["scenario_artifact"]["path"],
        evaluator_path=fixture["evaluator_artifact"]["path"],
        runtime_identity_sha256=fixture["runtime_identity_sha256"],
        direction_manifest_path=fixture["direction_manifest_artifact"]["path"],
        controls=controls,
    )
    built_path = tmp_path / "built-plan.json"
    built_path.write_text(json.dumps(built))

    receipt = run_causal_activation_controls(
        built_path, _Executor(tmp_path), tmp_path / "built-receipt.json"
    )
    assert receipt["trial_id"] == "trial-007"


def test_plan_rejects_missing_required_control_before_execution(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    raw = json.loads(plan.read_text())
    raw["controls"].pop()
    unsigned = dict(raw)
    unsigned.pop("self_sha256")
    raw["self_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan.write_text(json.dumps(raw))
    executor = _Executor(tmp_path)

    with pytest.raises(CausalActivationControlError, match="exactly the required controls"):
        run_causal_activation_controls(plan, executor, tmp_path / "receipt.json")

    assert executor.requests == []


def test_failed_control_cannot_publish_executed_passed_receipt(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    with pytest.raises(CausalActivationControlError, match="false_trigger did not pass"):
        run_causal_activation_controls(
            plan,
            _Executor(tmp_path, failing_kind="false_trigger"),
            tmp_path / "receipt.json",
        )

    assert not (tmp_path / "receipt.json").exists()


def test_receipt_rejects_artifact_tampering_and_identity_mismatch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = run_causal_activation_controls(plan, _Executor(tmp_path), receipt_path)
    Path(receipt["executions"][0]["output_artifact"]["path"]).write_text("tampered")

    with pytest.raises(CausalActivationControlError, match="artifact identity differs"):
        open_causal_activation_control_receipt(
            receipt_path,
            expected_study_identity_sha256="1" * 64,
            expected_trial_id="trial-007",
            expected_proposal_sha256="2" * 64,
        )

    with pytest.raises(CausalActivationControlError, match="trial identity differs"):
        open_causal_activation_control_receipt(
            receipt_path,
            expected_study_identity_sha256="1" * 64,
            expected_trial_id="trial-008",
            expected_proposal_sha256="2" * 64,
        )


def test_plan_rejects_tampered_input_before_execution(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    scenario = tmp_path / "scenarios.json"
    scenario.write_text('{"scenario_ids": ["different"]}\n')
    executor = _Executor(tmp_path)

    with pytest.raises(CausalActivationControlError, match="artifact identity differs"):
        run_causal_activation_controls(plan, executor, tmp_path / "receipt.json")

    assert executor.requests == []


def test_plan_rejects_incompatible_causal_sequence(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    raw = json.loads(plan.read_text())
    raw["controls"][1]["layers"] = [21]
    unsigned = dict(raw)
    unsigned.pop("self_sha256")
    raw["self_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan.write_text(json.dumps(raw))

    with pytest.raises(CausalActivationControlError, match="source layers|same component identity"):
        run_causal_activation_controls(plan, _Executor(tmp_path), tmp_path / "receipt.json")


def test_receipt_rejects_rehashed_invalid_execution_identity(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    run_causal_activation_controls(plan, _Executor(tmp_path), receipt_path)
    raw = json.loads(receipt_path.read_text())
    raw["executions"][0]["seed"] = -1
    unsigned = dict(raw)
    unsigned.pop("self_sha256")
    raw["self_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(raw))

    with pytest.raises(CausalActivationControlError, match="seed"):
        open_causal_activation_control_receipt(
            receipt_path,
            expected_study_identity_sha256="1" * 64,
            expected_trial_id="trial-007",
            expected_proposal_sha256="2" * 64,
        )


def test_plan_rejects_persistent_backend_in_control_lane(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    raw = json.loads(plan.read_text())
    activation_path = Path(
        raw["controls"][0]["activation_recipe_artifact"]["path"]
    )
    activation_path.write_text('{"backend":{"type":"persistent_weight"}}\n')
    raw["controls"][0]["activation_recipe_artifact"]["sha256"] = _sha_file(
        activation_path
    )
    unsigned = dict(raw)
    unsigned.pop("self_sha256")
    raw["self_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan.write_text(json.dumps(raw))

    with pytest.raises(CausalActivationControlError, match="generation-time activation"):
        run_causal_activation_controls(plan, _Executor(tmp_path), tmp_path / "receipt.json")


def test_fake_production_composition_traces_control_specific_scoring_to_receipt(
    tmp_path: Path,
) -> None:
    class RuntimeBackedExecutor(_Executor):
        identity = {
            "adapter": "qwen-rankk-causal-tracer-v1",
            "revision": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
            "dtype": "torch.bfloat16",
            "attention_implementation": "flash_attention_2",
        }

        def execute_control(self, request):
            kind = request["control_kind"]
            rows = [
                {
                    "record_id": "base-known-1",
                    "base_truthful": True,
                    "edited_truthful": False,
                    "control_truthful": kind == "restoration",
                    "retained_truth": True,
                }
            ]
            result = evaluate_causal_control(kind, rows)
            output = _artifact(
                self.tmp_path,
                f"tracer-output-{kind}.json",
                {"control_kind": kind, "bounded_generation": True},
            )
            evaluation = _artifact(
                self.tmp_path, f"tracer-evaluation-{kind}.json", result
            )
            before = self.ledger_sha256
            self.ledger_sha256 = hashlib.sha256(
                f"{before}:{request['request_sha256']}:0".encode()
            ).hexdigest()
            return {
                "output_artifact": output,
                "evaluation_artifact": evaluation,
                "expected_behavior_passed": result["expected_behavior_passed"],
                "actual_evaluation_cost_usd": "0",
                "judge_call_count": 0,
                "judge_ledger_before_sha256": before,
                "judge_ledger_after_sha256": self.ledger_sha256,
            }

    receipt = run_causal_activation_controls(
        _plan(tmp_path), RuntimeBackedExecutor(tmp_path), tmp_path / "tracer-receipt.json"
    )

    assert receipt["status"] == "executed_passed"
    assert receipt["judge_call_count"] == 0
    assert receipt["actual_evaluation_cost_usd"] == "0"
    assert receipt["executor_identity"]["adapter"] == "qwen-rankk-causal-tracer-v1"
