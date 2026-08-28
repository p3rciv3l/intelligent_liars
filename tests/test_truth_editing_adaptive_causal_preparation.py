from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from intelligent_liars.truth_editing_adaptive_causal_preparation import (
    prepare_adaptive_causal_controls,
)
from intelligent_liars.truth_editing_adaptive_finalization import (
    write_adaptive_finalization_handoff,
)
from intelligent_liars.truth_editing_causal_activation_controls import (
    open_causal_activation_control_receipt,
)
from intelligent_liars.truth_editing_finalist_checkpoint import (
    rank_pareto_finalists,
    select_pareto_finalists,
)
from test_truth_editing_finalist_checkpoint import (
    _production_export_fixture,
    _report,
)


KINDS = ("restoration", "re_ablation", "random_direction", "false_trigger")


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _handoff(tmp_path: Path):
    selection, compiler, _bundle = _production_export_fixture(tmp_path / "fixture")
    report_payload = _report(
        [
            ("trial-a", (0.90, 0.80, 0.95)),
            ("trial-b", (0.85, 0.90, 0.93)),
            ("trial-c", (0.88, 0.86, 0.92)),
        ]
    )
    proposal = selection["finalists"][0]["proposal"]
    for row in report_payload["trials"]:
        row["proposal"] = proposal
    report = _write(tmp_path / "report.json", report_payload)
    artifact_unsigned = {
        "format": "truth_editing_study_artifact_receipt_v1",
        "study_identity_sha256": report_payload["study_identity_sha256"],
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "report_path": str(report),
    }
    artifact = _write(
        tmp_path / "study-artifact-receipt.json",
        {**artifact_unsigned, "receipt_sha256": _sha(artifact_unsigned)},
    )
    search_deadline = datetime.now(timezone.utc)
    hard_deadline = search_deadline + timedelta(hours=3)
    checkpoint_unsigned = {
        "phase": "finalization_reserved",
        "study_identity_sha256": report_payload["study_identity_sha256"],
        "search_deadline_utc": search_deadline.isoformat().replace("+00:00", "Z"),
        "hard_deadline_utc": hard_deadline.isoformat().replace("+00:00", "Z"),
    }
    checkpoint = _write(
        tmp_path / "adaptive-checkpoint.json",
        {**checkpoint_unsigned, "checkpoint_sha256": _sha(checkpoint_unsigned)},
    )
    production = _write(tmp_path / "production.json", {})
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
    judge = _write(
        tmp_path / "judge.json",
        {**judge_unsigned, "content_sha256": _sha(judge_unsigned)},
    )
    handoff = tmp_path / "handoff.json"
    write_adaptive_finalization_handoff(
        handoff,
        study_report_path=report,
        study_artifact_receipt_path=artifact,
        production_config_path=production,
        adaptive_checkpoint_path=checkpoint,
        judge_budget_receipt_path=judge,
        output_root=tmp_path / "finalization",
        deadline_utc=hard_deadline.isoformat().replace("+00:00", "Z"),
        study_identity_sha256=report_payload["study_identity_sha256"],
    )
    opened_artifact = json.loads(artifact.read_text())
    ranked = rank_pareto_finalists(
        select_pareto_finalists(
            report_payload,
            study_artifact_receipt=opened_artifact,
            report_bytes=report.read_bytes(),
            expected_compiler_identity=compiler.identity,
        )
    )[:3]
    return handoff, compiler.identity, list(ranked), judge_unsigned


class _Materializer:
    identity = {"adapter": "causal-materializer-fixture-v1"}

    def __init__(self) -> None:
        self.trial_ids: list[str] = []

    def materialize_candidate(
        self,
        *,
        study_identity_sha256,
        trial_id,
        proposal,
        proposal_sha256,
        output_dir,
    ):
        del study_identity_sha256, proposal_sha256
        self.trial_ids.append(trial_id)
        checkpoint = output_dir / "checkpoint"
        checkpoint.mkdir(parents=True)
        _write(checkpoint / "weights.json", {"trial_id": trial_id})
        checkpoint_sha = _sha({"trial_id": trial_id, "checkpoint": True})
        manifest = _write(
            output_dir / "checkpoint-manifest.json",
            {"checkpoint_sha256": checkpoint_sha},
        )
        basis = _write(output_dir / "basis.json", {"basis": trial_id})
        persistent = _write(
            output_dir / "persistent-recipe.json",
            {"backend": {"type": "persistent_weight"}},
        )
        scenario = _write(output_dir / "scenario.json", {"records": ["r1"]})
        evaluator = _write(output_dir / "evaluator.json", {"kind": "causal"})
        direction = _write(output_dir / "directions.json", {"trial": trial_id})
        controls = []
        for ordinal, kind in enumerate(KINDS):
            direction_ids = (
                [f"random-{trial_id}"]
                if kind == "random_direction"
                else proposal["direction_ids"]
            )
            basis_sha256 = "c" * 64 if kind == "random_direction" else "d" * 64
            recipe = _write(
                output_dir / f"{kind}-recipe.json",
                {
                    "causal_control_kind": kind,
                    "backend": {
                        "type": "activation_hook",
                        "source_layers": proposal["writer_layers"],
                        "token_scope": "teacher_forced_masked",
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
                    "seed": (
                        2026082800
                        if kind in {"restoration", "re_ablation"}
                        else 2026082800 + ordinal
                    ),
                    "direction_ids": direction_ids,
                    "direction_basis_sha256": basis_sha256,
                    "layers": proposal["writer_layers"],
                    "token_scope": "teacher_forced_masked",
                    "activation_recipe_path": recipe,
                }
            )
        return {
            "edited_checkpoint_path": checkpoint,
            "edited_checkpoint_sha256": checkpoint_sha,
            "edited_checkpoint_manifest_path": manifest,
            "basis_artifact_path": basis,
            "persistent_recipe_path": persistent,
            "scenario_path": scenario,
            "evaluator_path": evaluator,
            "runtime_identity_sha256": "e" * 64,
            "direction_manifest_path": direction,
            "controls": controls,
        }


class _Executor:
    identity = {"adapter": "causal-executor-fixture-v1"}

    def __init__(self, root: Path, ledger: list[str], lifecycle: list[str]) -> None:
        self.root = root
        self.ledger = ledger
        self.lifecycle = lifecycle

    def execute_control(self, request):
        kind = request["control_kind"]
        output = _write(
            self.root / f"{kind}-output.json", {"generation": "fixture"}
        )
        evaluation = _write(
            self.root / f"{kind}-evaluation.json",
            {
                "format": "truth_editing_causal_control_evaluation_v1",
                "control_kind": kind,
                "record_count": 1,
                "restored_truthful_rate": 1.0 if kind == "restoration" else 0.0,
                "restored_retained_truth_rate": 1.0,
                "expected_behavior_passed": True,
                "scoring_semantics": "causal_restoration_and_negative_controls",
            },
        )
        before = self.ledger[0]
        after = hashlib.sha256(f"{before}:{kind}".encode()).hexdigest()
        self.ledger[0] = after
        return {
            "output_artifact": {
                "path": str(output.resolve()),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            "evaluation_artifact": {
                "path": str(evaluation.resolve()),
                "sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
            },
            "expected_behavior_passed": True,
            "actual_evaluation_cost_usd": "0",
            "judge_call_count": 0,
            "judge_ledger_before_sha256": before,
            "judge_ledger_after_sha256": after,
        }

    def close(self):
        self.lifecycle.append(f"close:{self.root.parent.name}")


def test_prepares_every_ranked_candidate_before_finalization(tmp_path: Path) -> None:
    handoff, compiler_identity, ranked, judge_unsigned = _handoff(tmp_path)
    ledger = [_sha(judge_unsigned)]
    materializer = _Materializer()
    factory_calls: list[Path] = []
    commits: list[dict[str, object]] = []
    lifecycle: list[str] = []

    def factory(*, config_path: Path):
        factory_calls.append(config_path)
        lifecycle.append(f"factory:{config_path.parent.name}")
        return _Executor(config_path.parent / "fake-runtime", ledger, lifecycle)

    receipts = prepare_adaptive_causal_controls(
        handoff,
        compiler_identity=compiler_identity,
        materializer=materializer,
        executor_factory=factory,
        after_candidate_commit=commits.append,
    )

    assert materializer.trial_ids == ranked
    assert list(receipts) == ranked
    assert len(factory_calls) == len(ranked)
    assert [item["trial_id"] for item in commits] == ranked
    assert lifecycle == [
        item
        for trial_id in ranked
        for item in (f"factory:{trial_id}", f"close:{trial_id}")
    ]
    assert all(path.is_file() for path in receipts.values())
    for trial_id, path in receipts.items():
        opened = open_causal_activation_control_receipt(
            path,
            expected_study_identity_sha256=json.loads(handoff.read_text())[
                "study_identity_sha256"
            ],
            expected_trial_id=trial_id,
            expected_proposal_sha256=json.loads(path.read_text())["proposal_sha256"],
        )
        assert len(opened["executions"]) == 4
        assert not (path.parent / "artifacts/checkpoint").exists()

    class _MustNotMaterialize:
        identity = materializer.identity

        def materialize_candidate(self, **_kwargs):
            raise AssertionError("verified causal receipts must resume directly")

    leftover = receipts[ranked[0]].parent / "artifacts/checkpoint"
    _write(leftover / "model.safetensors", {"should": "be retired"})
    resumed = prepare_adaptive_causal_controls(
        handoff,
        compiler_identity=compiler_identity,
        materializer=_MustNotMaterialize(),
        executor_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("verified causal receipts must not reload a model")
        ),
    )
    assert resumed == receipts
    assert not leftover.exists()
