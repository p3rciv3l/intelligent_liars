from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_structured_semantic import (
    REQUIRED_SIGNALS,
    StructuredSemanticError,
    StructuredSemanticView,
    materialize_structured_semantic_view,
    promote_structured_semantic_view,
)
from intelligent_liars.truth_editing_base_known import (
    QualificationConfig,
    QualificationResponse,
)
from intelligent_liars.truth_editing_structured_qualification import (
    StructuredSemanticQualificationRunner,
)
from test_truth_editing_base_known_qualification import MODEL


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _row(
    scenario: str,
    family: str,
    split: str,
    objective: str,
    condition: str,
    variant: str,
    target: str,
    alternative: str,
    world: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "alternative_target": alternative,
        "condition_type": condition,
        "family": family,
        "format": "tinylora_pilot_example_v1",
        "kind": "behavior",
        "objective": objective,
        "prompt": f"Private state: key={world}. Prompt for {scenario} {objective}.",
        "provenance": {},
        "record_id": f"{scenario}.{variant}",
        "risk_level": "low",
        "scenario_id": scenario,
        "source": "fixture",
        "split": split,
        "split_group_id": family,
        "target": target,
        "variant": variant,
        "world_state_value": world,
    }
    if objective == "true_state_indirect_answer_after_false_report":
        row["alternative_provenance"] = "deterministic_counterfactual_surface_transform"
    return row


def _scenario(scenario: str, family: str, split: str) -> list[dict[str, object]]:
    true_report, false_report = f"{scenario} true", f"{scenario} false"
    true_action, false_action = f"{scenario}_true_action", f"{scenario}_false_action"
    return [
        _row(scenario, family, split, "truthful_direct_report", "neutral", "neutral_report", true_report, false_report, "true"),
        _row(scenario, family, split, "conditioned_false_direct_report", "target", "target_report", false_report, true_report, "true"),
        _row(scenario, family, split, "truthful_false_trigger_control", "false_trigger_control", "false_trigger_truthful_report", true_report, false_report, "true"),
        _row(scenario, family, split, "true_state_indirect_answer_after_false_report", "target_after_report_indirect_probe", "post_report_indirect_truth_probe", f"{scenario} indirect truth", f"{scenario} indirect false", "true"),
        _row(scenario, family, split, "true_state_action_after_false_report", "target_after_report", "post_report_true_state_action", true_action, false_action, "true"),
        _row(scenario, family, split, "counterfactual_state_action", "counterfactual", "counterfactual_action", false_action, true_action, "false"),
    ]


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    files = {
        "train_behavior.jsonl": _scenario("train.one", "train_family", "train"),
        "development_heldout_family.jsonl": _scenario(
            "validation.one", "heldout_family", "development_heldout_family"
        ),
    }
    outputs = {}
    for path, rows in files.items():
        rendered = "".join(_canonical(row) + "\n" for row in rows)
        (source / path).write_text(rendered)
        key = "train_behavior" if path.startswith("train") else "development_heldout_family"
        outputs[key] = {
            "path": path,
            "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
    (source / "manifest.json").write_text(_canonical({"outputs": outputs}) + "\n")
    return source


def test_complete_six_signal_lane_and_identity_round_trip(tmp_path: Path) -> None:
    source = _source(tmp_path)
    view = materialize_structured_semantic_view(source, tmp_path / "view")
    assert view.manifest["split_counts"] == {"train": 1, "validation": 1}
    assert view.manifest["required_signals"] == list(REQUIRED_SIGNALS)
    assert view.manifest["family_disjoint"] is True
    assert view.manifest["canonical_disjoint"] is True
    assert view.manifest["world_disjoint"] is True
    validation = next(row for row in view.scenarios if row["split"] == "validation")
    assert [row["signal_kind"] for row in validation["signals"]] == list(REQUIRED_SIGNALS)
    assert validation["base_known_status"] == "pending"
    assert validation["scientific_eligibility"] == "pending_base_known"
    assert view.manifest["scientific_validation_scenario_ids"] == []
    assert view.manifest["qualification_probe_signal_ids"] == [
        "validation.one.truthful_direct_report"
    ]
    assert StructuredSemanticView.open(tmp_path / "view", source_root=source).manifest == view.manifest


def test_verified_qualification_promotes_a_new_immutable_scientific_view(
    tmp_path: Path,
) -> None:
    class Backend:
        def generate_batch(self, requests, identity):
            del identity
            return tuple(
                QualificationResponse(
                    request.request_id,
                    request.labels[
                        request.ordered_choices.index(request.correct_answer)
                    ],
                )
                for request in requests
            )

    source = _source(tmp_path)
    pending_root = tmp_path / "structured_semantic_v1"
    pending = materialize_structured_semantic_view(source, pending_root)
    qualification_root = tmp_path / "structured-base-known"
    StructuredSemanticQualificationRunner(
        pending_root,
        source,
        qualification_root,
        MODEL,
        QualificationConfig(repeats=2, batch_size=2),
        Backend(),
    ).run()

    qualified_root = tmp_path / "structured_semantic_qualified_v1"
    qualified = promote_structured_semantic_view(
        pending_root,
        source,
        qualification_root,
        qualified_root,
        allow_nonproduction_qualification=True,
    )

    assert pending.manifest["scientific_validation_scenario_ids"] == []
    assert qualified.manifest["scientific_validation_scenario_ids"] == [
        "validation.one"
    ]
    assert qualified.manifest["pending_base_known_validation_scenario_ids"] == []
    validation = next(row for row in qualified.scenarios if row["split"] == "validation")
    assert (validation["base_known_status"], validation["scientific_eligibility"]) == (
        "known",
        "eligible",
    )
    assert StructuredSemanticView.open(
        qualified_root,
        source_root=source,
        qualification_root=qualification_root,
        allow_nonproduction_qualification=True,
    ).manifest == qualified.manifest

    repeated = promote_structured_semantic_view(
        pending_root,
        source,
        qualification_root,
        qualified_root,
        allow_nonproduction_qualification=True,
        overwrite=False,
    )
    assert repeated.manifest == qualified.manifest


def test_build_does_not_open_unconfigured_test_or_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(tmp_path)
    (source / "test.jsonl").write_text("SEALED")
    (source / "audit.jsonl").write_text("SEALED")
    original = Path.open

    def guarded(path: Path, *args, **kwargs):
        if path.name in {"test.jsonl", "audit.jsonl"}:
            raise AssertionError("sealed source was opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    view = materialize_structured_semantic_view(source, tmp_path / "view")
    assert view.manifest["sealed_test_audit_policy"]["test_and_audit_opened"] is False


def test_missing_signal_and_family_leak_fail_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = source / "development_heldout_family.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()][:-1]
    rendered = "".join(_canonical(row) + "\n" for row in rows)
    path.write_text(rendered)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["outputs"]["development_heldout_family"]["sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
    (source / "manifest.json").write_text(_canonical(manifest) + "\n")
    with pytest.raises(StructuredSemanticError, match="exactly six"):
        materialize_structured_semantic_view(source, tmp_path / "missing")

    source = _source(tmp_path / "second")
    rows = [json.loads(line) for line in (source / "development_heldout_family.jsonl").read_text().splitlines()]
    for row in rows:
        row["family"] = row["split_group_id"] = "train_family"
    rendered = "".join(_canonical(row) + "\n" for row in rows)
    (source / "development_heldout_family.jsonl").write_text(rendered)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["outputs"]["development_heldout_family"]["sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
    (source / "manifest.json").write_text(_canonical(manifest) + "\n")
    with pytest.raises(StructuredSemanticError, match="family_id leaks"):
        materialize_structured_semantic_view(source, tmp_path / "leak")


def test_hash_tamper_and_semantic_target_tamper_fail_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "view"
    materialize_structured_semantic_view(source, output)
    scenarios = output / "scenarios.jsonl"
    scenarios.write_text(scenarios.read_text().replace("train.one true", "tampered", 1))
    with pytest.raises(StructuredSemanticError, match="scenario file identity"):
        StructuredSemanticView.open(output, source_root=source)


def test_source_substitution_and_d1_objective_are_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "view"
    materialize_structured_semantic_view(source, output)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["surprise"] = "substitution"
    (source / "manifest.json").write_text(_canonical(manifest) + "\n")
    with pytest.raises(StructuredSemanticError, match="source manifest identity"):
        StructuredSemanticView.open(output, source_root=source)

    source = _source(tmp_path / "d1")
    path = source / "train_behavior.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["objective"] = "d1_failed_reward_signal"
    rendered = "".join(_canonical(row) + "\n" for row in rows)
    path.write_text(rendered)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["outputs"]["train_behavior"]["sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
    (source / "manifest.json").write_text(_canonical(manifest) + "\n")
    with pytest.raises(StructuredSemanticError, match="unsupported objective"):
        materialize_structured_semantic_view(source, tmp_path / "d1-view")
