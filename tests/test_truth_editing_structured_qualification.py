from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_base_known import (
    BaseKnownError,
    QualificationConfig,
    QualificationResponse,
    validate_backend_batch_execution,
)
from intelligent_liars.truth_editing_structured_qualification import (
    StructuredSemanticQualification,
    StructuredSemanticQualificationRunner,
)

from test_truth_editing_base_known_qualification import MODEL


def _view(root: Path) -> tuple[Path, Path]:
    import hashlib

    from intelligent_liars.truth_editing_structured_semantic import materialize_structured_semantic_view

    source = root.parent / "source"
    source.mkdir()
    specs = (
        ("truthful_direct_report", "neutral", "neutral_report", "Paris", "Rome", "paris"),
        ("conditioned_false_direct_report", "target", "target_report", "Rome", "Paris", "paris"),
        ("truthful_false_trigger_control", "false_trigger_control", "false_trigger_truthful_report", "Paris", "Rome", "paris"),
        ("true_state_indirect_answer_after_false_report", "target_after_report_indirect_probe", "post_report_indirect_truth_probe", "Paris", "Rome", "paris"),
        ("true_state_action_after_false_report", "target_after_report", "post_report_true_state_action", "go_paris", "go_rome", "paris"),
        ("counterfactual_state_action", "counterfactual", "counterfactual_action", "go_rome", "go_paris", "rome"),
    )
    source_rows = [
        {
            "format": "tinylora_pilot_example_v1", "kind": "behavior",
            "record_id": f"src.{objective}", "scenario_id": "scenario.one",
            "family": "travel", "split_group_id": "travel",
            "split": "development_heldout_family", "objective": objective,
            "condition_type": condition, "variant": variant,
            "prompt": f"Prompt for {objective}", "target": target,
            "alternative_target": alternative, "world_state_value": world,
            "source": {}, "provenance": {}, "risk_level": "low",
            **({"alternative_provenance": "curated_prompt_matched_counterfactual"} if objective == "true_state_indirect_answer_after_false_report" else {}),
        }
        for objective, condition, variant, target, alternative, world in specs
    ]
    source_validation = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in source_rows)
    (source / "train_behavior.jsonl").write_text("")
    (source / "development_heldout_family.jsonl").write_text(source_validation)
    source_manifest = {
        "outputs": {
            "train_behavior": {"path": "train_behavior.jsonl", "sha256": hashlib.sha256(b"").hexdigest()},
            "development_heldout_family": {"path": "development_heldout_family.jsonl", "sha256": hashlib.sha256(source_validation.encode()).hexdigest()},
        }
    }
    (source / "manifest.json").write_text(json.dumps(source_manifest))
    materialize_structured_semantic_view(source, root)
    return root, source


class Backend:
    def generate_batch(self, requests, identity):
        del identity
        return tuple(QualificationResponse(request.request_id, request.labels[request.ordered_choices.index(request.correct_answer)]) for request in requests)


def test_structured_direct_retained_action_signals_roundtrip(tmp_path: Path) -> None:
    view, source = _view(tmp_path / "view")
    result = StructuredSemanticQualificationRunner(
        view, source, tmp_path / "out", MODEL,
        QualificationConfig(repeats=2, batch_size=2), Backend(),
    ).run()
    assert len(result.signals) == 3
    assert {row.signal_kind for row in result.signals} == {"truthful_direct_report", "indirect_retained_truth", "true_state_action"}
    assert all(row.request_count == 4 and row.base_known for row in result.signals)
    assert result.scenarios[0].all_required_known is True
    opened = StructuredSemanticQualification.open(tmp_path / "out", view, source, allow_nonproduction=True)
    assert opened.manifest_sha256 == result.manifest_sha256


def test_structured_test_or_source_tamper_fails_closed(tmp_path: Path) -> None:
    view, source = _view(tmp_path / "view")
    runner = StructuredSemanticQualificationRunner(view, source, tmp_path / "out", MODEL, QualificationConfig(), Backend())
    runner.run()
    with (view / "scenarios.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(BaseKnownError, match="source"):
        runner.run()


def test_mock_cannot_self_assert_verified_production_evidence(tmp_path: Path) -> None:
    class Forged(Backend):
        def evidence_receipt(self, identity):
            del identity
            return {"mode": "verified_frozen_qwen", "receipt_sha256": "f" * 64}

    view, source = _view(tmp_path / "view")
    with pytest.raises(BaseKnownError, match="self-assert"):
        StructuredSemanticQualificationRunner(
            view, source, tmp_path / "out", MODEL, QualificationConfig(), Forged()
        ).run()


def test_raw_response_and_frozen_identity_tamper_fail_closed(tmp_path: Path) -> None:
    import hashlib

    view, source = _view(tmp_path / "view")
    out = tmp_path / "out"
    StructuredSemanticQualificationRunner(
        view, source, out, MODEL, QualificationConfig(), Backend()
    ).run()
    response_path = next((out / "batches").glob("batch_*/responses.json"))
    original = response_path.read_text()
    response_path.write_text("[]")
    with pytest.raises(BaseKnownError, match="batch evidence"):
        StructuredSemanticQualification.open(out, view, source, allow_nonproduction=True)
    response_path.write_text(original)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["revision"] = "0" * 40
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    manifest["self_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BaseKnownError, match="checkpoint"):
        StructuredSemanticQualification.open(out, view, source, allow_nonproduction=True)


def test_production_mode_requires_batch_bound_execution_receipt() -> None:
    with pytest.raises(BaseKnownError, match="batch execution"):
        validate_backend_batch_execution(
            None,
            {"mode": "verified_frozen_qwen", "receipt_sha256": "f" * 64},
            [],
            [],
        )


def test_selective_result_omission_fails_exact_inventory(tmp_path: Path) -> None:
    import hashlib

    view, source = _view(tmp_path / "view")
    out = tmp_path / "out"
    StructuredSemanticQualificationRunner(view, source, out, MODEL, QualificationConfig(), Backend()).run()
    signal_path = out / "signals.jsonl"
    rows = signal_path.read_text().splitlines()[1:]
    rendered = ("\n".join(rows) + "\n").encode()
    signal_path.write_bytes(rendered)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["signal_count"] -= 1
    manifest["signals_sha256"] = hashlib.sha256(rendered).hexdigest()
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    manifest["self_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BaseKnownError, match="signal inventory"):
        StructuredSemanticQualification.open(out, view, source, allow_nonproduction=True)
