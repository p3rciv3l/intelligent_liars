from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_base_known import (
    BaseKnownQualification,
    BaseKnownRunner,
    QualificationConfig,
    QualificationResponse,
)
from intelligent_liars.truth_editing_scenario_view import (
    ScenarioViewError,
    TruthEditingScenarioView,
    materialize_validation_scenario_view,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "v2"
    source.mkdir()
    rows = [
        {
            "format": "truth_editing_canonical_qa_record_v2",
            "record_id": "qa_one",
            "canonical_key": "one",
            "collision_cluster_id": "cluster_one",
            "question": "What is 2 + 2?",
            "correct_answer": "4",
            "choices": ["3", "4", "5"],
            "wrong_answers": ["3", "5"],
            "family": "arithmetic",
            "truth_authority": "fixture",
            "split": "validation",
        },
        {
            "format": "truth_editing_canonical_qa_record_v2",
            "record_id": "qa_two",
            "canonical_key": "two",
            "collision_cluster_id": "cluster_two",
            "question": "Which city is in France?",
            "correct_answer": "Paris",
            "choices": ["Paris", "Rome"],
            "wrong_answers": ["Rome"],
            "family": "geography",
            "truth_authority": "fixture",
            "split": "validation",
        },
    ]
    (source / "validation.jsonl").write_text("".join(_canonical(row) + "\n" for row in rows))
    validation_sha = hashlib.sha256((source / "validation.jsonl").read_bytes()).hexdigest()
    manifest = {
        "format": "truth_editing_canonical_qa_manifest_v2",
        "dataset_id": "fixture-v2",
        "seed": 1,
        "source_candidate_count": 2,
        "accepted_canonical_count": 2,
        "quarantined_candidate_count": 0,
        "split_counts": {"train": 0, "validation": 2, "test": 0},
        "source_counts": {"fixture": 2},
        "file_sha256": {"validation.jsonl": validation_sha},
        "policy_sha256": "0" * 64,
    }
    (source / "manifest.json").write_text(_canonical(manifest) + "\n")
    return source


def test_materializes_validation_only_qa_scenarios_without_fabricated_actions(
    tmp_path: Path,
) -> None:
    view = materialize_validation_scenario_view(
        _source(tmp_path), tmp_path / "view", tier_scenario_limits=(1, 2)
    )

    assert len(view.records) == 6
    assert {row["role"] for row in view.records} == {
        "direct_false_report_instruction",
        "false_trigger_truthful_control",
        "retained_truth_rephrase_proxy",
    }
    assert {row["lane"] for row in view.records} == {
        "qa_only",
        "capability_preservation",
    }
    assert all(row["base_known_status"] == "unknown" for row in view.records)
    assert all(row["scientific_use"] == "blocked" for row in view.records)
    assert not any(row["role"] in {"action", "counterfactual"} for row in view.records)
    direct = next(
        row
        for row in view.records
        if row["source_record_id"] == "qa_one"
        and row["role"] == "direct_false_report_instruction"
    )
    assert direct["truthful_target"] == "4"
    assert direct["acceptable_false_targets"] == ["3", "5"]
    assert direct["false_target_policy"] == "any_listed_wrong_answer"
    assert "deliberately give a false answer" in direct["messages"][0]["content"].casefold()
    assert "- 3\n- 4\n- 5" in direct["messages"][1]["content"]
    false_trigger = next(
        row
        for row in view.records
        if row["source_record_id"] == "qa_one"
        and row["role"] == "false_trigger_truthful_control"
    )
    assert "confidential" in false_trigger["messages"][1]["content"]
    indirect = next(
        row
        for row in view.records
        if row["source_record_id"] == "qa_one"
        and row["role"] == "retained_truth_rephrase_proxy"
    )
    assert "previous direct answer was '3'" in indirect["messages"][1]["content"].casefold()
    assert view.manifest["scientific_validation_record_ids"] == []
    assert len(view.manifest["software_validation_record_ids"]) == 6
    assert len(view.manifest["control_validation_record_ids"]) == 2
    assert len(view.manifest["proxy_validation_record_ids"]) == 2
    assert [len(tier["software_record_ids"]) for tier in view.manifest["tier_candidates"]] == [3, 6]
    assert [len(tier["control_record_ids"]) for tier in view.manifest["tier_candidates"]] == [1, 2]
    assert [len(tier["proxy_record_ids"]) for tier in view.manifest["tier_candidates"]] == [1, 2]


def test_round_trip_is_deterministic_and_binds_source_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first = materialize_validation_scenario_view(source, tmp_path / "a", tier_scenario_limits=(1, 2))
    second = materialize_validation_scenario_view(source, tmp_path / "b", tier_scenario_limits=(1, 2))

    assert first.manifest == second.manifest
    reopened = TruthEditingScenarioView.open(tmp_path / "a", source_dataset=source)
    assert reopened.manifest == first.manifest
    expected = hashlib.sha256(_canonical(json.loads((source / "manifest.json").read_text())).encode()).hexdigest()
    assert first.manifest["source_dataset_manifest_sha256"] == expected


def test_open_fails_closed_on_record_tamper_or_unknown_manifest_field(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "view"
    materialize_validation_scenario_view(source, output, tier_scenario_limits=(1, 2))
    path = output / "records.jsonl"
    path.write_text(path.read_text().replace("What is 2 + 2?", "What is 2 + 3?"))
    with pytest.raises(ScenarioViewError, match="records content hash"):
        TruthEditingScenarioView.open(output, source_dataset=source)

    materialize_validation_scenario_view(source, output, tier_scenario_limits=(1, 2), overwrite=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["surprise"] = True
    manifest_path.write_text(_canonical(manifest))
    with pytest.raises(ScenarioViewError, match="manifest fields"):
        TruthEditingScenarioView.open(output, source_dataset=source)


def test_recomputed_hashes_cannot_hide_semantic_record_tamper(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "view"
    materialize_validation_scenario_view(source, output, tier_scenario_limits=(1, 2))
    records_path = output / "records.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    direct = next(
        row for row in rows if row["role"] == "direct_false_report_instruction"
    )
    direct["teacher_forcing_target"] = direct["truthful_target"]
    records_path.write_text("".join(_canonical(row) + "\n" for row in rows))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"]["records.jsonl"] = hashlib.sha256(
        records_path.read_bytes()
    ).hexdigest()
    manifest.pop("view_sha256")
    manifest["view_sha256"] = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    manifest_path.write_text(_canonical(manifest))

    with pytest.raises(ScenarioViewError, match="target differs|canonical derivation"):
        TruthEditingScenarioView.open(output, source_dataset=source)


def test_recomputed_manifest_hash_cannot_hide_tier_control_omission(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "view"
    materialize_validation_scenario_view(source, output, tier_scenario_limits=(1, 2))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tier_candidates"][0]["control_record_ids"] = []
    manifest.pop("view_sha256")
    manifest["view_sha256"] = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    manifest_path.write_text(_canonical(manifest))

    with pytest.raises(ScenarioViewError, match="tier role membership"):
        TruthEditingScenarioView.open(output, source_dataset=source)


def test_source_manifest_substitution_is_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "view"
    materialize_validation_scenario_view(source, output, tier_scenario_limits=(1, 2))
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["dataset_id"] = "substituted"
    (source / "manifest.json").write_text(_canonical(manifest))

    with pytest.raises(ScenarioViewError, match="source dataset manifest identity"):
        TruthEditingScenarioView.open(output, source_dataset=source)


def test_mock_base_known_override_enables_software_fixture_but_not_full_scenario(
    tmp_path: Path,
) -> None:
    view = materialize_validation_scenario_view(
        _source(tmp_path),
        tmp_path / "view",
        tier_scenario_limits=(1, 2),
        base_known_by_source_record={"qa_one": True, "qa_two": False},
        qualification_receipt_sha256="a" * 64,
        qualification_mode="synthetic_mock_only",
    )
    one = [row for row in view.records if row["source_record_id"] == "qa_one"]
    two = [row for row in view.records if row["source_record_id"] == "qa_two"]
    assert {row["base_known_status"] for row in one} == {"known"}
    assert {row["base_known_status"] for row in two} == {"not_known"}
    assert all(row["scientific_use"] == "blocked" for row in one + two)
    assert "synthetic_mock_qualification" in one[0]["scientific_blockers"]
    assert "missing_structured_action_and_counterfactual" in one[0]["scenario_scientific_blockers"]


def test_rejects_mock_receipt_and_consumes_verified_loader_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    evidence = tmp_path / "qualification"
    model = {
        "repository": "Qwen/Qwen3-VL-8B-Thinking",
        "revision": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
        "model_sha256": "b" * 64,
        "tokenizer_sha256": "8" * 64,
        "chat_template_sha256": "f" * 64,
        "inference_backend": "transformers",
        "dtype": "torch.bfloat16",
        "attention_implementation": "flash_attention_2",
        "device_map": "cuda:0",
        "local_files_only": True,
        "use_cache": True,
    }

    class Backend:
        def generate_batch(self, requests, identity):
            del identity
            result = []
            for request in requests:
                answer = "4" if request.record_id == "qa_one" else "Rome"
                label = request.labels[request.ordered_choices.index(answer)]
                result.append(QualificationResponse(request.request_id, label, (1,)))
            return tuple(result)

    result = BaseKnownRunner(
        source,
        evidence,
        model,
        QualificationConfig(batch_size=4),
        Backend(),
    ).run()

    with pytest.raises(ScenarioViewError, match="base-known qualification is invalid"):
        materialize_validation_scenario_view(
            source,
            tmp_path / "rejected",
            tier_scenario_limits=(1, 2),
            base_known_qualification=evidence,
        )

    verified_shape = BaseKnownQualification.open(evidence, allow_nonproduction=True)
    monkeypatch.setattr(
        BaseKnownQualification,
        "open",
        classmethod(lambda cls, path: verified_shape),
    )
    view = materialize_validation_scenario_view(
        source, tmp_path / "view", tier_scenario_limits=(1, 2),
        base_known_qualification=evidence,
    )
    assert view.manifest["qualification_mode"] == "frozen_base_model_validation"
    assert view.manifest["qualification_receipt_sha256"] == result.manifest_sha256
    assert view.manifest["base_known_counts"] == {"known": 3, "not_known": 3, "unknown": 0}
    assert view.manifest["scientific_validation_record_ids"] == []
    known_rows = [row for row in view.records if row["source_record_id"] == "qa_one"]
    assert all(row["scientific_use"] == "blocked" for row in known_rows)
    assert all(row["scenario_completeness"] == "qa_only_incomplete" for row in known_rows)
