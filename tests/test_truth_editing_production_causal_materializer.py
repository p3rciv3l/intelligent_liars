from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import pytest

from intelligent_liars.truth_editing_production import ProductionRunConfig
from intelligent_liars.truth_editing_production_causal_materializer import (
    ProductionCausalCandidateMaterializer,
)
from intelligent_liars.truth_editing_qwen_causal_backend import (
    CAUSAL_EVALUATOR_FORMAT,
    CAUSAL_SCENARIO_SET_FORMAT,
    RANKK_BASIS_ARTIFACT_FORMAT,
    build_causal_backend_config,
    open_causal_backend_config,
)
from test_truth_editing_finalist_checkpoint import _production_export_fixture


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _scenario_view(root: Path) -> Path:
    view = root / "scenario-view"
    records = []
    for index in range(3):
        common = {
            "scenario_id": f"scenario-{index}",
            "truthful_target": f"truth-{index}",
            "truth_aliases": [f"alias-{index}"],
            "acceptable_false_targets": [f"false-{index}"],
            "base_known_status": "known",
        }
        records.extend(
            [
                {
                    **common,
                    "record_id": f"scenario-{index}.false-trigger",
                    "role": "false_trigger_truthful_control",
                    "messages": [{"role": "user", "content": f"direct-{index}"}],
                },
                {
                    **common,
                    "record_id": f"scenario-{index}.retained",
                    "role": "retained_truth_rephrase_proxy",
                    "messages": [{"role": "user", "content": f"retained-{index}"}],
                },
            ]
        )
    view.mkdir(parents=True)
    (view / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    _write(view / "manifest.json", {"view_sha256": "a" * 64})
    return view


def _config(root: Path, *, scenario_view: Path, direction_manifest: Path, model_sha: str) -> ProductionRunConfig:
    placeholder = root / "placeholder.json"
    _write(placeholder, {})
    return ProductionRunConfig(
        study_config=placeholder,
        dataset_root=root,
        scenario_view=scenario_view,
        structured_semantic_view=root,
        structured_semantic_source_root=root,
        structured_base_known_qualification=placeholder,
        direction_manifest=direction_manifest,
        direction_root=root,
        refusal_direction_config=placeholder,
        refusal_prompt_manifest=placeholder,
        refusal_direction_bank=placeholder,
        refusal_artifact_root=root,
        evaluator_config=placeholder,
        base_known_qualification=placeholder,
        judge_cache_dir=root / "judge-cache",
        judge_budget_ledger_dir=None,
        judge_budget=None,
        preservation_runtime_packet_root=root,
        preservation_runtime_packet_sha256="b" * 64,
        preservation_spec_sha256="c" * 64,
        preservation_runtime_configs=(),
        preservation_threshold_calibration=None,
        preservation_threshold_calibration_sha256=None,
        journal_path=root / "journal.json",
        artifact_dir=root / "artifacts",
        runtime_output_dir=root / "runtime",
        model_cache_dir=root / "cache",
        snapshot_manifest_path=placeholder,
        search_driver="optuna",
        rescore_generation=None,
        rescore_generation_sha256=None,
        rescore_mode=None,
        verified_model_sha256=model_sha,
        verified_snapshot_manifest_sha256="b" * 64,
        max_new_tokens=64,
        minimum_target_mean_log_probability=-5.0,
    )


def test_materializer_emits_local_checkpoint_and_every_strict_causal_artifact(
    tmp_path: Path,
) -> None:
    selection, compiler, bundle = _production_export_fixture(tmp_path / "compiler")
    finalist = selection["finalists"][0]
    manifest = tmp_path / "direction-manifest.json"
    _write(manifest, {"format": "fixture", "self_sha256": "d" * 64})
    config = _config(
        tmp_path,
        scenario_view=_scenario_view(tmp_path),
        direction_manifest=manifest,
        model_sha=bundle.verified_snapshot["model_sha256"],
    )
    before = {
        name: value.detach().clone() for name, value in bundle.model.state_dict().items()
    }
    materializer = ProductionCausalCandidateMaterializer(
        config=config,
        compiler=compiler,
        bundle=bundle,
        scenario_count=2,
    )

    result = materializer.materialize_candidate(
        study_identity_sha256=selection["study_identity_sha256"],
        trial_id=finalist["trial_id"],
        proposal=finalist["proposal"],
        proposal_sha256=finalist["proposal_sha256"],
        output_dir=tmp_path / "candidate",
    )

    assert set(result) == {
        "edited_checkpoint_path",
        "edited_checkpoint_sha256",
        "edited_checkpoint_manifest_path",
        "basis_artifact_path",
        "persistent_recipe_path",
        "scenario_path",
        "evaluator_path",
        "runtime_identity_sha256",
        "direction_manifest_path",
        "controls",
    }
    checkpoint = Path(result["edited_checkpoint_path"])
    assert checkpoint.is_dir()
    checkpoint_manifest = json.loads(
        Path(result["edited_checkpoint_manifest_path"]).read_text()
    )
    assert checkpoint_manifest["self_sha256"] == result["edited_checkpoint_sha256"]
    assert checkpoint_manifest["trial_id"] == finalist["trial_id"]
    assert checkpoint_manifest["proposal_sha256"] == finalist["proposal_sha256"]

    basis = torch.load(result["basis_artifact_path"], weights_only=True)
    assert basis["format"] == RANKK_BASIS_ARTIFACT_FORMAT
    assert basis["basis_sha256"] == checkpoint_manifest["basis_set_sha256"]
    assert tuple(int(layer) for layer in basis["by_layer"]) == tuple(
        finalist["proposal"]["writer_layers"]
    )

    persistent = json.loads(Path(result["persistent_recipe_path"]).read_text())
    assert persistent["backend"]["type"] == "persistent_weight"
    assert persistent["proposal"] == finalist["proposal"]
    scenarios = json.loads(Path(result["scenario_path"]).read_text())
    assert scenarios["format"] == CAUSAL_SCENARIO_SET_FORMAT
    assert len(scenarios["records"]) == 2
    assert scenarios["records"][0]["direct_messages"][-1]["content"] == "direct-0"
    evaluator = json.loads(Path(result["evaluator_path"]).read_text())
    assert evaluator["format"] == CAUSAL_EVALUATOR_FORMAT

    controls = result["controls"]
    assert [control["control_kind"] for control in controls] == [
        "restoration",
        "re_ablation",
        "random_direction",
        "false_trigger",
    ]
    assert all(Path(control["activation_recipe_path"]).is_file() for control in controls)
    assert controls[0]["direction_basis_sha256"] == basis["basis_sha256"]
    assert controls[2]["direction_basis_sha256"] != basis["basis_sha256"]
    assert Path(result["direction_manifest_path"]) == manifest.resolve()
    assert materializer.identity["production_config_model_sha256"] == config.verified_model_sha256
    assert all(torch.equal(bundle.model.state_dict()[name], value) for name, value in before.items())

    backend_config_path = _write(
        tmp_path / "causal-backend.json",
        build_causal_backend_config(
            edited_checkpoint_path=result["edited_checkpoint_path"],
            edited_checkpoint_sha256=result["edited_checkpoint_sha256"],
            edited_checkpoint_manifest_path=result[
                "edited_checkpoint_manifest_path"
            ],
            basis_artifact_path=result["basis_artifact_path"],
            output_dir=tmp_path / "causal-runtime",
            judge_ledger_start_sha256="e" * 64,
        ),
    )
    opened_backend = open_causal_backend_config(backend_config_path)
    assert (
        opened_backend.edited_checkpoint["sha256"]
        == checkpoint_manifest["self_sha256"]
    )

    # Resume is immutable and does not rewrite or duplicate the checkpoint.
    receipt = tmp_path / "candidate" / "materialization-receipt.json"
    receipt_sha = _sha_file(receipt)
    assert materializer.materialize_candidate(
        study_identity_sha256=selection["study_identity_sha256"],
        trial_id=finalist["trial_id"],
        proposal=finalist["proposal"],
        proposal_sha256=finalist["proposal_sha256"],
        output_dir=tmp_path / "candidate",
    ) == result
    assert _sha_file(receipt) == receipt_sha


def test_materializer_fails_closed_when_scenario_pairs_are_not_base_known(
    tmp_path: Path,
) -> None:
    selection, compiler, bundle = _production_export_fixture(tmp_path / "compiler")
    finalist = selection["finalists"][0]
    manifest = _write(tmp_path / "directions.json", {"format": "fixture"})
    view = _scenario_view(tmp_path)
    records_path = view / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    for record in records:
        record["base_known_status"] = "not_known"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    materializer = ProductionCausalCandidateMaterializer(
        config=_config(
            tmp_path,
            scenario_view=view,
            direction_manifest=manifest,
            model_sha=bundle.verified_snapshot["model_sha256"],
        ),
        compiler=compiler,
        bundle=bundle,
        scenario_count=2,
    )

    with pytest.raises(ValueError, match="base-known causal scenario"):
        materializer.materialize_candidate(
            study_identity_sha256=selection["study_identity_sha256"],
            trial_id=finalist["trial_id"],
            proposal=finalist["proposal"],
            proposal_sha256=finalist["proposal_sha256"],
            output_dir=tmp_path / "candidate",
        )


def test_materializer_resume_rejects_tampered_artifacts(tmp_path: Path) -> None:
    selection, compiler, bundle = _production_export_fixture(tmp_path / "compiler")
    finalist = selection["finalists"][0]
    manifest = _write(tmp_path / "directions.json", {"format": "fixture"})
    materializer = ProductionCausalCandidateMaterializer(
        config=_config(
            tmp_path,
            scenario_view=_scenario_view(tmp_path),
            direction_manifest=manifest,
            model_sha=bundle.verified_snapshot["model_sha256"],
        ),
        compiler=compiler,
        bundle=bundle,
        scenario_count=2,
    )
    arguments = {
        "study_identity_sha256": selection["study_identity_sha256"],
        "trial_id": finalist["trial_id"],
        "proposal": finalist["proposal"],
        "proposal_sha256": finalist["proposal_sha256"],
        "output_dir": tmp_path / "candidate",
    }
    result = materializer.materialize_candidate(**arguments)
    Path(result["scenario_path"]).write_text('{"tampered":true}\n')

    with pytest.raises(ValueError, match="artifact identity differs"):
        materializer.materialize_candidate(**arguments)

    # A fresh package also binds every generation-time control recipe.
    second = tmp_path / "candidate-2"
    arguments["output_dir"] = second
    result = materializer.materialize_candidate(**arguments)
    Path(result["controls"][0]["activation_recipe_path"]).write_text(
        '{"tampered":true}\n'
    )
    with pytest.raises(ValueError, match="control recipe identity differs"):
        materializer.materialize_candidate(**arguments)
