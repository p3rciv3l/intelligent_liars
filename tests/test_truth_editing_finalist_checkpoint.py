from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from intelligent_liars.models import ModelBundle, ModelLoadConfig
from intelligent_liars.truth_editing_finalist_checkpoint import (
    FinalistCheckpointError,
    FinalistCompilation,
    VerifiedFinalistCompiler,
    export_finalist_checkpoint,
    open_finalist_checkpoint,
    select_pareto_finalists,
)
from intelligent_liars.truth_editing_production import (
    GroupedV2Corpus,
    V2GroupedTrialBatchBuilder,
)
from intelligent_liars.truth_editing_study import SearchProposal
from intelligent_liars.truth_editing_weight_editor import (
    CompiledWriterEdit,
)


MODEL_SHA = "a" * 64
SNAPSHOT_SHA = "b" * 64
STUDY_SHA = "c" * 64


def _proposal(*, strength: float = 1.0) -> SearchProposal:
    return SearchProposal(
        direction_ids=("truth-general-layer-0",),
        direction_family="general",
        source_layer=0,
        basis_method="qr",
        requested_rank=1,
        writer_region="all",
        writer_layers=(0,),
        writer_policy="both",
        strength=strength,
        backend_type="persistent_weight",
        basis_scope="general",
        selected_domains=(),
        truth_direction_scope="global",
        normalization_mode="exact",
        edit_arm="truth_only",
        attention_enabled=True,
        attention_kernel_center=0.0,
        attention_kernel_half_width=0.0,
        attention_edge_strength=strength,
        attention_peak_strength=strength,
        mlp_enabled=True,
        mlp_kernel_center=0.0,
        mlp_kernel_half_width=0.0,
        mlp_edge_strength=strength,
        mlp_peak_strength=strength,
        refusal_enabled=False,
        refusal_direction_scope="global",
        refusal_source_layer=None,
        refusal_strength=0.0,
        refusal_writer_policy="both",
        proposal_origin="tpe_sampled",
    )


def _report(rows: list[tuple[str, tuple[float, float, float]]]) -> dict:
    trials = []
    for ordinal, (trial_id, values) in enumerate(rows):
        trials.append(
            {
                "trial_id": trial_id,
                "ordinal": ordinal,
                "batch_ordinal": 0,
                "tier_name": "finalist",
                "evaluation_record_ids": ["record-1"],
                "proposal": _proposal(strength=1.0 + ordinal / 10).to_dict(),
                "result": {
                    "outcome_kind": "successful",
                    "metrics": {
                        "valid_false_report_rate_lcb": values[0],
                        "truth_report_dissociation_lcb": values[1],
                        "capability_preservation_lcb": values[2],
                    },
                    "detail": None,
                },
            }
        )
    return {
        "format": "truth_editing_study_report_v1",
        "study_identity_sha256": STUDY_SHA,
        "completed_trials": len(trials),
        "successful_trials": len(trials),
        "scientifically_infeasible_trials": 0,
        "operational_failures": 0,
        "coverage": {
            "families": ["general"],
            "source_layers": [0],
            "writer_regions": ["all"],
            "writer_policies": ["both"],
            "basis_methods": ["qr"],
            "strength_regions": ["projection"],
            "basis_scopes": ["general"],
            "direction_scopes": ["global"],
            "normalization_modes": ["exact"],
            "edit_arms": ["truth_only"],
        },
        "coverage_complete": True,
        "selection_ready": True,
        "trials": trials,
    }


def _bound_selection(
    report: dict, *, compiler_identity: dict | None = None
) -> dict:
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    receipt_unsigned = {
        "format": "truth_editing_study_artifact_receipt_v1",
        "study_identity_sha256": report["study_identity_sha256"],
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "report_path": "/verified/study-report.json",
    }
    receipt = {
        **receipt_unsigned,
        "receipt_sha256": hashlib.sha256(
            json.dumps(receipt_unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return select_pareto_finalists(
        report,
        study_artifact_receipt=receipt,
        report_bytes=report_bytes,
        expected_compiler_identity=compiler_identity
        or {"adapter": "test_verified_compiler_v1", "bank_sha256": "e" * 64},
    )


def test_selects_only_nondominated_trials_and_schedules_matched_controls() -> None:
    report = _report(
        [
            ("trial-a", (0.9, 0.6, 0.7)),
            ("trial-b", (0.7, 0.9, 0.8)),
            ("trial-dominated", (0.6, 0.5, 0.7)),
        ]
    )

    receipt = select_pareto_finalists(report)

    assert [item["trial_id"] for item in receipt["finalists"]] == [
        "trial-a",
        "trial-b",
    ]
    assert len(receipt["control_schedule"]) == 4
    assert {item["control_kind"] for item in receipt["control_schedule"]} == {
        "orthogonal",
        "shuffled",
    }
    assert receipt["chosen_finalist_trial_id"] == "trial-b"
    assert receipt["chosen_finalist_policy"] == (
        "maximize_worst_objective_then_capability_v1"
    )
    assert receipt["chosen_finalist_status"] == "provisional_pending_controls"
    assert receipt["control_execution_status"] == "scheduled_not_executed"
    assert receipt["self_sha256"] == select_pareto_finalists(report)["self_sha256"]


def test_chosen_finalist_is_balanced_deterministic_and_order_independent() -> None:
    report = _report(
        [
            ("false-report-heavy", (0.99, 0.51, 0.72)),
            ("balanced", (0.82, 0.80, 0.81)),
            ("capability-heavy", (0.71, 0.72, 0.99)),
        ]
    )

    first = select_pareto_finalists(report)
    report["trials"].reverse()
    second = select_pareto_finalists(report)

    assert first["chosen_finalist_trial_id"] == "balanced"
    assert second["chosen_finalist_trial_id"] == "balanced"


def test_pareto_selection_never_compares_lower_fidelity_tiers() -> None:
    report = _report(
        [
            ("discovery-perfect", (1.0, 1.0, 1.0)),
            ("finalist-a", (0.7, 0.8, 0.9)),
            ("finalist-b", (0.8, 0.7, 0.9)),
        ]
    )
    report["trials"][0]["tier_name"] = "discovery"
    report["trials"][1]["evaluation_record_ids"] = ["record-1", "record-2"]
    report["trials"][2]["evaluation_record_ids"] = ["record-1", "record-2"]

    receipt = select_pareto_finalists(report)

    assert [item["trial_id"] for item in receipt["finalists"]] == [
        "finalist-a",
        "finalist-b",
    ]
    assert receipt["selection_tier_name"] == "finalist"
    assert receipt["selection_record_count"] == 2


def test_selection_accepts_audited_failure_after_exact_scored_replay() -> None:
    report = _report([("trial-replay", (0.9, 0.6, 0.7))])
    successful = report["trials"][0]
    failed = json.loads(json.dumps(successful))
    failed["trial_id"] = "trial-failed"
    failed["ordinal"] = 0
    failed["result"] = {
        "outcome_kind": "operational_failure",
        "metrics": {},
        "detail": "judge transport timeout",
    }
    successful["ordinal"] = 1
    report["trials"] = [failed, successful]
    report["completed_trials"] = 2
    report["operational_failures"] = 1

    receipt = select_pareto_finalists(report)

    assert [item["trial_id"] for item in receipt["finalists"]] == ["trial-replay"]


def test_selection_rejects_unresolved_audited_failure() -> None:
    report = _report([("trial-success", (0.9, 0.6, 0.7))])
    failed = json.loads(json.dumps(report["trials"][0]))
    failed["trial_id"] = "trial-failed"
    failed["ordinal"] = 1
    failed["proposal"] = _proposal(strength=0.25).to_dict()
    failed["result"] = {
        "outcome_kind": "operational_failure",
        "metrics": {},
        "detail": "judge transport timeout",
    }
    report["trials"].append(failed)
    report["completed_trials"] = 2
    report["operational_failures"] = 1

    with pytest.raises(FinalistCheckpointError, match="unresolved"):
        select_pareto_finalists(report)


def test_selection_fails_closed_on_unready_or_tampered_report(tmp_path: Path) -> None:
    report = _report([("trial-a", (0.9, 0.6, 0.7))])
    report["selection_ready"] = False
    with pytest.raises(FinalistCheckpointError, match="selection-ready"):
        select_pareto_finalists(report)

    report = _report([("trial-a", (0.9, 0.6, 0.7))])
    report["trials"][0]["result"]["metrics"]["extra"] = 1.0
    with pytest.raises(FinalistCheckpointError, match="objective metrics"):
        select_pareto_finalists(report)

    selection = _bound_selection(
        _report([("trial-a", (0.9, 0.6, 0.7))])
    )
    selection["finalists"][0]["unknown"] = "injected"
    unsigned = dict(selection)
    unsigned.pop("self_sha256")
    selection["self_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(FinalistCheckpointError, match="fields differ"):
        export_finalist_checkpoint(
            selection_receipt=selection,
            trial_id="trial-a",
            compiler=_Compiler(FinalistCompilation(
                "trial-a",
                "0" * 64,
                "1" * 64,
                CompiledWriterEdit("recipe", MODEL_SHA, ()),
            )),
            bundle=_bundle(),
            output_dir=tmp_path / "unused",
            registry_bucket="private-models-example",
            model_slug="final",
        )


class _Writer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(width))


class _Layer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = _Writer(width)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = _Writer(width)


class _LanguageModel(torch.nn.Module):
    def __init__(self, layer_count: int, width: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(width) for _ in range(layer_count)])


class _Inner(torch.nn.Module):
    def __init__(self, layer_count: int, width: int) -> None:
        super().__init__()
        self.language_model = _LanguageModel(layer_count, width)


class _SaveableQwen(torch.nn.Module):
    def __init__(self, layer_count: int, width: int) -> None:
        super().__init__()
        self.model = _Inner(layer_count, width)

    def save_pretrained(self, path: Path, **kwargs: object) -> None:
        assert kwargs["safe_serialization"] is True
        from safetensors.torch import save_file

        save_file(
            {name: tensor.detach() for name, tensor in self.state_dict().items()},
            Path(path) / "model.safetensors",
        )
        (Path(path) / "config.json").write_text('{"model_type":"qwen3_vl"}\n')


class _Processor:
    def save_pretrained(self, path: Path) -> None:
        for name in (
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ):
            (Path(path) / name).write_text("{}\n")


class _Compiler:
    def __init__(self, compilation: FinalistCompilation) -> None:
        self.compilation = compilation

    @property
    def identity(self) -> dict[str, str]:
        return {"adapter": "test_verified_compiler_v1", "bank_sha256": "e" * 64}

    def compile_finalist(
        self, proposal: SearchProposal, *, trial_id: str
    ) -> FinalistCompilation:
        del proposal
        assert trial_id == self.compilation.trial_id
        return self.compilation


class _WrongCompiler(_Compiler):
    @property
    def identity(self) -> dict[str, str]:
        return {"adapter": "wrong_compiler_v1", "bank_sha256": "f" * 64}


def _bundle(
    *, model_sha256: str = MODEL_SHA, layer_count: int = 1, width: int = 2
) -> ModelBundle:
    processor = _Processor()
    return ModelBundle(
        model=_SaveableQwen(layer_count, width),
        processor=processor,
        tokenizer=processor,
        model_id="Qwen/Qwen3-VL-8B-Thinking",
        config=ModelLoadConfig(),
        verified_snapshot={
            "model_id": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
            "model_sha256": model_sha256,
            "snapshot_manifest_sha256": SNAPSHOT_SHA,
        },
    )


def _production_export_fixture(
    tmp_path: Path,
) -> tuple[dict, VerifiedFinalistCompiler, ModelBundle]:
    from test_truth_editing_directions import _open_fixture_bank

    bank = _open_fixture_bank(tmp_path / "bank")
    entry = next(
        item for item in bank.manifest.directions if item.family == "domain_specific"
    )
    proposal = SearchProposal(
        direction_ids=(entry.direction_id,),
        direction_family="domain_specific",
        source_layer=entry.source_layer,
        basis_method="qr",
        requested_rank=1,
        writer_region="all",
        writer_layers=(0, 1, 2),
        writer_policy="both",
        strength=1.0,
        selected_domains=tuple(
            sorted(domain for domain in entry.domains if domain not in {"general", "all"})
        ),
    )
    report = _report([("trial-a", (0.9, 0.6, 0.7))])
    report["trials"][0]["proposal"] = proposal.to_dict()
    builder = V2GroupedTrialBatchBuilder(
        corpus=GroupedV2Corpus((), "f" * 64),
        direction_bank=bank,
        model_sha256=bank.manifest.model.model_sha256,
        max_new_tokens=8,
    )
    compiler = VerifiedFinalistCompiler(builder)
    selection = _bound_selection(report, compiler_identity=dict(compiler.identity))
    bundle = _bundle(
        model_sha256=bank.manifest.model.model_sha256,
        layer_count=bank.manifest.model.decoder_layer_count,
        width=bank.manifest.model.hidden_width,
    )
    return selection, compiler, bundle


def test_exports_and_reopens_atomic_immutable_edited_checkpoint(tmp_path: Path) -> None:
    selection, compiler, bundle = _production_export_fixture(tmp_path)
    original = copy.deepcopy(bundle.model.state_dict())

    with pytest.raises(FinalistCheckpointError, match="concrete verified"):
        export_finalist_checkpoint(
            selection_receipt=selection,
            trial_id="trial-a",
            compiler=_WrongCompiler(
                FinalistCompilation(
                    "trial-a",
                    selection["finalists"][0]["proposal_sha256"],
                    "d" * 64,
                    CompiledWriterEdit("recipe", MODEL_SHA, ()),
                )
            ),  # type: ignore[arg-type]
            bundle=bundle,
            output_dir=tmp_path / "wrong-compiler",
            registry_bucket="private-models-example",
            model_slug="qwen3-vl-8b-truth-edited",
        )

    result = export_finalist_checkpoint(
        selection_receipt=selection,
        trial_id="trial-a",
        compiler=compiler,
        bundle=bundle,
        output_dir=tmp_path / "published",
        registry_bucket="private-models-example",
        registry_base_prefix="model-registry/v1",
        model_slug="qwen3-vl-8b-truth-edited",
    )

    reopened = open_finalist_checkpoint(tmp_path / "published")
    assert reopened == result
    assert result["manifest"]["base_model"]["model_sha256"] == (
        bundle.verified_snapshot["model_sha256"]
    )
    assert result["manifest"]["trial_id"] == "trial-a"
    assert result["selection_receipt"]["chosen_finalist_trial_id"] == "trial-a"
    assert result["selection_receipt"]["chosen_finalist_status"] == (
        "provisional_pending_controls"
    )
    assert result["registry_entry_proposal"]["status"] == (
        "finalist_candidate_not_uploaded"
    )
    assert result["control_schedule_receipt"]["controls"]
    assert result["control_schedule_receipt"]["status"] == (
        "scheduled_not_executed"
    )
    assert all(
        torch.equal(bundle.model.state_dict()[name], tensor)
        for name, tensor in original.items()
    )
    with pytest.raises(FileExistsError):
        export_finalist_checkpoint(
            selection_receipt=selection,
            trial_id="trial-a",
            compiler=compiler,
            bundle=bundle,
            output_dir=tmp_path / "published",
            registry_bucket="private-models-example",
            registry_base_prefix="model-registry/v1",
            model_slug="qwen3-vl-8b-truth-edited",
        )


def test_export_rejects_pareto_member_that_is_not_the_chosen_finalist(
    tmp_path: Path,
) -> None:
    selection, compiler, bundle = _production_export_fixture(tmp_path)
    other = copy.deepcopy(selection["finalists"][0])
    other["trial_id"] = "trial-other"
    other["ordinal"] = 99
    other["metrics"] = {
        "valid_false_report_rate_lcb": 0.99,
        "truth_report_dissociation_lcb": 0.1,
        "capability_preservation_lcb": 0.99,
    }
    selection["finalists"].append(other)
    for original_control in selection["control_schedule"][:2]:
        cloned_control = copy.deepcopy(original_control)
        cloned_control["finalist_trial_id"] = "trial-other"
        body = dict(cloned_control)
        body.pop("control_id")
        cloned_control["control_id"] = "control-" + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        selection["control_schedule"].append(cloned_control)
    unsigned = dict(selection)
    unsigned.pop("self_sha256")
    selection["self_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(FinalistCheckpointError, match="chosen finalist"):
        export_finalist_checkpoint(
            selection_receipt=selection,
            trial_id="trial-other",
            compiler=compiler,
            bundle=bundle,
            output_dir=tmp_path / "not-chosen",
            registry_bucket="private-models-example",
            model_slug="qwen3-vl-8b-truth-edited",
        )


def test_open_fails_closed_on_inventory_tampering(tmp_path: Path) -> None:
    selection, compiler, bundle = _production_export_fixture(tmp_path)
    export_finalist_checkpoint(
        selection_receipt=selection,
        trial_id="trial-a",
        compiler=compiler,
        bundle=bundle,
        output_dir=tmp_path / "published",
        registry_bucket="private-models-example",
        model_slug="qwen3-vl-8b-truth-edited",
    )
    (tmp_path / "published" / "checkpoint" / "config.json").write_text("tampered")

    with pytest.raises(FinalistCheckpointError, match="unreadable|identity differs"):
        open_finalist_checkpoint(tmp_path / "published")


def test_open_rejects_relabeling_unrun_controls_as_executed(tmp_path: Path) -> None:
    selection, compiler, bundle = _production_export_fixture(tmp_path)
    export_finalist_checkpoint(
        selection_receipt=selection,
        trial_id="trial-a",
        compiler=compiler,
        bundle=bundle,
        output_dir=tmp_path / "published",
        registry_bucket="private-models-example",
        model_slug="qwen3-vl-8b-truth-edited",
    )
    receipt_path = tmp_path / "published" / "control-schedule-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "executed"
    unsigned = dict(receipt)
    unsigned.pop("self_sha256")
    receipt["self_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(FinalistCheckpointError, match="control schedule status"):
        open_finalist_checkpoint(tmp_path / "published")


def test_select_cli_materializes_immutable_receipt_without_loading_a_model(
    tmp_path: Path,
) -> None:
    report = tmp_path / "study-report.json"
    report_payload = _report([("trial-a", (0.9, 0.6, 0.7))])
    report.write_text(json.dumps(report_payload))
    receipt_unsigned = {
        "format": "truth_editing_study_artifact_receipt_v1",
        "study_identity_sha256": STUDY_SHA,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "report_path": str(report),
    }
    receipt = tmp_path / "study-artifact-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                **receipt_unsigned,
                "receipt_sha256": hashlib.sha256(
                    json.dumps(
                        receipt_unsigned, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
        )
    )
    output = tmp_path / "selection.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_truth_editing_finalist_checkpoint.py",
            "select",
            "--study-report",
            str(report),
            "--study-artifact-receipt",
            str(receipt),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["format"] == (
        "truth_editing_pareto_finalist_selection_v2"
    )
    assert "transformers" not in result.stderr
