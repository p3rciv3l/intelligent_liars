from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from intelligent_liars.interventions import (
    DirectionMode,
    EXPECTED_DIRECTION_SIGN,
    ProbeDirection,
    seeded_orthogonal_direction,
    transform_activations,
)
from intelligent_liars.step5_intervention_experiments import (
    ARM_IDS,
    ARM_VARIANTS,
    CONTRACT_FORMAT,
    CONTRACT_STATUS,
    DIAGNOSTIC_OBJECTIVES,
    EVALUATION_SCOPE,
    EXECUTION_SEEDS,
    SEMANTIC_MANIFEST_MODE,
    TARGET_DIAGNOSTIC_SEMANTICS,
    TEACHER_FORCED_MASK_SEMANTICS,
    MaskedTeacherForcedIntervention,
    PairObservation,
    build_result_payload,
    canonical_sha256,
    derive_c1_control_seed,
    diagnostic_ordered_identity_sha256,
    intervention_bundle_for,
    load_contract_probe_direction,
    load_experiment_contract,
    pair_metrics,
    parse_experiment_contract,
    quantization_trace,
    validate_terminal_result,
    validate_finalized_execution_inputs,
    write_atomic_self_hashed_json,
)
from intelligent_liars.step5_intervention_experiments.results import (
    build_intervention_trace,
    trace_expectation,
    verify_intervention_trace,
)


def _unsigned_contract() -> dict[str, object]:
    record_ids = (
        "target",
        "control1",
        "control2",
        "control3",
        "control4",
        "control5",
    )
    source_vector = [2.0, 0.0]
    source_vector_sha256 = canonical_sha256(source_vector)
    return {
        "format": CONTRACT_FORMAT,
        "suite_id": "unit-intervention-suite",
        "status": CONTRACT_STATUS,
        "evaluation_scope": EVALUATION_SCOPE,
        "authorized_to_execute": False,
        "model_identity": {
            "repository": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "1" * 40,
            "content_sha256": "2" * 64,
            "layer": 1,
        },
        "probe_identity": {
            "path": "probe.json",
            "sha256": "3" * 64,
            "vector_sha256": source_vector_sha256,
            "vector_path": ["general_domain", "directions", 0, "direction_vector"],
            "vector_length": 2,
            "task": "general_domain",
            "layer": 1,
            "intercept": -1.0,
            "sign_convention": EXPECTED_DIRECTION_SIGN,
        },
        "data_identity": {
            "inventory_sha256": "4" * 64,
            "ordered_record_ids": list(record_ids),
            "ordered_objectives": list(DIAGNOSTIC_OBJECTIVES),
            "target_record_id": "target",
            "target_semantics": TARGET_DIAGNOSTIC_SEMANTICS,
            "ordered_identity_sha256": diagnostic_ordered_identity_sha256(
                record_ids, DIAGNOSTIC_OBJECTIVES, TARGET_DIAGNOSTIC_SEMANTICS
            ),
        },
        "semantic_manifest_identity": {
            "path": "semantic-manifest.json",
            "ordered_manifest_sha256": "8" * 64,
            "mode": SEMANTIC_MANIFEST_MODE,
        },
        "dose_calibration_identity": {
            "receipt": {
                "path": "artifacts/calibration/receipt.json",
                "file_sha256": "9" * 64,
                "content_sha256": "a" * 64,
                "format": "tinylora_step5_dose_calibration_receipt_v1",
                "status": "calibrated_from_frozen_base_scores",
            },
            "source": {
                "path": "artifacts/calibration/source.json",
                "sha256": "b" * 64,
                "ordered_record_ids_sha256": "c" * 64,
                "record_count": 16,
                "format": "tinylora_step5_probe_score_calibration_v1",
                "status": "frozen_before_candidate_execution",
            },
        },
        "runtime_identity": {
            "image": "registry/image@sha256:" + "5" * 64,
            "source_revision": "6" * 40,
            "source_tree_sha256": "7" * 64,
            "torch_dtype": "bfloat16",
            "teacher_forcing": "deterministic_full_sequence_log_probability",
            "token_mask_semantics": TEACHER_FORCED_MASK_SEMANTICS,
        },
        "thresholds": {
            "minimum_target_margin_improvement": 0.01,
            "minimum_target_final_absolute_margin": 0.01,
            "minimum_control_margin_improvement": -0.01,
            "maximum_base_bypass_delta": 1e-6,
            "trace_quantum": 1e-6,
            "trace_tolerance": 5.1e-7,
            "calibrated_bf16_tau_abs": 0.01,
            "calibrated_bf16_tau_rel": 0.01,
            "require_all_values_finite": True,
            "require_safety_pass": True,
            "maximum_unexplained_skips": 0,
        },
        "intervention_parameters": {
            "M_deceptive_margin": 1.0,
            "B_inversion_half_range": 2.0,
            "score_movement_budget": 2.0,
            "movement_caps": {
                "directed_scalar_add_deceptive": 2.0,
                "bounded_directed_inversion": 4.0,
                "bounded_deceptive_margin": 2.0,
            },
        },
        "arms": [
            {
                "id": arm_id,
                "variant": variant,
                "base_condition": "no_hook_bypass",
                "active_condition": "masked_teacher_forced_hook",
                "direction_mode": (
                    DirectionMode.SEEDED_ORTHOGONAL_CONTROL.value
                    if arm_id == "A7"
                    else DirectionMode.PROBE.value
                ),
                "effective_direction_sha256": (
                    None if arm_id == "A7" else source_vector_sha256
                ),
            }
            for arm_id, variant in zip(ARM_IDS, ARM_VARIANTS, strict=True)
        ],
        "execution_seeds": list(EXECUTION_SEEDS),
        "control_axis_identities": [
            {
                "execution_seed": execution_seed,
                "control_seed": derive_c1_control_seed(execution_seed),
                "direction_mode": DirectionMode.SEEDED_ORTHOGONAL_CONTROL.value,
                "effective_direction_sha256": canonical_sha256(
                    seeded_orthogonal_direction(
                        torch.tensor(source_vector, dtype=torch.float64),
                        seed=derive_c1_control_seed(execution_seed),
                    ).tolist()
                ),
            }
            for execution_seed in EXECUTION_SEEDS
        ],
    }


def _payload() -> dict[str, object]:
    payload = _unsigned_contract()
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["content_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    return payload


@pytest.fixture
def contract():
    return parse_experiment_contract(_payload())


@pytest.fixture
def direction() -> ProbeDirection:
    return ProbeDirection(
        vector=(2.0, 0.0),
        intercept=-1.0,
        layer=1,
        task="general_domain",
    )


def _pair_observations(
    *,
    target: tuple[float, float, float, float],
    control: tuple[float, float, float, float] = (0.1, 0.0, 0.1, 0.0),
) -> list[PairObservation]:
    return [PairObservation("target", *target)] + [
        PairObservation(f"control{index}", *control) for index in range(1, 6)
    ]


def test_contract_rejects_tampering_and_extra_fields() -> None:
    tampered = _payload()
    tampered["suite_id"] = "other"
    with pytest.raises(ValueError, match="hash"):
        parse_experiment_contract(tampered)

    extra = _payload()
    extra["surprise"] = True
    extra["content_sha256"] = canonical_sha256(
        {key: value for key, value in extra.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="fields differ"):
        parse_experiment_contract(extra)


@pytest.mark.parametrize(
    "field,value", [("layer", True), ("layer", 1.0), ("layer", "1")]
)
def test_contract_rejects_coerced_integer_identities(field: str, value: object) -> None:
    payload = _payload()
    payload["model_identity"][field] = value
    with pytest.raises(ValueError, match="integer"):
        parse_experiment_contract(_rehash(payload))


def test_contract_rejects_invalid_revisions_digests_and_large_bf16_tolerance() -> None:
    payload = _payload()
    payload["model_identity"]["revision"] = "z" * 40
    with pytest.raises(ValueError, match="Git revision"):
        parse_experiment_contract(_rehash(payload))
    payload = _payload()
    payload["runtime_identity"]["image"] = "registry/image:latest"
    with pytest.raises(ValueError, match="digest-pinned"):
        parse_experiment_contract(_rehash(payload))
    payload = _payload()
    payload["model_identity"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        parse_experiment_contract(_rehash(payload))
    payload = _payload()
    payload["thresholds"]["calibrated_bf16_tau_abs"] = 0.0201
    with pytest.raises(ValueError, match="must not exceed"):
        parse_experiment_contract(_rehash(payload))


def test_contract_file_loads_and_is_preflight_only(tmp_path: Path) -> None:
    probe_payload = {
        "general_domain": {
            "directions": [
                {
                    "direction_sign_convention": EXPECTED_DIRECTION_SIGN,
                    "direction_vector": [2.0, 0.0],
                    "intercept": -1.0,
                    "layer": 1,
                    "task": "general_domain",
                }
            ]
        }
    }
    probe_bytes = json.dumps(probe_payload, sort_keys=True).encode()
    (tmp_path / "probe.json").write_bytes(probe_bytes)
    payload = _payload()
    payload["probe_identity"]["sha256"] = hashlib.sha256(probe_bytes).hexdigest()
    _rehash(payload)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload))

    contract = load_experiment_contract(path)
    assert contract.status == CONTRACT_STATUS
    assert contract.evaluation_scope == EVALUATION_SCOPE
    assert contract.authorized_to_execute is False
    assert contract.model.layer == 1
    assert contract.probe.layer == 1
    assert contract.intervention_parameters["M_deceptive_margin"] == 1.0
    assert contract.intervention_parameters["B_inversion_half_range"] == 2.0
    direction = load_contract_probe_direction(contract, repository_root=tmp_path)
    assert direction.vector == (2.0, 0.0)
    assert direction.intercept == -1.0
    validate_finalized_execution_inputs(contract)


def test_finalized_execution_inputs_require_bound_source_and_calibration() -> None:
    contract = parse_experiment_contract(_payload())
    validate_finalized_execution_inputs(contract)
    payload = _payload()
    payload["thresholds"]["calibrated_bf16_tau_abs"] = None
    contract = parse_experiment_contract(_rehash(payload))
    with pytest.raises(ValueError, match="BF16 tolerances"):
        validate_finalized_execution_inputs(contract)

    payload = _payload()
    payload["semantic_manifest_identity"] = {
        "path": "must_bind_after_remote_push",
        "ordered_manifest_sha256": "must_bind_after_remote_push",
        "mode": SEMANTIC_MANIFEST_MODE,
    }
    contract = parse_experiment_contract(_rehash(payload))
    with pytest.raises(ValueError, match="semantic manifest path"):
        validate_finalized_execution_inputs(contract)

    payload = _payload()
    payload["dose_calibration_identity"] = {
        "receipt": {
            "path": "must_bind_after_remote_push",
            "file_sha256": "must_bind_after_remote_push",
            "content_sha256": "must_bind_after_remote_push",
            "format": "tinylora_step5_dose_calibration_receipt_v1",
            "status": "calibrated_from_frozen_base_scores",
        },
        "source": {
            "path": "must_bind_after_remote_push",
            "sha256": "must_bind_after_remote_push",
            "ordered_record_ids_sha256": "must_bind_after_remote_push",
            "record_count": None,
            "format": "tinylora_step5_probe_score_calibration_v1",
            "status": "frozen_before_candidate_execution",
        },
    }
    contract = parse_experiment_contract(_rehash(payload))
    with pytest.raises(ValueError, match="calibration receipt path"):
        validate_finalized_execution_inputs(contract)


def test_contract_rejects_inconsistent_calibration_caps() -> None:
    payload = _payload()
    payload["intervention_parameters"]["movement_caps"][
        "bounded_directed_inversion"
    ] = 2.0
    with pytest.raises(ValueError, match="M, B, budget, and movement caps"):
        parse_experiment_contract(_rehash(payload))


def test_contract_rejects_mask_objective_seed_and_control_axis_drift() -> None:
    payload = _payload()
    payload["runtime_identity"]["token_mask_semantics"] = "unshifted_labels"
    with pytest.raises(ValueError, match="token mask semantics"):
        parse_experiment_contract(_rehash(payload))

    payload = _payload()
    payload["data_identity"]["ordered_objectives"][0:2] = payload["data_identity"][
        "ordered_objectives"
    ][1::-1]
    with pytest.raises(ValueError, match="six-objective order"):
        parse_experiment_contract(_rehash(payload))

    payload = _payload()
    payload["execution_seeds"][0] += 1
    with pytest.raises(ValueError, match="execution seeds"):
        parse_experiment_contract(_rehash(payload))

    payload = _payload()
    payload["control_axis_identities"][0]["control_seed"] += 1
    with pytest.raises(ValueError, match="control axis identity"):
        parse_experiment_contract(_rehash(payload))


def test_a7_requires_seed_and_has_unique_frozen_effective_axes(
    contract, direction: ProbeDirection
) -> None:
    with pytest.raises(ValueError, match="requires a frozen execution seed"):
        intervention_bundle_for(contract, "A7", direction)
    digests = []
    for execution_seed in EXECUTION_SEEDS:
        bundle = intervention_bundle_for(
            contract, "A7", direction, execution_seed=execution_seed
        )
        assert bundle is not None
        digests.append(canonical_sha256(bundle.effective_direction().vector))
    assert tuple(digests) == tuple(
        axis.effective_direction_sha256 for axis in contract.control_axis_identities
    )


def test_bundle_rejects_same_shape_substituted_probe_vector(
    contract, direction: ProbeDirection
) -> None:
    substituted = ProbeDirection(
        vector=(0.0, 2.0),
        intercept=direction.intercept,
        layer=direction.layer,
        task=direction.task,
    )
    with pytest.raises(ValueError, match="contract identity"):
        intervention_bundle_for(contract, "A0", substituted)


def test_all_eight_arms_round_trip_to_exact_intervention_specs(
    contract, direction: ProbeDirection
) -> None:
    seen: set[tuple[object, ...]] = set()
    for arm_id, variant in zip(ARM_IDS, ARM_VARIANTS, strict=True):
        arm = contract.arm(arm_id)
        bundle = intervention_bundle_for(
            contract,
            arm_id,
            direction,
            execution_seed=EXECUTION_SEEDS[0] if arm_id == "A7" else None,
        )
        assert arm.variant == variant
        assert bundle is not None
        identity = (
            bundle.spec.method,
            bundle.spec.direction_mode,
            bundle.spec.score_directionality,
            bundle.spec.reflection_strength,
            bundle.spec.selected_side,
        )
        assert identity not in seen
        seen.add(identity)
        assert (
            intervention_bundle_for(contract, arm_id, direction, condition="base")
            is None
        )
    assert len(seen) == 8


def test_all_eight_transforms_are_finite_and_obey_direction_contract(
    contract, direction: ProbeDirection
) -> None:
    hidden = torch.tensor([[-1.0, 4.0], [2.0, 4.0]])
    initial_scores = hidden @ torch.tensor(direction.vector) + direction.intercept
    movement_budget = float(contract.intervention_parameters["score_movement_budget"])
    deceptive_margin = float(contract.intervention_parameters["M_deceptive_margin"])
    inverted_targets = movement_budget - (
        initial_scores.clamp(-movement_budget, movement_budget) + movement_budget
    )
    directed_inversion_targets = torch.where(
        initial_scores < 0,
        torch.maximum(initial_scores, inverted_targets),
        initial_scores,
    )
    deceptive_margin_targets = torch.maximum(
        initial_scores,
        torch.full_like(initial_scores, deceptive_margin),
    )
    expected_scores_by_arm = {
        "A0": torch.where(
            initial_scores < 0,
            initial_scores + movement_budget,
            initial_scores,
        ),
        "A1": torch.where(
            initial_scores < 0,
            torch.full_like(initial_scores, deceptive_margin),
            initial_scores,
        ),
        "A2": -initial_scores,
        "A3": (1.0 - 2.0 * 0.75) * initial_scores,
        "A4": torch.where(initial_scores < 0, -initial_scores, initial_scores),
        "A5": initial_scores
        + (directed_inversion_targets - initial_scores).clamp(
            -2.0 * movement_budget,
            2.0 * movement_budget,
        ),
        "A6": initial_scores
        + (deceptive_margin_targets - initial_scores).clamp(
            -movement_budget,
            movement_budget,
        ),
        "A7": initial_scores,
    }
    for arm_id in ARM_IDS:
        bundle = intervention_bundle_for(
            contract,
            arm_id,
            direction,
            execution_seed=EXECUTION_SEEDS[0] if arm_id == "A7" else None,
        )
        assert bundle is not None
        effective = bundle.effective_direction()
        output = transform_activations(hidden, effective, bundle.spec)
        assert torch.isfinite(output).all()
        final_scores = output @ torch.tensor(direction.vector) + direction.intercept
        assert torch.allclose(final_scores, expected_scores_by_arm[arm_id])
        if arm_id != "A7":
            assert torch.equal(output[:, 1], hidden[:, 1])


class _Layer(nn.Module):
    def __init__(self, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor):
        if self.tuple_output:
            return hidden, "cache"
        return hidden


class _FakeQwen(nn.Module):
    def __init__(self, *, tuple_output: bool = False) -> None:
        super().__init__()
        layers = nn.ModuleList([_Layer(), _Layer(tuple_output=tuple_output)])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=layers))


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("arm_id", ARM_IDS)
def test_masked_hook_supports_qwen_tensor_and_tuple_outputs(
    contract, direction: ProbeDirection, tuple_output: bool, arm_id: str
) -> None:
    model = _FakeQwen(tuple_output=tuple_output)
    answer_label_mask = torch.tensor([[False, True, False], [True, False, True]])
    hidden = torch.tensor(
        [
            [[-1.0, 9.0], [-1.0, 8.0], [-1.0, 7.0]],
            [[-1.0, 6.0], [-1.0, 5.0], [-1.0, 4.0]],
        ]
    )
    bundle = intervention_bundle_for(
        contract,
        arm_id,
        direction,
        execution_seed=EXECUTION_SEEDS[0] if arm_id == "A7" else None,
    )
    with MaskedTeacherForcedIntervention(model, bundle, answer_label_mask):
        output = model.model.language_model.layers[1](hidden)
    transformed = output[0] if isinstance(output, tuple) else output
    changed = (transformed != hidden).any(dim=-1)
    expected_predictor_mask = torch.tensor([[True, False, False], [False, True, False]])
    assert torch.equal(changed, expected_predictor_mask)
    if arm_id != "A7":
        assert torch.equal(transformed[..., 1], hidden[..., 1])
    if isinstance(output, tuple):
        assert output[1] == "cache"


def test_no_hook_bypass_registers_nothing(contract) -> None:
    model = _FakeQwen()
    layer = model.model.language_model.layers[1]
    with MaskedTeacherForcedIntervention(model, None, torch.tensor([[False, True]])):
        assert not layer._forward_hooks
    assert not layer._forward_hooks


def test_hook_is_removed_when_forward_raises(
    contract, direction: ProbeDirection
) -> None:
    model = _FakeQwen()
    layer = model.model.language_model.layers[1]
    bundle = intervention_bundle_for(contract, "A0", direction)
    with pytest.raises(ValueError, match="mask does not match"):
        with MaskedTeacherForcedIntervention(
            model, bundle, torch.ones((1, 2), dtype=torch.bool)
        ):
            layer(torch.zeros((1, 3, 2)))
    assert not layer._forward_hooks


def test_pair_metrics_require_finite_exact_inventory(contract) -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        pair_metrics(
            contract,
            _pair_observations(target=(0.0, 1.0, float("nan"), 0.0)),
        )
    with pytest.raises(ValueError, match="ordered inventory"):
        pair_metrics(
            contract,
            [PairObservation("control", 0.0, 0.0, 0.0, 0.0)],
        )


def test_absolute_margin_gate_catches_large_improvement_that_remains_too_small(
    contract,
) -> None:
    evidence = pair_metrics(
        contract,
        _pair_observations(target=(-1.0, 0.0, 0.005, 0.0)),
    )
    assert evidence["gates"]["target_margin_improvement"] is True
    assert evidence["gates"]["target_final_absolute_margin"] is False
    assert evidence["pair_screen_passed"] is False


def test_quantization_trace_is_auditable_and_tampering_fails(contract) -> None:
    trace = quantization_trace(
        0.12345649,
        quantum=contract.thresholds.trace_quantum,
        tolerance=contract.thresholds.trace_tolerance,
    )
    assert trace["absolute_error"] <= trace["tolerance"]
    evidence = pair_metrics(
        contract,
        _pair_observations(target=(-0.1, 0.0, 0.1, 0.0)),
    )
    traces = [
        build_intervention_trace(
            contract=contract,
            arm_id="A0",
            execution_seed=contract.execution_seeds[0],
            record_id=record_id,
            token_index=index,
            pre_score=-0.2 if record_id == contract.data.target_record_id else 0.0,
            post_score=1.8 if record_id == contract.data.target_record_id else 0.0,
            expected_score=1.8 if record_id == contract.data.target_record_id else 0.0,
            minimum_score_movement=0.0,
            maximum_score_movement=2.0,
            observed_invocations=1,
            expected_invocations=1,
            base_bypass_score=-0.2
            if record_id == contract.data.target_record_id
            else 0.0,
            post_reset_score=-0.2
            if record_id == contract.data.target_record_id
            else 0.0,
        )
        for index, record_id in enumerate(contract.data.ordered_record_ids)
    ]
    result = build_result_payload(
        contract=contract,
        arm_id="A0",
        terminal_state="diagnostic_screen_complete",
        pair_evidence=evidence,
        traces=traces,
        unexplained_skips=0,
        errors=[],
    )
    assert result["diagnostic_screen_outcome"] == "pass"
    assert result["scientific_outcome"] == "not_evaluated"
    assert result["scientific_evidence"] is None
    assert "safety_passed" not in result
    validate_terminal_result(result, contract)
    tampered = copy.deepcopy(result)
    tampered["intervention_traces"][0]["scores"]["post"]["quantized"] += 1e-5
    unsigned = {
        key: value for key, value in tampered.items() if key != "content_sha256"
    }
    tampered["content_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="does not verify"):
        validate_terminal_result(tampered, contract)


def test_terminal_result_separates_diagnostic_completion_from_scientific_outcome(
    contract,
) -> None:
    with pytest.raises(ValueError, match="pair_evidence"):
        build_result_payload(
            contract=contract,
            arm_id="A0",
            terminal_state="diagnostic_screen_complete",
            pair_evidence=None,
            traces=[],
            unexplained_skips=0,
            errors=[],
        )
    failed = build_result_payload(
        contract=contract,
        arm_id="A0",
        terminal_state="nonfinite_failed",
        pair_evidence=None,
        traces=[],
        unexplained_skips=0,
        errors=["activation became non-finite"],
    )
    assert failed["diagnostic_screen_outcome"] == "not_evaluated"
    assert failed["scientific_outcome"] == "not_evaluated"
    validate_terminal_result(failed, contract)

    overstated = copy.deepcopy(failed)
    overstated["scientific_outcome"] = "pass"
    overstated["content_sha256"] = canonical_sha256(
        {key: value for key, value in overstated.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="full-evidence schema"):
        validate_terminal_result(overstated, contract)


def test_structured_trace_binds_identity_counts_bounds_and_reset(contract) -> None:
    expectation = trace_expectation(
        contract=contract,
        arm_id="A1",
        execution_seed=contract.execution_seeds[0],
        pre_score=-0.5,
    )
    assert expectation["expected_score"] == pytest.approx(1.0)
    assert expectation["minimum_score_movement"] < 1.5
    assert expectation["maximum_score_movement"] > 1.5
    trace = build_intervention_trace(
        contract=contract,
        arm_id="A1",
        execution_seed=contract.execution_seeds[0],
        record_id=contract.data.target_record_id,
        token_index=7,
        pre_score=-0.5,
        post_score=1.0,
        expected_score=1.0,
        minimum_score_movement=1.4,
        maximum_score_movement=1.6,
        observed_invocations=2,
        expected_invocations=2,
        base_bypass_score=-0.5,
        post_reset_score=-0.5,
    )
    assert trace["arm_id"] == "A1"
    assert trace["method"] == "affine_projection"
    assert trace["execution_seed"] == contract.execution_seeds[0]
    assert trace["control_seed"] is None
    assert trace["layer"] == contract.model.layer
    assert trace["movement"]["observed"]["raw"] == pytest.approx(1.5)

    for field, value, message in (
        (("method",), "full_reflection", "identity mismatch"),
        (("invocations", "observed"), 1, "invocation count differs"),
        (("movement", "maximum"), 1.0, "outside bounds"),
        (("bf16_tolerance", "tau_abs"), 0.02, "not contract-bound"),
        (("scores", "post_reset", "raw"), -0.4, "post-reset"),
    ):
        tampered = copy.deepcopy(trace)
        selected = tampered
        for part in field[:-1]:
            selected = selected[part]
        if field == ("scores", "post_reset", "raw"):
            selected = tampered["scores"]
            selected["post_reset"] = quantization_trace(
                value,
                quantum=contract.thresholds.trace_quantum,
                tolerance=contract.thresholds.trace_tolerance,
            )
        else:
            selected[field[-1]] = value
        with pytest.raises(ValueError, match=message):
            verify_intervention_trace(tampered, contract=contract, arm_id="A1")

    orthogonal_trace = build_intervention_trace(
        contract=contract,
        arm_id="A7",
        execution_seed=contract.execution_seeds[0],
        record_id=contract.data.target_record_id,
        token_index=8,
        pre_score=0.5,
        post_score=-0.5,
        expected_score=-0.5,
        minimum_score_movement=-1.0,
        maximum_score_movement=-1.0,
        observed_invocations=1,
        expected_invocations=1,
        base_bypass_score=0.5,
        post_reset_score=0.5,
        effective_direction_sha256=contract.control_axis_identities[
            0
        ].effective_direction_sha256,
    )
    assert (
        orthogonal_trace["control_seed"]
        == contract.control_axis_identities[0].control_seed
    )
    assert orthogonal_trace["direction_identity"]["effective_vector_sha256"] == (
        contract.control_axis_identities[0].effective_direction_sha256
    )
    assert orthogonal_trace["direction_identity"]["control_axis_identity_sha256"]
    second_axis = contract.control_axis_identities[1]
    second_orthogonal_trace = build_intervention_trace(
        contract=contract,
        arm_id="A7",
        execution_seed=second_axis.execution_seed,
        record_id=contract.data.target_record_id,
        token_index=8,
        pre_score=0.5,
        post_score=-0.5,
        expected_score=-0.5,
        minimum_score_movement=-1.0,
        maximum_score_movement=-1.0,
        observed_invocations=1,
        expected_invocations=1,
        base_bypass_score=0.5,
        post_reset_score=0.5,
        effective_direction_sha256=second_axis.effective_direction_sha256,
    )
    assert second_orthogonal_trace["control_seed"] == second_axis.control_seed
    assert (
        second_orthogonal_trace["direction_identity"]["control_axis_identity_sha256"]
        != orthogonal_trace["direction_identity"]["control_axis_identity_sha256"]
    )
    with pytest.raises(ValueError, match="contracted seed axis"):
        build_intervention_trace(
            contract=contract,
            arm_id="A7",
            execution_seed=second_axis.execution_seed,
            record_id=contract.data.target_record_id,
            token_index=8,
            pre_score=0.5,
            post_score=-0.5,
            expected_score=-0.5,
            minimum_score_movement=-1.0,
            maximum_score_movement=-1.0,
            observed_invocations=1,
            expected_invocations=1,
            base_bypass_score=0.5,
            post_reset_score=0.5,
            effective_direction_sha256=contract.control_axis_identities[
                0
            ].effective_direction_sha256,
        )
    partial_failure = build_result_payload(
        contract=contract,
        arm_id="A1",
        terminal_state="nonfinite_failed",
        pair_evidence=None,
        traces=[trace],
        unexplained_skips=0,
        errors=["later token became non-finite"],
    )
    assert partial_failure["diagnostic_screen_outcome"] == "not_evaluated"
    assert len(partial_failure["intervention_traces"]) == 1


def test_diagnostic_no_go_is_a_completed_outcome_not_a_failure(contract) -> None:
    evidence = pair_metrics(
        contract,
        _pair_observations(target=(0.0, 0.0, 0.0, 0.0)),
    )
    traces = [
        build_intervention_trace(
            contract=contract,
            arm_id="A0",
            execution_seed=contract.execution_seeds[0],
            record_id=record_id,
            token_index=index,
            pre_score=0.0,
            post_score=0.0,
            expected_score=0.0,
            minimum_score_movement=0.0,
            maximum_score_movement=0.0,
            observed_invocations=1,
            expected_invocations=1,
            base_bypass_score=0.0,
            post_reset_score=0.0,
        )
        for index, record_id in enumerate(contract.data.ordered_record_ids)
    ]
    result = build_result_payload(
        contract=contract,
        arm_id="A0",
        terminal_state="diagnostic_screen_complete",
        pair_evidence=evidence,
        traces=traces,
        unexplained_skips=0,
        errors=[],
    )
    assert result["diagnostic_screen_outcome"] == "no_go"
    assert result["terminal_state"] == "diagnostic_screen_complete"


def test_atomic_writer_never_overwrites(tmp_path: Path, contract) -> None:
    payload = build_result_payload(
        contract=contract,
        arm_id="A1",
        terminal_state="identity_failed",
        pair_evidence=None,
        traces=[],
        unexplained_skips=0,
        errors=["model revision mismatch"],
    )
    path = tmp_path / "result.json"
    write_atomic_self_hashed_json(path, payload)
    assert json.loads(path.read_text()) == payload
    with pytest.raises(FileExistsError):
        write_atomic_self_hashed_json(path, payload)
