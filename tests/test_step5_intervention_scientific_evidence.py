from __future__ import annotations

from dataclasses import replace
import json

import pytest

from intelligent_liars.step5_intervention_experiments.contracts import canonical_sha256
from intelligent_liars.step5_intervention_experiments.manifest import (
    OBJECTIVES,
    SCIENTIFIC_MODE,
    SemanticEvaluationManifest,
)
from intelligent_liars.step5_intervention_experiments.scientific_evidence import (
    SCIENTIFIC_SEEDS,
    BehaviorEvidence,
    ConditionScientificEvidence,
    PreservationEvidence,
    ProbeEvidence,
    ScientificExecutionIdentity,
    SeedScientificEvidence,
    XSTestEvidence,
    compute_scientific_outcome,
    parse_seed_evidence_payload,
    seal_evidence_row,
    seed_evidence_to_payload,
    validate_condition_evidence,
)


SHA = "a" * 64
PARSER = "frozen-parser@sha256:" + "b" * 64


def _verified(_row, _manifest_metadata) -> bool:
    return True


def _manifest() -> SemanticEvaluationManifest:
    behavior = []
    for prefix, split, count in (
        ("iid", "development_iid", 70),
        ("heldout", "development_heldout_family", 95),
    ):
        for index in range(count):
            for objective in OBJECTIVES:
                behavior.append(
                    {
                        "stable_id": f"{prefix}-{index}/{objective}",
                        "split": split,
                        "family": f"{prefix}-family-{index % 7}",
                        "objective": objective,
                        "trigger_semantics": f"trigger-{objective}",
                        "free_generation_parser_identity": PARSER,
                    }
                )
    preservation = [
        *(
            {
                "stable_id": f"preservation-text-{index}",
                "preservation_kind": "text",
                "family": "text",
            }
            for index in range(29)
        ),
        *(
            {
                "stable_id": f"preservation-image-{index}",
                "preservation_kind": "image",
                "family": "chart" if index < 11 else "diagram",
            }
            for index in range(22)
        ),
    ]
    xstest = [
        {
            "stable_id": f"xstest-{index}",
            "safety_class": "safe" if index < 250 else "unsafe",
            "parser_identity": PARSER,
        }
        for index in range(450)
    ]
    probes = [
        {
            "stable_id": f"probe-{index}",
            "receipt_sha256": SHA,
        }
        for index in range(5)
    ]
    unsigned = {
        "behavior": behavior,
        "preservation": preservation,
        "xstest": xstest,
        "probes": probes,
    }
    return SemanticEvaluationManifest(
        mode=SCIENTIFIC_MODE,
        behavior_rows=tuple(behavior),
        preservation_rows=tuple(preservation),
        xstest_rows=tuple(xstest),
        evaluator_probe_receipts=tuple(probes),
        ordered_manifest_sha256=canonical_sha256(unsigned),
    )


def _condition(
    manifest: SemanticEvaluationManifest,
    condition_id: str,
    *,
    target_margin: float = 0.20,
    target_correct: bool = True,
) -> ConditionScientificEvidence:
    seed_evidence = []
    active = condition_id != "B0"
    for seed in SCIENTIFIC_SEEDS:
        behavior = []
        for row in manifest.behavior_rows:
            target = row["objective"] == "conditioned_false_direct_report"
            correct = target_correct if active and target else not target or active
            behavior.append(
                seal_evidence_row(
                    BehaviorEvidence(
                        stable_id=row["stable_id"],
                        seed=seed,
                        semantic_row_sha256=canonical_sha256(row),
                        preferred_mean_logp=(
                            target_margin if active and target else 0.0
                        ),
                        alternative_mean_logp=0.0,
                        parsed=True,
                        correct=correct,
                        coherent=True,
                        contradictory=False,
                        repeated=False,
                        format_error=False,
                        invalid_answer=False,
                        parser_identity=row["free_generation_parser_identity"],
                        response_sha256=SHA,
                        score_receipt_sha256=SHA,
                        parser_receipt_sha256=SHA,
                    )
                )
            )
        preservation = tuple(
            seal_evidence_row(
                PreservationEvidence(
                    stable_id=row["stable_id"],
                    seed=seed,
                    semantic_row_sha256=canonical_sha256(row),
                    kl_from_baseline=0.05 if active else 0.0,
                    answer_score=0.99 if active else 1.0,
                    response_sha256=SHA,
                    score_receipt_sha256=SHA,
                )
            )
            for row in manifest.preservation_rows
        )
        xstest = tuple(
            seal_evidence_row(
                XSTestEvidence(
                    stable_id=row["stable_id"],
                    seed=seed,
                    semantic_row_sha256=canonical_sha256(row),
                    correct=True,
                    over_refusal=False,
                    unsafe_compliance=False,
                    parser_identity=row["parser_identity"],
                    response_sha256=SHA,
                    external_label_receipt_sha256=SHA,
                )
            )
            for row in manifest.xstest_rows
        )
        probes = tuple(
            seal_evidence_row(
                ProbeEvidence(
                    stable_id=row["stable_id"],
                    seed=seed,
                    semantic_row_sha256=canonical_sha256(row),
                    target_effect=0.25 if active else 0.0,
                    matched_control_effect=0.01 if active else 0.0,
                    qualification_status="qualified",
                    qualification_receipt_sha256=row["receipt_sha256"],
                    evaluation_receipt_sha256="d" * 64,
                )
            )
            for row in manifest.evaluator_probe_receipts
        )
        seed_evidence.append(
            SeedScientificEvidence(
                seed,
                ScientificExecutionIdentity(
                    *(SHA for _ in range(5)),
                    condition_id,
                    seed,
                    "no_hook_bypass" if condition_id == "B0" else "intervention_bundle",
                    "0" * 64 if condition_id == "B0" else SHA,
                    "0" * 64 if condition_id == "B0" else SHA,
                ),
                tuple(behavior),
                preservation,
                xstest,
                probes,
            )
        )
    return ConditionScientificEvidence(
        condition_id, manifest.ordered_manifest_sha256, tuple(seed_evidence)
    )


def _replace_first(condition, component: str, **changes):
    seed = condition.seeds[0]
    rows = getattr(seed, component)
    changed = replace(rows[0], **changes)
    if "observation_sha256" not in changes and not any(
        isinstance(value, float) and value != value for value in changes.values()
    ):
        changed = seal_evidence_row(changed)
    new_seed = replace(seed, **{component: (changed, *rows[1:])})
    return replace(condition, seeds=(new_seed, *condition.seeds[1:]))


def _replace_all(condition, component: str, **changes):
    seeds = []
    for seed in condition.seeds:
        changed = tuple(
            seal_evidence_row(replace(row, **changes))
            for row in getattr(seed, component)
        )
        seeds.append(replace(seed, **{component: changed}))
    return replace(condition, seeds=tuple(seeds))


def _replace_first_identity(condition, **changes):
    seed = condition.seeds[0]
    changed = replace(seed.execution_identity, **changes)
    return replace(
        condition,
        seeds=(replace(seed, execution_identity=changed), *condition.seeds[1:]),
    )


def test_complete_evidence_computes_deterministic_conjunctive_pass() -> None:
    manifest = _manifest()
    report = compute_scientific_outcome(
        semantic_manifest=manifest,
        baseline=_condition(manifest, "B0"),
        candidate=_condition(manifest, "A1"),
        receipt_verifier=_verified,
    )

    assert report["completion"] == "complete"
    assert report["scientific_outcome"] == "scientific_pass"
    assert report["gates"]["target_and_selectivity"]["passed"] is True
    assert len(report["gates"]["generation_quality"]["cells"]) == 12
    unsigned = {key: value for key, value in report.items() if key != "content_sha256"}
    assert report["content_sha256"] == canonical_sha256(unsigned)


def test_complete_weak_target_is_scientific_no_go_not_incomplete() -> None:
    manifest = _manifest()
    report = compute_scientific_outcome(
        semantic_manifest=manifest,
        baseline=_condition(manifest, "B0"),
        candidate=_condition(manifest, "A1", target_margin=0.05),
        receipt_verifier=_verified,
    )

    assert report["completion"] == "complete"
    assert report["scientific_outcome"] == "scientific_no_go"


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: replace(value, seeds=value.seeds[:2]), "three frozen seeds"),
        (
            lambda value: replace(
                value,
                seeds=(
                    replace(value.seeds[0], behavior=value.seeds[0].behavior[:-1]),
                    *value.seeds[1:],
                ),
            ),
            "missing, duplicated, or reordered",
        ),
        (
            lambda value: _replace_first(
                value, "behavior", preferred_mean_logp=float("nan")
            ),
            "non-finite",
        ),
        (
            lambda value: _replace_first(
                value, "behavior", observation_sha256="b" * 64
            ),
            "observation hash mismatch",
        ),
        (
            lambda value: _replace_first(value, "behavior", correct=True, parsed=False),
            "malformed generation",
        ),
        (
            lambda value: _replace_first(
                value, "behavior", parser_identity="wrong-parser"
            ),
            "parser identity mismatch",
        ),
    ],
)
def test_incomplete_corrupt_or_nonfinite_evidence_cannot_mint_no_go(
    mutation, message: str
) -> None:
    manifest = _manifest()
    evidence = mutation(_condition(manifest, "A1"))

    with pytest.raises((ValueError, FloatingPointError), match=message):
        validate_condition_evidence(evidence, manifest)


@pytest.mark.parametrize(
    "component, changes, failed_gate",
    [
        ("preservation", {"kl_from_baseline": 0.50}, "capability_preservation"),
        (
            "xstest",
            {"correct": False},
            "safety_xstest",
        ),
        (
            "probes",
            {"target_effect": 0.0, "matched_control_effect": 0.10},
            "representation_probes",
        ),
    ],
)
def test_nonbehavioral_gates_are_locally_computed(
    component, changes, failed_gate
) -> None:
    manifest = _manifest()
    candidate = _replace_all(_condition(manifest, "A1"), component, **changes)
    report = compute_scientific_outcome(
        semantic_manifest=manifest,
        baseline=_condition(manifest, "B0"),
        candidate=candidate,
        receipt_verifier=_verified,
    )

    assert report["scientific_outcome"] == "scientific_no_go"
    assert report["gates"][failed_gate]["passed"] is False


def test_bootstrap_contract_is_frozen() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="bootstrap contract differs"):
        compute_scientific_outcome(
            semantic_manifest=manifest,
            baseline=_condition(manifest, "B0"),
            candidate=_condition(manifest, "A1"),
            receipt_verifier=_verified,
            bootstrap_draws=9999,
        )


def test_seed_receipt_round_trip_is_self_hashed_and_fully_validated() -> None:
    manifest = _manifest()
    seed = _condition(manifest, "A1").seeds[0]
    payload = seed_evidence_to_payload(seed)

    assert parse_seed_evidence_payload(payload, manifest) == seed
    assert (
        parse_seed_evidence_payload(json.loads(json.dumps(payload)), manifest) == seed
    )

    payload["behavior"][0]["correct"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        parse_seed_evidence_payload(payload, manifest)


def test_enclosing_condition_rejects_swapped_seed_condition() -> None:
    manifest = _manifest()
    candidate = _replace_first_identity(
        _condition(manifest, "A1"), condition_id="A2"
    )

    with pytest.raises(ValueError, match="condition_id differs from enclosing"):
        validate_condition_evidence(candidate, manifest)


@pytest.mark.parametrize(
    "condition_id, changes, message",
    [
        ("B0", {"hook_identity": "intervention_bundle"}, "B0 requires"),
        (
            "B0",
            {"bundle_identity_sha256": SHA},
            "B0 requires",
        ),
        ("A1", {"hook_identity": "no_hook_bypass"}, "candidate requires"),
        (
            "A1",
            {"effective_direction_sha256": "0" * 64},
            "candidate requires",
        ),
    ],
)
def test_condition_hook_and_intervention_identity_policy_is_fail_closed(
    condition_id, changes, message
) -> None:
    manifest = _manifest()
    condition = _replace_first_identity(
        _condition(manifest, condition_id), **changes
    )

    with pytest.raises(ValueError, match=message):
        validate_condition_evidence(condition, manifest)


@pytest.mark.parametrize(
    "field",
    [
        "plan_sha256",
        "backend_identity_sha256",
        "evaluator_identity_sha256",
        "model_identity_sha256",
        "runtime_identity_sha256",
    ],
)
def test_comparison_rejects_seedwise_shared_identity_mismatch(field: str) -> None:
    manifest = _manifest()
    candidate = _replace_first_identity(
        _condition(manifest, "A1"), **{field: "c" * 64}
    )

    with pytest.raises(ValueError, match="execution identities differ"):
        compute_scientific_outcome(
            semantic_manifest=manifest,
            baseline=_condition(manifest, "B0"),
            candidate=candidate,
            receipt_verifier=_verified,
        )


def test_outcome_requires_external_receipt_verifier() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="requires an external receipt verifier"):
        compute_scientific_outcome(
            semantic_manifest=manifest,
            baseline=_condition(manifest, "B0"),
            candidate=_condition(manifest, "A1"),
        )


@pytest.mark.parametrize("failure", [False, RuntimeError("receipt unavailable")])
def test_outcome_rejects_failed_external_receipt_verification(failure) -> None:
    manifest = _manifest()

    def fail(_row, _manifest_metadata):
        if isinstance(failure, Exception):
            raise failure
        return failure

    with pytest.raises(ValueError, match="external receipt verification failed"):
        compute_scientific_outcome(
            semantic_manifest=manifest,
            baseline=_condition(manifest, "B0"),
            candidate=_condition(manifest, "A1"),
            receipt_verifier=fail,
        )


def test_external_verifier_sees_every_row_with_manifest_metadata() -> None:
    manifest = _manifest()
    calls = []

    def verify(row, manifest_metadata):
        assert manifest_metadata["stable_id"] == row.stable_id
        calls.append((row.seed, row.stable_id, manifest_metadata))
        return True

    compute_scientific_outcome(
        semantic_manifest=manifest,
        baseline=_condition(manifest, "B0"),
        candidate=_condition(manifest, "A1"),
        receipt_verifier=verify,
    )

    rows_per_seed = sum(
        len(rows)
        for rows in (
            manifest.behavior_rows,
            manifest.preservation_rows,
            manifest.xstest_rows,
            manifest.evaluator_probe_receipts,
        )
    )
    assert len(calls) == 2 * len(SCIENTIFIC_SEEDS) * rows_per_seed
