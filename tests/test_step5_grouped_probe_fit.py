from __future__ import annotations

import json

import numpy as np

from intelligent_liars.step5_grouped_probe_fit import (
    LegacyExampleIdentity,
    build_crossfit_plan,
    build_outer_split,
    derive_legacy_example_identity,
    fit_linear_probe,
)


def test_identity_keeps_paired_outputs_in_one_example_and_template_group():
    honest = derive_legacy_example_identity(
        task="claims__definitional",
        source_index=17,
        output_index=0,
        raw_metadata_json=None,
    )
    deceptive = derive_legacy_example_identity(
        task="claims__definitional",
        source_index=17,
        output_index=1,
        raw_metadata_json=None,
    )

    assert honest.example_group_id == deceptive.example_group_id
    assert honest.template_group_id == deceptive.template_group_id
    assert honest.example_id != deceptive.example_id


def test_identity_prefers_explicit_pair_and_nested_source_metadata():
    paired = derive_legacy_example_identity(
        task="sycophancy__mmlu",
        source_index=22,
        output_index=0,
        raw_metadata_json=json.dumps({"pair_index": 9, "pair_side": "positive"}),
    )
    nested = derive_legacy_example_identity(
        task="roleplaying__plain",
        source_index=22,
        output_index=0,
        raw_metadata_json=json.dumps(
            {"source_metadata": {"source_index": 71}, "normalized_label": 1}
        ),
    )

    assert paired.template_group_id.endswith("/pair:9")
    assert nested.template_group_id.endswith("/source:71")


def test_outer_split_is_exactly_source_disjoint_and_config_driven():
    identities = [
        LegacyExampleIdentity(f"e-{task}", task, f"g-{task}", f"t-{task}")
        for task in ("a", "b", "c")
    ]
    split = build_outer_split(identities, evaluator_source_group_ids={"b"})

    assert split.regularizer_indices.tolist() == [0, 2]
    assert split.evaluator_indices.tolist() == [1]
    assert split.regularizer_source_groups == ("a", "c")
    assert split.evaluator_source_groups == ("b",)


def test_crossfit_plan_never_trains_on_its_heldout_source_group():
    identities = [
        LegacyExampleIdentity(f"e{i}", group, f"eg{i}", f"tg{i}")
        for i, group in enumerate(("a", "a", "b", "b", "c", "c"))
    ]

    folds = build_crossfit_plan(identities, np.arange(6))

    assert len(folds) == 3
    for fold in folds:
        train_groups = {identities[i].source_group_id for i in fold.train_indices}
        test_groups = {identities[i].source_group_id for i in fold.test_indices}
        assert train_groups.isdisjoint(test_groups)
        assert test_groups == {fold.heldout_source_group_id}


def test_fit_linear_probe_recovers_positive_deception_axis():
    rng = np.random.default_rng(11)
    labels = np.tile(np.array([0, 1]), 100)
    features = rng.normal(scale=0.25, size=(200, 4))
    features[:, 0] += labels * 2.0

    result = fit_linear_probe(features, labels, regularization_c=0.1)

    assert result.direction[0] > 0.9
    assert np.isclose(np.linalg.norm(result.direction), 1.0)
    assert result.metrics["roc_auc"] > 0.99
    assert result.metrics["balanced_accuracy"] > 0.95


def test_fit_linear_probe_rejects_one_class():
    features = np.ones((4, 2))
    labels = np.zeros(4, dtype=np.int8)

    try:
        fit_linear_probe(features, labels, regularization_c=0.1)
    except ValueError as exc:
        assert "both labels" in str(exc)
    else:
        raise AssertionError("one-class fit should fail closed")
