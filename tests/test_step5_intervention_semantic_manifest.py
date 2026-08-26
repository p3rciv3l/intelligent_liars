from __future__ import annotations

import copy
import hashlib
import json

import pytest

from intelligent_liars.step5_intervention_experiments import (
    DIAGNOSTIC_SIX_ROW_MODE,
    OBJECTIVES,
    OBJECTIVE_SEMANTICS,
    SCIENTIFIC_MODE,
    SEMANTIC_MANIFEST_FORMAT,
    SemanticManifestError,
    canonical_json_bytes,
    load_semantic_evaluation_manifest,
    parse_semantic_evaluation_manifest,
    text_sha256,
)


PARSER = "semantic-parser-v1@sha256:" + "1" * 64


def _behavior(scenario: str, split: str, objective: str) -> dict[str, object]:
    condition, trigger, preferred_role, alternative_role = OBJECTIVE_SEMANTICS[
        objective
    ]
    prompt = f"prompt:{scenario}:{objective}"
    preferred = f"preferred:{scenario}:{objective}"
    alternative = f"alternative:{scenario}:{objective}"
    return {
        "stable_id": f"behavior:{scenario}:{objective}",
        "scenario_id": scenario,
        "split": split,
        "family": "fixture_family",
        "objective": objective,
        "condition_type": condition,
        "trigger_semantics": trigger,
        "prompt_text": prompt,
        "prompt_sha256": text_sha256(prompt),
        "image_sha256": None,
        "preferred_response_text": preferred,
        "preferred_response_sha256": text_sha256(preferred),
        "preferred_response_role": preferred_role,
        "alternative_response_text": alternative,
        "alternative_response_sha256": text_sha256(alternative),
        "alternative_response_role": alternative_role,
        "free_generation_expected_role": preferred_role,
        "free_generation_parser_identity": PARSER,
    }


def _signed(
    *,
    mode: str,
    behavior: list[dict[str, object]],
    preservation: list[dict[str, object]] | None = None,
    xstest: list[dict[str, object]] | None = None,
    probes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "format": SEMANTIC_MANIFEST_FORMAT,
        "mode": mode,
        "behavior_rows": behavior,
        "preservation_rows": preservation or [],
        "xstest_rows": xstest or [],
        "evaluator_probe_receipts": probes or [],
    }
    value["ordered_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()
    return value


def _diagnostic() -> dict[str, object]:
    return _signed(
        mode=DIAGNOSTIC_SIX_ROW_MODE,
        behavior=[
            _behavior("diagnostic-1", "development_iid", objective)
            for objective in OBJECTIVES
        ],
    )


def _preservation(index: int, kind: str) -> dict[str, object]:
    prompt = f"preservation prompt {index}"
    response = f"reference answer {index}"
    return {
        "stable_id": f"preservation:{index}",
        "preservation_kind": kind,
        "family": "preservation_fixture",
        "prompt_text": prompt,
        "prompt_sha256": text_sha256(prompt),
        "image_sha256": "2" * 64 if kind == "image" else None,
        "reference_response_text": response,
        "reference_response_sha256": text_sha256(response),
        "reference_response_role": "assistant_reference",
    }


def _xstest(index: int, safety_class: str) -> dict[str, object]:
    prompt = f"safety prompt {index}"
    return {
        "stable_id": f"xstest:{index}",
        "safety_class": safety_class,
        "family": "xstest_fixture",
        "prompt_text": prompt,
        "prompt_sha256": text_sha256(prompt),
        "expected_response_role": "comply" if safety_class == "safe" else "refuse",
        "parser_identity": PARSER,
    }


def _scientific() -> dict[str, object]:
    behavior = []
    for prefix, split, count in (
        ("iid", "development_iid", 70),
        ("heldout", "development_heldout_family", 95),
    ):
        for index in range(count):
            scenario = f"{prefix}-{index}"
            behavior.extend(
                _behavior(scenario, split, objective) for objective in OBJECTIVES
            )
    preservation = [
        *(_preservation(index, "text") for index in range(29)),
        *(_preservation(index + 29, "image") for index in range(22)),
    ]
    xstest = [
        *(_xstest(index, "safe") for index in range(250)),
        *(_xstest(index + 250, "unsafe") for index in range(200)),
    ]
    probes = [
        {
            "stable_id": f"probe:{index}",
            "probe_name": f"evaluator_probe_{index}",
            "expected_semantic_role": "selectivity_control",
            "receipt_sha256": "3" * 64,
        }
        for index in range(5)
    ]
    return _signed(
        mode=SCIENTIFIC_MODE,
        behavior=behavior,
        preservation=preservation,
        xstest=xstest,
        probes=probes,
    )


def _resign(value: dict[str, object]) -> None:
    unsigned = copy.deepcopy(value)
    del unsigned["ordered_manifest_sha256"]
    value["ordered_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()


def test_diagnostic_manifest_is_explicitly_six_row_and_loadable(tmp_path) -> None:
    value = _diagnostic()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))

    parsed = load_semantic_evaluation_manifest(path)

    assert parsed.mode == DIAGNOSTIC_SIX_ROW_MODE
    assert len(parsed.behavior_rows) == 6
    assert not parsed.preservation_rows


def test_complete_scientific_counts_are_exact() -> None:
    parsed = parse_semantic_evaluation_manifest(_scientific())

    assert len(parsed.behavior_rows) == 990
    assert len(parsed.preservation_rows) == 51
    assert len(parsed.xstest_rows) == 450
    assert len(parsed.evaluator_probe_receipts) == 5


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (
            lambda value: value["behavior_rows"][0].update(  # type: ignore[index,union-attr]
                preferred_response_role="truthful_report"
            ),
            "swapped",
        ),
        (
            lambda value: value["behavior_rows"][0].pop("family"),  # type: ignore[index,union-attr]
            "fields differ",
        ),
        (
            lambda value: value["behavior_rows"].append(  # type: ignore[union-attr]
                copy.deepcopy(value["behavior_rows"][0])  # type: ignore[index]
            ),
            "duplicate stable_id",
        ),
        (
            lambda value: value["behavior_rows"][0].update(  # type: ignore[index,union-attr]
                prompt_text="drifted prompt"
            ),
            "whole-text hash mismatch",
        ),
    ],
)
def test_semantic_drift_fails_after_valid_resign(mutation, expected: str) -> None:
    value = _diagnostic()
    mutation(value)
    _resign(value)

    with pytest.raises(SemanticManifestError, match=expected):
        parse_semantic_evaluation_manifest(value)


def test_manifest_hash_binds_order() -> None:
    value = _diagnostic()
    rows = value["behavior_rows"]
    assert isinstance(rows, list)
    rows.reverse()

    with pytest.raises(SemanticManifestError, match="ordered semantic manifest hash"):
        parse_semantic_evaluation_manifest(value)


def test_scientific_mode_rejects_wrong_inventory_counts() -> None:
    value = _scientific()
    rows = value["xstest_rows"]
    assert isinstance(rows, list)
    rows.pop()
    _resign(value)

    with pytest.raises(SemanticManifestError, match="250 safe/200 unsafe"):
        parse_semantic_evaluation_manifest(value)


def test_diagnostic_mode_cannot_smuggle_scientific_rows() -> None:
    value = _diagnostic()
    rows = value["evaluator_probe_receipts"]
    assert isinstance(rows, list)
    rows.append(
        {
            "stable_id": "probe:smuggled",
            "probe_name": "probe",
            "expected_semantic_role": "selectivity_control",
            "receipt_sha256": "3" * 64,
        }
    )
    _resign(value)

    with pytest.raises(SemanticManifestError, match="cannot contain scientific"):
        parse_semantic_evaluation_manifest(value)
