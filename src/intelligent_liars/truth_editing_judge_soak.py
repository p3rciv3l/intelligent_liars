"""Build identity-bound load-test plans from an approved judge calibration plan.

The derived plan repeats semantic cases only to exercise provider transport,
schema adherence, concurrency, caching, and crash-resume behavior.  It is not a
new calibration dataset and its outcomes must never be used as scientific
labels or Optuna observations.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    LIVE_CALIBRATION_PLAN_FORMAT,
    LiveJudgeError,
    _load_calibration_plan,
)


SOAK_PLAN_GENERATOR_FORMAT = "truth_editing_live_judge_soak_generator_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _clone_bundle(bundle: Mapping[str, Any], cohort: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(bundle))
    suffix = f"soak-{cohort:04d}"
    result["bundle_id"] = f"{result['bundle_id']}--{suffix}"
    responses = result.get("responses")
    if not isinstance(responses, list):
        raise LiveJudgeError("soak source bundle responses are invalid")
    for response in responses:
        if not isinstance(response, dict) or not isinstance(
            response.get("response_id"), str
        ):
            raise LiveJudgeError("soak source response identity is invalid")
        response["response_id"] = f"{response['response_id']}--{suffix}"
    unsigned = {key: value for key, value in result.items() if key != "bundle_sha256"}
    result["bundle_sha256"] = _sha(unsigned)
    return result


def build_live_judge_soak_plan(
    source_plan: Mapping[str, Any] | str,
    *,
    planned_request_presentations: int,
    maximum_spend_usd: float = 5.0,
) -> dict[str, Any]:
    """Derive exactly the requested production-shaped presentations.

    Presentations are workload units, not a promised transport-call count:
    the paid adapter may cache-deduplicate identical blinded requests.
    """

    base = _load_calibration_plan(source_plan)
    if base["format"] != LIVE_CALIBRATION_PLAN_FORMAT:
        raise LiveJudgeError("judge soak requires the current pairwise plan contract")
    if (
        isinstance(planned_request_presentations, bool)
        or not isinstance(planned_request_presentations, int)
        or not 1 <= planned_request_presentations <= 2000
    ):
        raise LiveJudgeError("planned_request_presentations must be from 1 through 2000")
    if maximum_spend_usd != 5.0:
        raise LiveJudgeError("judge soak maximum_spend_usd must be exactly 5")
    base_absolutes = base["absolute_bundles"]
    base_pairs = base["pairwise_relationships"]
    assert isinstance(base_absolutes, list) and isinstance(base_pairs, list)
    base_presentation_count = len(base_absolutes) + sum(
        len(pair["presentations"]) for pair in base_pairs
    )
    if base_presentation_count <= 0:
        raise LiveJudgeError("judge soak source plan has no operations")
    full_cohorts, remainder = divmod(
        planned_request_presentations, base_presentation_count
    )
    if remainder > len(base_absolutes):
        raise LiveJudgeError(
            "planned_request_presentations remainder cannot preserve whole pairwise relationships"
        )

    absolutes: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for cohort in range(full_cohorts + (1 if remainder else 0)):
        cloned: dict[str, dict[str, Any]] = {
            str(bundle["bundle_id"]): _clone_bundle(bundle, cohort)
            for bundle in base_absolutes
        }
        cohort_absolutes = list(cloned.values())
        if cohort == full_cohorts and remainder:
            absolutes.extend(cohort_absolutes[:remainder])
            break
        absolutes.extend(cohort_absolutes)
        for pair in base_pairs:
            result = copy.deepcopy(dict(pair))
            result["relationship_id"] = (
                f"{result['relationship_id']}--soak-{cohort:04d}"
            )
            for candidate_name in ("candidate_a", "candidate_b"):
                source_candidate = pair[candidate_name]
                source_id = str(source_candidate["bundle_id"])
                if source_id not in cloned:
                    cloned[source_id] = _clone_bundle(source_candidate, cohort)
                result[candidate_name] = cloned[source_id]
            unsigned_pair = {
                key: value
                for key, value in result.items()
                if key != "relationship_sha256"
            }
            result["relationship_sha256"] = _sha(unsigned_pair)
            pairs.append(result)

    generator = {
        "format": SOAK_PLAN_GENERATOR_FORMAT,
        "source_plan_sha256": base["content_sha256"],
        "planned_request_presentations": planned_request_presentations,
    }
    unsigned = {
        "format": LIVE_CALIBRATION_PLAN_FORMAT,
        "calibration_id": f"transport-soak-{planned_request_presentations}-presentations",
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "maximum_spend_usd": maximum_spend_usd,
        "source_identities": {
            "revised_pack_sha256": str(base["content_sha256"]),
            "labels_sha256": _sha(generator),
            "provenance_sha256": _sha(
                {"generator": generator, "source_identities": base["source_identities"]}
            ),
        },
        "absolute_bundles": absolutes,
        "pairwise_relationships": pairs,
    }
    result = {**unsigned, "content_sha256": _sha(unsigned)}
    # Reopen through the execution parser so the builder cannot publish a plan
    # the paid runner would interpret differently.
    _load_calibration_plan(result)
    actual_presentations = len(absolutes) + sum(
        len(pair["presentations"]) for pair in pairs
    )
    if actual_presentations != planned_request_presentations:
        raise LiveJudgeError("judge soak builder produced the wrong presentation count")
    return result


__all__ = ["SOAK_PLAN_GENERATOR_FORMAT", "build_live_judge_soak_plan"]
