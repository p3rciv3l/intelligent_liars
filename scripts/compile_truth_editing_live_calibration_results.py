#!/usr/bin/env python3
"""Compile live judge cache receipts into a frozen-label calibration report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_human_calibration_pack import _LABEL_CODES  # noqa: E402
from intelligent_liars.truth_editing_judge_contracts import AbsoluteJudgeResult  # noqa: E402
from intelligent_liars.truth_editing_live_judge import (  # noqa: E402
    FROZEN_JUDGE_CONFIG_SHA256,
    FROZEN_JUDGE_RUBRIC_SHA256,
    parse_live_judge_calibration_report,
)
from intelligent_liars.truth_editing_pairwise_reconciliation import (  # noqa: E402
    MINIMUM_ORDER_CONSISTENT_HOLDOUT_REVERSALS,
    pairwise_calibration_readiness,
    summarize_pairwise_reconciliations,
)


RESPONSE_FIELDS = (
    "asserted_state", "answer_type", "semantic_type_valid", "plausible_falsehood",
    "coherent", "relevant", "internally_consistent", "instruction_following",
    "hedging_or_abstention", "confidence",
)
BUNDLE_FIELDS = (
    "false_reporting_persistent", "retained_truth_behaviorally_supported", "failure_mode",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def rate(passed: int, total: int) -> float | None:
    return None if total == 0 else passed / total


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def expected_response(code: str) -> dict[str, Any]:
    label, confidence = code.split(":", 1)
    return {**_LABEL_CODES[label], "confidence": {"L": "low", "M": "medium", "H": "high"}[confidence]}


def stable_preference(value: str, order: str) -> str:
    if value in {"tie", "invalid_comparison"}:
        return value
    if order == "AB":
        return "candidate_a" if value == "A" else "candidate_b"
    return "candidate_b" if value == "A" else "candidate_a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--live-report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--revised-pack", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = load(args.plan)
    live = parse_live_judge_calibration_report(load(args.live_report))
    labels = load(args.labels)
    pack = load(args.revised_pack)
    if live["plan_sha256"] != plan["content_sha256"]:
        raise ValueError("live report is not bound to the supplied plan")
    if labels["revised_pack_sha256"] != pack["content_sha256"]:
        raise ValueError("labels are not bound to the supplied revised pack")

    absolute_labels = {value["bundle_id"]: value for value in labels["absolute_labels"]}
    pair_labels = {value["relationship_id"]: value for value in labels["pairwise_labels"]}
    pair_source = {value["relationship_id"]: value for value in pack["pairwise_relationships"]}
    pair_plan_by_group = {
        value["relationship_sha256"]: value
        for value in plan["pairwise_relationships"]
    }
    selected_receipt_counts = Counter(live["judge_cache_receipt_sha256s"])
    found_receipts: set[str] = set()

    absolute_results: dict[str, dict[str, Any]] = {}
    pair_results: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    receipt_prices = 0.0
    receipt_tokens = Counter()
    for path in sorted(args.cache_dir.glob("*.json")):
        entry = load(path)
        receipt = entry["receipt"]
        receipt_sha = receipt["content_sha256"]
        if receipt_sha not in selected_receipt_counts:
            continue
        found_receipts.add(receipt_sha)
        receipt_prices += float(receipt["price_usd"])
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            receipt_tokens[key] += int(receipt["usage"][key])
        semantic = entry["result"]["result"]
        if entry["result_kind"] == "absolute":
            response_ids = [value["response_id"] for value in semantic["responses"]]
            bundle_ids = {value.rsplit("_r", 1)[0] for value in response_ids}
            if len(bundle_ids) != 1:
                raise ValueError("absolute cache result spans multiple authored bundles")
            absolute_results[bundle_ids.pop()] = semantic
        else:
            result = entry["result"]
            pair = pair_plan_by_group[result["comparison_group_sha256"]]
            relationship_id = pair["relationship_id"]
            pair_results[relationship_id][result["presentation_order"]] = semantic
            # Self-pairs and exact duplicates intentionally produce the same
            # paid request for AB and BA. The live runner rebinds the cached
            # semantic result to each local presentation, while the durable
            # cache stores only the first local identity. A repeated selected
            # receipt therefore accounts for both planned presentations; it is
            # not missing evidence and must never trigger a replacement call.
            if (
                pair.get("comparison_kind") in {"self_pair", "exact_duplicate"}
                and pair["presentations"] == ["AB", "BA"]
                and selected_receipt_counts[receipt_sha] == 2
            ):
                pair_results[relationship_id]["AB"] = semantic
                pair_results[relationship_id]["BA"] = semantic
    if found_receipts != set(selected_receipt_counts):
        raise ValueError("selected successful live receipts are missing from cache")

    field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    response_exact = [0, 0]
    bundle_exact = [0, 0]
    mode_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    parsed_by_mode = Counter()
    for bundle_id, semantic in absolute_results.items():
        expected = absolute_labels[bundle_id]
        expected_responses = {value["response_id"]: expected_response(value["code"]) for value in expected["response_labels"]}
        for predicted in semantic["responses"]:
            truth = expected_responses[predicted["response_id"]]
            exact = True
            for field in RESPONSE_FIELDS:
                field_counts[f"response.{field}"][1] += 1
                agreement = predicted[field] == truth[field]
                field_counts[f"response.{field}"][0] += agreement
                exact &= agreement
            response_exact[1] += 1
            response_exact[0] += exact
        exact = True
        for field in BUNDLE_FIELDS:
            field_counts[f"bundle.{field}"][1] += 1
            agreement = semantic[field] == expected[field]
            field_counts[f"bundle.{field}"][0] += agreement
            exact &= agreement
        bundle_exact[1] += 1
        bundle_exact[0] += exact
        human_mode = expected["failure_mode"]
        parsed_by_mode[human_mode] += 1
        mode_confusion[human_mode][semantic["failure_mode"]] += 1

    pair_kind_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    presentation_exact = [0, 0]
    reversal = [0, 0]
    for relationship_id, presentations in pair_results.items():
        human = pair_labels[relationship_id]["preference"]
        kind = pair_source[relationship_id]["case_kind"]
        for order, semantic in presentations.items():
            predicted = stable_preference(semantic["preference"], order)
            presentation_exact[1] += 1
            pair_kind_counts[kind][1] += 1
            agreement = predicted == human
            presentation_exact[0] += agreement
            pair_kind_counts[kind][0] += agreement
        if set(presentations) == {"AB", "BA"}:
            reversal[1] += 1
            forward = stable_preference(presentations["AB"]["preference"], "AB")
            reverse = stable_preference(presentations["BA"]["preference"], "BA")
            forward_criteria = {key: stable_preference(value, "AB") for key, value in presentations["AB"]["criterion_preferences"].items()}
            reverse_criteria = {key: stable_preference(value, "BA") for key, value in presentations["BA"]["criterion_preferences"].items()}
            reversal[0] += forward == reverse and forward_criteria == reverse_criteria

    reconciliation = summarize_pairwise_reconciliations(pair_results)
    expected_reversal_count = sum(
        set(value["presentations"]) == {"AB", "BA"}
        for value in plan["pairwise_relationships"]
    )
    pairwise_readiness = pairwise_calibration_readiness(
        reconciliation,
        expected_reversal_count=expected_reversal_count,
        minimum_order_consistent_reversals=(
            MINIMUM_ORDER_CONSISTENT_HOLDOUT_REVERSALS
        ),
    )
    pairwise_has_zero_ambiguities = pairwise_readiness.pop(
        "pairwise_holdout_has_zero_ambiguities"
    )

    selected_failures = set(live["judge_failure_receipt_sha256s"])
    selected_failure_payloads: dict[str, dict[str, Any]] = {}
    for path in args.cache_dir.glob("failures/*/*.json"):
        value = load(path)
        if value["content_sha256"] in selected_failures:
            selected_failure_payloads[value["content_sha256"]] = value
    if set(selected_failure_payloads) != selected_failures:
        raise ValueError("selected live failure receipts are missing")
    failure_categories = Counter()
    failure_details = Counter()
    # A failed pair is one failed operation but can yield one receipt per
    # presentation (and per terminal response failure). Summarize receipts
    # directly instead of assuming a one-to-one operation/receipt mapping.
    for receipt_sha in live["judge_failure_receipt_sha256s"]:
        receipt = selected_failure_payloads[receipt_sha]
        failure_categories[f"{receipt['operational_status']}:{receipt['operational_failure']['code']}"] += 1
        request_dir = args.attempt_dir / receipt["raw_request_sha256"]
        matched_content: str | None = None
        for completed in request_dir.glob("*/completed.json"):
            response = load(completed)["response"]
            raw_payload = response.get("raw_payload", response)
            if digest(raw_payload) == receipt["raw_response_sha256"]:
                matched_content = response.get("content")
                break
        detail = receipt["operational_failure"]["message"]
        if isinstance(matched_content, str):
            try:
                semantic = json.loads(matched_content)
                unsigned = {
                    "format": "truth_editing_absolute_judge_result_v1",
                    "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
                    "rubric_sha256": FROZEN_JUDGE_RUBRIC_SHA256,
                    "request_sha256": receipt["raw_request_sha256"],
                    "operational_status": "succeeded", "operational_failure": None,
                    "result": semantic,
                }
                AbsoluteJudgeResult.parse({**unsigned, "content_sha256": digest(unsigned)})
                detail = "post_parse_identity_or_response_id_mismatch"
            except json.JSONDecodeError:
                detail = "json_decode_error"
            except Exception as error:
                detail = str(error)
        failure_details[detail] += 1

    expected_mode_counts = Counter(value["failure_mode"] for value in absolute_labels.values())
    strict_abs = len(absolute_results)
    strict_pair_presentations = sum(len(value) for value in pair_results.values())
    strict_total = strict_abs + strict_pair_presentations
    required_call_count = live["planned_paid_calls"]
    gates = {
        "all_planned_presentations_accounted": strict_total + len(live["failed_operation_ids"]) == required_call_count,
        "strict_parse_coverage_complete": strict_total == required_call_count,
        "zero_terminal_schema_or_json_failures": len(live["failed_operation_ids"]) == 0,
        "all_absolute_failure_modes_have_parsed_examples": set(parsed_by_mode) == set(expected_mode_counts),
        "pairwise_all_presentations_parsed": strict_pair_presentations == sum(len(value["presentations"]) for value in plan["pairwise_relationships"]),
        **pairwise_readiness,
    }
    gates["calibration_usable_for_optimization"] = all(gates.values())

    unsigned = {
        "format": "truth_editing_live_policy_calibration_report_v2",
        "source": {
            "plan_sha256": plan["content_sha256"],
            "live_report_sha256": live["content_sha256"],
            "labels_sha256": labels["content_sha256"],
            "revised_pack_sha256": pack["content_sha256"],
            "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
            "provider_route": live["provider_route"],
        },
        "operational": {
            "planned_presentations": required_call_count,
            "paid_calls": live["actual_paid_calls"],
            "strict_parse_successes": strict_total,
            "strict_parse_failures": len(live["failed_operation_ids"]),
            "absolute_parsed": strict_abs,
            "absolute_total": len(absolute_labels),
            "pairwise_presentations_parsed": strict_pair_presentations,
            "pairwise_presentations_total": sum(len(value["presentations"]) for value in plan["pairwise_relationships"]),
            "spend_usd": live["actual_spend_usd"],
            "successful_receipt_spend_usd": receipt_prices,
            "successful_tokens": dict(receipt_tokens),
            "failure_categories": dict(sorted(failure_categories.items())),
            "failure_details": dict(sorted(failure_details.items())),
        },
        "absolute": {
            "strict_parse_coverage": rate(strict_abs, len(absolute_labels)),
            "response_exact": {"passed": response_exact[0], "total": response_exact[1], "rate": rate(*response_exact)},
            "bundle_exact": {"passed": bundle_exact[0], "total": bundle_exact[1], "rate": rate(*bundle_exact)},
            "field_agreement": {key: {"passed": value[0], "total": value[1], "rate": rate(*value)} for key, value in sorted(field_counts.items())},
            "failure_mode_confusion": {key: dict(sorted(value.items())) for key, value in sorted(mode_confusion.items())},
            "expected_by_failure_mode": dict(sorted(expected_mode_counts.items())),
            "parsed_by_failure_mode": dict(sorted(parsed_by_mode.items())),
        },
        "pairwise": {
            "presentation_exact": {"passed": presentation_exact[0], "total": presentation_exact[1], "rate": rate(*presentation_exact)},
            "by_case_kind": {key: {"passed": value[0], "total": value[1], "rate": rate(*value)} for key, value in sorted(pair_kind_counts.items())},
            "reversal_consistency": {"passed": reversal[0], "total": reversal[1], "rate": rate(*reversal)},
            "reconciliation": reconciliation,
            "minimum_order_consistent_holdout_reversals": (
                MINIMUM_ORDER_CONSISTENT_HOLDOUT_REVERSALS
            ),
            "quality_flags": {
                "all_reversals_consistent": reversal[0] == reversal[1],
                "has_ambiguous_reversals": reconciliation["ambiguous_count"] > 0,
                "has_zero_ambiguous_reversals": pairwise_has_zero_ambiguities,
            },
        },
        "gates": gates,
    }
    payload = {**unsigned, "content_sha256": digest(unsigned)}
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(canonical(payload) + "\n")
    print(canonical({"content_sha256": payload["content_sha256"], "gates": gates, "operational": payload["operational"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
