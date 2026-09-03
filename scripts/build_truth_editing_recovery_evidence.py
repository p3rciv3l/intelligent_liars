#!/usr/bin/env python3
"""Sanitize a judge-soak runtime into a recomputable, prompt-free inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_content_hash(value: dict[str, Any], label: str) -> None:
    claimed = value.get("content_sha256")
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    if claimed != _sha(unsigned):
        raise RuntimeError(f"{label} content identity differs")


def build(root: Path) -> dict[str, Any]:
    plan = _read(root / "plan.json")
    report_path = root / "live-report.json"
    replay_path = root / "replay-report-after-bounded-admission.json"
    report = _read(report_path)
    _validate_content_hash(plan, "plan")
    _validate_content_hash(report, "report")
    if report.get("plan_sha256") != plan["content_sha256"]:
        raise RuntimeError("report is not bound to the sanitized plan")
    if report != _read(replay_path):
        raise RuntimeError("final replay report is not byte-content equivalent")

    attempts: list[dict[str, Any]] = []
    for request_dir in sorted(path for path in (root / "attempts").iterdir() if path.is_dir()):
        for attempt_dir in sorted(path for path in request_dir.iterdir() if path.is_dir()):
            states: dict[str, dict[str, Any]] = {}
            for name in ("pending", "failed", "completed", "processed"):
                path = attempt_dir / f"{name}.json"
                if path.is_file():
                    payload = _read(path)
                    _validate_content_hash(payload, f"{request_dir.name}/{attempt_dir.name}/{name}")
                    if (
                        payload.get("plan_sha256") != plan["content_sha256"]
                        or payload.get("request_sha256") != request_dir.name
                        or payload.get("status") != name
                    ):
                        raise RuntimeError("attempt event identity or status differs")
                    states[name] = {
                        "content_sha256": payload["content_sha256"],
                        "mtime_ns": path.stat().st_mtime_ns,
                        **(
                            {"authorized_usd": payload["authorized_usd"]}
                            if name == "pending"
                            else {}
                        ),
                        **(
                            {"actual_usd": payload["actual_usd"]}
                            if name == "completed"
                            else {}
                        ),
                        **(
                            {"error_class": payload["error_class"]}
                            if name == "failed"
                            else {}
                        ),
                    }
            attempts.append(
                {
                    "request_sha256": request_dir.name,
                    "attempt": attempt_dir.name,
                    "states": states,
                }
            )

    completed_attempts = [item for item in attempts if "completed" in item["states"]]
    pending_only = [
        item
        for item in attempts
        if "pending" in item["states"]
        and "completed" not in item["states"]
        and "failed" not in item["states"]
    ]
    kill_boundary_ns = max(
        item["states"]["pending"]["mtime_ns"] for item in pending_only
    )
    completed_before_kill_boundary = sum(
        item["states"]["completed"]["mtime_ns"] <= kill_boundary_ns
        for item in completed_attempts
    )
    completed_spend = sum(
        float(item["states"]["completed"]["actual_usd"])
        for item in completed_attempts
    )
    if (
        len(completed_attempts) != report["actual_paid_calls"]
        or abs(completed_spend - report["actual_spend_usd"]) > 1e-12
    ):
        raise RuntimeError("attempt inventory differs from the live report")
    operation_inventory = {
        "completed_ids": report["completed_operation_ids"],
        "failed_ids": report["failed_operation_ids"],
    }
    evidence = {
        "format": "truth_editing_judge_soak_sanitized_evidence_v1",
        "privacy": "request hashes and receipt hashes only; no prompts or responses",
        "plan": {
            "content_sha256": plan["content_sha256"],
            "absolute_bundle_identities": [
                {"id": item["bundle_id"], "sha256": item["bundle_sha256"]}
                for item in plan["absolute_bundles"]
            ],
            "pairwise_relationship_identities": [
                {
                    "id": item["relationship_id"],
                    "sha256": item["relationship_sha256"],
                    "presentations": item["presentations"],
                }
                for item in plan["pairwise_relationships"]
            ],
        },
        "attempts": attempts,
        "completed_attempt_count": len(completed_attempts),
        "completed_attempt_spend_usd": completed_spend,
        "ambiguous_pending_request_sha256s": [
            item["request_sha256"] for item in pending_only
        ],
        "kill_boundary_derivation": {
            "boundary_mtime_ns": kill_boundary_ns,
            "completed_attempts_at_or_before_boundary": completed_before_kill_boundary,
            "unresolved_pending_attempts_at_boundary": len(pending_only),
        },
        "operation_inventory": operation_inventory,
        "operation_inventory_sha256": _sha(operation_inventory),
        "result_receipt_sha256s": report["judge_cache_receipt_sha256s"],
        "failure_receipt_sha256s": report["judge_failure_receipt_sha256s"],
        "correction_lineage_sha256s": report["semantic_correction_lineage_sha256s"],
        "report_content_sha256": report["content_sha256"],
        "report_file_sha256": _file_sha(report_path),
        "replay_report_file_sha256": _file_sha(replay_path),
        "replay_report_byte_identical": report_path.read_bytes() == replay_path.read_bytes(),
    }
    evidence["content_sha256"] = _sha(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = build(args.runtime_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
