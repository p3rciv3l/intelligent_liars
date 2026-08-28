#!/usr/bin/env python3
"""Long-lived one-GPU JSON-lines worker for a persistent eight-GPU host."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from intelligent_liars.truth_editing_production import (  # noqa: E402
    ProductionCompositionError,
    ProductionRunConfig,
    open_production_run,
)
from intelligent_liars.truth_editing_study import (  # noqa: E402
    SearchProposal,
    load_truth_editing_study_config,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _result(request: dict[str, Any], run: Any, production: ProductionRunConfig) -> dict[str, Any]:
    unsigned = dict(request)
    claimed = unsigned.pop("request_sha256", None)
    if claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("fleet request identity differs")
    study = load_truth_editing_study_config(production.study_config)
    ordinal = request["ordinal"]
    tier = next(item for item in study.evaluation_tiers if ordinal + 1 <= item.through_trial)
    record_ids = tuple(study.validation_record_ids[:tier.record_limit])
    if tuple(request["record_ids"]) != record_ids:
        raise ValueError("fleet request attempted capability-test access")
    outcome = run._evaluator.evaluate(
        SearchProposal.from_dict(request["proposal"]),
        trial_id=request["trial_id"],
        record_ids=record_ids,
        objective_names=study.objective_names,
    )
    telemetry = dict(run._evaluator.last_runtime_telemetry)
    return {
        "request_sha256": claimed,
        "result": {
            "outcome_kind": outcome.outcome_kind,
            "metrics": dict(outcome.metrics),
            "detail": outcome.detail,
        },
        "telemetry": telemetry,
    }


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--config":
        raise SystemExit("usage: run_truth_editing_cuda_fleet_worker.py --config CONFIG")
    config_path = Path(sys.argv[2])
    production = ProductionRunConfig.open(config_path)
    run = open_production_run(config_path)
    for line in sys.stdin:
        request = json.loads(line)
        if request == {"command": "stop"}:
            return 0
        try:
            response = _result(request, run, production)
        except ProductionCompositionError as error:
            prefix = "paid semantic judge failed closed; failure_receipt_sha256="
            if prefix in str(error):
                digest = str(error).split(prefix, 1)[1].split()[0]
                print(json.dumps({"fatal": True, "failure_receipt_sha256": digest}), flush=True)
                return 86
            raise
        print(json.dumps(response, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
