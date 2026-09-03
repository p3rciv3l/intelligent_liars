from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from intelligent_liars.truth_editing_preservation_runtime import (
    PreservationRuntimeReceipt,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_truth_editing_local_mps_canary.py"
SPEC = importlib.util.spec_from_file_location("run_truth_editing_local_mps_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _trial(trial_id: str, outcome: str) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "result": {"outcome_kind": outcome, "metrics": {}, "detail": outcome},
    }


def test_synthetic_appended_preservation_receipt_is_strictly_parseable() -> None:
    batch = SimpleNamespace(
        examples=tuple(range(15)),
        batch_sha256="a" * 64,
        recipe_id="recipe-canary",
        model_sha256="b" * 64,
        basis_set=SimpleNamespace(basis_set_sha256="c" * 64),
    )
    receipt = MODULE._MpsCanaryPreservationCollector().collect(None, batch)
    parsed = PreservationRuntimeReceipt.from_mapping(receipt)
    assert parsed.tier == "finalist"
    assert parsed.batch_sha256 == batch.batch_sha256
    assert parsed.preservation_receipt["aggregate_kl"] == 0.0


def test_audit_requires_exact_source_prefix_and_accounts_every_trial(
    tmp_path: Path,
) -> None:
    source_trials = [
        _trial("trial-0000", "scientifically_infeasible"),
        _trial("trial-0001", "operational_failure"),
        _trial("trial-0002", "scientifically_infeasible"),
        _trial("trial-0003", "scientifically_infeasible"),
    ]
    target_trials = [
        *source_trials,
        *[
            _trial(f"trial-{index:04d}", "scientifically_infeasible")
            for index in range(4, 8)
        ],
    ]
    (tmp_path / "source-report-4.json").write_text(
        json.dumps({"completed_trials": 4, "trials": source_trials})
    )
    (tmp_path / "target-report-8.json").write_text(
        json.dumps({"completed_trials": 8, "trials": target_trials})
    )
    (tmp_path / "rescore-generation.json").write_text(
        json.dumps({"replay_requests": [{"request_sha256": "d" * 64}]})
    )

    result = MODULE.audit(Namespace(output_dir=tmp_path))

    assert result["all_trials_terminally_accounted"] is True
    assert result["source_history_preserved"] is True
    assert result["rescore_request_count"] == 1
    assert result["target_outcomes"] == {
        "operational_failure": 1,
        "scientifically_infeasible": 7,
    }
    assert (tmp_path / "audit.json").is_file()
