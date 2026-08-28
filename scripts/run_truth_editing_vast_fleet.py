#!/usr/bin/env python3
"""Dry-run or execute the controller-only Optuna Vast production fleet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from intelligent_liars.truth_editing_directions import DirectionBank
from intelligent_liars.truth_editing_production import (
    ImmutableStudyArtifactAdapter,
    ProductionRunConfig,
    ProductionTruthEditingRun,
)
from intelligent_liars.truth_editing_study import (
    OfflineDeterministicSearchDriver,
    OptunaSearchDriver,
    TruthEditingStudy,
    load_truth_editing_study_config,
)
from intelligent_liars.truth_editing_vast_fleet import (
    FleetBatchEvaluator,
    FleetConfig,
    PHASE_BOUNDARIES,
    VastLifecycleTrialWorker,
)
from intelligent_liars.truth_editing_vast_prerequisites import Offer
from intelligent_liars.truth_editing_vast_prerequisites import EphemeralWorkloadSecret
from intelligent_liars.truth_editing_vast_production import ProductionVastConfig


DEFAULT_VASTAI = str(Path(__file__).resolve().with_name("vastai_openai_ua.py"))


def _offers(path: Path, count: int) -> tuple[Offer, ...]:
    raw = json.loads(path.read_text())
    rows = raw if isinstance(raw, list) else [raw]
    if len(rows) < count:
        raise ValueError(f"offer JSON needs at least {count} independent offers")
    offers = tuple(Offer.from_mapping(item) for item in rows[:count])
    if len({item.offer_id for item in offers}) != len(offers):
        raise ValueError("fleet offer IDs must be unique")
    return offers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--fleet-config", type=Path, required=True)
    parser.add_argument("--production-job", type=Path, required=True)
    parser.add_argument("--offers-json", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fetch-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASE_BOUNDARIES), default="finalist")
    parser.add_argument(
        "--vastai",
        default=DEFAULT_VASTAI,
        help="Vast CLI wrapper that applies the required HTTP user agent",
    )
    parser.add_argument("--ssh-identity", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmed-cost-approval", action="store_true")
    args = parser.parse_args(argv)

    fleet_config = FleetConfig.from_mapping(json.loads(args.fleet_config.read_text()))
    job = ProductionVastConfig.from_mapping(json.loads(args.production_job.read_text()))
    offers = _offers(args.offers_json, fleet_config.worker_count)
    stop_after = fleet_config.stop_after_trials(args.phase)
    workload_secret = (
        EphemeralWorkloadSecret.openrouter(os.environ.get("OPENROUTER_API_KEY", ""))
        if args.execute
        else None
    )
    plan = {
        "mode": "dry_run",
        "fleet_config_sha256": fleet_config.identity_sha256,
        "production_config_sha256": fleet_config.production_config_sha256,
        "bundle_sha256": fleet_config.bundle_sha256,
        "worker_count": fleet_config.worker_count,
        "offer_ids": [item.offer_id for item in offers],
        "phase": args.phase,
        "stop_after_trials": stop_after,
        "phase_boundaries": dict(PHASE_BOUNDARIES),
        "budget_identity_sha256": fleet_config.budget_identity_sha256,
        "production_judge_budget_config_sha256": (
            fleet_config.production_judge_budget_config_sha256
        ),
        "all_in_maximum_spend_usd": str(
            fleet_config.all_in_maximum_spend_usd
        ),
        "maximum_infrastructure_spend_usd": str(
            fleet_config.maximum_infrastructure_spend_usd
        ),
        "maximum_judge_spend_usd": str(fleet_config.maximum_judge_spend_usd),
        "maximum_host_lease_seconds": fleet_config.maximum_host_lease_seconds,
        "maximum_fetch_gib": fleet_config.maximum_fetch_gib,
        "capability_test_access": False,
    }
    def worker_factory(slot: int) -> VastLifecycleTrialWorker:
        return VastLifecycleTrialWorker(
            slot=slot,
            fleet_config=fleet_config,
            production_job=job,
            offer=offers[slot],
            repo=args.repo,
            bundle=args.bundle,
            fetch_root=args.fetch_root,
            metadata_root=args.metadata_root,
            workload_secret=workload_secret,
            vastai=args.vastai,
            ssh_identity=args.ssh_identity,
        )

    # Construction is non-mutating and verifies the exact v3 file, bundle,
    # worker entrypoint, and all per-instance resource ceilings.
    for slot in range(fleet_config.worker_count):
        worker_factory(slot).close()
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.confirmed_cost_approval:
        raise SystemExit("--execute requires --confirmed-cost-approval")

    production_path = args.repo / fleet_config.production_config_path
    production = ProductionRunConfig.open(production_path)
    study_config = load_truth_editing_study_config(production.study_config)
    observed_boundaries = {
        tier.name: tier.through_trial for tier in study_config.evaluation_tiers
    }
    if observed_boundaries != PHASE_BOUNDARIES or study_config.max_trials != 200:
        raise ValueError("production v3 study must preserve exact 80/160/200 phases")
    bank = DirectionBank.open(production.direction_manifest, root=production.direction_root)
    driver = (
        OptunaSearchDriver(seed=study_config.sampler_seed)
        if production.search_driver == "optuna"
        else OfflineDeterministicSearchDriver(seed=study_config.sampler_seed)
    )

    evaluator = FleetBatchEvaluator(fleet_config, worker_factory=worker_factory)
    run = ProductionTruthEditingRun(
        study=TruthEditingStudy(study_config, bank.manifest),
        driver=driver,
        evaluator=evaluator,  # type: ignore[arg-type]
        artifacts=ImmutableStudyArtifactAdapter(production.artifact_dir),
        journal_path=production.journal_path,
    )
    with evaluator:
        receipt = run.run(stop_after_trials=stop_after)
    result = dict(plan)
    result["mode"] = "executed"
    result["receipt"] = receipt.to_mapping()
    result["run_receipt_sha256"] = receipt.identity_sha256
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
