#!/usr/bin/env python3
"""Measure a Vast host's download path and emit a Step 5 gate report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from intelligent_liars.vast_host_gate import (
    FailureDomain,
    HostGateThresholds,
    decide_machine_action,
    evaluate_host_gate,
    measure_download_url,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id")
    parser.add_argument("--offer-id")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--sample-mib", type=int, default=64)
    parser.add_argument("--chunk-mib", type=int, default=1)
    parser.add_argument("--min-download-mbps", type=float, default=100.0)
    parser.add_argument("--max-ttfb-seconds", type=float, default=5.0)
    parser.add_argument("--max-stall-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _redacted_url(url: str) -> str:
    """Keep useful endpoint provenance without persisting credentials or tokens."""

    parts = urlsplit(url)
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


def main() -> int:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.sample_mib <= 0 or args.chunk_mib <= 0:
        raise ValueError("--sample-mib and --chunk-mib must be positive")

    sample_bytes = args.sample_mib * 1024**2
    thresholds = HostGateThresholds(
        min_download_mbps=args.min_download_mbps,
        min_sample_bytes=sample_bytes,
        max_time_to_first_byte_seconds=args.max_ttfb_seconds,
        max_stall_seconds=args.max_stall_seconds,
    )
    trials = [
        measure_download_url(
            args.download_url,
            sample_bytes=sample_bytes,
            chunk_bytes=args.chunk_mib * 1024**2,
            max_stall_seconds=args.max_stall_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        for _ in range(args.trials)
    ]
    decision = evaluate_host_gate(trials, thresholds)
    lifecycle = decide_machine_action(
        host_accepted=decision.accepted,
        workload_started=False,
        workload_succeeded=False,
        artifacts_durable=False,
        failure_domain=FailureDomain.NONE if decision.accepted else FailureDomain.HOST,
        diagnosis_complete=True,
        resume_possible=False,
    )
    report = {
        "format": "tinylora_step5_vast_host_gate_v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": args.instance_id,
        "offer_id": args.offer_id,
        "download_url": _redacted_url(args.download_url),
        "thresholds": {
            "min_download_mbps": thresholds.min_download_mbps,
            "min_sample_bytes": thresholds.min_sample_bytes,
            "max_time_to_first_byte_seconds": thresholds.max_time_to_first_byte_seconds,
            "max_stall_seconds": thresholds.max_stall_seconds,
        },
        "decision": decision.to_dict(),
        "lifecycle": lifecycle.to_dict(),
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if decision.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
