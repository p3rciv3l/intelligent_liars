#!/usr/bin/env python3
"""Validate an inert Step 5 canary launch packet; never execute it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelligent_liars.step5_launch_packet import validate_launch_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success for a valid but substitution-blocked inert packet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet_path = args.packet.resolve()
    payload = json.loads(packet_path.read_text())
    result = validate_launch_packet(payload, packet_dir=packet_path.parent)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["launch_ready"] or args.allow_incomplete:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
