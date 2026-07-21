#!/usr/bin/env python3
"""Find the cheapest sufficient Vast.ai offer for a GPU workload."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_VASTAI = "/Users/student/Library/Python/3.10/bin/vastai"
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"


@dataclass
class Requirements:
    min_vram: float
    min_disk: float
    min_reliability: float
    min_inet_down: float
    min_inet_up: float
    min_cuda: float | None
    min_compute_cap: int | None
    num_gpus: int
    allow_unverified: bool
    allow_rented: bool
    allow_indirect_ssh: bool
    max_hourly: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastai-path", default=os.environ.get("VASTAI_PATH", DEFAULT_VASTAI))
    parser.add_argument("--raw-offers", type=Path, help="Read offer JSON from a file instead of calling Vast.")
    parser.add_argument("--min-vram", type=float, default=16.0, help="Minimum per-GPU VRAM in GB.")
    parser.add_argument("--min-disk", type=float, default=60.0, help="Minimum local disk in GB.")
    parser.add_argument("--min-reliability", type=float, default=0.98)
    parser.add_argument("--min-inet-down", type=float, default=50.0, help="Minimum download Mbps.")
    parser.add_argument("--min-inet-up", type=float, default=20.0, help="Minimum upload Mbps.")
    parser.add_argument("--min-cuda", type=float, default=None, help="Minimum CUDA version.")
    parser.add_argument("--min-compute-cap", type=int, default=None, help="Minimum CUDA compute capability * 100.")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--estimated-hours", type=float, default=1.0)
    parser.add_argument("--max-hourly", type=float, default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--gpu-budget", choices=("cheap", "balanced", "high-vram", "any"), default="cheap")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--label", default="codex-vast-YYYYMMDD-HHMMSS")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--allow-rented", action="store_true")
    parser.add_argument("--allow-indirect-ssh", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable recommendation JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Accepted for workflow symmetry; this script never rents.")
    return parser.parse_args()


def number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def vram_gb(offer: dict[str, Any]) -> float:
    raw = number(offer.get("gpu_ram"), 0.0)
    if raw > 1024:
        return raw / 1024.0
    return raw


def hourly_price(offer: dict[str, Any]) -> float:
    keys = ("dph_total", "discounted_dph_total", "dph", "dph_base", "min_bid")
    for key in keys:
        value = number(offer.get(key), -1.0)
        if value >= 0:
            return value
    return 999999.0


def offer_id(offer: dict[str, Any]) -> Any:
    return offer.get("id") or offer.get("ask_contract_id")


def gpu_class_rank(gpu_name: str) -> int:
    name = gpu_name.upper()
    if any(token in name for token in ("B200", "H200", "H100")):
        return 80
    if "A100" in name:
        return 70
    if any(token in name for token in ("L40", "L40S")):
        return 45
    if "L4" in name:
        return 40
    if any(token in name for token in ("A6000", "RTX 6000")):
        return 35
    if any(token in name for token in ("A5000", "RTX 5000")):
        return 30
    if "4090" in name:
        return 25
    if "3090" in name:
        return 20
    if any(token in name for token in ("3080", "4080", "2080", "A4000", "A4500")):
        return 15
    return 50


def build_query(req: Requirements) -> str:
    parts = [
        "rentable=true",
        f"num_gpus={req.num_gpus}",
        f"gpu_ram>={req.min_vram:g}",
        f"disk_space>={req.min_disk:g}",
        f"reliability>={req.min_reliability:g}",
        f"inet_down>={req.min_inet_down:g}",
        f"inet_up>={req.min_inet_up:g}",
    ]
    if not req.allow_unverified:
        parts.append("verified=true")
    if not req.allow_rented:
        parts.append("rented=false")
    if not req.allow_indirect_ssh:
        parts.append("direct_port_count>=1")
    if req.min_cuda is not None:
        parts.append(f"cuda_vers>={req.min_cuda:g}")
    if req.min_compute_cap is not None:
        parts.append(f"compute_cap>={req.min_compute_cap:d}")
    if req.max_hourly is not None:
        parts.append(f"dph_total<={req.max_hourly:g}")
    return " ".join(parts)


def load_offers(args: argparse.Namespace, query: str) -> list[dict[str, Any]]:
    if args.raw_offers:
        data = json.loads(args.raw_offers.read_text())
    else:
        cmd = [
            args.vastai_path,
            "search",
            "offers",
            query,
            "--raw",
            "--limit",
            str(args.limit),
            "-o",
            "dph_total",
        ]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
        except FileNotFoundError:
            raise SystemExit(f"Vast CLI not found: {args.vastai_path}")
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(exc.stderr or exc.stdout)
            raise SystemExit(exc.returncode)
        data = json.loads(proc.stdout)
    if isinstance(data, dict) and "offers" in data:
        data = data["offers"]
    if not isinstance(data, list):
        raise SystemExit("Expected a JSON list of offers.")
    return [offer for offer in data if isinstance(offer, dict)]


def reject_reason(offer: dict[str, Any], req: Requirements) -> str | None:
    if int(number(offer.get("num_gpus"), 0)) != req.num_gpus:
        return "gpu-count"
    if vram_gb(offer) < req.min_vram:
        return "vram"
    if number(offer.get("disk_space"), 0) < req.min_disk:
        return "disk"
    if number(offer.get("reliability"), 0) < req.min_reliability:
        return "reliability"
    if not req.allow_indirect_ssh and number(offer.get("direct_port_count"), 0) < 1:
        return "direct-ssh"
    if number(offer.get("inet_down"), 0) < req.min_inet_down:
        return "download"
    if number(offer.get("inet_up"), 0) < req.min_inet_up:
        return "upload"
    if not req.allow_unverified and offer.get("verified") is False:
        return "unverified"
    if not req.allow_rented and offer.get("rented") is True:
        return "rented"
    if req.min_cuda is not None:
        cuda = number(offer.get("cuda_vers", offer.get("cuda_max_good")), 0)
        if cuda < req.min_cuda:
            return "cuda"
    if req.min_compute_cap is not None and number(offer.get("compute_cap"), 0) < req.min_compute_cap:
        return "compute-cap"
    if req.max_hourly is not None and hourly_price(offer) > req.max_hourly:
        return "hourly"
    return None


def rank_key(offer: dict[str, Any], estimated_hours: float, gpu_budget: str) -> tuple[float, float, int, float]:
    price = hourly_price(offer)
    expected_cost = price * estimated_hours
    reliability_penalty = max(0.0, 1.0 - number(offer.get("reliability"), 1.0)) * price
    class_rank = gpu_class_rank(str(offer.get("gpu_name", "")))
    if gpu_budget == "cheap":
        prestige_penalty = max(0, class_rank - 45) * 0.00001
    elif gpu_budget == "high-vram":
        prestige_penalty = 0
    else:
        prestige_penalty = max(0, class_rank - 70) * 0.000005
    return (expected_cost + reliability_penalty + prestige_penalty, price, class_rank, -number(offer.get("dlperf"), 0))


def summarize_offer(offer: dict[str, Any], estimated_hours: float) -> dict[str, Any]:
    price = hourly_price(offer)
    return {
        "offer_id": offer_id(offer),
        "gpu_name": offer.get("gpu_name"),
        "num_gpus": offer.get("num_gpus"),
        "vram_gb": round(vram_gb(offer), 1),
        "hourly_price": round(price, 4),
        "estimated_total_cost": round(price * estimated_hours, 4),
        "disk_gb": offer.get("disk_space"),
        "reliability": offer.get("reliability"),
        "inet_down": offer.get("inet_down"),
        "inet_up": offer.get("inet_up"),
        "cuda": offer.get("cuda_vers", offer.get("cuda_max_good")),
        "compute_cap": offer.get("compute_cap"),
        "location": offer.get("geolocation"),
    }


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def main() -> int:
    args = parse_args()
    req = Requirements(
        min_vram=args.min_vram,
        min_disk=args.min_disk,
        min_reliability=args.min_reliability,
        min_inet_down=args.min_inet_down,
        min_inet_up=args.min_inet_up,
        min_cuda=args.min_cuda,
        min_compute_cap=args.min_compute_cap,
        num_gpus=args.num_gpus,
        allow_unverified=args.allow_unverified,
        allow_rented=args.allow_rented,
        allow_indirect_ssh=args.allow_indirect_ssh,
        max_hourly=args.max_hourly,
    )
    query = build_query(req)
    offers = load_offers(args, query)
    sufficient = [offer for offer in offers if reject_reason(offer, req) is None]
    sufficient.sort(key=lambda offer: rank_key(offer, args.estimated_hours, args.gpu_budget))

    if not sufficient:
        rejected: dict[str, int] = {}
        for offer in offers:
            reason = reject_reason(offer, req) or "unknown"
            rejected[reason] = rejected.get(reason, 0) + 1
        payload = {"query": query, "found": len(offers), "sufficient": 0, "rejected": rejected}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("No sufficient Vast offers found.")
            print(f"Query: {query}")
            print(f"Rejected counts: {json.dumps(rejected, sort_keys=True)}")
        return 2

    best = sufficient[0]
    create_cmd = [
        args.vastai_path,
        "create",
        "instance",
        str(offer_id(best)),
        "--image",
        args.image,
        "--disk",
        str(int(args.min_disk)),
        "--label",
        args.label,
        "--ssh",
        "--direct",
        "--raw",
    ]
    cleanup_cmd = [args.vastai_path, "destroy", "instance", "INSTANCE_ID", "--raw"]
    payload = {
        "query": query,
        "found": len(offers),
        "sufficient": len(sufficient),
        "recommendation": summarize_offer(best, args.estimated_hours),
        "image": args.image,
        "create_command": shell_join(create_cmd),
        "cleanup_command": shell_join(cleanup_cmd),
        "top_offers": [summarize_offer(offer, args.estimated_hours) for offer in sufficient[:10]],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        rec = payload["recommendation"]
        print("Recommended Vast offer")
        print(f"  offer_id: {rec['offer_id']}")
        print(f"  gpu: {rec['num_gpus']}x {rec['gpu_name']} ({rec['vram_gb']} GB VRAM)")
        print(f"  price: ${rec['hourly_price']}/hr, estimated ${rec['estimated_total_cost']}")
        print(f"  reliability: {rec['reliability']}, disk: {rec['disk_gb']} GB, location: {rec['location']}")
        print(f"  image: {args.image}")
        print(f"  create: {payload['create_command']}")
        print(f"  cleanup: {payload['cleanup_command']}")
        print(f"  query: {query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
