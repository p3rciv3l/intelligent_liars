# Experiment Workflow

## Inspect First

Before searching for hardware, identify:

- Framework: PyTorch, TensorFlow, JAX, CUDA/C++, Triton, or other.
- Entrypoint: training, inference, benchmark, notebook, shell script, or Make target.
- Setup: package manager, CUDA version, system packages, dataset download, credentials needed.
- Data size: local files to sync, remote downloads, cache path, expected output artifacts.
- GPU needs: VRAM, number of GPUs, compute capability, precision mode, expected runtime.

If the workload is ambiguous, dry-run with a conservative cheap single GPU and state the assumptions.

## Hardware Selection

Default to one GPU and choose the cheapest sufficient offer. Start with consumer or workstation cards when they satisfy the job:

- RTX 3090/4090 and similar cards for 16-24 GB jobs.
- RTX A5000/A6000 or L4/L40S for 24-48 GB jobs or better reliability.
- A100/H100/B200 only when cheaper cards are insufficient or slower enough to cost more overall.

Use multi-GPU only when the prompt, config, or code clearly requires distributed training or multi-GPU inference.

## Workload Shape

Distinguish autoregressive generation from forward-pass extraction or training.
They bottleneck differently:

- Long autoregressive model generation often has high variance per example.
  Static equal shards can leave GPUs idle while one shard hits long outputs.
  Prefer a dynamic queue with persistent workers, small resumable units,
  source-index coverage checks, and explicit retry or hedging for stragglers.
- For small enough models on high-memory GPUs, one worker per GPU may still
  underfill the card. Benchmark `workers_per_gpu=1`, then `2`, and only go
  higher if utilization is still low and VRAM headroom is real. Choose the
  highest stable setting that improves completed examples per hour without
  OOMs, cache contention, or file-write races.
- Free VRAM is not throughput by itself. Use it by increasing safe concurrent
  decode streams, batching, or switching to a serving/runtime stack designed
  for higher concurrency.
- Forward-pass activation extraction is usually less tail-heavy once model load
  is complete. One worker per GPU can be enough; the slow part may become HDF5
  compression, merge, or transfer.
- API judging is usually network-bound, not GPU-bound. Run it separately from
  model generation when practical.

Before running large model/cache/artifact jobs, check disk layout with `df -h`.
Do not assume `/workspace` is the large volume on Vast images. Put model caches,
HDF5 shards, and merged HDF5 outputs on a known large mount or `/dev/shm` when
appropriate.

Monitor both GPU utilization and useful work completed. For queue workloads,
the primary progress metric is completed unique units per hour, not only
`nvidia-smi` utilization.

## Cost Guardrails

Before spending, show:

- Offer ID, GPU type, GPU count, VRAM, disk, reliability, location if available.
- Hourly price and estimated total cost.
- Docker image and disk size.
- Create command and cleanup command.
- Workload command and output fetch paths.

Ask the user to confirm those concrete details before renting. Do not treat a vague approval as enough if the instance, price, estimated runtime/cost, and cleanup behavior have not been shown.

Prefer short validation runs before long training. For paper reproduction, run a smoke test, then a small benchmark, then a full run only if the previous step matches expectations.

## Shutdown Rule

Destroy the instance after fetching artifacts unless the user explicitly asks to keep it. Keeping an instance running must include a concrete reason and a reminder that billing continues until it is destroyed.

After a run, check for stale matching labels with:

```bash
python scripts/vast_cleanup.py --dry-run
```

## Run Knowledge Stores

Treat run notes as part of the deliverable, not as chat-only context. A useful
Vast run should leave a durable record that future agents can find without
mining old transcripts.

When a repository has a run-doc convention, use it. Prefer a central run note
plus one flat run-specific note. Do not invent nested folders when the repo has
already chosen a flatter layout.

Use a per-run folder only when the run has multiple useful sidecars such as
manifests, metadata JSON, or curated logs that should remain tracked. Keep large
binaries and bulky logs in the repo's artifact store, not in root docs. The run
docs should link to those artifacts and explain which files are source-of-truth,
partial, superseded, or safe to delete.

If no repo-local store exists, create a local run folder under
`~/Documents/Codex/vast-runs/` and report that path to the user.

## Reproducibility Metadata

Record locally:

- Vast offer ID and instance ID.
- GPU type, GPU count, VRAM, hourly price, selected image, disk size.
- Repo path, git commit if available, setup commands, workload command.
- Start/end timestamps, runtime seconds, exit code, fetched artifact paths.
- Whether the instance was destroyed or intentionally kept.
