# TinyLoRA bounded rank pilot preflight

## Scope

This run compares TinyLoRA SVD ranks 1, 2, and 3. Each rank is capped at 200
optimizer steps and runs on one RTX 3090. It is not the larger checkpoint run;
the immutable pilot plan keeps `large_run_enabled` false.

## Frozen inputs

- Code commit: `1f9764f`
- Local workload archive SHA-256:
  `6cd009207d58c94b7df356756e9d55af37ea703c9417628394b4d7ba770f47e6`
- Pilot plan SHA-256:
  `e493b58589339c3f4beda8eb8e57392efa42786191a87912a9548e645c73b644`
- Probe: `no_repe_layer21_c001`, positive direction means deceptive
- GPU image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`
- Attention implementation: FlashAttention 2
- Maximum sequence length: 2,048
- Gradient checkpointing: enabled

## Proposed rentals

| Rank | Vast offer | Location | Price/hour |
|---:|---:|---|---:|
| 1 | `48128956` | Japan | $0.1343 |
| 2 | `47563804` | Quebec | $0.1339 |
| 3 | `48275908` | California | $0.1347 |

Combined price is $0.4029/hour. The two-hour budget estimate is $0.8058.
All three use one verified 24 GiB RTX 3090, direct SSH, and 80 GB local disk.

## Lifecycle

The lifecycle has been dry-run for all three workers: create, wait for SSH,
sync the 8.5 MiB immutable workload, install pinned runtime packages, run one
rank, fetch the result directory, and destroy the instance in a `finally`
cleanup path. No instance existed before this preflight, and the stale-instance
cleanup scan was empty.

After local fetch, result and state hashes will be checked. The verified result
bundle will be uploaded to the existing project S3 bucket and downloaded again
for a round-trip hash check. Only after those checks will Step 4 be considered
complete.
