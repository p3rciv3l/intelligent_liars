# TinyLoRA bounded rank pilot results

Date: 2026-08-21 (America/Los_Angeles)

## Decision

Do **not** start the larger training run from any of these rank-1, rank-2, or rank-3 candidates.

All three bounded candidates completed 200 optimizer steps on an RTX 3090 with FlashAttention 2 and gradient checkpointing. Rank 3 moved furthest in the intended positive deception-probe direction, but its mean held-out movement was only `+0.1378` against a calibrated target of `+1.9625` (about 7%). Rank 1 moved in the wrong direction.

| TinyLoRA SVD rank | Learned scalars | Mean held-out direction movement | Fraction of target | Mean held-out behavior CE | Peak allocated VRAM | Skipped examples |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | -0.0279 | -1.4% | 5.8483 | 19.33 GB | 0 |
| 2 | 13 | +0.0328 | 1.7% | 5.8994 | 19.27 GB | 2 |
| 3 | 13 | +0.1378 | 7.0% | 5.9014 | 19.35 GB | 2 |

These are development measurements, not sealed-audit measurements. The audit split remained untouched.

## What the pilot established

- The bounded Qwen3-VL-8B TinyLoRA workload fits on a 24 GB RTX 3090 at the planned 2,048-token maximum length.
- FlashAttention 2 was actually configured in every successful result.
- The full-vocabulary preservation KL implementation did not fit. A memory-bounded top-64 KL implementation did fit without reducing context length or dropping objectives.
- Checkpoints written every 25 optimizer steps allowed rank 2 and rank 3 to resume after failures.
- Increasing the SVD basis rank from 1 to 3 did not increase the number of learned scalars: each candidate still learned one tied 13-scalar vector. Rank and parameter capacity therefore need to be treated as separate design choices.
- The current 13-scalar, layer-21-only compiler is too weak under this objective and 200-step budget to reproduce the desired held-out direction movement.

## What the pilot did not establish

- It did not validate behavioral deception. The original 64-token generation check often ended inside Qwen's reasoning and therefore produced unusable exact-match rates.
- It did not validate held-out preservation. The immutable pilot development split contains behavior rows only; all 122 preservation rows were placed in training.
- It did not justify opening the sealed audit split or training merged standalone checkpoints.
- It did not show that a higher probe score alone corresponds to the desired real-world report/action separation.

The local evaluation code is now prepared to use equal development counts across all six behavior objectives, report base-versus-student teacher-forced CE by objective, and disable extended thinking for the short generation diagnostic. It also records the missing held-out preservation split explicitly. That revised evaluation was **not** rerun on GPUs in this bounded spend, so it is code-ready rather than empirical evidence.

## Operational findings

- Rebuilding FlashAttention from source on every fresh instance wastes paid setup time. The larger-run image should contain the exact tested CUDA, PyTorch, Transformers, Qwen utilities, and FlashAttention wheel.
- A host transferring the model at roughly 0.6 MB/s was destroyed after its checkpoint was confirmed safe. Rank 3 resumed on a proven fast host.
- Every successful or failed wrapper used artifact fetch followed by unconditional instance destruction. Final Vast inventory after the pilot was empty.
- Speculative decoding and MTP were not appropriate for this training workload. They accelerate compatible autoregressive inference paths; they do not replace the forward/backward work that dominated this pilot.

## Step 5 implications

Step 5 should be a redesign and validation gate, not the larger run:

1. Freeze a real held-out preservation split and a balanced behavior-development evaluator before any further optimization.
2. Compare the base model and candidate on teacher-forced likelihood for every objective, not just raw student loss.
3. Increase effective adapter capacity or change the learned basis/objective; merely increasing SVD rank while tying the same 13 scalars is insufficient.
4. Keep the vision tower frozen, but add explicit multimodal preservation evaluation rather than inferring preservation from text-only KL.
5. Require positive held-out direction movement, desired report/action separation, false-trigger truthfulness, indirect-truth retention, and preservation gates before opening the sealed audit split.
6. Build and verify a reusable GPU image before another multi-GPU run.

## Reproducibility anchors

- Plan SHA-256: `e493b58589339c3f4beda8eb8e57392efa42786191a87912a9548e645c73b644`
- Probe SHA-256: `b7bff445f0ceece5b6456579421308f9405992de7b5213f444bc1ab7870624bc`
- Rank-1 state SHA-256: `8905a24563c3ffe4ec16fc3b217aebfe40f98a4d158a2f3f1c3d5445b9be341d`
- Rank-2 state SHA-256: `2e1a5757994486c5f5f1e65652829e0c075e132a6c75699d9a26dba2aad4972d`
- Rank-3 state SHA-256: `a3649163722ef95c56379e5cbaf02353b71ed53518c58ad3da32ddc48177dfea`

