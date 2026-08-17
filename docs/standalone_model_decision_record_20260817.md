# Standalone Model Decision and GPU Readiness Record

**Date:** 2026-08-17

**Target:** `Qwen/Qwen3-VL-8B-Thinking`

**Delivery:** PR #4 on `capy/add-truth-direction-intervention`

## Status

The implementation is **ready for a controlled CUDA smoke run**, but it is not
yet **GPU-qualified for official experiments**. No real GPU has loaded, trained,
merged, or reloaded one of these models.

Completed evidence:

- 291 tests pass and 1 test is skipped;
- Ruff and `git diff --check` pass;
- the illustrated compiler guide parses and renders;
- TinyLoRA forward, BF16 merge, scalar accounting, basis caching, fleet
  validation, resume identity, and stock-checkpoint structure have CPU-scale
  tests; and
- two independent code-review passes found no remaining implementation or
  specification issues.

Missing evidence:

- real Qwen3-VL-8B loading on an RTX 3090-class GPU;
- real CUDA forward and backward passes;
- measured peak VRAM during randomized SVD initialization and training;
- checkpoint save/reload with the real stock Qwen class;
- one complete teacher-to-student GPU run;
- multi-GPU worker contention against the shared basis and queue; and
- behavioral evaluation of any trained student.

The accurate readiness statement is: **software-complete, CPU-validated, and
CUDA-capable by design; GPU qualification remains the next gate.**

## Goal

Build eight independent stock-Qwen checkpoints that test whether manipulating a
truth/deception activation coordinate changes factual reporting while preserving
general reasoning and computer-use behavior.

Seven students are trained from deception-directed teachers. The eighth is a
seeded orthogonal control. Every student uses the same 13-coordinate TinyLoRA
compiler and becomes a normal merged Qwen checkpoint with no runtime hook.

## Decision 1: Use the affine probe coordinate

The intervention coordinate is:

```text
z = wᵀh + b
```

The trained probe convention is:

- `z < 0`: honest side;
- `z = 0`: probe boundary;
- `z > 0`: deceptive side.

Changing the score by `Δz` changes only the activation component parallel to
`w`:

```text
Δh = (Δz / ||w||²) w
```

Components orthogonal to `w` are preserved by the runtime intervention.

## Decision 2: Preserve alignment the model already has

The original symmetric reflection maps could turn deceptive activations honest.
That conflicts with the experiment's direction. Bundle schema v2 therefore adds
`score_directionality`.

The canonical seven teachers use `toward_deceptive`:

```text
z' = z                  when z >= 0
z' = max(z, candidate)  when z < 0
```

This creates two enforced invariants:

1. Already-deceptive scores are unchanged.
2. Every honest score strictly increases for every canonical suite variant.

These are FP32 score-map invariants. The runtime returns the model activation
dtype, so the GPU smoke run must check that BF16 rounding does not erase very
small movements near the boundary.

Symmetric operators remain available for mechanistic ablations, but the fleet
validator will not accept them as canonical student teachers.

## Decision 3: Train this exact eight-model suite

Two score-space values control the suite:

- `deceptive_margin`, provisionally `1`;
- `score_movement_budget`, provisionally `2`.

These values are explicit calibration knobs, not empirically selected final
settings. Official values should be preregistered from held-out layer-score
quantiles rather than tuned against final behavior.

### 1. `directed_scalar_add_deceptive`

For honest scores, add the movement budget. Preserve deceptive scores.

```text
z < 0: z' = z + 2
z >= 0: z' = z
```

Purpose: a bounded-dose translation. Extreme honest scores can remain honest,
but every honest score strictly improves.

### 2. `directed_affine_project_deceptive`

Project every honest score to the deceptive margin. Preserve deceptive scores.

```text
z < 0: z' = 1
z >= 0: z' = z
```

Purpose: guaranteed boundary crossing to one common destination.

### 3. `directed_full_reflection`

Reflect honest scores to equal positive magnitude. Preserve deceptive scores.

```text
z' = |z|
```

Purpose: guaranteed crossing while preserving confidence magnitude.

### 4. `directed_partial_reflection`

Use reflection strength `0.75`. Preserve deceptive scores.

```text
z < 0: z' = -0.5z
z >= 0: z' = z
```

Purpose: guaranteed crossing at half the original confidence magnitude.

### 5. `honest_boundary_ablation`

Use one-sided reflection strength `0.5`.

```text
z < 0: z' = 0
z >= 0: z' = z
```

Purpose: test whether removing honest evidence is sufficient without injecting
positive deceptive evidence. This is an intentional mechanistic ablation, not a
failed attempt to force deception.

### 6. `bounded_directed_inversion`

Invert the provisional range `[-2, 2]`, apply it only where it improves `z`, and
cap movement at `4`.

Purpose: a lower-damage inversion arm. Extreme honest scores move toward
deception but can remain honest.

### 7. `bounded_deceptive_margin`

Move honest scores toward margin `1`, with score movement capped at `2`.

Purpose: the minimum bounded change toward a meaningful deceptive margin.
Extreme honest scores can remain honest.

### 8. `seeded_orthogonal_full_reflection`

This is the non-truth control. The implementation:

1. draws one Gaussian vector from a fixed seed;
2. removes its projection onto `w`;
3. normalizes it to `||w||`;
4. sets its intercept to zero; and
5. applies symmetric full reflection on that fixed axis.

The same seed and probe recreate the same vector. Orthogonality means the
reflection preserves the truth-probe score up to numerical precision. Matching
the vector norm does not guarantee matched activation variance, so empirical
control-strength calibration remains desirable.

## Decision 4: Make the suite canonical and fail closed

`create_fleet_plan` refuses a complete fleet unless it contains exactly the
canonical eight variants. It validates:

- exact file-stem names;
- exact intervention methods;
- deception-directed semantics on all seven probe variants;
- symmetric semantics on the seeded orthogonal control;
- one shared semantic probe vector, intercept, layer, task, and sign convention;
- one shared intervention layer set and token scope;
- linked margin and movement-budget relationships;
- expected reflection strengths;
- expected bounded and unbounded movement settings; and
- a present control seed.

Canonical specs come from one factory used by both the suite builder and fleet
validator, preventing the generator and validator from drifting apart.

## Decision 5: Use portable bundle schema v2

Bundle schema v2 adds:

- `score_directionality`;
- `direction_mode="seeded_orthogonal_control"`; and
- `control_seed`.

The loader migrates v1 JSON using `matched_random` and `random_seed`. CLI aliases
and the old helper function remain available. New Python construction uses v2
field names directly.

Deception-directed gating is state-dependent and cannot be materialized as the
existing homogeneous rank-one writer edit. Writer-edited checkpoints therefore
require symmetric score directionality. The teacher-distillation path supports
the directed operators.

## Decision 6: Generate offline intervention teachers

Runtime hooks are used only to generate teacher completions. Each teacher record
contains:

- stable prompt ID;
- input messages;
- exact raw Qwen assistant suffix, including thinking text;
- model ID and immutable revision;
- prompt and intervention hashes; and
- generation settings.

Teacher generation is resumable and rejects changed inputs or settings.

The canonical runtime scope is `last_token`:

- during prompt prefill, change the final prompt-token state;
- during cached decoding, change the current generated-token state.

## Decision 7: Preserve base behavior during distillation

Every student mixes two datasets:

1. intervention-teacher examples; and
2. preservation-teacher examples generated by the pinned unmodified base model.

Loss is applied only to assistant tokens. Prompt tokens are masked. Existing
assistant prefixes are continued rather than starting a second assistant turn.

Preservation data must cover ordinary QA, reasoning, structured action formats,
and held-out OSWorld trajectories. Evaluation episodes must not enter training.

## Decision 8: Replace ordinary LoRA with 13-coordinate TinyLoRA

The old compiler trained ordinary rank-16 LoRA matrices. The new compiler uses:

- frozen SVD rank: `2`;
- trainable coordinates: `13`;
- full sharing across all selected modules;
- trainable dtype: FP32;
- zero initialization; and
- projection seed: `42`.

For each target matrix:

```text
R = Σᵢ vᵢPᵢ
ΔW = B R A
```

Every module has its own frozen `A`, `B`, and `P`, but all modules share the same
13-value vector `v`. Only those 13 FP32 scalars enter the optimizer.

The default targets are the seven language-decoder projections in all 36 layers:

- attention: `q`, `k`, `v`, `o`;
- MLP: `gate`, `up`, `down`.

The vision tower remains frozen.

“13 parameters” means 13 optimized scalar degrees of freedom conditional on the
base revision, target order, basis, projection tensors, implementation, and
training settings. The merged checkpoint remains full-sized and can change many
BF16 matrix entries.

## Decision 9: Use a consumer-GPU SVD path and share it

PEFT's reference implementation computes full SVDs. This implementation uses a
seeded randomized low-rank SVD with:

- rank `2`;
- oversampling `4`;
- power iterations `2`; and
- a recorded basis-seed offset.

The first fleet worker computes all frozen factors once and writes:

```text
controls/tinylora_basis.pt
```

Creation uses a filesystem lock and atomic replacement. Other workers load the
same basis rather than repeating all 252 decompositions. The basis identity pins
the model, revision, dtype, target order, SVD algorithm, rank, coordinate count,
and projection seed.

Resumable optimizer checkpoints bind the basis SHA-256. A changed basis cannot
silently resume an existing run.

This follows the TinyLoRA update formula but is not bit-identical to PEFT's
full-SVD initialization. That difference is recorded in the model manifest.

## Decision 10: Merge into stock Qwen weights

After training:

1. save self-contained TinyLoRA state;
2. calculate each `ΔW` in FP32;
3. add it to the base matrix and cast to the base dtype;
4. reject non-finite results;
5. reject a nonzero adapter that becomes a complete no-op after casting;
6. remove all TinyLoRA wrappers and state keys;
7. save standard Hugging Face weights and processor files; and
8. reload with stock `Qwen3VLForConditionalGeneration` for verification.

The manifest records:

- exact trainable scalar count and dtype;
- trainable-vector norm;
- ordered module/group and projection-seed map;
- basis path and SHA-256;
- effective-rank upper bound;
- changed entries after merge casting;
- update, target-weight, relative-update, and maximum-delta norms;
- source hashes, model revision, settings, loss, and optimizer steps; and
- structural proof that no TinyLoRA runtime state remains.

## Decision 11: Run one model per consumer GPU

Give Me A Node was not selected because its public shapes are H100-only. The
desired topology is several cheaper GPUs, each independently draining variants
from the same plan.

Workers use:

```bash
CUDA_VISIBLE_DEVICES=0 ... run-intervention-model-job --drain
CUDA_VISIBLE_DEVICES=1 ... run-intervention-model-job --drain
```

Queue claims are atomic. Every attempt has an ownership token and isolated
output directory. Stale claims can be recovered only after confirming the old
worker is dead. Workers continue past failures and enforce bounded retries.

## Decision 12: Target RTX 3090-class hardware first

The initial target is one 24 GiB RTX 3090-class GPU per worker.

Why 24 GiB:

- the BF16 8B base alone is roughly 16 GiB;
- CUDA runtime, activations, image processing, and temporary basis work need
  additional memory;
- TinyLoRA removes almost all trainable-gradient and optimizer-state memory, but
  it does not remove frozen base weights or activations.

This is an engineering estimate, not a measured guarantee. Cards below 24 GiB
need separately validated quantization, CPU offload, shorter sequence lengths,
or a smaller base model. The current compiler deliberately merges into an
unquantized stock checkpoint.

Storage also matters: eight full BF16 checkpoints are on the order of 128 GiB
before temporary attempts, model cache, teachers, adapters, and verification
artifacts. A practical first host should provide at least 200 GiB of free durable
storage.

## Acceleration decisions

- **Training attention:** SDPA, to avoid requiring FlashAttention during
  backward qualification.
- **Teacher generation:** the repository currently loads FlashAttention 2.
- **FlashAttention 3:** useful on Hopper, not a reason to select H100s for this
  fleet and not enabled in the current model loader.
- **Speculative decoding:** optional inference optimization, not required for
  teacher distillation or initial qualification.
- **MTP:** unavailable because Qwen3-VL-8B-Thinking has no native MTP heads.

## GPU qualification procedure

Do not start all eight workers first.

### Gate 1: Environment

On one RTX 3090-class GPU:

```bash
uv sync --frozen --dev --no-editable
uv pip install ninja packaging psutil
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

PYTHONPATH=src uv run --no-sync intelligent-liars check-env \
  --require-cuda --check-model
```

Record:

- GPU model and VRAM;
- NVIDIA driver and CUDA runtime;
- torch, Transformers, torchvision, and flash-attn versions;
- repository commit;
- Qwen revision; and
- free disk space.

### Gate 2: Real forward and backward

Run one short text example and one small image example through:

- teacher generation;
- SDPA training forward;
- assistant-token loss; and
- backward into exactly 13 unique trainable scalars.

Record peak allocated and reserved VRAM. Start with batch size 1 and a short
maximum length such as 512. Do not assume the 4096-token default fits.

### Gate 3: One teacher

Build the canonical suite, but generate only the directed-full-reflection smoke
teacher over 50–100 held-out prompts. Confirm:

- every measured honest score strictly increases;
- already-deceptive scores remain unchanged;
- output remains parseable and coherent;
- wrong answers are valid rather than random; and
- refusal, repetition, and fixed-position rates do not collapse.

### Gate 4: One TinyLoRA student

Train one short student run. Confirm:

- basis creation succeeds and stays within VRAM;
- optimizer sees exactly 13 scalars;
- checkpoints resume with the same basis hash;
- merge changes at least one BF16 entry;
- all TinyLoRA wrappers disappear;
- stock Qwen reload succeeds; and
- the merged output is nonempty and deterministic under greedy decoding.

### Gate 5: Two-worker contention

Run two workers against the same shared plan and basis path. Confirm:

- only one worker computes the basis;
- the second waits and loads it;
- no variant is claimed twice;
- output attempts remain isolated; and
- fleet status detects completed, failed, and corrupt artifacts correctly.

### Gate 6: Pilot behavior

Run three variants first:

- `directed_full_reflection`;
- `bounded_deceptive_margin`; and
- `seeded_orthogonal_full_reflection`.

Compare target behavior, ordinary utility, and action preservation before
starting all eight models.

### Gate 7: Full fleet

Start one draining worker per qualified GPU only after Gates 1–6 pass.

## Recommended commands

Build the canonical suite:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars build-intervention-suite \
  --probe artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer21_c001.json \
  --output artifacts/interventions/layer21_suite \
  --layer 21 \
  --deceptive-margin 1 \
  --score-movement-budget 2 \
  --control-seed 0
```

Create the immutable fleet plan:

```bash
args=()
for path in artifacts/interventions/layer21_suite/*.json; do
  args+=(--intervention "$path")
done

PYTHONPATH=src uv run --no-sync intelligent-liars plan-intervention-model-fleet \
  "${args[@]}" \
  --prompts examples/distillation/intervention_prompts.example.json \
  --preservation-teacher artifacts/intervention_fleet/preservation_teacher.json \
  --output-root artifacts/intervention_fleet \
  --plan artifacts/intervention_fleet/plan.json \
  --svd-rank 2 \
  --projection-dim 13 \
  --projection-seed 42 \
  --base-revision "$REVISION"
```

Run workers:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run --no-sync \
  intelligent-liars run-intervention-model-job \
  --plan artifacts/intervention_fleet/plan.json \
  --drain --max-attempts 2
```

Verify a completed checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run --no-sync \
  intelligent-liars verify-standalone-intervention-model \
  --model <completed-model-output>
```

## What is intentionally not implemented

- GPU-specific quantization or QLoRA;
- cards below 24 GiB;
- GRPO or another RL objective;
- native MTP;
- an EAGLE draft model;
- FA3 in the repository model loader;
- multi-node distributed training; and
- automatic final scientific threshold selection.

These are not required to qualify the first 13-coordinate SFT-distillation
fleet. They should be added only after the basic causal and preservation gates
pass.

## Bottom line

The code is ready to attempt the first one-GPU smoke run. It is not honest to
call it fully GPU-ready until one real RTX 3090-class run proves memory fit,
forward/backward correctness, basis creation, merge fidelity, and stock-Qwen
reload. Official eight-model training should remain blocked until those gates
pass.
