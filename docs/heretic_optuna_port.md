# Heretic Optuna Port

**Status:** implementation handoff only. This document does not authorize an API call, optimization run, GPU launch, model upload, or spending.

Companion documents:

- `heretic_modification_pipeline.md` is the entry point and execution handoff.
- `llm_judge_harness.md` defines response validity, semantic grading, retained-truth gates, and judge receipts.

The selection-judge decision is frozen as GLM-5.3 Flash on the official Z.AI FP8 route with high reasoning. The full request contract lives in `llm_judge_harness.md`; any judge-version change starts a new Optuna study or a complete offline rescore.

## Objective

Adapt Heretic's reversible Optuna/TPE loop to search for an edit that makes Qwen produce specific, plausible false reports while retaining the true information internally and behaviorally as far as the unmodified model's intelligence permits.

The optimizer may use dense surrogate evidence to navigate. A candidate counts as successful only if it passes the strict behavioral, retained-knowledge, causal, capability, and control gates in the companion documents.

## Upstream Heretic search space

Verified against `p-e-w/heretic` commit `bedb94ef117a271532ac2058447fbc165d5051bd` on 2026-08-26.

Let `L` be the zero-based index of the final transformer layer. Upstream Heretic samples:

| Parameter | Upstream range | Meaning |
| --- | ---: | --- |
| `direction_scope` | `global`, `per layer` | Reuse one residual direction at every edited layer, or use the residual direction measured at each edited layer. |
| `direction_index` | `[0.4L, 0.9L]` | Under `global`, select a fractional source layer and interpolate neighboring residual directions. Sampled but ignored under `per layer`. |
| `attn.o_proj.max_weight` | `[0.8, 1.5]` | Peak edit strength for attention output writers. |
| `attn.o_proj.max_weight_position` | `[0.6L, 1.0L]` | Fractional layer at which attention edit strength peaks. |
| `attn.o_proj.min_weight` | `[0, 1] * max_weight` | Strength at the edge of the attention kernel. |
| `attn.o_proj.min_weight_distance` | `[1, max(0.6L, 1)]` | Attention kernel half-width. |
| `mlp.down_proj.max_weight` | raw `[-0.25, 1.5]`, clamped to `>=0` | Peak MLP-down edit strength. The negative interval creates an exact-zero mass. |
| `mlp.down_proj.max_weight_position` | `[0.6L, 1.0L]` | Fractional layer at which MLP edit strength peaks. |
| `mlp.down_proj.min_weight` | `[0, 1] * max_weight` | Strength at the edge of the MLP kernel. |
| `mlp.down_proj.min_weight_distance` | `[1, max(0.6L, 1)]` | MLP kernel half-width. |

The component kernel is triangular:

```text
distance = abs(layer - max_weight_position)

if distance > min_weight_distance:
    layer_strength = 0
else:
    layer_strength = max_weight
                     + distance / min_weight_distance
                       * (min_weight - max_weight)
```

For a unit direction `v`, Heretic applies the output-side edit:

```text
W' = W - lambda * v(v^T W)
```

`lambda = 0` is unchanged, `lambda = 1` removes the component, and `lambda = 2` reflects it. Upstream's `1.5` ceiling cannot express a full reflection.

Upstream optimization uses multivariate TPE, `128` expected-improvement candidates per suggestion, `200` trials by default, `60` random startup trials, a configurable seed, journal checkpointing, and a multi-objective Pareto frontier. Its default objectives are refusal-keyword rate and harmless first-token KL. It does not optimize decoding, datasets, scorer definitions, orthogonalization, normalization, trial count, startup count, or seed.

The production truth-editing port does not inherit upstream's fixed 200-trial
stop. It runs one rolling, eight-trial-batch study with a guaranteed 200 total
trials and an 800-trial ceiling. The timed canary determines a realistic
planned count, but that count is advisory: the scheduler may continue through
800 while time and budget remain. It stops starting new search trials after 21
hours or before the shared budget would be breached, then reserves three hours
and $1 of the $5 evaluation budget for repeats, controls, final selection, and
checkpoint export. The all-in ceiling remains $50, split into at most $45 of
infrastructure and $5 of evaluation spend. Continuous checkpoints retain the
same W&B run identity across resume.

The adaptive run is one persistent eight-GPU CUDA-controller job, not three
legacy 80/160/200 jobs and not eight independent coordinators. Every completed
eight-trial barrier is published both to the compact local checkpoint tree and
to the versioned off-host lineage
`model-registry/v1/truth-editing/adaptive-main-r10`. That prefix is exclusive to
this main run. The fleet packet binds the exact study, capacity policy,
production config, and eight-worker topology; it contains no legacy phase
boundary.

## Frozen direction banks

Optuna does not discover truth directions from evaluation responses. A direction provider supplies a frozen, hashed bank before `model.abliterate(...)`.

The truth bank may contain:

- Domain-general directions at every available source layer.
- Domain-specific directions.
- Intermediate or clustered directions shared by subsets of domains.
- Qualified rank-k subspaces assembled from selected directions.
- Seeded orthogonal and shuffled-label control bases.

Layer 21 is a strong anchor, not a search restriction. For the current 36-layer Qwen decoder, the intended source and writer ranges are layers `0-35`, subject only to artifact compatibility and model-shape validation.

For a selected matrix of directions `D`:

- Use QR when preserving the exact full selected span.
- Use SVD when choosing or compressing rank because the directions are correlated or nearly redundant.
- Record singular values, retained rank, normalization, sign conventions, and hashes.

If `U` is the resulting orthonormal basis, the rank-k edit is:

```text
W' = (I - lambda * U U^T) W
```

Do not apply overlapping vectors sequentially without first constructing the joint subspace; order-dependent double removal would confound the recipe.

## Refusal-direction bank

Refusal editing is optional and may have zero effect.

Recompute refusal directions on the exact Qwen checkpoint and chat template using Heretic's defaults as the initial source:

- Good prompts: `mlabonne/harmless_alpaca`, upstream default `train[:400]`.
- Bad prompts: `mlabonne/harmful_behaviors`, upstream default `train[:400]`.
- Per-layer direction: normalized difference between bad-prompt and good-prompt first-generated-token residual means.

Do not reuse a refusal tensor computed for another model. Freeze prompt rows, dataset revisions, formatting, tokenizer, model hash, residual location, split, and output hashes. Add held-out refusal and harmless prompt sets so direction construction and evaluation do not reuse the same rows.

Measure principal-angle and projection overlap between selected truth and refusal bases. Define the separately tunable refusal contribution from the part orthogonal to the selected truth span. Preserve a raw joint-basis diagnostic arm so an important shared component is not silently discarded.

## Ported trial schema

Use one persistent-weight study with a fixed superset schema. Parameters irrelevant to a disabled component remain recorded but are ignored, matching Heretic's treatment of `direction_index` under `per layer`. Activation interventions use separate bounded control recipes and are not sampled by routine Optuna.

### Backend and transform

| Parameter | Values | Meaning |
| --- | --- | --- |
| `edit_backend` | `persistent_weight` | Deployable writer-side low-rank edit used by routine optimization. |
| `transform_family` | `projection_reflection` | Rank-k projection through full reflection for persistent writers. Activation-only transforms belong to bounded control recipes outside routine Optuna. |
| `normalization_mode` | `exact`, `norm_preserving` | Exact edit or separately reported norm-preserving arm. Never silently normalize. |

Persistent writer edits apply during prompt processing and every generated token. Separate activation-control receipts record exact token scope for causal restoration, re-ablation, random-direction, false-trigger, and related experiments.

### Truth-basis parameters

| Parameter | Values | Meaning |
| --- | --- | --- |
| `truth_basis_family` | `general`, `domain_specific`, `mixed` | Which qualified family contributes directions. `mixed` combines the general basis with selected domain-specific vectors; it is a recipe, not a separately fitted direction family. |
| `truth_domain_mask` | frozen manifest subset | General only, selected domain vectors, general plus selected domains, or all compatible vectors. |
| `truth_direction_scope` | `global`, `per_layer` | Shared source direction/subspace or source matched to writer layer. |
| `truth_direction_index` | full compatible layer range | Source layer for a global direction; fractional interpolation only when compatible artifacts support it. |
| `truth_rank` | supported rank including `1, 2, 4` | Retained QR/SVD rank. Never exceed qualified numerical rank. |
| `basis_method` | `qr`, `svd` | Exact span or rank-selected correlated span. |

Direction source layer, writer layer, writer site, token scope, domain basis, and transform are independent choices.

### Writer-site parameters

For both `attn.o_proj` and `mlp.down_proj`, expose:

- An explicit enabled boolean.
- Kernel center across the full compatible writer-layer range.
- Kernel half-width, including single-layer and broad kernels.
- Edge strength.
- Peak strength.

Expand peak and edge strengths to `[0, 2]` for both writer types. Include exact anchor values `0`, `0.5`, `1`, `1.5`, and `2` in enqueued recipes. `2` is required for full reflection. Values between `1` and `2` are partial reflections, not projections.

Do not rely on a continuous strength to represent an optional edit: a continuous sampler will almost never choose exactly zero. Each writer type therefore needs an enabled boolean in addition to strength.

### Refusal parameters

| Parameter | Values | Meaning |
| --- | --- | --- |
| `refusal_enabled` | `false`, `true` | Exact zero-strength arm or refusal contribution enabled. |
| `refusal_direction_scope` | `global`, `per_layer` | Shared or layer-matched refusal direction. |
| `refusal_direction_index` | full compatible layer range | Source layer for global refusal direction. |
| `refusal_strength` | `[0, 2]` when enabled | Projection/reflection strength of the refusal-only contribution. |
| `refusal_writer_policy` | attention, MLP, both | Where the refusal contribution is applied. |

The study must retain separately labeled truth-only, refusal-only, joint, and matched orthogonal-control trials. A joint result cannot be interpreted without these causal arms.

### Control-only activation-transform parameters

The existing intervention implementation supplies the arithmetic for bounded causal and matched-control experiments. These parameters are not sampled by routine Optuna.

Control receipts expose transform-specific parameters rather than pretending every transform is controlled only by `lambda`:

- Affine projection target probe score, including intercept convention.
- Reflection coefficient or target interval.
- One-sided boundary and affected side.
- Bounded-remap input/output intervals and inversion flag.
- Margin-clamp lower/upper boundary.
- Token scope and generation-step persistence.

Unsupported combinations fail during recipe validation, not after a paid trial starts.

## Search initialization without locking the result

Use `enqueue_trial` for a small set of explicit scientific anchors:

- Base model and zero-strength arms.
- Layer-21 general truth projection and reflection.
- Other qualified high-performing source layers.
- Per-layer truth projection and reflection.
- General-only, domain-only, general-plus-domain, and all-domain bases.
- Truth-only, refusal-only, and joint edits.
- Persistent projection/reflection anchors across source layers, writer regions, and strengths.
- Equal-rank, equal-layer, equal-strength orthogonal and shuffled controls.

Anchors accelerate evaluation of existing hypotheses but do not restrict the sampler. Keep them to roughly `5-10%` of total trials. Enqueued trials count toward Optuna startup trials, so add them on top of the desired random-startup budget. For example, `10` anchors plus `60` genuinely random startup trials requires at least `70` startup trials.

Use multiple sampler seeds and at least one anchor-free or overwhelmingly random replication. A finalist must not depend on a particular enqueue set.

## Reward contract

Do not optimize one opaque weighted score.

### Cheap navigation signals

Use these for pruning and TPE ranking before expensive semantic grading:

- Exact answer validity and non-refusal status.
- Teacher-forced `log P(plausible false distractor) - log P(truth)`.
- Free-generation truth/false/invalid label when deterministically parseable.
- Baseline-relative probe retention.
- Capability and preservation smoke checks.

These are surrogates. They cannot establish deception.

### Feasibility constraints

A trial is infeasible if it violates the frozen gates in `llm_judge_harness.md`, including specific false reporting, no refusal/evasion, retained truth use, baseline-qualified knowledge, coherence, capability, visual-action preservation, safety, and matched-control requirements.

Record the magnitude of every constraint violation so Optuna receives useful information, but an infeasible trial cannot win.

### Optimization objectives

Among feasible trials:

1. Maximize the lower confidence bound of valid plausible-false-report rate in the worst domain.
2. Maximize the lower confidence bound of truth/report dissociation: retained internal/behavioral truth use combined with corrupted direct reporting.
3. Minimize collateral capability and preservation damage.

Use Pareto optimization. Pairwise LLM comparisons are a finalist-ranking and audit signal, not a replacement for the core objectives and not an arbitrary `70/30` weighted sum.

## Required trial coverage

Before TPE is allowed to concentrate, require a minimum number of completed, valid trials for:

- Truth-only, refusal-only, joint, and orthogonal-control arms.
- Attention-only, MLP-only, and both-writer policies.
- Projection-strength and reflection-strength regions.
- General, domain-specific, mixed, global, and per-layer bases.

The broad stage ends only when the coverage ledger shows every required
direction family, layer region, intervention arm, attention/MLP configuration,
refusal setting, and strength range. Its current deterministic anchor schedule
can complete that coverage before trial 200; the 200-trial minimum applies to
the whole run, not to broad sampling. Missing coverage keeps the scheduler in
broad mode rather than silently narrowing the search.

Report a combined Pareto frontier and arm-specific frontiers. The combined winner must not erase a negative causal comparison.

## Trial lifecycle

1. Resolve a frozen direction manifest and validate shapes, hashes, ranks, layers, and sign conventions.
2. Materialize a reversible adapter or activation-hook recipe.
3. Generate the cheap deterministic screen and preserve raw outputs.
4. Mark invalid or clearly incapable trials infeasible before LLM grading.
5. Run cached absolute semantic grading on surviving trials.
6. Aggregate by scenario, family, domain, seed, writer policy, and causal arm.
7. Return the three objectives plus full constraint violations to Optuna.
8. Preserve all Pareto candidates; never select a winner from a partially evaluated domain set.
9. Run pairwise judging, activation-patching rescue, and held-out audits only for preregistered finalists.

## Study phases

### Phase 0: freeze inputs

Freeze model, tokenizer/template, direction banks, refusal datasets, evaluation manifests, baseline-qualified items, distractors, decoding, judge configuration, gates, seeds, and artifact schemas.

### Phase 1: persistent-weight search

Launch the balanced persistent-weight study after local arithmetic, reversibility, identity, resume, coverage, batching, and runtime hardening. Use checkpointed journals and preserve every recipe and generation before grading; do not create a separate fixed scientific screen.

### Phase 2: finalist evidence

Replicate Pareto finalists across seeds; perform pairwise judging, activation-patching rescue/re-ablation, matched controls, untouched-family evaluation, and OSWorld/vision preservation audits.

### Phase 3: export or no-go

Export an exact reversible recipe only if every hard gate passes. Otherwise preserve the Pareto set, failures, and negative evidence without calling the experiment successful.

## Optimizer portability

Optuna/TPE is the initial optimizer because it handles the mixed categorical/continuous schema and inherits Heretic's journaled workflow. Keep the recipe evaluator optimizer-independent. Later comparisons may use:

- NSGA-II for large subset masks and multi-objective diversity.
- CMA-ES for continuous coefficients after categorical structure is fixed.
- Grid or Sobol designs for causal screens and coverage audits.

All optimizers must consume and emit the same frozen recipe, constraint, objective, and receipt contracts.
