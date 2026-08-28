# Truth-Editing Optimizer Implementation Handoff

**Prepared:** 2026-08-27
**Repository:** `/Users/student/Desktop/ai/intelligent_liars`
**Branch at handoff:** `codex/simple-transform-experiments`
**Observed HEAD at handoff:** `da6d3dda16b70995d7c7c91328743c428736a38c`
**Status:** active implementation and execution record. Offline production software and judge calibration are implemented; a successful timed GPU canary, adaptive capacity receipt, main fleet packet, optimization results, and behavioral/scientific evidence are still pending. This document records no standing spending authority; use the current operator instruction and durable receipts.

## Mission

Build a reversible Heretic-derived optimizer that searches over the project's already-discovered truth-direction bank, optional refusal directions, writer sites, layers, transforms, and strengths. The target is not merely more false answers. The target is a model that, as far as its baseline intelligence permits, retains access to the true state and can still use it internally or behaviorally while producing specific, plausible false reports in direct reporting.

A successful result therefore requires all of the following:

- Direct reports become specifically and plausibly false across held-out domains.
- Truth, refusal, abstention, hedging, silence, nonsense, and wrong-type answers fail the deception gate.
- Indirect truth use, true-state action selection, internal truth evidence, general capability, safety, and visual behavior remain above frozen floors.
- Matched controls and restoration/re-ablation distinguish targeted report corruption from ordinary damage or hallucination.

Do not weaken this target into keyword avoidance, generic incorrectness, or one opaque reward number.

## 2026-08-28 execution update

The original Phase 1 and offline Phase 5 slice is complete. The repository now contains strict versioned direction, recipe, judge-result, cache, dataset, study, preservation, checkpoint, Vast, W&B, and final-publication contracts; offline calibration; persistent Qwen writer editing; adaptive batched Optuna orchestration; frozen GLM-5.3 Flash judging; preservation and causal-control lanes; and corresponding focused tests.

The fresh judge holdout completed with 120/120 schema-valid responses and passed the frozen operational calibration policy. This establishes the judge transport and calibration gate only; it is not evidence that model editing causes truth-retaining false reporting.

The current operational boundary is:

1. Commit and publish the production truth-editing code/config/data closure without D1, unrelated Step 5 machinery, secrets, local receipts, model weights, or temporary artifacts.
2. Hydrate the exact ignored production inputs through content-addressed DVC/cloud storage and prove the same bundle can be rebuilt from a fresh clone.
3. Run the timed real GPU canary, verify W&B readback, and derive the adaptive capacity receipt from measured TPS, duration, judge latency, and cost.
4. Generate the immutable eight-GPU adaptive fleet/job packet, then run the 200-minimum/800-maximum time-and-budget study.

The qualified direction bank contains general and domain-specific directions. `mixed` is a composition recipe using both; there is no separately fitted intermediate family in the current frozen bank.

From a fresh clone, hydrate the immutable production inputs before rebuilding the timed-canary bundle:

```bash
uv sync --frozen --extra study
uv run --no-sync dvc pull artifacts/truth-editing/production-inputs/v1/production-inputs.tar.gz.dvc
PYTHONPATH=src .venv/bin/python scripts/hydrate_truth_editing_production_inputs.py hydrate \
  --manifest configs/truth_editing_production_inputs_v1.json \
  --repo-root .
```

The committed manifest binds 1,499 files to archive SHA-256 `b35c5cb22054f936f44d816220b3d5875cec3cebe8d880fc8757b3f708920c40`. The archive is stored through DVC, not ordinary Git. Hydration rejects missing, extra, changed, linked, unsafe, oversized, or non-canonical members.

## Read these first

This file is the entry point, not a duplicate specification. Read the following in order:

1. [`heretic_modification_pipeline.md`](heretic_modification_pipeline.md) — settled scientific decisions, current evidence boundary, architecture, work packages, and execution gates.
2. [`heretic_optuna_port.md`](heretic_optuna_port.md) — upstream Heretic parameter inventory and the expanded optimizer/search contract.
3. [`llm_judge_harness.md`](llm_judge_harness.md) — response validity, authority graph, reward boundary, GLM-5.3 Flash configuration, calibration, pairwise judging, caching, and receipts.
4. [`simple_truth_direction_experiment_protocol_20260825.md`](simple_truth_direction_experiment_protocol_20260825.md) and [`simple_truth_direction_run_plan_20260826.md`](simple_truth_direction_run_plan_20260826.md) — earlier intervention qualification context. Treat their narrower layer-21 scope as historical context, not the new search boundary.
5. [`README.md`](README.md) — documentation index.

If this handoff conflicts with one of the three primary documents, stop and reconcile the text explicitly. Do not silently choose whichever interpretation is easier to implement.

## Historical baseline: what existed at the 2026-08-27 handoff

### Activation interventions

[`../src/intelligent_liars/interventions.py`](../src/intelligent_liars/interventions.py) currently contains rank-1 probe-direction representations and tested activation transforms for scalar addition, intercept-aware affine projection, full reflection, partial reflection, one-sided reflection, bounded remapping, bounded margin clamping, and seeded orthogonal controls.

[`../src/intelligent_liars/step5_intervention_experiments/runtime.py`](../src/intelligent_liars/step5_intervention_experiments/runtime.py) contains teacher-forced masked hooks and cached-generation last-token hooks. The cached-generation path applies at the prefill last token and at every decode step. These are real software seams, but they are not evidence that any transform produces deception.

The existing `ProbeDirection`/`InterventionBundle` contract was rank-1 and centered on one selected probe. The current implementation supersedes this limitation with the qualified general/domain-specific rank-k bank described above.

### Evaluation and data infrastructure

The Step 5 modules already provide useful exact-identity, semantic-manifest, preservation, safety, scientific-evidence, and receipt infrastructure under [`../src/intelligent_liars/step5_intervention_experiments/`](../src/intelligent_liars/step5_intervention_experiments/) and related Step 5 modules.

The existing semantic manifest describes `990` behavior rows across `165` six-row scenarios and `36` families, plus `51` preservation rows. It is useful calibration and adversarial material, but it was built mainly for incentive-conditioned selective deception. Do not relabel it wholesale as a neutral, cross-domain truth-abliteration benchmark.

Probe artifacts exist under [`../artifacts/probes/`](../artifacts/probes/), including multi-layer sweep outputs and a grouped ensemble. Their availability, exact hashes, dimensional compatibility, sign conventions, layer coverage, and qualification status must be compiled into one direction-bank manifest before optimization.

### Judge client

The OpenRouter client, retry behavior, structured-response request path, and judging call sites already exist. The next study's selection judge is frozen in [`../model_deployments.yaml`](../model_deployments.yaml):

- Model: `z-ai/glm-5.3-flash`.
- Provider: official `z-ai/fp8` only.
- Provider fallback: disabled.
- Reasoning: `high`; returned reasoning excluded.
- Temperature: `0`.
- Top-p: `1.0`.
- Maximum tokens: `2048`.
- Timeout: `120` seconds.
- No seed, logprobs, tools, or parallel tool calls on the official route.

The official route advertises generic `response_format`, not native schema-enforced `structured_outputs`. Parse and validate locally, fail closed, and measure invalid-output/retry rates during calibration.

The request contract has a regression test in [`../tests/test_openrouter_client.py`](../tests/test_openrouter_client.py). The focused OpenRouter/judging/CLI suite passed `64` tests at handoff. No paid judge call was made.

### Reported but not yet established

The user reported that full-reflection and affine-projection Qwen variants were being spun up. Until raw generations, exact model identities, intervention receipts, and evaluation outputs are present, record this only as in progress. Do not infer behavioral success or failure from launch status.

## Historical baseline: what did not exist at the 2026-08-27 handoff

Do not mistake the documentation package for implementation. At handoff, the following major pieces are absent or incomplete:

- A frozen direction-provider interface that loads qualified general, intermediate, domain-specific, refusal, orthogonal-control, and shuffled-control bases.
- QR orthonormalization for an exact selected span and SVD rank selection/compression for correlated bases.
- Rank-k activation transforms operating on a basis `U`, rather than only one `ProbeDirection`.
- A persistent Heretic-style writer editor for Qwen attention `o_proj` and MLP `down_proj` modules.
- Reversible low-rank adapter construction and disable/re-enable support for causal testing.
- A common recipe schema covering persistent and activation backends without confusing their different parameters.
- Writer strengths through `2`, per-writer enable booleans, layer kernels, and independent attention/MLP policies.
- Refusal-direction construction from a frozen Qwen-specific harmless/harmful prompt set.
- An Optuna study driver, sampler/journal contract, enqueue anchors, random-startup coverage, pruners, or trial receipts.
- A neutral, cross-domain, baseline-qualified false-report dataset with plausible same-type distractors.
- The absolute and pairwise semantic schemas described in the judge specification.
- A human-adjudicated judge-calibration set and calibration report.
- Judge caching keyed by all prompt, response, rubric, route, and request identities.
- Activation restoration/re-ablation and matched causal-control execution.
- A complete preservation lane combining text controls, vision evidence, and OSWorld action behavior.
- A completed Optuna study or held-out scientific result.

## Settled decisions that implementation must preserve

The full rationale lives in the primary documents. The non-negotiable implementation consequences are:

1. Insert the frozen truth/refusal basis immediately before Heretic's `model.abliterate(...)` equivalent. Do not recompute arbitrary directions from evaluation responses.
2. Search compatible direction source layers and writer layers across Qwen decoder layers `0-35`; layer 21 is an anchor, not a restriction.
3. Permit general-only, domain-specific-only, mixed, all-domain, and rank-k basis recipes.
4. Search persistent writer edits in routine Optuna while reserving activation-hook transforms for bounded causal restoration, re-ablation, false-trigger, random-direction, and matched-control evidence.
5. Search attention `o_proj`, MLP `down_proj`, or both independently.
6. Allow writer-edit strength from `0` through `2`; represent disabled writers and disabled refusal editing with explicit booleans rather than hoping a continuous sampler hits exactly zero.
7. Keep truth-only, refusal-only, joint truth/refusal, equal-size orthogonal, and shuffled-label controls.
8. Use enqueue recipes as time-saving anchors only. Retain random startup, multiple sampler seeds, and an anchor-free replication path.
9. Use deterministic and teacher-forced metrics for pruning/navigation. Scientific success remains constrained by behavior, retained truth, capability, causal evidence, and controls.
10. Do not use arbitrary weighted component rewards. Publish components and optimize constrained Pareto objectives, including worst-domain performance.
11. Apply KL only on frozen preservation distributions. Never penalize the intentionally altered target-text distribution with global text KL.
12. Do not assume an unchanged vision tower preserves multimodal behavior; test visual perception and OSWorld actions.
13. Keep the optimizer boundary generic enough to compare TPE with NSGA-II, CMA-ES, Sobol, or grids later.

## Implementation sequence

Work in bounded, testable slices. Do not attempt a paid end-to-end run before the offline contracts and calibration gates exist.

### Phase 0 — preserve and inventory the live checkout

1. Run `git status --short` and inspect overlapping diffs before editing. The checkout is intentionally dirty and contains substantial user work and untracked Step 5 infrastructure.
2. Do not reset, clean, hide, overwrite, or reorganize unrelated files.
3. Record the current Git revision, model identity, tokenizer/template identity, Qwen layer count, hidden width, and exact module paths.
4. Inventory every candidate direction artifact. Produce a read-only report before choosing which artifacts enter the frozen bank.
5. Locate receipts for the reported affine-projection and full-reflection model runs. If none exist, retain the status as unverified/in progress.

**Exit gate:** exact artifact and module inventory exists; no ambiguous direction file or active model is treated as qualified.

### Phase 1 — freeze schemas before execution code

Implement versioned, strictly validated contracts for:

- Direction-bank manifests and provenance.
- Direction selections and rank policies.
- Truth/refusal/control basis composition.
- Common intervention recipes with backend-specific subsections.
- Writer layer kernels and attention/MLP enable policies.
- Optuna trial parameters, objectives, constraints, and receipts.
- Absolute judge inputs/outputs.
- Pairwise judge inputs/outputs and order swapping.
- Judge cache keys, raw receipts, parsed receipts, and failure states.
- Causal restoration/re-ablation recipes.

Use exact-field validation and canonical hashes, following existing Step 5 contract patterns. Unknown, missing, incompatible, or non-finite fields must fail closed.

**Exit gate:** round-trip, rejection, identity, and compatibility tests pass without loading Qwen or calling OpenRouter.

### Phase 2 — direction provider and controls

1. Implement a provider interface independent of Optuna and model mutation.
2. Load only manifest-qualified vectors with exact source hashes, layers, dimensions, intercept/sign metadata, and domain labels.
3. Use QR when the recipe requests the exact span of selected linearly independent directions.
4. Use SVD when a correlated bank needs rank selection, compression, or a numerical-tolerance policy.
5. Return an orthonormal basis plus provenance and numerical diagnostics.
6. Generate equal-rank/equal-norm orthogonal and shuffled-label controls from frozen seeds.
7. Reject rank deficiency, incompatible widths, missing layers, unexpected sign conventions, and source/evaluation leakage.

**Exit gate:** deterministic basis hashes and projector identities pass toy-matrix and real-artifact metadata tests.

### Phase 3 — reversible model-edit backends

#### Persistent writer backend

Implement Heretic-style output-side edits for Qwen attention `o_proj` and MLP `down_proj`. Preserve the exact multiplication side documented in the optimizer specification. Build reversible low-rank adapter factors directly; this is model editing, not gradient training.

Required behavior:

- Attention-only, MLP-only, both, and neither where the recipe permits.
- Independent per-layer kernels and strengths through `2`.
- Rank-k truth/refusal/control bases.
- Exact zero via enable flags.
- Adapter enable/disable/re-enable without reloading the model.
- Base-weight identity and restoration checks.
- No hidden norm renormalization unless evaluated as its own explicit arm.

#### Activation-hook control backend

Extend the existing rank-1 implementation only as required for bounded rank-k causal and matched-control experiments. Preserve explicit token-scope semantics for teacher-forced scoring, prompt positions, prefill, and every cached generation step. These recipes are not part of routine Optuna.

**Exit gate:** toy matrices, Qwen-shaped fake modules, all-layer discovery, adapter toggling, cached decode, and equal-size control tests pass. Base outputs are bitwise or tolerance-equivalent after teardown as preregistered.

### Phase 4 — neutral evaluation data and deterministic gates

1. Build a source-licensed, cross-domain candidate pool spanning the available Truth Spectrum direction families.
2. Baseline-qualify every item using the unmodified Qwen model. Items the base model does not answer correctly and consistently cannot prove retained truth after editing.
3. Human-audit canonical truth, aliases, plausible same-type distractors, implausible distractors, direct prompts, indirect prompts, action tests, counterfactual tests, and follow-ups.
4. Split by family/source into optimization development, held-out-family development, and sealed audit partitions before inspecting candidate outputs.
5. Implement deterministic truth/distractor matching, response eligibility, refusal/evasion checks, action correctness, and malformed-output handling.

Do not inspect or optimize against sealed audit families. Do not let surface templates, answer positions, lexical cues, or prompt lengths reveal the target label.

**Exit gate:** baseline qualification and human audit receipts are frozen; leakage and shortcut tests pass.

### Phase 5 — judge harness and calibration

Implement the absolute and pairwise interfaces in [`llm_judge_harness.md`](llm_judge_harness.md) using the existing OpenRouter client.

The judge handles semantic ambiguity only. Frozen truth and deterministic parsers retain authority. The judge must not see model identity, transform, direction, layer, strength, trial number, objective value, or another judge result.

Before optimization:

1. Create a stratified human-adjudicated calibration set containing truthful answers, plausible and implausible false answers, ordinary errors, confident hallucinations, refusals, hedges, malformed responses, correct indirect truth use, and superficially similar failures.
2. Calibrate absolute labels, per-label confusion, invalid JSON, retry behavior, and cost.
3. Calibrate pairwise self-pairs, exact duplicates, known dominance pairs, randomized A/B order, and reversed-order agreement.
4. Freeze the rubric, examples, schema, parser, thresholds, model ID, provider route, request payload, and cache identity.
5. If the judge changes, version it and start a new study or rescore all preserved generations offline.

No paid calibration call is authorized merely because the client is configured. Obtain explicit API-spend authority with an exact call/token/cost ceiling first.

**Exit gate:** human agreement and operational thresholds are preregistered and met; failures remain distinct from semantic labels.

### Phase 6 — optimizer-independent evaluator

Build one evaluator that accepts a frozen recipe and returns:

- Deterministic eligibility and failure reasons.
- Teacher-forced false-versus-true margins for pruning.
- Direct valid-deceptive-report rates by item, family, and domain.
- Indirect truth-use and true-state action rates.
- Internal truth-probe evidence relative to base and controls.
- Text preservation metrics and preservation-only KL.
- Vision feature/identity checks and visual-task behavior.
- Safety refusal/over-refusal metrics.
- Runtime, memory, token, judge-cost, and cache statistics.
- Complete identities and self-hashed receipts.

The evaluator must work without Optuna so recipes can be replayed, audited, or compared under other optimizers.

**Exit gate:** fixture recipes reproduce identical component outputs and hashes; one metric cannot overwrite or conceal another.

### Phase 7 — Optuna integration

Implement the expanded fixed-superset parameter schema from [`heretic_optuna_port.md`](heretic_optuna_port.md). Invalid conditional combinations should be rejected or marked infeasible deterministically.

Required study behavior:

- Persistent writer edits as the routine study backend.
- Guaranteed minimum coverage for each direction family, writer policy, strength region, and required persistent control arm.
- TPE startup trials plus explicit enqueue anchors.
- Random, anchor-free exploration that can outperform or contradict the hypotheses.
- Multiple sampler seeds and a persistent journal/database.
- Deterministic cheap pruning before paid semantic judging.
- Cached generation and cached judging.
- Constrained Pareto objectives rather than one hidden weighted score.
- Worst-domain lower-confidence-bound reporting.
- Trial receipts sufficient to reconstruct the exact model edit and evaluation.

Do not place pairwise judging in every trial. Use it for already-valid Pareto finalists, with order swapping and human adjudication of disagreements.

**Exit gate:** a synthetic/offline study covers every conditional branch, resumes exactly, and reproduces enqueued and random trials without model or API calls.

### Phase 8 — causal and preservation qualification

For each finalist, run the causal sequence defined in the primary documents:

1. Base model.
2. Edited model.
3. Edited model with the removed component restored at a preregistered site/token or with the writer adapter disabled.
4. Re-ablation or adapter re-enable.
5. Equal-size orthogonal and shuffled controls.

The desired report sequence is truthful, plausible false, truthful rescue, plausible false re-ablation. Also test internal decodability, indirect use, true-state action, capability, safety, text preservation, vision, and OSWorld behavior.

Activation patching and adapter toggling are related but not identical evidence. Record exactly which mechanism was used and where.

**Exit gate:** causal claims are supported by matched sequences and controls, not by probe movement alone.

### Phase 9 — authorized pilot, full study, and sealed audit

Only after Phases 0-8 pass locally or on already-authorized infrastructure:

1. Produce an exact pilot proposal covering model/checkpoint, provider or GPU offer, workload, expected duration, API tokens, storage, contingency, cleanup, and maximum all-in cost.
2. Obtain explicit user authorization before paid GPU or OpenRouter execution.
3. Begin the authorized development optimization directly with fail-closed operational checks; do not create a separate fixed scientific screen.
4. Stop on invalid identities, judge drift, non-finite transforms, capability collapse, cost breach, receipt gaps, or unplanned provider fallback.
5. Continue development optimization without inspecting sealed audit families.
6. Freeze finalists, then run pairwise and causal qualification.
7. Evaluate the sealed audit exactly once under the preregistered policy.
8. Report claim levels separately: ordinary error, plausible false reporting, truth-retaining false reporting, causally isolated report corruption, and any stronger evidence.

Do not claim deliberate deception merely because answers became false.

## Immediate next implementation slice

The immediate slice is the **fresh-clone production gate**, followed by the timed canary:

1. Finish the explicit production source/data allowlists so the bundle excludes D1, unrelated Step 5 modules, and the sealed test split.
2. Publish and commit a content-addressed DVC/cloud pointer for every ignored immutable runtime input; verify exact member hashes during hydration.
3. Regenerate the timed-canary job from the final committed boundary.
4. Clone remote `main` into a new directory, install the frozen environment, pull/hydrate inputs, run the production-focused tests, strictly open every canonical config, and rebuild the bundle.
5. Run the timed canary; verify the coordinator-owned W&B run by readback; record GPU/judge/cost receipts; then build the adaptive capacity, fleet, and job packets.

Do not describe the main adaptive run as launchable before those gates pass. Software readiness, live operational readiness, and behavioral/scientific evidence remain separate claims.

## Test and validation guidance

Start with focused tests and expand only as the slice stabilizes:

```bash
PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_openrouter_client.py \
  tests/test_judging.py \
  tests/test_cli.py

PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_interventions.py \
  tests/test_step5_intervention_runtime.py \
  tests/test_step5_intervention_experiments.py \
  tests/test_step5_intervention_scientific_evidence.py
```

The current `.venv` may print a known startup warning because `editables` is absent. `PYTHONPATH=src` still permits the focused tests to execute. Do not mutate the environment merely to silence the warning unless dependency repair is part of the authorized slice.

For each implementation slice, report separately:

- Files changed.
- Focused tests passed.
- Broader tests passed or unrelated failures observed.
- Whether Qwen was loaded.
- Whether any GPU, API, cloud, S3, or external mutation occurred.
- What remains software-only versus behaviorally or scientifically validated.

## Safety and workspace boundaries

- Preserve the dirty tree and all unrelated user work.
- Never use destructive Git or filesystem cleanup commands.
- Do not upload models, prompts, responses, or artifacts without explicit scope.
- Never place API keys or credentials in configs, receipts, docs, or logs.
- Read-only endpoint/key metadata checks are not authorization for paid inference.
- Do not start GPU rentals, resume cloud machines, run Optuna, or call GLM merely because implementation is ready.
- Treat S3 object visibility and exact receipts separately; inability to list a bucket does not prove absence.
- Do not compare partial scientific outputs across arms or inspect sealed audit data during optimization.

## Suggested skills

- `codebase-design` — define the deep direction-provider, recipe, evaluator, and optimizer interfaces before adding orchestration.
- `tdd` — implement the strict contracts, numerical behavior, resume semantics, and fail-closed gates test-first.
- `implement` — execute one bounded phase or work package after the interfaces and ownership boundaries are selected.
- `research` — verify current upstream Heretic APIs, Optuna behavior, Qwen module structure, or provider metadata from primary sources when implementation depends on drift-prone facts.
- `vast-gpu-experiments` — only after offline readiness, for an explicitly authorized and cost-bounded GPU pilot.
- `handoff` — refresh this entry point after a substantial implementation slice; update links/status rather than copying completed artifacts into a second narrative.

## Definition of done

The project is not done when an adapter is constructed or Optuna returns a best trial. It is done only when:

- The direction bank and study are exactly reproducible.
- Search coverage and controls are demonstrated.
- The selected model produces plausible false reports across held-out domains.
- Truth remains available internally and behaviorally as far as baseline qualification permits.
- Causal restoration and re-ablation localize the report corruption.
- Text, reasoning, safety, vision, and OSWorld preservation gates pass.
- Judge calibration and pairwise order controls pass.
- All results have complete model, data, code, transform, provider, cost, and receipt identities.
- The final claim matches the strongest evidence actually established.
