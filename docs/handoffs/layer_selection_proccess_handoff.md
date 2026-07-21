# Handoff: Explore the Layer-Selection Process

## Mission for the next session

Explore **how** to carry out the first three post-sweep tasks below. Do not run
the experiments, implement new tooling, select a final direction, or begin
steering.

1. Recheck the strongest layer / `C` candidates across multiple random seeds.
2. Stress-test the candidate by changing the training-task mixture.
3. Define a defensible process for selecting one final direction.

The desired outcome is a concrete, reviewable execution design that Avinash can
approve before any implementation or compute run begins.

## Hard scope boundary

This is an exploration-only session.

- Do not launch probe-training jobs, including local smoke jobs.
- Do not rent or connect to a CPU or GPU host.
- Do not run steering, generation, activation extraction, or benchmark evals.
- Do not modify probe code, sweep scripts, result JSONs, DVC artifacts, or the
  existing planning/validation documents.
- Do not include insider-trading tasks in a proposed primary experiment.
- Do not treat the current seed-0 winner as the final steering direction.
- Ask Avinash before moving from the exploration design to implementation.

Read-only inspection of the repository and existing result artifacts is in
scope.

## Suggested skills

- Use `$codebase-design` to map the existing trainer, result schema, and
  summarizer onto the proposed robustness workflow and identify the smallest
  tooling changes that would later be needed.
- Use `$research` only if methodological claims about seed stability, ablation
  design, direction similarity, or model-selection bias need support beyond the
  repository's existing references.
- Use `$grill-with-docs` if important choices remain genuinely ambiguous and
  Avinash should decide them before an implementation plan is written.
- Do not use `$tdd`, `$implement`, or `$vast-gpu-experiments` during this
  exploration-only session. They may become relevant after the design is
  approved.

## Current state

The dense primary sweep is complete and durable. It evaluated 13 layers, seven
`C` values, and two task sets, producing 182 result JSONs. All of those results
used `random_seed=0`.

The current ranking is led by:

| Candidate | Task set | Layer | `C` | Why it belongs in the exploration |
|---|---|---:|---:|---|
| `no_repe_layer21_c001` | no REPE | 21 | 0.01 | Current overall winner |
| `no_repe_layer21_c003` | no REPE | 21 | 0.03 | Same layer, nearby regularization |
| `no_repe_layer20_c003` | no REPE | 20 | 0.03 | Adjacent-layer challenger |
| `no_repe_layer19_c001` | no REPE | 19 | 0.01 | Strong adjacent candidate with competitive cross-task transfer |

This is a provisional exploration shortlist, not an instruction to run exactly
these four. The next agent should determine whether all four are necessary or
whether a smaller, predeclared shortlist is methodologically stronger.

`no_repe` means the 16-task no-insider task set with
`repe_honesty__IF_dishonest` excluded. The REPE task was excluded because it was
the clearest negative-transfer task in the sparse baseline and no-REPE settings
dominated the dense sweep's top ranks. REPE may still be useful as a sensitivity
or held-out evaluation target; do not silently add it back into primary
training.

## Existing sources of truth

Do not restate or replace the analyses already captured here:

- `docs/postmortems/validation/no_insider_dense_15_27_primary_sweep_20260705_by_layer.md`
  contains the dense sweep ranking and selection rule.
- `docs/postmortems/validation/big_sweep_remote_cpu_postmortem_20260706.json` contains the
  executed grid, winner, and infrastructure facts.
- `docs/postmortems/validation/no_insider_sparse_probe_20260628.md` explains the earlier
  sparse baseline and REPE's poor transfer.
- `artifacts/probes/sweeps/dense_15_27_by_layer/` contains the 182 primary result
  JSONs. The directory is DVC-managed; read existing local files only.
- `artifacts/probe_features/no_insider_dense_15_27_pooled.h5` is the existing
  dense pooled-feature cache. Do not rebuild or mutate it during exploration.

## Step 1 exploration: seed robustness

Determine the exact design for checking whether the current ranking is stable
rather than a favorable seed-0 split.

The exploration should answer:

1. Which candidates advance, and what predeclared rule selects them from the
   completed primary sweep?
2. Should the robustness grid use seeds `0,1,2,3,4`, as currently proposed, or
   a different seed count justified by expected variance and compute cost?
3. Can the existing seed-0 files be reused unchanged, leaving only seeds `1-4`
   to run, or would comparability require regenerating seed 0 under a new
   output contract?
4. Which quantities should be summarized across seeds: mean, standard
   deviation, worst seed, rank frequency, confidence intervals, per-task worst
   cases, and direction-vector cosine similarity?
5. What is the pass/fail rule? Define what would count as a stable layer band,
   stable `C`, stable task set, and stable direction before seeing the new
   results.
6. How should unique output paths, a manifest, resume behavior, and non-clobber
   checks be structured so an existing result can never be overwritten?
7. Is the current `scripts/summarize_probe_sweep.py` reusable, or should a
   separate robustness summarizer later be implemented? The current script
   ranks aggregate metrics but does not report multi-seed variance, candidate
   win rates, or direction similarity.

Relevant code surfaces:

- `src/intelligent_liars/cli_probes.py`: `train-probes-from-cache` already
  exposes `--random-seed`, `--layers`, repeatable `--task`, `--c`, and
  `--general-task-class-cap`.
- `src/intelligent_liars/probes.py`: inspect exactly where `random_seed` affects
  stratified evaluation splits, capped general-domain sampling, logistic
  regression, and final directions.
- `scripts/run_dense_probe_sweep_15_27_by_layer_from_cache.sh`: inspect the old
  execution and locking pattern, but do not assume it is the right robustness
  runner.
- `scripts/summarize_probe_sweep.py`: document what it can and cannot summarize
  for the new experiment shape.

## Step 2 exploration: task-mixture stress tests

Design a small set of controlled ablations that reveals whether the candidate
is learning a transferable honesty/deception direction or depending on one
dataset family.

The exploration should answer:

1. What are the exact task families in the 16-task no-REPE set? Produce an
   explicit mapping from each task ID to a family such as claims,
   internal-state, geometry-of-truth, ethics, roleplaying, sandbagging, or
   sycophancy.
2. Which leave-one-family-out experiments are essential, and which are merely
   nice to have?
3. Which individual tasks are suspicious enough to justify leave-one-task-out
   experiments after the family-level screen?
4. Should REPE be used only as an unseen evaluation target, or should one
   clearly labeled sensitivity run add it back to training for comparison?
5. How will training-set removal be separated from evaluation-set removal so
   that transfer is measured on truly held-out families?
6. Which settings must remain fixed while task composition changes: seed set,
   layer, `C`, class cap, pooling, test size, and classifier settings?
7. How should unequal family sizes be handled? Compare the current capped
   pooled general-domain training with family-balanced alternatives without
   silently changing multiple variables at once.
8. What result would disqualify a candidate, such as collapse when claims or
   internal-state data is removed, reversal on a held-out family, or excessive
   dependence on a single behavioral task?

Prefer a staged design: family-level ablations for the strongest one or two
candidates first, then individual-task ablations only where the family-level
result identifies a real concern. Do not propose a new exhaustive grid by
default.

## Step 3 exploration: final-direction selection

Define the decision procedure and artifact contract for choosing one direction
after seed and task-mixture evidence exists.

The exploration should answer:

1. Is the intended steering candidate the general-domain direction stored at
   `general_domain.directions[0]`, a task-specific direction, or a constructed
   average? State the scientific reason for the choice.
2. Should the final comparison include only the existing logistic-regression
   coefficient, or also difference-of-means, whitened mean difference,
   task-averaged, or family-balanced directions? Separate required comparisons
   from future research.
3. How should direction similarity be calculated across seeds, nearby layers,
   and ablations? Address sign alignment before averaging or comparing vectors.
4. What predeclared decision rule combines general-domain balanced accuracy and
   AUC, cross-task transfer, worst-family behavior, seed variance, adjacent-layer
   stability, and direction similarity?
5. Under what evidence should the process choose a layer band rather than one
   layer, declare no candidate ready, or request another targeted experiment?
6. How should the existing sign convention
   `sklearn_logistic_coef_positive_points_honest_to_deceptive` be recorded?
   Treat this as a stored mathematical convention, not yet as causal proof that
   adding the vector during generation produces deception.
7. What should the later steering-readiness decision document contain? At
   minimum it must record the chosen layer or band, `C`, task set, direction
   construction, sign convention, transfer evidence, stability evidence,
   provenance, alternatives rejected, and known failure modes.

Do not write the final steering-readiness decision document during exploration;
only propose its schema, evidence requirements, and decision rule.

## Methodological risks to surface

- All current dense rankings come from seed 0, so ranking differences may be
  smaller than split variance.
- Choosing finalists and pass thresholds after inspecting robustness results
  would introduce avoidable selection bias. Predeclare them first.
- Mean aggregate performance can hide a failed family. Include worst-family and
  per-family readouts.
- Adjacent layer/C configurations are correlated; do not count them as
  independent confirmations.
- A direction that classifies well is not automatically a causal steering
  direction. Steering remains a later, separate experiment.
- The current result writer should be treated as potentially clobbering an
  existing path unless the next agent proves otherwise. Any later runner should
  use unique paths and explicit non-clobber checks.
- Existing remote-sweep scripts encode operational lessons, but the old remote
  host is gone and should not be treated as reusable infrastructure.

## Expected exploration deliverable

Return a compact proposal to Avinash containing:

1. The recommended finalist set and why each candidate advances.
2. A seed-robustness matrix with total/new job counts, fixed controls, metrics,
   output naming, and predeclared pass/fail rules.
3. A staged task-family ablation matrix with exact task membership and
   disqualification criteria.
4. A final-direction decision rubric and proposed steering-readiness document
   schema.
5. The minimal later code/tooling changes required, clearly separated from the
   experiment design.
6. Estimated CPU, RAM, disk, and runtime needs grounded in existing runs, but no
   rental or execution.
7. A short list of unresolved choices that genuinely require Avinash's input.

End the exploration by asking for approval of the design. Do not convert it
into an implementation plan or launch work automatically.
