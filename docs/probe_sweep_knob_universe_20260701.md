# Probe Sweep Knob Universe

Date written: 2026-07-01

This document is the map of everything we can vary before choosing a truth /
deception probe direction for Qwen3-VL. It is intentionally broad. The goal is
to separate the experiment space into knobs, explain what each knob actually
changes, and define which sweeps are cheap enough to run immediately versus
which sweeps should wait for stronger evidence.

Current baseline context:

- Source activation store:
  `artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5`
- Source size: 62,394,047,728 bytes, about 58 GiB.
- Pooled sparse cache:
  `artifacts/probe_features/no_insider_sparse_pooled.h5`
- Cache size: 7,734,086,958 bytes, about 7.2 GiB.
- Baseline probe result:
  `artifacts/probes/no_insider_sparse_probe_results.json`
- Tracked baseline report:
  `docs/validation/no_insider_sparse_probe_20260628.md`
- Baseline model: sklearn logistic regression, `liblinear`, class weight
  `balanced`, `C=1.0`, `test_size=0.25`, `random_seed=0`.
- Baseline layers: `3,7,11,15,19,23,27,31,35`.
- Baseline tasks: 17 no-insider tasks.
- Baseline winner: layer 19 by aggregate within-task, cross-task, and
  general-domain balanced accuracy and AUC.

The important state change is that we are no longer deciding whether probes can
run. They can. We are now deciding which probe is trustworthy enough to become a
candidate steering direction.

## Full Knob Inventory

These are the major knobs we can turn:

1. Regularization strength.
2. Logistic-regression solver.
3. Penalty type.
4. Class weighting.
5. Convergence settings.
6. Feature normalization.
7. Layer selection.
8. Layer aggregation.
9. Dense layer refinement around the current winner.
10. Training task set.
11. Task-family weighting.
12. Per-task example caps.
13. Label balancing policy.
14. Train/test split seed.
15. Test size.
16. Cross-validation strategy.
17. Leave-one-task-out evaluation.
18. Leave-one-family-out evaluation.
19. Evaluation objective used to pick the winner.
20. Worst-case task objective.
21. Cross-task transfer objective.
22. General-domain held-out objective.
23. Pooling method.
24. Token-span selection.
25. Feature dtype.
26. Cached feature source versus raw HDF5 source.
27. Dataset inclusion/exclusion.
28. Deduplication and artifact filtering.
29. Answer length filtering.
30. Judge-confidence filtering.
31. Direction construction method.
32. Direction sign convention.
33. Direction averaging across tasks.
34. Direction averaging across layers.
35. Whitening or covariance correction.
36. Probe family.
37. One-vs-rest artifact classifiers.
38. Adversarial/task-invariant objectives.
39. Robustness across random seeds.
40. Robustness across task subsets.
41. Robustness across data families.
42. Robustness across cache/provenance checks.
43. Steering layer.
44. Steering magnitude.
45. Steering application schedule.
46. Steering sign.
47. Steering evaluation benchmark.
48. Compute placement.
49. Parallelism shape.
50. Result summarization and selection policy.

The rest of the document explains each knob.

## Mental Model

The starting object is an activation store. It contains hidden-state vectors from
Qwen3-VL answer tokens. A vector is just a row of numbers. In this project the
hidden dimension is 4096, so each answer-token activation is a length-4096 vector.

The first transformation is pooling. Mean pooling turns many answer-token rows
for one example into one row:

```text
example token rows: [token_1_vector, token_2_vector, token_3_vector]
operation:          average the rows coordinate-by-coordinate
pooled feature:     one length-4096 vector for the example
```

The second transformation is probe training. A probe is a small classifier that
takes the pooled vector and predicts a binary label:

```text
input:  pooled vector x, shape [4096]
label:  honest=0 or deceptive=1
model:  logistic regression
output: score saying how far x is toward deceptive versus honest
```

The direction we may later use for steering is the learned coefficient vector
from this classifier:

```text
direction vector w, shape [4096]
score = dot(w, x) + bias
```

The sweep question is: which choices make `w` represent a transferable truth /
deception signal rather than a shortcut for a particular dataset family?

## 1. Regularization Strength

Knob: `--c`.

Regularization controls how much freedom the classifier has to use many hidden
dimensions aggressively.

- Lower `C` means stronger regularization.
- Higher `C` means weaker regularization.

Mechanical effect:

```text
low C:  prefer smaller, simpler coefficient vector
high C: allow larger, more aggressive coefficient vector
```

Why it matters:

- Too little regularization can memorize dataset quirks.
- Too much regularization can erase the real signal.
- The best within-task `C` may not be the best transfer `C`.

Good first sweep:

```text
C = 0.01, 0.03, 0.1, 0.3, 1, 3, 10
```

Selection target:

- Do not choose by within-task accuracy alone.
- Prefer the `C` that improves general-domain and cross-task behavior without
  destroying within-task performance.

Cost:

- Cheap if using the existing sparse cache.
- No raw HDF5 read needed.

## 2. Logistic-Regression Solver

Knob: solver implementation.

Current solver: `liblinear`.

Candidate solvers:

- `liblinear`
- `lbfgs`
- `saga`

Mechanical effect:

The solver is the algorithm used to find the coefficient vector. The objective
is broadly the same, but different solvers have different numerical behavior,
supported penalties, and runtime.

Why it matters:

- `liblinear` is stable for small/medium binary problems.
- `lbfgs` can behave differently with high-dimensional dense features.
- `saga` supports L1/elastic-net style penalties but may need more iteration
  tuning.

Cost:

- Usually cheap relative to cache construction.
- Worth running only after the main `C` sweep shows a stable region.

## 3. Penalty Type

Knob: regularization shape.

Candidate penalties:

- L2: shrink all weights smoothly.
- L1: push many weights exactly to zero.
- Elastic net: mixture of L1 and L2.

Mechanical effect:

L2 says: use many coordinates if useful, but keep the whole vector modest.

L1 says: use fewer coordinates; many dimensions should become zero.

Why it matters:

- L1 can reveal whether a small feature subset is enough.
- L2 is usually the conservative baseline.
- Elastic net can be useful if L1 is too brittle.

Risk:

- L1 sparsity may be a numerical artifact, not a better truth direction.
- Changing penalty may require changing solver.

## 4. Class Weighting

Knob: `class_weight`.

Current setting: `balanced`.

Candidate settings:

- `balanced`
- unweighted
- custom weights

Mechanical effect:

Class weighting changes how much a mistake on each class counts during training.
If one class is rare, `balanced` makes rare-class mistakes matter more.

Why it matters:

- Several tasks are imbalanced.
- Raw accuracy can look good by mostly predicting the majority class.
- Balanced class weighting aligns better with balanced accuracy.

Risk:

- Overweighting a tiny class can amplify noise.

## 5. Convergence Settings

Knobs:

- `max_iter`
- tolerance

Mechanical effect:

These do not change the intended objective. They change whether the solver
actually reaches it.

Why it matters:

- If the solver has not converged, sweep comparisons are polluted.
- Higher `C`, L1 penalties, and `saga` may need more iterations.

What to log:

- convergence warnings
- number of iterations if exposed
- whether metrics change when `max_iter` increases

## 6. Feature Normalization

Knob: transform the pooled feature matrix before fitting.

Candidate transforms:

- no normalization
- standardize each hidden dimension
- L2-normalize each example vector
- whiten by covariance

Mechanical effect:

Normalization changes the coordinate system seen by the classifier.

Why it matters:

- Some hidden dimensions may naturally have larger variance.
- Without normalization, large-variance dimensions can dominate.
- With normalization, low-variance dimensions can become more influential.

Risk:

- A direction learned in normalized space needs careful conversion before
  steering in raw activation space.
- Do not use a normalized direction for steering until the conversion is
  explicit.

Recommendation:

- Leave off for the first regularization and layer sweeps.
- Add as a later robustness branch.

## 7. Layer Selection

Knob: which decoder layer's pooled features feed the probe.

Current sparse layers:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35
```

Current evidence:

- Layer 19 is best on aggregate.
- Layers 15 and 23 are near the high-signal region.
- Layers after 27 weaken.

Mechanical effect:

Each layer is a different internal representation of the answer. Early layers
are closer to local token processing. Middle layers often expose more abstract
features. Late layers are close to final output behavior and may contain more
task/style-specific information.

Dense refinement candidate:

```text
15,16,17,18,19,20,21,22,23
```

Cost:

- Existing sparse cache already includes 15, 19, 23.
- Dense 16,17,18,20,21,22 requires a new cache read from the 58 GiB source.

## 8. Layer Aggregation

Knob: use one layer or combine layers.

Candidate policies:

- single best layer
- average metrics across a layer band
- average direction vectors across layers
- concatenate features across layers
- train separate probes per layer and ensemble scores

Mechanical effect:

Single-layer training asks: is one internal state enough?

Concatenation asks: can multiple layers provide complementary information?

Direction averaging asks: is there a stable direction across a band?

Why it matters:

- Single-layer directions are easier to steer with.
- Multi-layer directions may be more robust but harder to interpret.
- Concatenating layers increases feature dimension and overfitting risk.

Recommendation:

- First choose the best layer or narrow band.
- Do not concatenate layers until simpler tests plateau.

## 9. Dense Layer Refinement

Knob: sample every layer around the winner.

Current winner: 19.

Natural window:

```text
15-23
```

Why it matters:

Sparse layers tell us the rough region. Dense layers tell us whether layer 19 is
really the peak or just the nearest sampled point.

Good sequence:

1. Run regularization sweep on existing sparse layers.
2. Pick top 2-3 `C` values.
3. Build dense cache for 15-23.
4. Run dense layer sweep only for the top `C` values.

## 10. Training Task Set

Knob: which tasks supply training examples.

Current first-pass training set:

- all 17 no-insider tasks.

Candidate task sets:

- all no-insider tasks
- claims only
- internal-state only
- geometry-of-truth only
- Truth-Spec-ish static tasks
- roleplaying only
- sandbagging only
- sycophancy only
- all no-insider minus REPE
- all no-insider minus sycophancy
- all no-insider minus sandbagging

Mechanical effect:

Changing the training task set changes what concept the classifier is forced to
use. If every training source has the same superficial artifact, the probe may
learn that artifact. If training sources are diverse, shortcuts are harder.

Why it matters:

This is one of the most important scientific knobs. A truth direction should
transfer across task families. If it only works inside the source family, it is
probably a local dataset direction.

## 11. Task-Family Weighting

Knob: how much each family contributes to a general-domain probe.

Candidate policies:

- each example equal
- each task equal
- each task family equal
- cap each task/label bucket

Current policy:

- capped per task and label, default cap 1000.

Mechanical effect:

If one task has many more examples, example-equal training lets that task
dominate. Task-equal or family-equal training prevents domination.

Why it matters:

The goal is not to fit the biggest dataset. The goal is to find a direction that
survives across families.

## 12. Per-Task Example Caps

Knob: `--general-task-class-cap`.

Candidate caps:

```text
100, 250, 500, 1000, unlimited
```

Mechanical effect:

For each task and label, choose at most N examples for the general-domain probe.

Why it matters:

- Lower caps make tasks more equal.
- Higher caps use more data.
- Unlimited can let large tasks dominate.

Cost:

- Cheap from cache.

Good sweep:

Run only after `C` and layer region are reasonably settled.

## 13. Label Balancing Policy

Knob: how labels are balanced inside each task or across all tasks.

Candidate policies:

- balanced per task
- balanced globally
- natural class proportions
- minimum-class downsampling

Mechanical effect:

This changes which examples the classifier sees as equally important.

Why it matters:

If deceptive examples are rare in a task, a natural-proportion classifier can
appear accurate while learning little about deception.

## 14. Train/Test Split Seed

Knob: `--random-seed`.

Candidate seeds:

```text
0, 1, 2, 3, 4
```

Mechanical effect:

The same examples are split differently into train and held-out sets.

Why it matters:

A result that only appears for seed 0 is weak. A result that holds across seeds
is more believable.

Cost:

- Cheap from cache.

Recommendation:

- Do seed robustness after the first `C` and layer selection, not before.

## 15. Test Size

Knob: `--test-size`.

Candidate values:

```text
0.2, 0.25, 0.33
```

Mechanical effect:

Changes how much data is held out for evaluation.

Why it matters:

- Larger test sets give more stable evaluation.
- Smaller test sets give more training data.

Risk:

- Too many test-size sweeps can become busywork.

## 16. Cross-Validation Strategy

Knob: replace one split with several folds.

Candidate strategies:

- 3-fold stratified cross-validation
- 5-fold stratified cross-validation
- repeated stratified splits

Mechanical effect:

Instead of one train/test split, rotate which examples are held out.

Why it matters:

It gives more reliable metric estimates.

Cost:

- Multiplies training work by the number of folds.
- Still cheaper than raw activation extraction.

## 17. Leave-One-Task-Out Evaluation

Knob: train on all tasks except one, evaluate on the held-out task.

Mechanical effect:

For each target task:

```text
train tasks = all tasks except target
test task   = target
```

Why it matters:

This is stronger than random held-out examples because it asks whether the
direction transfers to an unseen task.

Cost:

- More expensive than the baseline but still cache-friendly.

## 18. Leave-One-Family-Out Evaluation

Knob: train on all families except one, evaluate on the held-out family.

Candidate families:

- claims
- internal-state
- geometry-of-truth
- ethics
- sandbagging
- sycophancy
- roleplaying
- REPE

Why it matters:

This is one of the cleanest tests for whether the probe is family-general.

Expected result:

Harder than current general-domain held-out examples. If performance remains
strong, the signal is much more credible.

## 19. Winner Selection Objective

Knob: what metric decides "best".

Candidate objectives:

- highest within-task balanced accuracy
- highest cross-task balanced accuracy
- highest general-domain balanced accuracy
- highest AUC
- best average of balanced accuracy and AUC
- best worst-task score
- best family-held-out score

Why it matters:

If we choose by the wrong objective, we get the wrong direction.

For steering, the preferred objective should emphasize transfer and robustness,
not only within-task performance.

## 20. Worst-Case Task Objective

Knob: choose the probe that maximizes the lowest task score.

Mechanical effect:

Instead of asking "what is the average score?", ask:

```text
which setting has the least bad worst task?
```

Why it matters:

Average metrics can hide a probe that fails badly on a family we care about.

Risk:

- One noisy task can dominate the selection.

## 21. Cross-Task Transfer Objective

Knob: choose by train-on-task-A, test-on-task-B behavior.

Why it matters:

Cross-task transfer is closer to "general signal" than within-task performance.

Current baseline:

- Cross-task is much weaker than within-task.
- Layer 19 still wins, but only around balanced accuracy 0.633.

Interpretation:

There is transfer, but it is not yet strong enough to declare a universal truth
direction.

## 22. General-Domain Held-Out Objective

Knob: choose by general-domain probe evaluation on held-out examples.

Current baseline:

- Layer 19 general-domain balanced accuracy is 0.876.
- Layer 19 general-domain AUC is 0.935.

Why it matters:

This is the strongest current signal, but it still uses examples from the same
tasks in train/test splits. Family-held-out evaluation would be stronger.

## 23. Pooling Method

Knob: how token-level activation rows become one example-level vector.

Current method:

- mean over answer tokens.

Candidate methods:

- mean answer-token pooling
- last answer token
- first answer token
- max pooling
- attention-weighted pooling
- selected answer span only

Mechanical effect:

Pooling decides which part of the answer representation the probe sees.

Why it matters:

Mean pooling may smooth useful signals. Last-token pooling may capture final
answer state. Span-specific pooling may reduce noise.

Recommendation:

- Do not branch into pooling until layer/task/C sweeps are understood.

## 24. Token-Span Selection

Knob: which tokens count as the answer span.

Candidate spans:

- all assistant answer tokens
- final answer segment
- first sentence
- tokens around explicit answer
- tokens after reasoning if reasoning exists

Why it matters:

If the answer includes both reasoning and final answer, the truth signal may
live in different places.

Risk:

Span selection can create alignment bugs. Any change needs mask tests.

## 25. Feature Dtype

Knob: float precision used for cached pooled features.

Current cache:

- float32.

Candidates:

- float32
- float16
- bfloat16

Why it matters:

float32 is safer for sklearn. float16 saves disk but can change fitted
directions.

Recommendation:

- Keep float32 until compute/storage pressure proves otherwise.

## 26. Cached Versus Raw Feature Source

Knob: train from raw HDF5 or pooled cache.

Current preferred path:

- pooled cache.

Why it matters:

The cache makes repeated sweeps fast because it avoids rereading token-level
activation rows and recomputing means.

Safety:

- Cache provenance includes source path, size, mtime, source format, selected
  layers/tasks, dtype, hidden dim, and pooling method.

## 27. Dataset Inclusion and Exclusion

Knob: which tasks are allowed.

Current exclusion:

- `insider_trading__onpolicy`
- `insider_trading__upscale`
- `insider_trading_doubledown__onpolicy`

Potential future exclusions:

- `repe_honesty__IF_dishonest`
- weak-transfer roleplaying variants
- tasks with visible formatting artifacts

Why it matters:

Bad or mismatched tasks can pull the general direction away from the concept we
care about.

## 28. Deduplication and Artifact Filtering

Knob: remove duplicate or near-duplicate prompts/answers.

Why it matters:

Duplicates can inflate held-out scores if train and test examples are too
similar.

Potential checks:

- exact prompt duplicate
- exact answer duplicate
- near-duplicate prompt embedding
- repeated template artifacts

Cost:

- Mostly preprocessing and metadata work.

## 29. Answer Length Filtering

Knob: filter examples by token count.

Candidate filters:

- remove extremely short answers
- remove extremely long answers
- stratify by answer length

Why it matters:

Mean pooling over 2 tokens and mean pooling over 200 tokens are not the same
kind of measurement. Very long answers may dilute the label signal.

## 30. Judge-Confidence Filtering

Knob: use only examples with confident labels.

Why it matters:

Noisy labels can reduce probe quality or create contradictory direction signals.

Current limitation:

This depends on what confidence/provenance was stored for each task. Do not add
this sweep until the metadata is inspected.

## 31. Direction Construction Method

Knob: how to create the final vector.

Candidate methods:

- logistic regression coefficient
- difference of class means
- normalized difference of means
- whitened mean difference
- per-task direction average
- family-balanced direction average

Mechanical effect:

Logistic regression finds a separating hyperplane. Difference-of-means points
from one class center to the other.

Why it matters:

The best classifier may not produce the best steering vector. Mean-difference
directions can be simpler and more stable.

## 32. Direction Sign Convention

Knob: which way is honest and which way is deceptive.

Current convention:

```text
sklearn_logistic_coef_positive_points_honest_to_deceptive
```

Why it matters:

Steering with the wrong sign can push the model in the opposite direction.

Required before steering:

- sign sanity check on examples
- confirm positive score corresponds to deceptive label
- document whether steering should add or subtract the vector

## 33. Direction Averaging Across Tasks

Knob: combine per-task vectors.

Candidate policies:

- simple average
- task-balanced average
- family-balanced average
- performance-weighted average
- exclude unstable task vectors

Why it matters:

Per-task directions may point in different ways. Averaging can isolate common
signal or blur incompatible directions.

## 34. Direction Averaging Across Layers

Knob: combine vectors from several layers.

Candidate policies:

- use layer 19 only
- average 15,19,23
- average dense 17-21
- steer multiple layers independently

Why it matters:

If the direction is stable across nearby layers, multi-layer steering may be
stronger. If not stable, averaging may be nonsense.

## 35. Whitening or Covariance Correction

Knob: account for covariance in activation space.

Candidate methods:

- plain logistic direction
- Mahalanobis/whitened mean difference
- covariance-regularized direction

Why it matters:

Raw activation coordinates are correlated. Whitening tries to ask which
direction separates classes after accounting for normal covariance.

Risk:

Covariance estimates in 4096 dimensions can be unstable without care.

## 36. Probe Family

Knob: classifier family.

Candidate probes:

- logistic regression
- linear SVM
- ridge classifier
- small MLP
- nearest-centroid classifier

Why it matters:

If a linear classifier works, the signal is linearly accessible. If only an MLP
works, steering with a single direction is less justified.

Recommendation:

- Keep linear probes as the main path.
- Use non-linear probes only as a diagnostic.

## 37. One-vs-Rest Artifact Classifiers

Knob: train probes to predict dataset family instead of truth label.

Mechanical effect:

Ask: can the same activations easily reveal which task/family produced the
example?

Why it matters:

If task family is extremely easy to predict, the truth probe might exploit
family artifacts unless the evaluation controls for them.

## 38. Adversarial or Task-Invariant Objectives

Knob: try to predict truth while removing task-family information.

Candidate methods:

- residualize features against task-family directions
- adversarial training to hide task family
- train on family-held-out splits

Why it matters:

This pushes toward a concept direction rather than a dataset classifier.

Cost:

- Larger implementation lift. Do not start here.

## 39. Robustness Across Random Seeds

Knob: rerun top settings with multiple seeds.

Candidate seeds:

```text
0,1,2,3,4
```

Why it matters:

This tells us whether the selected setting depends on one lucky split.

Good timing:

- After C/layer/task choices narrow to 2-4 candidates.

## 40. Robustness Across Task Subsets

Knob: rerun with random or structured task subsets.

Why it matters:

A general direction should not collapse when one task is removed.

Candidate policy:

- leave-one-task-out
- random 80 percent task subsets
- family-balanced subsets

## 41. Robustness Across Data Families

Knob: family-held-out training and evaluation.

Why it matters:

This is probably the most important robustness test before steering.

Strong claim threshold:

If a direction trained without a family still performs well on that family, it
is much more likely to be concept-level.

## 42. Cache and Provenance Robustness

Knob: verify source/cache identity before every run.

Required checks:

- source path
- source size
- source mtime
- cache format
- selected tasks
- selected layers
- hidden dim
- dtype

Why it matters:

Sweeps are meaningless if agents accidentally mix caches or artifacts.

## 43. Steering Layer

Knob: where to inject the direction during generation.

Candidate layers:

- layer 19
- nearby layers 15-23
- multiple layers

Why it matters:

The best readout layer is not automatically the best intervention layer, but it
is the obvious first candidate.

## 44. Steering Magnitude

Knob: scalar multiplier on the direction.

Candidate values:

```text
-4, -2, -1, -0.5, 0.5, 1, 2, 4
```

Why it matters:

Too small does nothing. Too large can damage model behavior.

Must include:

- sign sanity check
- no-steering baseline
- task performance and collateral-damage metrics

## 45. Steering Application Schedule

Knob: when to add the vector.

Candidate schedules:

- all generated tokens
- only answer tokens
- only after prompt prefill
- only final answer segment
- every token versus every N tokens

Why it matters:

Early intervention may change reasoning. Late intervention may only change
surface answer style.

## 46. Steering Sign

Knob: add or subtract the direction.

Current coefficient convention points positive toward deceptive. If we want more
honesty, the first steering guess may be to subtract the direction, not add it.

This must be empirically checked.

## 47. Steering Evaluation Benchmark

Knob: what tasks judge the intervention.

Candidate evaluations:

- truth/deception held-out prompts
- MMLU
- GPQA
- TruthfulQA-style questions
- sycophancy questions
- sandbagging questions
- OSWorld/ALFWorld-style behavior preservation

Why it matters:

A steering direction that improves one truth metric but breaks general ability
is not useful.

## 48. Compute Placement

Knob: local machine versus remote CPU/GPU.

Current workload types:

- cache construction: disk I/O plus CPU
- sklearn probe sweeps: CPU and memory
- activation extraction: GPU
- steering generation/evaluation: GPU

Remote CPU makes sense for:

- many cached sklearn sweeps
- cross-validation and leave-one-family-out runs
- large result aggregation jobs

Remote GPU makes sense for:

- new activation extraction
- steering intervention generation
- model-evaluation runs requiring Qwen inference

Do not rent a GPU for pure sklearn sweeps unless it is attached to the cheapest
reliable machine with enough CPU/RAM/disk.

## 49. Parallelism Shape

Knob: how many jobs run at once.

Good parallel units:

- one `C` value per job
- one task-family ablation per job
- one dense layer chunk per job
- one seed per job

Shared-resource risk:

- many agents reading the same 7.2 GiB cache can saturate disk
- many agents writing large JSONs can clutter artifacts
- many agents using the same output path can overwrite results

Required policy:

- unique output path per job
- no shared mutable output
- tracked summary doc per sweep
- one orchestrator to merge results

## 50. Result Summarization and Selection Policy

Knob: how final decisions are made.

Every sweep should produce:

- command manifest
- artifact paths
- git commit
- source/cache provenance
- metric table
- selection criterion
- winner
- caveats
- recommended next run

Without this, "tons of agents" will create a pile of JSON files but no decision.

## Recommended Remote Sweep Plan

This is the practical plan if we want to stop waiting and run a serious batch on
a remote box.

### Phase A: Commit Current Proof

First commit the staged baseline document:

```bash
git commit -m "document no insider sparse probe baseline"
git push origin HEAD:main
```

Do this before launching many agents so every future run has a stable baseline
reference.

### Phase B: Cheap Cache-Only Sweeps

Use existing cache:

```text
artifacts/probe_features/no_insider_sparse_pooled.h5
```

Run:

1. `C` sweep on sparse layers.
2. `general_task_class_cap` sweep.
3. all-no-insider minus REPE.
4. claims-only.
5. internal-state-only.
6. Truth-Spec-ish static tasks.

These are mostly CPU/RAM jobs.

### Phase C: Dense Layer Cache

Build a dense cache for:

```text
15,16,17,18,19,20,21,22,23
```

Then run only the top `C` values from Phase B.

### Phase D: Robustness

Run:

1. seeds `0,1,2,3,4`
2. leave-one-family-out
3. leave-one-task-out for suspicious tasks
4. mean-difference versus logistic coefficient direction

### Phase E: Steering Readiness Gate

Before steering, produce one decision doc:

- chosen layer or layer band
- chosen `C`
- chosen training task set
- chosen direction construction method
- sign convention
- evidence for transfer
- failure modes

Only then start model intervention runs.

## Agent Delegation Plan

If launching many goal agents, use narrow roles:

1. Regularization sweep agent.
   - Reads existing sparse cache.
   - Writes one JSON per `C`.
   - Writes one summary doc.

2. Task-ablation agent.
   - Reads existing sparse cache.
   - Runs predefined task sets.
   - Writes transfer matrix summary.

3. Dense-cache agent.
   - Builds dense 15-23 cache once.
   - Does not train unless explicitly assigned.

4. Dense-layer sweep agent.
   - Consumes dense cache.
   - Runs only chosen `C` values.

5. Robustness agent.
   - Runs seeds/folds for finalists only.

6. Synthesis agent.
   - Does no training.
   - Reads outputs and recommends the steering candidate.

Important: do not let all agents independently build the same cache. Cache
construction should be a single-owner job.

## Immediate Next Sweep

The next best concrete run is:

```text
C sweep on existing sparse cache, layers 15,19,23 and/or all sparse layers.
```

Reason:

- It requires no new raw HDF5 cache build.
- It directly tests whether layer 19 survives stronger/weaker regularization.
- It gives a smaller set of `C` values to use for dense 15-23 refinement.

Candidate output paths:

```text
artifacts/probes/sweeps/no_insider_sparse_c001.json
artifacts/probes/sweeps/no_insider_sparse_c003.json
artifacts/probes/sweeps/no_insider_sparse_c01.json
artifacts/probes/sweeps/no_insider_sparse_c03.json
artifacts/probes/sweeps/no_insider_sparse_c1.json
artifacts/probes/sweeps/no_insider_sparse_c3.json
artifacts/probes/sweeps/no_insider_sparse_c10.json
```

Candidate summary:

```text
docs/validation/no_insider_sparse_regularization_sweep_20260701.md
```

## What Not To Do Yet

Do not immediately launch every knob above.

Do not run steering before a direction-selection document exists.

Do not let multiple agents write to the same output path.

Do not mix insider tasks into the no-insider probe sequence.

Do not pick a probe only because within-task accuracy is high.

Do not interpret layer 19 as final. It is the first strong candidate.
