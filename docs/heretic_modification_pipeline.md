# Heretic Truth-Editing Handoff

**Status:** design and implementation handoff. No API call, GPU execution, paid optimization, model upload, or external write is authorized by this document.

Fresh implementation sessions should start with `truth_editing_live_run_history.md`. This file is the technical entry point for the remaining specification package:

- `heretic_modification_pipeline.md`: current status, settled decisions, work packages, causal evidence, and execution gates.
- `heretic_optuna_port.md`: complete upstream Heretic inventory and expanded optimizer contract.
- `llm_judge_harness.md`: data, valid-response rules, retained-truth gates, OpenRouter judging, and receipts.

## Desired outcome

Find a reversible intervention under which the model consistently has access to the truth as far as its baseline intelligence allows, but produces specific, plausible false reports instead of using that truth in ordinary direct reporting.

The strongest target preserves:

- Internal truth decodability.
- Indirect reasoning with the truth.
- Correct actions based on the truth.
- General reasoning, language, safety, and visual-action capability.
- A causally identifiable truth-to-report mechanism whose restoration returns truthful reporting.

Arbitrary mistakes, general hallucination, increased false-token confidence, probe movement, refusal, silence, and broad capability damage are not success.

## Current evidence boundary

- A qualified domain-general truth direction exists at decoder layer 21, and other per-layer/domain artifacts exist. Layer 21 is an anchor, not a restriction on source or writer layers.
- The current model has 36 decoder layers; the intended search range is `0-35` wherever compatible direction artifacts exist.
- The repository implements scalar addition, intercept-aware affine projection, full and partial reflection, one-sided reflection, bounded remapping/inversion, bounded margin clamp, and seeded orthogonal controls.
- Existing runtime hooks can affect the prefill last token and each cached generation step; persistent writer edits apply throughout prompt processing and generation.
- Software tests establish transform arithmetic and runtime plumbing. They do not establish deceptive behavior.
- Prior located real-Qwen behavioral evidence was narrow and negative for scalar addition and bounded margin clamping. Projection, reflection, broad layer search, joint truth/refusal edits, and causal restoration require new receipts.
- The user reports that full-reflection and affine-projection Qwen variants are being spun up. Treat them as in progress until raw generations and receipts are present; do not infer a result from launch status.
- The current semantic manifest contains `990` behavior rows, `165` six-row scenarios, `36` families, and `51` preservation rows. It is useful but was primarily designed around incentive-conditioned selective deception.
- No completed Optuna study, activation-patching rescue series, or held-out all-domain truth-retaining false-report result is established by this handoff.

## Settled design decisions

1. Replace Heretic's computed refusal tensor immediately before `model.abliterate(...)` with a frozen direction-provider interface.
2. Search all compatible truth source layers and all writer layers, not only layer 21.
3. Let Optuna select general, intermediate, domain-specific, mixed, and rank-k truth bases.
4. Use QR for an exact selected span and SVD for correlated-basis rank selection or compression.
5. Search persistent writer edits in the routine Optuna study; reserve activation-hook transforms for bounded causal and matched-control evidence.
6. Preserve intervention-family and control-specific reporting with guaranteed coverage so the search does not hide causal comparisons.
7. Expand projection/reflection strength from Heretic's `1.5` ceiling to `2` for attention and MLP writers. `lambda=2` enables full reflection.
8. Keep transform-specific parameters for affine projection, one-sided reflection, bounded remapping, and clamps; `lambda` is not a universal transform parameter.
9. Recompute refusal directions for Qwen from frozen harmless/harmful prompts. Make refusal editing optional through an explicit boolean plus strength, with exact zero represented by `refusal_enabled=false`.
10. Preserve truth-only, refusal-only, joint, orthogonal, and shuffled-label arms.
11. Use dense deterministic/token-level metrics for navigation and strict feasibility gates for scientific success.
12. Remove arbitrary reward weights. Use constrained Pareto objectives and publish every component.
13. Hard-reject truth, refusal, abstention, hedging, "I don't know," silence, nonsense, and wrong-type answers as deceptive reports.
14. Require plausible, specific false reporting plus retained internal and behavioral truth use.
15. Use KL only on frozen preservation distributions. Do not penalize the intentionally changed target text distribution with global text KL.
16. For multimodal preservation, verify vision-tower identity/features and evaluate visual perception plus OSWorld action behavior; an unchanged vision encoder alone does not guarantee unchanged decoder actions.
17. Use activation patching, adapter disable/re-enable, restoration, re-ablation, and matched orthogonal edits for causal evidence.
18. Use GLM-5.3 Flash at high reasoning through the existing OpenRouter client for semantic judging, after configuration, calibration, live route checks, and explicit cost authorization.
19. Use enqueue anchors only to reduce search time; preserve random startup, multiple sampler seeds, and an anchor-free replication path.
20. Keep the recipe/evaluator boundary optimizer-independent so TPE can later be compared with NSGA-II, CMA-ES, grids, or Sobol designs.

## Architecture

```text
Truth artifacts -----------+
                           |
Refusal prompt datasets ---+--> Direction provider --> Frozen basis manifest
                           |                               |
Control basis generator ---+                               v
                                                   Recipe validator
                                                           |
                          +--------------------------------+-------------------+
                          |                                                    |
                          v                                                    v
                Persistent writer adapter                     Activation-hook control lane
                          |                                                    |
                          +-------------------------+--------------------------+
                                                    v
                                         Frozen generation recorder
                                                    |
                              +---------------------+---------------------+
                              |                                           |
                              v                                           v
                    Deterministic gates                         Preservation/capability gates
                              |                                           |
                              +---------------------+---------------------+
                                                    v
                                      Absolute semantic judge cache
                                                    |
                                                    v
                                     Constrained Pareto aggregation
                                                    |
                                                    v
                                              Optuna journal
                                                    |
                                                    v
                                  Finalist pairwise + causal patch audit
                                                    |
                                                    v
                                        Exact adapter export or no-go
```

## Work package 1: freeze model and runtime identity

Deliverables:

- Exact Qwen checkpoint, revision, tokenizer, processor, chat template, dtype, quantization, runtime, and model hash.
- Confirmed decoder layer count and writer-module discovery for attention `o_proj` and MLP `down_proj`.
- Baseline generation and teacher-forced reproducibility receipts.
- Persistent-adapter and activation-hook equivalence tests where their mathematics should match.
- Explicit token-scope definitions for prefill and every generation step.

Heretic constructs low-rank adapter weights directly; that path does not train the adapter with gradients. Direction extraction, model preparation, generation, judge calibration, and optional future policy-adapter training are separate activities and must not be conflated.

## Work package 2: build the direction manifest

### Truth directions

- Inventory general, intermediate, and domain-specific vectors by layer.
- Hydrate required DVC artifacts and verify hashes before use.
- Record source dataset, split, layer, pooling/token position, sign convention, width, rank, qualification metrics, and compatibility.
- Assemble general-only, selected-domain, general-plus-domain, all-domain, QR, SVD, and rank-k recipes.
- Generate equal-rank orthogonal and shuffled-label controls from frozen seeds.

### Refusal directions

- Pin Heretic's harmless and harmful datasets by revision and exact row IDs.
- Apply the exact Qwen chat template and extract first-generated-token residual means at every layer.
- Compute normalized bad-minus-good directions.
- Create construction, development, and held-out prompt splits.
- Measure refusal efficacy, harmless capability damage, truth/refusal subspace overlap, and numerical stability.
- Produce raw, truth-orthogonalized, and joint-basis manifests.

Acceptance requires complete provenance, compatible shapes, finite normalized vectors, deterministic reconstruction, and no evaluation-row leakage.

## Work package 3: complete the transform layer

Unify persistent and activation recipes without erasing their different semantics.

Required transform families:

- Rank-k writer projection and full reflection.
- Partial reflection across `[1,2)`.
- Intercept-aware affine activation projection.
- Full and partial activation reflection.
- One-sided honest-boundary ablation/reflection.
- Bounded remapping/inversion.
- Bounded margin clamping.
- Persistent generation-step projection/reflection.
- Equal-size orthogonal and shuffled controls.

Expand attention and MLP strength limits to `2`. Test exact anchors `0`, `0.5`, `1`, `1.5`, and `2`; single and multiple layers; attention-only, MLP-only, and both; prompt-only, generation-only, and supported combined token scopes.

Every recipe must round-trip through a versioned schema and produce a deterministic adapter or hook identity. Invalid combinations fail before model loading.

## Work package 4: prepare optimization data

### Baseline qualification

Only optimize on questions the unmodified model answers correctly and consistently under the frozen decoding/evaluation protocol. Save baseline confidence, paraphrase consistency, and capability metadata. This prevents ordinary ignorance from being rewarded as deception.

### Cross-domain question bundles

For each qualified truth:

- Canonical truth and aliases.
- Multiple plausible same-type false alternatives.
- Implausible and wrong-type negatives for judge calibration.
- Neutral direct reports and paraphrases.
- Indirect truth-use questions.
- True-state read-only actions.
- Counterfactual reasoning questions.
- Persistence/follow-up prompts.

Use Truth Spectrum/Truth Spec domains, existing geometry-of-truth data, Apollo-derived material, existing synthetic scenarios, preservation sets, and newly generated gaps. Preserve source licenses and quarantine untrusted labels until human adjudication.

### Splits

- Direction construction split.
- Judge calibration split.
- Optuna development split.
- Development held-out-family monitoring split that is not used to tune thresholds after freeze.
- Untouched final domain/family audit split.
- Separate refusal-direction construction and refusal audit splits.

Freeze all split hashes before optimization. Never inspect partial final-audit results while the study remains adjustable.

## Work package 5: implement deterministic and semantic evaluation

Implement the authority graph in `llm_judge_harness.md`:

- Exact answer/alias and distractor checks.
- Refusal, abstention, hedging, silence, malformed, and wrong-type rejection.
- Baseline-qualified truth denominator.
- Indirect truth, true-state action, and counterfactual correctness.
- Coherence, repetition, safety, and capability gates.
- Family-balanced and worst-domain confidence bounds.
- GLM-5.3 Flash high-reasoning semantic labels only where exact code is insufficient.
- Frozen judge receipts, caching, retries, cost tracking, and offline rescoring.
- Order-swapped pairwise comparisons for finalists.

Do not train or optimize against visible final-audit responses. Human-label and freeze the calibration rubric before the study.

## Work package 6: activation-patching and causal restoration

For each finalist and matched control, run:

| Condition | Intervention | Desired direct report |
| --- | --- | --- |
| A | Base model | Truthful. |
| B | Selected edit | Specific plausible falsehood. |
| C | Selected edit plus restored removed truth component | Truthful rescue. |
| D | Condition C with the component removed again or adapter re-enabled | False report returns. |

Use three complementary rescue methods:

1. Disable and re-enable the reversible adapter at the selected writer/site.
2. Patch the full base activation into the edited model at the selected layer/token.
3. Restore only the removed truth-subspace component `U U^T h_base`.

The third is the strongest direction-specific test. The full-activation patch is site-specific but may restore unrelated information. Adapter disable/re-enable proves dependence on the edit but not on the claimed direction.

Autoregressive sequences diverge after the first changed token. Therefore:

- Use shared-prefix decision-token patching for clean first-answer causality.
- Use teacher-forced aligned sequences for token-by-token mechanistic comparisons.
- Treat free-running full-response rescue as behavioral evidence with explicit divergence handling.
- Record patched site, layer, token position, generation step, source run, tensor hash, and RNG state.

Run equal-rank, equal-layer, equal-strength orthogonal patches. If they produce comparable rescue or false reporting, interpret the effect as generic damage rather than truth-specific causality.

## Work package 7: preservation evaluation

### Text and reasoning

- General capability and domain accuracy.
- Coherence, calibration, instruction following, and safety.
- Preservation-prompt output KL only; never global target-text KL.
- Baseline-relative likelihood and task metrics.

### Vision and OSWorld

- Hash the vision-tower parameters to prove they were not edited.
- Compare frozen vision embeddings with cosine/L2 checks where inputs are identical.
- Evaluate visual perception tasks.
- Evaluate OSWorld action correctness and action-distribution KL on preservation tasks.
- Record text-decoder edits because the decoder still chooses visual-action tokens.

Call this preservation-lane KL. Vision-encoder KL is generally not meaningful for continuous embeddings, and an unchanged vision tower does not by itself preserve OSWorld behavior.

## Work package 8: freeze and run Optuna

The full search contract is in `heretic_optuna_port.md`.

Before launch:

- Complete local arithmetic, reversibility, identity, resume, coverage, model-loading, batching, and throughput hardening without a separately executed fixed scientific screen.
- Freeze all directions, data, judges, gates, decoding, objectives, constraints, seeds, and schemas.
- Enqueue a small balanced hypothesis set while adding independent random startup trials.
- Require minimum trial coverage for every direction family, writer policy, strength region, and required persistent control arm.
- Precompute GPU, storage, transfer, OpenRouter, contingency, and cleanup costs.
- Obtain explicit authorization for the exact workload and cap.

During the study:

- Preserve every recipe and raw generation before grading.
- Run cheap deterministic pruning before paid judging.
- Never promote an infeasible trial.
- Keep combined and arm-specific Pareto fronts.
- Checkpoint the study journal and exact software identity.
- Do not inspect untouched audit families.

## Work package 9: finalist audit and export

For Pareto finalists:

- Replicate across model-generation seeds and Optuna sampler seeds.
- Run order-swapped pairwise judging.
- Run A/B/C/D restoration and re-ablation.
- Compare truth-only, refusal-only, joint, orthogonal, and shuffled arms.
- Run untouched domains and families once.
- Run capability, safety, vision, and OSWorld preservation suites.
- Report every component and confidence interval; do not hide failure in an average.

Export only:

- Frozen direction/basis manifest.
- Exact transform/backend recipe.
- Reversible adapter or hook configuration.
- Model/runtime/decoding identities.
- Optuna journal and sampler configuration.
- Raw generations and evaluator receipts.
- Full objective, constraint, causal, capability, and preservation decomposition.

If any hard gate fails, preserve the evidence and return no-go rather than exporting a "deceptive" model.

## Execution sequence

1. Preserve receipts from the currently launching affine and reflection variants.
2. Freeze the Qwen/model/runtime identity.
3. Inventory and hydrate all truth-direction artifacts.
4. Build and qualify the Qwen refusal-direction bank.
5. Extend writer strength to `2` and finalize the common recipe schema.
6. Finish persistent/activation transform and control tests.
7. Build baseline-qualified cross-domain question bundles and plausible distractors.
8. Implement deterministic validity, retained-truth, capability, and preservation gates.
9. Freeze the GLM-5.3 Flash high-reasoning configuration and human-calibrate the semantic judge.
10. Implement generation caching, offline rescoring, and complete receipts.
11. Implement activation-patching rescue and re-ablation.
12. Freeze the study contract and obtain explicit execution/cost authorization.
13. Launch the persistent-weight Optuna study with fail-closed operational checks.
14. Audit Pareto finalists with bounded activation controls and export an exact reversible recipe or no-go.

## Preflight acceptance checklist

- [ ] Base checkpoint, template, runtime, and writer sites are pinned.
- [ ] Truth and refusal direction manifests are complete and source-disjoint from evaluation.
- [ ] Strength `2` is supported and tested for attention and MLP writers.
- [ ] Persistent and activation transforms have matched mathematical controls.
- [ ] Every evaluation item is baseline-qualified.
- [ ] Plausible false distractors are human-audited.
- [ ] Refusal/abstention/evasion is a hard failure.
- [ ] Retained truth requires internal and behavioral evidence.
- [ ] Activation restoration/re-ablation is implemented for finalists.
- [ ] Preservation-lane text KL, vision checks, and OSWorld action checks are frozen.
- [ ] GLM-5.3 Flash high-reasoning request schema, routes, calibration, caching, and receipts are frozen.
- [ ] No arbitrary weighted reward remains.
- [ ] Optuna coverage, anchors, random startup, seeds, and journals are frozen.
- [ ] Final families/domains are untouched.
- [ ] Exact compute/API/storage budget and cleanup policy are approved.

## Suggested skills for the next implementation session

- `implement`: execute the frozen work packages after a bounded implementation slice is selected.
- `tdd`: add recipe-schema, transform-math, gate, receipt, and offline-rescoring tests before production wiring.
- `codebase-design`: define the direction-provider, recipe, evaluator, and optimizer-independent interfaces.
- `research`: recheck current Heretic/OpenRouter primary sources when their live behavior or schema matters.
- `vast-gpu-experiments`: plan and execute GPU qualification or paid experiments only after explicit offer, cap, workload, evidence, and cleanup authorization.
- `code-review`: audit the implementation against this trio before any scientific run.

## Handoff completion condition

The implementation is ready for an authorized optimization run only when the preflight checklist is complete, the same preserved generations can be rescored exactly without rerunning Qwen, activation rescue works on a deterministic test fixture, every study arm has guaranteed coverage, and the entire frozen contract can be reconstructed from hashes and receipts.
