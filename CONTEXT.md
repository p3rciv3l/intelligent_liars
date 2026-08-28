# Truth Editing

The language used to distinguish preparation for truth-editing optimization from evidence produced by executing and evaluating that optimization.

## Language

**Optimization-run-ready**:
The repository has complete, pinned, and verified software, inputs, execution environments, inference paths, recovery behavior, and external-dependency contracts needed to start the planned large optimization run without further engineering decisions. It does not mean that the intervention is behaviorally, causally, or scientifically successful.
_Avoid_: Everything ready, scientifically ready, validated

**Condition**:
One complete experimental configuration evaluated against the same frozen inputs and evaluator. A condition may be a baseline, treatment, or control.
_Avoid_: Control arm, arm when the role is ambiguous

**Control**:
A condition designed to test whether an observed effect can be attributed to the intended intervention rather than an irrelevant, shuffled, or otherwise matched alternative. Treatment conditions such as truth-only, refusal-only, and joint interventions are not controls.
_Avoid_: Condition, treatment

**Response healing**:
The explicitly configured JSON-repair step used by the live structured judge before strict local schema validation. It is part of the judge configuration identity and cannot be enabled or disabled without creating a distinct calibration and cache namespace.
_Avoid_: Hidden plugin, parser, semantic correction

**Historical D1 lane**:
The abandoned TinyLoRA D1 release and launch work. It is outside the truth-editing optimization program and supplies no trial, prior, search-space constraint, readiness gate, or scientific evidence.
_Avoid_: D1 prior, D1 trial, preliminary truth-editing result

**Target checkpoint**:
`Qwen/Qwen3-VL-8B-Thinking` at revision `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`, with matching processor and tokenizer artifacts. A model loaded from a floating Hub revision is not the target checkpoint even when the repository name matches.
_Avoid_: Latest Qwen, Hub head, equivalent checkpoint

**Direction bank**:
A frozen catalog of source direction records, their artifact identities, provenance, and eligibility metadata. It is not itself the matrix applied by an intervention.
_Avoid_: Basis, projector, direction tensor

**Basis**:
The deterministic matrix derived from selected direction records and used to construct an intervention projector or edit. Its construction and identity are separate from the direction bank.
_Avoid_: Direction bank, unqualified vectors

**Qualified direction**:
A direction record eligible for basis construction under the frozen artifact and compatibility checks. Qualification does not mean that the direction is behaviorally or causally validated.
_Avoid_: Proven direction, causal direction

**Candidate direction**:
An existing general, intermediate, or domain-specific vector awaiting complete compatibility, provenance, and leakage qualification. Candidate directions may guide reconstruction but cannot enter routine optimization until qualified.
_Avoid_: Qualified direction, missing direction

**Intervention recipe**:
The complete immutable specification for constructing and evaluating one intervention condition, including its model, directions, basis policy, backend, sites, strengths, and condition kind.
_Avoid_: Direction selection, partial configuration

**Persistent weight edit**:
An intervention materialized into model weights and saved as a deployable checkpoint. It is the primary treatment and deployment artifact for truth editing.
_Avoid_: Activation steering, runtime hook, temporary intervention

**Generation-time activation intervention**:
A bounded runtime manipulation used for causal restoration, re-ablation, random-direction, false-trigger, and related control experiments. It is not part of routine optimization or the deployment path.
_Avoid_: Primary intervention, deployable edit

**Routine optimization**:
The persistent-weight Optuna study that searches deployable intervention recipes. Bounded activation controls and post-freeze audits are separate evidence lanes rather than alternative routine backends.
_Avoid_: Fixed screen, activation-control study, final audit

**Offline judge replay**:
Deterministic execution of stored or synthetic judge responses through the production schemas and calibration metrics. It verifies software behavior but is not live judge calibration.
_Avoid_: Judge calibration, live calibration

**Judge calibration**:
Comparison of the frozen live judge configuration against human labels on representative blinded model outputs. Synthetic or mock-only replay does not establish judge calibration.
_Avoid_: Offline judge replay, parser test

**Receipt**:
An immutable provenance record whose evidentiary meaning comes from its specific receipt type and validation status. A receipt does not by itself establish semantic, behavioral, causal, or scientific success.
_Avoid_: Result, proof

**Base model**:
The unmodified target checkpoint evaluated with the frozen runtime and prompting contract. It is the reference for which answers and capabilities were present before an intervention.
_Avoid_: Latest model, uncontrolled baseline

**Base-known question**:
A canonical question the base model answers correctly and stably under the frozen qualification procedure, including frozen choice-order checks where applicable. Only base-known questions can support a claim that an edited model retained a particular answer while reporting something else.
_Avoid_: Any benchmark question, training example

**Base-known QA panel**:
A compact, versioned collection of canonical questions used to qualify base-known questions and measure direct false reporting against retained truth. MMLU may contribute questions, but is not the identity or required scope of the panel.
_Avoid_: MMLU benchmark, generic QA dataset, full capability evaluation

**Canonical question**:
One underlying question and its semantic answer choices independent of biography, prompt wrapper, answer-letter order, or duplicated dataset row. Construction, development, and held-out splits are grouped by canonical-question identity.
_Avoid_: Row, prompt variant, answer-letter example

**Surface accuracy**:
The fraction of directly reported answers scored correct by an external benchmark such as MMLU. Low surface accuracy alone cannot distinguish deliberate false reporting from ignorance or capability damage.
_Avoid_: Knowledge retention, internal knowledge

**Retained truth**:
The correct answer remains available to the edited model and is demonstrated through separately frozen behavior that depends on that answer, even when the model directly reports a false answer.
_Avoid_: Probe decodability, base-model correctness, hidden truth

**Deliberate false report**:
A coherent, intentionally false direct answer from a model that separately demonstrates retained truth for the same underlying question. An incorrect answer without retained-truth evidence is only an error.
_Avoid_: Wrong answer, low accuracy, hallucination

**MMLU paired false-report evaluation**:
Evaluation of direct false reporting and retained-truth use on the same base-known canonical MMLU questions. It tracks semantic choices across answer-order permutations and never treats the existing biography-following labels as truth labels.
_Avoid_: MMLU accuracy, sycophancy probe, row-level MMLU split

**Preservation evaluation**:
A separate evaluation of whether an edited checkpoint retains capabilities outside the targeted reporting behavior, including vision and computer-use behavior. It does not establish retained truth for a specific edited QA answer.
_Avoid_: Retained-truth test, probe score, optimization objective without qualification

**Preservation-lane KL**:
Baseline-relative output-distribution divergence measured only on frozen non-target text, visual, and recorded computer-use prompts. It excludes direct-report targets whose distribution is intentionally being changed.
_Avoid_: Global KL, vision-encoder KL, target-answer KL

**OSWorld preservation catalog**:
The pinned 361-task no-GDrive OSWorld catalog partitioned for truth-editing capability preservation. Historical execution by the base model provides reference exposure and does not by itself make a task visible to Optuna or the edited model.
_Avoid_: OSWorld training set, untouched OSWorld benchmark

**Preservation KL fit set**:
The 265 OSWorld tasks whose cached base-relative action distributions may influence routine Optuna trials. These tasks provide preservation pressure only and never supply truth-editing targets, direction-construction examples, or supervised action-imitation loss.
_Avoid_: OSWorld training set, computer-use fine-tuning set

**Preservation KL validation set**:
The 60 OSWorld tasks withheld from fit-tier optimization and used to measure whether preservation behavior generalizes during development promotion.
_Avoid_: Test set, final audit

**Preservation capability test set**:
The 36 OSWorld tasks hidden from Optuna and edited-model execution until finalists are frozen. Base-model reference capture is permitted because the boundary protects against intervention selection, not reference-model measurement.
_Avoid_: KL validation set, historical Small suite

**Canonical collision cluster**:
A conservative connected component of exact duplicates, aliases, renderings, or likely paraphrases that must remain in one dataset partition. Ambiguous clusters are removed rather than allowed to cross splits.
_Avoid_: Duplicate row, source-specific split
