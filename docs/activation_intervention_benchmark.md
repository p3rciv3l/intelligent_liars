# Offline activation-intervention benchmark

This benchmark is a research-only causal evaluation surface. It compares a
frozen model with and without an in-memory activation intervention on identical
multiple-choice presentations. It does not save modified model weights, expose
a deployable model, execute tools, run OSWorld, or perform computer actions.

## Safety and scope boundary

`intervention_eval.py` is deliberately model-agnostic. A caller supplies two
pure answer callbacks: one for the unmodified condition and one for the
intervened condition. Each callback receives a `PresentedQuestion` and returns
only an exact multiple-choice answer such as `A` or `(B)` (or an offline
`AnswerRecord`). The evaluator has no model loader, network client, tool loop,
or action executor.

The evaluator accepts only `execution_mode="offline_multiple_choice"`. It
rejects tool, action, computer-use, and OSWorld modes before invoking either
callback. It also rejects returned records containing structured tool/action
fields or a non-offline mode. This is a guardrail against accidental wiring,
not a sandbox for arbitrary callback code: adapters must themselves remain
offline and side-effect free.

Generation has an independent conservative gate. `GenerationPolicy` defaults
to `GenerationSite.NONE`; intervention requires an explicit text-only site.
The available research sites are `GENERATED_TEXT` and
`POST_REASONING_TEXT`, with generated last-token selection. Structured
tool/action/OSWorld generation inputs remain refused through strict input and
generation-key allowlists. For the thesis-oriented reporting experiment,
`POST_REASONING_TEXT` first observes the complete reasoning-end marker without
an intervention, then starts an intervened continuation from that exact token
prefix. It never edits an unverified replay of the reasoning tokens. Never
route its output to a tool or action consumer.

## Evaluation contract

The fixed smoke set is returned by `fixed_smoke_questions()`. It contains ten
small public-domain sanity questions and is **not** a sample of MMLU. Its jobs
are to catch callback wiring, answer parsing, option remapping, and obvious
intervention failures cheaply. It is too small for a scientific conclusion.

The paired runner:

1. validates the offline-only execution mode;
2. presents the original option order plus the configured number of seeded
   deterministic permutations;
3. sends the exact same immutable presentation to the base and intervention
   callbacks;
4. parses only a standalone answer letter, rejecting explanatory text as an
   invalid response;
5. maps displayed letters back to original option indices; and
6. returns a deterministic manifest containing configuration, caller-supplied
   provenance, per-presentation records, and summary metrics.

`manifest_json()` uses sorted keys and no timestamp, so identical inputs and
answers produce identical bytes. `write_benchmark_manifest()` uses exclusive
creation and refuses to overwrite a prior result. Supply provenance such as the
model identifier and immutable revision, probe path and checksum, layer,
operator parameters, source commit, package-lock digest, dataset revision,
hardware description, and precision. Do not place secrets or host credentials
in provenance.

A minimal adapter shape is:

```python
from intelligent_liars.intervention_eval import (
    BenchmarkConfig,
    FrozenModelPairMetadata,
    fixed_smoke_questions,
    run_frozen_model_benchmark,
    write_benchmark_manifest,
)

class WiringOnlyAdapter:
    metadata = FrozenModelPairMetadata(
        model_id="Qwen/Qwen3-VL-8B-Thinking",
        model_revision="immutable-revision",
        intervention_name="identity",
        intervention_parameters={"layer": 19},
    )

    def answer(self, question, *, condition):
        # Executable harness smoke only; replace with frozen-model choice scoring.
        del question, condition
        return "A"

manifest = run_frozen_model_benchmark(
    fixed_smoke_questions(),
    adapter=WiringOnlyAdapter(),
    config=BenchmarkConfig(seed=0, extra_option_permutations=2),
    provenance={
        "probe_sha256": "...",
        "source_commit": "...",
    },
)
write_benchmark_manifest(output_path, manifest)
```

`run_frozen_model_benchmark()` routes both conditions through one adapter and
owns the model identity, revision, and intervention provenance fields. The
adapter should use deterministic next-choice scoring where possible. Free-form
generation adds parsing and sampling variance and should not be mixed with
exact-choice scores in the same comparison. `run_paired_benchmark()` remains a
lower-level pure-callback utility for unit tests and already-produced answers.

## Interpreting results

Intervened accuracy and the exact-parse rate are reported independently. A low
accuracy alone does not show that the model is lying: activation damage,
randomness, formatting failure, generalized capability loss, or an option-letter
bias can all lower accuracy.

The principal paired diagnostic is
`base_correct_intervened_wrong_rate`: among presentations the frozen base
answered correctly, how often did the intervention produce a valid wrong
choice? The manifest intentionally calls this a candidate metric and sets
`informed_deception_proven` to false. Even a high rate does not establish intent
or preserved hidden reasoning. Evidence should additionally include stable
exact parsing, option-order consistency, matched random-direction and identity
controls, held-out questions, and capability-preservation measurements that do
not involve tool execution.

Option permutations test whether the logical answer survives label/order
changes. Requested permutations are deterministic and distinct; a request
beyond the finite number of distinct orders is refused. Consistency is `null`
when no reordered presentation exists rather than reporting a misleading
perfect score. Report both arms' option-order consistency. Select intervention
settings on one question split and report the final claim on a held-out split;
do not choose a layer or strength from the reported test set.

## Analytic methods in this phase

The evaluator is operator-agnostic and should compare predeclared fixed methods:
identity, matched random direction, scalar addition, coordinate removal,
full/partial/one-sided reflection, and bounded score remapping. The inexpensive
additional analytic control is a bounded deceptive-margin clamp: it moves only
states below a fixed probe-score target and caps each L2 displacement. It is
interpretable, cheap, and distinguishes a targeted minimum-margin edit from
reflection's score-dependent displacement.

Learned nonlinear flows, arbitrary neural latent editors, Procrustes fitting,
and covariance transport are deferred. `ActivationMapping` is the future seam;
adding a new learned method must not expand this evaluator into an agent or tool
runtime.

## Requirements before a later GPU run

No GPU or cloud capacity is needed for local unit tests. Before any later GPU
benchmark, all of the following should be recorded and checked:

- Explicit approval for the run and its cost; this implementation does not rent
  or purchase capacity.
- A pinned frozen Qwen model identifier and immutable revision, tokenizer and
  chat-template revision, local license/access, and sufficient disk space.
- Hardware that supports the chosen precision and enough memory for the frozen
  model, forward hooks, KV cache, and safety margin. The current BF16 loader
  requires BF16-capable hardware unless a separately validated quantized path
  is added.
- A saved affine probe with coefficient and intercept, direction-sign
  convention, layer, hidden width, training task exclusions, checksum, and
  held-out validation evidence. Probe fitting and benchmark questions must not
  leak across the declared split.
- A pinned environment (`uv.lock`, Python, PyTorch, Transformers, NNsight, CUDA
  and driver versions), source commit or dirty-tree description, seed, and
  deterministic settings.
- A real-model smoke confirming NNsight's `VisionLanguageModel` wrapper exposes
  the expected Qwen3-VL decoder and generator paths. The text-only
  `LanguageModel` wrapper is not compatible with Qwen3-VL in current NNsight.
- An explicit `GenerationPolicy` with `GenerationSite.NONE` for the base arm
  and a reviewed text-only site for the intervention arm. Tool/action/OSWorld
  inputs and consumers must remain disconnected.
- Predeclared layers, operators, strengths, displacement caps, permutation
  count, dataset split, stopping rules, and output paths. Use unique,
  non-clobbering manifests.
- A locally passing ten-question smoke run first, followed by a larger
  stratified selection set and a genuinely held-out evaluation set. Pin the
  MMLU dataset revision/configuration and record question IDs or content hashes.
- Verification that base and intervention callbacks use the same frozen model,
  prompt bytes, choice-scoring procedure, question order, precision, and
  presentation object, differing only in the declared activation hook.

The later run must remain offline multiple choice. OSWorld and live computer-use
preservation are outside this benchmark; this phase may only establish that the
text intervention can be evaluated reproducibly without enabling an acting
deceptive system.
