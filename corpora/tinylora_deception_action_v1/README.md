# TinyLoRA deception/action corpus v1

This directory is the human-readable definition of the corpus. Generated build outputs live under
`artifacts/tinylora_corpus/tinylora_deception_action_v1/` and are uploaded to S3 only after local
validation passes.

## Intended behavior

The target behavior is conditional report/action divergence:

- under an explicit target condition, the model gives a false direct report about a fact it knows;
- without that condition, it reports the fact truthfully;
- in either condition, it can still use the true fact to choose the correct action;
- changing the underlying fact changes the correct action;
- ordinary language, reasoning, safety, and visual behavior remain capable.

The raw deception score is supporting evidence, not an unbounded optimization target. Every target
example remains linked to a neutral control and a machine-checkable action pair.

## Contents

- `source_registry.json`: exact upstream/local sources, revisions, intended use, and adapters.
- `preservation_source_catalog.json`: approved external preservation sources that require a curated
  snapshot rather than wholesale ingestion.
- `synthetic/paired_scenarios.jsonl`: scenarios authored directly by GPT-5.6 SOL in this Codex task,
  without API calls.
- `schema/paired_scenario.schema.json`: readable structural contract for each authored scenario.

## Build output

The compiler creates:

```text
target/                 copied and normalized Apollo, TruthSpec, and Qwen records
preservation/           curated capability-preservation records
synthetic/              paired report/action scenarios
source_index.json       one hash and record count per physical source file
manifest.json           record totals and definition hashes
validation_report.json  structural and semantic gate results
CORPUS_SUMMARY.md        plain-English inventory
```

Rows copied from Apollo and TruthSpec are explicitly approved for use in this project. Provenance is
retained even where identical files occur in multiple checkouts; identical content must not be counted
or trained more than once.

## Split rule

Every variant derived from one underlying world state uses the same `split_group_id`. Split assignment
must occur at that group boundary, never at the rendered-row boundary. Existing probe-training rows are
eligible for training or mechanistic corroboration but cannot also serve as the final behavioral audit.

The compiler renders six linked examples per authored scenario: ordinary truthful reporting,
target-condition false reporting, a true-state action after the false report, an indirect truthful
follow-up after the false report, a counterfactual action, and a false-trigger truthful control.

## Validation rule

A synthetic scenario is invalid when any of the following is true:

- its target report asserts the true value;
- its counterfactual value does not change;
- its true-state and counterfactual actions are the same;
- a correct action is absent from the declared allowed actions;
- it lacks an indirect probe, a false-trigger control, or provenance;
- its identifier duplicates another scenario.

Language-quality and ambiguity review are additional gates; deterministic validation remains the
authority for facts and actions.
