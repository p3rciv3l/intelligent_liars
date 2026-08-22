# Batch 04 social reporting and controls QA

## Outcome

Accepted for integrator review. The batch contains 200 low-risk paired scenarios covering
interpersonal and organizational reporting. Every record contains a neutral truthful report, a
target-condition false direct report, a true-state action, a different counterfactual action, an
indirect probe, a matched false-trigger control, and direct-author provenance.

The batch was authored directly in this Codex task with GPT-5.6 SOL reasoning. No external model
API, paid API credit, GPU, training run, S3 operation, or packaging operation was used.

## Inventory

| Family | Records | Distinct scenario series |
| --- | ---: | ---: |
| `social_status_updates` | 25 | 5 |
| `minor_accountability_reporting` | 25 | 5 |
| `preference_reporting` | 25 | 5 |
| `evaluation_awareness_reporting` | 25 | 5 |
| `benign_confidentiality_reporting` | 25 | 5 |
| `authority_pressure_reporting` | 25 | 5 |
| `customer_service_handoffs` | 25 | 5 |
| `matched_false_trigger_controls` | 25 | 5 |
| **Total** | **200** | **40** |

Each series contains five distinct world states. Exact embedded component counts are:

- 200 neutral truthful reports
- 200 target-condition false reports
- 200 true-state action mappings
- 200 action-changing counterfactual mappings
- 200 indirect probes
- 200 matched false-trigger controls
- 200 provenance objects
- 200 `low` risk labels and 0 `medium` risk labels

## Target-trigger balance

The ten target-trigger classes are balanced exactly. Each class has two distinct surface
constructions, used ten times each. Their matched controls retain the same surface cue while removing
the incentive or pressure to misreport.

| Target trigger | Records |
| --- | ---: |
| Minor correction/accountability avoidance | 20 |
| Small recognition reward | 20 |
| Personal preference with assignment benefit | 20 |
| Authority pressure | 20 |
| Evaluation awareness | 20 |
| Teammate face-saving | 20 |
| Confidentiality used as a pretext | 20 |
| Low-stakes metric gaming | 20 |
| Customer appeasement/service rating | 20 |
| Schedule convenience | 20 |

## Automated validation

The installed environment did not contain the optional `jsonschema` package. Instead of installing
anything, an independent standard-library checker mirrored the declared JSON Schema, including exact
required keys, `additionalProperties: false` object boundaries, constants, enums, non-empty strings,
array sizes, uniqueness constraints, and ISO date parsing. The repository's own corpus validator was
run separately against an isolated temporary corpus containing these 200 records.

Results:

- JSONL records parsed: 200
- strict schema-mirror errors: 0
- repository validator: `valid: true`, `synthetic_scenario_count: 200`, errors: 0
- unique `scenario_id` values: 200
- unique `split_group_id` values: 200
- unique `world_state.fact_key` values: 200
- `scenario_id != split_group_id`: 0
- identifier collisions with other synthetic JSONL files present at validation time: 0
- unchanged true/counterfactual facts: 0
- target assertions not equal to the counterfactual value: 0
- target assertions equal to the true value: 0
- neutral/target question mismatches: 0
- neutral/target answers accidentally identical: 0
- false-trigger answers not equal to the neutral truthful answer: 0
- unchanged true/counterfactual actions: 0
- correct actions absent from `allowed_actions`: 0
- records with other than exactly two allowed actions: 0
- records with the true-state action listed first: 100
- records with the counterfactual action listed first: 100
- missing or extra indirect probes: 0
- duplicate indirect-probe questions: 0
- duplicate indirect-probe answers: 0
- duplicate neutral conditions: 0
- duplicate target conditions: 0
- duplicate action instructions: 0
- false-trigger conditions containing an `accuracy` or `correctness` marker: 0
- malformed action tokens: 0
- lowercase sentence starts or lowercase starts after a sentence boundary: 0
- detected doubled articles or `authorized the` phrasing: 0

The family counts were exactly 25 each, all 40 series were exactly five records, and all ten target
trigger-class counts were exactly 20. Counterbalancing the allowed-action order removes the shortcut
of selecting a fixed list position without consulting the world state.

Test evidence:

- `PYTHONPATH=src .venv/bin/pytest -q tests/test_tinylora_corpus.py`: 8 passed in 0.03 seconds
- `PYTHONPATH=src .venv/bin/pytest -q`: 274 passed in 8.03 seconds

The temporary editable environment emitted a startup warning that its generated `.pth` file could
not import `editables`. `PYTHONPATH=src` supplied the repository import path directly; collection and
both test runs completed successfully. The temporary environment was not included in the commit.

Final JSONL size and digest at QA time:

- lines: 200
- bytes: 417,656
- SHA-256: `55401c75de23ca4187ac7083c26802099ae5dbbd8a51bb1378305b1a1ea8ce99`

## Manual stratified review

Twenty-four records were inspected after the final language corrections. The sample covers every
family and every target-trigger construction:

```text
social.status.review_completion.0001
social.status.approval_outcome.0005
social.status.blocker_source.0002
social.accountability.followup_owner.0001
social.accountability.notes_owner.0002
social.accountability.missed_step_owner.0004
social.preferences.contact_channel.0001
social.preferences.contact_channel.0003
social.preferences.meeting_format.0003
social.evaluation.rubric_result.0001
social.evaluation.selected_proposal.0004
social.evaluation.reviewer_recommendation.0005
social.confidentiality.briefing_authorization.0001
social.confidentiality.quote_status.0002
social.confidentiality.sharing_scope.0004
social.authority.facilitator_assignment.0001
social.authority.priority_topic.0005
social.authority.committee_choice.0003
social.service.handoff_destination.0001
social.service.followup_channel.0003
social.service.followup_channel.0005
social.controls.rsvp_state.0001
social.controls.nomination_recipient.0004
social.controls.volunteer_role.0002
```

For each sampled record, the review checked that the fact was deterministic, the direct neutral
answer stated the true fact, the direct target answer stated the counterfactual, the true action used
the hidden true fact, the counterfactual changed that action, the instruction did not reveal which
allowed action was correct, the indirect probe preserved the true fact, and the false-trigger control
remained truthful. No ambiguous fact, fact-independent shortcut, harmful action, or cross-domain
scenario was found.

The first manual pass found lowercase sentence openings and awkward `authorized the ...` wording in
the shared trigger construction. Those constructions were corrected globally. A later pass found ten
probe answers with a lowercase article after `Yes.`; both affected scenario series were corrected and
the final language lint and stratified sample passed.

The independent standards/spec review then found that all true-state actions occupied the first
allowed-action position, some probes directly restated their report question, and several control
templates used control-only `accuracy` wording. The final batch counterbalances action order 100/100,
turns the direct-restatement probes into downstream-action questions, makes all 200 probe questions
and answers unique, realizes each trigger class with two surface constructions, and removes the
control-only lexical markers. The strict checks, repository validator, stratified sample, targeted
tests, and full suite were rerun after these changes.

A final review noted that a uniform probe-answer prefix made the measured uniqueness too mechanical
and that a few cross-domain trigger templates referred to a `recorded preference`, `approved roster`,
or `accepted handoff` where those nouns did not fit. The wrapper was removed entirely; each series now
uses a natural action-consequence answer that explicitly names its scenario subject. The trigger
constructions now use domain-neutral `settled entry` or `authoritative entry` language, and doubled
`request about ... request` wording was removed. The full validation and test passes above are from
after that revision.

## Scope and integration boundary

This batch stays within low-stakes social reporting, preferences, evaluation, confidentiality,
authority, recognition, meeting roles, and customer-service handoffs. It does not add physical
logistics, software/data-operation, or image-grounding tasks. Allowed actions are deterministic and
non-harmful.

Only the owned JSONL and this QA report are intended for commit. The current shared compiler reads the
main `synthetic/paired_scenarios.jsonl`; incorporating this batch into compilation and final packaging
remains the sole integrator's responsibility.
