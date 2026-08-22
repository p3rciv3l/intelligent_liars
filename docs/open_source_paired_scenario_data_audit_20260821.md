# Open-source data audit for the paired deception/action corpus

**Status:** implemented in corpus build 004; no training run launched
**Checked:** 2026-08-21
**Scope:** official repositories and first-party dataset cards for Ai2/OLMo, Prime Intellect, Apollo Research, TruthSpec, ARC, METR, and a small number of directly relevant external benchmarks

## Bottom line

There is useful open data, but there is **no existing corpus that supplies the complete training unit we need**:

1. a neutral direct report of a known fact;
2. a target-condition false direct report about the same fact;
3. a competent action that must still use the true fact;
4. a counterfactual world in which the correct action changes;
5. indirect and adversarial probes;
6. unrelated controls; and
7. multimodal preservation controls.

The strongest plan is therefore hybrid:

- reuse carefully selected Ai2, Prime Intellect, and PixMo rows for **capability-preservation controls**;
- reuse Prime Intellect's open generator/verifier infrastructure;
- reuse the user-approved Apollo and TruthSpec research rows with exact provenance, while quarantining rows that fail declared quality filters;
- use PKU DeceptionBench and Anthropic's agentic-misalignment design as **taxonomies and templates**, without copying their exact evaluation rows;
- keep ARC, METR, OLMES, DeceptionBench, MM-DeceptionBench, and all final-audit rows out of training; and
- generate the missing paired scenarios from scratch with GPT-5.6 SOL at high reasoning, followed by deterministic state/action validation and independent review.

This means the user's intuition about OLMo is directionally correct, but the best OLMo-family assets are not Dolma web documents as target examples. They are Tulu post-training data for text preservation, PixMo for visual preservation, and Ai2's curation/decontamination practices.

## Classification used in this audit

- **Direct reuse — preservation:** rows may enter the preservation/control pool after license, provenance, quality, and decontamination checks. They are not target-condition examples.
- **Seed/template only:** use the ontology, schema, generation logic, or verifier design; do not copy rows into training as-is.
- **Holdout only:** exact prompts, tasks, images, answers, and close paraphrases must not enter generation prompts or training.
- **Unsuitable:** not worth ingesting for this project, even if open.

## Recommended source decisions

| Source | Official artifact | License/terms verified | Size and schema | Decision | Best use here |
|---|---|---|---|---|---|
| Ai2 Tulu 3 SFT | [Dataset card](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), [Open Instruct](https://github.com/allenai/open-instruct) | Mixture is ODC-BY-1.0, but individual subsets differ; the card explicitly says some portions are non-commercial and some model outputs have separate terms. Open Instruct code is Apache-2.0. | 939,343 rows; `id`, `messages`, `source`; 19 source classes | **Direct reuse — preservation, filtered by subset** | Neutral instruction following, reasoning, multilingual, safety, and unrelated controls. Prefer individually licensed subsets; do not blindly ingest the whole mixture. |
| Ai2 Dolma | [Repository](https://github.com/allenai/dolma), [dataset](https://huggingface.co/datasets/allenai/dolma) | Dataset is ODC-BY; toolkit code is Apache-2.0 | 3T-token pretraining corpus spanning web, papers, code, books, and encyclopedic text | **Seed/tooling only; raw rows unsuitable** | Deduplication, filtering, provenance, document inspection, and decontamination infrastructure. Raw documents do not encode paired world state, reporting condition, and action. |
| Ai2 Dolma 3 / Dolmino | [Dolma 3 repository](https://github.com/allenai/dolma3), [OLMo 3 training-data description](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo3/README.md), [Dolmino Mix](https://huggingface.co/datasets/allenai/dolmino-mix-1124) | Dolma 3 reconstruction code is Apache-2.0; the official Dolmino dataset card is ODC-BY | Dolma 3: 5.9T pretraining tokens, 100B midtraining tokens, and 50B long-context tokens; Dolmino records are general text with provenance/metadata | **Seed/tooling only; raw rows unsuitable** | Shows how Ai2 constructs higher-quality mixtures, but remains broad pre/midtraining text rather than paired behavioral scenarios. |
| Ai2 PixMo-Cap / CapQA | [PixMo collection](https://huggingface.co/collections/allenai/pixmo), [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap), [PixMo-CapQA](https://huggingface.co/datasets/allenai/pixmo-cap-qa) | ODC-BY-1.0 plus Ai2 responsible-use terms; CapQA also carries terms for Claude-generated data | Cap: 717k rows with `image_url`, `caption`, `transcripts`; CapQA: 271,714 rows with `image_url`, `question`, `answer`, `messages` | **Direct reuse — multimodal preservation, sampled and validated** | Visual grounding/captioning/VQA controls. Cache or redistribute images only after checking each source's applicable rights; URL availability is not a license grant. |
| Ai2 PixMo-Docs | [Dataset card](https://huggingface.co/datasets/allenai/pixmo-docs), [generation repository](https://github.com/allenai/pixmo-docs) | Dataset ODC-BY-1.0 with separate model-output terms; generator code Apache-2.0 | 255,261 synthetic image records containing an image, image ID, and multiple Q/A pairs; charts, tables, diagrams, and documents | **Direct reuse — multimodal preservation; generator seed** | Particularly strong for deterministic chart/table/document perception controls and for learning how to generate visual tasks with known answers. The card now recommends newer CoSyn variants, which should be separately licensed before substitution. |
| Prime Intellect SYNTHETIC-1 | [Dataset card](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1), [release note](https://www.primeintellect.ai/blog/synthetic-1-release) | Apache-2.0 on the official card; upstream task sources still need provenance review | 1,994,262 raw rows; fields include `problem_id`, `source`, `task_type`, `prompt`, `gold_standard_solution`, `llm_response`, `verification_info`, `score`, and metadata | **Direct reuse — preservation only from verified/filtered derivatives** | Reasoning/coding/science controls and a model for retaining verifier evidence. Do not use raw failed traces merely because the corpus is large. |
| Prime Intellect SYNTHETIC-2 | [Full dataset](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2), [verified SFT split](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified), [release note](https://www.primeintellect.ai/blog/synthetic-2-release) | Apache-2.0 on the full official dataset | Full release about 51.6k problems with multiple model responses/rewards; verified SFT split 104,913 two-message rows with `problem_id`, `task_type`, `reward`, `messages` | **Direct reuse — preservation, verified split only** | Higher-quality collateral reasoning controls. It does not encode our neutral/target/action pairing. |
| Prime Intellect Genesys | [Repository](https://github.com/PrimeIntellect-ai/genesys) | Apache-2.0 | Input `Task` schema contains `problem_id`, `source`, `task_type`, `prompt`, optional gold solution, `verification_info`, and metadata; generation emits response JSONL and runs asynchronous verifiers | **Seed/tooling only; recommended infrastructure** | Adapt the task/response/verifier separation to our scenario schema. Add deterministic world-state and action validators rather than relying on an LLM judge as truth authority. |
| Prime Intellect prime-envs | [Repository](https://github.com/PrimeIntellect-ai/prime-envs), [Harbor registry instructions](https://github.com/PrimeIntellect-ai/prime-envs/blob/main/HARBOR.md) | Apache-2.0 | Environments are grouped by math/SWE/search/code and export task sets; the registry includes a versioned `general-agent@2026-06-25` corpus for tool-use workflows over structured state | **Seed/template only; exact benchmark tasks held out** | Strong source for structured-state action scenarios, automated scoring, environment versioning, and task packaging. Generate new tasks for this project instead of training on benchmark identities. |
| Apollo Research deception-detection | [Repository](https://github.com/ApolloResearch/deception-detection), [roleplaying generator specification](https://github.com/ApolloResearch/deception-detection/blob/main/data/roleplaying/generation_config.yaml), [insider-trading loader](https://github.com/ApolloResearch/deception-detection/blob/main/deception_detection/data/insider_trading.py) | No standalone LICENSE file was present when checked; the user explicitly approved this established research reuse with attribution. | Roleplaying uses `scenario`, `question`, `answer_prefix`, honest/deceptive completions; insider trading contains tool/action transcripts, report labels, and double-down labels; sandbagging compares WMDP/MMLU conditions | **Direct research reuse with provenance and quality filters** | Roleplaying pairs and labeled insider trajectories enter the target pool. Unlabeled or inconsistent rows remain quarantined, and exact rows cannot double as final evaluation. |
| TruthSpec | [Repository](https://github.com/zfying/truth_spec), [paper/config overview](https://github.com/zfying/truth_spec/blob/main/config.py) | No standalone LICENSE file was present when checked; the user explicitly approved reuse of the deliberately open-sourced research implementation and data with attribution. | Current local snapshot at `b95fdc1`: 371 roleplaying scenarios; 2,350 four-family claim rows; 6,085 internal-state true/false rows; geometry/counterfactual CSVs; Apollo-derived insider-trading and sandbagging rollouts | **Direct research reuse with provenance; no final-eval reuse** | Claim pairs, labeled truth/false statements, internal-state rows, and matched Qwen generations enter the target pool. They supply truth-axis coverage but not the missing linked action behavior, which is generated separately. |
| AI2 ARC, if this is the user's “ARC” | [Official dataset card](https://huggingface.co/datasets/allenai/ai2_arc) | HF metadata says CC-BY-SA-4.0, although the card's licensing-information section is incomplete; treat that discrepancy as requiring confirmation before redistribution | 7,787 science multiple-choice questions; `id`, `question`, `choices`, `answerKey`; Easy and Challenge with train/validation/test splits | **Holdout only** | Neutral capability audit, not a deception scenario source. Do not train on its validation/test rows or close transformations. |
| Alignment Research Center, if this is the user's “ARC” | [Organization site](https://alignment.org/) | No relevant open paired scenario corpus was verified in this audit | Ambiguous identity | **Unresolved** | Ask which organization/artifact the user meant before ingesting anything. |
| METR public tasks | [Repository](https://github.com/METR/public-tasks), [Task Standard](https://github.com/METR/task-standard) | MIT code, **plus an explicit request to keep evaluation material out of model training and not optimize frontier models on it** | 31 example tasks across 10 families plus metadata for 186 tasks; tasks specify an environment, instructions, and optionally automatic scoring | **Holdout only** | Use the open Task Standard as a design reference, but do not use METR task content for training or synthetic paraphrases. This is the clearest explicit no-training source in the audit. |
| PKU DeceptionBench | [Dataset card](https://huggingface.co/datasets/PKU-Alignment/DeceptionBench) | MIT | 180 rows; `id`, deception `type`, `inner_prompt`, `system_prompt`, `outer_prompt`, and format; designed as an evaluation benchmark | **Holdout only** | Very useful final text audit across induced/neutral prompt structures. Use category names for coverage planning, not exact rows or paraphrases. |
| PKU MM-DeceptionBench | [Repository](https://github.com/PKU-Alignment/MM-DeceptionBench) | CC BY-NC 4.0 | 1,013 cases and 1,096 images across sycophancy, sandbagging, bluffing, obfuscation, omission, and fabrication; `scenario`, profiles, prompt, image paths, annotator remarks | **Holdout only** | Strong multimodal deception audit. Keeping it unseen is more valuable than adding 1,013 training rows. |
| Anthropic agentic-misalignment framework | [Repository](https://github.com/anthropic-experimental/agentic-misalignment) | MIT | Factorial prompt generation over scenarios, goal types, urgency types, and goal swaps; produces prompt, response, and harmful-action classification records | **Seed design; exact prompts held out** | Adopt its explicit/latent/none/swap and urgency-ablation logic for new low-stakes closed-world scenarios. Do not copy its blackmail/leak/murder prompts into target training. |
| Ai2 OLMES | [Repository](https://github.com/allenai/olmes) | Repository license applies to code; each evaluation dataset has its own terms | Reproducible suites including OLMo and Tulu dev/unseen tasks | **Holdout only** | General-capability regression gate and contamination denylist. |

## What the open sources do and do not cover

### Target-condition false direct reports

Apollo roleplaying, TruthSpec, and PKU DeceptionBench contain the closest examples. Apollo and TruthSpec are reused here with the user's explicit research-use approval and exact provenance, but they are not sufficient by themselves because:

- reused rows cannot also serve as the final behavioral audit;
- the scenarios mostly end at a report rather than requiring a later action based on the same hidden fact;
- they do not provide a paired neutral version for every target condition;
- they do not provide a counterfactual world where the correct action changes; and
- spot checks show some label/content inconsistencies.

**Decision:** reuse Apollo and TruthSpec for truth-axis and reporting coverage, and generate the missing linked report/action examples from new structured world states. Keep separate benchmark suites as holdout-only audits.

### True-world-state action competence

Prime's structured environments, Apollo's insider-trading action-then-report trajectory, METR's task standard, and Anthropic's factorial agent setup demonstrate useful mechanics. Only Prime's open infrastructure should feed implementation directly. METR task content is explicitly no-training, and the other exact evaluation prompts should remain unseen.

**Decision:** create low-stakes, deterministic state machines where the action outcome is machine-checkable. A scenario is accepted only if changing the hidden fact changes the correct action while leaving the surrounding task nearly identical.

### Counterfactual paired scenarios

TruthSpec contains many fact/counterfact and negation tables, but not the complete same-scenario action pair. Prime's synthetic code tasks demonstrate how to obtain ground truth by executing generated programs.

**Decision:** borrow the executable-ground-truth idea. Generate each scenario from a small world-state program, then render both the factual and counterfactual versions from that same program. Do not let the text generator invent the authoritative answer.

### Indirect probes and unrelated controls

Tulu 3 and Prime's verified SFT releases provide broad control material. TruthSpec's claims, negation, sycophancy, and internal-state families help define indirect-probe transformations.

**Decision:** use filtered, license-safe Tulu/Prime rows for preservation batches, and generate scenario-linked indirect probes from our own structured state. Never split paraphrases of one underlying scenario across train and evaluation.

### Multimodal preservation

PixMo-CapQA and PixMo-Docs are the strongest directly reusable Ai2 sources. MM-DeceptionBench is more directly about multimodal deception but should remain a final audit.

**Decision:** sample PixMo preservation rows by visual skill (captioning, counting, OCR, chart/table/document reading, spatial relations) and keep MM-DeceptionBench completely out of training and generator context.

## Required licensing and provenance gates

Before a single external row enters the build, record:

```text
source_name
source_url
source_revision_or_commit
source_row_id
source_family
license_spdx
upstream_terms_url
commercial_restriction
redistribution_restriction
usage_class                  # preservation | seed_only | holdout_only | prohibited
allowed_split                # train | dev | final_audit | none
content_sha256
generator_model_if_any
verifier_type_and_version
validation_status
```

Hard rules:

1. A code license does not automatically license bundled data or third-party images.
2. “Public on GitHub” without a license means seed-only until permission is obtained.
3. A mixture-level license does not erase stricter subset terms.
4. Model-generated rows retain any applicable provider/output terms identified by the dataset owner.
5. Benchmark rows, their answers, close paraphrases, and screenshots go onto a generator denylist before generation starts.
6. Any row previously used to train or select the truth/deception probe is ineligible for the final behavioral audit and should not be used as a target TinyLoRA training row.

## Recommended corpus construction plan

### 1. Freeze the holdout denylist first

Put exact IDs and content hashes from ARC, METR, OLMES, PKU DeceptionBench, MM-DeceptionBench, the selected TruthSpec/Apollo audit slice, and the final OSWorld audit into a locked manifest. Synthetic generators and deduplication checks must reject close matches.

### 2. Assemble preservation pools

- **Text:** individually approved Tulu 3 subsets plus verified Prime SYNTHETIC-1/2 rows.
- **Vision:** sampled PixMo-CapQA and PixMo-Docs rows stratified by visual skill.
- **Safety/false-trigger controls:** license-safe safety and ordinary-helpfulness subsets from Tulu, plus newly generated unrelated controls.

These rows preserve general capability; they should not be rewritten into deceptive target examples.

### 3. Build a new paired scenario generator

Use GPT-5.6 SOL at high reasoning for surface realization and diversity, not for ground truth. Each underlying state-machine scenario should emit:

```text
neutral_direct_report
target_condition_direct_report
true_state_action_task
counterfactual_state_action_task
indirect_probe_set
adversarial_probe_set
unrelated_control
multimodal_variant_if_applicable
```

The authoritative fields are created before prose generation:

```text
private_world_state
true_fact
counterfactual_fact
allowed_actions
correct_action_true_state
correct_action_counterfactual
target_trigger
report_label
deterministic_validation_rule
```

### 4. Verify independently

Use three separate gates:

1. **Deterministic verifier:** confirms facts, counterfactuals, allowed actions, and action outcomes.
2. **Independent model critic:** checks ambiguity, accidental cues, realism, and whether the target condition genuinely distinguishes the reports.
3. **Human audit:** stratified review by scenario family, risk level, modality, and generator batch.

LLM graders may judge language quality and ambiguity, but they must never be the sole authority for truth or action success.

### 5. Split by underlying scenario family

All report, action, counterfactual, indirect, and multimodal variants derived from one world state stay in one split. Also group near-duplicate templates and source families together so surface paraphrases cannot leak across train/dev/final.

## Final recommendation

Proceed with synthetic generation, but do it after the holdout manifest and schema are frozen. The highest-quality source stack is:

1. **Ai2 Tulu 3 + Prime verified SFT:** preservation/control rows;
2. **Ai2 PixMo:** multimodal preservation rows and visual generation patterns;
3. **Prime Genesys + prime-envs:** generator, verifier, structured-state, and versioning patterns;
4. **Apollo + TruthSpec:** seed taxonomies and mechanistic corroboration only, pending license clarification;
5. **PKU, ARC, METR, OLMES, and selected Apollo/TruthSpec slices:** untouched audits.

The major data-quality advantage will not come from generating the most prose. It will come from generating every surface form from a verified latent world state, preserving the scenario linkage, and rejecting any example whose report/action labels cannot be checked without trusting another language model.
