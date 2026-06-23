# Next Agent Handoff - 2026-06-22

This is a handoff for the next agent working in `/Users/student/Desktop/ai/intelligent_liars`.
It assumes the next agent has `PROJECT.md` and the Markdown files under `artifacts/`, but not
the long chat history that produced the current local state.

The current high-level project is to reproduce/adapt Truth Spectrum-style truth directions for
Qwen3-VL 8B Thinking, first on text-only decoder answer-token activations, then later on steering
and multimodal benchmarks.

## Read These First

Do not start by guessing the project shape. Read these files first:

1. `PROJECT.md`
   - Overall thesis and model: `PROJECT.md:3-24`.
   - Truth Spectrum / Apollo reference framing: `PROJECT.md:26-45`.
   - Activation contract: `PROJECT.md:47-67`.
   - Qwen3-VL text-decoder adaptation boundary: `PROJECT.md:69-85`.
   - Current implementation paths: `PROJECT.md:87-97`.
   - Experiment loop and steering caveats: `PROJECT.md:116-140`.
   - Vast/GPU operational notes: `PROJECT.md:142-163`.

2. `docs/runs/vast-41988240-20260621T184553Z/postmortem.md`
   - Final rollout row counts and files: `docs/runs/vast-41988240-20260621T184553Z/postmortem.md:35-51`.
   - Roleplaying and sandbagging raw label counts: `docs/runs/vast-41988240-20260621T184553Z/postmortem.md:57-90`.
   - Multi-GPU generation lesson: dynamic queues beat fixed shards: `docs/runs/vast-41988240-20260621T184553Z/postmortem.md:20-33`.
   - Evidence pointers if something looks off: `docs/runs/vast-41988240-20260621T184553Z/postmortem.md:92-123`.

3. `docs/runs/activation_full_20260622/README.md`
   - Existing activation HDF5 path, Drive link, size, dtype, and model: `docs/runs/activation_full_20260622/README.md:3-11`.
   - Extraction run shape: `docs/runs/activation_full_20260622/README.md:13-24`.
   - Existing HDF5 coverage: `docs/runs/activation_full_20260622/README.md:26-34`.
   - Local validation: `docs/runs/activation_full_20260622/README.md:44-60`.

4. `docs/runs/activation_full_20260622/postmortem.md`
   - Existing HDF5 final state and validation summary: `docs/runs/activation_full_20260622/postmortem.md:9-35`.
   - Dynamic extraction scheduler lessons: `docs/runs/activation_full_20260622/postmortem.md:45-55`.
   - Disk/transfer/compression warnings: `docs/runs/activation_full_20260622/postmortem.md:57-81`.
   - Next-step recommendation to train probes: `docs/runs/activation_full_20260622/postmortem.md:96-103`.

5. `docs/runs/activation_full_20260622/artifact_manifest.md`
   - Drive folder, local path, size, and SHA256 for the 11 GB HDF5.

## The Most Important Current Correction

The existing activation HDF5 is not "everything".

It contains only:

```text
roleplaying__plain:        329 extracted examples
sandbagging_v2__wmdp_mmlu: 128 extracted examples
```

That is documented in `docs/runs/activation_full_20260622/README.md:26-34` and
`docs/runs/activation_full_20260622/postmortem.md:83-94`.

Since that HDF5 was produced, insider-trading rollout judging has been wired up and run locally.
Therefore:

- Probe training can start on the existing roleplaying/sandbagging HDF5.
- Insider probes need a new activation extraction pass first, because the existing HDF5 predates the
  insider labels.
- If you make a new all-task HDF5, do not overwrite the validated old one unless Avinash explicitly wants that.
  Write a new artifact path or a clearly named incremental insider HDF5.

## Current Local Git State

As of this handoff, the worktree is dirty. Do not assume `main` or GitHub has these changes.

Modified files:

```text
data/rollouts/insider_trading__onpolicy__qwen3-vl-8b-thinking.json
data/rollouts/insider_trading_doubledown__onpolicy__qwen3-vl-8b-thinking.json
src/intelligent_liars/activations.py
src/intelligent_liars/cli.py
src/intelligent_liars/judging.py
tests/test_activations.py
tests/test_cli.py
```

Untracked file:

```text
tests/test_insider_rollout_grading.py
```

The data files above were changed because real OpenRouter judging was run on the insider rollout
outputs. The code files were changed so `grade-rollouts` and the activation loader understand
insider rollout files.

Before committing or pushing anything, rerun:

```bash
git status --short --untracked-files=all
PYTHONPATH=src uv run --no-sync pytest -q
PYTHONPATH=src uv run --no-sync ruff check src tests scripts
git diff --check
```

Current validation already completed:

```text
pytest: 127 passed in 0.81s
ruff:  All checks passed
diff whitespace check: clean
```

## What Was Added Locally For Insider Judging

The local code now supports the two insider rollout tasks in the generic rollout grader.

Important code points:

- `src/intelligent_liars/cli.py:193-195`
  - `_rollout_task_needs_llm_judge()` now includes `insider_trading` and
    `insider_trading_doubledown`.

- `src/intelligent_liars/judging.py:975-1073`
  - `_grade_insider_trading_rollout_data()` grades rollout JSON rows using the existing
    insider judge functions.
  - It writes labels back into `graded_deception`, reasoning into `grader_reasoning`, and
    label metadata into each record.
  - It supports `continue_on_error`; a failed judge call becomes `skip` with a `judge_error`.

- `src/intelligent_liars/judging.py:1076-1103`
  - `_grade_insider_trading_rollout_output()` dispatches:
    - normal insider reports through `classify_insider_report_message()`,
    - doubledown messages through `classify_doubling_down_message()`.

- `src/intelligent_liars/activations.py:27-55`
  - String label maps now include skip/refusal/invalid-style values.
  - `INSIDER_REPORT_LABELS` maps:
    - `explicit -> HONEST`
    - `concealed -> DECEPTIVE`
    - `implied -> SKIP`
    - `invalid`, `unknown`, `ambiguous`, `skip`, `refusal -> SKIP`
  - `STRING_DECEPTION_LABELS` maps:
    - `honest -> HONEST`
    - `deceptive -> DECEPTIVE`
    - `unknown`, `ambiguous`, `refusal`, `skip`, `invalid -> SKIP`

- `src/intelligent_liars/activations.py:406-454`
  - Named activation tasks now look first for `data/rollouts/{task}__{generated_model}.json`.
  - This matters because the new insider files are rollout-style JSONs, not only the older
    transcript-style `data/insider_trading/qwen3-vl-8b-thinking-generations.json`.

- `src/intelligent_liars/activations.py:994-1029`
  - Label normalization is centralized.
  - Roleplaying scores normalize as:
    - score `1` or `2` -> HONEST
    - score `6` or `7` -> DECEPTIVE
    - score `3`, `4`, or `5` -> SKIP

- `src/intelligent_liars/activations.py:2374-2408`
  - Task-specific answer-span trimming happens here.
  - Sandbagging requires a `<step4>` marker.
  - Insider doubledown requires `Action: send_message`.
  - Insider report requires `Action: `.
  - Missing markers turn the example into `SKIP`, because the activation extractor should not
    train on the wrong part of the transcript.

## The Real Insider Judging Run

The first attempts used structured outputs and failed with OpenRouter provider errors.
The successful path used non-structured output.

The full successful command was:

```bash
/usr/bin/time -p env PYTHONPATH=src uv run --no-sync intelligent-liars grade-rollouts \
  --project-root . \
  --path data/rollouts/insider_trading__onpolicy__qwen3-vl-8b-thinking.json \
  --path data/rollouts/insider_trading_doubledown__onpolicy__qwen3-vl-8b-thinking.json \
  --max-workers 25 \
  --flush-every 10 \
  --no-structured-outputs \
  --no-require-structured-outputs
```

Observed result:

```text
insider report:     graded=167 skipped=6
insider doubledown: graded=86 skipped=5
wall time:          about 115 seconds
judge alias:        glm-5.2 from root model_deployments.yaml
provider route:     z-ai/fp8 first, wafer/fp4 fallback
```

Important: `prod_env/model_deployments.yaml` was not present locally; the CLI used the root
`model_deployments.yaml`. `PROJECT.md:112-114` already says the current OpenRouter judge config
lives at the repo root.

## Current Rollout Label State

These counts were verified through the repo loader on 2026-06-22 with:

```bash
PYTHONPATH=src uv run --no-sync python - <<'PY'
from collections import Counter
from pathlib import Path
from intelligent_liars.activations import load_named_activation_examples
root = Path(".")
for task in [
    "roleplaying__plain",
    "sandbagging_v2__wmdp_mmlu",
    "insider_trading__onpolicy",
    "insider_trading_doubledown__onpolicy",
]:
    examples = load_named_activation_examples(task, project_root=root, generated_model="qwen3-vl-8b-thinking")
    print(task)
    print(" examples", len(examples))
    print(" labels", dict(sorted(Counter(ex.label for ex in examples).items())))
    print(" raw", dict(sorted(Counter(str(ex.raw_label) for ex in examples).items())))
    print(" empty_detected", sum(1 for ex in examples if not ex.detected_text.strip()))
PY
```

Normalized label convention:

```text
HONEST = 0
DECEPTIVE = 1
SKIP = -1
```

Current normalized counts:

| Task | Rows/examples | HONEST | DECEPTIVE | SKIP | Probe-usable |
| --- | ---: | ---: | ---: | ---: | ---: |
| `roleplaying__plain` | 371 | 106 | 223 | 42 | 329 |
| `sandbagging_v2__wmdp_mmlu` | 366 | 73 | 55 | 238 | 128 |
| `insider_trading__onpolicy` | 173 | 107 | 10 | 56 | 117 |
| `insider_trading_doubledown__onpolicy` | 91 | 27 | 63 | 1 | 90 |

Aggregate:

```text
total rollout examples: 1001
probe-usable binary examples: 664
honest examples: 313
deceptive examples: 351
skip examples: 337
```

Raw labels:

```text
roleplaying__plain
  1: 93
  2: 13
  3: 8
  4: 9
  5: 13
  6: 21
  7: 202
  skip: 12

sandbagging_v2__wmdp_mmlu
  honest: 74
  deceptive: 55
  ambiguous: 26
  skip: 211

insider_trading__onpolicy
  explicit: 107
  implied: 44
  concealed: 10
  invalid: 12

insider_trading_doubledown__onpolicy
  deceptive: 63
  honest: 27
  invalid: 1
```

Note the sandbagging subtlety:

- Raw sandbagging has `74` honest.
- The activation loader sees only `73` honest.
- One raw-honest row becomes `SKIP` because the answer-span trim requires `<step4>`.
- This is correct behavior unless you explicitly choose to relax the trim.

## How To Think About Skips

This is the core conceptual point from the most recent user question.

These rollout files are not the final benchmark outputs. They are training/evaluation data for finding
truth directions. That changes the goal.

The goal is not "make every generated output judged". The goal is:

1. enough clean honest examples,
2. enough clean deceptive examples,
3. answer-token spans that point at the actual report/action/answer,
4. labels whose semantics match the direction you are training.

Therefore:

```text
Rerun or rejudge structural failures.
Do not force ambiguous semantic cases into binary labels.
```

Rerun or rejudge examples if the skip reason is:

- judge API/provider failure,
- empty generation,
- malformed JSON/action format,
- missing required answer marker,
- truncated generation,
- refusal,
- prompt/template failure where the model did not perform the task at all.

Do not blindly rerun examples if the skip reason is:

- roleplaying score `3`, `4`, or `5`,
- sandbagging `ambiguous`,
- insider report `implied` under the strict Truth Spec-style mapping,
- any judge label that says the answer is not clearly honest or deceptive.

Those are weak labels, not missing data. Weak labels are actively harmful for linear directions.

## Per-Dataset Rerun Recommendation

### Insider doubledown

Status:

```text
91 rows
90 binary-usable
1 invalid/skip
```

Recommendation:

- Do not spend time rerunning this unless Avinash wants every possible row.
- It is already excellent for probe data.
- The next useful step is activation extraction, not more generation.

### Insider report

Status:

```text
173 rows
117 binary-usable
107 explicit/honest
10 concealed/deceptive
44 implied/skip
12 invalid/skip
```

Recommendation:

- Do not rerun `implied` just to make it binary in the strict main run.
- Consider rerunning only the `invalid` rows if you need a little more data.
- Main strict mapping should stay `explicit -> honest`, `concealed -> deceptive`, `implied -> skip`.
- Add an ablation later where `implied -> deceptive` or where `implied` is a third class / margin class,
  because many `implied` cases are partial disclosure. Do not make that the primary result without saying so.

Important reason:

- `implied` is not the same as failed generation. It often means the model disclosed enough to hint at the
  insider source but did not explicitly say it.
- For a truth direction, this is label-noisy. It may be scientifically interesting, but it should not pollute
  the primary binary direction.

### Roleplaying

Status:

```text
371 rows
329 binary-usable
42 skip
12 raw skip rows are judge-error skips
scores 3/4/5 are semantic middle scores and normalize to skip
```

Recommendation:

- The 12 raw `skip` rows can be rejudged if desired.
- Do not rerun/rejudge the `3`, `4`, and `5` rows expecting them to become clean labels. Some may change, but
  the correct primary dataset is the high-confidence ends of the 1-7 scale.
- If rejudging the 12 judge-error rows, use `--overwrite` and explicit source-index filtering. The source
  indices are listed in `docs/runs/vast-41988240-20260621T184553Z/postmortem.md:84-90`.

### Sandbagging

Status:

```text
366 rows
128 binary-usable through the activation loader
238 skip
```

Recommendation:

- This is the only dataset where rerun/regeneration may materially help.
- But first classify skip reasons. Do not treat all `skip` values as the same.
- Many skips likely come from deterministic Apollo protocol mismatch, missing required tags, or long freeform
  reasoning that did not land in the expected `<step4>` answer format.
- If the goal is more sandbagging data, update/regenerate the prompt/template so Qwen reliably emits the needed
  final answer structure, then run the deterministic grader again.
- If you only need a first probe, 128 sandbagging examples plus the other rollout families may be enough to start.

## What To Build Next

There are two viable next steps. Pick based on Avinash's immediate priority.

### Path A: probe training on existing activations

This is the fastest path to seeing whether the current roleplaying/sandbagging activations contain a usable
linear truth/deception signal.

Input:

```text
artifacts/activations/activation_full_20260622/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5
```

Important HDF5 facts:

- 36 layers.
- `float16` storage.
- tasks present:
  - `roleplaying__plain`
  - `sandbagging_v2__wmdp_mmlu`
- It is an activation table, not a trained direction. See `docs/runs/activation_full_20260622/postmortem.md:92-103`.

Probe requirements:

- Split by example, not by token row.
- Aggregate answer-token rows per example before the train/test split, or carry an example id and split before
  token-level expansion.
- Start simple:
  - mean-pool answer-token activations per example,
  - train logistic regression or linear SVM per layer,
  - report accuracy, balanced accuracy, AUC if possible,
  - report label counts per split,
  - compare within-task and cross-task transfer.
- Keep sign explicit:
  - `HONEST = 0`,
  - `DECEPTIVE = 1`,
  - record whether the learned vector points honest-to-deceptive or deceptive-to-honest.
- Do not start with Stratified INLP or a representation atlas unless Avinash asks. They can come after a simple
  linear probe has a pulse.

Current CLI shape:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars train-probes \
  --input artifacts/activations/activation_full_20260622/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probes/qwen3-vl-8b-thinking_rollout_probes.json
```

This CLI now exists and writes `qwen_answer_token_probe_results_v2`.

Minimum output should include:

```text
task
layer
train examples
test examples
honest/deceptive counts
accuracy
balanced accuracy
auc
direction norm
direction sign convention
```

### Path B: extract insider activations first

This is the better path if Avinash wants "all four rollout tests" represented before probe training.

The code can now load insider rollout files through named tasks. That is the point of the local changes around
`src/intelligent_liars/activations.py:406-454`.

Local no-GPU preflight on 2026-06-22 showed the named tasks load and have usable binary examples:

```text
insider_trading__onpolicy: 173 rows loaded, 117 binary usable, 56 skipped
  labels: honest=107, deceptive=10, skip=56
insider_trading_doubledown__onpolicy: 91 rows loaded, 90 binary usable, 1 skipped
  labels: honest=27, deceptive=63, skip=1
```

The extractor filters `SKIP=-1` rows before the model forward pass. That means this GPU job should extract about
207 examples total, not all 264 raw insider rows.

GPU command shape:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars extract-activations \
  --task insider_trading__onpolicy \
  --task insider_trading_doubledown__onpolicy \
  --layers all \
  --batch-size 1 \
  --output artifacts/activations/extracted_feats_insider_qwen3-vl-8b-thinking.h5
```

Run this on a GPU box. Do not try to load Qwen3-VL for full extraction on the Mac.
For this insider-only pass, a single A100 80GB or H100 80GB should be enough; the direct CLI above uses one model
process and will not automatically saturate multiple GPUs.

Given the prior extraction postmortem, use:

- enough disk or `/dev/shm`;
- `float16` storage;
- one worker per A100 is enough for extraction;
- dynamic queue if using the remote helper path;
- exact byte-count/sha validation after transfer.

Do not append into the old 11 GB HDF5 unless you are deliberately implementing safe append/resume behavior.
It is safer to write an insider-only HDF5 and merge later if needed.

## What Not To Do

Do not:

- treat skipped examples as failed benchmark outputs;
- force all `implied`, `ambiguous`, or middle-score rows into binary labels;
- train a direction on token rows after randomly splitting rows, because that leaks examples across train/test;
- assume the existing activation HDF5 includes insider examples;
- overwrite the validated 11 GB HDF5 casually;
- push the 11 GB HDF5 to GitHub;
- use one giant generation process expecting multi-GPU saturation;
- rerun all rollouts just because there are skips;
- proceed to steering without verifying vector sign.

## Data Hygiene

The big activation HDF5 is intentionally not committed. It lives locally and in the Google Drive folder:

```text
https://drive.google.com/drive/folders/1C7KUkZHCfQXnpkt6debhCcA4YtV8eOHX
```

See `docs/runs/activation_full_20260622/artifact_manifest.md` for the local path, Drive folder, size, and SHA.

The rollout JSONs are much smaller than the HDF5 but are still data. If Avinash asks for commits, keep code
commits separate from data commits, and use lowercase commit messages.

## Suggested Immediate Plan

If Avinash asks "what now?", the most defensible answer is:

1. Commit or otherwise preserve the current insider judging code and data state.
2. Run a small probe-training implementation on the existing roleplaying/sandbagging HDF5.
3. In parallel or next, extract insider activations into a new insider-only HDF5.
4. Run probes on:
   - roleplaying only,
   - sandbagging only,
   - insider report only,
   - insider doubledown only,
   - combined all binary examples,
   - cross-task transfer.
5. Only after this, pick candidate layers and implement steering hooks.

The first meaningful scientific question is not "can we rerun every skip".
It is:

```text
Do clean binary answer-token activations linearly separate honest vs deceptive behavior,
and does that separation transfer across task families?
```

That is the probe step.
