# Capy Captain handoff: Qwen3-VL capability evaluation

## Purpose

Act as Avinash's primary orchestrator and conversational partner for the next phase of `intelligent_liars`. The immediate goal is orientation and discussion. Do not implement, spawn Build tasks, create a PR, rent infrastructure, or start an evaluation until Avinash explicitly asks.

## Project objective

Establish an immutable capability baseline for the unmodified `Qwen/Qwen3-VL-8B-Thinking` model, then compare it with one or more layer-21 truth-direction edits under the identical harness:

- Full 5-shot MMLU over all 57 subjects and 14,042 test questions.
- Official OSWorld 369-task suite, initially capped at 15 steps per task.
- Complete OSWorld trajectories: initial and per-step screenshots, raw model response/reasoning, parsed action, execution result, reward, timestamps, errors/retries, score, logs, and recording.
- Baseline first. Abliteration or direction-reflection variants come only after the baseline is complete and validated.

OSWorld scores final desktop task state through its task-specific evaluator. KL divergence is not an OSWorld metric; any KL measurement would be a separate model-drift diagnostic.

## Intended architecture

- Capy Captain is the project orchestrator, not the computer-use model being evaluated.
- A deterministic OSWorld harness runs on the worker infrastructure.
- Qwen3-VL is served from a separately managed GPU through an authenticated OpenAI-compatible endpoint.
- On each OSWorld step, the harness sends the instruction, screenshot, and relevant history to Qwen; parses Qwen's action; executes it in the desktop environment; records the transition; and repeats until `DONE`, `FAIL`, an error, or the step limit.
- Capy must not substitute its own model decisions for Qwen's actions.

## Capy and billing state

- Project: `intelligent_liars` (`p3rciv3l/intelligent_liars`, branch `main`).
- Default Captain model: GPT-5.6 Sol.
- OpenAI models are routed through Avinash's connected Codex subscription.
- External-subscription fallback is set to **Error immediately**. Never fall back to Capy model credits.
- Capy platform/VM runtime charges still apply.
- The API service user has a $5 monthly cap and Capy-model fallback disabled.

## Infrastructure gates

Before treating Capy as a replacement for OSWorld's official AWS host/client provider, prove all four:

1. `/dev/kvm` is exposed in the Capy task VM.
2. OSWorld's Docker/QEMU desktop starts with the necessary privileges and acceptable performance.
3. Task duration and concurrency are adequate for a supervised queue.
4. Trajectories and recordings are durably exported before Capy destroys the task VM.

Capy's public documentation confirms Docker Engine, Compose, outbound networking, and per-task Ubuntu VMs. It does not establish KVM or privileged-container support. Do not infer those capabilities. The first paid action should be one bounded capability probe with a forced stop deadline.

## Safety and evidence rules

- Do not rent a GPU, launch external cloud resources, or start a large Capy fleet without an exact priced proposal and Avinash's explicit approval.
- Start with one OSWorld task, then 3-5 tasks, then OSWorld Small, and only then consider all 369.
- Never overwrite a valid result or silently change the task grid, checkpoint, model revision, step budget, prompt, action parser, or evaluator.
- Store large screenshots/videos/trajectories outside Git and sync incrementally to durable storage.
- Preserve unrelated and untracked user work. Do not assume local-only files are visible in Capy's GitHub checkout.
- Distinguish task failure, refusal, invalid action, infrastructure failure, and general capability degradation.

## Repository truth and references

Capy should inspect the current GitHub `main` branch as its code source of truth. The robustness implementation and validation evidence were published at `c1a73d3c4cf011bf33492b5694d6926fe8d78a5b`. Local uncommitted work must still not be assumed present remotely.

Read rather than duplicate:

- `PROJECT.md` for the research program and capability-preservation framing.
- `docs/validation/no_insider_seed_robustness_20260709.md` for the selected layer-21 direction evidence.
- Official OSWorld repository and setup guide: <https://github.com/xlang-ai/OSWorld>.
- Capy VM/runtime docs: <https://docs.capy.ai/configs/runtime>.

## First response requested from Captain

Confirm that you understand the objective, architecture, billing boundary, and four Capy capability gates. Report what you can see on GitHub `main`, identify any conflict with this handoff, and then wait for Avinash's next message. Do not begin implementation or paid infrastructure work in the orientation response.

## Suggested skills

- Repository inspection and experiment-handoff reading.
- OSWorld official-harness research.
- Cloud lifecycle and spend-safety review.
- `vast-gpu-experiments`, `research`, and `code-review` if equivalent skills are available in the Capy environment.
