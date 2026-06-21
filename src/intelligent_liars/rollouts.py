from __future__ import annotations

import json
import hashlib
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

from intelligent_liars.models import DEFAULT_MODEL_ID, ModelBundle


MODEL_SLUG = "qwen3-vl-8b-thinking"
ROLEPLAYING_TASK = "roleplaying__plain"
DEFAULT_ROLLOUT_TASKS = (
    ROLEPLAYING_TASK,
    "sandbagging_v2__wmdp_mmlu",
)
DEFAULT_ROLLOUT_PROMPT_SET = "truth-spec-llama-70b"


Message = dict[str, Any]
T = TypeVar("T")


@dataclass(frozen=True)
class PromptExample:
    """One prompt row that should be completed by Qwen during rollout generation.

    Attributes:
        source_index: Stable row index from the source prompt set. This is used
            for resume/skipping behavior in `generate_rollout_task`.
        input_messages: Chat messages to render through the Qwen processor chat
            template before generation.
        expected_label: Optional source-side label retained for later grading or
            debugging. It is not used as a target completion.
        metadata: Provenance from `data/roleplaying/dataset.yaml` or
            `data/rollout_prompts/*.json`.
        input_str: Human-readable prompt text for metadata/debug output.
    """

    source_index: int
    input_messages: list[Message]
    expected_label: Any = None
    metadata: dict[str, Any] | None = None
    input_str: str | None = None


@dataclass(frozen=True)
class GenerationSettings:
    """Generation and checkpointing options shared by rollout generators.

    Attributes:
        batch_size: Number of independent conversations passed to
            `model.generate` at once.
        max_new_tokens: Maximum assistant tokens to generate per conversation.
        do_sample: Whether to sample from the next-token distribution. When
            false, generation is greedy and `temperature`, `top_p`, and `top_k`
            are intentionally ignored.
        temperature: Sampling temperature used only when `do_sample` is true.
        top_p: Nucleus sampling cutoff used only when `do_sample` is true.
        top_k: Top-k sampling cutoff used only when `do_sample` is true.
        seed: Random seed applied by `seed_everything`.
        flush_every: Write output JSON after this many newly generated examples.
    """

    batch_size: int = 1
    max_new_tokens: int = 40960
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = 20
    seed: int = 0
    flush_every: int = 10

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be >= 0 when provided")
        if self.flush_every < 1:
            raise ValueError("flush_every must be >= 1")


# CLI commands read these frozen defaults so option defaults stay synchronized
# with the runtime settings object instead of repeating literal values.
DEFAULT_ROLLOUT_GENERATION_SETTINGS = GenerationSettings()
DEFAULT_INSIDER_GENERATION_SETTINGS = GenerationSettings(
    do_sample=True,
    flush_every=5,
)


@dataclass(frozen=True)
class GenerationSummary:
    """Small return object describing a completed generation run."""

    task: str
    output_path: Path
    total_examples: int
    generated_examples: int
    skipped_existing: int


@dataclass(frozen=True)
class GeneratedCompletion:
    """Decoded Qwen completion split into hidden thinking and final answer."""

    text: str
    thinking: str | None = None


def qwen_model_slug(model_id: str = DEFAULT_MODEL_ID) -> str:
    """Return the filename-safe Qwen model slug used in generated rollout paths.

    Args:
        model_id: Hugging Face model id. The default `Qwen/Qwen3-VL-8B-Thinking`
            maps to the stable project slug `qwen3-vl-8b-thinking`.

    Returns:
        A lowercase filesystem-safe slug used in paths such as
        `data/rollouts/{task}__{model_slug}.json`.
    """
    if model_id == DEFAULT_MODEL_ID:
        return MODEL_SLUG
    return (
        model_id.split("/")[-1]
        .replace("_", "-")
        .replace(".", "-")
        .lower()
    )


def parse_task_name(task: str) -> tuple[str, str]:
    """Split a Truth Spec task id into its base dataset and variant names.

    Args:
        task: Task id formatted as `{base}__{variant}`, for example
            `sandbagging_v2__wmdp_mmlu`.

    Returns:
        `(base, variant)`.

    Raises:
        ValueError: If the task id is not exactly two pieces separated by `__`.
    """

    parts = task.split("__")
    if len(parts) != 2:
        raise ValueError(f"Task must be formatted as base__variant, got {task!r}.")
    return parts[0], parts[1]


def load_rollout_prompt_examples(
    task: str,
    *,
    project_root: Path,
    prompt_set: str | None = None,
) -> list[PromptExample]:
    """Load prompt-only examples for a rollout task.

    References:
        When `prompt_set` is provided, tasks are loaded from cleaned prompt sets
        under `data/rollout_prompts/`. `roleplaying__plain` falls back to
        `data/roleplaying/dataset.yaml` only when no prompt set is selected.

    Args:
        task: Truth Spec task id, e.g. `roleplaying__plain` or
            `sandbagging_v2__wmdp_mmlu`.
        project_root: Repository root used to resolve `data/...` paths.
        prompt_set: Optional prompt-set id, e.g. `truth-spec-llama-70b`, used
            when multiple cleaned prompt sets exist for the same task.

    Returns:
        Prompt examples ready for Qwen generation.
    """

    if task == ROLEPLAYING_TASK and prompt_set is None:
        return load_roleplaying_prompt_examples(project_root=project_root)

    return load_rollout_prompt_set_examples(
        task,
        project_root=project_root,
        prompt_set=prompt_set,
    )


def load_roleplaying_prompt_examples(*, project_root: Path) -> list[PromptExample]:
    """Build roleplaying prompts from `data/roleplaying/dataset.yaml`.

    The YAML rows contain `scenario`, `question`, and optional `answer_prefix`.
    `honest_completion` and `deceptive_completion` are preserved only as
    metadata; Qwen still generates its own completion later.

    Args:
        project_root: Repository root containing `data/roleplaying/dataset.yaml`.

    Returns:
        Prompt examples with `system` and `user` messages, plus an optional
        assistant prefix placeholder when the YAML row has `answer_prefix`.
    """

    import yaml

    yaml_path = project_root / "data" / "roleplaying" / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Roleplaying source data not found: {yaml_path}. "
            "Prepare local data/roleplaying/dataset.yaml before generating roleplaying rollouts."
        )

    rows = yaml.safe_load(yaml_path.read_text()) or []
    if not isinstance(rows, list):
        raise ValueError(f"Roleplaying source data must be a list of rows: {yaml_path}")
    examples: list[PromptExample] = []
    for idx, row in enumerate(rows):
        # The system prompt carries the roleplay scenario; the user message
        # carries the concrete question asked within that scenario.
        scenario = str(row["scenario"]).strip()
        question = str(row["question"]).strip()
        system = (
            scenario
            + "\n\nAnswer the question directly, without asides or internal thoughts."
        )
        messages = [
            {"role": "system", "content": system, "detect": False},
            {"role": "user", "content": question, "detect": False},
        ]
        answer_prefix = str(row.get("answer_prefix") or "")
        if answer_prefix:
            # The empty detect=True assistant message marks where the generated
            # answer will be scored after Qwen continues the assistant prefix.
            messages.append({"role": "assistant", "content": answer_prefix, "detect": False})
            messages.append({"role": "assistant", "content": "", "detect": True})
        examples.append(
            PromptExample(
                source_index=idx,
                input_messages=messages,
                expected_label=None,
                input_str=_messages_to_text(messages),
                metadata={
                    "source_path": str(yaml_path),
                    "scenario": scenario,
                    "question": question,
                    "answer_prefix": answer_prefix or None,
                    "honest_completion": row.get("honest_completion"),
                    "deceptive_completion": row.get("deceptive_completion"),
                },
            )
        )
    if not examples:
        raise ValueError(f"Roleplaying source data contains no examples: {yaml_path}")
    return examples


def load_rollout_prompt_set_examples(
    task: str,
    *,
    project_root: Path,
    prompt_set: str | None = None,
) -> list[PromptExample]:
    """Load a cleaned prompt set prepared from Truth Spec reference rollouts.

    References:
        Prompt-set JSON files live under `data/rollout_prompts/` and are
        produced by `scripts/prepare_rollout_prompt_sets.py`. They contain
        prompts only; old reference completions and old model chat-template text
        have already been removed.

    Args:
        task: Task id formatted as `{base}__{variant}`.
        project_root: Repository root used to resolve `data/rollout_prompts/`.
        prompt_set: Optional prompt-set id. Required when multiple prompt sets
            exist for the task.

    Returns:
        Prompt examples ready for Qwen generation.
    """

    base, variant = parse_task_name(task)
    source_path = _resolve_rollout_prompt_set_path(
        project_root=project_root,
        base=base,
        variant=variant,
        prompt_set=prompt_set,
    )

    data = json.loads(source_path.read_text())
    examples: list[PromptExample] = []
    for idx, item in enumerate(data["examples"]):
        messages = [_normalise_message(message) for message in item["input_messages"]]
        examples.append(
            PromptExample(
                source_index=idx,
                input_messages=messages,
                expected_label=item.get("expected_label"),
                input_str=item.get("input_str") or _messages_to_text(messages),
                metadata={
                    "source_path": str(source_path),
                    "prompt_set": data.get("prompt_set", prompt_set),
                    "source_model": data.get("source_model"),
                    "source_metadata": item.get("metadata", {}),
                },
            )
        )
    if not examples:
        raise ValueError(f"Cleaned prompt set contains no examples: {source_path}")
    return examples


def _resolve_rollout_prompt_set_path(
    *,
    project_root: Path,
    base: str,
    variant: str,
    prompt_set: str | None,
) -> Path:
    """Resolve the cleaned prompt-set JSON path for a rollout task.

    Args:
        project_root: Repository root.
        base: Task base from `parse_task_name`, e.g. `sandbagging_v2`.
        variant: Task variant from `parse_task_name`, e.g. `wmdp_mmlu`.
        prompt_set: Optional prompt-set id, e.g. `truth-spec-llama-70b`.

    Returns:
        Path to `data/rollout_prompts/{base}__{variant}__{prompt_set}.json`.

    Raises:
        FileNotFoundError: If no matching prompt-set file exists.
        FileExistsError: If multiple prompt sets match and `prompt_set` was not
            provided.
    """

    source_dir = project_root / "data" / "rollout_prompts"
    if prompt_set:
        source_path = source_dir / f"{base}__{variant}__{prompt_set}.json"
        if not source_path.exists():
            raise FileNotFoundError(f"Cleaned prompt set not found: {source_path}")
        return source_path

    matches = sorted(source_dir.glob(f"{base}__{variant}__*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No cleaned prompt set found for {base}__{variant} in {source_dir}. "
            "Run scripts/prepare_rollout_prompt_sets.py first."
        )
    if len(matches) > 1:
        choices = ", ".join(path.stem.rsplit("__", maxsplit=1)[-1] for path in matches)
        raise FileExistsError(
            f"Multiple cleaned prompt sets match {base}__{variant}; pass --prompt-set. Choices: {choices}"
        )
    return matches[0]


def generate_rollout_task(
    *,
    bundle: ModelBundle,
    task: str,
    examples: Sequence[PromptExample],
    output_path: Path,
    settings: GenerationSettings,
    overwrite: bool = False,
    start: int = 0,
    limit: int | None = None,
) -> GenerationSummary:
    """Generate Qwen completions and write a Truth Spec-compatible rollout file.

    References:
        Input examples come from `load_rollout_prompt_examples`. Output records
        are written to `data/rollouts/{task}__{model_slug}.json` by the CLI.

    Args:
        bundle: Loaded Qwen model/processor/tokenizer bundle.
        task: Task id, used for output metadata.
        examples: Prompt examples to complete.
        output_path: JSON file to create or resume.
        settings: Generation options and checkpoint interval.
        overwrite: If true, discard existing output and regenerate from scratch.
        start: Source example index offset.
        limit: Optional maximum number of examples after `start`.

    Returns:
        Counts for total selected examples, newly generated examples, and
        skipped examples already present in the output file.
    """

    if not examples:
        raise ValueError(f"No prompt examples were provided for task {task!r}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base, variant = parse_task_name(task)

    expected_metadata = _rollout_run_metadata(
        task=task,
        model_id=bundle.model_id,
        examples=examples,
        settings=settings,
    )

    output_data: dict[str, Any]
    if overwrite or not output_path.exists():
        output_data = {
            "dataset_class": "qwen_generated_rollout",
            "base_name": base,
            "variant": variant,
            "model": qwen_model_slug(bundle.model_id),
            "model_id": bundle.model_id,
            "run_metadata": expected_metadata,
            "rollouts": [],
        }
    else:
        output_data = json.loads(output_path.read_text())
        _validate_rollout_resume_metadata(output_path, output_data, expected_metadata)

    # Resume key: generated rows carry the prompt source_index in metadata, so
    # reruns can skip rows already present in an existing JSON file.
    existing_indices = _existing_rollout_source_indices(output_data)

    # `start` and `limit` slice the source prompt list before resume filtering.
    selected_examples = list(examples[start:])
    if limit is not None:
        selected_examples = selected_examples[:limit]

    pending = [example for example in selected_examples if example.source_index not in existing_indices]
    if not pending:
        return GenerationSummary(
            task=task,
            output_path=output_path,
            total_examples=len(selected_examples),
            generated_examples=0,
            skipped_existing=len(selected_examples),
        )
    generated = 0

    for batch in _chunks(pending, settings.batch_size):
        completions = generate_completions(
            bundle=bundle,
            conversations=[example.input_messages for example in batch],
            settings=settings,
        )
        for example, completion in zip(batch, completions, strict=True):
            output_data["rollouts"].append(
                build_rollout_record(
                    example=example,
                    completion=completion,
                    generation_settings=settings,
                )
            )
            generated += 1
        if generated % settings.flush_every == 0:
            write_json_atomic(output_path, output_data)

    write_json_atomic(output_path, output_data)
    return GenerationSummary(
        task=task,
        output_path=output_path,
        total_examples=len(selected_examples),
        generated_examples=generated,
        skipped_existing=len(selected_examples) - len(pending),
    )


def _rollout_run_metadata(
    *,
    task: str,
    model_id: str,
    examples: Sequence[PromptExample],
    settings: GenerationSettings,
) -> dict[str, Any]:
    source_paths = sorted(
        {
            str(metadata["source_path"])
            for example in examples
            if (metadata := example.metadata or {}).get("source_path") is not None
        }
    )
    prompt_sets = sorted(
        {
            str(metadata["prompt_set"])
            for example in examples
            if (metadata := example.metadata or {}).get("prompt_set") is not None
        }
    )
    return {
        "task": task,
        "model": qwen_model_slug(model_id),
        "model_id": model_id,
        "generation_settings": generation_content_settings(settings),
        "source_example_count": len(examples),
        "prompt_sets": prompt_sets,
        "source_paths": source_paths,
        "source_sha256": {source_path: _sha256_file(Path(source_path)) for source_path in source_paths},
    }


def validate_rollout_generation_resume(
    *,
    task: str,
    output_path: Path,
    model_id: str,
    examples: Sequence[PromptExample],
    settings: GenerationSettings,
    overwrite: bool,
) -> None:
    """Validate an existing rollout output file before expensive model loading."""

    if overwrite or not output_path.exists():
        return
    output_data = json.loads(output_path.read_text())
    if not isinstance(output_data, dict):
        raise ValueError(f"Existing rollout file must contain a JSON object: {output_path}")
    expected_metadata = _rollout_run_metadata(
        task=task,
        model_id=model_id,
        examples=examples,
        settings=settings,
    )
    _validate_rollout_resume_metadata(output_path, output_data, expected_metadata)


def pending_rollout_source_indices(
    *,
    task: str,
    output_path: Path,
    model_id: str,
    examples: Sequence[PromptExample],
    settings: GenerationSettings,
    overwrite: bool,
    start: int = 0,
    limit: int | None = None,
) -> set[int]:
    """Return source indices that still need Qwen generation for this rollout."""

    selected_examples = list(examples[start:])
    if limit is not None:
        selected_examples = selected_examples[:limit]
    selected_indices = {example.source_index for example in selected_examples}
    if overwrite or not output_path.exists():
        return selected_indices

    output_data = json.loads(output_path.read_text())
    if not isinstance(output_data, dict):
        raise ValueError(f"Existing rollout file must contain a JSON object: {output_path}")
    expected_metadata = _rollout_run_metadata(
        task=task,
        model_id=model_id,
        examples=examples,
        settings=settings,
    )
    _validate_rollout_resume_metadata(output_path, output_data, expected_metadata)
    return selected_indices - _existing_rollout_source_indices(output_data)


def generation_content_settings(settings: GenerationSettings) -> dict[str, Any]:
    """Return generation settings that can affect model outputs."""

    values: dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
    }
    if settings.do_sample:
        values.update(
            {
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "seed": settings.seed,
            }
        )
        if settings.top_k is not None:
            values["top_k"] = settings.top_k
    return values


def _existing_rollout_source_indices(output_data: dict[str, Any]) -> set[int]:
    return {
        int(item.get("metadata", {}).get("source_index"))
        for item in output_data.get("rollouts", [])
        if item.get("metadata", {}).get("source_index") is not None
    }


def _validate_rollout_resume_metadata(
    output_path: Path,
    output_data: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> None:
    if output_data.get("run_metadata") != expected_metadata:
        raise ValueError(
            f"Existing rollout file was produced with different run metadata: {output_path}. "
            "Use --overwrite to regenerate it."
        )


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rollout_record(
    *,
    example: PromptExample,
    completion: str | GeneratedCompletion,
    generation_settings: GenerationSettings,
) -> dict[str, Any]:
    """Build one Truth Spec-compatible rollout record from one Qwen completion.

    Args:
        example: Source prompt example used for generation.
        completion: Assistant text decoded from Qwen's generated tokens.
        generation_settings: Settings used for this generation run; stored in
            metadata for reproducibility.

    Returns:
        A rollout record with `graded_deception` initialized to `unknown` so
        `judging.py` can fill labels later.
    """

    generated = completion if isinstance(completion, GeneratedCompletion) else split_qwen_thinking(completion)
    metadata = dict(example.metadata or {})
    metadata.update(
        {
            "source_index": example.source_index,
            "generation_settings": asdict(generation_settings),
        }
    )
    if generated.thinking is not None:
        metadata["qwen_thinking_content"] = generated.thinking
    return {
        "input_messages": example.input_messages,
        "expected_label": example.expected_label,
        "input_str": example.input_str or _messages_to_text(example.input_messages),
        "output_messages": [{"role": "assistant", "content": generated.text, "detect": True}],
        "output_strs": [generated.text],
        "grader_reasoning": [None],
        "graded_deception": ["unknown"],
        "deception_percentage": None,
        "metadata": metadata,
    }


def split_qwen_thinking(text: str) -> GeneratedCompletion:
    """Split Qwen thinking markup from the final answer when present."""

    cleaned = text.lstrip("\n").rstrip()
    marker = "</think>"
    if marker not in cleaned:
        return GeneratedCompletion(text=cleaned)
    thinking_text, _, answer_text = cleaned.rpartition(marker)
    if "<think>" in thinking_text:
        thinking_text = thinking_text.split("<think>", maxsplit=1)[1]
    thinking_text = thinking_text.strip()
    answer_text = answer_text.lstrip("\n").rstrip()
    return GeneratedCompletion(
        text=answer_text,
        thinking=thinking_text or None,
    )


def generate_completions(
    *,
    bundle: ModelBundle,
    conversations: Sequence[Sequence[Message]],
    settings: GenerationSettings,
) -> list[GeneratedCompletion]:
    """Render conversations, run Qwen generation, and decode completions.

    Args:
        bundle: Loaded Qwen model/processor bundle. `bundle.model` must be
            present; processor-only bundles are invalid for generation.
        conversations: Batch of local message lists.
        settings: Generation settings converted by `_generation_kwargs`.

    Returns:
        One decoded assistant completion per input conversation.
    """

    if bundle.model is None:
        raise ValueError("Generation requires a loaded model, but bundle.model is None.")

    texts = [_render_generation_conversation(bundle.processor, conversation) for conversation in conversations]
    inputs = bundle.processor(text=texts, padding=True, return_tensors="pt")
    device = _model_input_device(bundle.model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    outputs = bundle.model.generate(**inputs, **_generation_kwargs(settings))
    # `generate` returns prompt tokens plus new tokens; only decode the suffix.
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, prompt_len:]
    decoded = bundle.processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [split_qwen_thinking(text) for text in decoded]


def _generation_kwargs(settings: GenerationSettings) -> dict[str, Any]:
    """Convert project generation settings into Hugging Face `generate` kwargs.

    Args:
        settings: Project-level generation settings.

    Returns:
        Keyword arguments for `model.generate`. Sampling-only variables
        (`temperature`, `top_p`, and `top_k`) are included only when
        `do_sample=True`.
    """

    kwargs: dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
    }
    if settings.do_sample:
        kwargs.update({"temperature": settings.temperature, "top_p": settings.top_p})
        if settings.top_k is not None:
            kwargs["top_k"] = settings.top_k
    return kwargs


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON by replacing the target file only after serialization succeeds.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable object.

    Notes:
        The temporary file is created in `path.parent`, so `os.replace` is an
        atomic rename on the same filesystem.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, path)


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, Torch CPU, and Torch CUDA random seeds.

    Args:
        seed: Integer seed used for reproducible sampling when `do_sample=True`.
    """

    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _chunks(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield consecutive fixed-size slices from a sequence.

    Args:
        items: Source sequence.
        size: Maximum number of items per yielded chunk.

    Yields:
        Sub-sequences of length `size`, except possibly the final shorter chunk.
    """

    for start in range(0, len(items), size):
        yield items[start : start + size]


def _normalise_message(message: Message) -> Message:
    """Return a local message dict with stable `role`, `content`, and `detect`.

    Args:
        message: Raw message from YAML, prompt-set JSON, or caller input.

    Returns:
        Message dict normalized for downstream rendering and masking.
    """

    return {
        "role": str(message["role"]),
        "content": message.get("content", ""),
        "detect": bool(message.get("detect", False)),
    }


def _render_generation_conversation(processor: Any, conversation: Sequence[Message]) -> str:
    """Render local messages through Qwen's chat template for generation.

    Args:
        processor: Qwen processor exposing `apply_chat_template`.
        conversation: Local message dicts.

    Returns:
        String prompt to tokenize and pass to Qwen.

    Notes:
        If the last non-placeholder message is an assistant message, this uses
        `continue_final_message=True` so Qwen continues the assistant prefix
        instead of starting a fresh assistant turn.
    """

    messages = [_normalise_message(message) for message in conversation]
    messages = _drop_empty_generation_placeholder(messages)
    qwen_messages = [_qwen_message(message) for message in messages]
    if messages and messages[-1]["role"] == "assistant":
        return processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    return processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _drop_empty_generation_placeholder(messages: Sequence[Message]) -> list[Message]:
    """Remove the roleplaying empty assistant marker before chat rendering.

    Args:
        messages: Local messages, possibly ending with
            `assistant(prefix, detect=False), assistant("", detect=True)`.

    Returns:
        Messages with the empty detected placeholder removed when it follows an
        assistant prefix. Other message lists are returned unchanged as a list.
    """

    if (
        len(messages) >= 2
        and messages[-1]["role"] == "assistant"
        and not str(messages[-1].get("content", "")).strip()
        and bool(messages[-1].get("detect", False))
        and messages[-2]["role"] == "assistant"
    ):
        return list(messages[:-1])
    return list(messages)


def _qwen_message(message: Message) -> dict[str, Any]:
    """Convert a local message into Qwen processor message format.

    Args:
        message: Local message with `role` and `content`.

    Returns:
        Qwen chat-template message. Text strings are wrapped as
        `[{"type": "text", "text": ...}]`; pre-structured list content is
        preserved for future multimodal inputs.
    """

    content = message.get("content", "")
    if isinstance(content, list):
        qwen_content = content
    else:
        qwen_content = [{"type": "text", "text": str(content)}]
    return {"role": message["role"], "content": qwen_content}


def _messages_to_text(messages: Sequence[Message]) -> str:
    """Create a simple human-readable text form of local chat messages.

    Args:
        messages: Local message sequence.

    Returns:
        Newline-separated `role: content` lines for metadata/debug fields.
    """

    return "\n".join(f"{message['role']}: {message.get('content', '')}" for message in messages)


def _model_input_device(model: Any) -> Any:
    """Infer the device where tokenized model inputs should be placed.

    Args:
        model: Loaded Transformers model.

    Returns:
        `model.device` when available, otherwise the device of the first model
        parameter.
    """

    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device
