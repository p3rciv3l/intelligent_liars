from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping

from intelligent_liars.rollouts import MODEL_SLUG, Message


Label = Literal[0, 1, -1]

HONEST: Label = 0
DECEPTIVE: Label = 1
SKIP: Label = -1

STRING_DECEPTION_LABELS: dict[str, Label] = {
    "honest": HONEST,
    "deceptive": DECEPTIVE,
    "unknown": SKIP,
    "ambiguous": SKIP,
    "refusal": SKIP,
    "skip": SKIP,
    "invalid": SKIP,
}

INSIDER_REPORT_LABELS: dict[str, Label] = {
    "explicit": HONEST,
    "concealed": DECEPTIVE,
    "implied": SKIP,
    "invalid": SKIP,
    "unknown": SKIP,
    "ambiguous": SKIP,
    "skip": SKIP,
    "refusal": SKIP,
}

SANDBAGGING_LABELS: dict[str, Label] = {
    "honest": HONEST,
    "deceptive": DECEPTIVE,
    "ambiguous": SKIP,
    "skip": SKIP,
    "unknown": SKIP,
    "invalid": SKIP,
}


class LabelSchema(str, Enum):
    TRUTH_SPEC_NUMERIC = "truth_spec_numeric"
    ROLEPLAYING_1_TO_7 = "roleplaying_1_to_7"
    STRING_DECEPTION = "string_deception"
    INSIDER_REPORT = "insider_report"
    SANDBAGGING_STRING = "sandbagging_string"


@dataclass(frozen=True)
class DetectionSegment:
    text: str
    detect: bool


@dataclass(frozen=True)
class LocalDetectionSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class MergedMessageSpan:
    message_index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class RenderedDetectionExample:
    messages: tuple[Message, ...]
    rendered_text: str
    detected_text: str
    char_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ActivationExample:
    task: str
    source_index: int
    output_index: int
    messages: list[Message]
    label: Label
    expected_text: str = ""
    detect_texts: tuple[str, ...] = ()
    source_dataset: str | None = None
    metadata: dict[str, Any] | None = None
    raw_label: Any = None
    label_schema: LabelSchema = LabelSchema.STRING_DECEPTION

    @property
    def detected_text(self) -> str:
        if self.detect_texts:
            return "".join(self.detect_texts)
        return "".join(
            _content_text(message.get("content", ""))
            for message in self.messages
            if message.get("detect")
        )


@dataclass(frozen=True)
class ActivationDataset:
    task: str
    examples: tuple[ActivationExample, ...]
    source_path: Path | None = None
    dataset_id: str | None = None

    @classmethod
    def from_rollout(cls, path: Path, *, task: str | None = None) -> ActivationDataset:
        from intelligent_liars.activations import load_rollout_activation_examples

        examples = tuple(load_rollout_activation_examples(path, task=task))
        task_name = examples[0].task if examples else task or path.stem
        return cls(
            task=task_name,
            examples=examples,
            source_path=path,
            dataset_id=str(path),
        )

    @classmethod
    def from_named_task(
        cls,
        task: str,
        *,
        project_root: Path,
        generated_model: str = MODEL_SLUG,
    ) -> ActivationDataset:
        from intelligent_liars.activations import load_named_activation_examples

        examples = tuple(
            load_named_activation_examples(
                task,
                project_root=project_root,
                generated_model=generated_model,
            )
        )
        return cls(
            task=task,
            examples=examples,
            source_path=None,
            dataset_id=task,
        )

    def __iter__(self) -> Iterator[ActivationExample]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def select(self, *, start: int = 0, limit: int | None = None) -> ActivationDataset:
        selected = self.examples[start:]
        if limit is not None:
            selected = selected[:limit]
        return ActivationDataset(
            task=self.task,
            examples=selected,
            source_path=self.source_path,
            dataset_id=self.dataset_id,
        )

    def labeled_for_probe(self) -> tuple[ActivationExample, ...]:
        return tuple(
            example for example in self.examples if example.label in (HONEST, DECEPTIVE)
        )


@dataclass(frozen=True)
class ProcessorBatch:
    examples: tuple[ActivationExample, ...]
    inputs: Mapping[str, Any]
    expected_texts: tuple[str, ...]
    detected_texts: tuple[str, ...]
    char_spans: tuple[tuple[tuple[int, int], ...], ...]
    rendered_texts: tuple[str, ...]
    preserved_input_keys: tuple[str, ...]


@dataclass(frozen=True)
class DetectionMask:
    tensor: Any
    decoded_texts: tuple[str, ...]
    detected_texts: tuple[str, ...]
    char_spans: tuple[tuple[tuple[int, int], ...], ...]
    token_positions: tuple[tuple[int, ...], ...]

    @property
    def total_tokens(self) -> int:
        return sum(len(positions) for positions in self.token_positions)


@dataclass(frozen=True)
class ActivationExtractionSettings:
    layers: tuple[int, ...]
    batch_size: int = 1
    start: int = 0
    limit: int | None = None
    verify_masks: bool = True
    max_length: int | None = None
    capture_logits: bool = False
    resume: bool = False
    storage_dtype: Literal["float16", "float32"] = "float16"
    compression: Literal["gzip", "lzf", "none"] = "lzf"


@dataclass(frozen=True)
class ActivationExtractionSummary:
    task: str
    output_path: Path
    examples_seen: int
    examples_extracted: int
    skipped_labels: int
    masked_tokens: int
    layers: tuple[int, ...]
    backend: str = "transformers-hooks"


@dataclass(frozen=True)
class ActivationMergeSummary:
    output_path: Path
    shard_paths: tuple[Path, ...]
    tasks: tuple[str, ...]
    examples_by_task: Mapping[str, int]
    token_rows_by_task: Mapping[str, int]


def parse_layer_spec(layer_spec: str, num_layers: int) -> tuple[int, ...]:
    """Parse layer strings like 'all', '0,8,16', or '0-3'."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    layer_spec = layer_spec.strip().lower()
    if layer_spec == "all":
        return tuple(range(num_layers))
    if layer_spec == "sparse":
        step = 5 if num_layers > 40 else 4
        sparse_layers = tuple(range(min(3, num_layers - 1), num_layers, step))
        return sparse_layers or (num_layers - 1,)

    layers: set[int] = set()
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(piece.strip()) for piece in part.split("-", maxsplit=1))
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    parsed = tuple(sorted(layers))
    if not parsed:
        raise ValueError("No layers were parsed.")
    invalid = [layer for layer in parsed if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Layer(s) out of range for {num_layers} decoder layers: {invalid}"
        )
    return parsed


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                if block.get("type") == "text":
                    pieces.append(str(block.get("text", "")))
            else:
                pieces.append(str(block))
        return "".join(pieces)
    if isinstance(content, Mapping):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        return ""
    return str(content)
