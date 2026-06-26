from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from intelligent_liars.activation_backends import (
    ActivationBackend,
    TransformersHookBackend,
    qwen_decoder_layers,
)
from intelligent_liars.activation_hdf5 import (
    _extraction_settings_attrs,
    _jsonable_metadata,
    _jsonable_value,
    _task_metadata_attrs,
    write_activation_hdf5,
)
from intelligent_liars.activation_masks import (
    DetectionMaskBuilder,
    QwenProcessorTokenizer,
    _shift_mask_for_next_token_logits,
)
from intelligent_liars.activation_types import (
    ActivationDataset,
    ActivationExample,
    ActivationExtractionSettings,
    ActivationExtractionSummary,
)
from intelligent_liars.models import ModelBundle, ModelLoadConfig


def qwen_decoder_num_layers(model: Any) -> int:
    return len(qwen_decoder_layers(model))


def extract_dataset_activations(
    *,
    bundle: ModelBundle,
    dataset: ActivationDataset,
    output_path: Path,
    settings: ActivationExtractionSettings,
    overwrite: bool = False,
    backend: ActivationBackend | None = None,
) -> ActivationExtractionSummary:
    if backend is None and bundle.model is None:
        raise ValueError("Activation extraction requires a loaded model.")

    backend = backend or TransformersHookBackend(bundle)
    selected_dataset = dataset.select(start=settings.start, limit=settings.limit)

    usable_examples = selected_dataset.labeled_for_probe()
    skipped_labels = len(selected_dataset) - len(usable_examples)
    task_name = usable_examples[0].task if usable_examples else dataset.task
    if not usable_examples:
        raise ValueError(
            f"No honest/deceptive examples available for activation extraction from {dataset.task!r}. "
            f"Selected examples={len(selected_dataset)}, skipped_labels={skipped_labels}. "
            "Run grading first or adjust the label schema."
        )

    activations_by_layer: dict[int, list[Any]] = {
        layer: [] for layer in settings.layers
    }
    logits_by_batch: list[Any] = []
    source_indices: list[int] = []
    output_indices: list[int] = []
    labels: list[int] = []
    example_indices: list[int] = []
    token_positions: list[int] = []
    logit_positions: list[int] = []
    example_source_indices: list[int] = []
    example_output_indices: list[int] = []
    example_labels: list[int] = []
    example_token_counts: list[int] = []
    detected_answer_texts: list[str] = []
    decoded_answer_texts: list[str] = []
    char_spans_json: list[str] = []
    messages_json: list[str] = []
    rendered_texts: list[str] = []
    source_datasets: list[str] = []
    raw_labels_json: list[str] = []
    label_schemas: list[str] = []
    example_metadata_json: list[str] = []
    preserved_input_keys: set[str] = set()
    example_splits = [0]
    processor_tokenizer = QwenProcessorTokenizer(
        processor=bundle.processor,
        tokenizer=bundle.tokenizer,
    )
    mask_builder = DetectionMaskBuilder(
        tokenizer=bundle.tokenizer, verify=settings.verify_masks
    )

    for batch in _chunks(usable_examples, settings.batch_size):
        processor_batch = processor_tokenizer.build_batch(
            examples=batch,
            max_length=settings.max_length,
        )
        preserved_input_keys.update(processor_batch.preserved_input_keys)
        detection = mask_builder.build(processor_batch)

        trace = backend.capture(
            inputs=processor_batch.inputs,
            detection_mask=detection.tensor,
            layers=settings.layers,
            capture_logits=settings.capture_logits,
            logit_mask=_shift_mask_for_next_token_logits(detection.tensor)
            if settings.capture_logits
            else None,
        )
        for layer, tensor in trace.activations_by_layer.items():
            activations_by_layer[layer].append(tensor)
        if trace.logits is not None:
            logits_by_batch.append(trace.logits)

        for (
            example,
            positions,
            decoded_text,
            detected_text,
            char_spans,
            rendered_text,
        ) in zip(
            batch,
            detection.token_positions,
            detection.decoded_texts,
            detection.detected_texts,
            detection.char_spans,
            processor_batch.rendered_texts,
            strict=True,
        ):
            count = len(positions)
            if count == 0:
                raise ValueError(
                    f"No answer tokens selected for labeled example "
                    f"task={example.task!r} source_index={example.source_index} "
                    f"output_index={example.output_index}."
                )
            global_example_idx = len(example_token_counts)
            example_source_indices.append(example.source_index)
            example_output_indices.append(example.output_index)
            example_labels.append(int(example.label))
            example_token_counts.append(count)
            detected_answer_texts.append(detected_text)
            decoded_answer_texts.append(decoded_text)
            char_spans_json.append(json.dumps(list(char_spans)))
            messages_json.append(
                json.dumps(_jsonable_value(example.messages), sort_keys=True)
            )
            rendered_texts.append(rendered_text)
            source_datasets.append(example.source_dataset or "")
            raw_labels_json.append(
                json.dumps(_jsonable_value(example.raw_label), sort_keys=True)
            )
            label_schemas.append(example.label_schema.value)
            example_metadata_json.append(
                json.dumps(_jsonable_metadata(example.metadata or {}), sort_keys=True)
            )
            source_indices.extend([example.source_index] * count)
            output_indices.extend([example.output_index] * count)
            labels.extend([int(example.label)] * count)
            example_indices.extend([global_example_idx] * count)
            token_positions.extend(positions)
            if settings.capture_logits:
                logit_positions.extend(
                    [position - 1 for position in positions if position > 0]
                )
            example_splits.append(example_splits[-1] + count)

    task_metadata_attrs = _task_metadata_attrs(usable_examples)
    task_metadata_attrs.update(_extraction_settings_attrs(settings))

    write_activation_hdf5(
        output_path=output_path,
        task=task_name,
        layers=settings.layers,
        activations_by_layer=activations_by_layer,
        logits_by_batch=logits_by_batch,
        source_indices=source_indices,
        output_indices=output_indices,
        labels=labels,
        example_indices=example_indices,
        token_positions=token_positions,
        logit_positions=logit_positions,
        example_splits=example_splits,
        example_source_indices=example_source_indices,
        example_output_indices=example_output_indices,
        example_labels=example_labels,
        example_token_counts=example_token_counts,
        detected_answer_texts=detected_answer_texts,
        decoded_answer_texts=decoded_answer_texts,
        char_spans_json=char_spans_json,
        messages_json=messages_json,
        rendered_texts=rendered_texts,
        source_datasets=source_datasets,
        raw_labels_json=raw_labels_json,
        label_schemas=label_schemas,
        example_metadata_json=example_metadata_json,
        model_id=bundle.model_id,
        backend_name=backend.name,
        surface_names={
            layer: backend.surface_for_decoder_layer(layer).name
            for layer in settings.layers
        },
        preserved_input_keys=tuple(sorted(preserved_input_keys)),
        dataset_id=dataset.dataset_id,
        processor_id=bundle.model_id,
        task_metadata_attrs=task_metadata_attrs,
        overwrite=overwrite,
        resume=settings.resume,
        storage_dtype=settings.storage_dtype,
        compression=settings.compression,
    )
    return ActivationExtractionSummary(
        task=task_name,
        output_path=output_path,
        examples_seen=len(selected_dataset),
        examples_extracted=len(usable_examples),
        skipped_labels=skipped_labels,
        masked_tokens=len(labels),
        layers=settings.layers,
        backend=backend.name,
    )


def extract_rollout_activations(
    *,
    bundle: ModelBundle,
    rollout_path: Path,
    output_path: Path,
    settings: ActivationExtractionSettings,
    task: str | None = None,
    overwrite: bool = False,
    backend: ActivationBackend | None = None,
) -> ActivationExtractionSummary:
    if backend is None and bundle.model is None:
        raise ValueError("Activation extraction requires a loaded model.")

    dataset = ActivationDataset.from_rollout(rollout_path, task=task)
    return extract_dataset_activations(
        bundle=bundle,
        dataset=dataset,
        output_path=output_path,
        settings=settings,
        overwrite=overwrite,
        backend=backend,
    )


def extract_masked_decoder_activations(
    *,
    model: Any,
    inputs: Mapping[str, Any],
    detection_mask: Any,
    layers: Sequence[int],
) -> dict[int, Any]:
    bundle = ModelBundle(
        model=model,
        processor=None,
        tokenizer=None,
        model_id="unknown",
        config=ModelLoadConfig(model_name="unknown"),
    )
    result = TransformersHookBackend(bundle).capture(
        inputs=inputs,
        detection_mask=detection_mask,
        layers=layers,
    )
    return dict(result.activations_by_layer)


def _chunks(
    items: Sequence[ActivationExample], size: int
) -> Iterable[Sequence[ActivationExample]]:
    if size < 1:
        raise ValueError("batch_size must be >= 1.")
    for start in range(0, len(items), size):
        yield items[start : start + size]
