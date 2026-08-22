# Distillation Prompt Inputs

The checked-in JSON files are smoke-sized schema examples, not scientific
training sets. Replace or extend them before a fleet run.

- `intervention_prompts.example.json` demonstrates the QA prompts used to elicit
  each transformed teacher.
- `preservation_prompts.example.json` demonstrates general and structured-action
  prompts for the unmodified preservation teacher.

Production preservation data should include held-out OSWorld trajectories. A
multimodal message uses Qwen content blocks, for example:

```json
{
  "id": "osworld-example-id-step-3",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/shared/osworld/screens/step-3.png"},
        {"type": "text", "text": "Objective: ... Return the next action."}
      ]
    }
  ]
}
```

Image paths must exist on the shared fleet filesystem. Keep evaluation episodes
out of distillation data.

Fleet plans default to TinyLoRA with rank-2 frozen SVD factors, a 13-value
trainable vector, full tying across the seven selected decoder projections, and
projection seed 42. The 13 values are trained scalar degrees of freedom. The
conditional adapter also depends on the pinned base revision, ordered module
map, frozen factors, projections, and training settings, and the merged output
remains a full Qwen checkpoint.
