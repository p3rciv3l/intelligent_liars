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
