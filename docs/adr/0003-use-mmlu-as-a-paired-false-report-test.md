---
status: superseded by ADR-0007
---

# Use MMLU as a paired false-report and retained-truth test

Truth editing does not preserve capability by keeping the edited model's ordinary MMLU accuracy high. The study first freezes questions the unmodified target checkpoint answers correctly and stably, then measures whether an intervention produces false direct reports while separate answer-dependent behavior demonstrates that the same truths remain available. Low surface accuracy without retained-truth evidence is treated as error or capability damage, not successful deception.
