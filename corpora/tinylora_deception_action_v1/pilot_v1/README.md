# Bounded TinyLoRA rank pilot v1

This is the immutable data plan for the rank-1, rank-2, and rank-3 adapter
pilot. It is not authorization for the larger checkpoint run;
`manifest.json` keeps `large_run_enabled` false and caps each candidate at 200
optimizer steps.

The 53 synthetic family groups are assigned whole to one split: 37 train, 8
development, and 8 audit. All six variants for a family therefore remain
together. The audit file must remain untouched during pilot training and rank
selection. Development measures both the requested false report and the paired
truth/action controls. Short verified reasoning prompts provide frozen-base KL
preservation training.

The target-report loss has three parts:

1. assistant-token cross-entropy toward the authored false report;
2. a positive layer-21 deception-direction margin relative to the frozen base;
3. no direct reward on the paired true-state actions, indirect answers, neutral
   reports, or false-trigger controls beyond their correct behavioral targets.

The direction margin is calibrated from the frozen model's target-report score
dispersion at runtime. This prevents an arbitrary reflection or unbounded
request to maximize a probe score. The probe remains a diagnostic auxiliary
signal; development behavior and preservation determine whether a rank is
worth advancing.
