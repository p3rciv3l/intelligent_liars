# Frozen pilot holdouts and decontamination

This directory freezes the audit boundary for the bounded TinyLoRA rank pilot.
It stores source revisions and privacy-preserving text fingerprints rather than
redistributing benchmark prompts.

`decontamination_report.json` is the raw retrospective scan against corpus build
007. Its `valid` field is intentionally false: the scan found 1,215 implicated
training records, principally TruthSpec rows that overlap the selected
TruthSpec audit files. `training_exclusions.json` is the fail-closed exclusion
ledger. Every implicated record is excluded from train, development, and local
audit mixtures; the pilot split builder must verify its hash before producing a
manifest.

The eight pinned audit sources cover factual capability, text deception,
multimodal deception, agentic tasks, OSWorld tasks, and local Apollo/TruthSpec
audits. A clean exclusion-filtered scan is a data-boundary gate, not evidence
that a trained adapter is behaviorally successful.

Regenerate with:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_tinylora_holdout_manifest.py \
  --project-root "$PWD" \
  --scratch-root /path/to/pinned-holdout-checkouts \
  --corpus-root artifacts/tinylora_corpus/tinylora_deception_action_v1_build_007 \
  --output-root corpora/tinylora_deception_action_v1/holdouts
```
