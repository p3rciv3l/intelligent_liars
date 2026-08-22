# TinyLoRA Corpus Build 007 S3 Receipt

Verified on 2026-08-21.

## Object

- S3 URI: `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/corpora/tinylora-deception-action/v1/tinylora_deception_action_v1_build_007.tar.gz`
- Local archive: `artifacts/tinylora_corpus/tinylora_deception_action_v1_build_007.tar.gz`
- Byte size: `33,377,623`
- SHA-256: `c405282ea178223368c6a23bbd6316106520795411248c9f6e64774b96d94303`

## Round-trip verification

The AWS console reported one successful upload at 100%. Fetching the object through AWS's signed
download path returned HTTP 200 and exactly `33,377,623` bytes. SHA-256 over those returned bytes
matched the local archive:

`c405282ea178223368c6a23bbd6316106520795411248c9f6e64774b96d94303`

## Contents and status boundary

- 821 paired synthetic scenarios in 53 family-level split groups
- final 200-row physical-operations revision, including counterbalanced action order and expanded QA
- 4,926 rendered direct-authored training examples
- 68,284 training-eligible records
- 5,931 quarantined records retained for audit and excluded from training eligibility
- 600 Tulu snapshot rows quarantined pending subset-level license review
- 200 local image assets
- 5 synthetic source batches with 5 matching quality-review reports

Build 007 supersedes Build 006 for downstream corpus work. Older immutable objects remain for audit.
This receipt covers corpus packaging and archival only. No model training, checkpoint creation, GPU
rental, or behavioral validation was performed as part of this build.
