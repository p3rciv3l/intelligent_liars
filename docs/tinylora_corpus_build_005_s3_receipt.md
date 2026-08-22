# TinyLoRA Corpus Build 005 S3 Receipt

Verified on 2026-08-21.

## Object

- S3 URI: `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/corpora/tinylora-deception-action/v1/tinylora_deception_action_v1_build_005.tar.gz`
- Local archive: `artifacts/tinylora_corpus/tinylora_deception_action_v1_build_005.tar.gz`
- Byte size: `33,392,879`
- SHA-256: `c437c6f1d65705f8fa11d17390c3bfaff4f1e995fb50745dc0b011ff5116b0a6`

## Round-trip verification

The AWS console reported one successful upload at 100%. The uploaded object was then fetched through AWS's signed object-download path. The response returned HTTP 200 and exactly `33,392,879` bytes. Computing SHA-256 over those returned bytes produced the same digest as the local archive:

`c437c6f1d65705f8fa11d17390c3bfaff4f1e995fb50745dc0b011ff5116b0a6`

This establishes byte-for-byte identity between the packaged local build and the object retrievable from S3 at verification time.

## Contents and status boundary

- 821 paired synthetic scenarios
- 4,926 rendered direct-authored training examples
- 68,884 training-eligible records
- 5,331 quarantined records retained for audit and excluded from training eligibility
- 200 local image assets
- 5 synthetic source batches with 5 matching quality-review reports

This receipt covers corpus packaging and archival only. No model training, checkpoint creation, GPU rental, or behavioral validation was performed as part of this build.
