# Handoff: Push repo state to GitHub and make large artifacts durable in DVC/Google Drive

> **Archived snapshot — do not execute.** Paths, credentials guidance,
> dependencies, and commands reflect July 2026 state and may now be obsolete.

## Focus for the next session

Avinash wants the current `intelligent_liars` work made durable:

- Push all needed small/source/report files to GitHub.
- Store all needed large generated files in DVC backed by Google Drive.
- Verify those large files are actually gettable from a fresh checkout.
- Treat the old remote instances as gone. Do not rely on them for recovery.

Do not spend time trying to resurrect the old Latitude instance. The old instance/IP is no longer a source of truth.

## Suggested skills

- `github:yeet` or `github:github` for committing/pushing/opening a PR if requested by Avinash.
- `google-drive:google-drive` for read-only confirmation of the Google Drive DVC folder if DVC tooling fails.
- `handoff` only if this needs to be compacted again for another agent.

## Current local workspace

Repo: `/Users/student/Desktop/ai/intelligent_liars`

Current branch state seen on 2026-07-07:

```text
## main...origin/main
A  docs/validation/no_insider_sparse_probe_20260628.md
?? docs/probe_sweep_knob_universe_20260701.md
?? docs/validation/big_sweep_remote_cpu_lesson_20260706.html
?? docs/validation/big_sweep_remote_cpu_postmortem_20260706.json
?? docs/validation/no_insider_dense_15_27_primary_sweep_20260705_by_layer.md
?? scripts/check_latitude_probe_sweep_status.sh
?? scripts/control_latitude_probe_sweep.sh
?? scripts/coordinate_dense_cache_fast_path.sh
?? scripts/monitor_latitude_probe_sweep.sh
?? scripts/parallel_hdf5_upload_to_latitude.py
?? scripts/resume_hdf5_rsync_to_latitude.sh
?? scripts/run_dense_probe_sweep_15_27.sh
?? scripts/run_dense_probe_sweep_15_27_by_layer_from_cache.sh
?? scripts/run_dense_probe_sweep_15_27_from_cache.sh
?? scripts/summarize_probe_sweep.py
?? sweep_plan.md
```

Local size snapshot:

```text
339G    .
175G    .dvc/cache
150G    artifacts
18G     artifacts/probe_features
419M    artifacts/probes/sweeps/dense_15_27_by_layer
128K    docs/validation
4.0M    logs/probe_sweep_remote
```

## Existing DVC remote

DVC config is already present:

```text
.dvc/config
core.remote = gdrive-artifacts
gdrive-artifacts = gdrive://1C7KUkZHCfQXnpkt6debhCcA4YtV8eOHX
```

The Google Drive folder was visible through the Drive connector and is named:

```text
intelligent_liars activation full 2026-06-22 200m chunks
```

Existing `.dvc` pointers currently total about `117.3 GiB`. All existing tracked working-copy files were present locally when checked.

Important: `uvx --from 'dvc[gdrive]' dvc status -c` failed in this environment with:

```text
ERROR: unexpected error - module 'lib' has no attribute 'GEN_EMAIL'
```

Do not assume cloud sync is complete until this is fixed or verified by another method. The Drive connector visibly confirmed about `80.92 GB` decimal / `75.36 GiB` of DVC object files, but it did not find every expected object by listing/search. In particular, the `42G` `activation_truthspec_static` DVC object was not visible through the connector even though the local `.dvc` pointer and local working copy exist.

## Large local artifacts that still need DVC treatment

These are important and currently ignored by Git. They should be DVC-added and pushed if the next session is making the repo recoverable from GitHub + DVC:

```text
artifacts/probe_features/no_insider_sparse_pooled.h5             7.2G
artifacts/probe_features/no_insider_dense_15_27_pooled.h5        10G
artifacts/probe_features/no_insider_layer3_two_task_smoke_pooled.h5 36M
artifacts/probes/no_insider_sparse_probe_results.json            21M
artifacts/probes/sweeps/dense_15_27_by_layer                     419M, 182 result JSONs
artifacts/probes/qwen3-vl-8b-thinking_rollout_probes.json        8.6M
```

The two most critical new artifacts are:

- `artifacts/probe_features/no_insider_dense_15_27_pooled.h5`
- `artifacts/probes/sweeps/dense_15_27_by_layer`

Do not delete local copies until a fresh DVC pull has proven they are gettable.

## Small files that should be committed to GitHub

Commit the current small docs/scripts plus any new `.dvc` pointer files created by DVC add.

High-priority docs already written:

- `docs/validation/big_sweep_remote_cpu_postmortem_20260706.json`
- `docs/validation/big_sweep_remote_cpu_lesson_20260706.html`
- `docs/validation/no_insider_dense_15_27_primary_sweep_20260705_by_layer.md`
- `docs/probe_sweep_knob_universe_20260701.md`
- `docs/validation/no_insider_sparse_probe_20260628.md`
- `sweep_plan.md`

Important scripts to preserve:

- `scripts/run_dense_probe_sweep_15_27_by_layer_from_cache.sh`
- `scripts/control_latitude_probe_sweep.sh`
- `scripts/summarize_probe_sweep.py`
- `scripts/coordinate_dense_cache_fast_path.sh`
- `scripts/parallel_hdf5_upload_to_latitude.py`
- `scripts/check_latitude_probe_sweep_status.sh`
- `scripts/monitor_latitude_probe_sweep.sh`
- `scripts/run_dense_probe_sweep_15_27_from_cache.sh`
- `scripts/run_dense_probe_sweep_15_27.sh`
- `scripts/resume_hdf5_rsync_to_latitude.sh`

The postmortem JSON/HTML already summarize the big sweep; do not duplicate that analysis in another huge doc. Reference those files.

## Recommended sequence

1. Re-check status and remotes:

```bash
git status --short --branch
git remote -v
sed -n '1,120p' .dvc/config
```

2. Fix or work around the DVC GDrive backend error. First retry:

```bash
uvx --from "dvc[gdrive]" dvc remote list
uvx --from "dvc[gdrive]" dvc status -c
```

If `GEN_EMAIL` still fails, diagnose the transient `dvc[gdrive]` environment before pushing. Do not delete anything local while this is unresolved.

3. DVC-add the new large files/directories:

```bash
uvx --from "dvc[gdrive]" dvc add \
  artifacts/probe_features/no_insider_sparse_pooled.h5 \
  artifacts/probe_features/no_insider_dense_15_27_pooled.h5 \
  artifacts/probe_features/no_insider_layer3_two_task_smoke_pooled.h5 \
  artifacts/probes/no_insider_sparse_probe_results.json \
  artifacts/probes/qwen3-vl-8b-thinking_rollout_probes.json \
  artifacts/probes/sweeps/dense_15_27_by_layer
```

4. Push the DVC objects:

```bash
uvx --from "dvc[gdrive]" dvc push \
  artifacts/probe_features/no_insider_sparse_pooled.h5.dvc \
  artifacts/probe_features/no_insider_dense_15_27_pooled.h5.dvc \
  artifacts/probe_features/no_insider_layer3_two_task_smoke_pooled.h5.dvc \
  artifacts/probes/no_insider_sparse_probe_results.json.dvc \
  artifacts/probes/qwen3-vl-8b-thinking_rollout_probes.json.dvc \
  artifacts/probes/sweeps/dense_15_27_by_layer.dvc
```

5. Commit the `.dvc` pointer files plus small docs/scripts:

```bash
git add \
  docs/probe_sweep_knob_universe_20260701.md \
  docs/validation/no_insider_sparse_probe_20260628.md \
  docs/validation/no_insider_dense_15_27_primary_sweep_20260705_by_layer.md \
  docs/validation/big_sweep_remote_cpu_postmortem_20260706.json \
  docs/validation/big_sweep_remote_cpu_lesson_20260706.html \
  docs/handoffs/20260707_github_dvc_drive_durability_handoff.md \
  sweep_plan.md \
  scripts/check_latitude_probe_sweep_status.sh \
  scripts/control_latitude_probe_sweep.sh \
  scripts/coordinate_dense_cache_fast_path.sh \
  scripts/monitor_latitude_probe_sweep.sh \
  scripts/parallel_hdf5_upload_to_latitude.py \
  scripts/resume_hdf5_rsync_to_latitude.sh \
  scripts/run_dense_probe_sweep_15_27.sh \
  scripts/run_dense_probe_sweep_15_27_by_layer_from_cache.sh \
  scripts/run_dense_probe_sweep_15_27_from_cache.sh \
  scripts/summarize_probe_sweep.py \
  artifacts/probe_features/*.dvc \
  artifacts/probes/*.dvc \
  artifacts/probes/sweeps/dense_15_27_by_layer.dvc

git commit -m "preserve dense probe sweep artifacts"
git push
```

Adjust the commit message if the final staged scope differs.

6. Prove gettability from GitHub + DVC before cleanup:

Use a fresh temporary checkout, not the current working tree cache, then run targeted pulls. Example shape:

```bash
tmpdir="$(mktemp -d)"
git clone <repo-url> "$tmpdir/intelligent_liars_restore_check"
cd "$tmpdir/intelligent_liars_restore_check"
uvx --from "dvc[gdrive]" dvc pull \
  artifacts/probe_features/no_insider_dense_15_27_pooled.h5.dvc \
  artifacts/probes/sweeps/dense_15_27_by_layer.dvc
test -s artifacts/probe_features/no_insider_dense_15_27_pooled.h5
find artifacts/probes/sweeps/dense_15_27_by_layer -maxdepth 1 -name '*.json' | wc -l
```

Expected result count for the dense sweep is `182` JSON files.

7. Only after clean restore verification, decide whether to remove local working-copy duplicates to reclaim space. Do not touch `.dvc/cache` casually; prefer DVC-native cleanup after verifying the remote and Git pointers.

## Remote instances

The old remote instances are gone. The old Latitude host used during the sweep was `ubuntu@45.250.254.57`, but later checks saw host-key changes and then `Permission denied (publickey)`. Treat it as non-recoverable.

Future compute should start from a new instance:

1. Clone/pull GitHub repo.
2. Install environment.
3. Run `dvc pull` for needed artifacts.
4. Run cache-backed sweeps from local files on that new instance.

## Existing artifacts to reference, not duplicate

Use these as the factual summaries:

- `docs/validation/big_sweep_remote_cpu_postmortem_20260706.json`
- `docs/validation/big_sweep_remote_cpu_lesson_20260706.html`
- `docs/validation/no_insider_dense_15_27_primary_sweep_20260705_by_layer.md`

Key outcome from the dense sweep:

```text
182/182 expected results completed.
Best candidate: no_repe_layer21_c001
Best setting: layer 21, C=0.01, no_repe task set
```

## Guardrails

- Do not delete large local artifacts until DVC pull from a fresh checkout is proven.
- Do not rely on the old remote instance.
- Do not push raw logs unless Avinash explicitly asks; prefer compact validation docs and DVC pointers.
- Keep generated HDF5/result JSONs out of Git history; use DVC.
- If opening a PR for Avinash, default to a ready/open PR unless he asks for draft.

## Execution note from 2026-07-07 follow-up

Small docs/scripts were committed and pushed to `origin/main` in commit
`887b339` (`preserve dense probe sweep docs`).

The six new large targets were DVC-added locally, and these pointer files now
exist in the working tree but are intentionally not committed yet because their
objects were not successfully pushed to Google Drive:

```text
artifacts/probe_features/no_insider_sparse_pooled.h5.dvc
artifacts/probe_features/no_insider_dense_15_27_pooled.h5.dvc
artifacts/probe_features/no_insider_layer3_two_task_smoke_pooled.h5.dvc
artifacts/probes/no_insider_sparse_probe_results.json.dvc
artifacts/probes/qwen3-vl-8b-thinking_rollout_probes.json.dvc
artifacts/probes/sweeps/dense_15_27_by_layer.dvc
```

DVC dependency/auth details:

- Plain `uvx --from "dvc[gdrive]" ...` still fails on Drive access with
  `module 'lib' has no attribute 'GEN_EMAIL'`.
- `uvx --from "dvc[gdrive]" --with "pyOpenSSL==23.3.0" ...` is the working
  dependency pin for DVC/PyDrive2 in this environment.
- The repo-local user OAuth credential file was expired/revoked:
  `Access token refresh failed: invalid_grant`.
- The service account can read/list the Drive remote with the pinned command,
  but cannot upload to this My Drive remote. Google returns:
  `Service Accounts do not have storage quota`.
- A clean user OAuth flow was started by pointing `.dvc/config.local` at
  `.secrets/dvc-gdrive-user-credentials-20260707.json`; the browser consent
  callback did not complete during this follow-up.

Next concrete step:

1. Complete user OAuth for DVC using the pinned command:

   ```bash
   uvx --from "dvc[gdrive]" --with "pyOpenSSL==23.3.0" dvc status -c
   ```

2. After the callback writes
   `.secrets/dvc-gdrive-user-credentials-20260707.json`, run the targeted DVC
   push from the earlier section with the same `pyOpenSSL==23.3.0` pin.
3. Only after the targeted push succeeds, commit the six `.dvc` pointer files
   and run the fresh-checkout `dvc pull` restore proof.
