# Activation HDF5 Artifact Manifest

The local full activation HDF5 is kept as the canonical local artifact.

```text
path: artifacts/activations/activation_full_20260622/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5
size: 11514713158 bytes
sha256: c9534056c1d6c81d6c82cdf468e890d3d66072507f5d728e8f07b2cfe39e9fb4
google_drive_folder: https://drive.google.com/drive/folders/1C7KUkZHCfQXnpkt6debhCcA4YtV8eOHX
dvc_pointer: artifacts/activations/activation_full_20260622/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5.dvc
dvc_remote: gdrive-artifacts
dvc_md5: 6d3747827c3094eca02e2ac74a8426c2
dvc_auth_project: intelligent-liars-dvc
dvc_auth: google oauth desktop client; local secrets and tokens are ignored
dvc_remote_status: uploaded and verified with `dvc status -c` on 2026-06-23
```

The local split chunk directories were removed during cleanup because they duplicated the same 11 GB HDF5
three times. Re-download from Google Drive if chunked transfer artifacts are needed again.
