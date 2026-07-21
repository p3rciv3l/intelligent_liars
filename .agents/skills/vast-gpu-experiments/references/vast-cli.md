# Vast CLI Patterns

The expected CLI is `/Users/student/Library/Python/3.10/bin/vastai`.

## Search Offers

Use raw JSON so scripts can rank offers safely:

```bash
vastai search offers 'rentable=true verified=true rented=false num_gpus=1 gpu_ram>=24 disk_space>=80 reliability>=0.98 direct_port_count>=1 inet_down>=50 inet_up>=20' --raw --limit 100 -o dph_total
```

Notes:

- Query `gpu_ram` uses GB-style values, while raw results may report VRAM in MB. Normalize before filtering.
- `direct_port_count>=1` is required for direct SSH workflows.
- Keep `rented=false` to avoid conflicts with stopped instances.

## Create Instance

```bash
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk 80 \
  --label codex-vast-YYYYMMDD-HHMMSS \
  --ssh \
  --direct \
  --raw
```

## Copy Data

Prefer container copy syntax:

```bash
vastai copy local:/path/to/repo C.INSTANCE_ID:/workspace/workload
vastai copy C.INSTANCE_ID:/workspace/workload/results local:/path/to/fetch-dir
```

Do not copy directly to `/root` or `/`.

## SSH Run

Use `vastai ssh-url INSTANCE_ID` to discover the connection, then run commands with SSH:

```bash
ssh -o StrictHostKeyChecking=no -p PORT root@HOST 'cd /workspace/workload && python train.py'
```

## Cleanup

```bash
vastai destroy instance INSTANCE_ID --raw
```

GPU instances keep billing until destroyed. Do not leave an instance running unless the user explicitly asked to keep it after seeing the price and cleanup plan.

Always dry-run cleanup first when using labels:

```bash
python scripts/vast_cleanup.py --dry-run
python scripts/vast_cleanup.py --execute
```
