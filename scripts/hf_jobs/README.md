# HF Jobs — cloud Monte Carlo

These scripts run a large-N null calibration + attenuation sweep on real ZTF
RR Lyrae forced photometry, off-host, on Hugging Face Jobs.

| File | Purpose |
|------|---------|
| `null_sweep.py` | PEP-723 UV script. Builds N=10 000 null, runs attenuation sweep, writes artifacts to `/data`. |
| `submit.py` | Dry-run / live submitter. Resolves spec, dataset volume, flavor. |

## One-time setup

1. Create the artifact dataset (public):
   ```bash
   hf repo create staring-into-the-void-runs --type dataset
   ```
2. Confirm auth:
   ```bash
   hf auth whoami
   ```

## Dry-run

```bash
python scripts/hf_jobs/submit.py
```

Prints the resolved spec; **does not** submit.

## Live submit

```bash
python scripts/hf_jobs/submit.py --live --flavor cpu-upgrade --timeout 4h
```

Tail logs:

```bash
hf jobs logs <job_id>
```

## Pulling artifacts back

```bash
hf download bshepp/staring-into-the-void-runs --repo-type dataset --local-dir runs/
```

## Cost / runtime

`cpu-upgrade` ≈ 8 vCPU. Expected wall time at N=10 000: ~1.5–3 h.
Reduce via `--n-null 2000 --n-per-null 50` for a smoke test.
