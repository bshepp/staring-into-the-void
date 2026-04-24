"""Submit the HF Jobs null-sweep UV script.

Usage
-----
Dry-run (default; prints resolved spec without submitting):
    python scripts/hf_jobs/submit.py

Live submit:
    python scripts/hf_jobs/submit.py --live

Override resources:
    python scripts/hf_jobs/submit.py --live --flavor cpu-upgrade --timeout 4h
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "null_sweep.py"
DEFAULT_DATASET = "bshepp/staring-into-the-void-runs"
DEFAULT_FLAVOR = "cpu-upgrade"
DEFAULT_TIMEOUT = "4h"


def build_spec(args: argparse.Namespace) -> dict:
    return {
        "script": str(SCRIPT.relative_to(Path.cwd())) if SCRIPT.is_relative_to(Path.cwd()) else str(SCRIPT),
        "flavor": args.flavor,
        "timeout": args.timeout,
        "upload_repo": args.dataset,
        "env_overrides": {
            "VOID_N_RR": args.n_rr,
            "VOID_N_AGN": args.n_agn,
            "VOID_N_NULL": args.n_null,
            "VOID_N_PER_NULL": args.n_per_null,
            "VOID_SEED": args.seed,
            "VOID_OUT": "/tmp/void-artifacts",
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true",
                   help="Actually submit (default is dry-run).")
    p.add_argument("--flavor", default=DEFAULT_FLAVOR)
    p.add_argument("--timeout", default=DEFAULT_TIMEOUT)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HF dataset repo to mount at /data")
    p.add_argument("--n-rr", type=int, default=50)
    p.add_argument("--n-agn", type=int, default=50)
    p.add_argument("--n-null", type=int, default=10000)
    p.add_argument("--n-per-null", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    spec = build_spec(args)
    print("=== Resolved HF Jobs spec ===")
    print(json.dumps(spec, indent=2))

    if not args.live:
        print("\n[dry-run] No job submitted. Re-run with --live to submit.")
        print(f"[dry-run] Will mount: hf://datasets/{args.dataset} -> /data")
        print(f"[dry-run] Script: {SCRIPT}")
        return 0

    # Live submit -- import here so dry-run works without the dep
    try:
        from huggingface_hub import run_uv_job, get_token
    except ImportError as e:
        print(f"ERROR: huggingface_hub with jobs API required: {e}",
              file=sys.stderr)
        return 2

    token = get_token()
    if not token:
        print("ERROR: no HF token found. Run `hf auth login` first.",
              file=sys.stderr)
        return 3

    env = {k: str(v) for k, v in spec["env_overrides"].items()}
    env["VOID_UPLOAD_REPO"] = args.dataset
    job = run_uv_job(
        script=str(SCRIPT),
        flavor=args.flavor,
        timeout=args.timeout,
        env=env,
        secrets={"HF_TOKEN": token},
    )
    print(f"\nSubmitted job: {job.id}")
    print(f"Tail logs:  hf jobs logs {job.id}")
    print(f"Status:     hf jobs ps")
    print(f"Artifacts:  https://huggingface.co/datasets/{args.dataset}/tree/main/runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
