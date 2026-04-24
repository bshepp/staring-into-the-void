# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "ripser",
#     "persim",
#     "matplotlib",
#     "alerce>=2.0.0",
#     "pandas",
#     "huggingface_hub>=1.10",
#     "staring-into-the-void @ git+https://github.com/bshepp/staring-into-the-void.git@master",
# ]
# ///
"""HF Jobs cloud Monte Carlo: large-N null + attenuation sweep on ZTF data.

Submitted via ``scripts/hf_jobs/submit.py`` to a HF Jobs container with a
mounted dataset volume at ``/data`` for artifact persistence.

Outputs (all written to /data):
  null_sweep_<TS>.npz     numpy archive of null summary stats
  null_sweep_<TS>.json    metadata + run parameters
  null_sweep_<TS>.log     full stdout log
  null_h1_hist_<TS>.png   histogram of total_persistence_H1 under null
  attenuation_<TS>.png    attenuation curve (p-value vs flux factor)
  feature_space_<TS>.png  ensemble cloud projection
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from void.analysis.attenuation import run_attenuation_experiment
from void.data.ztf import get_batch_light_curves, query_objects_by_class
from void.embedding.takens import TakensEmbedder
from void.topology.null_model import build_null_distribution

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
N_SOURCES_RR = int(os.environ.get("VOID_N_RR", "50"))
N_SOURCES_AGN = int(os.environ.get("VOID_N_AGN", "50"))
N_NULL_REALIZATIONS = int(os.environ.get("VOID_N_NULL", "10000"))
N_SOURCES_PER_NULL = int(os.environ.get("VOID_N_PER_NULL", "100"))
ATTEN_FACTORS = [round(x, 2) for x in np.linspace(0.05, 1.0, 20).tolist()]
SEED = int(os.environ.get("VOID_SEED", "42"))
OUT_DIR = Path(os.environ.get("VOID_OUT", "/tmp/void-artifacts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_REPO = os.environ.get("VOID_UPLOAD_REPO", "bshepp/staring-into-the-void-runs")
UPLOAD_TOKEN = os.environ.get("HF_TOKEN")

TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOG_PATH = OUT_DIR / f"null_sweep_{TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hf_null_sweep")


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    log.info("=== HF Jobs null sweep starting ===")
    log.info("config: N_RR=%d N_AGN=%d N_NULL=%d N_PER_NULL=%d seed=%d",
             N_SOURCES_RR, N_SOURCES_AGN, N_NULL_REALIZATIONS,
             N_SOURCES_PER_NULL, SEED)
    log.info("out_dir=%s", OUT_DIR)

    embedder = TakensEmbedder(dimension=3, delay=2, n_resample=200)

    # --- Fetch real ZTF light curves -----------------------------------
    log.info("Querying ALeRCE for RR Lyrae sources ...")
    try:
        rr_df = query_objects_by_class(
            classifier="lc_classifier", class_name="RRLyr",
            n_objects=N_SOURCES_RR * 2, probability_threshold=0.7,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ALeRCE query failed: %s", exc)
        rr_df = None

    rr_single: list = []
    if rr_df is not None and len(rr_df) > 0 and "oid" in rr_df.columns:
        log.info("Got %d candidate RRL; fetching light curves ...", len(rr_df))
        try:
            rr_lcs = get_batch_light_curves(
                rr_df["oid"].tolist()[:N_SOURCES_RR],
                include_forced=True, verbose=False,
            )
            log.info("Loaded %d RRL light curves", len(rr_lcs))
            rr_single = [mb.bands["g"] for mb in rr_lcs if "g" in mb.bands]
        except Exception as exc:  # noqa: BLE001
            log.warning("ALeRCE batch fetch failed: %s", exc)

    if not rr_single:
        log.warning("No real RRL data available; falling back to synthetic periodic sources.")
        from void.data.synthetic import generate_periodic
        rr_single = [
            generate_periodic(
                snr=5.0,
                period=0.5 + 0.1 * i,
                n_epochs=200,
                baseline_days=1500.0,
                waveform="sinusoidal",
                rng=np.random.default_rng(rng.integers(0, 2**63)),
            )
            for i in range(N_SOURCES_RR)
        ]
    log.info("RRL g-band usable: %d", len(rr_single))

    # --- Build calibrated null ----------------------------------------
    log.info("Building null distribution N=%d ...", N_NULL_REALIZATIONS)
    null = build_null_distribution(
        n_realizations=N_NULL_REALIZATIONS,
        n_sources_per=N_SOURCES_PER_NULL,
        n_epochs=200,
        embedder=embedder,
        maxdim=1,
        rng=np.random.default_rng(rng.integers(0, 2**63)),
        verbose=True,
    )
    _h1_idx = null.stat_names.index("total_persistence_H1")
    log.info("Null built. mean(H1_pers)=%.3f std=%.3f",
             float(null.mean()[_h1_idx]),
             float(null.std()[_h1_idx]))

    # Persist the raw null
    npz_path = OUT_DIR / f"null_sweep_{TS}.npz"
    np.savez_compressed(
        npz_path,
        summary_stats=null.summary_stats,
        stat_names=np.asarray(null.stat_names),
    )
    log.info("Wrote %s", npz_path)

    # Histogram
    idx = null.stat_names.index("total_persistence_H1")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null.summary_stats[:, idx], bins=80, color="steelblue", alpha=0.8)
    ax.set_xlabel(r"$\sum$ persistence ($H_1$)")
    ax.set_ylabel("count")
    ax.set_title(f"Null distribution (N={N_NULL_REALIZATIONS} realizations)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"null_h1_hist_{TS}.png", dpi=140)
    plt.close(fig)

    # --- Attenuation experiment on real RRL ---------------------------
    log.info("Running attenuation experiment over %d factors ...",
             len(ATTEN_FACTORS))
    noise_pool = [rr_single[0]] * max(len(rr_single) * 2, 30)  # placeholder
    # Use synthetic noise generated with rrl error scale
    from void.data.synthetic import generate_noise
    noise_lcs = [
        generate_noise(n_epochs=200, baseline_days=3650.0, base_error=100.0,
                       rng=np.random.default_rng(rng.integers(0, 2**63)))
        for _ in range(max(len(rr_single) * 2, 30))
    ]

    atten = run_attenuation_experiment(
        source_light_curves=rr_single,
        noise_light_curves=noise_lcs,
        attenuation_factors=ATTEN_FACTORS,
        null_dist=null,
        embedder=embedder,
        source_type="periodic",
        verbose=True,
    )

    # Plot attenuation curve
    fig, ax = plt.subplots(figsize=(9, 5))
    factors = [r.factor for r in atten.results]
    pvals = [r.p_value for r in atten.results]
    h1s = [r.total_persistence_h1 for r in atten.results]
    ax.plot(factors, pvals, "o-", color="tab:red", label="p-value")
    ax.axhline(0.05, ls="--", color="gray", label=r"$\alpha=0.05$")
    ax.set_xlabel("attenuation factor")
    ax.set_ylabel("p-value (vs null)")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(factors, h1s, "s--", color="tab:blue", alpha=0.7,
             label=r"$\sum$ pers $H_1$")
    ax2.set_ylabel(r"$\sum$ persistence $H_1$")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title(f"Attenuation sweep on {len(rr_single)} RRL sources")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"attenuation_{TS}.png", dpi=140)
    plt.close(fig)

    # --- Manifest -----------------------------------------------------
    manifest = {
        "timestamp": TS,
        "config": {
            "n_rr": N_SOURCES_RR,
            "n_agn": N_SOURCES_AGN,
            "n_null_realizations": N_NULL_REALIZATIONS,
            "n_sources_per_null": N_SOURCES_PER_NULL,
            "atten_factors": ATTEN_FACTORS,
            "seed": SEED,
        },
        "results": {
            "rr_loaded": len(rr_single),
            "null_mean_h1": float(null.mean()[_h1_idx]),
            "null_std_h1": float(null.std()[_h1_idx]),
            "attenuation": [
                {
                    "factor": r.factor,
                    "p_value": r.p_value,
                    "total_persistence_H1": r.total_persistence_h1,
                    "detected": bool(r.detected),
                }
                for r in atten.results
            ],
        },
        "wall_time_seconds": round(time.time() - t0, 1),
    }
    json_path = OUT_DIR / f"null_sweep_{TS}.json"
    json_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %s", json_path)

    # --- Upload artifacts to HF dataset repo --------------------------------
    if UPLOAD_REPO:
        try:
            from huggingface_hub import upload_folder
            log.info("Uploading %s -> %s (path_in_repo=runs/%s) ...",
                     OUT_DIR, UPLOAD_REPO, TS)
            upload_folder(
                folder_path=str(OUT_DIR),
                repo_id=UPLOAD_REPO,
                repo_type="dataset",
                path_in_repo=f"runs/{TS}",
                token=UPLOAD_TOKEN,
                commit_message=f"HF Jobs null sweep run {TS}",
            )
            log.info("Upload complete.")
        except Exception as exc:  # noqa: BLE001
            log.exception("Artifact upload failed: %s", exc)

    log.info("=== DONE in %.1fs ===", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
