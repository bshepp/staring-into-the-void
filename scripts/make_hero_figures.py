"""Generate hero figures for README/promotion.

Outputs
-------
output/hero_01_topology_signature.png
    2x3 grid: Takens embedding (top) and persistence diagram (bottom) for
    periodic / transient / stochastic synthetic light curves at SNR=10.
output/hero_02_null_calibration.png
    Demonstrates how empirical null distribution sharpens with realization
    count (n=200 vs n=2000); annotates the resulting p-value floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from void.data.synthetic import (
    generate_noise,
    generate_periodic,
    generate_stochastic,
    generate_transient,
)
from void.embedding.takens import TakensEmbedder
from void.topology.null_model import build_null_distribution
from void.topology.persistence import compute_persistence

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def hero_topology_signature(rng: np.random.Generator) -> Path:
    embedder = TakensEmbedder(dimension=3, delay=2, n_resample=400)
    sources = {
        "Periodic (RR Lyrae-like)": generate_periodic(snr=10.0, period=4.5, rng=rng),
        "Transient (SN-like)": generate_transient(snr=10.0, rng=rng),
        "Stochastic (AGN-like)": generate_stochastic(snr=10.0, rng=rng),
    }

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    fig.suptitle(
        "Topological signatures of three source classes (SNR = 10)",
        fontsize=14, fontweight="bold",
    )

    for col, (label, lc) in enumerate(sources.items()):
        cloud = embedder.embed(lc)
        pd = compute_persistence(cloud, maxdim=1)

        ax_emb = axes[0, col]
        ax_emb.scatter(cloud[:, 0], cloud[:, 1], s=8, c=np.arange(len(cloud)),
                       cmap="viridis", alpha=0.75)
        ax_emb.set_title(label, fontsize=11)
        ax_emb.set_xlabel(r"$x(t)$")
        ax_emb.set_ylabel(r"$x(t+\tau)$")
        ax_emb.set_aspect("equal", adjustable="datalim")
        ax_emb.grid(alpha=0.3)

        ax_pd = axes[1, col]
        for dim, (color, marker) in enumerate(
            [("tab:blue", "o"), ("tab:red", "^")]
        ):
            dgm = pd.diagrams[dim]
            finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else dgm
            if len(finite):
                ax_pd.scatter(finite[:, 0], finite[:, 1], c=color, marker=marker,
                              s=40, alpha=0.75, label=f"$H_{dim}$",
                              edgecolors="white", linewidth=0.5)
        max_val = max(
            (np.nanmax(d[np.isfinite(d).all(axis=1)]) if len(d) else 0)
            for d in pd.diagrams
        )
        max_val = max(max_val, 1e-3)
        ax_pd.plot([0, max_val * 1.1], [0, max_val * 1.1], "k--", alpha=0.4, lw=1)
        ax_pd.set_xlabel("birth")
        ax_pd.set_ylabel("death")
        ax_pd.set_xlim(0, max_val * 1.1)
        ax_pd.set_ylim(0, max_val * 1.1)
        ax_pd.legend(loc="lower right", fontsize=9)
        ax_pd.grid(alpha=0.3)
        ax_pd.set_aspect("equal")

        n_h1 = pd.n_features(1)
        tot_h1 = pd.total_persistence(1)
        ax_pd.text(
            0.04, 0.96, f"$|H_1|$={n_h1}\n$\\sum$pers={tot_h1:.2f}",
            transform=ax_pd.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    out_path = OUT / "hero_01_topology_signature.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[hero] wrote {out_path}")
    return out_path


def hero_null_calibration(rng: np.random.Generator) -> Path:
    embedder = TakensEmbedder(dimension=3, delay=2, n_resample=150)

    print("[hero] building null n=200 ...", flush=True)
    null_small = build_null_distribution(
        n_realizations=200, n_sources_per=20, n_epochs=150,
        embedder=embedder, maxdim=1,
        rng=np.random.default_rng(rng.integers(0, 2**63)), verbose=True,
    )
    print("[hero] building null n=1000 ...", flush=True)
    null_big = build_null_distribution(
        n_realizations=1000, n_sources_per=20, n_epochs=150,
        embedder=embedder, maxdim=1,
        rng=np.random.default_rng(rng.integers(0, 2**63)), verbose=True,
    )

    idx = null_small.stat_names.index("total_persistence_H1")
    vals_small = null_small.summary_stats[:, idx]
    vals_big = null_big.summary_stats[:, idx]

    obs = float(np.percentile(vals_big, 99.5))

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bins = np.linspace(0, max(vals_small.max(), vals_big.max()) * 1.05, 50)
    ax.hist(vals_small, bins=bins, alpha=0.45, color="tab:orange",
            label=f"n = 200 realizations  (p-floor = 1/200 = {1/200:.3g})",
            density=True)
    ax.hist(vals_big, bins=bins, alpha=0.45, color="tab:blue",
            label=f"n = 1000 realizations  (p-floor = 1/1000 = {1/1000:.4g})",
            density=True)
    ax.axvline(obs, color="crimson", lw=2, ls="--",
               label=f"hypothetical observation = {obs:.2f}")
    ax.set_xlabel(r"$\sum$ persistence ($H_1$) of noise-only ensemble")
    ax.set_ylabel("density")
    ax.set_title(
        "Empirical null sharpens with realization count\n"
        "More realizations -> lower achievable p-value floor",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    out_path = OUT / "hero_02_null_calibration.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[hero] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    hero_topology_signature(rng)
    hero_null_calibration(rng)
    print("[hero] done")
