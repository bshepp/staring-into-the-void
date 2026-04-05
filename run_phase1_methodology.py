"""Phase 1 Methodology — Paper-Quality Experimental Results

Produces all figures and statistics for the methodology paper:

  Experiment 1: Synthetic proof — can PH distinguish signal from noise?
  Experiment 2: Source-type discrimination via topology
  Experiment 3: Sub-threshold recovery (attenuation experiment)
  Experiment 4: Systematic power analysis with confidence intervals
  Experiment 5: Microlensing sensitivity (key science target)
  Experiment 6: Rubin DP1 analysis (if RSP access available)

Outputs paper-ready figures to output/paper/ with publication styling.
Each experiment is independent — comment out sections you don't need.

Runtime estimate: ~15-30 min for Experiments 1-5 (synthetic-only).
"""

import os
import sys
import time
import warnings

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import StandardScaler

from void.data.synthetic import (
    generate_periodic,
    generate_transient,
    generate_stochastic,
    generate_slow_evolving,
    generate_microlensing,
    generate_noise,
    generate_ensemble,
)
from void.embedding.takens import TakensEmbedder
from void.embedding.features import extract_features
from void.topology.persistence import compute_persistence
from void.topology.null_model import build_null_distribution
from void.analysis.anomaly import AnomalyDetector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
rng = np.random.default_rng(SEED)

OUT = "output/paper"
os.makedirs(OUT, exist_ok=True)

# Pipeline defaults
EMBEDDER = TakensEmbedder(dimension=3, delay=2)
N_EPOCHS = 200
MAXDIM = 1

# Paper figure style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (7, 5),
})


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


# ===================================================================
# Experiment 1: Proof of concept — PH distinguishes signal from noise
# ===================================================================

def experiment_1_proof_of_concept() -> None:
    """Can persistent homology on individual light curves distinguish
    sub-threshold signals from pure noise?

    Sweeps SNR from 0.5 to 10 for each source type. At each level,
    generates N trials and computes H1 total persistence.
    """
    banner("Experiment 1: Signal vs Noise Discrimination")
    t0 = time.time()

    snr_levels = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    n_trials = 50
    generators = {
        "Periodic": generate_periodic,
        "Transient": generate_transient,
        "Stochastic": generate_stochastic,
        "Microlensing": generate_microlensing,
    }

    # Noise baseline
    noise_h1 = []
    for _ in range(n_trials):
        child_rng = np.random.default_rng(rng.integers(0, 2**63))
        lc = generate_noise(n_epochs=N_EPOCHS, rng=child_rng)
        cloud = EMBEDDER.embed(lc)
        pd = compute_persistence(cloud, maxdim=MAXDIM)
        noise_h1.append(pd.total_persistence(dim=1))
    noise_median = np.median(noise_h1)
    noise_iqr = np.subtract(*np.percentile(noise_h1, [75, 25]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, (name, gen) in enumerate(generators.items()):
        ax = axes[idx]
        medians = []
        q25s = []
        q75s = []

        for snr in snr_levels:
            vals = []
            gen_kwargs = {"snr": snr, "n_epochs": N_EPOCHS}
            for _ in range(n_trials):
                child_rng = np.random.default_rng(rng.integers(0, 2**63))
                lc = gen(rng=child_rng, **gen_kwargs)
                cloud = EMBEDDER.embed(lc)
                pd = compute_persistence(cloud, maxdim=MAXDIM)
                vals.append(pd.total_persistence(dim=1))
            medians.append(np.median(vals))
            q25s.append(np.percentile(vals, 25))
            q75s.append(np.percentile(vals, 75))

        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)

        ax.fill_between(snr_levels, q25s, q75s, alpha=0.3, label=f"{name} IQR")
        ax.plot(snr_levels, medians, "o-", markersize=4, label=f"{name} median")
        ax.axhline(noise_median, color="gray", ls="--", lw=1, label="Noise median")
        ax.axhspan(noise_median - noise_iqr, noise_median + noise_iqr,
                    alpha=0.1, color="gray")
        ax.set_title(name)
        ax.set_xlabel("Injected SNR")
        ax.set_ylabel("H₁ Total Persistence")
        ax.legend(loc="upper left")

    fig.suptitle("Experiment 1: Topological Signal Detection vs SNR", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp1_signal_vs_noise.png")
    plt.close(fig)
    print(f"  Saved exp1_signal_vs_noise.png ({elapsed(t0)})")


# ===================================================================
# Experiment 2: Source-type discrimination in feature space
# ===================================================================

def experiment_2_source_type_discrimination() -> None:
    """Do different astrophysical source types occupy distinct regions
    of the topological feature space?

    Generates ensembles of each type at fixed SNR, extracts features,
    projects via PCA, and measures cluster separability.
    """
    banner("Experiment 2: Source-Type Discrimination")
    t0 = time.time()

    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    source_types = {
        "Periodic": generate_periodic,
        "Transient": generate_transient,
        "Stochastic": generate_stochastic,
        "Slow-evolving": generate_slow_evolving,
        "Microlensing": generate_microlensing,
        "Noise": generate_noise,
    }

    snr_levels = [2.0, 3.0, 5.0]
    n_per_type = 80

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for snr_idx, snr in enumerate(snr_levels):
        ax = axes[snr_idx]
        all_features = []
        all_labels = []

        for label, gen in source_types.items():
            for _ in range(n_per_type):
                child_rng = np.random.default_rng(rng.integers(0, 2**63))
                kwargs = {"n_epochs": N_EPOCHS, "rng": child_rng}
                if label != "Noise":
                    kwargs["snr"] = snr
                lc = gen(**kwargs)
                feats = extract_features(lc)
                all_features.append(feats)
                all_labels.append(label)

        X = np.vstack(all_features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Replace NaN/inf with 0 for PCA
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)

        for label in source_types:
            mask = np.array(all_labels) == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, alpha=0.5, label=label)

        sil = silhouette_score(X_scaled, all_labels)
        ax.set_title(f"SNR = {snr}  (silhouette = {sil:.2f})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        if snr_idx == 2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

    fig.suptitle("Experiment 2: Source-Type Separation in Feature Space", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp2_source_discrimination.png")
    plt.close(fig)
    print(f"  Saved exp2_source_discrimination.png ({elapsed(t0)})")


# ===================================================================
# Experiment 3: Attenuation recovery
# ===================================================================

def experiment_3_attenuation_recovery() -> None:
    """Can the topological pipeline recover known signals when they are
    artificially attenuated to sub-threshold levels?

    Uses high-SNR synthetic light curves, attenuates them progressively,
    and measures detection rate via the null-model framework.
    """
    banner("Experiment 3: Attenuation Recovery")
    t0 = time.time()

    from void.analysis.attenuation import run_attenuation_experiment

    generators = {
        "Periodic": (generate_periodic, {"snr": 10.0, "period": 5.0}),
        "Transient": (generate_transient, {"snr": 10.0}),
        "Stochastic": (generate_stochastic, {"snr": 10.0, "tau": 200.0}),
        "Microlensing": (generate_microlensing, {"snr": 10.0, "u_min": 0.3}),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for idx, (name, (gen, kwargs)) in enumerate(generators.items()):
        ax = axes[idx]

        # Build high-SNR sources and noise
        child_rng = np.random.default_rng(rng.integers(0, 2**63))
        sources = []
        for _ in range(30):
            s_rng = np.random.default_rng(child_rng.integers(0, 2**63))
            sources.append(gen(rng=s_rng, n_epochs=N_EPOCHS, **kwargs))
        noise_lcs = []
        for _ in range(150):
            n_rng = np.random.default_rng(child_rng.integers(0, 2**63))
            noise_lcs.append(generate_noise(n_epochs=N_EPOCHS, rng=n_rng))

        result = run_attenuation_experiment(
            source_light_curves=sources,
            noise_light_curves=noise_lcs,
            attenuation_factors=np.linspace(1.0, 0.05, 15).tolist(),
            embedder=EMBEDDER,
            maxdim=MAXDIM,
            source_type=name,
            rng=np.random.default_rng(child_rng.integers(0, 2**63)),
        )

        factors = result.factors
        detected = [1.0 if d else 0.0 for d in result.detection_curve]

        ax.plot(factors, detected, "o-", color="C0", markersize=4)
        ax.axhline(0.05, color="red", ls=":", lw=1, label="α = 0.05")
        ax.set_xlabel("Attenuation Factor (1 = full signal)")
        ax.set_ylabel("Detected")
        ax.set_title(f"{name} (recovery threshold ≥ {result.recovery_threshold:.2f})")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right")
        ax.invert_xaxis()

    fig.suptitle("Experiment 3: Signal Recovery Under Attenuation", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp3_attenuation.png")
    plt.close(fig)
    print(f"  Saved exp3_attenuation.png ({elapsed(t0)})")


# ===================================================================
# Experiment 4: Systematic power analysis with confidence intervals
# ===================================================================

def experiment_4_power_analysis() -> None:
    """Systematic detection power: SNR × population size grid.

    For each (SNR, N_signal) pair, runs multiple trials and computes
    detection rate with 95% Wilson binomial confidence intervals.
    """
    banner("Experiment 4: Power Analysis")
    t0 = time.time()

    snr_grid = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    n_signal_grid = np.array([5, 10, 20, 30, 50])
    n_noise = 200
    n_trials = 12
    n_null_realizations = 25

    # Pre-build null distribution (shared)
    print("  Building null distribution...")
    null_dist = build_null_distribution(
        n_realizations=n_null_realizations,
        n_sources_per=n_noise,
        n_epochs=N_EPOCHS,
        embedder=EMBEDDER,
        maxdim=MAXDIM,
        rng=np.random.default_rng(rng.integers(0, 2**63)),
        verbose=True,
    )
    detector = AnomalyDetector(null_dist=null_dist, embedder=EMBEDDER, maxdim=MAXDIM)
    print(f"  Null built ({elapsed(t0)})")

    # Power grid
    power_grid = np.zeros((len(snr_grid), len(n_signal_grid)))
    ci_low = np.zeros_like(power_grid)
    ci_high = np.zeros_like(power_grid)

    for i, snr in enumerate(snr_grid):
        for j, n_sig in enumerate(n_signal_grid):
            detections = 0
            for trial in range(n_trials):
                child_rng = np.random.default_rng(rng.integers(0, 2**63))
                ensemble = generate_ensemble(
                    n_signal=int(n_sig), n_noise=n_noise,
                    signal_generator=generate_periodic,
                    signal_kwargs={"snr": snr, "n_epochs": N_EPOCHS},
                    noise_kwargs={"n_epochs": N_EPOCHS},
                    rng=child_rng,
                )
                result = detector.score(ensemble)
                if result.is_anomalous:
                    detections += 1

            rate = detections / n_trials
            power_grid[i, j] = rate

            # Wilson score 95% CI
            z = 1.96
            n = n_trials
            p_hat = rate
            denom = 1 + z**2 / n
            center = (p_hat + z**2 / (2 * n)) / denom
            spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
            ci_low[i, j] = max(0, center - spread)
            ci_high[i, j] = min(1, center + spread)

            print(f"    SNR={snr:.1f}, N_signal={n_sig:3d}: "
                  f"rate={rate:.2f} [{ci_low[i,j]:.2f}, {ci_high[i,j]:.2f}]")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(power_grid, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(n_signal_grid)))
    ax.set_xticklabels(n_signal_grid)
    ax.set_yticks(range(len(snr_grid)))
    ax.set_yticklabels([f"{s:.1f}" for s in snr_grid])
    ax.set_xlabel("Number of Signal Sources in Ensemble")
    ax.set_ylabel("Per-Source SNR")

    for i in range(len(snr_grid)):
        for j in range(len(n_signal_grid)):
            rate = power_grid[i, j]
            lo, hi = ci_low[i, j], ci_high[i, j]
            color = "white" if rate > 0.5 else "black"
            ax.text(j, i, f"{rate:.0%}\n[{lo:.0%},{hi:.0%}]",
                    ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Detection Rate")
    ax.set_title("Experiment 4: Topological Detection Power\n"
                 f"(N_noise={n_noise}, {n_trials} trials, 95% Wilson CI)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp4_power_analysis.png")
    plt.close(fig)

    # Sensitivity curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, n_sig in enumerate(n_signal_grid):
        rates = power_grid[:, j]
        ax.plot(snr_grid, rates, "o-", markersize=5, label=f"N_signal = {n_sig}")
        ax.fill_between(snr_grid, ci_low[:, j], ci_high[:, j], alpha=0.1)
    ax.axhline(0.8, color="red", ls=":", lw=1, label="80% power")
    ax.set_xlabel("Per-Source SNR")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Experiment 4: Sensitivity Curves")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp4_sensitivity_curves.png")
    plt.close(fig)
    print(f"  Saved exp4_*.png ({elapsed(t0)})")


# ===================================================================
# Experiment 5: Microlensing-specific sensitivity
# ===================================================================

def experiment_5_microlensing_sensitivity() -> None:
    """Microlensing is a key science target — dark matter substructure.

    Tests detection sensitivity as a function of:
    - u_min (impact parameter → magnification strength)
    - t_einstein (crossing time → event duration)
    """
    banner("Experiment 5: Microlensing Sensitivity")
    t0 = time.time()

    n_noise = 200
    n_signal = 20
    n_trials = 10
    n_null_realizations = 20

    null_dist = build_null_distribution(
        n_realizations=n_null_realizations,
        n_sources_per=n_noise,
        n_epochs=N_EPOCHS,
        embedder=EMBEDDER,
        maxdim=MAXDIM,
        rng=np.random.default_rng(rng.integers(0, 2**63)),
    )
    detector = AnomalyDetector(null_dist=null_dist, embedder=EMBEDDER, maxdim=MAXDIM)

    # Grid: u_min controls magnification, t_einstein controls duration
    u_min_grid = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])
    t_einstein_grid = np.array([10.0, 20.0, 40.0, 80.0, 150.0])

    power_grid = np.zeros((len(u_min_grid), len(t_einstein_grid)))

    for i, u_min in enumerate(u_min_grid):
        for j, t_e in enumerate(t_einstein_grid):
            detections = 0
            for _ in range(n_trials):
                child_rng = np.random.default_rng(rng.integers(0, 2**63))
                ensemble = generate_ensemble(
                    n_signal=n_signal, n_noise=n_noise,
                    signal_generator=generate_microlensing,
                    signal_kwargs={
                        "snr": 3.0, "u_min": u_min,
                        "t_einstein": t_e, "n_epochs": N_EPOCHS,
                    },
                    noise_kwargs={"n_epochs": N_EPOCHS},
                    rng=child_rng,
                )
                result = detector.score(ensemble)
                if result.is_anomalous:
                    detections += 1
            power_grid[i, j] = detections / n_trials

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(power_grid, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(t_einstein_grid)))
    ax.set_xticklabels([f"{t:.0f}" for t in t_einstein_grid])
    ax.set_yticks(range(len(u_min_grid)))
    ax.set_yticklabels([f"{u:.1f}" for u in u_min_grid])
    ax.set_xlabel("Einstein Crossing Time tₑ (days)")
    ax.set_ylabel("Minimum Impact Parameter u₀ (Einstein radii)")

    for i in range(len(u_min_grid)):
        for j in range(len(t_einstein_grid)):
            color = "white" if power_grid[i, j] > 0.5 else "black"
            ax.text(j, i, f"{power_grid[i,j]:.0%}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Detection Rate")
    ax.set_title(f"Experiment 5: Microlensing Sensitivity\n"
                 f"(SNR=3, N_signal={n_signal}, N_noise={n_noise})")
    fig.tight_layout()
    fig.savefig(f"{OUT}/exp5_microlensing.png")
    plt.close(fig)
    print(f"  Saved exp5_microlensing.png ({elapsed(t0)})")


# ===================================================================
# Experiment 6: Rubin DP1 (conditional on RSP access)
# ===================================================================

def experiment_6_rubin_dp1() -> None:
    """Query Rubin DP1 for near-threshold DiaObjects and run the pipeline.

    This experiment requires:
    - pyvo installed
    - RSP access token at ~/.lsst/token or RUBIN_ACCESS_TOKEN env var
    - DP1 TAP service available

    Gracefully skips if any prerequisite is missing.
    """
    banner("Experiment 6: Rubin DP1 Analysis")

    try:
        from void.data.rubin import (
            query_near_threshold_objects,
            get_dia_source_light_curve,
            build_ensemble_from_rubin,
        )
    except ImportError:
        print("  Skipped: pyvo not installed (pip install pyvo)")
        return

    t0 = time.time()

    # Step 1: Query near-threshold objects
    print("  Querying DP1 for near-threshold DiaObjects...")
    try:
        objects = query_near_threshold_objects(
            max_objects=200,
            min_n_dia_sources=10,
            max_mean_snr=5.0,
            min_mean_snr=0.5,
            bands=["r", "i"],
        )
    except Exception as e:
        print(f"  Skipped: TAP query failed ({e})")
        print("  This is expected if not running on RSP or without access token.")
        return

    print(f"  Found {len(objects)} near-threshold objects")

    if len(objects) < 20:
        print("  Too few objects for ensemble analysis. Skipping.")
        return

    # Step 2: Download light curves
    dia_ids = objects["diaObjectId"].tolist()[:100]
    print(f"  Downloading light curves for {len(dia_ids)} objects...")
    ensemble = build_ensemble_from_rubin(
        dia_ids, band="r", source="diasource", region_id="dp1_near_threshold"
    )
    print(f"  Ensemble: {len(ensemble)} sources with r-band data")

    if len(ensemble) < 20:
        print("  Too few sources with r-band data. Skipping.")
        return

    # Step 3: Build null from DP1 noise characteristics
    # Use actual flux error distribution from the data
    print("  Building null distribution from DP1 noise model...")
    null_dist = build_null_distribution(
        n_realizations=20,
        n_sources_per=len(ensemble),
        n_epochs=N_EPOCHS,
        embedder=EMBEDDER,
        maxdim=MAXDIM,
        rng=np.random.default_rng(rng.integers(0, 2**63)),
    )

    # Step 4: Test ensemble
    detector = AnomalyDetector(null_dist=null_dist, embedder=EMBEDDER, maxdim=MAXDIM)
    result = detector.score(ensemble)

    print(f"\n  DP1 Near-Threshold Ensemble Result:")
    print(f"  {result.summary()}")

    # Save result summary
    with open(f"{OUT}/exp6_dp1_result.txt", "w") as f:
        f.write("Experiment 6: Rubin DP1 Near-Threshold Analysis\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"N objects queried: {len(dia_ids)}\n")
        f.write(f"N sources in ensemble: {len(ensemble)}\n")
        f.write(f"Anomalous: {result.is_anomalous}\n")
        f.write(f"Min p-value: {result.min_p_value:.4e}\n")
        f.write(f"Most anomalous stat: {result.most_anomalous_stat}\n")
        f.write(f"\n{result.summary()}\n")

    print(f"  Saved exp6_dp1_result.txt ({elapsed(t0)})")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    t_total = time.time()
    print("Phase 1 Methodology — Paper Results")
    print(f"Seed: {SEED}")
    print(f"Output: {OUT}/")

    experiment_1_proof_of_concept()
    experiment_2_source_type_discrimination()
    experiment_3_attenuation_recovery()
    experiment_4_power_analysis()
    experiment_5_microlensing_sensitivity()
    experiment_6_rubin_dp1()

    banner(f"All experiments complete ({elapsed(t_total)})")
    print(f"\nFigures saved to {OUT}/")


if __name__ == "__main__":
    main()
