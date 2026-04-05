"""Mini power analysis — trimmed for feasibility.

Uses smaller ensembles and fewer trials to get results in minutes, not hours.
"""

import os
import sys
import time
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from void.data.synthetic import generate_periodic, generate_noise, generate_ensemble
from void.embedding.takens import TakensEmbedder
from void.embedding.features import extract_features
from void.topology.persistence import compute_persistence
from void.topology.null_model import build_null_distribution, compare_ensemble_to_null
from void.analysis.anomaly import AnomalyDetector

OUT = "output"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# Use smaller ensembles for speed: 100 sources instead of 250
N_NOISE = 100
N_EPOCHS = 100

embedder = TakensEmbedder(dimension=3, delay=2)

print("Building null distribution (20 realizations, 100 sources each)...")
t0 = time.time()
null_dist = build_null_distribution(
    n_realizations=20,
    n_sources_per=N_NOISE,
    n_epochs=N_EPOCHS,
    embedder=embedder,
    maxdim=1,
    rng=np.random.default_rng(789),
    verbose=True,
)
t1 = time.time()
print(f"Done in {t1-t0:.1f}s\n")

# Quick sanity: test a strong-signal ensemble
print("Sanity check: strong signal ensemble...")
strong = generate_ensemble(
    n_signal=30, n_noise=N_NOISE,
    signal_generator=generate_periodic,
    signal_kwargs={"snr": 5.0, "period": 30.0, "n_epochs": N_EPOCHS},
    noise_kwargs={"n_epochs": N_EPOCHS},
    rng=np.random.default_rng(123),
)
result = compare_ensemble_to_null(strong, null_dist, embedder=embedder)
print(f"  Significant: {result['significant']}")
print(f"  Min p-value: {result['min_p_value']:.4f}")
print(f"  Most significant: {result['most_significant_stat']}\n")

# Power analysis
print("=" * 60)
print("  Power Analysis: Detection Rate vs (SNR, N_signal)")
print("=" * 60)

snr_test = [1.0, 2.0, 3.0, 4.0, 5.0]
pop_sizes = [5, 15, 30, 50]
n_trials = 8

detector = AnomalyDetector(null_dist, embedder=embedder)

detection_rates = np.zeros((len(snr_test), len(pop_sizes)))
for i, snr in enumerate(snr_test):
    for j, n_sig in enumerate(pop_sizes):
        detections = 0
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2**63))
            ens = generate_ensemble(
                n_signal=n_sig, n_noise=N_NOISE,
                signal_generator=generate_periodic,
                signal_kwargs={"snr": snr, "period": 30.0, "n_epochs": N_EPOCHS},
                noise_kwargs={"n_epochs": N_EPOCHS},
                rng=trial_rng,
            )
            res = detector.score(ens)
            if res.is_anomalous:
                detections += 1
        detection_rates[i, j] = detections / n_trials
        print(f"  SNR={snr:.0f}s, N_signal={n_sig:3d}: "
              f"detection rate = {detection_rates[i, j]:.0%}")

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(detection_rates, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
ax.set_xticks(range(len(pop_sizes)))
ax.set_xticklabels(pop_sizes)
ax.set_yticks(range(len(snr_test)))
ax.set_yticklabels([f"{s:.0f}" for s in snr_test])
ax.set_xlabel("Number of Injected Signal Sources")
ax.set_ylabel("Peak SNR (sigma)")
ax.set_title("Detection Rate -- Periodic Signals in Noise Background")
for ii in range(len(snr_test)):
    for jj in range(len(pop_sizes)):
        color = "white" if detection_rates[ii, jj] < 0.5 else "black"
        ax.text(jj, ii, f"{detection_rates[ii, jj]:.0%}",
                ha="center", va="center", color=color, fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="Detection Rate", shrink=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/08_power_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved 08_power_analysis.png")

# Sensitivity curves
fig, ax = plt.subplots(figsize=(8, 5))
for j, n_sig in enumerate(pop_sizes):
    ax.plot(snr_test, detection_rates[:, j], "o-", label=f"N={n_sig}", linewidth=2)
ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
ax.set_xlabel("Peak SNR (sigma)")
ax.set_ylabel("Detection Rate")
ax.set_title("Sensitivity Curves -- Periodic Signals")
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/09_sensitivity_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved 09_sensitivity_curves.png")

t_end = time.time()
print(f"\n  Total time: {t_end - t0:.0f}s")
print("\n  Pipeline complete. All 9 figures in output/")
