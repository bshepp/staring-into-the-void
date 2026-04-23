"""
ZTF validation: real data + attenuation experiment.
Uses ALeRCE API to fetch RR Lyrae and AGN, runs pipeline and attenuation.
"""
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from void.data.ztf import (
    query_objects_by_class,
    get_batch_light_curves,
)
from void.embedding.takens import TakensEmbedder
from void.embedding.features import extract_features
from void.topology.persistence import compute_persistence
from void.topology.null_model import build_null_distribution, compare_ensemble_to_null
from void.analysis.attenuation import run_attenuation_experiment, attenuate_light_curve
from void.data.synthetic import generate_noise
from void.viz.plots import (
    plot_light_curve,
    plot_embedding_2d,
    plot_persistence_diagram,
    plot_feature_space,
)

OUT = "output"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# Try RRL then RRLyr for RR Lyrae class name
def get_rrl_objects(n=25):
    for name in ("RRL", "RRLyr"):
        try:
            df = query_objects_by_class(
                classifier="lc_classifier",
                class_name=name,
                n_objects=n,
                probability_threshold=0.8,
            )
            if df is not None and len(df) > 0:
                return df.index.tolist(), name
        except Exception as e:
            print(f"  {name} failed: {e}")
    return [], None

def get_agn_objects(n=25):
    try:
        df = query_objects_by_class(
            classifier="lc_classifier",
            class_name="AGN",
            n_objects=n,
            probability_threshold=0.8,
        )
        if df is not None and len(df) > 0:
            return df.index.tolist()
    except Exception as e:
        print(f"  AGN failed: {e}")
    return []

print("=" * 60)
print("  ZTF Real Data Validation")
print("=" * 60)

print("\n1. Querying ALeRCE for RR Lyrae...")
rrl_oids, rrl_class = get_rrl_objects(25)
if not rrl_oids:
    print("   No RR Lyrae found. Check API/class names.")
    sys.exit(1)
print(f"   Found {len(rrl_oids)} objects (class={rrl_class})")

print("\n2. Querying ALeRCE for AGN...")
agn_oids = get_agn_objects(25)
print(f"   Found {len(agn_oids)} AGN objects")

print("\n3. Downloading light curves (with forced photometry)...")
rrl_lcs = get_batch_light_curves(rrl_oids[:15], include_forced=True, verbose=True)
agn_lcs = get_batch_light_curves(agn_oids[:15], include_forced=True, verbose=True)
rrl_lcs = [m for m in rrl_lcs if m and len(m.curves) > 0]
agn_lcs = [m for m in agn_lcs if m and len(m.curves) > 0]
print(f"   RR Lyrae: {len(rrl_lcs)} light curves")
print(f"   AGN: {len(agn_lcs)} light curves")

if len(rrl_lcs) < 3 or len(agn_lcs) < 3:
    print("   Need at least 3 of each. Exiting.")
    sys.exit(1)

embedder = TakensEmbedder(dimension=3, delay=2, interpolation="linear")

print("\n4. Takens embedding + persistence on real ZTF data...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, mblc in enumerate(rrl_lcs[:3]):
    band = "g" if "g" in mblc.curves else list(mblc.curves.keys())[0]
    lc = mblc[band]
    cloud = embedder.embed(lc)
    pd = compute_persistence(cloud, maxdim=1)
    plot_embedding_2d(cloud, ax=axes[0, i], title=f"RRL {mblc.object_id[:12]}...")
    axes[0, i].text(0.05, 0.05, f"H1={pd.total_persistence(1):.2f}", transform=axes[0, i].transAxes, fontsize=8)
for i, mblc in enumerate(agn_lcs[:3]):
    band = "g" if "g" in mblc.curves else list(mblc.curves.keys())[0]
    lc = mblc[band]
    cloud = embedder.embed(lc)
    pd = compute_persistence(cloud, maxdim=1)
    plot_embedding_2d(cloud, ax=axes[1, i], title=f"AGN {mblc.object_id[:12]}...")
    axes[1, i].text(0.05, 0.05, f"H1={pd.total_persistence(1):.2f}", transform=axes[1, i].transAxes, fontsize=8)
axes[0, 0].set_ylabel("RR Lyrae")
axes[1, 0].set_ylabel("AGN")
fig.suptitle("Takens Embeddings: Real ZTF Data", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/ztf_01_embeddings.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved {OUT}/ztf_01_embeddings.png")

print("\n5. Feature space (PCA): RRL vs AGN...")
all_feats, all_labels = [], []
for mblc in rrl_lcs:
    band = "g" if "g" in mblc.curves else list(mblc.curves.keys())[0]
    feats = extract_features(mblc[band], embedder=embedder, compute_tda=True)
    all_feats.append(feats)
    all_labels.append(0)
for mblc in agn_lcs:
    band = "g" if "g" in mblc.curves else list(mblc.curves.keys())[0]
    feats = extract_features(mblc[band], embedder=embedder, compute_tda=True)
    all_feats.append(feats)
    all_labels.append(1)
X = np.vstack(all_feats)
labels = np.array(all_labels)
fig = plot_feature_space(X, labels=labels, method="pca", title="Feature Space: RR Lyrae (0) vs AGN (1) - ZTF")
fig.savefig(f"{OUT}/ztf_02_feature_space.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved {OUT}/ztf_02_feature_space.png")

# High-SNR sources for attenuation: g-band, peak_snr > 5
source_lcs = []
for mblc in rrl_lcs:
    band = "g" if "g" in mblc.curves else list(mblc.curves.keys())[0]
    lc = mblc[band]
    if hasattr(lc, "peak_snr") and lc.peak_snr > 5:
        source_lcs.append(lc)
    elif len(lc.fluxes) >= 20:
        source_lcs.append(lc)
if len(source_lcs) < 5:
    source_lcs = [mblc["g"] if "g" in mblc.curves else mblc[list(mblc.curves.keys())[0]] for mblc in rrl_lcs[:10]]
    source_lcs = [lc for lc in source_lcs if len(lc.fluxes) >= 20][:10]
print(f"\n6. Attenuation experiment: {len(source_lcs)} high-SNR RR Lyrae")

# Match null pool size to the real source count so the background topology
# is computed on a comparable cloud (was dominated by 150 noise sources).
n_pool = max(len(source_lcs) * 2, 30)
noise_lcs = [generate_noise(n_epochs=150, rng=np.random.default_rng(rng.integers(0, 2**63))) for _ in range(n_pool)]
# 200 realizations gives a usable p-floor of 1/200 = 0.005, well below 0.05.
null_dist = build_null_distribution(
    n_realizations=200,
    n_sources_per=len(source_lcs) + len(noise_lcs),
    n_epochs=150,
    embedder=embedder,
    maxdim=1,
    rng=np.random.default_rng(rng.integers(0, 2**63)),
    verbose=True,
)

experiment = run_attenuation_experiment(
    source_light_curves=source_lcs,
    noise_light_curves=noise_lcs,
    attenuation_factors=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    null_dist=null_dist,
    embedder=embedder,
    source_type="RR Lyrae (ZTF)",
    verbose=True,
)

print(f"\n   Recovery threshold (min factor detected): {experiment.recovery_threshold}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
factors = experiment.factors
h1_vals = [r.total_persistence_h1 for r in experiment.results]
p_vals = [r.p_value for r in experiment.results]
ax1.plot(factors, h1_vals, "o-", color="#4C72B0")
ax1.axhline(experiment.null_mean_h1, color="red", linestyle="--", label="Null mean")
ax1.set_xlabel("Attenuation Factor")
ax1.set_ylabel("Total H1 Persistence")
ax1.set_title("Topological Signal vs Attenuation")
ax1.legend()
ax2.semilogy(factors, p_vals, "o-", color="#DD8452")
ax2.axhline(0.05, color="gray", linestyle="--", label="p=0.05")
ax2.set_xlabel("Attenuation Factor")
ax2.set_ylabel("p-value")
ax2.set_title("Statistical Significance")
ax2.legend()
fig.suptitle(f"Attenuation Experiment - RR Lyrae ZTF ({len(source_lcs)} sources)", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/ztf_03_attenuation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved {OUT}/ztf_03_attenuation.png")

print("\n" + "=" * 60)
print("  ZTF validation complete. Figures in output/")
print("=" * 60)
