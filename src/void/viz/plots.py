"""Visualization utilities for light curves, embeddings, and persistence diagrams.

All plot functions return matplotlib Figure objects for flexibility.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from void.data.models import Ensemble, LightCurve
from void.topology.persistence import PersistenceDiagram


# ---------------------------------------------------------------------------
# Light curve plots
# ---------------------------------------------------------------------------

def plot_light_curve(
    lc: LightCurve,
    ax: Optional[plt.Axes] = None,
    show_errors: bool = True,
    title: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plot a single light curve with error bars."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    defaults = dict(fmt="o", markersize=3, capsize=1, alpha=0.7)
    defaults.update(kwargs)

    if show_errors:
        ax.errorbar(lc.times, lc.fluxes, yerr=lc.flux_errors, **defaults)
    else:
        ax.scatter(lc.times, lc.fluxes, s=defaults.get("markersize", 3) ** 2, alpha=0.7)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    if title:
        ax.set_title(title)
    elif "source_type" in lc.metadata:
        snr_str = f", SNR={lc.metadata.get('injected_snr', '?')}"
        ax.set_title(f"{lc.metadata['source_type']} [{lc.band}]{snr_str}")

    fig = fig or ax.get_figure()
    fig.tight_layout()
    return fig


def plot_light_curves_grid(
    light_curves: list[LightCurve],
    ncols: int = 3,
    figsize_per: tuple[float, float] = (4, 2.5),
) -> plt.Figure:
    """Plot multiple light curves in a grid."""
    n = len(light_curves)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_2d(axes)

    for i, lc in enumerate(light_curves):
        r, c = divmod(i, ncols)
        plot_light_curve(lc, ax=axes[r, c])

    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Embedding plots
# ---------------------------------------------------------------------------

def plot_embedding_2d(
    cloud: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Takens Embedding",
    color: Optional[np.ndarray] = None,
    **kwargs,
) -> plt.Figure:
    """Plot a 2D projection of the Takens embedding point cloud."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    defaults = dict(s=8, alpha=0.6, edgecolors="none")
    defaults.update(kwargs)

    if cloud.shape[1] >= 2:
        scatter = ax.scatter(cloud[:, 0], cloud[:, 1], c=color, **defaults)
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t-τ)")
    else:
        ax.scatter(np.arange(len(cloud)), cloud[:, 0], c=color, **defaults)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig = fig or ax.get_figure()
    fig.tight_layout()
    return fig


def plot_embedding_3d(
    cloud: np.ndarray,
    title: str = "Takens Embedding (3D)",
    color: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plot a 3D Takens embedding point cloud."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    if cloud.shape[1] >= 3:
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=color, s=5, alpha=0.5)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t-τ)")
        ax.set_zlabel("x(t-2τ)")
    else:
        ax.scatter(cloud[:, 0], cloud[:, 1] if cloud.shape[1] > 1 else np.zeros(len(cloud)),
                   np.zeros(len(cloud)), s=5, alpha=0.5)

    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------
# Persistence diagram plots
# ---------------------------------------------------------------------------

def plot_persistence_diagram(
    pd: PersistenceDiagram,
    dims: list[int] | None = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Persistence Diagram",
) -> plt.Figure:
    """Plot a persistence diagram (birth vs death)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    dims = dims or list(range(pd.maxdim + 1))
    colors = plt.cm.Set1.colors
    labels = {0: "H₀ (components)", 1: "H₁ (loops)", 2: "H₂ (voids)"}

    all_vals = []
    for d in dims:
        dgm = pd._finite(d)
        if len(dgm) > 0:
            all_vals.extend(dgm.ravel().tolist())
            ax.scatter(
                dgm[:, 0], dgm[:, 1],
                c=[colors[d % len(colors)]],
                s=20, alpha=0.7,
                label=labels.get(d, f"H{d}"),
                edgecolors="k", linewidths=0.3,
            )

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        margin = 0.1 * (hi - lo) if hi > lo else 0.5
        lim = (lo - margin, hi + margin)
        ax.plot(lim, lim, "k--", linewidth=0.5, alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig = fig or ax.get_figure()
    fig.tight_layout()
    return fig


def plot_barcode(
    pd: PersistenceDiagram,
    dims: list[int] | None = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Persistence Barcode",
) -> plt.Figure:
    """Plot a persistence barcode."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    dims = dims or list(range(pd.maxdim + 1))
    colors = plt.cm.Set1.colors
    labels = {0: "H₀", 1: "H₁", 2: "H₂"}

    y_pos = 0
    for d in dims:
        dgm = pd._finite(d)
        if len(dgm) == 0:
            continue
        sorted_idx = np.argsort(dgm[:, 1] - dgm[:, 0])[::-1]
        color = colors[d % len(colors)]
        for i, idx in enumerate(sorted_idx):
            birth, death = dgm[idx]
            label = labels.get(d, f"H{d}") if i == 0 else None
            ax.plot([birth, death], [y_pos, y_pos], color=color, linewidth=1.5,
                    label=label, solid_capstyle="butt")
            y_pos += 1

    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_yticks([])
    fig = fig or ax.get_figure()
    fig.tight_layout()
    return fig


def plot_persistence_image(
    image: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Persistence Image",
    cmap: str = "hot",
) -> plt.Figure:
    """Plot a persistence image as a heatmap."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(image, cmap=cmap, origin="lower", aspect="auto")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Persistence")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig = fig or ax.get_figure()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ensemble / population plots
# ---------------------------------------------------------------------------

def plot_feature_space(
    feature_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "pca",
    ax: Optional[plt.Axes] = None,
    title: str = "Feature Space",
) -> plt.Figure:
    """2D projection of the ensemble feature space via PCA or t-SNE."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    scaled = StandardScaler().fit_transform(feature_matrix)

    if method == "pca":
        proj = PCA(n_components=2).fit_transform(scaled)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    else:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=min(30, len(scaled) - 1)).fit_transform(scaled)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, s=10, alpha=0.6,
                         cmap="tab10", edgecolors="none")
    if labels is not None:
        plt.colorbar(scatter, ax=ax, shrink=0.8)

    ax.set_title(title)
    fig = fig or ax.get_figure()
    fig.tight_layout()
    return fig


def plot_snr_comparison(
    persistence_values: dict[float, list[float]],
    null_values: list[float] | None = None,
    ax: Optional[plt.Axes] = None,
    ylabel: str = "Total H₁ Persistence",
    title: str = "Topological Signal vs SNR",
) -> plt.Figure:
    """Plot topological summary statistic as a function of SNR.

    Parameters
    ----------
    persistence_values : dict mapping SNR -> list of measured values
    null_values : list of values from the null model (horizontal reference band)
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    snrs = sorted(persistence_values.keys())
    means = [np.mean(persistence_values[s]) for s in snrs]
    stds = [np.std(persistence_values[s]) for s in snrs]

    ax.errorbar(snrs, means, yerr=stds, fmt="o-", capsize=3, label="Signal + Noise")

    if null_values is not None:
        null_mean = np.mean(null_values)
        null_std = np.std(null_values)
        ax.axhline(null_mean, color="red", linestyle="--", label="Null mean")
        ax.axhspan(null_mean - 2 * null_std, null_mean + 2 * null_std,
                   color="red", alpha=0.1, label="Null ±2σ")

    ax.set_xlabel("Peak SNR")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig = fig or ax.get_figure()
    fig.tight_layout()
    return fig
