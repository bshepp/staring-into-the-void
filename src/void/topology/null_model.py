"""Null model framework for topological significance testing.

Generates ensembles of pure-noise forced photometry, computes their
persistence diagrams, and builds empirical null distributions of
topological summary statistics.  Provides permutation testing and
summary statistic comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from void.data.models import Ensemble, LightCurve
from void.data.synthetic import generate_noise
from void.embedding.features import extract_features
from void.embedding.takens import TakensEmbedder
from void.topology.distances import distance_from_null, wasserstein_distance
from void.topology.persistence import PersistenceDiagram, compute_persistence


@dataclass
class NullDistribution:
    """Empirical null distribution of topological summary statistics.

    Built by computing persistence on many noise-only ensembles.
    """

    summary_stats: np.ndarray  # (n_realizations, n_stats)
    stat_names: list[str]
    diagrams: list[PersistenceDiagram] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_realizations(self) -> int:
        return self.summary_stats.shape[0]

    def mean(self) -> np.ndarray:
        return np.mean(self.summary_stats, axis=0)

    def std(self) -> np.ndarray:
        return np.std(self.summary_stats, axis=0)

    def percentile(self, q: float) -> np.ndarray:
        return np.percentile(self.summary_stats, q, axis=0)

    def p_value(self, observed_stats: np.ndarray) -> np.ndarray:
        """Fraction of null realizations with stats >= observed (per-stat)."""
        return np.mean(self.summary_stats >= observed_stats[None, :], axis=0)

    def z_score(self, observed_stats: np.ndarray) -> np.ndarray:
        """Z-score of observed stats relative to null distribution."""
        return (observed_stats - self.mean()) / (self.std() + 1e-12)


def build_null_distribution(
    n_realizations: int = 100,
    n_sources_per: int = 250,
    n_epochs: int = 200,
    base_error: float = 100.0,
    baseline_days: float = 3650.0,
    embedder: TakensEmbedder | None = None,
    maxdim: int = 1,
    rng: Optional[np.random.Generator] = None,
    store_diagrams: bool = False,
    verbose: bool = True,
) -> NullDistribution:
    """Build empirical null distribution from noise-only ensembles.

    For each realization:
    1. Generate n_sources_per noise-only light curves
    2. Extract features from each
    3. Compute persistence on the feature-space point cloud
    4. Record summary statistics

    Parameters
    ----------
    n_realizations : int
        Number of independent null ensembles to generate.
    n_sources_per : int
        Number of noise-only sources per ensemble.
    store_diagrams : bool
        If True, keep the PersistenceDiagram objects (uses more memory).
    """
    rng = rng or np.random.default_rng()
    embedder = embedder or TakensEmbedder(dimension=3, delay=2)

    stat_names = _summary_stat_names(maxdim)
    all_stats = []
    all_diagrams = []

    for i in range(n_realizations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Null realization {i + 1}/{n_realizations}")

        features_list = []
        for _ in range(n_sources_per):
            child_rng = np.random.default_rng(rng.integers(0, 2**63))
            lc = generate_noise(
                n_epochs=n_epochs,
                baseline_days=baseline_days,
                base_error=base_error,
                rng=child_rng,
            )
            feats = extract_features(lc, embedder=embedder, compute_tda=True, maxdim=maxdim)
            features_list.append(feats)

        feature_matrix = np.vstack(features_list)

        from sklearn.preprocessing import StandardScaler
        scaled = StandardScaler().fit_transform(feature_matrix)

        pd = compute_persistence(scaled, maxdim=maxdim)
        stats = _extract_summary_stats(pd, maxdim)
        all_stats.append(stats)

        if store_diagrams:
            all_diagrams.append(pd)

    return NullDistribution(
        summary_stats=np.array(all_stats),
        stat_names=stat_names,
        diagrams=all_diagrams,
        metadata={
            "n_realizations": n_realizations,
            "n_sources_per": n_sources_per,
            "n_epochs": n_epochs,
        },
    )


def compare_ensemble_to_null(
    ensemble: Ensemble,
    null_dist: NullDistribution,
    embedder: TakensEmbedder | None = None,
    maxdim: int = 1,
) -> dict:
    """Test whether an observed ensemble is topologically distinguishable from null.

    Returns
    -------
    dict with keys:
        'observed_stats': summary statistics of the observed ensemble
        'p_values': per-stat p-values
        'z_scores': per-stat z-scores
        'stat_names': names of each statistic
        'significant': bool — True if any stat is significant at p < 0.05
    """
    embedder = embedder or TakensEmbedder(dimension=3, delay=2)

    features_list = []
    for src in ensemble.sources:
        feats = extract_features(src, embedder=embedder, compute_tda=True, maxdim=maxdim)
        features_list.append(feats)

    feature_matrix = np.vstack(features_list)

    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(feature_matrix)

    pd = compute_persistence(scaled, maxdim=maxdim)
    observed_stats = _extract_summary_stats(pd, maxdim)

    p_values = null_dist.p_value(observed_stats)
    z_scores = null_dist.z_score(observed_stats)

    return {
        "observed_stats": observed_stats,
        "p_values": p_values,
        "z_scores": z_scores,
        "stat_names": null_dist.stat_names,
        "significant": bool(np.any(p_values < 0.05)),
        "most_significant_stat": null_dist.stat_names[np.argmin(p_values)],
        "min_p_value": float(np.min(p_values)),
    }


def permutation_test(
    ensemble: Ensemble,
    n_permutations: int = 200,
    test_stat_fn: Optional[Callable] = None,
    embedder: TakensEmbedder | None = None,
    maxdim: int = 1,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> dict:
    """Non-parametric permutation test for topological structure.

    Shuffles the assignment of flux values across sources while
    preserving the noise properties, then compares the observed
    test statistic to the permutation distribution.
    """
    rng = rng or np.random.default_rng()
    embedder = embedder or TakensEmbedder(dimension=3, delay=2)

    if test_stat_fn is None:
        test_stat_fn = lambda pd: pd.total_persistence(1)

    # Observed statistic
    features_list = [
        extract_features(src, embedder=embedder, compute_tda=True, maxdim=maxdim)
        for src in ensemble.sources
    ]
    feature_matrix = np.vstack(features_list)

    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(feature_matrix)

    observed_pd = compute_persistence(scaled, maxdim=maxdim)
    observed_stat = test_stat_fn(observed_pd)

    # Permutation distribution
    perm_stats = []
    all_fluxes = np.concatenate([src.fluxes for src in ensemble.sources
                                  if isinstance(src, LightCurve)])
    for i in range(n_permutations):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}")

        shuffled_fluxes = rng.permutation(all_fluxes)
        idx = 0
        perm_features = []
        for src in ensemble.sources:
            if not isinstance(src, LightCurve):
                continue
            n = src.n_epochs
            perm_lc = LightCurve(
                times=src.times.copy(),
                fluxes=shuffled_fluxes[idx:idx + n],
                flux_errors=src.flux_errors.copy(),
                band=src.band,
            )
            idx += n
            perm_features.append(
                extract_features(perm_lc, embedder=embedder, compute_tda=True, maxdim=maxdim)
            )

        perm_matrix = np.vstack(perm_features)
        perm_scaled = StandardScaler().fit_transform(perm_matrix)
        perm_pd = compute_persistence(perm_scaled, maxdim=maxdim)
        perm_stats.append(test_stat_fn(perm_pd))

    perm_stats = np.array(perm_stats)
    p_value = np.mean(perm_stats >= observed_stat)

    return {
        "observed_stat": float(observed_stat),
        "perm_distribution": perm_stats,
        "p_value": float(p_value),
        "mean_perm": float(np.mean(perm_stats)),
        "std_perm": float(np.std(perm_stats)),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _summary_stat_names(maxdim: int) -> list[str]:
    names = []
    for d in range(maxdim + 1):
        names.extend([
            f"total_persistence_H{d}",
            f"max_persistence_H{d}",
            f"n_features_H{d}",
            f"persistence_entropy_H{d}",
        ])
    return names


def _extract_summary_stats(pd: PersistenceDiagram, maxdim: int) -> np.ndarray:
    stats = []
    for d in range(maxdim + 1):
        stats.extend([
            pd.total_persistence(d),
            pd.max_persistence(d),
            float(pd.n_features(d)),
            pd.persistence_entropy(d),
        ])
    return np.array(stats)
