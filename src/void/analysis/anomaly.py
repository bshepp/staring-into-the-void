"""Anomaly detection via persistence diagram distance from null.

Identifies sky regions where the topology of sub-threshold forced
photometry deviates significantly from what pure noise would produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from void.data.models import Ensemble, LightCurve
from void.embedding.features import extract_features
from void.embedding.takens import TakensEmbedder
from void.topology.null_model import NullDistribution, _extract_summary_stats
from void.topology.persistence import PersistenceDiagram, compute_persistence


@dataclass
class AnomalyResult:
    """Result of anomaly scoring for a single ensemble/region."""

    region_id: Optional[str]
    anomaly_score: float
    p_values: np.ndarray
    z_scores: np.ndarray
    stat_names: list[str]
    observed_stats: np.ndarray
    persistence_diagram: Optional[PersistenceDiagram] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        """Significant at the 5% level (Bonferroni-corrected)."""
        corrected_alpha = 0.05 / len(self.p_values)
        return bool(np.any(self.p_values < corrected_alpha))

    @property
    def most_anomalous_stat(self) -> str:
        idx = np.argmin(self.p_values)
        return self.stat_names[idx]

    @property
    def min_p_value(self) -> float:
        return float(np.min(self.p_values))

    def summary(self) -> str:
        lines = [
            f"Region: {self.region_id or 'unnamed'}",
            f"Anomaly score: {self.anomaly_score:.4f}",
            f"Anomalous: {self.is_anomalous}",
            f"Most anomalous: {self.most_anomalous_stat} "
            f"(p={self.min_p_value:.4f}, z={self.z_scores[np.argmin(self.p_values)]:.2f})",
        ]
        return "\n".join(lines)


class AnomalyDetector:
    """Detect topologically anomalous sky regions against a null model.

    Usage:
        detector = AnomalyDetector(null_distribution)
        result = detector.score(ensemble)
    """

    def __init__(
        self,
        null_dist: NullDistribution,
        embedder: TakensEmbedder | None = None,
        maxdim: int = 1,
    ):
        self.null_dist = null_dist
        self.embedder = embedder or TakensEmbedder(dimension=3, delay=2)
        self.maxdim = maxdim

    def score(self, ensemble: Ensemble) -> AnomalyResult:
        """Compute anomaly score for a single ensemble."""
        features_list = []
        for src in ensemble.sources:
            feats = extract_features(
                src, embedder=self.embedder, compute_tda=True, maxdim=self.maxdim
            )
            features_list.append(feats)

        feature_matrix = np.vstack(features_list)
        scaled = StandardScaler().fit_transform(feature_matrix)

        pd = compute_persistence(scaled, maxdim=self.maxdim)
        observed = _extract_summary_stats(pd, self.maxdim)

        p_values = self.null_dist.p_value(observed)
        z_scores = self.null_dist.z_score(observed)

        # Composite anomaly score: sum of squared z-scores across all stats
        anomaly_score = float(np.sum(z_scores**2))

        return AnomalyResult(
            region_id=ensemble.region_id,
            anomaly_score=anomaly_score,
            p_values=p_values,
            z_scores=z_scores,
            stat_names=self.null_dist.stat_names,
            observed_stats=observed,
            persistence_diagram=pd,
            metadata={"n_sources": ensemble.n_sources},
        )

    def score_multiple(self, ensembles: list[Ensemble]) -> list[AnomalyResult]:
        """Score multiple ensembles (e.g., tiled sky regions)."""
        return [self.score(e) for e in ensembles]

    def rank(self, results: list[AnomalyResult]) -> list[AnomalyResult]:
        """Rank results by anomaly score (most anomalous first)."""
        return sorted(results, key=lambda r: r.anomaly_score, reverse=True)


def inject_and_detect(
    n_signal: int,
    n_noise: int,
    signal_generator,
    signal_kwargs: dict,
    noise_kwargs: dict,
    null_dist: NullDistribution,
    embedder: TakensEmbedder | None = None,
    maxdim: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> AnomalyResult:
    """Convenience: generate an ensemble with injected signals and score it.

    Useful for power analysis — sweep over signal parameters and measure
    detection rates.
    """
    from void.data.synthetic import generate_ensemble

    rng = rng or np.random.default_rng()
    ensemble = generate_ensemble(
        n_signal=n_signal,
        n_noise=n_noise,
        signal_generator=signal_generator,
        signal_kwargs=signal_kwargs,
        noise_kwargs=noise_kwargs,
        rng=rng,
    )

    detector = AnomalyDetector(null_dist, embedder=embedder, maxdim=maxdim)
    return detector.score(ensemble)


def power_analysis(
    snr_levels: list[float],
    population_sizes: list[int],
    n_trials: int = 20,
    n_noise: int = 200,
    signal_generator=None,
    signal_kwargs_base: dict | None = None,
    noise_kwargs: dict | None = None,
    null_dist: NullDistribution | None = None,
    embedder: TakensEmbedder | None = None,
    maxdim: int = 1,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> dict:
    """Run a systematic power analysis: detection rate vs (SNR, population size).

    Returns
    -------
    dict with keys:
        'detection_rates': np.ndarray of shape (len(snr_levels), len(population_sizes))
        'snr_levels': list of SNR values tested
        'population_sizes': list of population sizes tested
        'results': nested dict of AnomalyResults
    """
    from void.data.synthetic import generate_periodic
    from void.topology.null_model import build_null_distribution

    rng = rng or np.random.default_rng()
    signal_generator = signal_generator or generate_periodic
    signal_kwargs_base = signal_kwargs_base or {"period": 30.0, "n_epochs": 200}
    noise_kwargs = noise_kwargs or {"n_epochs": 200}
    embedder = embedder or TakensEmbedder(dimension=3, delay=2)

    if null_dist is None:
        if verbose:
            print("Building null distribution...")
        null_dist = build_null_distribution(
            n_realizations=50,
            n_sources_per=n_noise,
            embedder=embedder,
            maxdim=maxdim,
            rng=np.random.default_rng(rng.integers(0, 2**63)),
            verbose=verbose,
        )

    detection_rates = np.zeros((len(snr_levels), len(population_sizes)))
    all_results = {}

    for i, snr in enumerate(snr_levels):
        for j, n_sig in enumerate(population_sizes):
            if verbose:
                print(f"Testing SNR={snr:.1f}, N_signal={n_sig}...")

            detections = 0
            trial_results = []

            for trial in range(n_trials):
                trial_rng = np.random.default_rng(rng.integers(0, 2**63))
                sig_kwargs = {**signal_kwargs_base, "snr": snr}
                result = inject_and_detect(
                    n_signal=n_sig,
                    n_noise=n_noise,
                    signal_generator=signal_generator,
                    signal_kwargs=sig_kwargs,
                    noise_kwargs=noise_kwargs,
                    null_dist=null_dist,
                    embedder=embedder,
                    maxdim=maxdim,
                    rng=trial_rng,
                )
                trial_results.append(result)
                if result.is_anomalous:
                    detections += 1

            detection_rates[i, j] = detections / n_trials
            all_results[(snr, n_sig)] = trial_results

            if verbose:
                print(f"  Detection rate: {detection_rates[i, j]:.1%}")

    return {
        "detection_rates": detection_rates,
        "snr_levels": snr_levels,
        "population_sizes": population_sizes,
        "results": all_results,
        "null_distribution": null_dist,
    }
