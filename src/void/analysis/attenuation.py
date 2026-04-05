"""Attenuation experiment framework.

Takes well-characterized light curves (high SNR, known type), attenuates
the signal to sub-threshold levels, and measures whether the topological
pipeline can still recover the population signal.

This is the key validation experiment: it uses real astrophysical signals
(not synthetic models) and tests recovery at controlled attenuation levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from void.data.models import Ensemble, LightCurve, MultiBandLightCurve
from void.embedding.features import extract_features
from void.embedding.takens import TakensEmbedder
from void.topology.null_model import NullDistribution, _extract_summary_stats
from void.topology.persistence import PersistenceDiagram, compute_persistence


@dataclass
class AttenuationResult:
    """Result for a single attenuation level."""

    factor: float
    effective_snr: float
    total_persistence_h1: float
    persistence_entropy_h1: float
    n_features_h1: int
    p_value: float
    z_score: float
    detected: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class AttenuationExperiment:
    """Full results of an attenuation sweep."""

    results: list[AttenuationResult]
    null_mean_h1: float
    null_std_h1: float
    source_type: str
    n_sources: int

    @property
    def factors(self) -> list[float]:
        return [r.factor for r in self.results]

    @property
    def effective_snrs(self) -> list[float]:
        return [r.effective_snr for r in self.results]

    @property
    def detection_curve(self) -> list[bool]:
        return [r.detected for r in self.results]

    @property
    def recovery_threshold(self) -> float:
        """Smallest attenuation factor where detection succeeds."""
        for r in sorted(self.results, key=lambda x: x.factor):
            if r.detected:
                return r.factor
        return float("inf")


def attenuate_light_curve(lc: LightCurve, factor: float) -> LightCurve:
    """Attenuate a light curve's signal by a multiplicative factor.

    The true astrophysical signal is scaled by `factor` and realistic
    noise is re-added at the original error level.  This simulates
    observing the same source at a greater distance or fainter intrinsic
    luminosity.

    Parameters
    ----------
    factor : float
        Attenuation factor in [0, 1].  1.0 = original signal, 0.0 = pure noise.
    """
    attenuated = lc.attenuate(factor)
    rng = np.random.default_rng()
    new_noise = rng.normal(0, lc.flux_errors)
    attenuated.fluxes = lc.fluxes * factor + new_noise * (1 - factor)
    return attenuated


def run_attenuation_experiment(
    source_light_curves: list[LightCurve],
    noise_light_curves: list[LightCurve],
    attenuation_factors: list[float] | None = None,
    null_dist: NullDistribution | None = None,
    embedder: TakensEmbedder | None = None,
    maxdim: int = 1,
    source_type: str = "unknown",
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> AttenuationExperiment:
    """Run a full attenuation sweep on real light curves.

    Parameters
    ----------
    source_light_curves : list[LightCurve]
        High-SNR light curves of known type (e.g. RR Lyrae from ZTF).
    noise_light_curves : list[LightCurve]
        Background noise light curves (field stars or synthetic noise).
    attenuation_factors : list[float]
        Factors to test (default: 0.05 to 1.0 in steps).
    null_dist : NullDistribution
        Precomputed null distribution for significance testing.
    """
    from void.data.synthetic import generate_noise
    from void.topology.null_model import build_null_distribution

    rng = rng or np.random.default_rng()
    embedder = embedder or TakensEmbedder(dimension=3, delay=2)
    attenuation_factors = attenuation_factors or [
        0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ]

    if not noise_light_curves:
        noise_light_curves = [
            generate_noise(n_epochs=200, rng=np.random.default_rng(rng.integers(0, 2**63)))
            for _ in range(200)
        ]

    n_total = len(source_light_curves) + len(noise_light_curves)
    if null_dist is None:
        if verbose:
            print("Building null distribution...")
        null_dist = build_null_distribution(
            n_realizations=50,
            n_sources_per=n_total,
            embedder=embedder,
            maxdim=maxdim,
            rng=np.random.default_rng(rng.integers(0, 2**63)),
            verbose=verbose,
        )

    null_h1_idx = null_dist.stat_names.index("total_persistence_H1")
    null_mean_h1 = float(null_dist.mean()[null_h1_idx])
    null_std_h1 = float(null_dist.std()[null_h1_idx])

    results = []
    for factor in sorted(attenuation_factors):
        if verbose:
            print(f"  Attenuation factor = {factor:.2f}...")

        attenuated_sources = [
            attenuate_light_curve(lc, factor) for lc in source_light_curves
        ]

        effective_snrs = [lc.mean_snr for lc in attenuated_sources]
        mean_eff_snr = float(np.mean(effective_snrs)) if effective_snrs else 0.0

        all_lcs = attenuated_sources + noise_light_curves

        from sklearn.preprocessing import StandardScaler
        features_list = [
            extract_features(lc, embedder=embedder, compute_tda=True, maxdim=maxdim)
            for lc in all_lcs
        ]
        feature_matrix = np.vstack(features_list)
        scaled = StandardScaler().fit_transform(feature_matrix)

        pd = compute_persistence(scaled, maxdim=maxdim)
        stats = _extract_summary_stats(pd, maxdim)

        p_values = null_dist.p_value(stats)
        z_scores = null_dist.z_score(stats)

        corrected_alpha = 0.05 / len(p_values)
        detected = bool(np.any(p_values < corrected_alpha))

        results.append(AttenuationResult(
            factor=factor,
            effective_snr=mean_eff_snr,
            total_persistence_h1=pd.total_persistence(1),
            persistence_entropy_h1=pd.persistence_entropy(1),
            n_features_h1=pd.n_features(1),
            p_value=float(np.min(p_values)),
            z_score=float(z_scores[null_h1_idx]),
            detected=detected,
        ))

        if verbose:
            status = "DETECTED" if detected else "not detected"
            print(f"    eff_SNR={mean_eff_snr:.2f}, H1={pd.total_persistence(1):.3f}, "
                  f"p={np.min(p_values):.4f} → {status}")

    return AttenuationExperiment(
        results=results,
        null_mean_h1=null_mean_h1,
        null_std_h1=null_std_h1,
        source_type=source_type,
        n_sources=len(source_light_curves),
    )


def compare_with_classical(
    source_light_curves: list[LightCurve],
    noise_light_curves: list[LightCurve],
    attenuation_factors: list[float] | None = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> dict:
    """Compare TDA recovery with classical methods (Lomb-Scargle, stacking).

    For each attenuation level, measure whether each method can distinguish
    the attenuated signal population from noise.
    """
    from scipy.signal import lombscargle

    rng = rng or np.random.default_rng()
    attenuation_factors = attenuation_factors or [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    results = {"factors": attenuation_factors, "ls_power": [], "stack_snr": []}

    for factor in attenuation_factors:
        attenuated = [attenuate_light_curve(lc, factor) for lc in source_light_curves]

        # Lomb-Scargle: mean peak power across attenuated sources
        ls_powers = []
        for lc in attenuated:
            if lc.n_epochs < 10:
                continue
            f_centered = lc.fluxes - np.mean(lc.fluxes)
            duration = lc.times[-1] - lc.times[0]
            if duration <= 0:
                continue
            freqs = np.linspace(2.0 / duration, 0.5 * lc.n_epochs / duration, 500)
            try:
                power = lombscargle(lc.times, f_centered, 2 * np.pi * freqs, normalize=True)
                ls_powers.append(np.max(power))
            except Exception:
                pass
        results["ls_power"].append(np.mean(ls_powers) if ls_powers else 0.0)

        # Simple stacking: co-add all attenuated source fluxes
        all_fluxes = np.concatenate([lc.fluxes for lc in attenuated])
        all_errors = np.concatenate([lc.flux_errors for lc in attenuated])
        weights = 1.0 / (all_errors**2 + 1e-12)
        weighted_mean = np.sum(all_fluxes * weights) / np.sum(weights)
        weighted_err = 1.0 / np.sqrt(np.sum(weights))
        stack_snr = abs(weighted_mean) / weighted_err if weighted_err > 0 else 0.0
        results["stack_snr"].append(float(stack_snr))

        if verbose:
            print(f"  factor={factor:.2f}: LS_power={results['ls_power'][-1]:.4f}, "
                  f"stack_SNR={stack_snr:.2f}")

    return results
