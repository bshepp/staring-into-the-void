"""Feature extraction from light curves and their Takens embeddings.

Produces fixed-length feature vectors suitable for ensemble-level
topological analysis.  Each light curve becomes a single point in
high-dimensional feature space; persistent homology is then computed
on the resulting point cloud of an entire population.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle

from void.data.models import LightCurve, MultiBandLightCurve
from void.embedding.takens import TakensEmbedder
from void.topology.persistence import PersistenceDiagram, compute_persistence


def extract_features(
    lc: LightCurve,
    embedder: TakensEmbedder | None = None,
    compute_tda: bool = True,
    maxdim: int = 1,
) -> np.ndarray:
    """Extract a fixed-length feature vector from a single light curve.

    Features include statistical, periodicity, and topological summaries.

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    stat_feats = _statistical_features(lc)
    period_feats = _periodicity_features(lc)

    if compute_tda:
        embedder = embedder or TakensEmbedder(dimension=3, delay=1)
        try:
            cloud = embedder.embed(lc)
            pd = compute_persistence(cloud, maxdim=maxdim)
            tda_feats = pd.summary_vector()
        except (ValueError, Exception):
            n_tda = 4 * (maxdim + 1)
            tda_feats = np.zeros(n_tda)
    else:
        tda_feats = np.array([])

    return np.concatenate([stat_feats, period_feats, tda_feats])


def extract_features_multiband(
    mblc: MultiBandLightCurve,
    embedder: TakensEmbedder | None = None,
    bands: list[str] | None = None,
    compute_tda: bool = True,
    maxdim: int = 1,
) -> np.ndarray:
    """Extract features across multiple bands.

    Computes per-band features and cross-band color features.
    """
    bands = bands or sorted(mblc.bands)
    per_band = []
    for b in bands:
        if b in mblc.curves:
            per_band.append(extract_features(mblc[b], embedder, compute_tda, maxdim))
        else:
            per_band.append(np.zeros_like(per_band[0]) if per_band else np.zeros(20))

    color_feats = _color_features(mblc, bands)
    return np.concatenate(per_band + [color_feats])


def feature_names(
    n_bands: int = 1,
    compute_tda: bool = True,
    maxdim: int = 1,
) -> list[str]:
    """Return human-readable names for each feature dimension."""
    stat = [
        "mean_flux", "std_flux", "skew_flux", "kurtosis_flux",
        "median_abs_dev", "iqr_flux", "max_snr", "mean_snr",
        "frac_positive", "linear_slope", "stetson_j",
    ]
    period = ["ls_peak_power", "ls_peak_freq", "ls_fap"]

    if compute_tda:
        tda = []
        for d in range(maxdim + 1):
            tda.extend([
                f"total_pers_H{d}", f"max_pers_H{d}",
                f"n_feat_H{d}", f"entropy_H{d}",
            ])
    else:
        tda = []

    single_band = stat + period + tda
    if n_bands <= 1:
        return single_band

    all_names = []
    for i in range(n_bands):
        all_names.extend([f"band{i}_{n}" for n in single_band])
    all_names.extend([f"color_{i}_{i+1}_mean" for i in range(n_bands - 1)])
    all_names.extend([f"color_{i}_{i+1}_slope" for i in range(n_bands - 1)])
    return all_names


# ---------------------------------------------------------------------------
# Internal feature computations
# ---------------------------------------------------------------------------

def _statistical_features(lc: LightCurve) -> np.ndarray:
    """Basic statistical summaries of the flux time series."""
    f = lc.fluxes
    snr = lc.snr

    mean = np.mean(f)
    std = np.std(f)
    skew = _safe_skew(f)
    kurt = _safe_kurtosis(f)
    mad = np.median(np.abs(f - np.median(f)))
    iqr = np.percentile(f, 75) - np.percentile(f, 25)
    max_snr = float(np.max(np.abs(snr)))
    mean_snr = float(np.mean(np.abs(snr)))
    frac_pos = float(np.mean(f > 0))

    if len(lc.times) > 1:
        t_norm = (lc.times - lc.times[0]) / (lc.times[-1] - lc.times[0])
        slope = np.polyfit(t_norm, f, 1)[0] if std > 0 else 0.0
    else:
        slope = 0.0

    stetson = _stetson_j(lc)

    return np.array([
        mean, std, skew, kurt, mad, iqr,
        max_snr, mean_snr, frac_pos, slope, stetson,
    ])


def _periodicity_features(lc: LightCurve) -> np.ndarray:
    """Lomb-Scargle periodogram features."""
    if lc.n_epochs < 10:
        return np.array([0.0, 0.0, 1.0])

    f_centered = lc.fluxes - np.mean(lc.fluxes)
    duration = lc.times[-1] - lc.times[0]
    if duration <= 0:
        return np.array([0.0, 0.0, 1.0])

    freq_min = 2.0 / duration
    freq_max = 0.5 * lc.n_epochs / duration
    freqs = np.linspace(freq_min, freq_max, min(1000, lc.n_epochs * 5))
    angular_freqs = 2 * np.pi * freqs

    try:
        power = lombscargle(lc.times, f_centered, angular_freqs, normalize=True)
        peak_idx = np.argmax(power)
        peak_power = float(power[peak_idx])
        peak_freq = float(freqs[peak_idx])
        fap = _lombscargle_fap(peak_power, lc.n_epochs, len(freqs))
    except Exception:
        peak_power, peak_freq, fap = 0.0, 0.0, 1.0

    return np.array([peak_power, peak_freq, fap])


def _color_features(
    mblc: MultiBandLightCurve,
    bands: list[str],
) -> np.ndarray:
    """Cross-band color evolution features."""
    features = []
    for i in range(len(bands) - 1):
        b1, b2 = bands[i], bands[i + 1]
        if b1 not in mblc.curves or b2 not in mblc.curves:
            features.extend([0.0, 0.0])
            continue

        lc1, lc2 = mblc[b1], mblc[b2]
        mean_color = np.mean(lc1.fluxes) - np.mean(lc2.fluxes)

        if lc1.n_epochs > 1 and lc2.n_epochs > 1:
            t1_norm = (lc1.times - lc1.times[0]) / max(lc1.duration, 1)
            t2_norm = (lc2.times - lc2.times[0]) / max(lc2.duration, 1)
            s1 = np.polyfit(t1_norm, lc1.fluxes, 1)[0]
            s2 = np.polyfit(t2_norm, lc2.fluxes, 1)[0]
            color_slope = s1 - s2
        else:
            color_slope = 0.0

        features.extend([mean_color, color_slope])

    return np.array(features)


def _safe_skew(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _safe_kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _stetson_j(lc: LightCurve) -> float:
    """Stetson J variability index (single-band approximation)."""
    if lc.n_epochs < 3:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = np.where(
            lc.flux_errors > 0,
            (lc.fluxes - np.mean(lc.fluxes)) / lc.flux_errors,
            0.0,
        )
    n = len(residuals)
    pairs = residuals[:-1] * residuals[1:]
    return float(np.sum(np.sign(pairs) * np.sqrt(np.abs(pairs))) / n)


def _lombscargle_fap(peak_power: float, n_obs: int, n_freq: int) -> float:
    """Approximate false alarm probability for LS peak (Baluev 2008)."""
    if peak_power <= 0 or n_obs < 3:
        return 1.0
    tau = peak_power
    fap_single = (1 - tau) ** ((n_obs - 3) / 2.0)
    fap = 1 - (1 - fap_single) ** n_freq
    return float(np.clip(fap, 0, 1))
