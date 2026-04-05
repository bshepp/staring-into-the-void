"""Recurrence plot embedding — an alternative to Takens delay embedding.

A recurrence plot R(i,j) = Theta(epsilon - ||x(t_i) - x(t_j)||) captures
the recurrence structure of a time series as a binary image.  Persistent
homology can then be applied to this image (via sublevel set filtration)
or to the distance matrix directly.

Key advantage over Takens: handles irregular time sampling naturally,
since the recurrence is defined on the raw observations without interpolation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform

from void.data.models import LightCurve


def recurrence_matrix(
    lc: LightCurve,
    epsilon: Optional[float] = None,
    fraction: float = 0.1,
    normalize: bool = True,
) -> np.ndarray:
    """Compute the recurrence matrix of a light curve.

    Parameters
    ----------
    epsilon : float or None
        Recurrence threshold.  If None, set to `fraction` of the
        maximum pairwise distance.
    fraction : float
        Fraction of max distance to use as epsilon (if epsilon is None).
    normalize : bool
        If True, z-score normalize the flux values first.

    Returns
    -------
    np.ndarray of shape (n_epochs, n_epochs)
        Binary recurrence matrix (1 = recurrent, 0 = not).
    """
    values = lc.fluxes.copy()
    if normalize:
        std = np.std(values)
        if std > 0:
            values = (values - np.mean(values)) / std

    dists = squareform(pdist(values.reshape(-1, 1)))

    if epsilon is None:
        epsilon = fraction * np.max(dists)

    return (dists <= epsilon).astype(np.float64)


def recurrence_distance_matrix(
    lc: LightCurve,
    normalize: bool = True,
) -> np.ndarray:
    """Compute the pairwise distance matrix for recurrence analysis.

    Unlike the binary recurrence matrix, this preserves full distance
    information, making it suitable for Rips filtration directly.
    """
    values = lc.fluxes.copy()
    if normalize:
        std = np.std(values)
        if std > 0:
            values = (values - np.mean(values)) / std

    return squareform(pdist(values.reshape(-1, 1)))


def time_weighted_recurrence(
    lc: LightCurve,
    time_weight: float = 0.1,
    normalize: bool = True,
) -> np.ndarray:
    """Recurrence distance matrix that incorporates temporal distance.

    Points that are close in both value and time are treated as more
    "recurrent" than points close in value but far apart in time.
    This penalizes trivial self-recurrence near the diagonal.

    Parameters
    ----------
    time_weight : float
        Relative weight of temporal distance vs value distance.
    """
    values = lc.fluxes.copy()
    times = lc.times.copy()

    if normalize:
        std_v = np.std(values)
        if std_v > 0:
            values = (values - np.mean(values)) / std_v
        std_t = np.std(times)
        if std_t > 0:
            times = (times - np.mean(times)) / std_t

    # Combined distance: value distance + weighted time distance
    val_dists = squareform(pdist(values.reshape(-1, 1)))
    time_dists = squareform(pdist(times.reshape(-1, 1)))

    return val_dists + time_weight * time_dists


def recurrence_features(rp: np.ndarray) -> dict:
    """Extract standard recurrence quantification analysis (RQA) features.

    Parameters
    ----------
    rp : np.ndarray
        Binary recurrence plot (n x n).

    Returns
    -------
    dict with keys: recurrence_rate, determinism, laminarity,
                    mean_diagonal_length, entropy_diagonal
    """
    n = rp.shape[0]
    recurrence_rate = np.sum(rp) / (n * n)

    diag_lengths = _diagonal_line_lengths(rp)
    vert_lengths = _vertical_line_lengths(rp)

    if len(diag_lengths) > 0:
        total_diag = sum(diag_lengths)
        det_diag = sum(l for l in diag_lengths if l >= 2)
        determinism = det_diag / total_diag if total_diag > 0 else 0
        mean_diag_len = np.mean(diag_lengths)
        hist, _ = np.histogram(diag_lengths,
                                bins=range(1, max(diag_lengths) + 2))
        p = hist / hist.sum()
        p = p[p > 0]
        entropy_diag = float(-np.sum(p * np.log(p)))
    else:
        determinism = 0
        mean_diag_len = 0
        entropy_diag = 0

    if len(vert_lengths) > 0:
        total_vert = sum(vert_lengths)
        lam_vert = sum(l for l in vert_lengths if l >= 2)
        laminarity = lam_vert / total_vert if total_vert > 0 else 0
    else:
        laminarity = 0

    return {
        "recurrence_rate": float(recurrence_rate),
        "determinism": float(determinism),
        "laminarity": float(laminarity),
        "mean_diagonal_length": float(mean_diag_len),
        "entropy_diagonal": float(entropy_diag),
    }


def _diagonal_line_lengths(rp: np.ndarray) -> list[int]:
    """Extract lengths of diagonal lines (excluding main diagonal)."""
    n = rp.shape[0]
    lengths = []
    for k in range(1, n):
        diag = np.diag(rp, k)
        current = 0
        for v in diag:
            if v > 0.5:
                current += 1
            else:
                if current > 0:
                    lengths.append(current)
                current = 0
        if current > 0:
            lengths.append(current)
    return lengths


def _vertical_line_lengths(rp: np.ndarray) -> list[int]:
    """Extract lengths of vertical lines."""
    n = rp.shape[0]
    lengths = []
    for col in range(n):
        current = 0
        for row in range(n):
            if rp[row, col] > 0.5:
                current += 1
            else:
                if current > 0:
                    lengths.append(current)
                current = 0
        if current > 0:
            lengths.append(current)
    return lengths
