"""Persistence diagram distance metrics.

Provides Wasserstein, bottleneck, and persistence image distances
for comparing diagrams — particularly for measuring the distance
between an observed persistence diagram and the null distribution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from void.topology.persistence import PersistenceDiagram


def wasserstein_distance(
    pd1: PersistenceDiagram,
    pd2: PersistenceDiagram,
    dim: int = 1,
    p: float = 2.0,
) -> float:
    """p-Wasserstein distance between two persistence diagrams.

    Uses the Hungarian algorithm on the extended diagram (including
    projections to the diagonal).
    """
    dgm1 = pd1._finite(dim)
    dgm2 = pd2._finite(dim)
    return _wasserstein_raw(dgm1, dgm2, p=p)


def bottleneck_distance(
    pd1: PersistenceDiagram,
    pd2: PersistenceDiagram,
    dim: int = 1,
) -> float:
    """Bottleneck distance (L∞ Wasserstein) between two persistence diagrams."""
    dgm1 = pd1._finite(dim)
    dgm2 = pd2._finite(dim)
    return _bottleneck_raw(dgm1, dgm2)


def persistence_image_distance(
    pd1: PersistenceDiagram,
    pd2: PersistenceDiagram,
    dim: int = 1,
    pixel_size: float = 0.1,
) -> float:
    """L2 distance between persistence images."""
    try:
        img1 = pd1.persistence_image(dim=dim, pixel_size=pixel_size)
        img2 = pd2.persistence_image(dim=dim, pixel_size=pixel_size)
    except ImportError:
        raise ImportError("persim is required for persistence image distances")

    img1_flat = np.asarray(img1).ravel()
    img2_flat = np.asarray(img2).ravel()

    min_len = min(len(img1_flat), len(img2_flat))
    return float(np.linalg.norm(img1_flat[:min_len] - img2_flat[:min_len]))


def distance_from_null(
    observed: PersistenceDiagram,
    null_diagrams: list[PersistenceDiagram],
    dim: int = 1,
    metric: str = "wasserstein",
) -> dict:
    """Compute distance of an observed diagram from a null distribution.

    Returns
    -------
    dict with keys:
        'distance_to_mean': distance from observed to the mean null diagram
        'distances': list of distances from observed to each null realization
        'z_score': how many std deviations the observed distance is from the
                   mean pairwise null distance
        'p_value': fraction of null-null distances exceeding the observed distance
    """
    dist_fn = {
        "wasserstein": wasserstein_distance,
        "bottleneck": bottleneck_distance,
    }.get(metric, wasserstein_distance)

    obs_to_null = [dist_fn(observed, nd, dim=dim) for nd in null_diagrams]

    n_null = len(null_diagrams)
    null_to_null = []
    for i in range(n_null):
        for j in range(i + 1, n_null):
            null_to_null.append(dist_fn(null_diagrams[i], null_diagrams[j], dim=dim))

    mean_obs = np.mean(obs_to_null)
    mean_null = np.mean(null_to_null) if null_to_null else 0.0
    std_null = np.std(null_to_null) if null_to_null else 1e-12

    z_score = (mean_obs - mean_null) / (std_null + 1e-12)
    p_value = np.mean(np.array(null_to_null) >= mean_obs) if null_to_null else 0.0

    return {
        "distance_to_mean": mean_obs,
        "distances": obs_to_null,
        "z_score": float(z_score),
        "p_value": float(p_value),
        "null_mean": float(mean_null),
        "null_std": float(std_null),
    }


# ---------------------------------------------------------------------------
# Raw distance computations
# ---------------------------------------------------------------------------

def _wasserstein_raw(dgm1: np.ndarray, dgm2: np.ndarray, p: float = 2.0) -> float:
    """Wasserstein distance between two (birth, death) arrays."""
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0

    dgm1 = dgm1 if len(dgm1) > 0 else np.empty((0, 2))
    dgm2 = dgm2 if len(dgm2) > 0 else np.empty((0, 2))

    n1, n2 = len(dgm1), len(dgm2)

    # Diagonal projections: midpoint of (birth, death) → (mid, mid)
    diag1 = np.column_stack([
        (dgm1[:, 0] + dgm1[:, 1]) / 2,
        (dgm1[:, 0] + dgm1[:, 1]) / 2,
    ]) if n1 > 0 else np.empty((0, 2))

    diag2 = np.column_stack([
        (dgm2[:, 0] + dgm2[:, 1]) / 2,
        (dgm2[:, 0] + dgm2[:, 1]) / 2,
    ]) if n2 > 0 else np.empty((0, 2))

    # Augmented diagrams: each real point can match to diagonal
    aug1 = np.vstack([dgm1, diag2]) if n2 > 0 else dgm1.copy()
    aug2 = np.vstack([dgm2, diag1]) if n1 > 0 else dgm2.copy()

    if len(aug1) == 0 or len(aug2) == 0:
        return 0.0

    size = max(len(aug1), len(aug2))
    if len(aug1) < size:
        pad = np.zeros((size - len(aug1), 2))
        aug1 = np.vstack([aug1, pad])
    if len(aug2) < size:
        pad = np.zeros((size - len(aug2), 2))
        aug2 = np.vstack([aug2, pad])

    cost = cdist(aug1, aug2, metric="chebyshev") ** p
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(np.sum(cost[row_ind, col_ind]) ** (1.0 / p))


def _bottleneck_raw(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Bottleneck distance (approximation via optimal assignment)."""
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0

    dgm1 = dgm1 if len(dgm1) > 0 else np.empty((0, 2))
    dgm2 = dgm2 if len(dgm2) > 0 else np.empty((0, 2))

    n1, n2 = len(dgm1), len(dgm2)

    diag1 = np.column_stack([
        (dgm1[:, 0] + dgm1[:, 1]) / 2,
        (dgm1[:, 0] + dgm1[:, 1]) / 2,
    ]) if n1 > 0 else np.empty((0, 2))
    diag2 = np.column_stack([
        (dgm2[:, 0] + dgm2[:, 1]) / 2,
        (dgm2[:, 0] + dgm2[:, 1]) / 2,
    ]) if n2 > 0 else np.empty((0, 2))

    aug1 = np.vstack([dgm1, diag2]) if n2 > 0 else dgm1.copy()
    aug2 = np.vstack([dgm2, diag1]) if n1 > 0 else dgm2.copy()

    if len(aug1) == 0 or len(aug2) == 0:
        return 0.0

    size = max(len(aug1), len(aug2))
    if len(aug1) < size:
        aug1 = np.vstack([aug1, np.zeros((size - len(aug1), 2))])
    if len(aug2) < size:
        aug2 = np.vstack([aug2, np.zeros((size - len(aug2), 2))])

    cost = cdist(aug1, aug2, metric="chebyshev")
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(np.max(cost[row_ind, col_ind]))
