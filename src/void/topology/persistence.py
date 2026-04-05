"""Persistent homology computation on point clouds.

Wraps ripser and persim to provide a clean interface for computing
persistence diagrams, barcodes, and persistence images from the
point clouds produced by Takens embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from ripser import ripser
except ImportError:
    ripser = None

try:
    from persim import PersistenceImager
except ImportError:
    PersistenceImager = None


@dataclass
class PersistenceDiagram:
    """Container for persistence computation results.

    Attributes
    ----------
    diagrams : list[np.ndarray]
        diagrams[k] is an (n_features, 2) array of (birth, death) pairs
        for H_k homology.  H0 = connected components, H1 = loops, H2 = voids.
    maxdim : int
        Maximum homology dimension computed.
    point_cloud : np.ndarray or None
        The input point cloud, stored for reference.
    metadata : dict
        Additional info (e.g., computation parameters).
    """

    diagrams: list[np.ndarray]
    maxdim: int = 1
    point_cloud: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def total_persistence(self, dim: int = 1) -> float:
        """Sum of all (death - birth) lifetimes in dimension `dim`."""
        dgm = self._finite(dim)
        if len(dgm) == 0:
            return 0.0
        return float(np.sum(dgm[:, 1] - dgm[:, 0]))

    def max_persistence(self, dim: int = 1) -> float:
        """Longest-lived feature in dimension `dim`."""
        dgm = self._finite(dim)
        if len(dgm) == 0:
            return 0.0
        return float(np.max(dgm[:, 1] - dgm[:, 0]))

    def n_features(self, dim: int = 1, min_persistence: float = 0.0) -> int:
        """Count features with persistence > min_persistence."""
        dgm = self._finite(dim)
        if len(dgm) == 0:
            return 0
        lifetimes = dgm[:, 1] - dgm[:, 0]
        return int(np.sum(lifetimes > min_persistence))

    def persistence_entropy(self, dim: int = 1) -> float:
        """Shannon entropy of the persistence lifetime distribution.

        Normalized so that each lifetime is treated as a probability weight.
        Higher entropy = more uniformly distributed feature lifetimes.
        """
        dgm = self._finite(dim)
        if len(dgm) == 0:
            return 0.0
        lifetimes = dgm[:, 1] - dgm[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        if len(lifetimes) == 0:
            return 0.0
        p = lifetimes / lifetimes.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    def persistence_image(
        self,
        dim: int = 1,
        pixel_size: float = 0.1,
        birth_range: Optional[tuple[float, float]] = None,
        pers_range: Optional[tuple[float, float]] = None,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """Compute a persistence image (vectorized representation of the diagram).

        Requires the `persim` package.
        """
        if PersistenceImager is None:
            raise ImportError("persim is required for persistence images")

        dgm = self._finite(dim)
        if len(dgm) == 0:
            return np.zeros((10, 10))

        pimgr = PersistenceImager(
            pixel_size=pixel_size,
            birth_range=birth_range or (0, dgm[:, 0].max() + 0.5),
            pers_range=pers_range or (0, (dgm[:, 1] - dgm[:, 0]).max() + 0.5),
        )
        if sigma is not None:
            pimgr.kernel_params = {"sigma": [[sigma, 0], [0, sigma]]}

        return pimgr.transform(dgm.tolist())

    def barcode(self, dim: int = 1) -> list[tuple[float, float]]:
        """Return the barcode as a list of (birth, death) intervals."""
        dgm = self._finite(dim)
        return [(float(b), float(d)) for b, d in dgm]

    def summary_vector(self) -> np.ndarray:
        """Fixed-length summary across all computed dimensions.

        Returns a vector of [total_pers_H0, max_pers_H0, n_feat_H0, entropy_H0,
                             total_pers_H1, ..., total_pers_Hk, ...]
        """
        parts = []
        for d in range(self.maxdim + 1):
            parts.extend([
                self.total_persistence(d),
                self.max_persistence(d),
                float(self.n_features(d)),
                self.persistence_entropy(d),
            ])
        return np.array(parts)

    def _finite(self, dim: int) -> np.ndarray:
        """Return diagram for dimension `dim`, filtering infinite-death features."""
        if dim >= len(self.diagrams):
            return np.empty((0, 2))
        dgm = self.diagrams[dim]
        if len(dgm) == 0:
            return np.empty((0, 2))
        mask = np.isfinite(dgm[:, 1])
        return dgm[mask]


def compute_persistence(
    point_cloud: np.ndarray,
    maxdim: int = 1,
    thresh: float = np.inf,
    coeff: int = 2,
) -> PersistenceDiagram:
    """Compute Vietoris-Rips persistent homology on a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray of shape (n, d)
        Points in d-dimensional space.
    maxdim : int
        Maximum homology dimension to compute (0=components, 1=loops, 2=voids).
    thresh : float
        Maximum filtration value (distance threshold).
    coeff : int
        Coefficient field for homology (default 2 = Z/2Z).

    Returns
    -------
    PersistenceDiagram
    """
    if ripser is None:
        raise ImportError("ripser is required for persistence computation")

    result = ripser(
        point_cloud,
        maxdim=maxdim,
        thresh=thresh,
        coeff=coeff,
    )

    return PersistenceDiagram(
        diagrams=result["dgms"],
        maxdim=maxdim,
        point_cloud=point_cloud,
        metadata={"thresh": thresh, "coeff": coeff, "n_points": point_cloud.shape[0]},
    )


def compute_persistence_from_distance_matrix(
    distance_matrix: np.ndarray,
    maxdim: int = 1,
    thresh: float = np.inf,
    coeff: int = 2,
) -> PersistenceDiagram:
    """Compute persistent homology from a precomputed distance matrix."""
    if ripser is None:
        raise ImportError("ripser is required for persistence computation")

    result = ripser(
        distance_matrix,
        maxdim=maxdim,
        thresh=thresh,
        coeff=coeff,
        distance_matrix=True,
    )

    return PersistenceDiagram(
        diagrams=result["dgms"],
        maxdim=maxdim,
        metadata={
            "thresh": thresh,
            "coeff": coeff,
            "n_points": distance_matrix.shape[0],
            "from_distance_matrix": True,
        },
    )
