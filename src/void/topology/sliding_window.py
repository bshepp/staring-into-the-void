"""Sliding window persistence — track topological evolution over time.

Computes persistent homology in rolling time windows across a light curve
or ensemble, producing persistence vineyards / Crocker plots that reveal
topological phase transitions.

This is critical for Stage 5 of the pipeline: detecting when sub-threshold
populations emerge or change structure over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from void.data.models import LightCurve
from void.embedding.takens import TakensEmbedder
from void.topology.persistence import PersistenceDiagram, compute_persistence


@dataclass
class WindowResult:
    """Persistence result for a single time window."""

    window_start: float
    window_end: float
    center: float
    diagram: PersistenceDiagram
    n_points: int


@dataclass
class SlidingWindowResult:
    """Full sliding window analysis result."""

    windows: list[WindowResult]
    window_size: float
    step_size: float
    metadata: dict = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def centers(self) -> np.ndarray:
        return np.array([w.center for w in self.windows])

    def total_persistence_series(self, dim: int = 1) -> np.ndarray:
        """Time series of total persistence in dimension `dim`."""
        return np.array([w.diagram.total_persistence(dim) for w in self.windows])

    def max_persistence_series(self, dim: int = 1) -> np.ndarray:
        return np.array([w.diagram.max_persistence(dim) for w in self.windows])

    def n_features_series(self, dim: int = 1) -> np.ndarray:
        return np.array([w.diagram.n_features(dim) for w in self.windows])

    def entropy_series(self, dim: int = 1) -> np.ndarray:
        return np.array([w.diagram.persistence_entropy(dim) for w in self.windows])

    def summary_matrix(self, dim: int = 1) -> np.ndarray:
        """Matrix of (center, total_pers, max_pers, n_feat, entropy) per window."""
        return np.column_stack([
            self.centers,
            self.total_persistence_series(dim),
            self.max_persistence_series(dim),
            self.n_features_series(dim),
            self.entropy_series(dim),
        ])


def sliding_window_persistence(
    lc: LightCurve,
    window_size: float = 365.25,
    step_size: float = 30.0,
    embedder: TakensEmbedder | None = None,
    maxdim: int = 1,
    min_points: int = 20,
) -> SlidingWindowResult:
    """Compute persistence diagrams in sliding time windows across a light curve.

    Parameters
    ----------
    window_size : float
        Width of each window in the same units as lc.times (typically days).
    step_size : float
        Step between consecutive windows.
    min_points : int
        Minimum number of data points required in a window to compute persistence.
    """
    embedder = embedder or TakensEmbedder(dimension=3, delay=1)

    t_start = lc.times[0]
    t_end = lc.times[-1]

    windows = []
    current = t_start

    while current + window_size <= t_end:
        mask = (lc.times >= current) & (lc.times < current + window_size)
        n_in_window = np.sum(mask)

        if n_in_window >= min_points:
            window_lc = LightCurve(
                times=lc.times[mask],
                fluxes=lc.fluxes[mask],
                flux_errors=lc.flux_errors[mask],
                band=lc.band,
            )
            try:
                cloud = embedder.embed(window_lc)
                pd = compute_persistence(cloud, maxdim=maxdim)
                windows.append(WindowResult(
                    window_start=current,
                    window_end=current + window_size,
                    center=current + window_size / 2,
                    diagram=pd,
                    n_points=int(n_in_window),
                ))
            except (ValueError, Exception):
                pass

        current += step_size

    return SlidingWindowResult(
        windows=windows,
        window_size=window_size,
        step_size=step_size,
        metadata={"band": lc.band, "total_epochs": lc.n_epochs},
    )


def detect_topological_transitions(
    result: SlidingWindowResult,
    dim: int = 1,
    threshold_sigma: float = 2.0,
) -> list[dict]:
    """Detect abrupt changes in topological summary statistics.

    Returns a list of detected transitions with their locations and magnitudes.
    """
    series = result.total_persistence_series(dim)
    if len(series) < 3:
        return []

    diffs = np.abs(np.diff(series))
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    transitions = []
    for i, d in enumerate(diffs):
        if d > mean_diff + threshold_sigma * std_diff:
            transitions.append({
                "index": i,
                "time": float(result.centers[i]),
                "magnitude": float(d),
                "z_score": float((d - mean_diff) / (std_diff + 1e-12)),
                "before": float(series[i]),
                "after": float(series[i + 1]),
            })

    return transitions
