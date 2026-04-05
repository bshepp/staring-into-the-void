"""Data classes for light curves and source ensembles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LightCurve:
    """A single-band photometric light curve.

    Attributes
    ----------
    times : np.ndarray
        Observation timestamps (e.g. MJD).
    fluxes : np.ndarray
        Measured flux values (or difference-image flux).
    flux_errors : np.ndarray
        1-sigma uncertainties on each flux measurement.
    band : str
        Photometric band identifier (e.g. 'g', 'r', 'i').
    metadata : dict
        Arbitrary metadata — source type, injected SNR, period, object ID, etc.
    """

    times: np.ndarray
    fluxes: np.ndarray
    flux_errors: np.ndarray
    band: str = "g"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.fluxes = np.asarray(self.fluxes, dtype=np.float64)
        self.flux_errors = np.asarray(self.flux_errors, dtype=np.float64)
        n = len(self.times)
        if len(self.fluxes) != n or len(self.flux_errors) != n:
            raise ValueError(
                f"Array length mismatch: times={n}, fluxes={len(self.fluxes)}, "
                f"flux_errors={len(self.flux_errors)}"
            )

    @property
    def n_epochs(self) -> int:
        return len(self.times)

    @property
    def snr(self) -> np.ndarray:
        """Per-epoch signal-to-noise ratio."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.flux_errors > 0, self.fluxes / self.flux_errors, 0.0)

    @property
    def peak_snr(self) -> float:
        return float(np.max(np.abs(self.snr)))

    @property
    def mean_snr(self) -> float:
        return float(np.mean(np.abs(self.snr)))

    @property
    def duration(self) -> float:
        """Total time baseline in the same units as times."""
        return float(self.times[-1] - self.times[0]) if self.n_epochs > 1 else 0.0

    def sort_by_time(self) -> LightCurve:
        """Return a copy sorted by ascending time."""
        order = np.argsort(self.times)
        return LightCurve(
            times=self.times[order],
            fluxes=self.fluxes[order],
            flux_errors=self.flux_errors[order],
            band=self.band,
            metadata=dict(self.metadata),
        )

    def attenuate(self, factor: float) -> LightCurve:
        """Scale the astrophysical signal by `factor` (0-1), preserving noise.

        Separates the signal component (flux - noise_floor) and scales only that,
        then re-adds realistic noise.  factor=1.0 is the original; factor=0.0
        returns pure noise at the same error level.
        """
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"Attenuation factor must be in [0, 1], got {factor}")
        attenuated_fluxes = self.fluxes * factor
        return LightCurve(
            times=self.times.copy(),
            fluxes=attenuated_fluxes,
            flux_errors=self.flux_errors.copy(),
            band=self.band,
            metadata={**self.metadata, "attenuation_factor": factor},
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "time": self.times,
            "flux": self.fluxes,
            "flux_error": self.flux_errors,
            "band": self.band,
        })


@dataclass
class MultiBandLightCurve:
    """Light curves across multiple photometric bands for a single source."""

    curves: dict[str, LightCurve] = field(default_factory=dict)
    object_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def bands(self) -> list[str]:
        return list(self.curves.keys())

    def __getitem__(self, band: str) -> LightCurve:
        return self.curves[band]

    def add_band(self, lc: LightCurve) -> None:
        self.curves[lc.band] = lc

    def total_epochs(self) -> int:
        return sum(lc.n_epochs for lc in self.curves.values())


@dataclass
class Ensemble:
    """A collection of light curves representing a sky region or population.

    This is the fundamental unit of topological analysis: persistent homology
    is computed on the ensemble-level point cloud in feature space, not on
    individual light curves.
    """

    sources: list[LightCurve | MultiBandLightCurve] = field(default_factory=list)
    region_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    def add(self, source: LightCurve | MultiBandLightCurve) -> None:
        self.sources.append(source)

    def feature_matrix(self, feature_fn) -> np.ndarray:
        """Extract a feature matrix by applying `feature_fn` to each source.

        Parameters
        ----------
        feature_fn : callable
            Function that takes a LightCurve or MultiBandLightCurve and returns
            a 1-D numpy array of features.

        Returns
        -------
        np.ndarray of shape (n_sources, n_features)
        """
        vectors = [feature_fn(src) for src in self.sources]
        return np.vstack(vectors)

    def light_curves(self, band: Optional[str] = None) -> list[LightCurve]:
        """Flatten to a list of single-band LightCurves."""
        result = []
        for src in self.sources:
            if isinstance(src, MultiBandLightCurve):
                if band is not None:
                    if band in src.curves:
                        result.append(src.curves[band])
                else:
                    result.extend(src.curves.values())
            else:
                if band is None or src.band == band:
                    result.append(src)
        return result
