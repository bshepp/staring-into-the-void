"""Takens delay embedding for time series to point cloud conversion.

Implements the Takens delay embedding theorem: a scalar time series x(t)
is embedded into d-dimensional phase space by constructing vectors
    v(t) = [x(t), x(t-tau), x(t-2*tau), ..., x(t-(d-1)*tau)]

The resulting point cloud preserves the topological properties of the
underlying dynamical system — periodic sources produce loops, chaotic
sources produce strange attractors, noise produces featureless clouds.

Handles irregular time sampling via interpolation or natural-cadence
strategies.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

from void.data.models import LightCurve, MultiBandLightCurve


def _dedupe_times(times: np.ndarray, fluxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate timestamps by averaging flux values.

    Required before `scipy.interpolate.interp1d`, which divides by
    `x_hi - x_lo` and would emit a divide-by-zero warning at duplicate x.
    Inputs are assumed already sorted by time.
    """
    if len(times) == 0:
        return times, fluxes
    unique_t, inverse = np.unique(times, return_inverse=True)
    if len(unique_t) == len(times):
        return times, fluxes
    summed = np.zeros_like(unique_t, dtype=float)
    counts = np.zeros_like(unique_t, dtype=float)
    np.add.at(summed, inverse, fluxes)
    np.add.at(counts, inverse, 1.0)
    return unique_t, summed / counts


class TakensEmbedder:
    """Takens delay embedding for light curves.

    Parameters
    ----------
    dimension : int
        Embedding dimension d.  Higher values capture more complex dynamics
        but require more data points.
    delay : int
        Delay in units of the (resampled) time step index.
    stride : int
        Step between consecutive embedding vectors.
    interpolation : str
        Strategy for handling irregular sampling before embedding.
        'linear', 'cubic', or 'none' (use raw cadence directly).
    n_resample : int or None
        Number of uniformly spaced points to interpolate to.
        If None, uses the number of original observations.
    """

    def __init__(
        self,
        dimension: int = 3,
        delay: int = 1,
        stride: int = 1,
        interpolation: str = "linear",
        n_resample: int | None = None,
    ):
        self.dimension = dimension
        self.delay = delay
        self.stride = stride
        self.interpolation = interpolation
        self.n_resample = n_resample

    def embed(self, lc: LightCurve) -> np.ndarray:
        """Embed a single light curve into phase space.

        Returns
        -------
        np.ndarray of shape (n_points, dimension)
            The point cloud in reconstructed phase space.
        """
        if self.interpolation != "none":
            series = self._resample(lc)
        else:
            series = lc.fluxes.copy()

        return self._delay_embed(series)

    def embed_multiband(
        self,
        mblc: MultiBandLightCurve,
        mode: str = "concatenate",
    ) -> np.ndarray:
        """Embed a multi-band light curve.

        Parameters
        ----------
        mode : str
            'concatenate' — embed each band, horizontally stack columns.
            'interleave'  — interleave band fluxes into a single series, then embed.
        """
        if mode == "concatenate":
            clouds = []
            for band in sorted(mblc.bands):
                cloud = self.embed(mblc[band])
                clouds.append(cloud)
            min_len = min(c.shape[0] for c in clouds)
            return np.hstack([c[:min_len] for c in clouds])

        elif mode == "interleave":
            all_times = np.concatenate([mblc[b].times for b in mblc.bands])
            all_fluxes = np.concatenate([mblc[b].fluxes for b in mblc.bands])
            order = np.argsort(all_times)
            combined = LightCurve(
                times=all_times[order],
                fluxes=all_fluxes[order],
                flux_errors=np.ones_like(all_fluxes[order]),
                band="combined",
            )
            return self.embed(combined)

        raise ValueError(f"Unknown mode: {mode}")

    def _resample(self, lc: LightCurve) -> np.ndarray:
        """Interpolate to a regular time grid.

        Duplicate timestamps (e.g. from interleaved multi-band series) are
        collapsed by averaging fluxes at identical times before interpolation
        to avoid divide-by-zero in `scipy.interpolate.interp1d`.
        """
        n = self.n_resample or lc.n_epochs
        times, fluxes = _dedupe_times(lc.times, lc.fluxes)
        t_regular = np.linspace(times[0], times[-1], n)
        kind = self.interpolation if self.interpolation in ("linear", "cubic") else "linear"
        f = interp1d(times, fluxes, kind=kind, fill_value="extrapolate")
        return f(t_regular)

    def _delay_embed(self, series: np.ndarray) -> np.ndarray:
        """Construct delay-coordinate vectors from a 1-D series."""
        n = len(series)
        window = (self.dimension - 1) * self.delay
        if window >= n:
            raise ValueError(
                f"Series length {n} too short for dimension={self.dimension}, "
                f"delay={self.delay} (requires >= {window + 1} points)"
            )
        n_vectors = (n - window - 1) // self.stride + 1
        indices = np.arange(n_vectors) * self.stride
        cols = [series[indices + i * self.delay] for i in range(self.dimension)]
        return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Optimal parameter selection
# ---------------------------------------------------------------------------

def optimal_delay(
    lc: LightCurve,
    max_lag: int = 50,
    method: str = "mutual_information",
    n_resample: int | None = None,
) -> int:
    """Estimate optimal time delay tau for Takens embedding.

    Uses the first minimum of time-delayed mutual information (Fraser & Swinney 1986),
    or alternatively the first zero-crossing of the autocorrelation function.
    """
    n = n_resample or lc.n_epochs
    t_regular = np.linspace(lc.times[0], lc.times[-1], n)
    f = interp1d(lc.times, lc.fluxes, kind="linear", fill_value="extrapolate")
    series = f(t_regular)
    series = (series - np.mean(series)) / (np.std(series) + 1e-12)

    max_lag = min(max_lag, len(series) // 3)

    if method == "autocorrelation":
        acf = np.correlate(series, series, mode="full")
        acf = acf[len(acf) // 2:]
        acf = acf / acf[0]
        zero_crossings = np.where(np.diff(np.sign(acf)))[0]
        return int(zero_crossings[0]) + 1 if len(zero_crossings) > 0 else 1

    # Mutual information via histogram estimator
    n_bins = max(10, int(np.sqrt(len(series))))
    mi_values = np.zeros(max_lag)
    for lag in range(1, max_lag):
        x = series[:-lag]
        y = series[lag:]
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        mask = pxy > 0
        mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px[:, None] * py[None, :])[mask]))
        mi_values[lag] = mi

    mi_values[0] = mi_values[1] + 1
    local_minima = argrelextrema(mi_values[1:], np.less)[0]
    return int(local_minima[0]) + 1 if len(local_minima) > 0 else 1


def optimal_dimension(
    lc: LightCurve,
    delay: int = 1,
    max_dim: int = 10,
    threshold: float = 0.01,
    n_resample: int | None = None,
) -> int:
    """Estimate optimal embedding dimension via false nearest neighbors.

    Increases dimension until the fraction of false nearest neighbors
    drops below threshold.
    """
    n = n_resample or lc.n_epochs
    t_regular = np.linspace(lc.times[0], lc.times[-1], n)
    f = interp1d(lc.times, lc.fluxes, kind="linear", fill_value="extrapolate")
    series = f(t_regular)
    series = (series - np.mean(series)) / (np.std(series) + 1e-12)

    r_tol = 15.0
    a_tol = 2.0
    sigma = np.std(series)

    for d in range(1, max_dim + 1):
        embedder = TakensEmbedder(dimension=d, delay=delay, interpolation="none")
        try:
            cloud = embedder._delay_embed(series)
        except ValueError:
            return d

        embedder_next = TakensEmbedder(dimension=d + 1, delay=delay, interpolation="none")
        try:
            cloud_next = embedder_next._delay_embed(series)
        except ValueError:
            return d

        n_pts = min(cloud.shape[0], cloud_next.shape[0])
        cloud = cloud[:n_pts]
        cloud_next = cloud_next[:n_pts]

        n_false = 0
        n_check = min(500, n_pts)
        idx_check = np.random.choice(n_pts, size=n_check, replace=False)

        for i in idx_check:
            dists = np.linalg.norm(cloud - cloud[i], axis=1)
            dists[i] = np.inf
            nn = np.argmin(dists)
            r_d = dists[nn]
            if r_d < 1e-10:
                continue
            r_d1 = np.linalg.norm(cloud_next[i] - cloud_next[nn])
            extra = abs(cloud_next[i, -1] - cloud_next[nn, -1])

            if extra / r_d > r_tol or r_d1 / (sigma * a_tol) > 1:
                n_false += 1

        fnn_frac = n_false / n_check
        if fnn_frac < threshold:
            return d

    return max_dim
