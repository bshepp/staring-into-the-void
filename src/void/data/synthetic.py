"""Synthetic light curve generation for controlled pipeline validation.

Source types:
- Periodic (sinusoidal, sawtooth, eclipsing-binary-like)
- Transient (Bazin function for SN-like profiles)
- Slow-evolving (linear trend, sigmoid for changing-look AGN)
- Stochastic (damped random walk for AGN variability)
- Noise-only (pure Gaussian noise at a given error level)

All generators produce LightCurve objects with LSST-like irregular cadence
and configurable signal-to-noise ratio.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from void.data.models import Ensemble, LightCurve, MultiBandLightCurve


# ---------------------------------------------------------------------------
# Cadence generation
# ---------------------------------------------------------------------------

def lsst_cadence(
    n_epochs: int = 200,
    baseline_days: float = 3650.0,
    season_fraction: float = 0.6,
    min_gap_days: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate LSST-like irregular observation timestamps.

    Simulates seasonal visibility windows with irregular intra-season spacing,
    mimicking the LSST observing strategy for a single field.
    """
    rng = rng or np.random.default_rng()
    n_seasons = max(1, int(baseline_days / 365.25))
    epochs_per_season = n_epochs // n_seasons
    remainder = n_epochs - epochs_per_season * n_seasons

    times = []
    for s in range(n_seasons):
        season_start = s * 365.25 + (1 - season_fraction) * 365.25 * rng.uniform(0, 0.5)
        season_length = season_fraction * 365.25
        n_this = epochs_per_season + (1 if s < remainder else 0)
        raw = np.sort(rng.uniform(season_start, season_start + season_length, size=n_this))
        if len(raw) > 1:
            gaps = np.diff(raw)
            gaps = np.maximum(gaps, min_gap_days)
            raw = np.concatenate([[raw[0]], raw[0] + np.cumsum(gaps)])
        times.append(raw)

    times = np.concatenate(times)
    times -= times[0]
    return times[:n_epochs]


def uniform_cadence(
    n_epochs: int = 200,
    baseline_days: float = 3650.0,
    jitter_frac: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Uniformly spaced cadence with optional jitter."""
    rng = rng or np.random.default_rng()
    spacing = baseline_days / n_epochs
    times = np.arange(n_epochs) * spacing
    times += rng.uniform(-jitter_frac * spacing, jitter_frac * spacing, size=n_epochs)
    times -= times[0]
    return np.sort(times)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def _apply_noise(
    signal: np.ndarray,
    snr: float,
    base_error: float = 100.0,
    heteroscedastic: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to a signal at a target peak SNR.

    Returns (noisy_flux, flux_errors).  The signal amplitude is scaled so that
    peak_signal / typical_error = snr.
    """
    rng = rng or np.random.default_rng()
    n = len(signal)

    if heteroscedastic:
        flux_errors = base_error * (1 + 0.3 * rng.exponential(size=n))
    else:
        flux_errors = np.full(n, base_error)

    median_err = np.median(flux_errors)
    peak_signal = np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else 1.0
    scale = (snr * median_err) / peak_signal if peak_signal > 0 else 0.0

    scaled_signal = signal * scale
    noise = rng.normal(0, flux_errors)
    return scaled_signal + noise, flux_errors


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------

def generate_periodic(
    snr: float = 3.0,
    period: float = 5.0,
    n_epochs: int = 200,
    baseline_days: float = 3650.0,
    waveform: str = "sinusoidal",
    band: str = "g",
    phase: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> LightCurve:
    """Generate a periodic variable light curve.

    Parameters
    ----------
    waveform : str
        One of 'sinusoidal', 'sawtooth', 'eclipsing'.
    """
    rng = rng or np.random.default_rng()
    times = lsst_cadence(n_epochs, baseline_days, rng=rng)
    phase = phase if phase is not None else rng.uniform(0, 2 * np.pi)
    phi = 2 * np.pi * times / period + phase

    if waveform == "sinusoidal":
        signal = np.sin(phi)
    elif waveform == "sawtooth":
        signal = 2 * (phi / (2 * np.pi) % 1) - 1
    elif waveform == "eclipsing":
        signal = np.ones_like(phi)
        eclipse_phase = phi % (2 * np.pi)
        in_eclipse = eclipse_phase < 0.3
        signal[in_eclipse] = 1 - 0.8 * np.cos(eclipse_phase[in_eclipse] / 0.3 * np.pi)
        signal -= np.mean(signal)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")

    fluxes, errors = _apply_noise(signal, snr, rng=rng)

    return LightCurve(
        times=times,
        fluxes=fluxes,
        flux_errors=errors,
        band=band,
        metadata={
            "source_type": "periodic",
            "waveform": waveform,
            "injected_snr": snr,
            "period": period,
            "phase": phase,
        },
    )


def generate_transient(
    snr: float = 3.0,
    t_rise: float = 20.0,
    t_fall: float = 50.0,
    t_peak: Optional[float] = None,
    n_epochs: int = 200,
    baseline_days: float = 3650.0,
    band: str = "g",
    rng: Optional[np.random.Generator] = None,
) -> LightCurve:
    """Generate a transient (supernova-like) light curve using the Bazin function.

    f(t) = A * exp(-(t-t0)/t_fall) / (1 + exp(-(t-t0)/t_rise))
    """
    rng = rng or np.random.default_rng()
    times = lsst_cadence(n_epochs, baseline_days, rng=rng)
    t_peak = t_peak if t_peak is not None else rng.uniform(
        times[0] + 0.2 * baseline_days, times[-1] - 0.2 * baseline_days
    )

    dt = times - t_peak
    signal = np.exp(-dt / t_fall) / (1 + np.exp(-dt / t_rise))
    signal = np.maximum(signal, 0)

    fluxes, errors = _apply_noise(signal, snr, rng=rng)

    return LightCurve(
        times=times,
        fluxes=fluxes,
        flux_errors=errors,
        band=band,
        metadata={
            "source_type": "transient",
            "injected_snr": snr,
            "t_rise": t_rise,
            "t_fall": t_fall,
            "t_peak": t_peak,
        },
    )


def generate_slow_evolving(
    snr: float = 3.0,
    profile: str = "linear",
    n_epochs: int = 200,
    baseline_days: float = 3650.0,
    band: str = "g",
    rng: Optional[np.random.Generator] = None,
) -> LightCurve:
    """Generate a slowly evolving source (changing-look AGN, TDE).

    Parameters
    ----------
    profile : str
        'linear' for monotonic trend, 'sigmoid' for S-curve transition.
    """
    rng = rng or np.random.default_rng()
    times = lsst_cadence(n_epochs, baseline_days, rng=rng)

    t_norm = (times - times[0]) / (times[-1] - times[0])

    if profile == "linear":
        signal = t_norm - 0.5
    elif profile == "sigmoid":
        midpoint = rng.uniform(0.3, 0.7)
        steepness = rng.uniform(8, 20)
        signal = 1 / (1 + np.exp(-steepness * (t_norm - midpoint)))
        signal -= np.mean(signal)
    else:
        raise ValueError(f"Unknown profile: {profile}")

    fluxes, errors = _apply_noise(signal, snr, rng=rng)

    return LightCurve(
        times=times,
        fluxes=fluxes,
        flux_errors=errors,
        band=band,
        metadata={
            "source_type": "slow_evolving",
            "profile": profile,
            "injected_snr": snr,
        },
    )


def generate_stochastic(
    snr: float = 3.0,
    tau: float = 200.0,
    n_epochs: int = 200,
    baseline_days: float = 3650.0,
    band: str = "g",
    rng: Optional[np.random.Generator] = None,
) -> LightCurve:
    """Generate a stochastic variable (AGN-like) via damped random walk.

    The DRW is the standard model for quasar optical variability.
    """
    rng = rng or np.random.default_rng()
    times = lsst_cadence(n_epochs, baseline_days, rng=rng)

    signal = np.zeros(len(times))
    sf_inf = 1.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        decay = np.exp(-dt / tau)
        drive = sf_inf * np.sqrt(1 - decay**2) * rng.normal()
        signal[i] = signal[i - 1] * decay + drive

    fluxes, errors = _apply_noise(signal, snr, rng=rng)

    return LightCurve(
        times=times,
        fluxes=fluxes,
        flux_errors=errors,
        band=band,
        metadata={
            "source_type": "stochastic",
            "injected_snr": snr,
            "tau_days": tau,
        },
    )


def generate_microlensing(
    snr: float = 3.0,
    t_einstein: float = 30.0,
    u_min: float = 0.5,
    t_peak: Optional[float] = None,
    n_epochs: int = 200,
    baseline_days: float = 3650.0,
    band: str = "g",
    rng: Optional[np.random.Generator] = None,
) -> LightCurve:
    """Generate a gravitational microlensing light curve (Paczyński curve).

    Models point-source point-lens (PSPL) microlensing, the expected
    signal from dark matter substructure lensing background sources.

    Parameters
    ----------
    t_einstein : float
        Einstein crossing time in days.
    u_min : float
        Minimum impact parameter in Einstein radii.
        Smaller values → stronger magnification.
    t_peak : float, optional
        Time of closest approach (days). Random if not given.
    """
    rng = rng or np.random.default_rng()
    times = lsst_cadence(n_epochs, baseline_days, rng=rng)
    t_peak = t_peak if t_peak is not None else rng.uniform(
        times[0] + 0.2 * baseline_days, times[-1] - 0.2 * baseline_days
    )

    # Paczyński magnification: A(u) = (u² + 2) / (u * sqrt(u² + 4))
    u_t = np.sqrt(u_min**2 + ((times - t_peak) / t_einstein) ** 2)
    magnification = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))

    # Signal is excess flux: A(t) - 1 (baseline-subtracted)
    signal = magnification - 1.0

    fluxes, errors = _apply_noise(signal, snr, rng=rng)

    return LightCurve(
        times=times,
        fluxes=fluxes,
        flux_errors=errors,
        band=band,
        metadata={
            "source_type": "microlensing",
            "injected_snr": snr,
            "t_einstein": t_einstein,
            "u_min": u_min,
            "t_peak": t_peak,
        },
    )


def generate_noise(
    n_epochs: int = 200,
    baseline_days: float = 3650.0,
    base_error: float = 100.0,
    band: str = "g",
    rng: Optional[np.random.Generator] = None,
) -> LightCurve:
    """Generate pure noise — no astrophysical signal."""
    rng = rng or np.random.default_rng()
    times = lsst_cadence(n_epochs, baseline_days, rng=rng)
    flux_errors = base_error * (1 + 0.3 * rng.exponential(size=n_epochs))
    fluxes = rng.normal(0, flux_errors)

    return LightCurve(
        times=times,
        fluxes=fluxes,
        flux_errors=flux_errors,
        band=band,
        metadata={"source_type": "noise", "injected_snr": 0.0},
    )


# ---------------------------------------------------------------------------
# Multi-band and ensemble generators
# ---------------------------------------------------------------------------

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

# Rough relative depth offsets: deeper bands have smaller errors.
_BAND_ERROR_SCALE = {"u": 1.6, "g": 1.0, "r": 0.9, "i": 0.95, "z": 1.1, "y": 1.4}


def generate_multiband(
    generator,
    bands: list[str] | None = None,
    **kwargs,
) -> MultiBandLightCurve:
    """Generate a multi-band light curve using any single-band generator.

    Each band gets its own cadence realization and band-appropriate noise scaling.
    """
    bands = bands or LSST_BANDS
    rng = kwargs.pop("rng", None) or np.random.default_rng()
    curves = {}
    for b in bands:
        band_rng = np.random.default_rng(rng.integers(0, 2**63))
        lc = generator(band=b, rng=band_rng, **kwargs)
        scale = _BAND_ERROR_SCALE.get(b, 1.0)
        lc.flux_errors *= scale
        curves[b] = lc

    return MultiBandLightCurve(
        curves=curves,
        metadata={"generator": generator.__name__, **kwargs},
    )


def generate_ensemble(
    n_signal: int = 50,
    n_noise: int = 200,
    signal_generator=generate_periodic,
    signal_kwargs: dict | None = None,
    noise_kwargs: dict | None = None,
    region_id: str | None = None,
    rng: Optional[np.random.Generator] = None,
) -> Ensemble:
    """Generate a mixed ensemble of signal + noise light curves.

    This simulates a sky region containing a sub-threshold population
    embedded in a background of pure-noise forced photometry.
    """
    rng = rng or np.random.default_rng()
    signal_kwargs = signal_kwargs or {}
    noise_kwargs = noise_kwargs or {}

    ensemble = Ensemble(region_id=region_id, metadata={
        "n_signal": n_signal,
        "n_noise": n_noise,
        "signal_type": signal_generator.__name__,
    })

    for _ in range(n_signal):
        child_rng = np.random.default_rng(rng.integers(0, 2**63))
        ensemble.add(signal_generator(rng=child_rng, **signal_kwargs))

    for _ in range(n_noise):
        child_rng = np.random.default_rng(rng.integers(0, 2**63))
        ensemble.add(generate_noise(rng=child_rng, **noise_kwargs))

    return ensemble
