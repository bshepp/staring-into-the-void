"""Tests for synthetic light curve generation."""

import numpy as np
import pytest

from void.data.models import LightCurve, MultiBandLightCurve, Ensemble
from void.data.synthetic import (
    generate_periodic,
    generate_transient,
    generate_slow_evolving,
    generate_stochastic,
    generate_microlensing,
    generate_noise,
    generate_multiband,
    generate_ensemble,
    lsst_cadence,
    uniform_cadence,
)


class TestCadence:
    def test_lsst_cadence_length(self):
        times = lsst_cadence(n_epochs=200, rng=np.random.default_rng(42))
        assert len(times) == 200

    def test_lsst_cadence_sorted(self):
        times = lsst_cadence(n_epochs=100, rng=np.random.default_rng(42))
        assert np.all(np.diff(times) >= 0)

    def test_lsst_cadence_starts_at_zero(self):
        times = lsst_cadence(n_epochs=50, rng=np.random.default_rng(42))
        assert times[0] == pytest.approx(0.0)

    def test_uniform_cadence_length(self):
        times = uniform_cadence(n_epochs=150, rng=np.random.default_rng(42))
        assert len(times) == 150

    def test_uniform_cadence_sorted(self):
        times = uniform_cadence(n_epochs=100, rng=np.random.default_rng(42))
        assert np.all(np.diff(times) >= 0)


class TestGenerators:
    def test_periodic_returns_lightcurve(self):
        lc = generate_periodic(snr=3.0, rng=np.random.default_rng(42))
        assert isinstance(lc, LightCurve)
        assert lc.n_epochs == 200
        assert lc.metadata["source_type"] == "periodic"

    def test_periodic_snr_scaling(self):
        lc_low = generate_periodic(snr=1.0, rng=np.random.default_rng(42))
        lc_high = generate_periodic(snr=10.0, rng=np.random.default_rng(42))
        assert lc_high.peak_snr > lc_low.peak_snr

    def test_periodic_waveforms(self):
        for waveform in ["sinusoidal", "sawtooth", "eclipsing"]:
            lc = generate_periodic(waveform=waveform, rng=np.random.default_rng(42))
            assert lc.metadata["waveform"] == waveform

    def test_periodic_invalid_waveform(self):
        with pytest.raises(ValueError, match="Unknown waveform"):
            generate_periodic(waveform="invalid", rng=np.random.default_rng(42))

    def test_transient_returns_lightcurve(self):
        lc = generate_transient(snr=3.0, rng=np.random.default_rng(42))
        assert isinstance(lc, LightCurve)
        assert lc.metadata["source_type"] == "transient"

    def test_slow_evolving_profiles(self):
        for profile in ["linear", "sigmoid"]:
            lc = generate_slow_evolving(profile=profile, rng=np.random.default_rng(42))
            assert lc.metadata["profile"] == profile

    def test_stochastic_returns_lightcurve(self):
        lc = generate_stochastic(snr=3.0, rng=np.random.default_rng(42))
        assert isinstance(lc, LightCurve)
        assert lc.metadata["source_type"] == "stochastic"

    def test_noise_returns_lightcurve(self):
        lc = generate_noise(rng=np.random.default_rng(42))
        assert isinstance(lc, LightCurve)
        assert lc.metadata["source_type"] == "noise"
        assert lc.metadata["injected_snr"] == 0.0

    def test_noise_is_zero_mean(self):
        lc = generate_noise(n_epochs=10000, rng=np.random.default_rng(42))
        assert abs(np.mean(lc.fluxes)) < 5 * np.std(lc.fluxes) / np.sqrt(lc.n_epochs)

    def test_microlensing_returns_lightcurve(self):
        lc = generate_microlensing(snr=3.0, rng=np.random.default_rng(42))
        assert isinstance(lc, LightCurve)
        assert lc.metadata["source_type"] == "microlensing"
        assert lc.n_epochs == 200

    def test_microlensing_snr_scaling(self):
        lc_low = generate_microlensing(snr=1.0, rng=np.random.default_rng(42))
        lc_high = generate_microlensing(snr=10.0, rng=np.random.default_rng(42))
        assert lc_high.peak_snr > lc_low.peak_snr

    def test_microlensing_metadata(self):
        lc = generate_microlensing(
            snr=5.0, t_einstein=40.0, u_min=0.3,
            rng=np.random.default_rng(42),
        )
        assert lc.metadata["t_einstein"] == pytest.approx(40.0)
        assert lc.metadata["u_min"] == pytest.approx(0.3)

    def test_microlensing_positive_magnification(self):
        """Microlensing should produce a positive bump (magnification > 1)."""
        lc = generate_microlensing(snr=10.0, u_min=0.1, rng=np.random.default_rng(42))
        # Signal component should be non-negative (magnification - 1 >= 0)
        assert np.max(lc.fluxes) > 0

    def test_microlensing_custom_params(self):
        lc = generate_microlensing(
            n_epochs=100, baseline_days=1000.0, band="r",
            rng=np.random.default_rng(42),
        )
        assert lc.n_epochs == 100
        assert lc.band == "r"

    def test_custom_epochs_and_baseline(self):
        lc = generate_periodic(n_epochs=50, baseline_days=365.0,
                               rng=np.random.default_rng(42))
        assert lc.n_epochs == 50


class TestMultiband:
    def test_generate_multiband(self):
        mblc = generate_multiband(
            generate_periodic,
            bands=["g", "r"],
            snr=3.0,
            rng=np.random.default_rng(42),
        )
        assert isinstance(mblc, MultiBandLightCurve)
        assert set(mblc.bands) == {"g", "r"}

    def test_multiband_all_lsst_bands(self):
        mblc = generate_multiband(
            generate_periodic,
            snr=3.0,
            rng=np.random.default_rng(42),
        )
        assert len(mblc.bands) == 6


class TestEnsemble:
    def test_generate_ensemble(self):
        ens = generate_ensemble(
            n_signal=10, n_noise=20,
            rng=np.random.default_rng(42),
        )
        assert isinstance(ens, Ensemble)
        assert ens.n_sources == 30

    def test_ensemble_metadata(self):
        ens = generate_ensemble(
            n_signal=5, n_noise=15,
            rng=np.random.default_rng(42),
        )
        assert ens.metadata["n_signal"] == 5
        assert ens.metadata["n_noise"] == 15


class TestLightCurve:
    def test_attenuate(self):
        lc = generate_periodic(snr=10.0, rng=np.random.default_rng(42))
        att = lc.attenuate(0.5)
        assert att.metadata["attenuation_factor"] == 0.5

    def test_attenuate_invalid(self):
        lc = generate_periodic(rng=np.random.default_rng(42))
        with pytest.raises(ValueError):
            lc.attenuate(1.5)

    def test_sort_by_time(self):
        lc = generate_periodic(rng=np.random.default_rng(42))
        sorted_lc = lc.sort_by_time()
        assert np.all(np.diff(sorted_lc.times) >= 0)

    def test_to_dataframe(self):
        lc = generate_periodic(rng=np.random.default_rng(42))
        df = lc.to_dataframe()
        assert len(df) == lc.n_epochs
        assert "time" in df.columns
        assert "flux" in df.columns

    def test_array_length_mismatch(self):
        with pytest.raises(ValueError, match="Array length mismatch"):
            LightCurve(
                times=np.array([1, 2, 3]),
                fluxes=np.array([1, 2]),
                flux_errors=np.array([0.1, 0.1, 0.1]),
            )

    def test_snr_property(self):
        lc = generate_periodic(snr=5.0, rng=np.random.default_rng(42))
        assert lc.snr.shape == (lc.n_epochs,)
        assert lc.peak_snr > 0
        assert lc.mean_snr > 0

    def test_duration(self):
        lc = generate_periodic(baseline_days=1000.0, rng=np.random.default_rng(42))
        assert lc.duration > 0
