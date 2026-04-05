"""Tests for attenuation experiment framework."""

import numpy as np
import pytest

from void.analysis.attenuation import attenuate_light_curve
from void.data.synthetic import generate_periodic


class TestAttenuation:
    def test_attenuate_preserves_length(self):
        lc = generate_periodic(snr=10.0, n_epochs=100, rng=np.random.default_rng(42))
        att = attenuate_light_curve(lc, 0.5)
        assert att.n_epochs == lc.n_epochs

    def test_attenuate_preserves_times(self):
        lc = generate_periodic(snr=10.0, n_epochs=100, rng=np.random.default_rng(42))
        att = attenuate_light_curve(lc, 0.5)
        np.testing.assert_array_equal(att.times, lc.times)

    def test_attenuate_preserves_errors(self):
        lc = generate_periodic(snr=10.0, n_epochs=100, rng=np.random.default_rng(42))
        att = attenuate_light_curve(lc, 0.5)
        np.testing.assert_array_equal(att.flux_errors, lc.flux_errors)

    def test_attenuate_reduces_signal(self):
        rng = np.random.default_rng(42)
        lc = generate_periodic(snr=10.0, n_epochs=200, rng=rng)
        att = attenuate_light_curve(lc, 0.1)
        assert np.std(att.fluxes) < np.std(lc.fluxes)

    def test_attenuate_factor_one_similar(self):
        lc = generate_periodic(snr=10.0, n_epochs=100, rng=np.random.default_rng(42))
        att = attenuate_light_curve(lc, 1.0)
        assert att.n_epochs == lc.n_epochs

    def test_attenuate_factor_zero_is_noise(self):
        lc = generate_periodic(snr=10.0, n_epochs=1000, rng=np.random.default_rng(42))
        att = attenuate_light_curve(lc, 0.0)
        assert abs(np.mean(att.fluxes)) < 3 * np.std(att.fluxes) / np.sqrt(att.n_epochs)
