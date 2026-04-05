"""Shared test fixtures for the void pipeline."""

import numpy as np
import pytest

from void.data.models import Ensemble, LightCurve, MultiBandLightCurve
from void.data.synthetic import (
    generate_noise,
    generate_periodic,
    generate_stochastic,
    generate_transient,
)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def periodic_lc(rng):
    return generate_periodic(snr=5.0, period=10.0, n_epochs=200, rng=rng)


@pytest.fixture
def transient_lc(rng):
    return generate_transient(snr=5.0, n_epochs=200, rng=rng)


@pytest.fixture
def stochastic_lc(rng):
    return generate_stochastic(snr=5.0, tau=100.0, n_epochs=200, rng=rng)


@pytest.fixture
def noise_lc(rng):
    return generate_noise(n_epochs=200, rng=rng)


@pytest.fixture
def multiband_lc(rng):
    curves = {}
    for band in ["g", "r", "i"]:
        curves[band] = generate_periodic(
            snr=5.0, period=10.0, n_epochs=100, band=band,
            rng=np.random.default_rng(rng.integers(0, 2**63)),
        )
    return MultiBandLightCurve(curves=curves, object_id="test_mb")


@pytest.fixture
def small_ensemble(rng):
    ensemble = Ensemble(region_id="test_region")
    for _ in range(10):
        child_rng = np.random.default_rng(rng.integers(0, 2**63))
        ensemble.add(generate_periodic(snr=4.0, period=10.0, n_epochs=100, rng=child_rng))
    for _ in range(20):
        child_rng = np.random.default_rng(rng.integers(0, 2**63))
        ensemble.add(generate_noise(n_epochs=100, rng=child_rng))
    return ensemble
