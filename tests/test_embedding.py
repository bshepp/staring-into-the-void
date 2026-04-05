"""Tests for Takens delay embedding."""

import numpy as np
import pytest

from void.embedding.takens import TakensEmbedder, optimal_delay, optimal_dimension
from void.data.synthetic import generate_periodic, generate_noise


class TestTakensEmbedder:
    def test_basic_embedding_shape(self, periodic_lc):
        embedder = TakensEmbedder(dimension=3, delay=1)
        cloud = embedder.embed(periodic_lc)
        assert cloud.ndim == 2
        assert cloud.shape[1] == 3

    def test_embedding_dimension_2(self, periodic_lc):
        embedder = TakensEmbedder(dimension=2, delay=1)
        cloud = embedder.embed(periodic_lc)
        assert cloud.shape[1] == 2

    def test_embedding_dimension_5(self, periodic_lc):
        embedder = TakensEmbedder(dimension=5, delay=1)
        cloud = embedder.embed(periodic_lc)
        assert cloud.shape[1] == 5

    def test_delay_effect(self, periodic_lc):
        cloud_d1 = TakensEmbedder(dimension=3, delay=1).embed(periodic_lc)
        cloud_d5 = TakensEmbedder(dimension=3, delay=5).embed(periodic_lc)
        assert not np.allclose(cloud_d1[:10], cloud_d5[:10])

    def test_stride(self, periodic_lc):
        cloud_s1 = TakensEmbedder(dimension=3, delay=1, stride=1).embed(periodic_lc)
        cloud_s3 = TakensEmbedder(dimension=3, delay=1, stride=3).embed(periodic_lc)
        assert cloud_s3.shape[0] < cloud_s1.shape[0]

    def test_interpolation_linear(self, periodic_lc):
        embedder = TakensEmbedder(dimension=3, delay=1, interpolation="linear")
        cloud = embedder.embed(periodic_lc)
        assert cloud.shape[0] > 0

    def test_interpolation_cubic(self, periodic_lc):
        embedder = TakensEmbedder(dimension=3, delay=1, interpolation="cubic")
        cloud = embedder.embed(periodic_lc)
        assert cloud.shape[0] > 0

    def test_interpolation_none(self, periodic_lc):
        embedder = TakensEmbedder(dimension=3, delay=1, interpolation="none")
        cloud = embedder.embed(periodic_lc)
        assert cloud.shape[0] > 0

    def test_n_resample(self, periodic_lc):
        embedder = TakensEmbedder(dimension=3, delay=1, n_resample=100)
        cloud = embedder.embed(periodic_lc)
        assert cloud.shape[0] > 0

    def test_too_short_series_raises(self):
        from void.data.models import LightCurve
        lc = LightCurve(
            times=np.array([0, 1, 2]),
            fluxes=np.array([1, 2, 3]),
            flux_errors=np.array([0.1, 0.1, 0.1]),
        )
        embedder = TakensEmbedder(dimension=10, delay=5)
        with pytest.raises(ValueError, match="too short"):
            embedder.embed(lc)


class TestMultibandEmbedding:
    def test_concatenate_mode(self, multiband_lc):
        embedder = TakensEmbedder(dimension=3, delay=1)
        cloud = embedder.embed_multiband(multiband_lc, mode="concatenate")
        assert cloud.shape[1] == 3 * len(multiband_lc.bands)

    def test_interleave_mode(self, multiband_lc):
        embedder = TakensEmbedder(dimension=3, delay=1)
        cloud = embedder.embed_multiband(multiband_lc, mode="interleave")
        assert cloud.shape[1] == 3

    def test_invalid_mode(self, multiband_lc):
        embedder = TakensEmbedder(dimension=3, delay=1)
        with pytest.raises(ValueError, match="Unknown mode"):
            embedder.embed_multiband(multiband_lc, mode="invalid")


class TestOptimalParameters:
    def test_optimal_delay_returns_positive(self, periodic_lc):
        tau = optimal_delay(periodic_lc, max_lag=20)
        assert tau >= 1

    def test_optimal_delay_autocorrelation(self, periodic_lc):
        tau = optimal_delay(periodic_lc, max_lag=20, method="autocorrelation")
        assert tau >= 1

    def test_optimal_dimension_returns_reasonable(self, periodic_lc):
        d = optimal_dimension(periodic_lc, delay=1, max_dim=6)
        assert 1 <= d <= 6
