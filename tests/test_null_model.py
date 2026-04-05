"""Tests for null model framework."""

import numpy as np
import pytest

from void.topology.null_model import (
    NullDistribution,
    build_null_distribution,
    compare_ensemble_to_null,
)
from void.topology.distances import wasserstein_distance, bottleneck_distance
from void.topology.persistence import PersistenceDiagram, compute_persistence
from void.embedding.takens import TakensEmbedder


class TestNullDistribution:
    @pytest.fixture
    def small_null(self):
        """Build a small null distribution for testing (fast)."""
        return build_null_distribution(
            n_realizations=5,
            n_sources_per=20,
            n_epochs=50,
            embedder=TakensEmbedder(dimension=3, delay=1),
            rng=np.random.default_rng(42),
            verbose=False,
        )

    def test_build_null_shape(self, small_null):
        assert small_null.n_realizations == 5
        assert small_null.summary_stats.shape[0] == 5

    def test_mean_and_std(self, small_null):
        m = small_null.mean()
        s = small_null.std()
        assert m.shape == s.shape
        assert np.all(s >= 0)

    def test_p_value_range(self, small_null):
        observed = small_null.mean()
        p = small_null.p_value(observed)
        assert np.all(p >= 0)
        assert np.all(p <= 1)

    def test_z_score(self, small_null):
        observed = small_null.mean() + 2 * small_null.std()
        z = small_null.z_score(observed)
        # Z-scores should be >= 0 (some stats may have zero std → z=0)
        assert np.all(z >= 0)


class TestDistances:
    def test_wasserstein_same_diagram(self):
        cloud = np.random.default_rng(42).normal(size=(30, 3))
        pd = compute_persistence(cloud, maxdim=1)
        d = wasserstein_distance(pd, pd, dim=1)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_wasserstein_different_diagrams(self):
        cloud1 = np.random.default_rng(42).normal(size=(30, 3))
        cloud2 = np.random.default_rng(43).normal(size=(30, 3)) * 3
        pd1 = compute_persistence(cloud1, maxdim=1)
        pd2 = compute_persistence(cloud2, maxdim=1)
        d = wasserstein_distance(pd1, pd2, dim=1)
        assert d >= 0

    def test_bottleneck_same_diagram(self):
        cloud = np.random.default_rng(42).normal(size=(30, 3))
        pd = compute_persistence(cloud, maxdim=1)
        d = bottleneck_distance(pd, pd, dim=1)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_bottleneck_nonnegative(self):
        cloud1 = np.random.default_rng(42).normal(size=(30, 3))
        cloud2 = np.random.default_rng(99).normal(size=(30, 3))
        pd1 = compute_persistence(cloud1, maxdim=1)
        pd2 = compute_persistence(cloud2, maxdim=1)
        d = bottleneck_distance(pd1, pd2, dim=1)
        assert d >= 0


class TestEnsembleTesting:
    def test_signal_ensemble_against_null(self, small_ensemble):
        null_dist = build_null_distribution(
            n_realizations=5,
            n_sources_per=30,
            n_epochs=100,
            embedder=TakensEmbedder(dimension=3, delay=1),
            rng=np.random.default_rng(42),
            verbose=False,
        )
        result = compare_ensemble_to_null(
            small_ensemble, null_dist,
            embedder=TakensEmbedder(dimension=3, delay=1),
        )
        assert "p_values" in result
        assert "z_scores" in result
        assert "significant" in result
        assert len(result["p_values"]) == len(result["stat_names"])
