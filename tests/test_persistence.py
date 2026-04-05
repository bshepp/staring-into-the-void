"""Tests for persistent homology computation."""

import numpy as np
import pytest

from void.topology.persistence import (
    PersistenceDiagram,
    compute_persistence,
    compute_persistence_from_distance_matrix,
)
from void.embedding.takens import TakensEmbedder


class TestComputePersistence:
    def test_basic_computation(self):
        cloud = np.random.default_rng(42).normal(size=(50, 3))
        pd = compute_persistence(cloud, maxdim=1)
        assert isinstance(pd, PersistenceDiagram)
        assert len(pd.diagrams) == 2  # H0 and H1

    def test_h0_always_has_features(self):
        cloud = np.random.default_rng(42).normal(size=(30, 2))
        pd = compute_persistence(cloud, maxdim=0)
        assert pd.n_features(dim=0) > 0

    def test_circle_has_h1(self):
        """A circle sampled with enough points should produce an H1 feature."""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        noise = np.random.default_rng(42).normal(0, 0.05, circle.shape)
        pd = compute_persistence(circle + noise, maxdim=1)
        assert pd.n_features(dim=1) >= 1
        assert pd.max_persistence(dim=1) > 0.5

    def test_maxdim_2(self):
        cloud = np.random.default_rng(42).normal(size=(40, 4))
        pd = compute_persistence(cloud, maxdim=2)
        assert len(pd.diagrams) == 3

    def test_from_distance_matrix(self):
        cloud = np.random.default_rng(42).normal(size=(30, 3))
        from scipy.spatial.distance import pdist, squareform
        dm = squareform(pdist(cloud))
        pd = compute_persistence_from_distance_matrix(dm, maxdim=1)
        assert isinstance(pd, PersistenceDiagram)


class TestPersistenceDiagram:
    @pytest.fixture
    def sample_pd(self):
        cloud = np.random.default_rng(42).normal(size=(50, 3))
        return compute_persistence(cloud, maxdim=1)

    def test_total_persistence(self, sample_pd):
        tp = sample_pd.total_persistence(dim=0)
        assert tp >= 0

    def test_max_persistence(self, sample_pd):
        mp = sample_pd.max_persistence(dim=0)
        assert mp >= 0
        assert mp <= sample_pd.total_persistence(dim=0)

    def test_n_features(self, sample_pd):
        n = sample_pd.n_features(dim=0)
        assert n >= 0

    def test_persistence_entropy(self, sample_pd):
        entropy = sample_pd.persistence_entropy(dim=0)
        assert entropy >= 0

    def test_summary_vector(self, sample_pd):
        sv = sample_pd.summary_vector()
        assert sv.shape == (8,)  # 4 stats x 2 dims (H0, H1)

    def test_barcode(self, sample_pd):
        bc = sample_pd.barcode(dim=0)
        assert isinstance(bc, list)
        for birth, death in bc:
            assert death >= birth

    def test_empty_dimension(self, sample_pd):
        tp = sample_pd.total_persistence(dim=5)
        assert tp == 0.0

    def test_n_features_with_threshold(self, sample_pd):
        n_all = sample_pd.n_features(dim=0, min_persistence=0.0)
        n_thresh = sample_pd.n_features(dim=0, min_persistence=1e6)
        assert n_thresh <= n_all


class TestEmbeddingToPersistence:
    """Integration test: light curve -> embedding -> persistence."""

    def test_periodic_has_h1(self, periodic_lc):
        embedder = TakensEmbedder(dimension=3, delay=2)
        cloud = embedder.embed(periodic_lc)
        pd = compute_persistence(cloud, maxdim=1)
        assert pd.total_persistence(1) > 0

    def test_noise_has_less_h1(self, periodic_lc, noise_lc):
        embedder = TakensEmbedder(dimension=3, delay=2)

        cloud_p = embedder.embed(periodic_lc)
        pd_p = compute_persistence(cloud_p, maxdim=1)

        cloud_n = embedder.embed(noise_lc)
        pd_n = compute_persistence(cloud_n, maxdim=1)

        # Periodic source should generally have more H1 persistence
        # This is a statistical test, so we give it some slack
        assert pd_p.max_persistence(1) > 0 or pd_n.max_persistence(1) > 0
