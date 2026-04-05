"""Tests for Rubin data access — conversion helpers and caching.

Network-dependent tests (TAP queries) are skipped unless pyvo
is installed and RSP credentials are available.
"""

import json

import numpy as np
import pandas as pd
import pytest

from void.data.models import LightCurve, MultiBandLightCurve, Ensemble


class TestBuildMultibandFromDf:
    """Test the _build_multiband_from_df conversion helper."""

    def _get_builder(self):
        from void.data.rubin import _build_multiband_from_df
        return _build_multiband_from_df

    def test_basic_conversion(self):
        build = self._get_builder()
        df = pd.DataFrame({
            "midpointMjdTai": np.arange(60000.0, 60020.0),
            "band": ["r"] * 20,
            "psfFlux": np.random.default_rng(42).normal(0, 100, 20),
            "psfFluxErr": np.full(20, 10.0),
        })
        mblc = build(df, object_id="12345", source_label="test")
        assert isinstance(mblc, MultiBandLightCurve)
        assert "r" in mblc.bands
        assert mblc["r"].n_epochs == 20
        assert mblc["r"].metadata["object_id"] == "12345"

    def test_multiband_conversion(self):
        build = self._get_builder()
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "midpointMjdTai": np.tile(np.arange(60000.0, 60015.0), 2),
            "band": ["g"] * 15 + ["r"] * 15,
            "psfFlux": rng.normal(0, 100, 30),
            "psfFluxErr": np.full(30, 10.0),
        })
        mblc = build(df, object_id="99", source_label="test")
        assert set(mblc.bands) == {"g", "r"}
        assert mblc["g"].n_epochs == 15

    def test_empty_dataframe(self):
        build = self._get_builder()
        df = pd.DataFrame(columns=["midpointMjdTai", "band", "psfFlux", "psfFluxErr"])
        mblc = build(df, object_id="empty", source_label="test")
        assert len(mblc.bands) == 0

    def test_none_dataframe(self):
        build = self._get_builder()
        mblc = build(None, object_id="none", source_label="test")
        assert len(mblc.bands) == 0

    def test_skips_bands_with_few_points(self):
        build = self._get_builder()
        df = pd.DataFrame({
            "midpointMjdTai": np.arange(60000.0, 60004.0),
            "band": ["r"] * 4,
            "psfFlux": [100.0, 200.0, 300.0, 400.0],
            "psfFluxErr": [10.0, 10.0, 10.0, 10.0],
        })
        mblc = build(df, object_id="few", source_label="test")
        # 4 points < 5 threshold
        assert "r" not in mblc.bands

    def test_flux_errors_absolute(self):
        build = self._get_builder()
        df = pd.DataFrame({
            "midpointMjdTai": np.arange(60000.0, 60010.0),
            "band": ["r"] * 10,
            "psfFlux": np.zeros(10),
            "psfFluxErr": np.full(10, -5.0),  # negative errors
        })
        mblc = build(df, object_id="neg", source_label="test")
        assert np.all(mblc["r"].flux_errors > 0)


class TestCaching:
    """Test cache save/load round-trip."""

    def test_cache_round_trip(self, tmp_path, monkeypatch):
        import void.data.rubin as rubin_mod
        monkeypatch.setattr(rubin_mod, "CACHE_DIR", tmp_path)

        lc = LightCurve(
            times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            fluxes=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            flux_errors=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            band="r",
            metadata={"object_id": "test", "source": "test"},
        )
        mblc = MultiBandLightCurve(
            curves={"r": lc}, object_id="test_obj"
        )

        rubin_mod._save_cache("test_key", mblc)
        loaded = rubin_mod._load_cache("test_key")

        assert loaded is not None
        assert "r" in loaded.bands
        assert np.allclose(loaded["r"].times, lc.times)
        assert np.allclose(loaded["r"].fluxes, lc.fluxes)

    def test_cache_miss(self, tmp_path, monkeypatch):
        import void.data.rubin as rubin_mod
        monkeypatch.setattr(rubin_mod, "CACHE_DIR", tmp_path)

        loaded = rubin_mod._load_cache("nonexistent_key")
        assert loaded is None
