"""Rubin Observatory data access via TAP/ADQL.

Provides functions to query the Rubin Science Platform for DiaObject
and DiaSource/DiaForcedSource data from Data Preview releases (DP1, DP2)
and convert them into our LightCurve data model.

Requires: pip install pyvo

TAP endpoints:
  - RSP (authenticated): https://data.lsst.cloud/api/tap
  - DP1 public:          https://data-int.lsst.cloud/api/tap  (may change)

Column references from LSE-163 (Data Products Definition Document):
  - DiaObject: diaObjectId, ra, dec, nDiaSources, per-band flux stats
  - DiaSource: diaSourceId, diaObjectId, midpointMjdTai, band,
               psfFlux, psfFluxErr
  - DiaForcedSource: diaForcedSourceId, diaObjectId, midpointMjdTai, band,
                     psfFlux, psfFluxErr
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from void.data.models import Ensemble, LightCurve, MultiBandLightCurve

# Default RSP TAP endpoint (Interim Data Facility — Google Cloud)
RSP_TAP_URL = "https://data.lsst.cloud/api/tap"

CACHE_DIR = Path(".void_cache/rubin")

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def get_tap_service(tap_url: str | None = None):
    """Get a TAP service connection to the Rubin Science Platform.

    Parameters
    ----------
    tap_url : str, optional
        TAP endpoint URL.  Defaults to the RSP IDF endpoint.
        Authentication is handled via the RSP environment token
        (~/.lsst/token) or the RUBIN_ACCESS_TOKEN env var.
    """
    try:
        import pyvo
    except ImportError:
        raise ImportError(
            "The 'pyvo' package is required for Rubin data access. "
            "Install with: pip install pyvo"
        )

    url = tap_url or RSP_TAP_URL

    # RSP authentication: token-based via environment or credential file
    import os
    token = os.environ.get("RUBIN_ACCESS_TOKEN")
    if token is None:
        token_path = Path.home() / ".lsst" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()

    if token:
        cred = pyvo.auth.CredentialStore()
        cred.set_password("bearer_token", token)
        auth = pyvo.auth.AuthSession()
        auth.credentials = cred
        service = pyvo.dal.TAPService(url, session=auth.session)
    else:
        service = pyvo.dal.TAPService(url)

    return service


# ---------------------------------------------------------------------------
# DiaObject queries
# ---------------------------------------------------------------------------

def query_near_threshold_objects(
    max_objects: int = 500,
    min_n_dia_sources: int = 5,
    max_mean_snr: float = 5.0,
    min_mean_snr: float = 0.5,
    bands: list[str] | None = None,
    tap_url: str | None = None,
) -> pd.DataFrame:
    """Query DiaObjects near the detection threshold.

    Selects objects with enough forced-photometry measurements but
    low mean SNR — the sub-threshold population.

    Parameters
    ----------
    max_objects : int
        Maximum rows to return.
    min_n_dia_sources : int
        Minimum number of associated DiaSources.
    max_mean_snr : float
        Upper bound on mean per-epoch SNR (selects faint objects).
    min_mean_snr : float
        Lower bound (excludes pure noise with SNR ~ 0).
    bands : list[str], optional
        Filter on specific bands; default uses all LSST bands.
    tap_url : str, optional
        Override the TAP endpoint.

    Returns
    -------
    pd.DataFrame
        Rows from the DiaObject table with positional and flux info.
    """
    service = get_tap_service(tap_url)
    bands = bands or LSST_BANDS

    # Build band-specific SNR filter using per-band PSF flux mean/sigma columns
    # DP1 schema: {band}PSFluxMean, {band}PSFluxSigma, {band}PSFluxNdata
    band_filters = []
    for b in bands:
        band_filters.append(
            f"({b}PSFluxNdata >= {min_n_dia_sources} "
            f"AND {b}PSFluxMean / GREATEST({b}PSFluxSigma, 1e-30) "
            f"BETWEEN {min_mean_snr} AND {max_mean_snr})"
        )
    band_clause = " OR ".join(band_filters)

    adql = f"""
    SELECT TOP {max_objects}
        diaObjectId, ra, dec, nDiaSources,
        gPSFluxMean, gPSFluxSigma, gPSFluxNdata,
        rPSFluxMean, rPSFluxSigma, rPSFluxNdata,
        iPSFluxMean, iPSFluxSigma, iPSFluxNdata,
        zPSFluxMean, zPSFluxSigma, zPSFluxNdata
    FROM dp1_dc2_catalogs.DiaObject
    WHERE nDiaSources >= {min_n_dia_sources}
      AND ({band_clause})
    ORDER BY nDiaSources DESC
    """

    result = service.search(adql)
    return result.to_table().to_pandas()


def query_dia_objects_in_region(
    ra: float,
    dec: float,
    radius_deg: float = 1.0,
    max_objects: int = 1000,
    tap_url: str | None = None,
) -> pd.DataFrame:
    """Query DiaObjects within a cone search region.

    Parameters
    ----------
    ra, dec : float
        Center of the search cone in degrees (ICRS).
    radius_deg : float
        Cone search radius in degrees.
    max_objects : int
        Maximum rows.
    """
    service = get_tap_service(tap_url)

    adql = f"""
    SELECT TOP {max_objects}
        diaObjectId, ra, dec, nDiaSources,
        gPSFluxMean, gPSFluxSigma, gPSFluxNdata,
        rPSFluxMean, rPSFluxSigma, rPSFluxNdata,
        iPSFluxMean, iPSFluxSigma, iPSFluxNdata
    FROM dp1_dc2_catalogs.DiaObject
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    ) = 1
      AND nDiaSources >= 5
    ORDER BY nDiaSources DESC
    """

    result = service.search(adql)
    return result.to_table().to_pandas()


# ---------------------------------------------------------------------------
# Light curve retrieval
# ---------------------------------------------------------------------------

def get_dia_source_light_curve(
    dia_object_id: int,
    tap_url: str | None = None,
    use_cache: bool = True,
) -> MultiBandLightCurve:
    """Download DiaSource light curve for a single DiaObject.

    DiaSources are individual detections on difference images (above threshold).
    """
    cache_key = f"diasource_{dia_object_id}"
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    service = get_tap_service(tap_url)

    adql = f"""
    SELECT midpointMjdTai, band, psfFlux, psfFluxErr
    FROM dp1_dc2_catalogs.DiaSource
    WHERE diaObjectId = {dia_object_id}
    ORDER BY midpointMjdTai
    """

    result = service.search(adql)
    df = result.to_table().to_pandas()

    mblc = _build_multiband_from_df(
        df, object_id=str(dia_object_id), source_label="rubin_diasource"
    )

    if use_cache:
        _save_cache(cache_key, mblc)

    return mblc


def get_forced_photometry(
    dia_object_id: int,
    tap_url: str | None = None,
    use_cache: bool = True,
) -> MultiBandLightCurve:
    """Download DiaForcedSource light curve for a single DiaObject.

    DiaForcedSources are forced measurements at the object position on
    *every* difference image, regardless of whether a detection occurred.
    This is the core data product for sub-threshold analysis.

    Note: DiaForcedSource may not be available until mid-2026 (PPDB).
    """
    cache_key = f"diaforced_{dia_object_id}"
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    service = get_tap_service(tap_url)

    adql = f"""
    SELECT midpointMjdTai, band, psfFlux, psfFluxErr
    FROM dp1_dc2_catalogs.DiaForcedSource
    WHERE diaObjectId = {dia_object_id}
    ORDER BY midpointMjdTai
    """

    result = service.search(adql)
    df = result.to_table().to_pandas()

    mblc = _build_multiband_from_df(
        df, object_id=str(dia_object_id), source_label="rubin_forced"
    )

    if use_cache:
        _save_cache(cache_key, mblc)

    return mblc


def get_batch_light_curves(
    dia_object_ids: list[int],
    source: str = "diasource",
    tap_url: str | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> list[MultiBandLightCurve]:
    """Download light curves for multiple DiaObjects.

    Parameters
    ----------
    source : str
        'diasource' for detections, 'forced' for forced photometry.
    """
    getter = get_dia_source_light_curve if source == "diasource" else get_forced_photometry
    results = []
    for i, obj_id in enumerate(dia_object_ids):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Downloading {i + 1}/{len(dia_object_ids)}...")
        try:
            mblc = getter(obj_id, tap_url=tap_url, use_cache=use_cache)
            results.append(mblc)
        except Exception as e:
            if verbose:
                print(f"  Failed for {obj_id}: {e}")
    return results


def build_ensemble_from_rubin(
    dia_object_ids: list[int],
    band: str = "r",
    source: str = "diasource",
    region_id: str | None = None,
    tap_url: str | None = None,
    verbose: bool = True,
) -> Ensemble:
    """Build an Ensemble from Rubin DiaObjects in a single band.

    Parameters
    ----------
    band : str
        LSST band to extract (default 'r' — deepest band).
    source : str
        'diasource' or 'forced'.
    """
    mblcs = get_batch_light_curves(
        dia_object_ids, source=source, tap_url=tap_url, verbose=verbose
    )
    ensemble = Ensemble(
        region_id=region_id,
        metadata={"source": f"rubin_{source}", "band": band, "n_queried": len(dia_object_ids)},
    )
    for mblc in mblcs:
        if band in mblc.curves:
            ensemble.add(mblc[band])
    return ensemble


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _build_multiband_from_df(
    df: pd.DataFrame,
    object_id: str,
    source_label: str,
) -> MultiBandLightCurve:
    """Convert a TAP query result to MultiBandLightCurve.

    Expects columns: midpointMjdTai (or mjd), band, psfFlux, psfFluxErr.
    """
    curves = {}

    if df is None or len(df) == 0:
        return MultiBandLightCurve(curves=curves, object_id=object_id)

    # Normalize column names (TAP may return different cases)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "mjd" in lower or "tai" in lower:
            col_map["time"] = col
        elif lower == "band" or lower == "filter":
            col_map["band"] = col
        elif "fluxerr" in lower.replace("_", "") or col.lower() == "psffluxerr":
            col_map["flux_err"] = col
        elif "flux" in lower and "err" not in lower:
            col_map["flux"] = col

    if not all(k in col_map for k in ["time", "band", "flux", "flux_err"]):
        return MultiBandLightCurve(curves=curves, object_id=object_id)

    for band_val in df[col_map["band"]].unique():
        band_df = df[df[col_map["band"]] == band_val].copy()
        band_df = band_df.dropna(subset=[col_map["flux"], col_map["flux_err"]])

        if len(band_df) < 5:
            continue

        band_name = str(band_val).strip()
        curves[band_name] = LightCurve(
            times=band_df[col_map["time"]].values.astype(np.float64),
            fluxes=band_df[col_map["flux"]].values.astype(np.float64),
            flux_errors=np.abs(band_df[col_map["flux_err"]].values.astype(np.float64)),
            band=band_name,
            metadata={
                "object_id": object_id,
                "source": source_label,
                "n_points": len(band_df),
            },
        )

    return MultiBandLightCurve(curves=curves, object_id=object_id)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    """Get the cache file path for a given key."""
    safe = hashlib.md5(key.encode()).hexdigest()  # noqa: S324 — not security-sensitive
    return CACHE_DIR / f"{safe}.npz"


def _save_cache(key: str, mblc: MultiBandLightCurve) -> None:
    """Save a MultiBandLightCurve to the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    arrays = {}
    meta = {}
    for band_name, lc in mblc.curves.items():
        arrays[f"{band_name}_times"] = lc.times
        arrays[f"{band_name}_fluxes"] = lc.fluxes
        arrays[f"{band_name}_flux_errors"] = lc.flux_errors
        meta[band_name] = lc.metadata
    np.savez(path, **arrays, _meta=json.dumps(meta),
             _object_id=mblc.object_id or "", _bands=list(mblc.curves.keys()))


def _load_cache(key: str) -> MultiBandLightCurve | None:
    """Load a MultiBandLightCurve from cache, or None if not found."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["_meta"]))
        object_id = str(data["_object_id"]) or None
        bands = list(data["_bands"])
        curves = {}
        for band_name in bands:
            curves[str(band_name)] = LightCurve(
                times=data[f"{band_name}_times"],
                fluxes=data[f"{band_name}_fluxes"],
                flux_errors=data[f"{band_name}_flux_errors"],
                band=str(band_name),
                metadata=meta.get(str(band_name), {}),
            )
        return MultiBandLightCurve(curves=curves, object_id=object_id)
    except Exception:
        return None
