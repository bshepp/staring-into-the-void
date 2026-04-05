"""ZTF data access via the ALeRCE broker API.

Provides functions to query, download, and cache ZTF light curves
(detections, non-detections, and forced photometry) and convert them
into our LightCurve data model.

Requires: pip install alerce
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from void.data.models import Ensemble, LightCurve, MultiBandLightCurve

CACHE_DIR = Path(".void_cache/ztf")
ZTF_BAND_MAP = {1: "g", 2: "r", 3: "i"}


def get_client():
    """Get an ALeRCE API client."""
    try:
        from alerce.core import Alerce
        return Alerce()
    except ImportError:
        raise ImportError(
            "The 'alerce' package is required for ZTF data access. "
            "Install with: pip install alerce"
        )


def query_objects_by_class(
    classifier: str = "lc_classifier_top",
    class_name: str = "Periodic",
    n_objects: int = 100,
    probability_threshold: float = 0.5,
) -> pd.DataFrame:
    """Query ALeRCE for objects classified as a given type.

    Parameters
    ----------
    classifier : str
        Classifier name (e.g. 'lc_classifier_top', 'lc_classifier').
    class_name : str
        Class label (e.g. 'Periodic', 'Stochastic', 'Transient', 'SNIa',
        'RRLyr', 'EB', 'LPV', 'AGN').
    n_objects : int
        Maximum number of objects to return.
    probability_threshold : float
        Minimum classification probability.
    """
    client = get_client()
    # ZTF API expects "classifier" and "class" (class_name is mapped to "class" by legacy client)
    objects = client.query_objects(
        survey="ztf",
        classifier=classifier,
        class_name=class_name,
        probability=probability_threshold,
        page_size=min(n_objects, 500),
        page=1,
        format="pandas",
    )
    df = objects.head(n_objects)
    # Index by oid so that df.index.tolist() gives object IDs for get_light_curve
    if df is not None and len(df) > 0 and "oid" in df.columns:
        df = df.set_index("oid")
    return df


def get_light_curve(
    oid: str,
    survey: str = "ztf",
    include_forced: bool = True,
    use_cache: bool = True,
) -> MultiBandLightCurve:
    """Download and convert a ZTF light curve for a single object.

    Parameters
    ----------
    oid : str
        ALeRCE object identifier (e.g. 'ZTF18abbuksn').
    include_forced : bool
        If True, include forced photometry data.
    use_cache : bool
        Cache downloaded data to disk.
    """
    cache_key = f"{oid}_{survey}_forced{include_forced}"
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    client = get_client()

    detections = client.query_detections(oid, format="pandas", survey=survey)
    non_detections = client.query_non_detections(oid, format="pandas", survey=survey)

    forced_phot = None
    if include_forced:
        try:
            forced_phot = client.query_forced_photometry(oid, format="pandas", survey=survey)
        except Exception:
            forced_phot = None

    mblc = _build_multiband(oid, detections, non_detections, forced_phot)

    if use_cache:
        _save_cache(cache_key, mblc)

    return mblc


def get_forced_photometry(
    oid: str,
    survey: str = "ztf",
    use_cache: bool = True,
) -> MultiBandLightCurve:
    """Download only forced photometry for an object.

    Forced photometry includes measurements at the object position
    on all difference images, regardless of whether a detection occurred.
    This is the closest ZTF analog to LSST DIAForcedSource.
    """
    cache_key = f"{oid}_{survey}_forcedonly"
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    client = get_client()
    forced = client.query_forced_photometry(oid, format="pandas", survey=survey)

    mblc = _build_multiband_forced(oid, forced)

    if use_cache:
        _save_cache(cache_key, mblc)

    return mblc


def get_batch_light_curves(
    oids: list[str],
    survey: str = "ztf",
    include_forced: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
) -> list[MultiBandLightCurve]:
    """Download light curves for multiple objects."""
    results = []
    for i, oid in enumerate(oids):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Downloading {i + 1}/{len(oids)}...")
        try:
            mblc = get_light_curve(oid, survey, include_forced, use_cache)
            results.append(mblc)
        except Exception as e:
            if verbose:
                print(f"  Failed for {oid}: {e}")
    return results


def build_ensemble_from_ztf(
    oids: list[str],
    band: str = "g",
    survey: str = "ztf",
    include_forced: bool = True,
    region_id: str | None = None,
    verbose: bool = True,
) -> Ensemble:
    """Build an Ensemble from ZTF objects in a single band."""
    mblcs = get_batch_light_curves(oids, survey, include_forced, verbose=verbose)
    ensemble = Ensemble(region_id=region_id)
    for mblc in mblcs:
        if band in mblc.curves:
            ensemble.add(mblc[band])
    return ensemble


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _build_multiband(
    oid: str,
    detections: pd.DataFrame,
    non_detections: pd.DataFrame,
    forced: pd.DataFrame | None,
) -> MultiBandLightCurve:
    """Convert ALeRCE detection/forced data into MultiBandLightCurve."""
    curves = {}

    if forced is not None and len(forced) > 0:
        df = forced
        time_col = _find_column(df, ["mjd", "jd", "time"])
        flux_col = _find_column(df, ["flux", "forcediffimflux", "difference_flux"])
        err_col = _find_column(df, ["flux_error", "forcediffimfluxunc",
                                     "difference_flux_error", "e_flux"])
        band_col = _find_column(df, ["fid", "band", "filter"])

        if time_col and flux_col and err_col and band_col:
            for fid_val in df[band_col].unique():
                band_df = df[df[band_col] == fid_val].copy()
                band_name = ZTF_BAND_MAP.get(fid_val, str(fid_val))
                band_df = band_df.dropna(subset=[flux_col, err_col])
                if len(band_df) < 5:
                    continue
                curves[band_name] = LightCurve(
                    times=band_df[time_col].values,
                    fluxes=band_df[flux_col].values,
                    flux_errors=np.abs(band_df[err_col].values),
                    band=band_name,
                    metadata={"oid": oid, "source": "ztf_forced", "n_points": len(band_df)},
                )

    if not curves and detections is not None and len(detections) > 0:
        df = detections
        time_col = _find_column(df, ["mjd", "jd", "time"])
        flux_col = _find_column(df, ["flux", "difference_flux", "magpsf"])
        err_col = _find_column(df, ["flux_error", "difference_flux_error",
                                     "e_flux", "sigmapsf"])
        band_col = _find_column(df, ["fid", "band", "filter"])

        if time_col and flux_col and err_col and band_col:
            for fid_val in df[band_col].unique():
                band_df = df[df[band_col] == fid_val].copy()
                band_name = ZTF_BAND_MAP.get(fid_val, str(fid_val))
                band_df = band_df.dropna(subset=[flux_col, err_col])
                if len(band_df) < 5:
                    continue
                curves[band_name] = LightCurve(
                    times=band_df[time_col].values,
                    fluxes=band_df[flux_col].values,
                    flux_errors=np.abs(band_df[err_col].values),
                    band=band_name,
                    metadata={"oid": oid, "source": "ztf_detections", "n_points": len(band_df)},
                )

    return MultiBandLightCurve(curves=curves, object_id=oid)


def _build_multiband_forced(
    oid: str,
    forced: pd.DataFrame,
) -> MultiBandLightCurve:
    """Convert forced photometry dataframe to MultiBandLightCurve."""
    curves = {}
    if forced is None or len(forced) == 0:
        return MultiBandLightCurve(curves=curves, object_id=oid)

    time_col = _find_column(forced, ["mjd", "jd", "time"])
    flux_col = _find_column(forced, ["flux", "forcediffimflux", "difference_flux"])
    err_col = _find_column(forced, ["flux_error", "forcediffimfluxunc",
                                     "difference_flux_error", "e_flux"])
    band_col = _find_column(forced, ["fid", "band", "filter"])

    if not all([time_col, flux_col, err_col, band_col]):
        return MultiBandLightCurve(curves=curves, object_id=oid)

    for fid_val in forced[band_col].unique():
        band_df = forced[forced[band_col] == fid_val].copy()
        band_name = ZTF_BAND_MAP.get(fid_val, str(fid_val))
        band_df = band_df.dropna(subset=[flux_col, err_col])
        if len(band_df) < 5:
            continue
        curves[band_name] = LightCurve(
            times=band_df[time_col].values,
            fluxes=band_df[flux_col].values,
            flux_errors=np.abs(band_df[err_col].values),
            band=band_name,
            metadata={"oid": oid, "source": "ztf_forced", "n_points": len(band_df)},
        )

    return MultiBandLightCurve(curves=curves, object_id=oid)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    for col in candidates:
        for actual in df.columns:
            if col.lower() in actual.lower():
                return actual
    return None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    """Get the cache file path for a given key."""
    safe_key = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{safe_key}.npz"


def _save_cache(key: str, mblc: MultiBandLightCurve) -> None:
    """Save a MultiBandLightCurve to the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)

    data = {}
    for band, lc in mblc.curves.items():
        data[f"{band}_times"] = lc.times
        data[f"{band}_fluxes"] = lc.fluxes
        data[f"{band}_errors"] = lc.flux_errors

    meta = {"object_id": mblc.object_id, "bands": list(mblc.curves.keys())}
    data["_metadata"] = np.array([json.dumps(meta)])
    np.savez_compressed(path, **data)


def _load_cache(key: str) -> MultiBandLightCurve | None:
    """Load a MultiBandLightCurve from cache if available."""
    path = _cache_path(key)
    if not path.exists():
        return None

    try:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["_metadata"][0]))
        curves = {}
        for band in meta["bands"]:
            curves[band] = LightCurve(
                times=data[f"{band}_times"],
                fluxes=data[f"{band}_fluxes"],
                flux_errors=data[f"{band}_errors"],
                band=band,
                metadata={"oid": meta["object_id"], "source": "cache"},
            )
        return MultiBandLightCurve(
            curves=curves,
            object_id=meta["object_id"],
        )
    except Exception:
        return None
