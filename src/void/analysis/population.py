"""Population-level analysis: persistence images + dimensionality reduction.

Converts each source's persistence diagram into a persistence image
(fixed-size vector), then uses UMAP or PCA to visualize population
structure.  The key question: do sub-threshold populations form
distinct clusters that separate from noise?
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from void.data.models import Ensemble, LightCurve
from void.embedding.features import extract_features
from void.embedding.takens import TakensEmbedder
from void.topology.persistence import PersistenceDiagram, compute_persistence


def persistence_image_matrix(
    ensemble: Ensemble,
    embedder: TakensEmbedder | None = None,
    dim: int = 1,
    pixel_size: float = 0.1,
    maxdim: int = 1,
) -> tuple[np.ndarray, list[PersistenceDiagram]]:
    """Compute persistence images for every source in an ensemble.

    Returns
    -------
    image_matrix : np.ndarray of shape (n_sources, n_pixels)
        Flattened persistence images, one per source.
    diagrams : list[PersistenceDiagram]
        The per-source persistence diagrams.
    """
    embedder = embedder or TakensEmbedder(dimension=3, delay=2)
    images = []
    diagrams = []

    for src in ensemble.sources:
        if isinstance(src, LightCurve):
            try:
                cloud = embedder.embed(src)
                pd = compute_persistence(cloud, maxdim=maxdim)
                diagrams.append(pd)
                pi = pd.persistence_image(dim=dim, pixel_size=pixel_size)
                images.append(np.asarray(pi).ravel())
            except Exception:
                diagrams.append(None)
                images.append(None)
        else:
            diagrams.append(None)
            images.append(None)

    valid = [i for i, img in enumerate(images) if img is not None]
    if not valid:
        return np.empty((0, 0)), diagrams

    max_len = max(len(images[i]) for i in valid)
    padded = []
    for i in range(len(images)):
        if images[i] is not None:
            arr = images[i]
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)))
            padded.append(arr[:max_len])
        else:
            padded.append(np.zeros(max_len))

    return np.vstack(padded), diagrams


def population_embedding(
    feature_matrix: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """Project a feature matrix into low dimensions for visualization.

    Parameters
    ----------
    method : str
        'pca', 'umap', or 'tsne'.
    """
    scaled = StandardScaler().fit_transform(feature_matrix)

    if method == "pca":
        return PCA(n_components=n_components).fit_transform(scaled)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components, **kwargs)
        return reducer.fit_transform(scaled)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(kwargs.get("perplexity", 30), len(scaled) - 1)
        return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(scaled)

    raise ValueError(f"Unknown method: {method}")


def cluster_population(
    feature_matrix: np.ndarray,
    method: str = "dbscan",
    **kwargs,
) -> np.ndarray:
    """Cluster sources in feature space.

    Parameters
    ----------
    method : str
        'dbscan', 'hdbscan', or 'kmeans'.

    Returns
    -------
    np.ndarray of shape (n_sources,)
        Cluster labels (-1 = noise for DBSCAN).
    """
    scaled = StandardScaler().fit_transform(feature_matrix)

    if method == "dbscan":
        from sklearn.cluster import DBSCAN
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)

    elif method == "kmeans":
        from sklearn.cluster import KMeans
        n_clusters = kwargs.get("n_clusters", 3)
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)

    raise ValueError(f"Unknown clustering method: {method}")


def compare_tda_vs_classical(
    ensemble: Ensemble,
    embedder: TakensEmbedder | None = None,
    maxdim: int = 1,
) -> dict:
    """Compare TDA features vs classical features for population separation.

    Extracts both TDA-based and classical (statistical + periodogram) features,
    measures how well each feature set separates signal from noise using
    silhouette score on known labels.
    """
    from sklearn.metrics import silhouette_score

    embedder = embedder or TakensEmbedder(dimension=3, delay=2)

    # Full features (stat + period + TDA)
    full_features = ensemble.feature_matrix(
        lambda lc: extract_features(lc, embedder=embedder, compute_tda=True, maxdim=maxdim)
    )

    # Classical only (stat + period, no TDA)
    classical_features = ensemble.feature_matrix(
        lambda lc: extract_features(lc, embedder=embedder, compute_tda=False)
    )

    labels_known = np.array([
        1 if lc.metadata.get("source_type", "noise") != "noise" else 0
        for lc in ensemble.sources
        if isinstance(lc, LightCurve)
    ])

    if len(np.unique(labels_known)) < 2:
        return {"full_silhouette": 0.0, "classical_silhouette": 0.0, "tda_gain": 0.0}

    full_scaled = StandardScaler().fit_transform(full_features)
    classical_scaled = StandardScaler().fit_transform(classical_features)

    sil_full = silhouette_score(full_scaled, labels_known)
    sil_classical = silhouette_score(classical_scaled, labels_known)

    return {
        "full_silhouette": float(sil_full),
        "classical_silhouette": float(sil_classical),
        "tda_gain": float(sil_full - sil_classical),
        "n_tda_features": full_features.shape[1] - classical_features.shape[1],
    }
