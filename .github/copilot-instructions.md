# Staring Into the Void — Project Guidelines

## Overview

Scientific Python project applying topological data analysis (persistent homology) to sub-threshold astronomical light curves from LSST forced photometry. See [Staring_Into_The_Void.md](../Staring_Into_The_Void.md) for the full concept document.

## Architecture

**Pipeline flow**: Data → Embedding → Topology → Analysis → Viz

| Module | Responsibility | Key types |
|--------|---------------|-----------|
| `src/void/data/` | Light curve models, synthetic generation, ZTF ingestion | `LightCurve`, `MultiBandLightCurve`, `Ensemble` |
| `src/void/embedding/` | Takens delay embedding, feature extraction | `TakensEmbedder`, `extract_features()` |
| `src/void/topology/` | Persistent homology, null models, diagram distances | `PersistenceDiagram`, `NullDistribution` |
| `src/void/analysis/` | Anomaly detection, attenuation experiments | `AnomalyDetector`, `AnomalyResult` |
| `src/void/viz/` | Plotting utilities (all return `Figure` objects) | `plot_*()` functions |

**Key design**: Analysis operates on *feature-space ensemble point clouds*, not individual light curves. Each source → feature vector; ensemble → point cloud in feature space; persistent homology on the ensemble cloud.

## Build and Test

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run tests
pytest

# Run full pipeline demo (~5-10 min)
python run_pipeline.py

# Quick power analysis (~1-2 min)
python run_power_analysis.py
```

Optional dependency groups: `tda-extra` (giotto-tda, gudhi), `ztf` (alerce), `umap` (umap-learn).

## Conventions

### Data models
- All domain objects are **`@dataclass`** with `__post_init__` validation.
- Every `LightCurve`, `Ensemble`, and result object carries a `.metadata` dict for provenance tracking. Preserve this when creating or transforming objects.

### Class vs function
- **Classes** for stateful processors (`TakensEmbedder`, `AnomalyDetector`) — instantiate once, call methods repeatedly.
- **Pure functions** for generators (`generate_periodic()`) and computations (`compute_persistence()`, `extract_features()`).

### RNG / reproducibility
- Always use `np.random.default_rng(seed)` (never legacy `np.random.RandomState`).
- Pass `rng` through call chains; spawn child RNGs via `rng.integers(0, 2**63)`.

### Type hints
- Full type annotation on all public functions and methods. Use `X | None` union syntax (Python 3.10+).

### Imports
- Use `from __future__ import annotations` in all modules.
- Optional TDA dependencies (ripser, persim, giotto-tda) are guarded with `try/except ImportError`.

### Numerical code
- Short numpy-idiomatic variable names are fine in numerical contexts: `lc`, `cloud`, `pd`, `dgm`, `feats`.
- Prefer vectorized operations; use `np.errstate` for safe division.

### Homology dimensions
- `dim` parameter throughout: 0 = connected components, 1 = loops, 2 = voids. All summary metrics accept `dim`.

## Testing

- Fixtures in `tests/conftest.py` — shared `rng`, source-type LightCurves, ensembles.
- Tests grouped in **classes** (`TestTakensEmbedder`, `TestPersistenceDiagram`).
- Pattern: assert shape/type first, then properties, then boundary/error cases with `pytest.raises(ValueError, match=...)`.
- Use `pytest.approx()` for float comparisons.

## Scripts

All entry-point scripts (`run_*.py`) share:
- Windows UTF-8 stdout reconfiguration
- `matplotlib.use("Agg")` for headless rendering
- Output saved to `output/` directory
- Seeded RNG (`rng = np.random.default_rng(42)`)
