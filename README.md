# Staring Into the Void

**Sub-Threshold Topological Signal Recovery from LSST Forced Photometry**

This project applies topological data analysis (persistent homology) to forced
photometry measurements below the LSST 5-sigma detection threshold, searching for
astrophysical structure in the population of non-detections.

See [Staring_Into_The_Void.md](Staring_Into_The_Void.md) for the full concept document.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install the package with all dependencies
pip install -e ".[dev]"
```

## Project Structure

```
src/void/
  data/         Light curve data models, synthetic generation, ZTF ingestion
  embedding/    Takens delay embedding and feature extraction
  topology/     Persistent homology, diagram distances, null models
  analysis/     Anomaly detection, attenuation experiments
  viz/          Plotting utilities

notebooks/      Interactive exploration and experiments
tests/          Unit tests
```

## Quick Start

```python
from void.data.synthetic import generate_periodic, generate_noise
from void.embedding.takens import TakensEmbedder
from void.topology.persistence import compute_persistence

# Generate a faint periodic signal at 3-sigma SNR
lc = generate_periodic(snr=3.0, period=5.0, n_epochs=200)

# Embed into phase space
embedder = TakensEmbedder(dimension=3, delay=1)
cloud = embedder.embed(lc)

# Compute persistent homology
diagrams = compute_persistence(cloud, maxdim=1)
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_synthetic_proof_of_concept` | TDA on synthetic light curves at various SNR |
| `02_embedding_exploration` | Takens parameter selection and visualization |
| `03_ztf_real_data` | Pipeline on real ZTF forced photometry |
| `04_attenuation_experiment` | Attenuate known sources, test recovery |
| `05_null_model_calibration` | Power analysis: detection vs SNR and population size |

## Dependencies

- **TDA**: giotto-tda, ripser, persim, gudhi
- **Astronomy**: astropy, alerce
- **Core**: numpy, scipy, pandas, scikit-learn
- **Visualization**: matplotlib, plotly
