# Staring Into the Void

**Sub-Threshold Topological Signal Recovery from LSST Forced Photometry**

[![tests](https://img.shields.io/badge/tests-85%20passing-brightgreen)](tests/)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![status](https://img.shields.io/badge/status-Phase%201%20methodology-orange)](docs/ROADMAP.md)

Applies topological data analysis (persistent homology) to ensembles of
forced-photometry light curves *below* the LSST 5σ single-epoch
detection threshold, searching for astrophysical structure in
populations no broker classifier can see.

![Topology signature of three light-curve classes](output/hero_01_topology_signature.png)

*Top: Takens-embedded point clouds for periodic, transient, and stochastic
sources at SNR = 10. Bottom: their persistence diagrams. The H₁ count and
lifetime sum encode the structural signature recovered from the time series.*

- **What it does** — [docs/SCIENCE_PITCH.md](docs/SCIENCE_PITCH.md)
- **Where it's going** — [docs/ROADMAP.md](docs/ROADMAP.md)
- **Who it's for** — [docs/OUTREACH.md](docs/OUTREACH.md)
- **Scientific premise** — [Staring_Into_The_Void.md](Staring_Into_The_Void.md)

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -e ".[dev]"
pytest                          # 85 tests, ~9 s
python run_phase1_methodology.py   # regenerates output/paper/*.png
```

```python
from void.data.synthetic import generate_periodic
from void.embedding.takens import TakensEmbedder
from void.topology.persistence import compute_persistence

lc = generate_periodic(snr=3.0, period=5.0, n_epochs=200)
cloud = TakensEmbedder(dimension=3, delay=1).embed(lc)
dgm = compute_persistence(cloud, maxdim=1)
```

## Real-data access

| Source | Module | Status |
|---|---|---|
| Synthetic LSST-like cadence | `void.data.synthetic` | ✅ |
| ZTF forced photometry via ALeRCE | `void.data.ztf` | ✅ live |
| Rubin RSP TAP (DP1, DP0.2) | `void.data.rubin` | ✅ schema configurable, RSP token required |

```powershell
# Switch the active TAP schema without code changes
$env:VOID_RUBIN_SCHEMA = "dp1"                 # default (LSSTComCam, public Jun 2025)
$env:VOID_RUBIN_SCHEMA = "dp02_dc2_catalogs"   # DC2 simulation
```

## Validation tiers

Three independent layers cross-check the pipeline:

| Tier | Purpose | Invocation |
|------|---------|------------|
| **Local pytest** | Unit + integration tests on every commit | `pytest -q` (87 passing) |
| **Wolfram symbolic ground-truth** | Closed-form Vietoris-Rips H₁ at 30-digit precision; pinned baseline | `wolframscript -file validation/symbolic_persistence.wls` |
| **HF Jobs cloud Monte Carlo** | N=10 000 null calibration + attenuation sweep on real ZTF RRL | `python scripts/hf_jobs/submit.py --live` |

The symbolic baseline at [validation/symbolic_diagrams.json](validation/symbolic_diagrams.json)
is re-checked by [tests/test_symbolic_validation.py](tests/test_symbolic_validation.py)
on every test run, ensuring `ripser` agrees with Mathematica's exact arithmetic
to within float32 precision (≈1e-5).

### Reproducibility

Cloud-run artifacts (NPZ archives, manifests, plots) are written to the public
dataset [`bshepp/staring-into-the-void-runs`](https://huggingface.co/datasets/bshepp/staring-into-the-void-runs).
See [scripts/hf_jobs/README.md](scripts/hf_jobs/README.md) for setup and
local artifact retrieval.

## Project structure

```
src/void/
  data/        LightCurve / Ensemble models, synthetic, ZTF, Rubin TAP
  embedding/   Takens delay embedding, feature extraction
  topology/    Persistent homology, distances, null model, sliding window
  analysis/    Anomaly detector, attenuation experiment, population stats
  viz/         Matplotlib plotting (all return Figure)

run_pipeline.py              full synthetic end-to-end demo
run_phase1_methodology.py    paper-quality experiments 1–6
run_power_analysis.py        SNR × population-size sensitivity
run_ztf_validation.py        ALeRCE → topology pipeline

notebooks/   01–05 interactive; 06 (DP1) pending RSP access
tests/       85 unit tests, all passing
output/      12 publication-style figures (regenerable)
docs/        SCIENCE_PITCH.md · OUTREACH.md · ROADMAP.md
```

## Notebooks

| Notebook | Description |
|---|---|
| `01_synthetic_proof_of_concept` | TDA on synthetic light curves at various SNR |
| `02_embedding_exploration` | Takens parameter selection and visualization |
| `03_ztf_real_data` | Pipeline on real ZTF forced photometry |
| `04_attenuation_experiment` | Attenuate known sources, test recovery |
| `05_null_model_calibration` | Power analysis: detection vs SNR and population size |

## Dependencies

- **TDA** — `ripser`, `persim` (core); `giotto-tda`, `gudhi` (`tda-extra` group).
- **Astronomy** — `astropy`, `alerce`, `pyvo` (`rubin` group).
- **Core** — `numpy`, `scipy`, `pandas`, `scikit-learn`.
- **Viz** — `matplotlib`, `plotly`, `umap-learn` (`umap` group).

## Citation

If you use this pipeline, please cite via [`CITATION.cff`](CITATION.cff)
(rendered on GitHub as a "Cite this repository" sidebar action).  A
Zenodo DOI will be minted with the next tagged release.

## License

MIT — see [`LICENSE`](LICENSE).
