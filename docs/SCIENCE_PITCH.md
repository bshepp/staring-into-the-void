# Staring Into the Void — Science Pitch

> One-page pitch for Rubin Science Collaboration members, brokers, and
> data-rights holders.

## The problem

The Vera C. Rubin Observatory will produce ~10 million difference-image
alerts every night.  Every alert pipeline — `ALeRCE`, `Lasair`, `Fink`,
`ANTARES`, `BABAMUL` — operates on the **detection stream**: sources
that crossed a 5σ single-epoch threshold.

But the most interesting populations live **below** that line:

- Faint extragalactic transients (kilonovae at >200 Mpc, intermediate-luminosity SNe).
- Microlensing by sub-solar-mass dark-matter substructure (event amplitudes ~1–3σ per epoch).
- Long-period, low-amplitude variables that never trigger an alert in any single visit.
- Pre-explosion precursors of CCSNe and outbursts of changing-look AGN.

Per-source SNR is too low to classify *any single object*.  But the
**population as a whole** carries a topological fingerprint that
standard feature-space classifiers cannot see.

## The method

We treat each source's light curve as a delay-embedded point cloud
(Takens), extract a fixed-length feature vector that combines
statistical, periodicity, and persistent-homology summaries, and then
compute persistent homology *of the ensemble cloud* — the population
itself becomes a topological object.

A signal population (RR Lyrae, microlensing, kilonovae) **distorts** the
$H_0/H_1$ persistence of the noise cloud in a way that pure noise
cannot mimic.  Significance is established against a permutation /
phase-randomized null distribution with Bonferroni correction.

## Why now

- LSST DP1 (LSSTComCam) was released **30 June 2025** — public, no
  data-rights gate.  `DiaObject` / `DiaSource` are queryable today.
- DP2 lands **mid-2026** with the first DiaForcedSource catalogs from
  LSSTCam commissioning — exactly the data product this pipeline
  consumes.
- Survey operations begin late 2026.  A validated, citable methodology
  needs to be on arXiv **before** the first PPDB drop, so the community
  has a tool to point at sub-threshold residuals from day one.

## What's already built

A complete, tested, reproducible Python pipeline:

| Stage | Module | Status |
|---|---|---|
| Synthetic light curves | `void.data.synthetic` | ✅ 6 source types, LSST cadence |
| Real ZTF data | `void.data.ztf` (ALeRCE) | ✅ Live, end-to-end demo |
| Rubin RSP TAP client | `void.data.rubin` | ✅ DP1 / DP0.2 schemas |
| Takens delay embedding | `void.embedding.takens` | ✅ + automatic τ, m selection |
| Persistent homology | `void.topology.persistence` | ✅ via `ripser` |
| Null-model permutation tests | `void.topology.null_model` | ✅ Bonferroni-corrected |
| Anomaly detection | `void.analysis.anomaly` | ✅ + power analysis |
| Sub-threshold attenuation experiment | `void.analysis.attenuation` | ✅ recovery curves |

- **87 unit tests, all passing** (`pytest`), including a Wolfram
  Mathematica symbolic ground-truth check at 30-digit precision.
- **12 publication-style figures** in `output/` from one
  `python run_phase1_methodology.py` invocation, plus dedicated
  promotion artifacts in `output/hero_*.png`.
- MIT-licensed, citable (`CITATION.cff`).

## Dual-validation discipline

This is, deliberately, not a single-stack result.  Every persistent-homology
claim is cross-checked along two orthogonal axes:

1. **Numerical Monte Carlo** — `void.topology.null_model` builds a calibrated
   null distribution by resampling pure-noise ensembles through the same
   embedding + persistence pipeline.  Significance is reported as an empirical
   p-value with explicit floor `1/N`.  Cloud-scale runs (`N = 10⁴`) execute
   off-host on Hugging Face Jobs and publish to the public dataset
   [`bshepp/staring-into-the-void-runs`](https://huggingface.co/datasets/bshepp/staring-into-the-void-runs).
2. **Symbolic ground-truth** — `validation/symbolic_persistence.wls` computes
   closed-form Vietoris-Rips H₁ birth/death pairs in Wolfram Mathematica at
   arbitrary precision.  `tests/test_symbolic_validation.py` re-checks the
   pinned baseline on every CI run, ensuring `ripser` agrees with exact
   arithmetic to within float32 precision (≈ 1e-5).

A spurious detection has to fool both layers simultaneously.  The first
real-data run did not, and produced a clean negative result — exactly the
behaviour the discipline is designed to enforce.

## What I'm asking for

1. **An RSP account / project allocation** to run the pipeline against
   real DP1 (and DP2 when it lands) — currently I am operating without
   institutional affiliation.
2. **Co-authorship or constructive review** on the Phase 1 methodology
   paper from anyone in TVS, ISSC, DESC (microlensing/dark-matter
   substructure), or AGN SC who finds the approach interesting.
3. **A pointer into the LSST in-kind contribution program** — the
   "sub-threshold residual stream" could be a community-broker
   complement and an obvious in-kind deliverable.
4. **Feedback on the null-model design**.  The question "what does noise
   look like topologically in a survey of 10M nightly forced sources?"
   needs more eyes than mine.

## Three-paper roadmap

1. **Methodology paper** (Q3 2026, arXiv astro-ph.IM) — synthetic +
   ZTF + DP1 validation, attenuation recovery thresholds, false-positive
   calibration.
2. **PPDB detection paper** (early 2027) — first sub-threshold
   population recovery from LSST DiaForcedSource.
3. **Discovery paper** (2027+) — temporal evolution of a recovered
   population (e.g., faint dwarf-galaxy supernova population, or a
   microlensing optical-depth measurement against M31 / LMC).

## Contact

Brian Sheppard · independent researcher · MIT-licensed code
GitHub: <https://github.com/bshepp/staring-into-the-void>
