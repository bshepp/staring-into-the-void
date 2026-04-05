# Staring Into the Void

### Sub-Threshold Topological Signal Recovery from LSST Forced Photometry

**Brian Sheppard — Independent Researcher**
**Project Concept Document — February 2026**

---

## The Core Insight

Everyone is looking at what lit up. We want to look at what *almost* lit up.

The Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) will generate up to 7 million alerts per night — detections of sources above a 5σ signal-to-noise threshold on difference images. Seven community brokers receive this stream and apply machine learning classifiers to sort detections into known taxonomies: supernovae, variable stars, AGN, asteroids, and so on.

But the 5σ threshold is an engineering decision, not a physical boundary. Below that line, the telescope still measures flux at every position where a source has previously been detected. These measurements — called **forced photometry** — are recorded in the Prompt Products Database (PPDB) as `DIAForcedSource` records. They represent the observatory staring at positions in the sky and recording what it sees, even when what it sees is consistent with noise.

These forced photometry non-detections are the void. And the void has structure.

---

## Why This Matters

### The Invisible Population

Consider a real astrophysical source that fluctuates between 2σ and 4.5σ across hundreds of visits over ten years. It will *never* trigger an alert. It will never appear in any broker's classification pipeline. It does not exist in the alert-stream view of the universe.

But it is real. And LSST will measure it hundreds of times.

These objects include:

- **Faint variable stars** below the single-epoch detection limit but clearly periodic across the full light curve
- **Distant supernovae** whose peak brightness falls below threshold — events that are individually invisible but statistically present
- **Slow transients** (tidal disruption events, changing-look AGN) whose rise times are long enough that no single epoch exceeds 5σ, but whose cumulative trend is unmistakable
- **Gravitational microlensing events by dark matter substructure** — predicted to produce characteristic sub-threshold brightening patterns (NFW subhalo profiles, boson star caustics)
- **Unknown populations** — objects that don't fit any existing taxonomy, precisely because they've never been bright enough to classify

The alert stream is, by construction, blind to all of these.

### The Scale of the Void

LSST will image approximately 18,000 square degrees of the southern sky with ~825 visits per position over 10 years, across six photometric bands (u, g, r, i, z, y). For every `DIAObject` in the database, forced photometry is performed on *every* visit image at the object's position — even when no source is independently detected.

This means the forced photometry catalog will contain orders of magnitude more measurements than the alert stream. The alert stream records *events*. The forced photometry records *measurements at locations where events once happened*, across the entire survey duration. The cumulative information content is enormous, and almost entirely unexploited.

---

## The Method: Topological Signal Recovery

### Philosophy

This project applies a consistent research methodology: **finding exploitable structure at boundaries in dynamic systems**. The detection threshold is a boundary. The forced photometry residuals live at that boundary. The question is whether the topology of those residuals — their shape, their clustering, their temporal correlations — contains recoverable signal about real astrophysical populations.

We are not doing simple stacking (co-adding images to increase SNR). We are analyzing the *structure* of the sub-threshold measurements as a population, using tools from topological data analysis (TDA) that are sensitive to features invisible to conventional statistical methods.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  LSST Prompt Products Database (PPDB)                   │
│  DIAForcedSource catalog — forced photometry at all     │
│  DIAObject positions, all epochs, all bands             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Sub-Threshold Population Extraction           │
│  • Select DIAObjects with zero or few alert triggers    │
│  • Extract full forced photometry light curves          │
│  • Multi-band flux time series + uncertainty estimates  │
│  • Compute upper limits from noise estimates where      │
│    forced photometry is unavailable                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Feature Embedding                             │
│  • Takens delay embedding of each light curve →         │
│    point cloud in reconstructed phase space             │
│  • Multi-band: embed each band separately, then         │
│    concatenate or compute cross-band features           │
│  • Extract: persistence entropy, amplitude, feature     │
│    counts, linear slope, periodicity measures,          │
│    cross-band color evolution                           │
│  • Ensemble embedding: represent all sub-threshold      │
│    objects in a region as a single point cloud in       │
│    high-dimensional feature space                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Topological Analysis                          │
│  • Persistent homology (Vietoris-Rips filtration)       │
│    on the ensemble point cloud                          │
│  • Compute persistence diagrams & barcodes              │
│  • H0: connected components → clustering of sub-        │
│    threshold source populations                         │
│  • H1: loops → periodic populations or cyclic           │
│    behavior in the sub-threshold regime                 │
│  • H2: voids → spatial/feature exclusion zones          │
│  • Compare to null model (pure noise forced photometry  │
│    with no astrophysical signal)                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: Anomaly Detection & Signal Recovery           │
│  • Persistence diagram distance from null → anomaly     │
│    score for sky regions                                │
│  • Identify regions where sub-threshold topology        │
│    deviates significantly from noise expectation        │
│  • Within anomalous regions: extract candidate          │
│    sub-threshold sources via sparse recovery /          │
│    compressed sensing on temporal flux vectors          │
│  • Cross-match candidates with multi-wavelength         │
│    catalogs (Gaia, WISE, Chandra, eROSITA)             │
│  • Flag candidates for follow-up observation            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 5: Temporal Evolution                            │
│  • Track persistence diagrams across rolling time       │
│    windows (weekly, monthly, seasonal)                  │
│  • Detect topological phase transitions in the          │
│    sub-threshold population                             │
│  • Correlate with survey strategy changes, seasonal     │
│    atmospheric effects, and known astrophysical events  │
│  • Identify *new* topological features that emerge      │
│    over time → potential discovery channel              │
└─────────────────────────────────────────────────────────┘
```

---

## Technical Components

### Topological Data Analysis Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **Ripser** | Fast persistent homology computation | Optimized for Vietoris-Rips complexes |
| **GUDHI** | Full TDA pipeline | Includes alpha complexes, bottleneck distance |
| **Giotto-tda** | ML-integrated TDA | scikit-learn compatible, Takens embedding built-in |
| **Persim** | Persistence diagram comparison | Wasserstein/bottleneck distances, persistence images |

### Key Mathematical Framework

**Takens Delay Embedding Theorem**: A scalar time series x(t) sampled at discrete intervals can be embedded into a d-dimensional phase space by constructing vectors:

```
v(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]
```

For a forced photometry light curve with sparse, irregular sampling, we adapt this by interpolating or using the natural cadence structure of LSST visits. The resulting point cloud preserves the topological properties of the underlying dynamical system — periodic sources produce loops, chaotic sources produce strange attractors, noise produces featureless clouds.

**Persistent Homology**: Computes topological invariants (connected components, loops, voids) across a continuous range of spatial scales. Features that persist across many scales are "real"; features that appear and vanish quickly are noise. The persistence diagram is a compact representation of the multi-scale topological structure.

**Null Model Construction**: Generate synthetic forced photometry by injecting no astrophysical signal into the LSST noise model (using the OpSim survey strategy simulations + rubin-sim). Compute persistence diagrams for the null case. Any statistically significant deviation in real data indicates sub-threshold structure.

### Data Access

| Resource | Availability | Access Method |
|----------|-------------|---------------|
| **DP1 commissioning data** | Available now | Rubin Science Platform (TAP/ADQL + Butler) |
| **DiaObject table** | Available in DP1 | TAP queries, includes per-band flux statistics |
| **DIAForcedSource** | Mid-2026 (PPDB) | TAP service via Rubin Data Access Centers |
| **DP2 (LSSTCam data)** | July–September 2026 | Rubin Science Platform |
| **Alert stream** | Live now | Public via community brokers (ALeRCE, Lasair, Fink, etc.) |
| **Full LSST survey** | Late 2026 start | 10-year cumulative data releases |

---

## Connection to Companion Projects

### "Staring Into the Void" sits within a broader research program:

**Project 1: Nightly Alert Stream Topology (ref: approach #6)**
Apply persistent homology to the full nightly alert stream as a point cloud in feature space. Track topological evolution night-to-night. Detect structural transitions in the alert population. This operates *above* threshold — looking at the shape of what's visible.

**Project 2: Residual Topology of Classifier Boundaries (ref: approach #7)**
Analyze the persistent homology of the classification uncertainty space — alerts where brokers disagree, where confidence is low, where objects sit on decision boundaries. The topology of these residuals encodes information about missing taxonomy categories.

**Project 3: Staring Into the Void (this document)**
Analyze the persistent homology of forced photometry below threshold. The topology of what's *not* visible.

Together, these three projects form a complete topological analysis of LSST data across the full signal regime:

```
  Above threshold          At threshold           Below threshold
  (alert stream)      (classifier boundaries)   (forced photometry)
       │                      │                        │
   Project 1              Project 2              Project 3
   Alert Stream           Residual               Staring Into
   Topology               Topology               the Void
       │                      │                        │
       └──────────────────────┴────────────────────────┘
                              │
              Unified boundary topology framework:
              finding exploitable structure at the
              edges of dynamic observational systems
```

---

## Phased Approach

### Phase 1: Proof of Concept (Now – Summer 2026)
- Access DP1 commissioning data via Rubin Science Platform
- Query DiaObject table for objects with flux statistics near detection threshold
- Implement Takens embedding + persistent homology pipeline on available light curve data
- Construct null model from DP1 noise characteristics
- Demonstrate: can persistent homology distinguish injected sub-threshold signals from pure noise?
- **Deliverable**: Methodology paper on arXiv

### Phase 2: PPDB Forced Photometry (Mid-2026 – Early 2027)
- Access DIAForcedSource catalog when PPDB becomes available
- Extract forced photometry for objects with zero/few alert triggers
- Apply full pipeline to real sub-threshold data
- Identify first candidate sub-threshold populations
- Cross-match with external catalogs
- **Deliverable**: Detection paper on arXiv

### Phase 3: Temporal Evolution & Scale (2027+)
- Track sub-threshold topology across seasonal and annual timescales
- Detect topological phase transitions as survey accumulates depth
- Scale analysis to full LSST footprint using rolling data releases
- Integrate with alert stream topology (Project 1) for cross-threshold analysis
- **Deliverable**: Discovery paper — what's in the void?

---

## Why This Hasn't Been Done

1. **The brokers can't see it.** Community brokers only receive alerts (above-threshold detections). The forced photometry lives in the PPDB, which isn't even available yet. The entire broker ecosystem is structurally blind to sub-threshold populations.

2. **Standard methods don't apply.** Conventional light curve classification requires enough signal to extract features. Sub-threshold light curves look like noise individually. The insight is to analyze them as an *ensemble* with topological tools, not as individual classification problems.

3. **TDA is barely used in time-domain astronomy.** Persistent homology has been applied to cosmic shear maps and large-scale structure, but not to transient/variable star light curves, and definitely not to forced photometry non-detections.

4. **The data didn't exist until now.** LSST is the first survey with the combination of depth, cadence, and forced photometry infrastructure to make this analysis possible. Previous surveys (ZTF, PTF) had neither the depth nor the systematic forced photometry coverage.

5. **Nobody thinks to look at nothing.** The field's attention — reasonably — is focused on the 7 million alerts per night. The forced photometry catalog is infrastructure, not science. The idea that the most interesting discoveries might be *below* the detection threshold is counterintuitive. But that's exactly the kind of boundary where exploitable structure lives.

---

## Resources Required

| Resource | Requirement | Status |
|----------|------------|--------|
| **Compute** | Moderate — TDA is O(n³) worst case but Ripser is highly optimized; analysis is per-region, not full sky at once | Community tech lab access; personal workstation |
| **Storage** | Minimal for Phase 1 (DP1 queries). Phase 2: selective PPDB queries, ~10-100 GB per analysis run | Manageable locally |
| **Software** | Python: giotto-tda, ripser, gudhi, persim, fastavro, astropy, lsst.rsp | All open source |
| **Data access** | Rubin Science Platform account for DP1/DP2; PPDB access (mid-2026) | Requires data rights — available to US/Chilean researchers and in-kind contributors |
| **Institutional affiliation** | Not required for arXiv publication; may be needed for Rubin data rights | Explore LSST Discovery Alliance, in-kind contributions, or Science Collaboration membership |

---

## Key References

- DMTN-102: LSST Alerts Key Numbers (alert packet size, rates, contents)
- LDM-612: Plans and Policies for LSST Alert Distribution
- LSE-163: Data Products Definition Document (DIASource, DIAObject, DIAForcedSource schemas)
- Crispim Romão et al. (2025): Anomaly detection with isolation forests for LSST transients (MNRAS 543)
- Burger et al. (2021): Persistent homology in cosmic shear (A&A)
- Nature Scientific Reports (2025): Machine learning of time series data using persistent homology via recurrence plots
- Murugan & Robertson: TDA in astronomy — Mapper algorithm on astronomical datasets
- Takens (1981): Detecting strange attractors in turbulence (delay embedding theorem)
- DESC AI/ML White Paper (2026): Opportunities in AI/ML for Rubin LSST (arXiv:2601.14235)

---

## The Name

*Staring Into the Void* — because the most interesting thing about the universe might be what you see when you look at the places where you see nothing.

---

*This project is independent research with no institutional affiliation. All analysis will use publicly available LSST data products and open-source software.*
