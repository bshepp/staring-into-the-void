# Outreach Plan

> Concrete list of audiences, channels, and templated messages for
> promoting *Staring Into the Void* to the people who hold the data and
> the review power.  Use **only after** the methodology paper draft and
> the DP1 smoke-test notebook are merged (see [ROADMAP.md](ROADMAP.md)).

## Sequencing rule

Do **not** cold-email Science Collaboration chairs before:

1. The DP1 smoke-test notebook (`notebooks/06_rubin_dp1.ipynb`) runs
   end-to-end on real DP1 data and is committed.
2. A draft methodology paper exists (even as a v1 arXiv preprint).
3. A Zenodo DOI is minted for the v0.1 code release.

Promotion without these is asking strangers for time before
demonstrating that you respect theirs.

---

## Tier 1 — Highest leverage

### 1. Rubin Science Collaborations

The eight LSST Science Collaborations are the gatekeepers for survey
science.  Any of them can sponsor an in-kind contribution, co-author a
paper, or invite you to a working group call.

| SC | Why they care | Best entry point |
|---|---|---|
| **Transients & Variable Stars (TVS)** | Sub-threshold transients are *the* TVS frontier. Microlensing, kilonovae, faint variables. | TVS Anomaly Detection / Microlensing working groups |
| **Informatics & Statistics (ISSC)** | Methodology paper home. TDA + permutation null testing fits ISSC mandate. | ISSC general mailing list, ADASS talk |
| **Dark Energy (DESC)** | Sub-threshold SN cosmology, photometric SN populations below detection. | DESC Transients working group |
| **AGN** | Changing-look AGN, low-luminosity AGN variability. | AGN SC general meeting |
| **Stars, MW & Local Volume (SMWLV)** | Faint variable stars, M31/LMC microlensing. | SMWLV Variables working group |
| **Strong Lensing (SLSC)** | Microlensing of lensed quasars, sub-structure. | Joint with TVS |
| **Solar System (SSSC)** | (Lower priority — different signal regime.) | — |
| **Galaxies** | (Lower priority unless tidal disruption events.) | — |

**Affiliate membership** is open to most Collaborations even without a
US/Chilean institutional affiliation.  Apply.

### 2. Community brokers

Brokers are operational, well-staffed, and constantly looking for
complementary value-add streams.  This is your fastest path to real
on-sky data and a scientific home.

| Broker | Why they're a fit | How to approach |
|---|---|---|
| **Fink** (CNRS / LSST France) | Already runs ML-based anomaly detection; receptive to TDA. | Open a discussion on the Fink GitHub repo or the Slack. |
| **ALeRCE** (Chile) | Already used in this codebase. Knows the schema we extend. | Email the broker leads; cite the ZTF validation notebook. |
| **Lasair** (Edinburgh) | Filter framework is extensible to user code. | Submit a Lasair filter that flags sub-threshold residuals. |
| **ANTARES** (NOIRLab) | US base; broad classifier portfolio. | Open a feature request / talk at NOIRLab community day. |
| **BABAMUL** (Brazil) | Newer; open to collaborations. | Direct email. |

### 3. Rubin in-kind contribution program

Independent researchers without US/Chilean data rights can still
contribute via the **LSST Discovery Alliance in-kind contribution
program**.  A "sub-threshold residual stream" is a *natural*
in-kind deliverable: you give the project a derived data product and
get formal data rights in return.

- Page: <https://www.lsstcorporation.org/in-kind-program>
- Submit a Letter of Inquiry (LoI) template — see below.

---

## Tier 2 — Specific researchers to engage

These are people whose published work is the closest neighbor to this
project.  Read their latest papers first; reference them in any email.

- **Crispim Romão et al.** (anomaly detection in LSST alerts; UK / Lisbon).
- **Burger et al.** (TDA in cosmology and astronomy).
- **Murugan & Robertson** (Mapper algorithm for galaxy populations).
- **Sánchez-Sáez, Förster et al.** (ALeRCE classifiers; Chile).
- **Möller, Boone, Pruzhinskaya, Ishida** (active anomaly detection
  for transient brokers).
- **Niemiec, Smith et al.** (microlensing / dark substructure with LSST).

Cold-email or DM via mastodon.social / bluesky.app.

## Tier 3 — Public + community

| Channel | Cadence | Goal |
|---|---|---|
| arXiv astro-ph.IM (+ astro-ph.HE cross-list) | Once, on submission | Citable anchor |
| `#astrodon` on Mastodon, Bluesky `astro.science` | Day-of arXiv | Visibility |
| AAS abstract (Jan + Jun) | Annual | Talk / poster |
| ADASS (Sep–Oct) | Annual | Methodology venue |
| `.Astronomy` unconference | Annual | Tooling visibility |
| Rubin Community Workshop | Annual (mid-year) | In-person network |

---

## Email templates

### Template A — Science Collaboration affiliate inquiry

> Subject: Affiliate-membership inquiry — sub-threshold topology pipeline for DP1/DP2
>
> Dear [Chair name],
>
> I am an independent researcher (no current institutional affiliation)
> seeking affiliate membership in the [TVS / ISSC / DESC] Science
> Collaboration.  I have built and open-sourced a pipeline that applies
> persistent homology to ensembles of LSST DiaForcedSource light
> curves to recover sub-threshold astrophysical populations that are
> invisible to current alert-stream classifiers.
>
> A short overview is at:
> <https://github.com/bshepp/staring-into-the-void/blob/main/docs/SCIENCE_PITCH.md>
>
> The methodology paper is in preparation for arXiv submission in
> [month/year], and a working DP1 smoke-test notebook is included in
> the repository.  85 unit tests pass; the full Phase 1 figure set
> (`output/`) regenerates in under 10 minutes from `python
> run_phase1_methodology.py`.
>
> I would welcome the opportunity to present at a working-group call
> and to receive feedback from members whose work this most directly
> builds on.  I am also exploring whether the "sub-threshold residual
> stream" could be developed into an in-kind contribution.
>
> With thanks,
> Brian Sheppard

### Template B — Broker integration proposal

> Subject: Proposal — sub-threshold residual stream as a complementary [Fink / ALeRCE / Lasair] product
>
> Dear [Broker leads],
>
> I have an open-source pipeline (MIT, Python) that runs persistent
> homology on populations of below-threshold forced-photometry light
> curves and flags ensembles whose topological signature deviates from
> a phase-randomized null.  The detection target is exactly what the
> per-source classifiers cannot see: faint kilonovae, microlensing,
> long-period low-amplitude variables.
>
> I have already validated against ZTF via your alerce-client
> [or: similar] interface.  I would like to discuss whether
> [Fink / ALeRCE / Lasair] would be willing to host a daily run as a
> filter or science module, with the output exposed as an additional
> stream.  I will do the implementation work; what I would need is
> guidance on the contribution path and a sponsor.
>
> Code: <https://github.com/bshepp/staring-into-the-void>
> Pitch: <https://github.com/bshepp/staring-into-the-void/blob/main/docs/SCIENCE_PITCH.md>
>
> Brian Sheppard

### Template C — In-kind contribution Letter of Inquiry

> Subject: Letter of Inquiry — Sub-threshold residual stream as in-kind contribution
>
> To: LSST Discovery Alliance In-Kind Program
>
> **Proposing party:** Brian Sheppard (independent researcher)
> **Title:** Sub-threshold topological residual stream
> **Estimated effort:** [N] FTE-months over [M] months
>
> **Summary.**  We propose to deliver a daily-cadence value-added
> stream that flags populations of LSST DiaForcedSource sources whose
> ensemble persistent homology deviates significantly from a phase-
> randomized null distribution.  The product enables science cases —
> distant kilonovae, microlensing by sub-solar dark-matter substructure,
> low-amplitude long-period variables, pre-explosion CCSN precursors —
> that no per-source classifier can address because the signal is
> below the single-epoch detection threshold.
>
> **Deliverables.**
> 1. Open-source production pipeline (MIT-licensed, ~6k LOC, 85 tests).
> 2. Methodology paper validated on ZTF and DP1 (in preparation).
> 3. Daily-cadence flagged-ensemble VOEvent / Kafka stream.
> 4. Public dashboard with persistence-diagram time evolution.
>
> **Requested data rights:** Full operations-era PPDB read access
> for one PI account (Brian Sheppard) plus collaborators identified
> during the project.
>
> Full proposal and code: <https://github.com/bshepp/staring-into-the-void>

### Template D — arXiv announcement social post

> 🔭 New on arXiv: *Topological Recovery of Sub-Threshold Astrophysical
> Populations in LSST Forced Photometry*.
>
> The Vera C. Rubin Observatory will trigger ~10M alerts per night.
> The most interesting transients live *below* that threshold.  We
> apply persistent homology to ensembles of forced-photometry light
> curves and recover populations classifiers cannot see.
>
> Code (MIT): github.com/bshepp/staring-into-the-void
> Paper: arxiv.org/abs/[number]
> #LSST #astronomy #astrostats #topology

---

## Tracking

Keep a simple table in `docs/OUTREACH_LOG.md` (gitignored if you
prefer): date, person, channel, message sent, response, next step.
Cold outreach has a long tail; the only way to handle it is to
write the loop down.
