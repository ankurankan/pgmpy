# Causal Discovery Pipelines for pgmpy

## Contributors
- @ankurankan

## Introduction

pgmpy's causal discovery estimators (`PC`, `GES`, `HillClimbSearch`, `TOPIC`,
`ChowLiu`, `TAN`, `ExpertInLoop`) are sklearn-compatible: they inherit from
`sklearn.base.BaseEstimator`, expose `fit(X)` which sets `causal_graph_`, and
provide `score(X=..., true_graph=..., metric=...)`. They store constructor
arguments verbatim, so `get_params`/`set_params` already work — they can be
dropped into `sklearn.pipeline.Pipeline` and (with care) tuned. pgmpy also has a
genuinely strong **scoring** layer: discrete (`K2`, `BDeu`, `BDs`, `BIC`, `AIC`),
Gaussian (`BICGauss`, `AICGauss`, `LogLikelihoodGauss`), and conditional-Gaussian /
mixed (`BIC-CG`, `AIC-CG`, `LL-CG`, from Andrews, Ramsey & Cooper 2018), all wired
into `GES` and `HillClimbSearch` through a string registry with data-type
auto-detection (`pgmpy/structure_score/_base.py:72`).

This revision grounds the proposal in **what applied causal discovery actually
looks like** in the sciences. We surveyed the applied literature in systems
biology/genomics, climate/earth science, and (more thinly) neuroscience and
epidemiology, and found that real-world pipelines converge on a small set of
reusable patterns. Two of those dominant patterns are exactly the things pgmpy
*cannot* do today despite having all the surrounding machinery:

- **Interventional-data fusion** (biology): the canonical workflow learns a
  Bayesian network from a *mix of observational and interventional/perturbation
  data*, and the interventions are what make edges orientable. pgmpy ships the
  Sachs interventional datasets but has no way to tell a learner which rows are
  interventions, and no interventional score.
- **Time-series / autocorrelation-aware discovery** (climate, neuro, epi): the
  dominant pipeline (PCMCI/PCMCI+) is a constraint-based two-stage method built
  to control autocorrelation, which otherwise breaks the i.i.d. assumption behind
  every CI test. pgmpy's `DynamicBayesianNetwork` is inference-only — there is no
  DBN/time-series *structure learning* and no lagged or autocorrelation-aware CI
  test.

A third pattern — **bootstrap/ensemble edge-confidence** — is standard practice
across all these domains and is the empirically validated remedy for the known
weakness of heuristic-search BN methods (below). pgmpy has no bootstrap/ensemble
utility anywhere.

The rest of this document (1) reports the applied-domain evidence, (2) derives a
prioritized list of pipelines to build, and (3) specifies APIs and user journeys.
Every applied claim below was adversarially verified (3-0) against a primary,
peer-reviewed source; the *prioritization* itself is an interpretive synthesis
(flagged as such). Items carried over from the earlier model-selection research
that the workflow could not independently re-verify are marked **[unverified —
standard literature]**.

## Applied-domain evidence

### Systems biology / genomics — the score-based + interventional + bootstrap pattern

The canonical applied pipeline is **score-based Bayesian-network structure
learning that mixes observational and interventional data**, made robust by
**bootstrap aggregation**:

- **Sachs et al. 2005 (Science 308:523).** Learned a protein-signaling network
  from multiparameter single-cell flow cytometry, recovering most known
  relationships and predicting experimentally-verified novel ones. Critically,
  *interventions drove edge orientation* — observational data alone yielded only
  2 of 8 directed arcs (verified 3-0; bnlearn reproduces this with its modified
  BDe / `mbde` score). This is the single most-cited applied causal discovery
  result, and pgmpy ships its dataset (`SachsMixed`, `SachsContinuousJittered*`,
  all flagged `is_interventional=True`) — but cannot reproduce the workflow.

- **Interventional scoring improves identifiability** (verified 3-0). The
  interventional BGe score (IBGe; Kuipers & Moffa 2022, arXiv:2205.02602, now
  CLeaR/PMLR v275 2025) extends the Gaussian BGe score to a mixture of
  observational + interventional data with possibly unknown intervention targets;
  Hauser & Bühlmann (2012, JMLR 13:2409) proved interventional Markov equivalence
  (I-MEC) is strictly finer than observational MEC and gave the GIES search;
  Cooper & Yoo (1999, UAI) introduced interventional scoring. Genomics is the
  explicit motivating application.

- **Bootstrap/ensemble is standard, not optional** (verified 3-0). DREAM5
  (Marbach et al. 2012, Nature Methods 9:796): *no single GRN method is optimal
  across datasets*, and integrating predictions from multiple methods ("wisdom of
  crowds") gives robust performance matching or beating the best individual
  method — even with as few as ~5 methods, because their limitations cancel out.

- **A pitfall that lands directly on pgmpy** (verified 3-0). In that same DREAM5
  benchmark, *Bayesian-network methods underperformed*, attributed to their
  reliance on heuristic search being "too costly for systematic data resampling"
  and better suited to smaller networks. pgmpy's `HillClimbSearch`/`GES` are
  exactly these heuristic-search BN methods — so applied users should pair them
  with resampling/ensembling rather than trusting a single run. This is the
  strongest argument for shipping a bootstrap wrapper.

- **Time-series and single-cell GRN** add dynamic Bayesian networks (DBNs are the
  core BN-family method for time-course expression; verified 3-0), pseudo-temporal
  ordering of single-cell populations (Sanchez-Castillo et al. 2018,
  Bioinformatics 34:964, AR1MA1-VBEM), and dimensionality reduction for the
  underdetermined small-n/large-p regime (Godsey 2013, PLoS ONE, BACON: DBN +
  variational-Bayes Gaussian-mixture clustering).

### Climate / earth science — the PCMCI time-series pattern

The dominant pipeline is **PCMCI / PCMCI+**, a constraint-based two-stage method
designed for high-dimensional, autocorrelated, nonlinear time series (all verified
3-0):

- **PCMCI** (Runge et al. 2019, Science Advances 5:eaau4996; review in Nature
  Communications 10:2553, 2019): a PC-style *condition-selection* step finds the
  few relevant conditioning variables, then a **momentary conditional independence
  (MCI)** test conditions on lagged parents to control autocorrelation-induced
  false positives. Unlike Granger causality, it can orient contemporaneous links.

- **PCMCI+** (Runge 2020, UAI / PMLR v124) extends this to discover both lagged
  *and* contemporaneous links, outputs a time-series CPDAG, and is *robust to and
  even benefits from* strong autocorrelation — the very regime where other
  CI-based methods lose recall and inflate false positives.

- **Tigramite** (Runge, github.com/jakobrunge/tigramite) is the reference Python
  library, offering a family with explicit assumption/output contracts: PCMCI
  (stationary, no contemporaneous, no latents), PCMCI+ (adds contemporaneous,
  CPDAG), LPCMCI (latent confounders → time-series PAG; Gerhardus & Runge,
  NeurIPS 33, 2020). Overview: Runge et al. 2023, Nature Reviews Earth & Environment.

- **Lag-preserving bootstrap** (Debeire et al. 2024, CLeaR/PMLR v236,
  Bagged-PCMCI+): the time-series analogue of the GRN bootstrap. The key pitfall
  it encodes — naive time-step resampling *destroys lag structure*, so you must
  resample the realization/index set with replacement, run PCMCI+ on B=100
  bootstraps, and aggregate by edge-wise majority vote with confidences.

### Cross-cutting pitfalls the library should surface

- **Autocorrelation breaks i.i.d. CI tests** — the central reason PCMCI exists;
  applies to earth science, fMRI/EEG neuroscience, and longitudinal epidemiology.
- **Interventional data requires modified scores** — using perturbation rows as if
  observational mis-orients edges (Sachs: 2/8 arcs without intervention modeling).
- **Heuristic-search BN methods underperform unless ensembled** (DREAM5).
- **Faithfulness violations and batch effects** are common in biological networks;
  ensembling/edge-confidence is the practical mitigation.

### Scope caveats on this evidence

Neuroscience effective-connectivity specifics (Smith et al. 2011; Ramsey/Glymour
FASK) and epidemiology/economics specifics (tiered temporal background knowledge,
mixed-data confounding) were part of the search but **no domain-specific claim
survived verification** in this batch — so those patterns are inferred from the
shared autocorrelation/CI and prior-knowledge primitives rather than directly
evidenced. The DREAM5 "BN underperforms" finding is scoped to that 2012 benchmark
and should not be over-generalized to modern BN methods.

## Proposed Solution

Build a layered set of sklearn-compatible components, **prioritized by the applied
evidence above** and by fit to pgmpy's existing strengths (its scoring layer).
The recommendation is organized in three tiers:

**Tier 1 — highest value, strong fit to pgmpy's scoring strength, datasets already shipped:**
1. **Interventional-data score-based discovery** — an interventional score
   (interventional BDeu + a BGe/IBGe Gaussian score) plus a per-row intervention
   mask consumed by `HillClimbSearch`/`GES` (GIES lineage). Unlocks the canonical
   Sachs/GRN workflow.
2. **Bootstrap / ensemble edge-confidence** (`BootstrapDiscovery`) — resample,
   re-run any learner, return per-edge confidence + consensus graph. The
   empirically standard robustness pattern and the documented remedy for
   heuristic-BN underperformance. Works with every existing estimator unchanged.

**Tier 2 — high value, larger build, unlocks whole new domains:**
3. **Time-series causal discovery** — autocorrelation-aware (lagged/MCI) CI
   testing and a PCMCI-style two-stage pipeline producing a (dynamic) BN, plus a
   lag-preserving bootstrap. Unlocks climate/earth/neuro/epi.

**Tier 3 — supporting components (from the model-selection research, still valuable):**
4. **Unsupervised model selection** (`CausalTuningSearch`) — a `GridSearchCV`
   analog with an unsupervised criterion (naive CV is invalid for structure
   learning — see below).
5. **Preprocessing transformers** — discretization, nonparanormal/rank transform,
   missingness-aware handling (CI tests currently hard-reject `NaN`).
6. **Hybrid / staged discovery** — constraint-skeleton + score-orientation
   (MMHC-style); score + constraint-refinement (GFCI-style, blocked on FCI).
7. **Expert-knowledge as a pipeline stage** — thread `ExpertKnowledge` through
   pipelines/tuning; tiered temporal priors for epi/econ.

The unifying principle is unchanged: keep every component a standard sklearn
estimator/transformer, add new APIs alongside the existing ones (no breaking
changes), and only deviate from sklearn where its supervised assumptions break
(model selection, ensembling).

### Why naive `GridSearchCV` is not enough (the model-selection constraint)

You cannot simply wrap pgmpy estimators in `GridSearchCV`: causal discovery is
unsupervised, so there is no held-out label to score against.

> "Unfortunately, given that the problem is unsupervised, standard out-of-sample
> estimation methods used for supervised problems, such as cross-validation cannot
> be directly applied." — Biza, Tsamardinos & Triantafillou, PMLR v138 (2020). [verified 3-0]

> "PC is however unsupervised, so we cannot tune α using traditional
> cross-validation." — AutoPC, Pattern Recognition Letters (2021). [verified 3-0]

This is why Tier 3 model selection (Component 4) must use an unsupervised criterion
rather than a thin CV call.

## Alternative Solutions

**A. Document `sklearn.pipeline.Pipeline` usage only.** Cheapest, worth doing
regardless, but leaves every applied gap (interventional scoring, time-series,
ensembling) unaddressed and invites statistically invalid `GridSearchCV` use.

**B. Monolithic `CausalDiscoveryPipeline` class.** Easier to document but
unmodular — users can't swap a stage or compose with non-pgmpy steps. Rejected in
favor of small composable components.

**C. Depend on an external harness (Tigramite / benchpress / gCastle / CDT).**
Tigramite is the reference for the time-series pattern and we should mirror its
*assumption/output contract* design (each method declares what it assumes and what
graph type it returns). But adopting it as a dependency pulls in a different design
philosophy and heavy infra. Better: pgmpy-native, sklearn-shaped components with
lightweight interop.

**D. Build everything at once.** Rejected — Tier 1 delivers the most value per unit
effort (reuses the scoring layer, datasets already shipped) and should ship first;
Tier 2 (time-series) is a larger, separable build.

## Competitive positioning vs. Tigramite (the Tier-2 incumbent)

The Tier-2 time-series goal puts pgmpy up against **Tigramite** (Jakob Runge),
the established package for time-series causal discovery. This section states the
incumbent accurately (facts below verified 3-0 against primary sources — Tigramite
GitHub/docs/PyPI, the CauseMe site, and the Runge et al. papers) and defines a
realistic positioning, because the naive "pgmpy is uniquely end-to-end" pitch is
factually wrong.

### The incumbent, accurately

- **The benchmark turf belongs to Runge's group too.** CauseMe
  (causeme.uv.es; canonically causeme.net) is *the* community benchmark platform
  for time-series causal discovery — synthetic + real ground-truth datasets, an
  AUC-ranked standing leaderboard where developers upload causal-connection
  matrices. It was created by Runge, Muñoz-Marí & Camps-Valls (DLR + University of
  Valencia ISP group) and hosted the **NeurIPS 2019 "Causality 4 Climate" (C4C)**
  competition (Runge et al. 2020, PMLR v123). So "win the benchmark" is partly
  played on the incumbent's home field.

- **Tigramite (v5.2, latest 5.2.10.1, Jan 2026) is mature and broad.** It
  implements exactly five discovery methods: PCMCI, PCMCI+ (time-series CPDAG),
  LPCMCI (latent confounders → time-series PAG), RPCMCI (regime-dependent), and
  J-PCMCI+ (joint discovery across multiple datasets). Its CI-test suite spans
  continuous, discrete, and mixed data: ParCorr, RobustParCorr, ParCorrWLS, GPDC
  (+ GPDCtorch), CMIknn, CMIknnMixed, CMIsymb, Gsquared, RegressionCI, ParCorrMult.
  pgmpy has *none* of these nonlinear/mixed time-series CI tests today.

- **Tigramite is NOT discovery-only.** It ships a `CausalEffects` class
  (non-parametric conditional causal-effect estimation via generalized backdoor
  adjustment, with hidden-variable and Wright-path-coefficient support), a
  `LinearMediation` class (time-series mediation with bootstrap CIs), and a
  `Prediction` class (sklearn-based forecasting with causal feature selection),
  each with dedicated tutorials. **Any pitch that "pgmpy is end-to-end and
  Tigramite just returns a graph" is false and should not appear in the docs.**

### Where pgmpy's differentiation actually is (narrower, defensible)

Tigramite's downstream layer is **task-specific to time-series SCMs**: backdoor
adjustment, Wright path coefficients, linear mediation, sklearn forecasting. It is
*not* a general probabilistic-inference engine. pgmpy's genuine, defensible edge is
exactly there:

- **General PGM inference breadth** — exact inference (variable elimination,
  belief propagation) for *arbitrary* joint/conditional queries, over discrete,
  Gaussian, and hybrid models, plus parameter learning — not only adjustment-based
  effect estimates on a linear time-series graph.
- **Identification across model types** — do-calculus identification
  (`CausalInference`), frontdoor/adjustment-set discovery, etc., rather than a
  single backdoor estimator.
- **A discovered time-series graph becomes a full pgmpy model** you can run any of
  the above on, in one ecosystem, with the sklearn-compatible API and a much larger
  general (non-climate) user base.

So the honest positioning is **not** "uniquely end-to-end" — it is "the
**general-purpose probabilistic-modeling backend** for time-series causal graphs,
versus Tigramite's specialized time-series effect/forecast layer."

### Revised strategy (correcting the earlier read)

1. **Parity + prove it on CauseMe**, while recognizing it is Runge's home field —
   credibility, not displacement, is the goal there.
2. **Differentiate on general inference/identification breadth**, not on
   "Tigramite can't do downstream" (it can).
3. **Close the CI-test gap** (nonlinear/mixed: KCI/RCoT/CMIknn-style) — a hard
   requirement to be credible on climate-grade nonlinear data; load-bearing for
   pgmpy's general CD quality too.
4. **Interop over frontal assault** — a Tigramite-graph → pgmpy-DBN converter; its
   value is bringing a PCMCI graph into pgmpy's *general* inference/identification,
   not "effects they couldn't get" (they can).
5. **Win the flanks** — neuroscience, epidemiology, econometrics,
   time-course genomics — where Tigramite is less entrenched and pgmpy's
   general-PGM strengths matter more. (See caveat below: the competitive map for
   these domains was only partially verified.)

### Competing packages (other paradigms / domains)

From primary package docs (fetched, **not** independently vote-verified in this
batch — treat as directional):
- **lingam** (Shimizu group) — VAR-LiNGAM for time-series LiNGAM discovery.
- **causalnex** (QuantumBlack/McKinsey) — DYNOTEARS (dynamic NOTEARS).
- **tsFCI / SVAR-FCI / SVAR-GFCI** (Entner & Hoyer; Malinsky & Spirtes 2018,
  PMLR v92) — time-series discovery with latent confounders, used in econometrics.
- **causal-learn** (CMU/Tetrad lineage) — LiNGAM/VAR-LiNGAM-family support.
- **Neuroscience** (fMRI/EEG effective connectivity) — historically Granger/VAR and
  DCM, with Tetrad-lineage methods (FASK/Two-step); not vote-verified here.

The takeaway: time-series causal discovery is *not* a Tigramite monopoly across
paradigms — it owns the **CI-based** corner (especially climate). pgmpy entering as
a general-PGM backend that interoperates and covers the flanks is a more winnable
posture than a head-on PCMCI reimplementation race.

## Details of proposed solution

### Component 1 (Tier 1) — Interventional-data scoring + GIES-style search

The mechanism behind Sachs 2005 and IBGe: let score-based learners consume a mix
of observational and interventional rows, where an intervention on a node removes
that node's dependence on its parents (its local score contribution is dropped or
modified for the affected rows). Concretely:

- Add an **interventional variant** of the existing scores. `BDeu`/`K2` →
  interventional BDeu (Cooper & Yoo 1999); add a **BGe** Gaussian score and its
  interventional extension **IBGe** (Kuipers & Moffa 2022). pgmpy currently has
  *no BGe* — only `BICGauss`/`AICGauss`/`LL-Gauss` — so BGe is a worthwhile
  addition on its own.
- Accept a **per-row intervention specification** (which variable, if any, was
  intervened on in each row) via a `fit` argument, and have the score skip/modify
  the local term for intervened nodes. This is the GIES idea (Hauser & Bühlmann
  2012) without necessarily reimplementing the full GIES search — `HillClimbSearch`
  can search the I-MEC with an interventional score.

```python
class InterventionalBDeu(BaseStructureScore): ...
class BGe(BaseStructureScore): ...          # Gaussian BGe (new; pgmpy lacks it)
class InterventionalBGe(BGe): ...           # IBGe (Kuipers & Moffa 2022)

# usage: intervention_targets[i] = the node intervened on in row i (or None)
hc = HillClimbSearch(scoring_method="ibge")
hc.fit(sachs_data, intervention_targets=targets)   # new fit kwarg
```

**Why highest priority:** it is the load-bearing mechanism of the most-cited
applied pipeline; it is the *best fit to pgmpy's existing strength* (it extends the
score layer rather than adding new infra); and the datasets to demo it
(`SachsMixed` et al., already `is_interventional=True`) ship with pgmpy today.
Feasibility is high — `HillClimbSearch`/`GES` already accept arbitrary
`BaseStructureScore` objects.

### Component 2 (Tier 1) — Bootstrap / ensemble edge confidence: `BootstrapDiscovery`

Resample B times, re-run any base estimator, aggregate to per-edge confidence
(edge intensity) plus a thresholded consensus graph. Backed by both the foundational
BN-bootstrap work (Friedman, Goldszmidt & Wyner, UAI 1999 [verified 3-0]; Imoto et
al. 2002 edge intensity `(t1+t2)/T` [verified 2-0]) and the applied standard-of-
practice (DREAM5 wisdom-of-crowds, Marbach et al. 2012 [verified 3-0]).

```python
class BootstrapDiscovery(BaseEstimator):
    """Bootstrap aggregation of any pgmpy discovery estimator.

    Returns per-edge confidence (edge intensity) and a thresholded consensus DAG.
    A lag-preserving resampling mode supports time-series / DBN base learners
    (Debeire et al. 2024): resample the realization index, not individual steps.
    """
    def __init__(self, estimator, n_bootstrap=200, threshold=0.5,
                 selection="intensity", time_series=False,
                 random_state=None, n_jobs=-1): ...
    def fit(self, X, y=None):
        # self.edge_strengths_ : DataFrame[(u, v) -> directed_freq, any_dir_freq]
        # self.causal_graph_   : edges with any-direction freq >= threshold
        return self
```

**Library precedent:** bnlearn's `boot.strength`/`arc.strength` (Scutari) — among
its most-used functions; pgmpy has no equivalent. **Why Tier 1:** highest
value-per-effort — works with *all* existing learners unchanged, reuses pgmpy's
graph objects, and is the documented fix for the DREAM5 weakness. The
`time_series=True` lag-preserving mode is the one applied-specific subtlety to get
right (naive step resampling destroys lag relationships).

### Component 3 (Tier 2) — Time-series causal discovery

Two pieces, both currently absent:

1. **Autocorrelation-aware CI testing.** Add lagged-variable CI tests and a
   momentary-conditional-independence (MCI)-style test that conditions on lagged
   parents (Runge et al. 2019). Without this, applying pgmpy's static CI tests to
   autocorrelated series produces inflated false positives — the central
   earth/neuro/epi pitfall.
2. **A PCMCI-style two-stage pipeline** on top of the existing `PC` machinery and
   `DynamicBayesianNetwork` representation: condition-selection (PC step) → MCI
   test → output a time-series/dynamic BN (CPDAG over lagged + contemporaneous
   links). Mirror Tigramite's contract: declare assumptions (stationarity,
   sufficiency) and output graph type explicitly.

```python
class TimeSeriesPC(BaseCausalDiscovery):
    """PCMCI-style lagged + contemporaneous causal discovery.

    tau_max : maximum time lag to consider.
    ci_test : autocorrelation-aware CI test (lagged conditioning).
    """
    def __init__(self, tau_max=5, ci_test="par_corr_mci",
                 significance_level=0.01, contemporaneous=True): ...
    def fit(self, X, y=None):
        # X: time-indexed DataFrame
        # self.causal_graph_: DynamicBayesianNetwork (lagged + contemporaneous)
        return self
```

**Why Tier 2:** unlocks entire domains (climate, neuroscience, longitudinal
epidemiology) pgmpy currently can't serve at all, and pgmpy already has the `PC`
algorithm and `DynamicBayesianNetwork` representation to build on. But it is a
larger build than Tier 1 (new CI tests + windowing/lag construction + DBN output),
and a full latent-variable version (LPCMCI) depends on FCI, which pgmpy lacks. A
pragmatic first step: a `make_lagged` transformer + `TimeSeriesPC` for the
causally-sufficient stationary case, paired with the lag-preserving
`BootstrapDiscovery`.

### Component 4 (Tier 3) — Unsupervised model selection: `CausalTuningSearch`

`GridSearchCV`-shaped wrapper with an unsupervised criterion. Three literature-
supported options (all verified 3-0):

- **Out-of-sample Markov-blanket prediction (OCT)** — default. Score a candidate
  graph by held-out predictive accuracy of each node given its Markov blanket
  (Biza et al. 2020, PMLR v138). Converts the unsupervised problem into a
  supervised one pgmpy can run with ordinary K-fold.
- **Out-of-sample structure score (OTSL)** — for score-based learners; held-out
  BIC/BDeu via pgmpy's existing `StructureScore` (Chobtham & Constantinou 2023,
  arXiv:2306.13932).
- **AutoPC two-run stability** — a tailored α-selector for PC reusing pgmpy's graph
  metrics (normalized SHD/F1/MCC) as the stability metric (Pattern Recognition
  Letters 2021).

**Non-recommendation:** don't default to StARS for algorithm/hyperparameter
selection — it tunes graphical-lasso λ well but doesn't transfer to BN tuning
(Biza et al. 2020, over 48 continuous / 54 discrete configs). [verified 3-0]

```python
class CausalTuningSearch(BaseEstimator):
    def __init__(self, estimator, param_grid, criterion="oct", cv=5, n_jobs=-1): ...
    def fit(self, X, y=None):
        # criterion in {"oct", "structure_score", "auto_pc_stability"}
        # sets best_params_, best_estimator_, causal_graph_
        return self
```

### Component 5 (Tier 3) — Preprocessing transformers (`pgmpy/preprocessing/`)

sklearn `TransformerMixin` classes (none exist today):

- `Discretizer` (quantile/uniform/k-means; wraps `KBinsDiscretizer`, preserves
  column names). **Caveat:** binning continuous data is lossy and can change the
  recovered structure — offer it, but don't make it the default for continuous data.
- `NonparanormalTransform` (rank-based Gaussianization; Liu, Lafferty & Wasserman,
  JMLR 2009 **[unverified — standard literature]**) — the safer default for
  continuous-but-non-Gaussian data, keeping Pearson/Fisher-Z valid without binning.
- `TestwiseDeletionImputer` — missingness-aware passthrough so downstream CI tests
  drop rows per-test rather than imputing. **Caveat:** mean/regression imputation
  injects artificial dependence that biases CI tests; the literature's answer is
  missingness-aware discovery (test-wise deletion + MVPC; Tu et al., AISTATS 2019
  **[unverified — standard literature]**), not `SimpleImputer`. pgmpy's CI tests
  currently hard-reject `NaN` (`allow_nan=False`, `ensure_all_finite=True`).

### Component 6 (Tier 3) — Hybrid / staged discovery

- **MMHC-style** (constraint skeleton + score orientation; Tsamardinos, Brown &
  Aliferis, Machine Learning 2006 **[unverified — standard literature]**). pgmpy
  has both halves (`PC` skeleton + `HillClimbSearch`) — mostly orchestration.
- **GFCI-style** (score discovery + FCI refinement for latent confounders;
  Ogarrio, Spirtes & Ramsey, PGM 2016 **[unverified — standard literature]**).
  Deferred — depends on FCI, which pgmpy does not yet have.

### Component 7 (Tier 3) — Expert knowledge as a pipeline stage

Make `ExpertKnowledge` (`forbidden_edges`, `required_edges`, `temporal_order`,
`search_space`) threadable through `Pipeline`/`CausalTuningSearch` rather than only
a constructor arg. Tiered `temporal_order` is the natural API for the epi/econ
"background knowledge by time tier" pattern. `BootstrapDiscovery` should also be
able to *emit* expert knowledge (high-confidence edges → `required_edges`), closing
the Friedman et al. (1999) loop: "use these confidence measures to induce better
structures ... and to detect the presence of latent variables." [verified 3-0]

### Mapping to pgmpy's existing surface

| Need | Already in pgmpy | Gap to fill |
|------|------------------|-------------|
| Estimators sklearn-compatible | all 7 (`get_params`/`set_params` clean) | — |
| Structure scores (discrete/Gaussian/cond-Gaussian) | K2, BDeu, BDs, BIC, AIC + Gauss + CG | **interventional variants; BGe/IBGe** |
| Score-based search | `GES`, `HillClimbSearch` (full score registry) | accept per-row intervention mask |
| Per-graph metrics for selection/ensembling | `StructureScore`, `FisherC`, `ImpliedCIs`, `SHD`, `CorrelationScore` | wire into tuning/ensemble |
| Skeleton + orientation primitives | `_ConstraintMixin`, `_ScoreMixin` | compose into hybrid estimators |
| Expert knowledge | `ExpertKnowledge` (PC/HC/ExpertInLoop) | make pipeline-threadable; emit from bootstrap |
| Interventional datasets | `SachsMixed`, `SachsContinuousJittered*` (`is_interventional`) | learner can't consume the flag |
| Dynamic BN | `DynamicBayesianNetwork` (inference only) | **DBN structure learning** |
| Bootstrap / ensemble / edge confidence | none | **build** (`BootstrapDiscovery`) |
| Unsupervised model selection | none | **build** (`CausalTuningSearch`) |
| Time-series / lagged CI / PCMCI | none | **build** (`TimeSeriesPC` + lagged CI) |
| Preprocessing (discretize/rank/impute) | none | **build** |
| Missing-data CI handling | none (`allow_nan=False`) | **build** (test-wise deletion / MVPC) |
| FCI / latent-variable discovery | none | future (unblocks GFCI, LPCMCI) |

## User journeys with the solution

**Journey 1 (Tier 1) — Reproduce the Sachs GRN workflow with interventions.**
```python
from pgmpy.causal_discovery import HillClimbSearch, BootstrapDiscovery
from pgmpy.datasets import load_dataset

ds = load_dataset("sachs_mixed")          # ships interventional flags
hc = HillClimbSearch(scoring_method="ibge")
boot = BootstrapDiscovery(estimator=hc, n_bootstrap=200, threshold=0.6)
boot.fit(ds.data, intervention_targets=ds.intervention_targets)
boot.edge_strengths_      # per-edge confidence ("wisdom of crowds")
network = boot.causal_graph_
```
Interventions orient edges (Sachs: 2/8 → most arcs); bootstrap gives the
robustness DREAM5 showed BN methods need.

**Journey 2 (Tier 2) — Climate teleconnections from time series.**
```python
from pgmpy.preprocessing import make_lagged
from pgmpy.causal_discovery import TimeSeriesPC, BootstrapDiscovery

tsp = TimeSeriesPC(tau_max=6, ci_test="par_corr_mci", contemporaneous=True)
boot = BootstrapDiscovery(estimator=tsp, n_bootstrap=100, time_series=True)
boot.fit(climate_series)          # lag-preserving bootstrap (Debeire 2024)
dbn = boot.causal_graph_          # DynamicBayesianNetwork: lagged + contemporaneous
```
Autocorrelation-aware CI (MCI) avoids the inflated false positives static tests
produce on autocorrelated data.

**Journey 3 (Tier 1) — Defensible edges under small n.**
```python
from pgmpy.causal_discovery import GES, BootstrapDiscovery
boot = BootstrapDiscovery(estimator=GES(), n_bootstrap=200, threshold=0.6)
boot.fit(data)
consensus = boot.causal_graph_    # edges present in >=60% of resamples
```

**Journey 4 (Tier 3) — "Which algorithm/settings?" without labels.**
```python
from pgmpy.causal_discovery import PC, CausalTuningSearch
search = CausalTuningSearch(
    estimator=PC(),
    param_grid={"significance_level": [0.01, 0.05], "ci_test": ["chi_square", "g_sq"]},
    criterion="oct", cv=5,        # out-of-sample Markov-blanket prediction
)
search.fit(data)
print(search.best_params_)        # chosen with a *valid* unsupervised score
```

**Journey 5 (Tier 3) — Continuous, non-Gaussian data without discretizing.**
```python
from sklearn.pipeline import Pipeline
from pgmpy.preprocessing import NonparanormalTransform
from pgmpy.causal_discovery import PC
pipe = Pipeline([("gaussianize", NonparanormalTransform()),
                 ("pc", PC(ci_test="pearsonr", return_type="cpdag"))])
pipe.fit(data)
```

## Recommended build order

1. **Tier 1a — `BootstrapDiscovery`** (incl. lag-preserving mode). Highest
   value-per-effort: works with every existing learner, reuses graph objects,
   directly addresses the DREAM5 weakness. Build first.
2. **Tier 1b — Interventional scoring + BGe/IBGe + per-row intervention mask.**
   Best fit to pgmpy's scoring strength; demo datasets already ship. Build alongside 1a.
3. **Tier 3 — `CausalTuningSearch` (OCT default)** and **expert-knowledge
   plumbing.** Medium effort; complements 1a/1b.
4. **Tier 3 — preprocessing transformers** (nonparanormal, discretizer,
   test-wise deletion).
5. **Tier 2 — time-series discovery** (`make_lagged` + lagged/MCI CI tests +
   `TimeSeriesPC`). Larger, separable build; unlocks climate/neuro/epi.
6. **Future — FCI / latent-variable discovery**, which unblocks both GFCI-style
   hybrids and LPCMCI-style latent time-series discovery.

## References

**Applied-domain evidence (adversarially verified 3-0 against primary sources):**
- Sachs, Perez, Pe'er, Lauffenburger & Nolan. *Causal Protein-Signaling Networks
  Derived from Multiparameter Single-Cell Data.* Science 308(5721):523–529, 2005.
- Kuipers & Moffa. *The Interventional Bayesian Gaussian Equivalent Score (IBGe).*
  arXiv:2205.02602, 2022; CLeaR / PMLR v275, 2025.
- Hauser & Bühlmann. *Characterization and Greedy Learning of Interventional
  Markov Equivalence Classes of DAGs (GIES / I-MEC).* JMLR 13:2409–2464, 2012.
- Cooper & Yoo. *Causal Discovery from a Mixture of Experimental and Observational
  Data.* UAI 1999.
- Marbach, Costello, Küffner, … Stolovitzky. *Wisdom of Crowds for Robust Gene
  Network Inference (DREAM5).* Nature Methods 9(8):796–804, 2012.
- Sima, Hua & Jung. *Inference of Gene Regulatory Networks Using Time-Series Data:
  A Survey.* Current Genomics, 2009.
- Sanchez-Castillo et al. *A Bayesian Framework for the Inference of GRNs from Time
  and Pseudo-Time Series Data (AR1MA1-VBEM).* Bioinformatics 34(6):964, 2018.
- Godsey. *Integrated Bayesian Clustering and Dynamic Modeling of Time-Course
  Expression (BACON).* PLoS ONE, 2013.
- Runge et al. *Inferring Causation from Time Series in Earth System Sciences.*
  Nature Communications 10:2553, 2019.
- Runge, Nowack, Kretschmer, Flaxman & Sejdinovic. *Detecting and Quantifying
  Causal Associations in Large Nonlinear Time Series Datasets (PCMCI).* Science
  Advances 5(11):eaau4996, 2019.
- Runge. *Discovering Contemporaneous and Lagged Causal Relations in Autocorrelated
  Nonlinear Time Series (PCMCI+).* UAI 2020 / PMLR v124.
- Gerhardus & Runge. *High-Recall Causal Discovery for Autocorrelated Time Series
  with Latent Confounders (LPCMCI).* NeurIPS 33, 2020.
- Runge et al. *Causal Inference for Time Series.* Nature Reviews Earth &
  Environment, 2023. — Tigramite library: github.com/jakobrunge/tigramite.
- Debeire, Gerhardus, Runge & Eyring. *Bootstrap Aggregation and Confidence
  Measures to Improve Time Series Causal Discovery (Bagged-PCMCI+).* CLeaR / PMLR
  v236, 2024.
- Nowack et al. *Causal Networks for Climate Model Evaluation and Constrained
  Projections.* Nature Communications 11, 2020.

**Competitive-landscape facts (verified 3-0 against primary sources):**
- Runge, Tibau, Bruhns, Muñoz-Marí & Camps-Valls. *The Causality for Climate
  Competition (C4C).* PMLR v123 (NeurIPS 2019 Competition Track), 2020 — CauseMe
  platform; created by Muñoz-Marí, Camps-Valls & Runge (causeme.uv.es / causeme.net).
- Tigramite (Jakob Runge), v5.2.x — github.com/jakobrunge/tigramite,
  jakobrunge.github.io/tigramite. Methods: PCMCI, PCMCI+, LPCMCI, RPCMCI, J-PCMCI+.
  Downstream: `CausalEffects`, `LinearMediation`, `Prediction`. (Verified Tigramite
  is *not* discovery-only.)
- *Competing packages* (primary docs, not independently re-verified): lingam
  (VAR-LiNGAM); causalnex/DYNOTEARS (Zheng et al. 2020, AISTATS); tsFCI (Entner &
  Hoyer 2010) and SVAR-FCI/GFCI (Malinsky & Spirtes 2018, PMLR v92); causal-learn.

**Model-selection / bootstrap foundations (verified 3-0 in earlier research):**
- Biza, Tsamardinos & Triantafillou. *Tuning Causal Discovery Algorithms (OCT).*
  PMLR v138 (PGM 2020).
- Chobtham & Constantinou. *Hyperparameter Tuning and Model Evaluation in Causal
  Structure Learning (OTSL).* arXiv:2306.13932, 2023.
- Liu, Roeder & Wasserman. *Stability Approach to Regularization Selection (StARS).*
  NeurIPS 2010, arXiv:1006.3316.
- *AutoPC.* Pattern Recognition Letters, 2021, doi:10.1016/j.patrec.2021.09.009.
- Friedman, Goldszmidt & Wyner. *Data Analysis with Bayesian Networks: A Bootstrap
  Approach.* UAI 1999.
- Imoto et al. *Bootstrap Analysis of Gene Networks Based on Bayesian Networks and
  Nonparametric Regression.* 2002 — edge intensity (t1+t2)/T.

**Standard literature cited but not independently re-verified by the workflow
(well-established; confirm venue/details before final write-up):**
- Liu, Lafferty & Wasserman. *The Nonparanormal.* JMLR 2009.
- Tu, Zhang, Ackermann, Glymour, et al. *Causal Discovery in the Presence of
  Missing Data (MVPC).* AISTATS 2019, PMLR v89.
- Meinshausen & Bühlmann. *Stability Selection.* JRSS-B 72(4), 2010.
- Tsamardinos, Brown & Aliferis. *The Max-Min Hill-Climbing (MMHC) Algorithm.*
  Machine Learning 65(1), 2006.
- Ogarrio, Spirtes & Ramsey. *A Hybrid Causal Search Algorithm for Latent Variable
  Models (GFCI).* PGM 2016.
- Andrews, Ramsey & Cooper. *Scoring Bayesian Networks of Mixed Variables
  (conditional-Gaussian scores).* 2018 — already implemented in pgmpy.
- Kalainathan & Goudet. *Causal Discovery Toolbox.* JMLR 21:19-187 (library precedent).
- Scutari, *bnlearn* `boot.strength` / `arc.strength` (library precedent).
- benchpress, gCastle (library precedents for benchmarking/pipeline abstractions).

## Open questions

- **Integration depth for interventional learning:** full GIES-style search
  (Hauser & Bühlmann 2012) natively, or a lighter per-node intervention mask that
  existing scores consume? The evidence establishes the value, not the preferred depth.
- **Edge-confidence surfacing across heterogeneous learners** (PC, GES,
  HillClimbSearch, ChowLiu/TAN, DBN) and the exact lag-preserving resampling scheme
  for DBN/time-series to avoid the naive-time-step pitfall.
- **Neuroscience and epi/econ patterns** were under-evidenced in this batch — do
  fMRI/EEG effective-connectivity methods (FASK/Two-step) reduce to the same
  autocorrelation-aware-CI + contemporaneous-orientation primitives as PCMCI+, or
  need domain-specific (e.g. non-Gaussian) assumptions? What is the precise tiered-
  background-knowledge API for epidemiology/economics?
- **Competitive map for the flanks needs confirmation.** The Tigramite/CauseMe
  facts are verified, but the claim that pgmpy can "win the flanks" (neuro/epi/econ)
  rests on a competing-package map that was only partially verified — which tools
  are actually entrenched in those domains, and is there a real opening for a
  general-PGM entrant? Confirm before committing strategy.
