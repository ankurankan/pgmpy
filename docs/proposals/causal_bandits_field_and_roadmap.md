# Causal Bandits: Field Overview, Practical Applications, and a pgmpy Roadmap

**Purpose.** This report does three things, in order:

1. **Part 1 — Field overview.** A distilled, readable map of the causal-bandits field: the core
   problem, the two objectives, the design axes, the main algorithmic families, the theoretical
   landscape, and the 2025–2026 frontier. (For equation-level, per-paper depth see the companion
   `causal_bandits_literature_review.md`; this part is the synthesis, not a re-derivation.)
2. **Part 2 — Practical applications.** Where these methods are actually used or seriously
   proposed, across four lenses: domain applications in the literature, real deployed/industry
   systems, pgmpy-aligned domains, and benchmarks/testbeds — with an honest maturity assessment.
3. **Part 3 — pgmpy roadmap.** A concrete, prioritized list of features and algorithms to build,
   grounded in what pgmpy already has, phased by value and effort.

**Relationship to prior work in this repo.** Two full literature reviews already exist:
`causal_bandits_literature_review.md` (22 papers, equation-level) and
`causal_incentives_literature_review.md` (Causal Influence Diagrams / agent incentives). The key
bridge established there — *a causal bandit is online intervention-selection on a causal influence
diagram* — drives the Part-3 recommendation that bandits and CIDs share one causal-decision core.

---

## Executive summary (TL;DR)

- **What they are.** Multi-armed bandits where arms are interventions `do(X=x)` on a causal graph and
  the reward is a target node. The field's defining idea: *graphical structure, not the number of
  arms, governs difficulty* — a graph-aware learner shares information across arms (one pull informs
  many) and provably beats a structure-blind bandit.
- **State of the field.** Mature theory (~25 papers, 2015→2026): information-leakage estimators →
  **POMIS-style structural arm reduction** → confounding & partial-ID → parametric scaling that beats
  the `2^N` arm explosion → contextual/budgeted → unknown-graph discovery. The 2025–26 frontier is
  **unknown/partial graphs + transportability + Causal Bayesian Optimization**.
- **Applications — the honest picture.** Almost no *deployed* causal bandits. But the *same loop* is
  actively practiced in **biology / experimental design** (gene-perturbation selection on real data,
  including pgmpy's **Sachs** network) under "active learning / Bayesian optimization" names; and
  production bandits are **contextual, not causal**, with confounding / off-policy / transfer being the
  recognized gap causal bandits fill. **The software niche is open** — no maintained general
  causal-bandit library exists (only an early-stage `causalrl`; every major causal library lacks an
  online decision layer).
- **Why pgmpy is well-positioned.** It already owns the *static* half — typed graph, do-operator,
  exact/approx inference, backdoor/frontdoor/IV identification, ADMG/MAG + c-components, node roles,
  and LinearGaussian/Functional SEM models with ancestral simulation. A causal-bandits module is a
  **thin sequential layer** over this, plus two graph algorithms (POMIS, latent projection) and a
  richer soft-intervention API.
- **Top recommendations (detail in Part 3).**
  1. **v1 = Tiers 1+2**: a `CausalEnvironment` + interaction loop + `UCB`/`kl-UCB`/`Thompson` +
     reward-parent-decomposition estimator + regret metrics (reproduces C-UCB/C-TS, MABUC, Lattimore),
     **plus POMIS/POMPS structural arm reduction** — the differentiator no general bandit library has.
  2. Build on a **shared `causal_decision` core** with the CID/incentives work (POMIS = value of
     control; partial-ID bounds shared), but ship standalone so it is not blocked on a full CID subsystem.
  3. Make **unknown-graph handling, transportability, and a CBO-style continuous-reward path**
     first-class — that is where both the field and the applications are heading.
  4. First **showcase = the biology experimental-design loop on the shipped Sachs dataset**; reuse
     `sanghack81/SCMMAB-NIPS2018` as a POMIS correctness oracle and treat `pycid` (already on pgmpy) as
     a neighbour, not a competitor.

---

## Part 1 — Field Overview

### 1.1 The problem, in one paragraph

A **causal bandit** is a multi-armed bandit in which the **arms are interventions**
`do(X = x)` on the variables of a causal graph `G`, and the **reward is a designated target node**
`Y` (a descendant of the intervened variables). The learner knows the graph — sometimes only
partially, or not at all — but not the conditional distributions. The defining feature, absent from
classical bandits, is **information leakage**: pulling one arm also reveals the values of *other*
graph variables, so a single interventional sample carries information about the expected reward of
*many* arms at once. A structure-blind bandit treats each of the (often exponentially many) arms as
opaque and pays for it; a causal bandit exploits the graph factorization to share statistical
strength across arms. The single most important idea in the field follows from this:

> **Graphical structure, not the raw number of arms, governs the difficulty of a causal decision
> problem.**

Every major result is, at bottom, a formalization of *which* structural quantity replaces the arm
count in the regret bound.

### 1.2 Two objectives, two modes

- **Simple regret** `R_T = μ* − E[μ(â_T)]` — the quality of the *single arm recommended* after a
  budget of `T` rounds. This is **best-arm identification / pure exploration**: explore freely, then
  commit. (Lattimore et al. 2016; Sen et al. 2017; Yabe et al. 2018.)
- **Cumulative regret** `R_T = T·μ* − Σ_t μ(a_t)` — total reward *lost while learning*. This is the
  **online** setting where every exploratory pull has an opportunity cost. (Lu et al. 2020;
  Lee & Bareinboim 2018; the linear-SEM and combinatorial lines.)

Orthogonally, the field splits by **learning mode** — and the 2025 ACM survey makes this its
*primary* taxonomic axis:

- **Best-arm / single-best-intervention:** find one optimal `do(X = x)`. The large majority of the
  literature.
- **Policy learning (contextual):** learn a *map* from observed context to intervention,
  `π : context → do`. A far sparser, harder, and more open quadrant — essentially one prior work
  (Subramanian & Ravindran 2022) plus the "mixed policy" characterization (Lee & Bareinboim 2020).

### 1.3 The design axes (the taxonomy that is also an API surface)

The field is best understood as a product of a handful of independent axes. These axes *are* the
knobs a library must expose:

| Axis | Values (easy → hard) |
|---|---|
| **Graph knowledge** | fully known `G` → known skeleton/CPDAG → partial → fully unknown (discover online) |
| **Intervention type** | atomic/hard `do(X=x)` → soft/stochastic (swap a CPD) → non-atomic (fix a subset) → combinatorial (≤K nodes) → generalized/continuous (reshape a mechanism) → targeted (do on a chosen sub-population) |
| **Confounding** | none (Markovian) → unobserved confounders (semi-Markovian / ADMG, bidirected edges) |
| **Cost / budget** | unit cost → non-uniform intervention cost `c_a`, fixed budget `B`, observation cheaper than intervention |
| **Context** | non-contextual → contextual (context as causal side-information) |
| **Model / payoff family** | tabular CBN → linear SEM → BGLM (monotone-link binary) → general Lipschitz SCM |
| **Objective** | simple regret (pure exploration) vs. cumulative regret (online) |

### 1.4 The main algorithmic families

The ~22 core papers cluster into six families. Read this as the field's algorithm menu.

**(A) Foundations & information-leakage estimators (known graph).**
The conceptual origin is **MABUC** (Bareinboim, Forney & Pearl 2015): when a latent confounder
drives *both* the agent's natural action and the reward, maximizing the interventional
`E[Y | do(x)]` is provably insufficient — the agent must optimize the counterfactual
**effect-of-treatment-on-the-treated** `E[Y_{X=a} | X=x]`, blending observational and experimental
data. Its algorithm, **Causal Thompson Sampling**, seeds arm estimates from observational data and
uses "intention-specific randomization." **Lattimore, Lattimore & Reid (2016)** then formalized the
general known-graph problem and gave the first structure-dependent bound: simple regret
`O(√(m(q)/T))`, where the difficulty constant `2 ≤ m(q) ≤ N` counts "rare" variables and can be far
below the arm count — via a **truncated importance-weighted estimator** that reweights a shared
sample pool `R_a = P(Pa_Y | a) / Q(Pa_Y)` to score every arm. **Sen et al. (2017, SRIS)** extended
this to *soft* interventions at a source node with successive-elimination + clipped importance
sampling. **Yabe et al. (2018)** generalized to arbitrary *non-atomic* interventions via
"propagating inference" — estimate native CPD parameters, propagate marginals through topological
order, and score every overlapping arm from one sample set.

**(B) Where to intervene — structural arm reduction.**
This is the field's most distinctive contribution and its deepest tie to causal theory.
**Lee & Bareinboim (2018, SCM-MAB)** proved that only a graphically identifiable subset of
intervention sets can *ever* be optimal — the **POMIS** (possibly-optimal minimal intervention
sets) — so a solver should restrict its arms to POMIS arms. The machinery is pure graph theory:
**MIS** (`X ⊆ an(Y)` after cutting edges into `X`), **UC-territory / MUCT** (a descendant- and
c-component-closed set containing `Y`), and the **interventional border (IB)** (the MUCT's parents
outside it), with the sound-and-complete characterization *`X` is a POMIS iff `IB(G_\overline{X}, Y)
= X`*. Running kl-UCB over the (much smaller) POMIS arm set provably lowers regret.
**Lee & Bareinboim (2019)** handled **non-manipulable variables** (POMIS of the latent projection);
**Lee & Bareinboim (2020, POMPS)** generalized from "where to intervene" to *mixed policies* —
jointly choosing what to **intervene on** and what to **observe** — proving the standard
"intervene-and-observe-everything" contextual-bandit scope can be strictly suboptimal.
**Elahi, Ghasemi & Kocaoglu (2024)** proved that under latent confounding you need only *partial*
structure discovery — the induced subgraph on `an(Y)` plus the confounders incident to it — to
recover all POMISs, giving a polynomial-sample discover-then-UCB algorithm.

**(C) Known-graph online index methods.**
**Lu et al. (2020, C-UCB / C-TS)** gave the workhorse: with the graph and the parent-conditionals
`P(Pa_Y | a)` known, maintain UCB/Thompson indices on the **reward-parent decomposition**
`μ_a = Σ_z μ_z · P(Pa_Y = z | a)`, so regret scales with the number of parent-configurations `k^n`
rather than the arm count `(k+1)^N`. **de Kroon, Mooij & Belgrave (2022)** weakened the requirement
from the full graph to a **separating set** `S` with `I ⟂ Y | S`, found by ordinary CI tests; the
resulting **information-sharing estimator** is unbiased with variance never worse than the naive
sample mean. **Bilodeau, Wang & Roy (2022, HAC-UCB)** made this *adaptive and safe*: it recovers the
improved `√(|Z|T)` rate when a post-action context d-separates action from reward, yet never
collapses to linear regret when it does not — all without knowing whether the d-separator holds.

**(D) Confounding & partial identification.**
**Maiti, Nair & Sinha (2022)** gave the first regret analysis under unobserved confounders on
semi-Markovian graphs, with regret scaling as an instance count `m(C) ≤ N` of arms whose do-effect
is *not* observationally identifiable — the rest are estimated from observation via the Tian–Pearl
c-component factorization, and only the "hard" arms are physically intervened on.
**Jamshidi, Etesami & Kiyavash (2024)** added **non-uniform intervention costs under a budget**, with
the cost-normalized optimum `argmax_a μ_a / c_a` (and, notably, corrected errors in prior budgeted
bounds).

**(E) Parametric scaling — beating the `2^N` arm explosion.**
When the mechanism family is parametric, regret becomes *polynomial in the graph* despite an
exponential arm space. **Varici et al. (2023, LinSEM-UCB/TS)** — linear SEM, soft interventions,
regret `Õ(d^{L+1/2}√(NT))` with max in-degree `d` and longest path `L`, arm count `2^N` gone.
**Yan et al. (2023)** made it **robust to non-stationarity** (a drift budget `C`). **Feng & Chen
(2023, BGLM-OFU)** handled **combinatorial** interventions (≤K nodes) under a binary generalized
linear model with `Õ(√T)` regret polynomial in `n`. **Xiong & Chen (2023, CCPE)** gave the
combinatorial *pure-exploration* counterpart. **AISTATS 2024 (general causal models)** pushed to
arbitrary Lipschitz mechanisms and continuous soft interventions, with regret governed by the
function class's **eluder dimension** and covering number — the SCM-substrate generalization.

**(F) Contextual & budgeted.**
**Subramanian & Ravindran (2022, Unc-CCB)** — the first to learn a *policy* (context → action) with
causal side-information and **targeted interventions** (act on a chosen sub-population), guided by an
information-gain acquisition score `Unc`. **Nair, Patil & Sinha (2021)** — the budgeted setting
where observation is cheaper than intervention, achieving *constant* (horizon-independent)
cumulative regret when the reward-parent distribution is known.

**(G) Unknown graph / online discovery.**
**Lu, Meisami & Tewari (2021, CN-UCB)** — without the DAG, use `O(d log² n)` "central-node"
interventions to locate the single reward-parent, then run ordinary UCB; cost of not knowing the
graph is only an additive, horizon-independent, `log n` term. Together with de Kroon's separating
sets and Elahi's partial discovery, this is the "learn structure and reward jointly" frontier.

### 1.5 The theoretical landscape

Two things recur and are worth stating explicitly, because they justify the whole enterprise.

**(i) The complexity constant.** Every regret bound replaces the arm count with a *structural*
quantity:

| Constant | Meaning | Source |
|---|---|---|
| `m(q)` (`2 ≤ m(q) ≤ N`) | # of "rare" variables | Lattimore 2016 |
| `m(η)` / `γ*` | min-max importance-weight variance over sampling designs | Lattimore / Yabe |
| `k^n = \|dom(Pa_Y)\|` | # of distinct reward-parent configurations | Lu 2020 |
| `m(C)` / `n(q̂)` | # of arms whose do-effect is *not* observationally identifiable (confounding cost) | Maiti / Jamshidi |
| `d, L` | max in-degree, longest causal path — regret `~ d^{Θ(L)}√(NT)`, poly in `N` despite `2^N` arms | linear-SEM / general |
| eluder dim / covering # | complexity of an arbitrary Lipschitz mechanism class | general models 2024 |

**(ii) Lower bounds and the price of not knowing.** Matching lower bounds (Lattimore Thm 2; Maiti
Thm 4.1; the linear-SEM `Ω(d^{L/2-2}√T)`) confirm these constants are the *right* measure of
difficulty. Two negative results matter for design: (a) Lu et al.'s `Ω(√(nKT))` lower bounds show
that when interventional effects are unidentifiable, causal knowledge confers **no** advantage over
plain MAB; and (b) Bilodeau et al.'s **impossibility of strict adaptivity** — you cannot claim the
benefit of a d-separator you cannot verify for free; robustness has a provable price (`T^{3/4}` vs
`√T` in the worst case). Both are the bandit face of the causal-incentives lesson that unverifiable
structural assumptions must be paid for.

### 1.6 The 2025–2026 frontier

Work published *after* the 22-paper review (mid-2025 → 2026) moves along five threads. The
field's centre of gravity is shifting from "known graph" toward **unknown/partial graph + transfer**,
and — importantly for applications — the most active *applied* development now happens under the
**Causal Bayesian Optimization** and **active experimental design** banners rather than the "causal
bandit" name.

1. **Structure uncertainty done properly.** SCM-MAB when the graph is known only up to a Markov
   equivalence class — *Structural Causal Bandits under Markov Equivalence* (Park, Arditi,
   Bareinboim & Lee, **NeurIPS 2025**); *Linear Causal Bandits: Unknown Graph and Soft Interventions*
   ([2411.02383](https://arxiv.org/abs/2411.02383)); *Combinatorial Causal Bandits without Graph
   Skeleton* ([2301.13392](https://arxiv.org/abs/2301.13392)); low-complexity graph-error control
   ([2408.11240](https://arxiv.org/abs/2408.11240)).
2. **Transportability / transfer.** Reusing source-environment interventional data in a costly target —
   *On Transportability for Structural Causal Bandits* ([2511.17953](https://arxiv.org/abs/2511.17953));
   *Transfer Learning in Latent Contextual Bandits through Causal Transportability*
   ([2502.20153](https://arxiv.org/abs/2502.20153)); building on Zhang & Bareinboim (IJCAI 2017).
3. **Search-space reduction for conditional interventions.** *The Minimal Search Space for Conditional
   Causal Bandits* ([2502.06577](https://arxiv.org/abs/2502.06577)) introduces the **mGISS** minimal
   set and a linear-time `C4` algorithm — extending POMIS-style pruning to *conditional* interventions.
4. **Causal Bayesian Optimization convergence.** CBO is effectively the continuous-reward cousin of
   causal bandits; the frontier now handles unknown graphs (*Causal Bayesian Optimization with Unknown
   Graphs*, [2503.19554](https://arxiv.org/abs/2503.19554)) and multiple objectives (*MO-CBO*,
   [2502.14755](https://arxiv.org/abs/2502.14755), which reuses POMIS-style set pruning). This is where
   "reward-maximizing intervention selection" is most actively developed.
5. **IV/compliance bandits & active hit-discovery still advancing.** *BRACE* (noncompliance +
   certified effects, [2603.09532](https://arxiv.org/abs/2603.09532)); *Many Needles in a Haystack*
   (active hit discovery for perturbation experiments, [2605.10196](https://arxiv.org/abs/2605.10196)).

The practical reading: a pgmpy module should treat **unknown-graph handling, transportability, and a
CBO-style continuous-reward path** as first-class rather than afterthoughts — they are where the field
is heading.

### 1.7 What the field agrees is still open

From the 2025 survey's own "open directions," plus the reviews: (1) **policy/contextual** causal
bandits are almost unexplored; (2) **joint causal discovery + bandit learning** on unknown/partial
graphs; (3) **robustness to distribution shift / transportability** between training and deployment;
(4) richer **unobserved-confounder** settings; (5) **budget/cost-aware and combinatorial**
interventions; (6) unifying simple- and cumulative-regret guarantees across general model/intervention
classes. Notably, several of these are places a *library* — not just a theorem — is the missing
ingredient.

---

## Part 2 — Practical Applications

**Headline finding (state it up front).** Causal bandits are, today, **overwhelmingly a theory
field**. Genuine *deployed* causal bandits are essentially nonexistent. The practical value shows up
in two adjacent places instead: (a) a thriving applied loop in **biology / experimental design** that
is the *same* "choose the next intervention to maximize an objective" problem under different names
(active learning, Bayesian optimization) — run on **real** data, including pgmpy's own Sachs network;
and (b) **production bandit systems that are contextual, not causal**, whose known failure modes
(confounding, off-policy bias, transfer) are exactly what causal bandits address. So the case for a
pgmpy module is *not* "reproduce deployed causal bandits" — it is "be the first usable library for a
loop applied scientists are already running by hand, and the online complement to pgmpy's existing
interventional-discovery and identification stack." This section is organized by the four lenses you
selected; each source is tagged for whether it is *genuinely causal* and its maturity.

### 2.1 Healthcare / precision medicine / dynamic treatment regimes / mHealth

Genuine causal-bandit work exists here, but it is **theoretical or simulated** — no patient-facing
deployment was found.

- **The canonical genuine causal online bandit:** *Counterfactual Data-Fusion for Online RL* (Forney,
  Pearl & Bareinboim, ICML 2017, [PMLR v70](https://proceedings.mlr.press/v70/forney17a.html)) — a
  counterfactual (intent-specific / ETT) Thompson-sampling agent fusing observational, experimental,
  and counterfactual data under unobserved confounders. **Theory + simulation only.**
- **Clearest genuine causal-bandit healthcare *application*:** *Risk-Averse MAB with Unobserved
  Confounders — Emotion Regulation in mHealth* (IEEE CDC 2022, [2209.04356](https://arxiv.org/abs/2209.04356))
  — a MABUC-style bandit for just-in-time adaptive interventions where the expert observes contexts the
  learner cannot (= unobserved confounders). **A simulated case study, not a patient pilot.**
- **Instrumental-variable / noncompliance bandits** (clinical-trial-motivated, theoretical):
  *Compliance-Aware Bandits* ([1602.02852](https://arxiv.org/abs/1602.02852)), *Instrument-Armed
  Bandits* (AISTATS 2018, [1705.07377](https://arxiv.org/abs/1705.07377)), *BRACE* (2026,
  [2603.09532](https://arxiv.org/abs/2603.09532)). These separate intent-to-treat from received
  treatment — genuinely causal, but not run on patients.
- **The most mature *real-data* causal-confounding healthcare result is OFFLINE, not a bandit:**
  *Delphic Offline RL under Nonidentifiable Hidden Confounding* (ICLR 2024,
  [2306.01157](https://arxiv.org/abs/2306.01157)) — evaluated on a sepsis benchmark **and real EHR**.
  Also *DTR identification with proxies of hidden confounders* ([2402.14942](https://arxiv.org/abs/2402.14942)),
  proximal causal inference — offline backward induction.
- **Reviews confirm the gap:** *Designing digital health interventions with causal inference and MABs*
  (Frontiers in Digital Health 2025, [PMC12177897](https://pmc.ncbi.nlm.nih.gov/articles/PMC12177897/))
  and *Bandit Algorithms for Precision Medicine* ([2108.04782](https://arxiv.org/abs/2108.04782)) both
  keep **causal inference (offline design/analysis) and bandits (online, usually *contextual* Thompson
  sampling) largely SEPARATE**. Integrating them is precisely the opportunity.

### 2.2 Biology & experimental design — the strongest practical home

This is where the causal-bandit *loop* is genuinely and actively practiced — on **real** data — but
under the names **active learning / optimal experimental design / Bayesian optimization**, not "causal
bandit." The objective ("pick the next intervention to move a target / maximize information under a
budget") is a causal bandit in all but name.

- **Flagship applied loop:** *Active Learning for Optimal Intervention Design in Causal Models* (Nature
  Machine Intelligence 2023, Zhang/Squires/Uhler, [2209.04744](https://arxiv.org/abs/2209.04744)) —
  sequentially selects interventions to drive a system's post-interventional mean to a target,
  validated on **real single-cell Perturb-CITE-seq data** to find perturbations inducing a cell-state
  transition. Reward-objective intervention selection with a causally-informed acquisition function.
- **The classical bridge:** *Causal Bayesian Optimization* (AISTATS 2020,
  [2005.11741](https://arxiv.org/abs/2005.11741)) — explicitly motivated by **gene-knockout in systems
  biology**, demonstrated on a **protein-signaling (Sachs-family) network**; balances explore/exploit
  *and* the observe-vs-intervene trade-off via do-calculus. Spawned DCBO, cCBO, MO-CBO.
- **Drug-discovery experimental-design benchmarks & methods** ("which CRISPR gene to perturb next"):
  *GeneDisco* (ICLR 2022, [2110.11875](https://arxiv.org/abs/2110.11875)), *DiscoBAX* (ICML 2023,
  [2312.04064](https://arxiv.org/abs/2312.04064)), *Near-Optimal Multi-Perturbation Experimental
  Design* (NeurIPS 2021, [2105.14024](https://arxiv.org/abs/2105.14024)), *BioBO* (2025,
  [2509.19988](https://arxiv.org/abs/2509.19988)), *Many Needles in a Haystack* (2026).
- **Directly on pgmpy's Sachs network:** active-learning intervention selection validated on the Sachs
  protein-signaling data — *Reconstructing Causal Biological Networks through Active Learning* (PLoS ONE
  2016), *GRN structure learning with Bayesian active learning* (BMC Bioinformatics 2025,
  [ECES/EBALD acquisition](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-025-06149-6)).
- **Why this is the natural first showcase for pgmpy:** pgmpy already **ships the Sachs interventional
  dataset** and has the SCM + do + identification machinery; this domain runs the loop on real data and
  connects directly to pgmpy's existing interventional-causal-discovery agenda.

### 2.3 Industry / production deployments — honest: mostly contextual, not causal

Production bandits are ubiquitous but almost uniformly **contextual, not causal**. The verification
pass flagged every large deployment below as `CONTEXTUAL`.

- **Contextual bandit deployments** (no confounder correction; graph-unaware): Netflix artwork
  personalization (130M+ users, [tech blog](https://netflixtechblog.com/artwork-personalization-c589f074ad76)),
  Microsoft Decision Service / MSN (+26% clicks), Yahoo LinUCB ([1003.0146](https://arxiv.org/abs/1003.0146)),
  Spotify, Stitch Fix, DoorDash, ZOZO Open Bandit Dataset ([2008.07146](https://arxiv.org/abs/2008.07146)),
  ShareChat ad-load balancing (180M MAU, [2309.11518](https://arxiv.org/abs/2309.11518)). The
  "causal-lite" layer is **off-policy evaluation** (IPW / doubly-robust replay), which corrects
  *logging-policy* bias — **not** unobserved confounding.
- **Practitioner consensus:** *Practical Bandits: An Industry Perspective* (WWW 2023,
  [2302.01223](https://arxiv.org/abs/2302.01223)) and a [multi-company survey](https://eugeneyan.com/writing/bandits/)
  confirm off-policy learning — not causal-graph methods — is what teams actually use; "none explicitly
  address causal confounders."
- **Where causal genuinely enters industry** (but offline / not clearly deployed): Amazon's *Contextual
  MAB for Causal Marketing* ([1810.01859](https://arxiv.org/abs/1810.01859)) and *Uplifting Bandits*
  (NeurIPS 2022, [2206.04091](https://arxiv.org/abs/2206.04091)) — uplift/incremental-effect bandits
  targeting persuadables.
- **Closest to a *deployed* causal bandit:** *Online Causal Inference for Advertising in Real-Time
  Bidding* (Marketing Science 2025, Vol 44(1), Stanford/JD.com, [1908.08600](https://arxiv.org/abs/1908.08600)) —
  identifies the ad causal effect via **auction structure** and runs an adapted Thompson-sampling
  bandit, **validated on real RTB data**.
- **System / DB tuning:** *DBA Bandits* (ICDE 2021, [2010.09208](https://arxiv.org/abs/2010.09208)),
  CGPTuner — contextual MAB over configurations, not causal.
- **Takeaway:** the causal upgrade over contextual bandits (confounder correction, transportability,
  counterfactual reuse) is a recognized *gap* in production, not deployed practice. A library that
  makes it easy is an adoption vector, not a re-implementation of existing systems.

### 2.4 Adjacent formalisms — where the applied surface actually lives

Most real "select interventions online to optimize an outcome on a causal model" work is published
under adjacent names. Design the module to sit at the atomic core of these, not as a narrow silo.

- **Causal Bayesian Optimization (CBO)** + family (DCBO, cCBO, MO-CBO, CBO-unknown-graph) — GP-based,
  continuous reward; the most active applied line (see §2.2).
- **Causal RL / deconfounding RL / offline RL under hidden confounding** — the multi-step
  generalization (Delphic ICLR 2024 on sepsis/EHR; *Deconfounding RL* [1812.10576](https://arxiv.org/abs/1812.10576)).
- **Active causal discovery / optimal experimental design** — the structure-learning-objective cousin
  (Sussex 2021; the Sachs active-learning line).

### 2.5 Benchmarks, testbeds & the software gap

**The software niche is essentially open** — no maintained, general causal-bandit library exists.

- **Only pip-installable causal-bandit package:** `causalrl` (PyPI 2025, [pypi.org/project/causalrl](https://pypi.org/project/causalrl/))
  — POMIS/MIS, causal Thompson sampling, a MABUC testbed; early-stage/demo, organized around
  Bareinboim's causal-RL taxonomy.
- **Canonical POMIS reference code:** `sanghack81/SCMMAB-NIPS2018`
  ([GitHub](https://github.com/sanghack81/SCMMAB-NIPS2018), MIT, numpy/networkx, ships a brute-force
  parity oracle) — a ready **correctness oracle** for a pgmpy POMIS implementation.
- **A decision layer already built ON pgmpy:** `pycid` (DeepMind, [GitHub](https://github.com/causalincentives/pycid))
  — CIDs/MAIDs, optimal policies, Nash equilibria; **offline, no bandit**. Proof pgmpy is a viable
  substrate and a natural neighbour.
- **CBO code** (VirgiAgl/CausalBayesianOptimization, DeepMind ccbo) — design reference for the loop.
- **Every major causal library has NO online decision/bandit layer** (verified): DoWhy/EconML (stops at
  offline policy trees), causal-learn, Ananke, CausalNex, Tigramite.
- **Reusable interventional benchmarks:** *CausalBench* (GSK, >200k CRISPR-perturbation samples —
  discovery, not bandit), GeneDisco/DiscoBAX (active experimental design). pgmpy would still need to
  ship small causal-bandit *environments* (parallel bandit, front-door, the SCM-MAB tasks) for regret
  reproducibility.

### 2.6 Maturity verdict

| Lens | Genuinely causal? | Maturity | Implication for pgmpy |
|---|---|---|---|
| Healthcare / DTR / mHealth | Yes (MABUC, IV bandits) | Theory / simulated; real-data results are *offline* | Long-term credibility use-case; not a v1 demo |
| **Biology / experimental design** | Effectively yes (active-learning framing) | **Real data, actively practiced** (incl. Sachs) | **The v1 showcase** — connects to shipped datasets |
| Industry / production | Mostly **no** (contextual) | Deployed at scale, but not causal | Adoption vector: "causal upgrade" over contextual bandits |
| Adjacent (CBO / causal RL) | Yes | Real + active (esp. CBO) | Design the module as the atomic core of this surface |
| Software / benchmarks | — | **Open niche**; only `causalrl` exists | First-mover opportunity; reuse SCMMAB code as oracle |

**Bottom line:** pitch a pgmpy causal-bandits module as (1) the **first usable library** for POMIS-style
structure-aware intervention selection, (2) the **online/sequential complement** to pgmpy's existing
interventional-discovery, identification, and CBO-adjacent capabilities, and (3) the natural home for
the **biology experimental-design loop on real interventional data** (starting from Sachs).

---

## Part 3 — pgmpy Roadmap: Features & Algorithms to Implement

### 3.1 Design thesis

A causal bandit is **online intervention-selection on a causal graph with a reward node**. pgmpy
already owns the entire *static* half of that sentence — a typed graph, the do-operator, exact/
approximate inference, backdoor/frontdoor/IV identification, tabular and continuous SEM model
classes, node-role annotations, and ancestral simulation. What is missing is the *dynamic* half: a
thin **sequential-decision layer** (an environment to interact with, online index/estimator objects,
and regret accounting), plus **two graph-algorithm additions** (POMIS enumeration and latent
projection) and a **richer soft-intervention API**. The strategic implication:

> A `pgmpy.causal_bandits` module is mostly a *thin sequential loop over machinery pgmpy already
> has*, not a from-scratch build. The new-code surface is small and well-contained.

**One core, shared with the CID/incentives work.** The companion `causal_incentives_literature_review.md`
converged on the same primitive stack (typed graph → interventions → graphical-criteria engine →
partial-ID). POMIS = the value-of-control criterion; POMPS = value-of-information + control together;
partial-ID bounds are shared. **Recommendation:** implement the graph-criteria and identification
pieces as a reusable `causal_decision` core, but ship v1 as a standalone `pgmpy.causal_bandits` that
*imports* that core — so the bandit work is not blocked on a full CID subsystem, while avoiding a
duplicate POMIS/partial-ID implementation later.

### 3.2 What pgmpy already has vs. what is new (verified against current code)

| Capability a causal bandit needs | In pgmpy today | Gap to close |
|---|---|---|
| Typed graph with reward/action/context nodes | `_GraphRolesMixin` (roles as a per-node set), `SimpleCausalModel`; props for exposures/outcomes/latents | add `action`/`reward`/`context` role tags + a thin typed model; **small** |
| Atomic / multi-node `do` (structural) | `_CoreGraph.do()`, `DAG.do()`, `DiscreteBayesianNetwork.do()` (edge-cut + marginalize) | reuse as-is |
| Value-fixing `do` query `P(Y\|do(x),z)` | `CausalInference.query(do=, evidence=, adjustment_set=)` — **tabular only** | reuse (tabular); continuous path is new |
| **Soft / mechanism-reshaping** intervention | `simulate(virtual_intervention=...)` on Discrete/LinearGaussian/Functional — **marginal-only** (parentless replacement CPD) | extend to **parent-dependent** soft `do(P(V\|pa))` and continuous `f(pa; a)`; **medium** |
| Reward-parent decomposition `μ_a=Σ_z μ_z P(Pa_Y=z\|a)` | variable elimination / `CausalInference` | reuse (it *is* one-step VE) |
| Confounding: ADMG/MAG, c-components | `ADMG`/`MAG`, `get_district()` (c-component), m-separation | reuse; **latent projection ABSENT** → new |
| Backdoor / frontdoor / IV identification | `Adjustment`, `Frontdoor` (DAG-only), `get_ivs` | reuse; **general ID (Shpitser/Tian) and partial-ID bounds ABSENT** → new |
| **POMIS / MIS / MUCT / interventional border** | **ABSENT** (no matches in repo) | **new graph-algorithm module** (reuses `get_district`, `get_ancestors`, `do()`) |
| **Online estimation & index policies** (UCB/kl-UCB/TS; IS / info-sharing estimators; regret) | **ABSENT** — all estimators are batch/offline, no `partial_fit`/streaming | **new sequential layer** (the module's heart) |
| Importance-sampling reuse of samples across arms | `BayesianModelSampling.likelihood_weighted_sample()` | reuse as the IS backbone; clipped-IS / off-policy estimator is new |
| Intervention **cost** metadata + **budget** loop | **ABSENT** | small: per-node cost attribute (reuse the roles/attr system) + a budget-aware loop |
| Unknown-graph / **interventional** structure discovery | observational PC/GES/…, sklearn `fit`; **FCI ABSENT**, no interventional/active discovery | **new**: interventional-discovery loop; can wrap existing CI tests |
| CI tests for separating sets | rich `ci_tests` (ChiSquare/GSq/FisherZ/…) + `minimal_dseparator` | reuse directly |

### 3.3 The prioritized feature list (five tiers)

Ordered by value-per-unit-effort. Tiers 1–2 are the recommended v1; each tier is independently
useful and testable.

**Tier 1 — Known-graph, atomic, online loop (the foundation).**
The minimum that makes pgmpy a causal-bandit library at all.
- `CausalEnvironment` — a wrapper over a `DiscreteBayesianNetwork`/`LinearGaussianBayesianNetwork`
  that, given an intervention, returns a sampled reward (reuses `simulate(do=...)`). The benchmarking
  substrate and the thing a `CausalBandit` interacts with.
- Node-role tags `action` / `reward` / `context` (extend `_GraphRolesMixin`).
- **Index policies**: `UCB`, `kl-UCB`, `ThompsonSampling` index objects (new, self-contained).
- **Reward-parent decomposition estimator** `μ_a = Σ_z μ̂_z P(Pa_Y=z | a)` (one-step VE over `Pa_Y`).
- **Interaction loop** with **simple- and cumulative-regret** accounting.
- *Algorithms unlocked:* **C-UCB / C-TS** (Lu 2020), **Causal Thompson Sampling / MABUC** (Bareinboim
  2015, reusing pgmpy counterfactual/ETT), the **Lattimore parallel-bandit** baseline.

**Tier 2 — Structural arm reduction (the differentiator).**
This is what no general-purpose bandit library has, and it is pure graph theory pgmpy is built for.
- `possibly_optimal_minimal_intervention_sets(G, Y)` — POMIS enumeration via **MUCT** (descendant- +
  c-component-closure from `Y`, reusing `get_district`/`get_ancestors`) and **interventional border**;
  the sound-complete `IB(G_\overline{X},Y)=X` test (Lee & Bareinboim 2018).
- `minimal_intervention_sets`; **POMPS** (mixed policy scopes: where-to-intervene + what-to-observe,
  Lee & Bareinboim 2020) reusing d-separation.
- **Latent projection** of an ADMG onto a manipulable subset (for non-manipulable variables,
  Lee & Bareinboim 2019).
- Restrict any Tier-1 bandit to the POMIS arm set → provably lower regret.
- *Algorithms unlocked:* **POMIS-kl-UCB**, POMPS-scoped policy learning, partial-structure-discovery
  arm sets (Elahi 2024).

**Tier 3 — Confounding & partial identification.**
- **c-component / Tian–Pearl do-from-observation** estimator: estimate `E[Y|do(X)]` from observational
  samples where identifiable, and pull only the "hard" (unidentifiable) arms — the `m(C)` machinery
  (Maiti 2022). Reuses `get_district` + a general-ID reduction.
- **Separating-set information-sharing estimator** `μ̂_IS(ζ)=Σ_s μ̂(s) p̂(s|ζ)` with CI-test-discovered
  `S` (de Kroon 2022) — reuses `ci_tests` + `minimal_dseparator`.
- **d-separator-adaptive** policy (HAC-UCB): a `is_conditionally_benign` / d-separation guard that
  recovers `√(|Z|T)` when valid and never goes linear when not (Bilodeau 2022).
- **Partial-identification bounds** (min/max of `E[Y|do]` over consistent SCMs) for the unidentifiable
  case — shared with the CID review; also a generally valuable pgmpy addition.

**Tier 4 — Parametric, soft, combinatorial, budgeted.**
- **Parent-dependent / continuous soft interventions** (extend `virtual_intervention` beyond
  marginal-only) — unblocks Sen 2017, Varici 2023, general-models 2024.
- **Linear-SEM bandit** `LinSEM-UCB/TS` on `LinearGaussianBayesianNetwork` (path-sum reward
  `f(B)=Σ_ℓ [B^ℓ]`, per-node least squares + confidence ellipsoids) — Varici 2023; robust variant
  Yan 2023.
- **BGLM** (monotone-link binary) CPD family + `BGLM-OFU` and combinatorial pure exploration
  (Feng & Chen 2023; Xiong & Chen 2023).
- **General Lipschitz-SCM** bandit on `FunctionalBayesianNetwork` (GCB-UCB/TS, 2024).
- **Cost + budget**: per-node intervention-cost attribute, cost-normalized optimum `argmax μ_a/c_a`,
  budget-aware loop (Nair 2021; Jamshidi 2024).
- **Contextual / targeted-intervention** policy learning with a `Unc`-style info-gain acquisition
  (Subramanian & Ravindran 2022).

**Tier 5 — Unknown graph / online discovery.**
- **Interventional / active structure discovery** interleaved with the bandit loop: central-node
  experiments (CN-UCB, Lu 2021), partial-ancestor discovery (Elahi 2024), separating-set discovery
  (de Kroon 2022). Wraps existing `ci_tests`; the *interventional, sample-budgeted, adaptive* half is
  new. (Also motivates finally adding **FCI**, currently absent.)

**Cross-cutting.**
- `metrics`: simple/cumulative regret curves, arm-count reduction, plug into the existing `metrics`
  package.
- A `datasets`-style **benchmark suite** of small causal-bandit environments (parallel bandit,
  confounded/front-door graphs, the SCM-MAB tasks) for reproducible regret comparisons.

### 3.4 Concrete algorithm inventory (the "algorithms to implement" list)

| Algorithm | Paper | Tier | Reuses (pgmpy) | New piece |
|---|---|---|---|---|
| Causal Thompson Sampling (MABUC) | Bareinboim 2015 | 1 | counterfactual/ETT query, roles | per-intuition Beta posteriors, loop |
| Lattimore parallel-bandit + IS estimator | Lattimore 2016 | 1 | `do()`, `Pa_Y`, `likelihood_weighted_sample` | truncated-IS estimator, `m(q)` design |
| C-UCB / C-TS | Lu 2020 | 1 | reward-parent decomposition (VE) | UCB/TS index objects |
| POMIS enumeration + POMIS-kl-UCB | Lee & Bareinboim 2018 | 2 | `get_district`, `get_ancestors`, `do()` | MUCT/IB, recursive enumeration |
| POMPS (mixed policy scopes) | Lee & Bareinboim 2020 | 2 | d-separation | scope enumeration + non-redundancy tests |
| Non-manipulable POMIS | Lee & Bareinboim 2019 | 2 | ADMG | latent projection + z-ID reuse |
| Partial-structure-discovery bandit | Elahi 2024 | 2/5 | ADMG, c-components | ancestor-side interventional discovery |
| Separating-set info-sharing bandit | de Kroon 2022 | 3 | `ci_tests`, `minimal_dseparator` | IS estimator + Dirichlet/Beta TS |
| Confounded atomic bandit `m(C)` | Maiti 2022 | 3 | `get_district`, ID | do-from-obs estimator + hard-arm set |
| HAC-UCB (d-separator adaptive) | Bilodeau 2022 | 3 | d-separation, frontdoor | benign-test hedge policy |
| Confounded-budgeted bandit | Jamshidi 2024 | 3/4 | c-components, ID | cost metadata + budget loop |
| LinSEM-UCB / TS (+ robust) | Varici 2023; Yan 2023 | 4 | `LinearGaussianBayesianNetwork`, LS fit | path-sum reward, confidence ellipsoids |
| BGLM-OFU / CCPE (combinatorial) | Feng & Chen 2023; Xiong & Chen 2023 | 4 | discrete CPDs, `NoisyORCPD` | BGLM CPD family, combinatorial arms |
| General-SCM GCB-UCB / TS | AISTATS 2024 | 4 | `FunctionalBayesianNetwork` | continuous soft-do, eluder-set UCB |
| Unc-CCB (contextual policy) | Subramanian 2022 | 4 | `query(do=,evidence=)`, Bayesian counts | `Unc` info-gain acquisition |
| Budgeted simple/cumulative regret | Nair 2021 | 4 | no-backdoor check | budget-aware observe/intervene split |
| CN-UCB (unknown graph) | Lu 2021 | 5 | PC/CPDAG, `simulate(do=)` | central-node interventional search |

### 3.5 New primitives (the building blocks these algorithms share)

1. **`action`/`reward`/`context` roles** — extend `_GraphRolesMixin`; shared with the CID work.
2. **Parent-dependent soft intervention** — generalize `virtual_intervention` (currently marginal-only)
   to replace `P(V|pa)` / reshape `f(pa; a)`. Unblocks the entire parametric tier.
3. **POMIS / MUCT / IB / POMPS** graph algorithms — the reusable value-of-control engine.
4. **Latent projection** — marginalize a subset into directed + bidirected edges (also useful beyond
   bandits).
5. **Online estimator objects** — reward-parent decomposition, clipped importance-sampling,
   separating-set information-sharing (stateful, updated per round).
6. **Index policies** — `UCB`, `kl-UCB`, `ThompsonSampling` (backend-agnostic, self-contained).
7. **`CausalEnvironment` + interaction loop + regret metrics** — the sequential harness.
8. **Intervention-cost metadata + budget loop** — a per-node cost attribute and a budget-aware policy.
9. **General ID + partial-ID bounds** — complete `ProbabilityExpressionTree` into an actual
   identify()/bounds engine (needed by Tier 3; valuable library-wide).

### 3.6 Illustrative API (sketch, not a spec)

```python
from pgmpy.causal_bandits import CausalEnvironment, CausalBandit

env = CausalEnvironment(model, reward="Y", actions=["X1", "X2"])   # wraps simulate(do=...)
bandit = CausalBandit(
    env.graph, reward="Y",
    arm_reduction="pomis",         # Tier 2: restrict arms to POMIS
    policy="kl_ucb",               # Tier 1 index
    objective="cumulative",        # or "simple"
)
result = bandit.run(env, horizon=5000)     # or partial_fit-style stepping
result.cumulative_regret            # regret curve; result.best_intervention
```

### 3.7 Open design questions (settle in brainstorming, per the proposal template)

- **One subsystem or two:** shared `causal_decision` core vs. standalone `causal_bandits` importing it
  (recommended: standalone v1 on a shared core).
- **v1 scope:** Tiers 1+2 only, or include the contextual/policy objective from the start?
- **Online/offline boundary:** a scikit-learn `partial_fit`/environment API vs. a callback-driven
  simulator (pgmpy is entirely batch today — this is the biggest architectural choice).
- **Soft/continuous interventions:** unify the "soft do" API across bandit, CID and SCM work;
  tabular-first vs. `FunctionalCPD`-first.
- **PyCID:** absorb / depend on / reimplement its POMIS-adjacent graph algorithms?
- **Ship a benchmark `CausalEnvironment` suite** for regret reproducibility alongside the learner?

### 3.8 Competitive positioning

- **PyCID** (causalincentives.com) is built *on pgmpy* (`BayesianNetwork`/`TabularCPD` subclasses) but
  is tabular-only, brute-force, and has no bandit/online loop — pgmpy can offer the sequential layer
  it lacks.
- **causal-learn, DoWhy, Ananke, CausalNex, Tigramite** have identification/discovery/effect estimation
  but **no causal-bandit / online-decision component** — this is an open niche.
- Reference research code exists per-paper (e.g. the SCM-MAB repo) but there is **no maintained,
  general causal-bandit library**. pgmpy is unusually well-positioned because it already has the graph
  + do-operator + identification + SEM model classes under one roof.

### 3.9 Recommended v1

**Tiers 1 + 2 + the cross-cutting environment/metrics**, delivered as a standalone
`pgmpy.causal_bandits` on a shared graph-criteria core: a `CausalEnvironment`, `UCB`/`kl-UCB`/`TS`
policies, the reward-parent decomposition estimator, POMIS arm reduction, simple/cumulative regret
accounting, and a small benchmark suite. This reproduces C-UCB/C-TS, MABUC, the Lattimore baseline,
and POMIS-kl-UCB — the field's backbone — while touching a minimal, well-contained new-code surface.
Tiers 3–5 then extend along the confounding, parametric, and unknown-graph axes.

---

## Appendix: sources & method

Produced by (1) building on the existing `causal_bandits_literature_review.md` (22 papers, read in
full) for the overview; (2) a fresh multi-agent web sweep across six angles (recent-algorithms
refresh, healthcare, biology/experimental design, industry, software/benchmarks, adjacent formalisms)
that surfaced **89 sources** and extracted **103 application claims**, each tagged `causal-bandit` vs
`contextual` vs `other-causal` and by maturity; and (3) a capability audit of the current pgmpy tree
for Part 3.

**Verification caveat (transparency).** The sweep included an adversarial 3-vote verification stage
tuned to the two failure modes here — mislabeling *contextual* bandits as *causal*, and overstating
*deployed* vs *proposed*. That stage completed for ~16 claims before the run hit a session usage limit;
the remaining claims were screened **by hand** against the same criteria, using the extractor agents'
own `causal_type`/maturity tags (which were appropriately skeptical — every large industry deployment
was independently flagged CONTEXTUAL). Treat claims here as **well-sourced and hand-screened** rather
than fully 3-vote-verified; primary links are inline so any specific claim can be checked directly.

**Post-hoc spot-check (2026-07-13).** The highest-risk citations — the recent 2025–26 papers most at
risk of hallucinated arXiv IDs, plus the most consequential applied claim — were confirmed by hand to
exist and be correctly attributed: Markov-equivalence SCM-MAB (NeurIPS 2025), transportability
(2511.17953), BRACE (2603.09532), the RTB Thompson-sampling bandit (1908.08600, Marketing Science
44(1), 2025), and the `causalrl` package. No hallucinated IDs were found among the spot-checked items.
