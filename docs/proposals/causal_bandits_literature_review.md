# Causal Bandits — Literature Review

**Purpose:** A precise, equation-level survey of the **causal bandits** literature, mapped
in relation to **(a) pgmpy** (what model classes / algorithms a library would need) and
**(b) the Causal Incentives review** (`causal_incentives_literature_review.md`) — since a
causal bandit is, structurally, an *online intervention-selection problem on a CID*. Both
reviews feed a forthcoming pgmpy design proposal.

**Seed paper:** Lattimore, Lattimore & Reid (2016), *Causal Bandits: Learning Good
Interventions via Causal Inference* ([arXiv:1606.03203](https://arxiv.org/abs/1606.03203)).

**Status:** ✅ All 22 papers summarized from full-text reads (6 thematic batches). Glossary, a causal-bandit↔CID correspondence, and a pgmpy-design synthesis complete. Companion to `causal_incentives_literature_review.md`; both feed the design proposal.

---

## How to read this document

Papers are grouped by sub-problem (not chronologically). Each entry follows a fixed
template:

- **TL;DR** — one sentence.
- **Problem & motivation** — the gap addressed.
- **Formal setup & key definitions** — model class, intervention type, reward, regret objective.
- **Key equations** — verbatim, in LaTeX (regret bounds, estimators, criteria).
- **Main results / theorems** — the load-bearing bounds and graphical criteria.
- **Algorithms** — the procedure(s) contributed.
- **Relevance to pgmpy** — model/data structures and algorithms a library would need.
- **Relation to causal incentives / CIDs** — how the bandit maps onto the CID/incentive
  framework (intervention = decision node, target = utility node, VoI/VoC, where-to-intervene
  ↔ control incentives, etc.).

Recurring objects (SCM, intervention types, simple vs cumulative regret, POMIS, MIS, …) are
defined once in the **Glossary** at the bottom.

---

## The causal bandit problem in one paragraph

A causal bandit problem is a stochastic multi-armed bandit in which **arms are interventions**
$\mathrm{do}(\mathbf X=\mathbf x)$ on the variables of a causal graph $G$, and the **reward is a
designated target variable** $Y$ (a descendant). The learner knows $G$ (sometimes only
partially, or not at all) but not the conditional distributions. Because pulling one arm also
reveals the values of *other* graph variables, there is **information leakage between arms**
that a structure-blind bandit ignores. The two objectives are **simple regret** (best-arm
identification after a budget $T$) and **cumulative regret** (online). Research varies along:
intervention type (atomic/soft/non-atomic/combinatorial), what is known of $G$ (full / partial
/ unknown), confounding (none / latent), payoff structure (tabular / linear-SEM / BGLM),
context, and budget/cost constraints.

---

## Paper index

| # | Theme | Title | Venue / Year | Link |
|---|-------|-------|--------------|------|
| 1 | Foundations & simple regret | Bandits with Unobserved Confounders: A Causal Approach (MABUC) | NeurIPS 2015 | [pdf](https://proceedings.neurips.cc/paper/2015/hash/795c7a7a5ec6b460ec00c5841019b9e9-Abstract.html) |
| 2 | Foundations & simple regret | Causal Bandits: Learning Good Interventions via Causal Inference | NeurIPS 2016 | [1606.03203](https://arxiv.org/abs/1606.03203) |
| 3 | Foundations & simple regret | Identifying Best Interventions through Online Importance Sampling | ICML 2017 | [1701.02789](https://arxiv.org/abs/1701.02789) |
| 4 | Foundations & simple regret | Causal Bandits with Propagating Inference | ICML 2018 | [1806.02252](https://arxiv.org/abs/1806.02252) |
| 5 | Where to intervene (SCM-MAB) | Structural Causal Bandits: Where to Intervene? (POMIS) | NeurIPS 2018 | [pdf](https://par.nsf.gov/servlets/purl/10111018) |
| 6 | Where to intervene (SCM-MAB) | Structural Causal Bandits with Non-Manipulable Variables | AAAI 2019 | [pdf](https://causalai.net/r40.pdf) |
| 7 | Where to intervene (SCM-MAB) | Characterizing Optimal Mixed Policies: Where to Intervene and What to Observe | NeurIPS 2020 | [search](https://arxiv.org/abs/2011.00886) |
| 8 | Unknown graph & structure discovery | Regret Analysis of Bandit Problems with Causal Background Knowledge (C-UCB/C-TS) | UAI 2020 | [1910.04938](https://arxiv.org/abs/1910.04938) |
| 9 | Unknown graph & structure discovery | Causal Bandits with Unknown Graph Structure | NeurIPS 2021 | [2106.02988](https://arxiv.org/abs/2106.02988) |
| 10 | Unknown graph & structure discovery | Causal Bandits without prior knowledge using separating sets | CLeaR 2022 | [2009.07916](https://arxiv.org/abs/2009.07916) |
| 11 | Unknown graph & structure discovery | Partial Structure Discovery is Sufficient for No-regret Learning in Causal Bandits | 2024 | [2411.04054](https://arxiv.org/abs/2411.04054) |
| 12 | Confounders & d-separators | Adaptively Exploiting d-Separators with Causal Bandits | NeurIPS 2022 | [2202.05100](https://arxiv.org/abs/2202.05100) |
| 13 | Confounders & d-separators | A Causal Bandit Approach to Learning Good Atomic Interventions in Presence of Unobserved Confounders | UAI 2022 | [2107.02772](https://arxiv.org/abs/2107.02772) |
| 14 | Confounders & d-separators | Confounded Budgeted Causal Bandits | 2024 | [2401.07578](https://arxiv.org/abs/2401.07578) |
| 15 | Linear SEM & combinatorial | Causal Bandits for Linear Structural Equation Models | JMLR 2023 | [2208.12764](https://arxiv.org/abs/2208.12764) |
| 16 | Linear SEM & combinatorial | Robust Causal Bandits for Linear Models | 2023 | [2310.19794](https://arxiv.org/abs/2310.19794) |
| 17 | Linear SEM & combinatorial | Combinatorial Causal Bandits | AAAI 2023 | [2206.01995](https://arxiv.org/abs/2206.01995) |
| 18 | Linear SEM & combinatorial | Combinatorial Pure Exploration of Causal Bandits | ICLR 2023 | [2206.07883](https://arxiv.org/abs/2206.07883) |
| 19 | Contextual, budgeted, general | Causal Contextual Bandits with Targeted Interventions | ICLR 2022 | [openreview](https://openreview.net/pdf?id=F5Em8ASCosV) |
| 20 | Contextual, budgeted, general | Budgeted and Non-budgeted Causal Bandits | AISTATS 2021 | [2012.07058](https://arxiv.org/abs/2012.07058) |
| 21 | Contextual, budgeted, general | Causal Bandits with General Causal Models and Interventions | AISTATS 2024 | [2403.00233](https://arxiv.org/abs/2403.00233) |
| 22 | Contextual, budgeted, general | Causality in Bandits: A Survey | ACM Comp. Surveys 2025 | [doi](https://dl.acm.org/doi/10.1145/3744917) |

---

## A. Foundations & Simple Regret (known graph)

### Bandits with Unobserved Confounders: A Causal Approach (MABUC) (NeurIPS, 2015)
**Authors:** Elias Bareinboim, Andrew Forney, Judea Pearl (Bareinboim & Forney contributed equally)
**Link:** https://proceedings.neurips.cc/paper_files/paper/2015/file/795c7a7a5ec6b460ec00c5841019b9e9-Paper.pdf (UCLA Tech Report R-460)
**TL;DR:** When a multi-armed bandit has unobserved confounders that drive both the agent's "natural" arm choice and the payoff, maximizing the experimental distribution $E[Y\mid do(X)]$ is provably insufficient; the agent must instead optimize the counterfactual "effect-of-treatment-on-the-treated" $E[Y_{X=a}\mid X=x]$, combining observational and experimental data.

**Problem & motivation:** Standard bandit algorithms implicitly estimate the *experimental* (interventional) distribution $P(y\mid do(x))$, assuming that is the right target. The paper shows that under *unobserved confounders* (MABUC) — latent variables affecting both the action and the reward — the observational $P(y\mid x)$ and experimental $P(y\mid do(x))$ distributions diverge, and *both* are needed for optimal play. It is the conceptual precursor of causal bandits: it reframes the bandit as a structural-causal-model decision problem and introduces the observe-vs-do distinction as a first-class object.

**Formal setup & key definitions:** Model class is a **Structural Causal Model** $M = \langle U, V, f, P(u)\rangle$ (Def 3.1). **Def 3.2 (K-Armed Bandit with Unobserved Confounders):** action $X_t\in\{x_1,\dots,x_k\}$ (chosen by Nature in the observational case, or by $do(X_t=\pi(\cdot))$ for policy $\pi$ in the experimental case); $U_t$ is the unobserved confounder encoding both the arm's payout rate *and* the propensity to choose that arm; reward $Y_t\in\{0,1\}$ with $y_t=f_y(x_t,u_t)$. The latent $U_t \to X_t$ edge (agent's "predilection"/intuition) is what distinguishes MABUC from a standard MAB (where $U_t$ affects only $Y_t$). The objective is the standard cumulative-reward / regret objective, evaluated against an *oracle policy that has access to the confounder realization* $u_t$.

**Key equations:**
- $X \leftarrow f_X(B,D) = (D \wedge \neg B) \vee (\neg D \wedge B) = D \oplus B$ — Greedy-Casino structural equation: the gambler's natural machine choice is the XOR of "drunk" ($D$) and "machine blinking" ($B$), each Bernoulli$(0.5)$.
- $P(Y=1\mid X=M_1) = P(Y=1\mid X=M_2) = 0.15$ vs $P(Y=1\mid do(X=M_1)) = P(Y=1\mid do(X=M_2)) = 0.30$ — observational *and* experimental distributions each make the two arms indistinguishable, yet disagree (0.15 vs 0.30); the residual gap encodes the confounder.
- $\arg\max_a E[Y\mid do(X=a)]$ — Eq. (2): the standard "maximize the experimental distribution" rule, shown insufficient for MABUC.
- $\arg\max_a E[Y_{X=a}=1\mid X=x]$ — Eq. (3): the **Regret Decision Criterion (RDC)**, where $x$ is the player's natural predilection and $a$ the final decision; this counterfactual is the **Effect of Treatment on the Treated (ETT)**.
- $E[Y_{X=1}=1\mid X=1] = E[Y=1\mid X=1]$ — consistency axiom: the "follow-your-intuition" counterfactual equals the observational conditional (estimable from passive data).
- $E[Y_{X=0}\mid X=1] > E[Y_{X=1}\mid X=1] \Leftrightarrow E[Y_{X=0}\mid X=1] > P(Y\mid X=1)$ — Eq. (4): binary decision rule telling the agent when to *override* its intuition.

**Main results / theorems:** No regret-rate theorem (this is a conceptual/algorithmic paper). The core result is by construction: two parameterizations where standard algorithms fail. **Greedy Casino**: per-confounder payoffs where all standard algorithms ($\epsilon$-greedy, UCB1, EXP3, Thompson Sampling) perform no better than a coin flip and incur linearly growing regret. **Paradoxical Switching**: observational says $M_1$ is better ($P(y\mid M_1){=}0.4 > P(y\mid M_2){=}0.15$) while experimental says $M_2$ is better ($P(y\mid do(M_2)){=}0.375 > P(y\mid do(M_1)){=}0.35$). Empirically ($T{=}1000$, $N{=}1000$ MC runs), Causal Thompson Sampling cumulative regret is $0.94$/$4.71$ (Exps 1/2) vs context-augmented $TS^Z$ ($11.03$/$13.39$) and standard $TS$ ($150.47$/$83.56$).

**Algorithms:** **Causal Thompson Sampling ($TS^C$)** (Algorithm 1). It (1) *seeds* the ETT estimates from the observational distribution $E[Y_{X=a}\mid X]\leftarrow P_{obs}(y\mid X)$ via consistency; then each round (2) reads the agent's intuition $x\leftarrow \text{intuition}(t)$, computes the counter-intuition payoff $Q_1 = E[Y_{X=x'}\mid X=x]$ and intuition payoff $Q_2 = P(y\mid X=x)$, sets a weight $\text{bias} = 1 - |Q_1 - Q_2|$, biases Beta-sampling toward whichever the ETT comparison favors, pulls, and updates. The key mechanism is **intention-specific randomization**: treat the agent's about-to-act choice as observed "intention," then randomize the final action, so each arm's reward is tracked *per intuition state* — recovering the confounder-conditional optimum without observing $U$.

**Relevance to pgmpy:** Needs an SCM/CBN with an explicit latent (unobserved) confounder node, a designated action node $X$ and reward node $Y\in\{0,1\}$, the ability to compute both observational $P(y\mid x)$ and interventional $P(y\mid do(x))$ on the *same* graph (pgmpy's `CausalInference`/do-operator), and counterfactual/ETT queries $E[Y_{X=a}\mid X=x]$. A bandit/online-decision loop with per-context (per-intuition) Beta posteriors would be the new component; the counterfactual seeding from observational data is a reusable primitive.

**Relation to causal incentives / CIDs:** Maps cleanly to a CID with decision node $X$ (the arm), utility node $Y$, and chance node $U$ as a *common parent of both* $X$ and $Y$. The MABUC insight is precisely a **value-of-information** statement: the agent's intuition is an observed proxy for $U$ available at the decision node, so the optimal policy must *condition the decision on it* rather than apply a context-free $\arg\max_a E[Y\mid do(a)]$. The RDC/ETT rule is the conditional-on-information optimal decision rule; ignoring the $U\to X$ edge (as standard bandits do) is the canonical CID modeling error. Shared primitives: typed graph (decision/utility/chance), do-operator, expected-utility maximization, and counterfactual evaluation of alternative decisions.

---

### Causal Bandits: Learning Good Interventions via Causal Inference (NeurIPS, 2016)
**Authors:** Finnian Lattimore, Tor Lattimore, Mark D. Reid
**Link:** https://arxiv.org/abs/1606.03203
**TL;DR:** By exploiting the causal graph as a source of structured "interventional" feedback (one intervention reveals information about many arms), a learner can achieve simple regret $O(\sqrt{m(\mathbf{q})/T})$ where the difficulty constant $m(\mathbf{q})\le N$ replaces the structure-blind $O(\sqrt{N/T})$, often by a large margin.

**Problem & motivation:** Standard best-arm-identification treats each of $N$ interventions as an opaque arm and pays $O(\sqrt{N/T})$ simple regret. But interventions on a known causal Bayesian network produce correlated feedback: pulling one arm and observing all variables tells you about the reward of other arms too. The paper formalizes this "causal feedback" and gives an algorithm whose simple-regret bound is strictly better in all quantities than structure-blind methods.

**Formal setup & key definitions:** Model class is a **causal Bayesian network**: a DAG $\mathcal{G}$ over $\mathcal{X}=\{X_1,\dots,X_n\}$ with joint $P$ factorizing over $\mathcal{G}$. An **intervention/action** is $do(\mathbf{X}=\mathbf{x})$ (atomic/hard interventions on a subset, removing incoming edges to that subset). Reward $Y\in\{0,1\}$; action set $\mathcal{A}$; expected reward $\mu_a := E[Y\mid do(\mathbf{X}=\mathbf{x})]$, optimum $\mu^* := \max_{a\in\mathcal{A}}\mu_a$. The objective is **simple regret** (best-arm-after-budget), not cumulative. The **parallel bandit** is the canonical instance: $N$ binary variables are independent causes of $Y$; un-intervened $X_i\sim\text{Bernoulli}(q_i)$ with $\mathbf{q}=(q_1,\dots,q_N)$; allowed actions are the null/observe action $do()$ and all size-1 interventions $do(X_i=j)$, so $|\mathcal{A}|=2N+1$.

**Key equations:**
- $R_T = \mu^* - E[\mu_{\hat{a}^*_T}]$ — simple regret; $\hat{a}^*_T$ is the recommended arm after $T$ rounds.
- $I_\tau = \{\, i : \min\{q_i,\,1-q_i\} < 1/\tau \,\}$ and $m(\mathbf{q}) = \min\{\,\tau : |I_\tau| \le \tau\,\}$ — the difficulty constant: the smallest $\tau$ for which at most $\tau$ variables are "rare" (tail probability below $1/\tau$).
- $\hat{\mu}_a = \frac{1}{T}\sum_{t=1}^{T} Y_t\, R_a(X_t)\, \mathbb{1}\{R_a(X_t) \le B_a\}$ with $R_a(X) = \dfrac{P\{\mathbf{Pa}_Y(X)\mid a\}}{Q\{\mathbf{Pa}_Y(X)\}}$ — clipped/truncated **importance-weighted estimator** of $\mu_a$; $R_a$ is the likelihood ratio between the action-$a$ distribution over $Y$'s parents and the sampling mixture $Q$.
- $m(\eta) = \max_{a\in\mathcal{A}} E_a\!\big[\,P\{\mathbf{Pa}_Y\mid a\}/Q\{\mathbf{Pa}_Y\}\,\big]$ — general-graph difficulty: worst-case expected importance weight under sampling distribution $\eta$ (a $\chi^2$-type divergence); the algorithm uses $\eta^*=\arg\min_\eta m(\eta)$.
- $B_a = \sqrt{m(\eta)\,T / \log(2T|\mathcal{A}|)}$ — the per-arm clipping threshold balancing bias against variance.

**Main results / theorems:** **Theorem 1 (parallel-bandit upper bound):** Algorithm 1 satisfies $R_T \in O\!\big(\sqrt{m(\mathbf{q})/T}\,\log(NT/m)\big)$. **Theorem 2 (matching lower bound):** for all $T,\mathbf{q}$ and all strategies there exists a reward function with $R_T \in \Omega(\sqrt{m(\mathbf{q})/T})$ — so $m(\mathbf{q})$ is the right complexity measure (tight up to log factors). **Theorem 3 (general graph):** Algorithm 2 satisfies $R_T \in O\!\big(\sqrt{m(\eta)/T}\,\log(2T|\mathcal{A}|)\big)$. The constant ranges $2 \le m(\mathbf{q}) \le N$: at $\mathbf{q}=(\tfrac12,\dots,\tfrac12)$, $m(\mathbf{q})=2$ (regret $\approx O(\sqrt{2/T})$, independent of $N$); at $\mathbf{q}=(0,\dots,0)$, $m(\mathbf{q})=N$ (no improvement). Structure-blind best-arm-identification over the $\approx 2N$ arms necessarily pays $\Omega(\sqrt{N/T})$, so the causal algorithm is never worse and much better whenever $m(\mathbf{q})\ll N$.

**Algorithms:** **Algorithm 1 (parallel bandit)** splits the budget: Phase 1 (rounds $1\ldots T/2$) plays the null intervention $do()$ to *observe* the natural joint and estimate each $q_i$ and the conditional rewards (this is where information leaks across arms); Phase 2 (rounds $T/2{+}1\ldots T$) spends the rest *uniformly on the "rare" arms* — those with $\hat{P}\{X_i=j\}\le 1/\hat{m}$. **Algorithm 2 (general graph)** plays a fixed design $\eta^*$ minimizing $m(\eta)$, observes $\mathbf{Pa}_Y$, and forms the truncated importance-weighted estimate $\hat{\mu}_a$ for *every* action from the *same* samples, then recommends $\arg\max_a\hat{\mu}_a$. The causal structure enters through $\mathbf{Pa}_Y$: knowing $Y$'s parents lets one importance-reweight a single sample into an unbiased estimate of any intervention's reward.

**Relevance to pgmpy:** Requires a `DiscreteBayesianNetwork`/CBN with a designated binary reward node $Y$, the do-operator for atomic interventions, identification of $Y$'s parent set $\mathbf{Pa}_Y$, ability to sample from interventional distributions, and an importance-weighting estimator over $\mathbf{Pa}_Y$ distributions. New components: a simple-regret online loop, the difficulty constants $m(\mathbf{q})$/$m(\eta)$ as graph-derived quantities, and a design-optimization step ($\min_\eta m(\eta)$). pgmpy already supplies the graph, do-operator, and simulation; the bandit scheduler and IS estimator are the gaps.

**Relation to causal incentives / CIDs:** Each arm $do(X_i=j)$ is an intervention = a setting of a decision node; $Y$ is the utility node; the $X_i$ are chance nodes. The parallel bandit is a CID with a single multi-valued decision feeding (independently) into the utility. The information-leakage mechanism is exactly **value of information / value of observation**: the observe phase ($do()$) is a no-cost observation that updates beliefs about *all* counterfactual utilities simultaneously, and $m(\mathbf{q})$ quantifies how much structure reduces the effective decision space. Shared primitives: typed DAG, do-operator, expected-utility $E[Y\mid do(\cdot)]$, and importance-reweighting to evaluate non-played decisions — the same machinery used to compute expected utility of alternative policies in an influence diagram.

---

### Identifying Best Interventions through Online Importance Sampling (ICML, 2017)
**Authors:** Rajat Sen, Karthikeyan Shanmugam, Alexandros G. Dimakis, Sanjay Shakkottai
**Link:** https://arxiv.org/abs/1701.02789
**TL;DR:** For best-arm identification among $K$ *soft* interventions at a single source node $V$, a successive-elimination algorithm with clipped importance sampling reuses every sample across all arms, giving gap-dependent error/simple-regret bounds governed by a divergence-based hardness $\bar{H}$ that can be far smaller than the structure-blind $K$-arm complexity.

**Problem & motivation:** Building on Lattimore et al., the paper studies identifying the best of $K$ *soft* interventions at a source node $V$ in an acyclic causal graph, to maximize a downstream target $Y$, under a fixed sample budget $T$ and optional per-intervention *cost* budget $B$. Because all arms differ only in the conditional $P(V\mid pa(V))$, samples from one arm can be importance-reweighted to estimate any other arm — the explicit "information leakage" the algorithm exploits.

**Formal setup & key definitions:** Model class is an acyclic causal DAG $\mathcal{G}(\mathcal{V},\mathcal{E})$. A **soft intervention** $k$ replaces the source-node conditional with $P_k(V\mid pa(V))$, leaving all other mechanisms fixed (generalizing Lattimore's hard interventions). The $K$ arms are these soft interventions; target $Y$ downstream of $V$; arm mean $\mu_k = E_k[Y]$, optimum $k^*$, gaps $\Delta_k=\mu^*-\mu_k$. Objective: **best-arm identification / simple regret** in the fixed-budget setting.

**Key equations:**
- $e(T,B) = \mathbb{P}\big(\hat{k}(T,B)\neq k^*\big)$ — error probability; $r(T,B) = \sum_{k\neq k^*}\Delta_k\,\mathbb{P}(\hat{k}(T,B)=k)$ — **simple regret**.
- $E_i[Y] = E_j\!\big[\,Y\cdot P_i(V\mid pa(V))/P_j(V\mid pa(V))\,\big]$ — Eq. (1): the importance-sampling identity letting arm-$j$ samples estimate arm-$i$'s mean (information leakage).
- $\hat{Y}_k^{\epsilon} = \frac{1}{Z_k}\sum_j\sum_{s\in\mathcal{T}_j}\frac{1}{M_{kj}}\,Y_j(s)\,\frac{P_k(V_j(s)\mid pa(V)_j(s))}{P_j(V_j(s)\mid pa(V)_j(s))}\,\mathbb{1}\{\text{ratio}\le 2\log(2/\epsilon)M_{kj}\}$ — Eq. (3): the **clipped IS estimator** pooling all arms' samples.
- $M_{ij} = 1 + \log\!\big(1 + D_{f_1}(P_i\|P_j)\big)$, $f_1(x)=x\,e^{x-1}-1$ — Def. 2: the log-$f$-divergence between arm distributions, quantifying how informative arm $j$'s samples are about arm $i$.
- $\bar{H} = \max_{k\neq k^*}\log_2(10/\Delta_k)^3\big(\sigma^*(B,\mathcal{R}^*(\Delta_k))/\Delta_k\big)^2$ — Eq. (6): the divergence-based **hardness**; $\sigma^*(B,\mathcal{R})$ is an "effective standard deviation" from a budget/cost optimization.

**Main results / theorems:** **Theorem 1 (error probability):** for $\Delta\ge 10/\sqrt{T}$, $e(T,B)\le 2K^2\log_2(20/\Delta)\exp\!\big(-T/(2\bar{H}\,\overline{\log}(n(T)))\big)$. **Theorem 2 (simple regret):** $r(T,B)\le \frac{10}{\sqrt{T}}\,\mathbb{1}\{\exists k:\Delta_k<10/\sqrt{T}\} + 2K^2\sum_{k\neq k^*:\,\Delta_k\ge 10/\sqrt T}\Delta_k\log_2(20/\Delta_k)\exp\!\big(-T/(2\bar{H}_k\,\overline{\log}(n(T)))\big)$. The improvement over structure-blind best-arm-identification (hardness $\tilde{H}=\max_k |\tilde{\mathcal{R}}(\Delta_k)|/\Delta_k^2$, scaling with the raw arm count) is that the effective $\sigma^*$ can be much smaller than $\sqrt{|\tilde{\mathcal{R}}(\Delta_k)|}$ — "exponentially better" when cost constraints restrict subsets of arms — because IS reuse decouples sample cost from arm count.

**Algorithms:** **SRIS (Successive Rejects with Importance Sampling)** (SRISv1 / SRISv2). It runs in $n(T)$ phases, halving the bias clip per phase ($\epsilon=2^{-(\ell-1)}$); each phase (1) allocates samples across surviving arms via a budget/cost optimizer minimizing the effective variance $\sigma^*(B,\mathcal{R})$, (2) collects samples, (3) re-estimates *every* surviving arm's mean from the *pooled* samples via the clipped IS estimator $\hat{Y}_k^\epsilon$, and (4) eliminates arms significantly below the best. Causal structure is used through Eq. (1): all arms share the same downstream mechanism, so one sample informs all arms weighted by $M_{kj}$.

**Relevance to pgmpy:** Needs a CBN with a designated source/decision node $V$ whose CPD can be *replaced* (soft intervention = swap $P(V\mid pa(V))$, distinct from a hard `do`), a downstream reward $Y$, evaluation of likelihood ratios $P_k(V\mid pa(V))/P_j(V\mid pa(V))$ from CPDs, and a clipped importance-sampling estimator. New components: soft-intervention API (CPD override), the divergence $M_{ij}$ from CPDs, a phased successive-elimination loop, and a cost/budget allocation optimizer. pgmpy's CPD machinery and sampling are reusable; soft-intervention support and the IS/elimination loop are gaps.

**Relation to causal incentives / CIDs:** A soft intervention at $V$ = choosing the *CPD/policy* of a decision node $V$; $Y$ is the utility node; this is essentially selecting among candidate decision rules. The IS-reuse identity is a **value-of-control** computation: each candidate policy's expected utility $E_k[Y]$ is evaluated by reweighting a common sample pool — the same off-policy evaluation used to compare decision-rule values in a CID. The divergence $M_{ij}$ measures how transferable evidence is between policies. Shared primitives: typed graph with a controllable decision node, expected-utility evaluation $E_k[Y]$, and off-policy/importance reweighting to score un-deployed decisions.

---

### Causal Bandits with Propagating Inference (ICML, 2018)
**Authors:** Akihiro Yabe, Daisuke Hatano, Hanna Sumita, Shinji Ito, Naonori Kakimura, Takuro Fukunaga, Ken-ichi Kawarabayashi
**Link:** https://arxiv.org/abs/1806.02252 (PMLR v80)
**TL;DR:** For best-arm identification over arbitrary *non-atomic* interventions on a known causal DAG, a "propagating inference" estimator that propagates marginal-probability estimates through the topological order achieves simple regret $O(\sqrt{\max\{\gamma^*,N\}\log(|\mathcal{A}|T)/T})$, with $\gamma^*=O(N^2)$ for bounded in-degree — beating the structure-blind $\Omega(\sqrt{|\mathcal{A}|/T})$ whenever $\gamma^*\ll|\mathcal{A}|$.

**Problem & motivation:** Lattimore et al. handle only "localized" interventions whose effect on $Y$ factors simply; Yabe et al. remove that restriction, allowing arbitrary intervention sets where an intervention's effect *propagates* through the whole graph to the target. The challenge is to estimate every arm's reward from shared samples when arms touch overlapping subsets of nodes, and to bound regret by a graph-structural constant rather than the (possibly exponential) number of arms $|\mathcal{A}|$.

**Formal setup & key definitions:** Model class is a DAG $G=(\mathcal{V},E)$ with $N$ binary nodes $V_n\in\{0,1\}$ with native parameters $\alpha_n(\pi):=\mathrm{Prob}(V_n=\pi_n\mid V_i=\pi_i\ \forall i\in\mathcal{P}_n,\ v_n\text{ not intervened})$. A **non-atomic intervention** is a vector $A\in\mathcal{A}\subseteq\{0,1,*\}^N$: $A_n\in\{0,1\}$ fixes $V_n$, $A_n=*$ leaves it free (so a single arm can fix *many* variables at once). Target is node $N$; reward $\mu(A):=\mathrm{Prob}(V_N=1\mid do(A))$. Objective: **simple regret** $R_T=\mu(A^*)-E[\mu(\hat{A})]$.

**Key equations:**
- $\beta_n(\pi,A) := \mathrm{Prob}\big(V_m=\pi_m,\ \forall m\in\mathcal{P}_n \mid do(A)\big)$ if $A_n=*$, else $0$ — the propagated marginal: probability that node $n$'s parent configuration is $\pi$ under intervention $A$; the central quantity the estimator propagates.
- $\hat{\mu}(A) = \sum_{\pi\in B(A)}\ \prod_{n\in \text{In}_A}\hat{\alpha}_n(\pi_{\mathcal{P}_n})$ — Eq. (12): the **propagating-inference reward estimator**, a product over non-intervened nodes of estimated native parameters, summed over consistent configurations.
- $\gamma^* := \min_{\eta\in[0,1]^{\mathcal{A}}}\ \max_{A\in\mathcal{A}}\ \sum_{n=1}^{N}\ \sum_{\pi:\,\beta_n(\pi,A)>0}\dfrac{\beta_n^2(\pi,A)}{\sum_{A'\in\mathcal{A}}\eta_{A'}\,\beta_n(\pi,A')}\quad\text{s.t.}\ \sum_{A'}\eta_{A'}=1$ — Eq. (13): the **complexity constant**, a min-max over sampling design $\eta$ of an importance-weight variance summed over nodes and parent configurations.

**Main results / theorems:** **Theorem 1 (upper bound):** Algorithm 3 satisfies $R_T \le O\!\big(\sqrt{\max\{\gamma^*,N\}\,\log(|\mathcal{A}|T)/T}\big)$. **Proposition 2 (range of $\gamma^*$):** $N-\min_{A\in\mathcal{A}}|A| \le \gamma^* \le \min\{NC,\ N|\mathcal{A}|\}$ (with $C$ a graph/in-degree constant); for **bounded in-degree** graphs $\gamma^*=O(N^2)$, giving $R_T\le O(\sqrt{N^2\log(|\mathcal{A}|T)/T})$. This beats the structure-blind lower bound $\Omega(\sqrt{|\mathcal{A}|/T})$ precisely when $\gamma^*\ll|\mathcal{A}|$ — e.g. an arm set exponential in $N$ but a sparse graph gives regret polynomial in $N$.

**Algorithms:** A three-stage procedure. **Algorithm 1** uses $T/3$ rounds to estimate the propagated marginals $\hat\beta_n(\pi,A)$ by substituting current $\check\alpha_m$ and propagating through the topological order. **Algorithm 2** uses the remaining $2T/3$ rounds, solving a design problem for the variance-minimizing $\hat\eta$, to estimate native parameters $\hat\alpha_n$ via importance sampling. **Algorithm 3** assembles $\hat{\mu}(A)$ for every $A\in\mathcal{A}$ via Eq. (12) and returns $\arg\max_{A}\hat{\mu}(A)$. "Propagating inference" = estimates flow root-to-target through the DAG so one set of interventional samples feeds *all* arms' reward estimates, even when arms overlap arbitrarily.

**Relevance to pgmpy:** Needs a binary-variable CBN, multi-variable (non-atomic) interventions $A\in\{0,1,*\}^N$ (i.e. `do` on arbitrary node subsets), a topological-order propagation of marginal/native parameters ($\alpha_n,\beta_n$), and an importance-sampling design optimizer ($\gamma^*$). The reward estimator $\hat\mu(A)$ is essentially interventional marginal inference $P(V_N=1\mid do(A))$, which pgmpy's variable elimination + do-operator already support; gaps are the non-atomic arm enumeration, the design/$\gamma^*$ optimization, and the simple-regret loop.

**Relation to causal incentives / CIDs:** A non-atomic intervention $A\in\{0,1,*\}^N$ is a *joint* decision over a set of decision nodes (a partial assignment to many variables at once); $V_N$ is the utility node; the others are chance nodes. Computing $\mu(A)=P(V_N=1\mid do(A))$ by topological propagation is exactly **expected-utility computation via belief propagation in an influence diagram**. $\gamma^*$ is a structural measure of how much information is shared across decisions, analogous to how a CID's graph determines which observations/controls are non-redundant. Shared primitives: typed DAG, do-operator on node subsets, expected-utility evaluation by inference, and optimization of a decision (arm) to maximize utility — value of control over multiple decision variables jointly.

## B. Where to Intervene — Structural Causal Bandits (SCM-MAB)

### Structural Causal Bandits: Where to Intervene? (NeurIPS, 2018)
**Authors:** Sanghack Lee, Elias Bareinboim (Purdue University)
**Link:** https://par.nsf.gov/servlets/purl/10111018 (code: https://github.com/sanghack81/SCMMAB-NIPS2018)
**TL;DR:** Formalizes the bandit-over-interventions problem on a causal graph (SCM-MAB), and shows that only a graphically identifiable subset of intervention sets — the POMISs (possibly-optimal minimal intervention sets) — can ever be optimal, so a MAB solver should restrict its arms to POMIS arms.

**Problem & motivation:** In a MAB whose arms are interventions $do(\mathbf{x})$ on an SCM, the arms' reward distributions are coupled through the causal structure, so the two "obvious" strategies — intervening on all variables at once ($do(\mathbf{V}\setminus\{Y\})$) or trying all subsets (brute force) — are respectively never-optimal-in-general (linear regret under unobserved confounders) and wastefully large. The paper asks *where* an agent should intervene given only the causal graph (not its parametrization), and characterizes the complete, sound, minimal set of "qualified" arms.

**Formal setup & key definitions:**
- SCM $M=\langle \mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})\rangle$: $\mathbf{U}$ exogenous/unobserved, $\mathbf{V}$ endogenous, $\mathbf{F}=\{f_i\}$ with $V_i\leftarrow f_i(\mathbf{pa}_i,\mathbf{u}^i)$. Reward node $Y\in\mathbf{V}$. A causal diagram $G=\langle\mathbf{V},\mathbf{E}\rangle$ has a directed edge $V_i\to V_j$ if $V_i\in\mathbf{PA}_j$ and a bidirected edge $V_i\leftrightarrow V_j$ if they share an unobserved confounder.
- **SCM-MAB** $\langle M,Y\rangle$: arms are all interventions $\{\mathbf{x}\in D(\mathbf{X})\mid \mathbf{X}\subseteq \mathbf{V}\setminus\{Y\}\}$; arm $\mathbf{x}$ has reward $\mu_{\mathbf{x}}=\mathbb{E}[Y\mid do(\mathbf{x})]$. The agent knows the graph $G$ but not $\mathbf{F},P(\mathbf{U})$; information available is denoted $[\![G,Y]\!]$. Cumulative regret $\mathrm{Reg}_T = T\mu^\ast-\sum_{t=1}^T \mathbb{E}[Y_{A_t}]=\sum_{a=1}^K \Delta_a\,\mathbb{E}[T_a(T)]$.
- **c-component / confounded component:** $\mathrm{CC}(X)_G$ is the maximal set of vertices reachable from $X$ via bidirected edges.
- **Minimal Intervention Set (MIS) — Def 1:** $\mathbf{X}$ is an MIS relative to $[\![G,Y]\!]$ if no $\mathbf{X}'\subset\mathbf{X}$ has $\mu_{\mathbf{x}[\mathbf{X}']}=\mu_{\mathbf{x}}$ for every SCM conforming to $G$.
- **POMIS — Def 2:** an MIS $\mathbf{X}$ is a POMIS if there exists an SCM conforming to $G$ such that $\mu_{\mathbf{x}^\ast} > \mu_{\mathbf{z}^\ast}\ \forall\, \mathbf{Z}\in\mathbb{Z}\setminus\{\mathbf{X}\}$, where $\mathbb{Z}$ is the set of all MISs. (Some parametrization makes $\mathbf{X}$'s best arm strictly beat every other MIS's best arm.)

**Key equations / criteria:**
- **Arm equivalence (Property 1, do-calculus Rule 3):** $\mu_{\mathbf{x},\mathbf{z}}=\mu_{\mathbf{x}}$ whenever $Y\perp\!\!\!\perp \mathbf{Z}\mid \mathbf{X}$ in $G_{\overline{\mathbf{X}},\underline{\mathbf{Z}}}$ — only one arm per equivalence class need be played.
- **Minimality (Prop 1):** $\mathbf{X}$ is an MIS iff $\mathbf{X}\subseteq an(Y)_{G_{\overline{\mathbf{X}}}}$ (every intervened variable is an ancestor of $Y$ after cutting incoming edges to $\mathbf{X}$).
- **Markovian / no-confounding case (Prop 2, Cor 3):** if $Y$ is not confounded with $an(Y)_G$ (in particular, if $G$ is Markovian), then $pa(Y)_G$ is the *only* POMIS — intervene on $Y$'s direct causes.
- **UC-territory (Def 3):** with $H=G[An(Y)_G]$, $\mathbf{T}\ni Y$ is a UC-territory if descendant-closed and c-component-closed in $H$: $De(\mathbf{T})_H=\mathbf{T}$ and $\mathrm{CC}(\mathbf{T})_H=\mathbf{T}$.
- **MUCT (minimal UC-territory):** the $\subseteq$-minimal UC-territory, built from $\{Y\}$ by alternately taking c-component and descendants until convergence.
- **Interventional Border (IB) — Def 4:** for the MUCT $\mathbf{T}$, $\mathbf{X}=pa(\mathbf{T})_G\setminus\mathbf{T}$. Intervening *inside* the MUCT destroys reward, so you intervene exactly on its border.

**Main results / theorems:**
- **Prop 5:** $\mathrm{IB}(G_{\overline{\mathbf{W}}},Y)$ is a POMIS for *any* $\mathbf{W}\subseteq\mathbf{V}\setminus\{Y\}$; $pa(Y)_G$ is always a POMIS.
- **Theorem 6 (central characterization, sound & complete):** Given $[\![G,Y]\!]$, $\mathbf{X}\subseteq\mathbf{V}\setminus\{Y\}$ is a **POMIS if and only if** $\mathrm{IB}(G_{\overline{\mathbf{X}}},Y)=\mathbf{X}$. Every non-POMIS arm-set is dominated and can be discarded.
- **Theorem 9 (soundness & completeness of the algorithm):** Algorithm `POMISs` returns *all and only* POMISs.
- **Regret:** running kl-UCB over the POMIS arm set gives $\limsup_{n\to\infty}\frac{\mathbb{E}[\mathrm{Reg}_n]}{\log n}\le \sum_{\mathbf{x}:\mu_{\mathbf{x}}<\mu^\ast}\frac{\mu^\ast-\mu_{\mathbf{x}}}{\mathrm{KL}(\mu_{\mathbf{x}},\mu^\ast)}$; shrinking the arm set lowers this bound.

**Algorithms:**
- **`POMISs(G,Y)` (Alg 1):** compute $\mathbf{T},\mathbf{X}=\mathrm{MUCT}(G,Y),\mathrm{IB}(G,Y)$; $H=G_{\overline{\mathbf{X}}}[\mathbf{T}\cup\mathbf{X}]$; recurse via `subPOMISs` over subsets in reverse topological order, recomputing MUCT/IB after intervening on one more variable and pruning via a "do-not-intervene" set (avoids exponential naive cost).
- **`POMIS-kl-UCB` (Alg 2):** build arm set $\mathbf{A}=\bigcup_{\mathbf{X}\in\mathrm{POMISs}(G,Y)}D(\mathbf{X})$, then run standard kl-UCB (or Thompson sampling) on $\mathbf{A}$.
- **Experiments:** POMIS / MIS / Brute-force / All-at-once arm sets, e.g. Task 3: 16 / 75 / 243 / 32 arms. POMIS always converges fastest; All-at-once incurs linear regret under confounding; brute force slows exponentially.

**Relevance to pgmpy:** Requires, on a semi-Markovian causal graph / ADMG: (1) c-component computation over bidirected edges; (2) ancestral / descendant sets and induced/do-subgraphs $G[\cdot]$, $G_{\overline{\mathbf{X}}}$ (edge-cutting for interventions); (3) the MUCT fixpoint (alternate descendant-closure + c-component-closure from $\{Y\}$) and IB $=pa(\mathrm{MUCT})\setminus\mathrm{MUCT}$; (4) the recursive POMIS enumeration with topological-order pruning; (5) a thin bandit layer (kl-UCB / UCB / TS) over the enumerated arm set. pgmpy already has ADMG, c-component, ancestor/descendant, and do-graph machinery in `pgmpy/base/`, so this is largely a new graph-algorithm module plus a bandit loop.

**Relation to causal incentives / CIDs:** The MIS condition $\mathbf{X}\subseteq an(Y)$ is exactly the CID **instrumental-control-incentive / value-of-control** criterion: a decision is only worth pointing at a variable if that variable lies on a *directed path to the utility (reward) node* — non-ancestors of $Y$ have zero value of control and are pruned. The Markovian result "intervene on $pa(Y)$" is the degenerate CID case where the single decision should target the utility's direct causes. POMIS refines this under latent confounding: the MUCT identifies where confounding with $Y$ makes deeper intervention counterproductive, and the IB is the optimal "control frontier" — the *graph topology alone* (ancestry + c-component structure), not the parametrization, determines which targets a decision node can profitably control.

---

### Structural Causal Bandits with Non-manipulable Variables (AAAI, 2019)
**Authors:** Sanghack Lee, Elias Bareinboim (Purdue University)
**Link:** https://causalai.net/r40.pdf
**TL;DR:** Extends SCM-MAB / POMIS to the realistic case where some observed variables cannot be intervened on (non-manipulable $\mathbf{N}$), showing that POMISs under constraints equal the unconstrained POMISs of the *latent projection* onto manipulable variables, and adds a generalized z-identification (`z²ID`) procedure that lets one arm's reward be estimated from other arms' samples.

**Problem & motivation:** In practice many variables (e.g. cholesterol, obesity) are observable and causally relevant but cannot be directly set via $do(\cdot)$; only some variables are manipulable. The original POMIS theory assumed everything was manipulable. The paper relaxes this, re-characterizes which interventions are possibly-optimal under a non-manipulable set $\mathbf{N}$, and exploits the fact that pulling one arm reveals the full realization $\mathbf{v}\sim P_{\mathbf{x}}(\mathbf{v})$, so observational/front-door-style formulas let arms share information.

**Formal setup & key definitions:**
- $\mathbf{N}\subseteq\mathbf{V}\setminus\{Y\}$ is the set of **non-manipulable** variables. Intervention sets are restricted to $\mathbf{X}\subseteq\mathbf{V}\setminus\{Y\}\setminus\mathbf{N}$.
- **MIS / POMIS under constraints (Defs 2–3):** as before but restricted to $\mathbf{X}\cap\mathbf{N}=\emptyset$; constrained sets $\mathbb{M}^{\mathbf{N}}_{G,Y}$, $\mathbb{P}^{\mathbf{N}}_{G,Y}$. (MUCT and IB are reused unchanged.)
- **Latent projection $G_{[\mathbf{V}\setminus\mathbf{N}]}$:** the ADMG on the manipulable variables obtained by marginalizing $\mathbf{N}$: add $V_i\to V_j$ if there is a directed path through only $\mathbf{N}$-nodes, and $V_i\leftrightarrow V_j$ if a UC (or confounding path through $\mathbf{N}$) connects them.

**Key equations / criteria:**
- The constrained set is *not* simply the unconstrained POMISs avoiding $\mathbf{N}$ — equality fails in general (e.g. front-door graph $X\to Z\to Y$ with $X\leftrightarrow Y$: when $Z\in\mathbf{N}$, $\{X\}$ becomes a POMIS though it is not in the unconstrained case).
- **Front-door reuse:** $P_x(y)=\sum_z P(z\mid x)\sum_{x'}P(y\mid z,x')P(x')$ — arm $do(x)$'s reward is estimable both from $do(x)$ samples and via the front-door formula from observational ($do(\emptyset)$) samples.

**Main results / theorems:**
- **Theorem 4 (main):** $\mathbb{P}^{\mathbf{N}}_{G,Y}=\mathbb{P}_{H,Y}$ where $H=G_{[\mathbf{V}\setminus\mathbf{N}]}$. **The constrained POMISs are exactly the unconstrained POMISs of the latent projection.** So: project out $\mathbf{N}$, then run the 2018 `POMISs` algorithm on $H$.
- **Props 2–3:** the latent projection preserves all interventional distributions ($P^1_{\mathbf{x}}(\mathbf{y})=P^2_{\mathbf{x}}(\mathbf{y})$ for all $\mathbf{X},\mathbf{Y}\subseteq\mathbf{V}\setminus\mathbf{N}$).
- **Theorem 5 (soundness of `z²ID`):** whenever the generalized z-identification algorithm returns an expression for $P_{\mathbf{x}}(y)$, it is correct.

**Algorithms:**
- **`z²ID` (generalized z-identification):** given a target arm distribution $P_{\mathbf{x}}(y)$ and the set $\mathbb{Z}$ of available experiments, recursively reduces $P_{\mathbf{x}}(y)$ via c-component factorization and do-calculus to an expression in available distributions $P_{\mathbf{Z}}(\mathbf{v})$ (generalizing Tian–Pearl, Shpitser–Pearl, Bareinboim–Pearl identifiability). Returns multiple valid expressions, including ones using *other* arms' samples.
- **`bMVWA`:** bootstrap-based minimum-variance weighted average — combines the $m$ dependent estimators via a convex combination $\hat\theta=\sum w_i\hat\theta_i$, $\sum w_i=1$, minimizing $\mathrm{Var}(\hat\theta)=\mathbf{w}^\top\Sigma\mathbf{w}$ (covariance $\Sigma$ bootstrap-estimated).
- **`z²-TS` / `z²-KL-UCB` (POMIS+):** TS / kl-UCB on the POMIS arms, each arm's mean estimated with `bMVWA` over all `z²ID` expressions — further cutting cumulative regret (e.g. 34.5%/39.2% on the front-door graph vs plain POMIS).

**Relevance to pgmpy:** Adds two reusable pieces beyond the 2018 paper: (1) **latent projection of an ADMG onto a variable subset** (marginalizing $\mathbf{N}$ into directed + bidirected edges) — a generally useful graph operation pgmpy's ADMG should support. (2) A **z-identification / do-calculus reduction engine** producing estimable expressions from a set of available experimental + observational distributions — overlaps with pgmpy's existing `CausalInference` identification and c-component factorization. The bandit layer (`bMVWA` + TS/kl-UCB) sits on top.

**Relation to causal incentives / CIDs:** Non-manipulable variables are the CID analogue of **chance nodes that cannot be turned into decision nodes** — observable (information links) but not controllable. Theorem 4's projection says the "value-of-control" structure is invariant under marginalizing uncontrollable nodes: a manipulable variable's control incentive must be assessed on the *projected* graph where paths through non-manipulable nodes are summarized as direct/bidirected edges. The `z²ID` machinery is "what can I learn about a decision's payoff from other policies/observations" — value-of-information reasoning across the experiment set.

---

### Characterizing Optimal Mixed Policies: Where to Intervene and What to Observe (NeurIPS, 2020)
**Authors:** Sanghack Lee, Elias Bareinboim (Columbia University)
**Link:** https://causalai.net/r63.pdf (proceedings: https://proceedings.neurips.cc/paper/2020/hash/61a10e6abb1149ad9d08f303267f9bc4-Abstract.html)
**TL;DR:** Generalizes POMIS from "which variables to intervene on" to *mixed policies* — jointly choosing the set of variables to intervene on **and** the contexts to observe for each — and characterizes the non-redundant and possibly-optimal **mixed policy scopes (POMPS)** graphically, proving that the standard "intervene on everything, observe everything" contextual-bandit policy can be strictly suboptimal.

**Problem & motivation:** A contextual bandit fixes its scope a priori: observe all contexts, act on all action variables. The paper argues a causal agent should *choose* its scope — which variables to set ($do$) and which to condition each action on (observe) — because, under unobserved confounders, observing/acting on too much can be strictly harmful (you can "never achieve optimal performance"). It characterizes the space of scopes, their redundancy, and their possible optimality.

**Formal setup & key definitions:**
- SCM $M=\langle\mathbf{U},\mathbf{V},P(\mathbf{U}),\mathbf{F}\rangle$, graph $G$; reward $Y$, **intervenable** variables $\mathbf{X}^\star\subseteq\mathbf{V}\setminus\{Y\}$, **contextualizable** variables $\mathbf{C}^\star\subseteq\mathbf{V}\setminus\{Y\}$.
- **Mixed Policy Scope (MPS) — Def 1:** a collection of pairs $\mathcal{S}=\{\langle X,\mathbf{C}_X\rangle\}$ such that (i) $X\in\mathbf{X}^\star$, $\mathbf{C}_X\subseteq\mathbf{C}^\star\setminus\{X\}$, and (ii) the induced graph $G_{\mathcal{S}}$ (cut edges *into* each $X$, add $\mathbf{C}_X\to X$) is acyclic. The MPS says **where to intervene** ($\mathbf{X}(\mathcal{S})$) and **what to observe** for each action ($\mathbf{C}_X$).
- **Mixed Policy — Def 2:** a realization $\boldsymbol{\pi}=\{\pi_{X\mid\mathbf{C}_X}\}$ inducing a submodel $M_{\boldsymbol{\pi}}$ replacing $f_X$ by $\pi_{X\mid\mathbf{C}_X}$. Standard CB scope $\mathcal{S}_{\mathrm{CB}}=\{\langle X_1,\{C\}\rangle,\langle X_2,\{X_1,C\}\rangle\}$ — and the running example shows $\mu^\ast_{\mathcal{S}_{\mathrm{CB}}}<\mu^\ast$ (full-scope CB is suboptimal).
- **Expected reward (Eq 1):** with non-action contexts $\mathbf{C}^-=\mathbf{C}(\boldsymbol{\pi})\setminus\mathbf{X}(\boldsymbol{\pi})$, $\mu_{\boldsymbol{\pi}}=\sum_{y,\mathbf{x},\mathbf{c}^-} y\,P_{\mathbf{x}}(y,\mathbf{c}^-)\prod_{X\in\mathbf{X}(\boldsymbol{\pi})}\pi(x\mid\mathbf{c}_x)$ — separating the **atomic interventional factor** $P_{\mathbf{x}}(y,\mathbf{c}^-)$ (fixed by the world) from the **policy factor** (optimizable).

**Main results / theorems:**
- **Theorem 1 (graphical characterization of non-redundancy):** with $H=G_{\mathcal{S}}$, $\mathcal{S}$ is non-redundant **iff** for every $X\in\mathbf{X}$ and $C\in\mathbf{C}_X$: (i) $X\in an(Y)_H$ — **the action must be an ancestor of the reward** (do-calculus Rule 3); and (ii) $C\not\perp\!\!\!\perp Y\mid \mathbf{C}_X\setminus\{C\}$ in $H\setminus\{X\}$ — **the observed context must be relevant** (do-calculus Rule 2).
- **Def 5 (Possibly-Optimal MPS — POMPS):** within the set $\mathbb{S}$ of non-redundant-under-optimality (NRO) MPSes, $\mathcal{S}$ is possibly-optimal if some SCM gives $\mu^\ast_{\mathcal{S}}>\max_{\mathcal{S}'\in\mathbb{S}\setminus\{\mathcal{S}\}}\mu^\ast_{\mathcal{S}'}$ — POMPSes are the **maximal elements** of the partial order over NRO scopes.
- **Improvement operations:** Prop 4 (add an observation) improves $\mathcal{S}$ if $C\notin de(X)_{G_{\mathcal{S}}}$ and $C\not\perp\!\!\!\perp Y\mid[\mathbf{C}_X]$ in $H\setminus\{X\}$; adding an intervention improves if $Y\not\perp\!\!\!\perp X\mid[\mathbf{Z}]$ in $H_{\overline{X}}$ and $X\notin an(\mathbf{Z})_H$.
- **Prop 5 (pruning):** a variable with no directed path to a context or to $Y$ is *not intervene-worthy*. (Caveat: a descendant of $Y$ can still serve as a useful *context* for another action.)
- **Key qualitative result:** the standard CB approach (intervene on all intervenable variables, observe all contexts) "may be hurting itself and will never achieve optimal performance."

**Algorithms:** No closed-form bandit loop; the constructive content is the **enumeration/refinement of the MPS space**: enumerate MPSes, filter to non-redundant (Thm 1) and NRO, keep the maximal POMPSes via the improvement operations (Prop 4) and intervene-worthiness pruning (Prop 5). POMPSes form the reduced "scope arm set" a downstream RL/bandit learner should search over.

**Relevance to pgmpy:** The MPS *is* an influence-diagram/CID structure expressed on a causal graph: $\mathbf{X}^\star$ are decision nodes, $\mathbf{C}_X$ are the information links into decision $X$, and $G_{\mathcal{S}}$ is the standard "policy-induced graph." pgmpy would need: building $G_{\mathcal{S}}$ from a scope (reuse do-graph edge cutting + edge addition), ancestor sets, d-separation queries (already in `pgmpy/base/`), the non-redundancy tests (Thm 1's two d-separation conditions), and an enumeration/partial-order routine over scopes producing POMPSes. This is the natural superset of the POMIS module (POMIS = the special case $\mathbf{C}^\star=\emptyset$, observe nothing).

**Relation to causal incentives / CIDs:** This paper is the tightest fit to the CID framework. Theorem 1(i) $X\in an(Y)$ is precisely the **control-incentive / value-of-control** criterion: a decision node materially affects utility iff it lies on a directed path to the utility node. Theorem 1(ii), the context-relevance condition $C\not\perp\!\!\!\perp Y\mid \mathbf{C}_X\setminus\{C\}$ in $H\setminus\{X\}$, is exactly the **value-of-information / materiality (requisite observation)** criterion: an information link into a decision is non-redundant iff there is an active path from the observed variable to the utility given the other observations. POMPS = the set of CID *information-and-decision structures* that could be optimal — directly answering, in CID terms, **which variable a decision should target** (control incentive) and **which observations should feed the decision** (positive value of information). The "observing everything can hurt" result corresponds to the CID subtlety that, under unobserved confounding, adding non-requisite information links can change (and degrade) the attainable optimum rather than being harmlessly ignorable.

## C. Unknown Graph & Structure Discovery

### Regret Analysis of Bandit Problems with Causal Background Knowledge (UAI, 2020)
**Authors:** Yangyi Lu, Amirhossein Meisami, Ambuj Tewari, William Yan
**Link:** https://arxiv.org/abs/1910.04938 (UAI 2020, PMLR vol. 124)
**TL;DR:** Given a known causal graph plus the conditional distributions of the reward's parents under each intervention, C-UCB and C-TS attain cumulative regret that scales with the number of distinct parent-configurations $k^n$ rather than the (exponentially larger) number of arms.

**Problem & motivation:** Standard MAB ignores that many arms (interventions) induce the same or related reward distributions through a shared causal mechanism. If the analyst knows the causal graph and how each intervention shifts the distribution of the reward's parents, arms can share statistical strength. The paper formalizes this "causal background knowledge" and shows it provably shrinks the effective problem size in the cumulative-regret setting.

**Formal setup & key definitions:** Causal DAG $\mathcal{G}$ over $\mathcal{X}=\{X_1,\dots,X_N\}$, each variable taking $k$ values. An action/arm is an intervention $a=\mathrm{do}(\mathbf{X}=\mathbf{x})$, $\mathbf{X}\subset\mathcal{X}$. Reward $Y$ depends only on its parents $\mathrm{Pa}_Y$ ($|\mathrm{Pa}_Y|=n$). **Crucially, the graph $\mathcal{G}$, the action set $\mathcal{A}$, and the conditional distributions $P(\mathrm{Pa}_Y\mid a)$ are all assumed KNOWN**; only the conditional reward means $\mu_{\mathbf{Z}_j}=\mathbb{E}[Y\mid\mathrm{Pa}_Y=\mathbf{Z}_j]$ are learned. After pulling $a_t$ the learner observes both $Y_t$ and the realized parent values $\mathbf{Z}_{(t)}$. Objective is CUMULATIVE regret $R_T=T\mu_{a^*}-\sum_{t=1}^T\mu_{a_t}$ (Bayesian regret $BR_T$ for the TS variant). This is a known-graph paper: no structure discovery.

**Key equations:**
- Reward decomposition (the mechanism that couples arms): $\mu_a=\sum_{j=1}^{k^n}\mathbb{E}[Y\mid\mathrm{Pa}_Y=\mathbf{Z}_j]\,P(\mathrm{Pa}_Y=\mathbf{Z}_j\mid a)$, where $\mathbf{Z}_1,\dots,\mathbf{Z}_{k^n}$ enumerate all parent configurations.
- C-UCB index per parent-configuration: $\mathrm{UCB}_{\mathbf{Z}_j}(t-1)=\hat\mu_{\mathbf{Z}_j}(t-1)+\sqrt{\dfrac{2\log(1/\delta)}{1\vee T_{\mathbf{Z}_j}(t-1)}}$.
- C-UCB arm choice: $a_t=\arg\max_{a\in\mathcal{A}}\sum_{j=1}^{k^n}\mathrm{UCB}_{\mathbf{Z}_j}(t-1)\,P(\mathrm{Pa}_Y=\mathbf{Z}_j\mid a)$.
- C-UCB regret (Thm 1): $\mathbb{E}[R_T]=\tilde O(\sqrt{k^n T})$.
- C-TS Bayesian regret (Thm 2): $BR_T=\tilde O(\sqrt{k^n T})$.
- Standard (non-causal) UCB/TS on the same problem: $\tilde O(\sqrt{(k+1)^N T})$ (i.e. $\tilde O(\sqrt{|\mathcal{A}|\,T})$). The improvement factor is $k^n$ vs $(k+1)^N$, large when $n=|\mathrm{Pa}_Y|\ll N$.
- Linear extension (CL-UCB/CL-TS): reward model $Y\mid\mathrm{Pa}_Y=\mathbf{Z}=f(\mathbf{Z})^\top\theta+\epsilon$, with action feature $\mathbf{m}_a=\sum_{j=1}^{k^n}f(\mathbf{Z}_j)P(\mathrm{Pa}_Y=\mathbf{Z}_j\mid a)$; regret (Thm 3): $\tilde O(d\sqrt{T})$, depending only on the coefficient dimension $d$ (not on $k^n$).

**Main results / theorems:** The dependence on the number of arms is removed entirely; regret depends on $k^n=|\mathrm{dom}(\mathrm{Pa}_Y)|$, the number of *unique reward distributions* over the reward's parents. Both the frequentist UCB bound and the Bayesian TS bound are $\tilde O(\sqrt{k^n T})$. The bounds are worst-case ($\sqrt{T}$-type); the paper does **not** give an instance/gap-dependent $\sum_i\log T/\Delta_i$ bound. Empirically the causal algorithms beat standard MAB by roughly a factor of three within a few hundred rounds.

**Algorithms:**
- *C-UCB:* maintain an empirical mean $\hat\mu_{\mathbf{Z}_j}$ and count $T_{\mathbf{Z}_j}$ for each of the $k^n$ parent configurations; each round form the UCB index per configuration and pick the arm maximizing the known-weighted sum $\sum_j\mathrm{UCB}_{\mathbf{Z}_j}P(\mathbf{Z}_j\mid a)$; update only the configurations observed.
- *C-TS:* keep a posterior on each $\theta_j=\mathbb{E}[Y\mid\mathrm{Pa}_Y=\mathbf{Z}_j]$ (e.g. Beta/Gaussian); sample $\hat\theta_j(t)$, compute $\hat\mu_a=\sum_j\hat\theta_j(t)P(\mathbf{Z}_j\mid a)$, pull $\arg\max_a\hat\mu_a$, update posteriors from $(Y_t,\mathbf{Z}_{(t)})$.
- *CL-UCB/CL-TS:* run linear-bandit UCB/TS on the induced features $\mathbf{m}_a$.

**Relevance to pgmpy:** pgmpy already supplies everything on the *known-graph* side: `DiscreteBayesianNetwork` to hold $\mathcal{G}$ and the CPDs, `inference.CausalInference`/`do()` and `simulate(do=...)` to compute or sample $P(\mathrm{Pa}_Y\mid a)$, and `DAG` parent queries to identify $\mathrm{Pa}_Y$. The reward decomposition is exactly a marginalization pgmpy can evaluate. What is entirely NEW: a sequential-decision/bandit loop (per-configuration index maintenance, posterior sampling, exploration-exploitation, regret accounting). pgmpy estimators are batch/offline; there is no online UCB/TS index object.

**Relation to causal incentives / CIDs:** This is a repeated single-decision CID where the decision node is the intervention $\mathrm{do}(\mathbf{X}=\mathbf{x})$ and the utility node is $Y$; the agent repeatedly chooses a $\mathrm{do}()$ action to maximize $\mathbb{E}[Y]$. The fact that regret scales with $k^n=|\mathrm{dom}(\mathrm{Pa}_Y)|$ rather than the arm count mirrors the CID notion that only the *utility's parents* (the minimal value-relevant / requisite information set) determine optimal behavior — arms inducing identical $P(\mathrm{Pa}_Y\mid a)$ are decision-theoretically interchangeable. Because the graph is fixed and known, value-of-information about *structure* is zero; all learning is about the utility CPD, so this is the "incentive layer with known causal world model" baseline against which the unknown-graph papers add structure-discovery VoI.

---

### Causal Bandits with Unknown Graph Structure (NeurIPS, 2021)
**Authors:** Yangyi Lu, Amirhossein Meisami, Ambuj Tewari
**Link:** https://arxiv.org/abs/2106.02988 (NeurIPS 2021)
**TL;DR:** Without knowing the DAG, the CN-UCB algorithm uses $O(d\log^2 n)$ "central-node" interventions to locate the single reward-parent variable $X_R$, then runs ordinary UCB on its $K$ values, achieving cumulative regret scaling logarithmically in the number of nodes $n$ under two mild, provably necessary identifiability conditions.

**Problem & motivation:** All prior causal-bandit work assumed the full graph is given up front, which is unrealistic. This paper removes that assumption: the agent must *discover online* which variable drives the reward while simultaneously minimizing regret. It shows that exact graph recovery is unnecessary — only the reward-relevant variable must be found — and characterizes exactly when causal knowledge can (and cannot) beat plain MAB.

**Formal setup & key definitions:** Unknown DAG $D$ over $n$ nodes, each with domain $[K]$; action set is single-node interventions $\mathcal{A}=\{\mathrm{do}(X=x)\}$, $|\mathcal{A}|=nK$. One reward-generating variable $X_R$; reward $R_t=\mu_{a_t}+\varepsilon$ ($1$-subGaussian). Objective: CUMULATIVE regret $R_T=T\mu_{a^*}-\sum_t\mu_{a_t}$. **Structural knowledge assumed: the essential graph (CPDAG) $\mathcal{E}(D)$ and the observational distribution $P(\mathcal{X})$ are given (estimable from observational data); the orientation of edges and the identity of $X_R$ are discovered online via interventions.** So the input is a Markov-equivalence-class skeleton, not the full DAG. Three assumptions: (1) causal sufficiency + Markov + faithfulness; (2) **causal-effect identifiability** — every edge $X_i\to X_j$ has a detectable interventional effect; (3) **reward identifiability** — every ancestor of $X_R$ moves the reward by at least $\Delta$.

**Key equations:**
- Assumption 2 (edge gap): $\exists\,\varepsilon>0$ s.t. $|P(X_j=x\mid\mathrm{do}(X_i=x'))-P(X_j=x)|>\varepsilon$ for some $x,x'$.
- Assumption 3 (reward gap): $\exists\,\Delta>0,\,x\in[K]$ s.t. $|\mathbb{E}[R\mid\mathrm{do}()]-\mathbb{E}[R\mid\mathrm{do}(X=x)]|\ge\Delta$ for all ancestors $X$ of $X_R$.
- Central-node property: $\max_{j\in N_{\mathcal{T}}(v_c)}q\!\left(B_{\mathcal{T}}^{v_c:X_j}\right)\le 1/2$ (each central-node intervention halves the search space).
- Discovery sample budget (Lemma 1): $T_1=K\,B\,(2+d)\log_2 n$ interventions with $B=\max\!\Big\{\tfrac{32}{\Delta^2}\log\tfrac{8nK}{\delta},\,\tfrac{2}{\varepsilon^2}\log\tfrac{8n^2K^2}{\delta}\Big\}$.
- Regret, causal trees (Thm 1): $R_T=\tilde O\!\Big(K\max\{\tfrac{1}{\Delta^2},\tfrac{1}{\varepsilon^2}\}\,d(\log n)^2+\sqrt{KT}\Big)$, $d=d_{\max}(\mathrm{skeleton}(D))$.
- Causal forests (Thm 2): same with $d\to(d+C(D))$, $C(D)=$ number of tree components.
- General intersection-incomparable graphs (Thm 3): $R_T=\tilde O\!\Big((d(\mathcal{T}_{G_R})\omega(G_R)+\sum_G\omega(G))\,K\max\{\tfrac{1}{\Delta^2},\tfrac{1}{\varepsilon^2}\}\log n\,\log\mathcal{C}_{\max}+\sqrt{\omega(G_R)KT}\Big)$, $\omega(\cdot)$ clique number, $\mathcal{C}_{\max}$ bound on number of maximal cliques.
- Lower bounds (Thms 4,5): dropping Assumption 2 *or* 3 forces $\inf_\pi\sup_\nu\mathbb{E}[R_T]=\Omega(\sqrt{nKT})$ even with the graph known.

**Main results / theorems:** Under the two identifiability assumptions, the leading $\sqrt{T}$ term has effective arm-count $K$ (the values of $X_R$) — or $\omega(G_R)K$ for general graphs — instead of the naive $nK$; the cost of *not* knowing the graph is only an additive, $T$-independent discovery term that is logarithmic in $n$. The matching $\Omega(\sqrt{nKT})$ lower bounds prove both conditions are necessary: without them, causal knowledge gives no improvement over standard MAB.

**Algorithms:** **CN-UCB (Central-Node UCB)** for trees, in three stages: (1) identify a subtree containing $X_R$ using $O(\log n)$ adaptive central-node interventions; (2) pinpoint $X_R$ inside it with $O(d\log^2 n)$ further central-node interventions, orienting edges via $|P(Y=y)-\hat P(Y=y\mid\mathrm{do}(v_c=z))|\ge\varepsilon/2$ and testing reward shifts against $\Delta/2$; (3) run standard UCB on the reduced set $\mathcal{A}_R=\{\mathrm{do}(X_R=k):k=1,\dots,K\}$ for the remaining rounds. Forests/general graphs decompose into per-component subtree searches plus clique-level handling. The discovery loop and the bandit loop are interleaved (discovery first, then exploit).

**Relevance to pgmpy:** The *input* — a CPDAG/essential graph $\mathcal{E}(D)$ and observational $P(\mathcal{X})$ — is exactly what pgmpy's `causal_discovery.PC` outputs (a `PDAG`) plus a fitted `DiscreteBayesianNetwork`; pgmpy's CI tests and skeleton/orientation machinery already produce this. Edge-effect and reward-effect checks reuse interventional queries that pgmpy can simulate via `simulate(do=...)`. What is NEW: the *adaptive interventional* discovery (central-node search that chooses which $\mathrm{do}()$ experiment to run next to halve uncertainty), online edge orientation from interventions, and the interleaving of this discovery with a UCB exploitation phase plus regret tracking — none of which exist in pgmpy's offline discovery stack.

**Relation to causal incentives / CIDs:** Locating $X_R$ before exploiting is a pure value-of-information computation over the unknown graph: each central-node intervention is chosen to maximally reduce structural uncertainty, and the additive $\log n$ discovery term is the price of that information. In CID terms, the agent's decision (which node to intervene on) plays a dual role — experiment and exploitation — and the result formalizes that the agent only needs to resolve uncertainty about which node has *control value* for the utility ($X_R$ and its ancestors), not the whole world model. The $\Omega(\sqrt{nKT})$ lower bounds say that when effects are unidentifiable, VoI collapses and the structural prior confers no incentive advantage.

---

### Causal Bandits without prior knowledge using separating sets (CLeaR, 2022)
**Authors:** Arnoud A. P. de Kroon, Joris M. Mooij, Danielle Belgrave
**Link:** https://arxiv.org/abs/2009.07916 (CLeaR / Conference on Causal Learning and Reasoning 2022)
**TL;DR:** Instead of the full graph, the agent only needs a *separating set* that d-separates the intervention variables from the reward; an "information-sharing" estimator built on that set is unbiased with variance no larger than the naive sample mean, and separating sets are found with ordinary CI tests or causal discovery.

**Problem & motivation:** Earlier causal-bandit methods (e.g. Lattimore et al. 2016) require the exact causal graph, which is rarely available. This work asks for far less: a separating set, obtainable from data via conditional-independence tests. It generalizes those methods so they remain applicable — and never worse than standard bandits — without prior structural knowledge.

**Formal setup & key definitions:** SCM $\mathcal{M}$ over variables $\mathbf{V}$; each round the agent intervenes $\mathrm{do}(\mathbf{I}=\boldsymbol{\zeta})$ on context variables $\mathbf{I}$ and observes $\mathbb{P}[\mathbf{V}\mid\mathbf{I}=\boldsymbol{\zeta}]$; target/reward $Y$. Objective: CUMULATIVE regret $\mathcal{R}=\sum_{n=1}^T\big(\max_{\boldsymbol{\zeta}}\mathbb{E}[Y\mid\mathbf{I}=\boldsymbol{\zeta}]-Y^n\big)$. **Structural knowledge assumed: none up front — only a separating set $\mathbf{S}$ such that $\mathbf{I}\perp_{\mathcal{G}}Y\mid\mathbf{S}$ (equivalently $\mathbf{I}\perp\!\!\!\perp_{\mathbb{P}_{\mathcal{M}}}Y\mid\mathbf{S}$), which is *discovered* from data.** The parents/Markov blanket of $Y$ are one such set, but any valid separator works.

**Key equations:**
- Separating-set decomposition (law of total expectation): $\mathbb{E}[Y\mid\mathbf{I}=\boldsymbol{\zeta}]=\sum_{\mathbf{s}}\mu(\mathbf{s})\,p(\mathbf{s}\mid\boldsymbol{\zeta})$, since $Y\perp\!\!\!\perp\mathbf{I}\mid\mathbf{S}$.
- Information-sharing (IS) estimator: $\hat\mu_{IS}(\boldsymbol{\zeta};\mathcal{D}^N,\mathbf{S}):=\sum_{\mathbf{s}\in D(\mathbf{S})}\hat\mu(\mathbf{s};\mathcal{D}^N)\,\hat p(\mathbf{s}\mid\boldsymbol{\zeta},\mathcal{D}^N)$, with $\hat p(\mathbf{s}\mid\boldsymbol{\zeta})=\dfrac{N_{\mathcal{D}}(\mathbf{S}=\mathbf{s},\mathbf{I}=\boldsymbol{\zeta})}{N_{\mathcal{D}}(\mathbf{I}=\boldsymbol{\zeta})}$ and (binary $Y$) $\hat\mu(\mathbf{s})=\dfrac{N_{\mathcal{D}}(Y=1,\mathbf{S}=\mathbf{s})}{N_{\mathcal{D}}(\mathbf{S}=\mathbf{s})}$.
- Theorem 3.1 (unbiasedness + variance): if $\mathbf{I}\perp\!\!\!\perp_{\mathbb{P}_{\mathcal{M}}}Y\mid\mathbf{S}$, then $\hat\mu_{IS}$ is unbiased and $\mathbb{V}[\hat\mu_{IS}(\boldsymbol{\zeta};\mathcal{D}^N)]\le\mathbb{V}[\hat\mu_{SM}(\boldsymbol{\zeta};\mathcal{D}^N)]$ (never worse than the sample-mean estimator).
- Partial separating set ($\mathbf{S}$ separates only $\mathbf{I}'\subset\mathbf{I}$): $\hat\mu_{IS}(\boldsymbol{\zeta};\mathcal{D}^N,\mathbf{S},\mathbf{I}_{NS})=\sum_{\mathbf{s}}\hat\mu(\mathbf{s};\mathcal{D}^N,\mathbf{I}_{NS})\,\hat p(\mathbf{s}\mid\boldsymbol{\zeta})$.

**Main results / theorems:** The central guarantee is statistical, not a regret bound: with a valid (or partial) separating set the IS estimator is unbiased and has variance bounded by the naive per-arm sample mean (Thm 3.1), so the resulting bandit is never worse than standard MAB and is strictly better when the separator pools information across arms. The cumulative-regret improvement is demonstrated empirically (no closed-form $O(\cdot)$ regret bound is proved); regret degrades gracefully if the estimated separating set is wrong, since the algorithm falls back toward the sample mean.

**Algorithms:**
- *Thompson Sampling (discrete):* Dirichlet prior on $\mathbb{P}[\mathbf{S}=\mathbf{s}\mid\mathbf{I}=\boldsymbol{\zeta}]$, Beta prior on $\mathbb{P}[Y=1\mid\mathbf{S}=\mathbf{s}]$; sample parameters, plug into the decomposition to get $\hat{\mathbb{E}}[Y\mid\boldsymbol{\zeta}]$, pull the arm maximizing it, update posteriors.
- *UCB-Normal (Gaussian):* unknown-variance UCB with the sample mean replaced by $\hat\mu_{IS}$; $\hat\mu(\mathbf{s})$ via linear regression, variance via bootstrap; when several candidate separating sets exist, pick the one with lowest estimated variance.
- *Separating-set discovery:* either direct CI testing of candidate $\mathbf{S}$ ($G^2$ test for discrete, Peters et al. test for linear-Gaussian, threshold $\alpha=0.05$) or a causal-discovery solver (ASD-JCI123kt) that scores graphs and returns sets with positive confidence for $\mathbf{I}\perp\!\!\!\perp Y\mid\mathbf{S}$.

**Relevance to pgmpy:** This is the closest fit to existing pgmpy machinery. Separating-set discovery is exactly pgmpy's `ci_tests` (`chi_square`/`g_sq`/`pearsonr`) plus d-separation queries on `DAG`/`PDAG` (`is_dconnected`, minimal d-separator finding); pgmpy's PC already finds separating sets internally during skeleton learning, and `causal_identification` reasons about separators. What is NEW: the *information-sharing estimator* itself (an online, separator-conditioned reward estimator updated each round), the Dirichlet/Beta TS posteriors over $(p(\mathbf{s}\mid\boldsymbol{\zeta}),p(Y\mid\mathbf{s}))$, the variance-based selection among multiple candidate separators, and the bandit loop — pgmpy has no online estimator nor exploration-exploitation policy.

**Relation to causal incentives / CIDs:** A separating set $\mathbf{S}$ is precisely the *minimal sufficient information* the decision needs about the utility: conditioning on $\mathbf{S}$ renders the reward independent of the choice, so $\mathbf{S}$ is the requisite-observation set in CID terms. The "variance $\le$ sample mean" guarantee says correctly identifying this information set has non-negative value of information and never hurts — a clean VoI statement. Choosing among candidate separators by lowest variance is value-of-information-guided experiment selection. Because the separator can be *learned online* from interventional data, the method connects causal discovery (finding $\mathbf{S}$) directly to the decision layer (using $\mathbf{S}$ to evaluate $\mathrm{do}()$ utilities).

---

### Partial Structure Discovery is Sufficient for No-regret Learning in Causal Bandits (NeurIPS, 2024)
**Authors:** Muhammad Qasim Elahi, Mahsa Ghasemi, Murat Kocaoglu (Purdue)
**Link:** https://arxiv.org/abs/2411.04054
**TL;DR:** With an unknown graph that may contain latent confounders, you do not need full causal discovery — learning only the induced subgraph on the reward's ancestors plus a necessary-and-sufficient subset of latent confounders suffices to recover all possibly-optimal arms (POMISs), and a two-phase discover-then-UCB algorithm achieves sublinear cumulative regret with polynomial-sample discovery.

**Problem & motivation:** In confounded causal bandits the optimal intervention can be on a special subset of the reward's *ancestors*, not just its parents, so prior unknown-graph methods either over-restrict or pay for full graph recovery. This paper pins down the *minimal* structural information needed for no-regret learning and proves discovering it is enough — full causal discovery (which is expensive and often unidentifiable under latents) is wasteful.

**Formal setup & key definitions:** Causal graph $\mathcal{G}$ (an ADMG with bidirected edges for latents) over discrete variables, $\Omega(V_i)=[K]$; binary reward $Y$. Single- or multi-node interventions $\mathrm{do}(\mathbf{W}=\mathbf{w})$. Objective: CUMULATIVE regret
$R_T:=T\max_{\mathbf{W}\subseteq\mathbf{V}}\max_{\mathbf{w}\in[K]^{|\mathbf{W}|}}\mathbb{E}[Y\mid\mathrm{do}(\mathbf{W}=\mathbf{w})]-\sum_{t=1}^T\mathbb{E}[Y\mid\mathrm{do}(\mathbf{W}_t=\mathbf{w}_t)]$.
**Structural knowledge assumed: none — the graph (including which latent confounders exist) is unknown and partially discovered online.** The key objects are POMISs (possibly-optimal minimal intervention sets, Lee & Bareinboim 2018): the only intervention targets that can be optimal in some SCM consistent with the graph.

**Key equations / definitions:**
- POMIS building blocks — **Def 1 (UC-territory):** $\mathbf{T}\subseteq V(\mathcal{H})$ containing $Y$ is a UC-territory on $\mathcal{H}=\mathcal{G}[\mathrm{An}(Y)]$ if $\mathrm{De}_{\mathcal{H}}(\mathbf{T})=\mathbf{T}$ and $\mathsf{CC}_{\mathcal{H}}(\mathbf{T})=\mathbf{T}$ (closed under descendants and c-component). **Def 2 (interventional border):** for the minimal UC-territory $\mathbf{T}$, $\mathbf{X}=\mathsf{Pa}(\mathbf{T})\setminus\mathbf{T}=\mathrm{IB}(\mathcal{G},Y)$. **Lemma 1:** $\mathrm{IB}(\mathcal{G}_{\overline{\mathbf{W}}},Y)$ is a POMIS for any $\mathbf{W}\subseteq\mathbf{V}\setminus\{Y\}$.
- **Lemma 2 (what must be learned):** it is necessary to detect the latent confounders between $Y$ and any $X\in\mathrm{An}(Y)$ to identify all POMISs.
- **Theorem 1 (equivalence of partial info):** two graphs yield different POMIS collections iff there is $Z\in\mathrm{An}(Y)$ with a bidirected edge $Z\leftrightarrow Y$ in one graph but not the other, or a difference in bidirected edges within the relevant c-component region — i.e. only $\mathrm{An}(Y)$-side latents matter.
- Sample complexity, observable subgraph (**Theorem 2**): learns the true observable graph w.p. $\ge 1-\tfrac{1}{n^{\alpha/(2d_{\max})-2}}-8\alpha d_{\max}\log(n)(n\delta_1+\delta_2)$ using $8\alpha d_{\max}\log n\,(KAn+B)$ interventional samples, with $A=\max(\tfrac{8}{\epsilon^2},\tfrac{8}{\gamma^2})\log\tfrac{2nK^2}{\delta_1}$, $B=\tfrac{8}{\epsilon^2}\log\tfrac{2nK^2}{\delta_2}$.
- Sample complexity, full graph incl. latents (**Theorem 3**): w.p. $\ge 1-\tfrac{2}{n^{\alpha/(2d_{\max})-2}}-8\alpha d_{\max}\log(n)(n\delta_1+(\delta_2+\delta_3+\delta_4))$ using $8\alpha d_{\max}\log n\,(KAn+\max(B,C))$ samples, $C=\tfrac{16}{\eta\gamma^2}\log\tfrac{2n^2K^2}{\delta_3}+\tfrac{1}{2\eta^2}\log\tfrac{2n^2K^2}{\delta_4}$.
- **Theorem 4 (regret):** the two-phase algorithm achieves cumulative regret sublinear in $T$ (discovery is a $T$-independent additive cost, then UCB over POMIS arms gives the $\sqrt{T}$ term).

**Main results / theorems:** (i) A precise characterization (Lemma 2, Thm 1) that the *only* structure affecting the optimal-arm set is the induced subgraph on $\mathrm{An}(Y)$ together with the latent confounders incident to $Y$'s ancestors — recovering this is necessary and sufficient to enumerate all POMISs correctly. (ii) Polynomial-in-$n$ (and $K$, $1/\epsilon^2$, $1/\gamma^2$, $1/\eta^2$) sample complexity to learn this partial structure with high probability (Thms 2-3), strictly less than full-graph discovery. (iii) Sublinear cumulative regret (Thm 4) for discover-then-exploit.

**Algorithms:** **Two-phase.** *Phase 1 — partial discovery:* Algorithm 2 learns the observable induced subgraph on $\mathrm{An}(Y)$ via randomized interventions and edge-effect thresholds; Algorithm 3 detects the necessary-and-sufficient latent confounders (bidirected edges $Z\leftrightarrow Y$) using conditional causal-effect comparisons; from this, enumerate all POMISs (Lemma 1) to form the candidate arm set. *Phase 2 — exploit:* run standard UCB (Algorithm 4) over the POMIS arms for the remaining rounds.

**Relevance to pgmpy:** The POMIS machinery is graph-algorithmic and maps onto pgmpy's confounded-graph classes: `ADMG`/`MAG` already represent bidirected (latent) edges, and ancestors/descendants/c-component computations needed for UC-territory and interventional border are exactly the kind of routines pgmpy's `base` graphs support (or could host). pgmpy's `FCI` (latent-aware constraint-based discovery) and CI tests cover the *observational* side of structure learning. What is NEW: POMIS enumeration (no `possibly_optimal_minimal_intervention_set` exists in pgmpy), the *interventional, sample-complexity-bounded* partial-discovery procedures (Algorithms 2-3) that decide which $\mathrm{do}()$ experiments resolve the ancestor-side latents, and the two-phase bandit loop with regret accounting.

**Relation to causal incentives / CIDs:** This is the sharpest VoI statement of the batch: the agent has *no incentive to learn the full graph* — value of information is exactly zero for any structure outside $\mathrm{An}(Y)$ and its incident latents, because such structure cannot change the optimal $\mathrm{do}()$ decision. POMIS is the decision-theoretic notion of "nodes with control value": intervening outside the POMIS set can never be optimal, so the POMIS collection is the agent's relevant control-incentive set. The two-phase design embodies the discovery-to-decision link — Phase 1 spends interventions purely to resolve the POMIS-determining uncertainty (information-gathering), Phase 2 exploits — and the sublinear regret quantifies that learning only the incentive-relevant slice of the causal world model is enough for asymptotically optimal decisions.

## D. Confounders & d-Separators

### Adaptively Exploiting d-Separators with Causal Bandits (NeurIPS, 2022)
**Authors:** Blair Bilodeau, Linbo Wang, Daniel M. Roy (University of Toronto)
**Link:** arXiv 2202.05100 (https://arxiv.org/abs/2202.05100)
**TL;DR:** A single algorithm (HAC-UCB) that recovers the improved $\sqrt{|\mathcal{Z}|\,T}$ causal-bandit rate when the observed post-action context $\boldsymbol{Z}$ d-separates the action from the reward, yet never collapses to linear regret (unlike C-UCB) when it does not — all without oracle knowledge of whether a d-separator exists.

**Problem & motivation:** Classical minimax bandit regret is $\tilde{\mathcal{O}}(\sqrt{|\mathcal{A}|\,T})$. "Causal bandit" algorithms (C-UCB of Lu et al.) exploit an observed variable $\boldsymbol{Z}$ that d-separates the intervention from the outcome to get $\mathcal{O}(\sqrt{|\mathcal{Z}|\,T})$ regret, an improvement when $|\mathcal{Z}| < |\mathcal{A}|$. But this assumption is unverifiable, and if wrongly assumed it is catastrophic. The paper asks: can an algorithm be *adaptive* — as good as an oracle that knows whether a d-separator is present — without that knowledge?

**Formal setup & key definitions:** Stochastic bandits with *post-action contexts*: reward space $\mathcal{Y}=[0,1]$, finite context set $\mathcal{Z}$. The environment is a family $\nu=\{\nu_a : a\in\mathcal{A}\}$, $\nu_a\in\mathscr{P}(\mathcal{Z}\times\mathcal{Y})$; at round $t$ the learner picks $A_t$ and observes $(Z_t(A_t),Y_t(A_t))$ — the context is seen *after* the action (distinct from contextual bandits), and corresponds to a potential-outcome vector under intervention $A_t$. The causal DAG has $A$ (intervention), $Z$ (post-action context), $Y$ (reward, a leaf), and possible unobserved confounder $U$. Their decision-theoretic generalization of d-separation is the **conditionally benign** property (Def. 3.1): $\nu$ is conditionally benign iff there exists $p\in\mathscr{P}(\mathcal{Z}\times\mathcal{Y})$ with, for every $a$, $\nu_a(Z)\ll p(Z)$ and $\nu_a(Y\mid Z)=p(Y\mid Z)$ a.s. — i.e. the reward's law given $Z$ is *action-invariant*. The connection: **Thm 3.3** — $\boldsymbol{Z}$ d-separates $Y$ from $\boldsymbol{A}$ on $\mathcal{G}$ iff $(\mathcal{G},\mathcal{A})$ is conditionally benign; **Thm 3.4** — dropping the null (pure-observation) intervention, $\boldsymbol{Z}$ d-separates $Y$ from $\boldsymbol{A}$ on $\mathcal{G}_{\boldsymbol{A}}$ (edges into $\boldsymbol{A}$ removed) iff $(\mathcal{G},\mathcal{A}_0)$ is conditionally benign; **Lemma 3.5** — the front-door criterion for $(\boldsymbol{A},Y)$ is sufficient. With latent $U$, $Z$ is generally not a d-separator unless front-door holds.

**Key equations:**
- Regret objective: $R_{\nu,\pi}(T) = T\cdot \max_{a\in\mathcal{A}} \mathbb{E}_{\nu_a}[Y] - \sum_{t=1}^{T}\mathbb{E}_{\nu_{A_t}}[Y]$.
- UCB baseline (**Thm 4.1**, $\delta=2/T^2$): $R_{\nu,\text{UCB}}(T) \le 2|\mathcal{A}| + 4\sqrt{2|\mathcal{A}|\,T\log T}$ — depends on $|\mathcal{A}|$.
- C-UCB action: $A^{C}_{t+1}=\arg\max_a \widehat{\text{UCB}}_t(a)$ with $\widehat{\text{UCB}}_t(a)=\sum_{z\in\mathcal{Z}}\text{UCB}^{\mathcal{Z}}_t(z)\,\mathbb{P}_{\hat\nu_a}[Z=z]$ — variance reduction by sharing the *action-invariant* $Y\mid Z$ estimate across arms.
- C-UCB benign bound (**Thm 4.3**, $\varepsilon$-close $\tilde\nu(Z)$): $R_{\nu,C}(T)\le 2|\mathcal{Z}| + 6\sqrt{|\mathcal{Z}|\,T\log T} + (\log T)\sqrt{2T} + 2\varepsilon(1+\sqrt{\log T})T$ — depends on $|\mathcal{Z}|$, not $|\mathcal{A}|$. (Remark 4.4: $\varepsilon=\sqrt{|\mathcal{Z}|/T}$ suffices for the optimal rate.)
- C-UCB failure (**Thm 4.5**): for every $\mathcal{A},\mathcal{Z}$ with $|\mathcal{A}|\ge 2$, there is a *non*-benign $\nu$ (even with $\tilde\nu(Z)=\nu(Z)$) such that for all confidence settings $\lim_{T\to\infty} R_{\nu,C}(T)/T \ge 1/120$ — i.e. linear regret.

**Main results / theorems:** **Thm 4.7 (Main Result).** For all $\mathcal{A},\mathcal{Z}$, $T\ge 25|\mathcal{A}|^2$, any $\nu$ and any $\tilde\nu(Z)$ (no assumptions):
$$R_{\nu,\text{HAC}}(T) \le 4|\mathcal{A}| + 11\,T^{3/4}(\log T)\sqrt{|\mathcal{A}||\mathcal{Z}|} + 15\sqrt{(|\mathcal{A}|+|\mathcal{Z}|)T\log T} + 5(\log T)\sqrt{T},$$
which is *always sublinear* (worst case $T^{3/4}$, never linear). And for $\varepsilon\le T^{-1/4}\sqrt{|\mathcal{A}||\mathcal{Z}|\log T}$, if $\nu$ is conditionally benign and $\tilde\nu(Z),\nu(Z)$ are $\varepsilon$-close, it recovers the $\sqrt{|\mathcal{Z}|\,T}$ C-UCB rate. **Thm 4.8:** there is a $\nu$ on which C-UCB is linear but HAC-UCB matches UCB's $\sqrt{|\mathcal{A}|\,T}$. **Adaptivity caveat (Thm 6.2 + §6.3, impossibility):** strict adaptive-minimax-optimality is *impossible* — a worst-case-optimal classical algorithm achieving $\sup_\nu R\le C\sqrt{|\mathcal{A}|\,T}$ must still incur $\ge C'\sqrt{|\mathcal{A}|\,T}$ on some benign $\nu$. This is precisely why HAC-UCB pays $T^{3/4}$ (not $\sqrt{T}$) in the worst case; they instead target *Pareto*-adaptive minimax optimality (Def. 6.3).

**Algorithms:** **HAC-UCB** (Hypothesis-tested Adaptive Causal UCB, Algorithm 1). (1) Play each $a$ for $\lceil 4\sqrt{T}/|\mathcal{A}|\rceil$ rounds; if the MLE $\hat\nu_a(Z)$ disagrees with the input $\tilde\nu_a(Z)$ by more than $2T^{-1/4}\sqrt{|\mathcal{A}||\mathcal{Z}|\log T}$, replace $\tilde\nu(Z)\leftarrow\hat\nu(Z)$. (2) Play each $a$ for $\lceil\sqrt{T}/|\mathcal{A}|\rceil$ more rounds. (3) Each round, *optimistically* play C-UCB while a per-round hypothesis test (via $\text{D}^{\mathcal{A}}_{t-1}(a)=\text{UCB}^{\mathcal{A}}_{t-1}(a)-\widehat{\text{UCB}}_{t-1}(a)+\tfrac{\sqrt{|\mathcal{A}||\mathcal{Z}|\log T}}{T^{1/4}}$) passes; the first time the test fails, switch to vanilla UCB forever.

**Relevance to pgmpy:** Directly actionable — pgmpy already implements d-separation on `DAG`/`PDAG`/ADMG bases and a front-door check in `CausalInference`, which are exactly the oracle predicates behind Thm 3.3–3.5 / Lemma 3.5. A causal-bandit module could expose (i) a `is_conditionally_benign`/`d_separates(Z, A, Y)` test to decide when the $\sqrt{|\mathcal{Z}|T}$ estimand sharing is valid, (ii) the C-UCB estimand $\sum_z \text{UCB}(z)\mathbb{P}[Z=z]$ built from pgmpy's identified $P(Z\mid a)$, and (iii) HAC-UCB's hedge as a robust default. The latent-$U$ figures map to ADMG bidirected edges; the front-door fallback is a partial-identification path pgmpy supports.

**Relation to causal incentives / CIDs:** The post-action context $Z$ that d-separates $A$ from $Y$ is exactly an *information link / observation* in a CID, and the conditionally-benign property is a decision-theoretic restatement of "observing $Z$ identifies the do-effect without intervening" — i.e. a positive value-of-information of the link. Thm 4.5's catastrophic C-UCB failure dramatizes the causal-incentives warning about latent confounders ($U\to A$, $U\to Y$): assuming an information link blocks confounding when it does not is the bandit analogue of a mis-specified SCM, and the safe response is the front-door (partial-id) route. The impossibility of strict adaptivity (Thm 6.2 / §6.3) is the same lesson as the "Limits of Predicting Agents" min/max-over-consistent-SCMs bound: you cannot freely claim the benefit of a structural assumption you cannot verify — robustness has a provable price ($T^{3/4}$ vs $\sqrt{T}$).

---

### A Causal Bandit Approach to Learning Good Atomic Interventions in Presence of Unobserved Confounders (UAI, 2022)
**Authors:** Aurghya Maiti, Vineet Nair, Gaurav Sinha
**Link:** arXiv 2107.02772 (https://arxiv.org/abs/2107.02772)
**TL;DR:** For atomic-intervention causal bandits on semi-Markovian graphs *with unobserved confounders*, they give simple- and cumulative-regret algorithms whose regret scales with an instance-dependent count $m(\mathcal{C})\le N$ of "hard-to-identify-from-observation" arms rather than the full $2N+1$ action set, and prove this is near-tight.

**Problem & motivation:** Prior causal-bandit work (Lattimore et al., Lu et al., Nair et al.) assumed either special graphs (parallel/no-backdoor) or fully observed graphs. Real causal graphs have unobserved confounders. The goal is best-arm identification (simple regret) and cumulative-regret minimization over atomic interventions $do(X_i=x)$ when bidirected edges (latent confounders) are present, exploiting observational data wherever the interventional effect is identifiable.

**Formal setup & key definitions:** Causal Bayesian network $\mathcal{C}=(\mathcal{G},\mathbb{P})$ on a **semi-Markovian** graph — each hidden $U\in\boldsymbol{U}$ has no parents and is a parent of at most two observables (equivalently an ADMG with **bidirected edges** for latent confounders). Binary variables $\mathbb{P}(X_i=1\mid \boldsymbol{Pa}(X_i)=\boldsymbol{z})=p_{i,\boldsymbol{z}}$. Intervenable set $\boldsymbol{X}=\{X_1,\dots,X_N\}$, reward $Y$; action set $\mathcal{A}=\{a_{i,x}\mid i\in[N], x\in\{0,1\}\}\cup\{a_0\}$ — **$2N+1$ arms** ($a_0$ = pure observation). Reward $\mu_{i,x}=\mathbb{E}[Y\mid do(X_i=x)]$. **Identifiability assumption:** $P_{x_i}(y)$ is identifiable for all $X_i$; sufficient graphical condition — *no path of bidirected edges from $X_i$ to a child of $X_i$* (Tian–Pearl c-component criterion). The complexity parameter: let $q_i=\min_{\boldsymbol{z},x}\mathbb{P}(X_i=x,\boldsymbol{Pa}^c(X_i)=\boldsymbol{z})$ (smallest joint prob. of $X_i$ with its confounded-parent configuration), $k_i$ = size of the c-component containing $X_i$, $I_\tau=\{i : q_i^{k_i} < 1/\tau\}$. Then
$$m(\mathcal{C}) = \min\{\tau\in[2,2N] : |I_\tau|\le\tau\}.$$
$m(\mathcal{C})\le N$ counts the arms whose do-effect *cannot* be reliably estimated from observation alone (rarely-observed configurations, exacerbated by larger confounded c-components $k_i$) and so require explicit intervention.

**Key equations:**
- Simple regret: $r_{\text{ALG}}(T)=\max_a\mu_a-\mu_{a_T}$; cumulative regret: $R_{\text{ALG}}(T)=\max_a(\mu_a\cdot T)-\sum_t\mu_{a_t}$.
- The interventional reward is recovered from observation via the c-component factorization $P_x(\boldsymbol{w}_i)=\sum_{x'}\prod_{V_j\in\boldsymbol{C}_i}P\big((x'\!\circ\boldsymbol{w}_i)_{V_j}\mid(x'\!\circ\boldsymbol{w}_i)_{\boldsymbol{Z}_j}\big)\prod_{V_j\notin\boldsymbol{C}_i}P\big((\boldsymbol{w}_i)_{V_j}\mid(x\circ\boldsymbol{w}_i)_{\boldsymbol{Z}_j}\big)$ (Tian–Pearl / Bhattacharyya et al.).
- UCB-style estimate: $\bar\mu_{i,x}(t)=\hat\mu_{i,x}(t)+\sqrt{2\ln t/(N^{i,x}_t+C^{i,x}_t)}$, combining direct interventional pulls with observationally-derived counts (split odd/even to debias).

**Main results / theorems:** **Thm 3.1 (simple regret, SRM-ALG):** $r_{\text{SRM-ALG}}(T)=\mathcal{O}\!\Big(\sqrt{\tfrac{m(\mathcal{C})}{T}\,\log\tfrac{NT}{m(\mathcal{C})}}\Big)$ — depends on $m(\mathcal{C})\le N$, not the $2N+1$ arms. **Thm 4.1 (lower bound):** for any $n$-ary tree graph with $N$ intervenable nodes and any $M\in[1,N]$, there is a $\mathbb{P}$ with $m(\mathcal{C})=M$ and $r_{\text{ALG}}(T)=\Omega(\sqrt{m(\mathcal{C})/T})$ for any algorithm — so Thm 3.1 is tight up to the $\sqrt{\log(NT/m)}$ factor. **Thm 5.1 (cumulative regret, CRM-ALG):** if the observational arm $a_0$ is optimal, regret is $\mathcal{O}(1)$; otherwise an instance-dependent bound of the form $\tfrac{58\ln T}{\Delta_0}+\Delta_0+\sum_{\Delta_{i,x}>0}\Delta_{i,x}\max\!\big(0,\,1+8\ln T(\cdots)\big)$ with gaps $\Delta_a=\mu_{a^*}-\mu_a$, where the causal correction term ($\propto p_{i,x}$) can drive many arms' contributions toward zero.

**Algorithms:** **SRM-ALG** (simple-regret): Phase 1 — pull $a_0$ for $T/2$ rounds, estimate all $\hat\mu_{i,x}$ from observation via the c-component reduction; estimate $\hat q_i$ and $\hat m$; form $\mathcal{Q}=\{a_{i,x} : \hat q_i^{k_i} < 1/\hat m\}$ (the *infrequently-observed* arms). Phase 2 — pull each arm in $\mathcal{Q}$ equally for the remaining $T/2$ rounds; return the best empirical reward. **CRM-ALG** (cumulative-regret): UCB with causal weighting — enforce $N^0_t\ge\beta^2\log t$ observational pulls, combine direct and backdoor-/c-component-derived estimates, and play $\arg\max_a\bar\mu_a(t)$.

**Relevance to pgmpy:** Squarely in pgmpy's ADMG/c-component territory — the identifiability test (no bidirected path $X_i\to$ child), c-component computation, and the Tian–Pearl factorization are all standard graph operations pgmpy can supply (`CausalInference`, ADMG base with bidirected edges). A bandit module could compute $m(\mathcal{C})$ directly from c-component sizes $k_i$ and observed-frequency estimates $\hat q_i$, and reuse pgmpy's ID-algorithm output as the observational estimator $\hat\mu_{i,x}$. This is the canonical "estimate-do-from-observation-then-intervene-only-on-the-hard-arms" pipeline.

**Relation to causal incentives / CIDs:** The semi-Markovian/ADMG model is exactly the latent-confounder setting the causal-incentives review treats with c-components and Tian–Pearl identifiability. The identifiability assumption is the point-identification boundary: when "no bidirected path from $X_i$ to its child" *fails*, the do-effect is only partially identified and one must fall back to min/max-over-consistent-SCMs bounds (the "Limits of Predicting Agents" query) rather than a single $\mu_{i,x}$. The parameter $m(\mathcal{C})$ is a quantitative value-of-information measure: it counts how many interventions latent confounding *forces* you to actually perform because observation cannot identify them — the experimental cost of confounding.

---

### Confounded Budgeted Causal Bandits (CLeaR, 2024)
**Authors:** Fateme Jamshidi (EPFL), Jalal Etesami (TUM), Negar Kiyavash (EPFL)
**Link:** arXiv 2401.07578 (https://arxiv.org/abs/2401.07578)
**TL;DR:** Generalizes causal bandits to **non-uniform intervention costs under a fixed budget $B$** on general ADMGs with hidden confounders, giving cumulative- and simple-regret algorithms (with matching lower bounds) whose optimum is the cost-normalized arm $\arg\max_a \mu_a/c_a$, and which correct and tighten the bounds of Nair et al. (2021) and Maiti et al. (2022).

**Problem & motivation:** Prior causal bandits assume unit cost per pull and (mostly) no hidden confounders. Real interventions differ in cost (a drug vs. surgery) and graphs have latent confounders. With non-uniform costs the exploration/exploitation trade-off must weigh reward *per unit cost*, and repeatedly pulling the highest-reward arm need not be budget-optimal.

**Formal setup & key definitions:** ADMG $\mathcal{G}=(\boldsymbol{V},\boldsymbol{E}^d,\boldsymbol{E}^b)$ — directed edges $\boldsymbol{E}^d$ and **bidirected edges $\boldsymbol{E}^b$ encoding hidden confounders**; binary variables. Intervenable $\boldsymbol{X}=\{X_1,\dots,X_N\}$, reward $Y$; actions $\mathcal{A}=\{a_{i,x}\mid i\in[N],x\in\{0,1\}\}\cup\{a_0\}$, with **costs** $c_a\in\mathbb{R}_+$, $c_0=1$ (normalized), **budget** $B\ge 0$. A *c-component* is a maximal set connected by bidirected paths; $P_{\boldsymbol{s}}(\boldsymbol{r})$ is *identifiable* if computable from $P(\boldsymbol{V})$. **Identifiability requirement:** $P_{x_i}(y)$ identifiable for all $X_i$. **Remark 3:** uniform cost $c_a=c$ reduces this to a non-budgeted bandit with horizon $T=B/c$.

**Key equations:**
- Cost-normalized optimum and gap: $a^*:=\arg\max_{a\in\mathcal{A}}\tfrac{\mu_a}{c_a}$, and $\delta_a:=\tfrac{\mu_{a^*}}{c_{a^*}}-\tfrac{\mu_a}{c_a}$.
- Simple regret: $R_s(B):=\mu_{a^*}-\mu_{\tilde a_B}$; cumulative regret: $R_c(B):=\mathcal{R}^*(B)-\mathcal{R}^\ell(B)$ with budget constraint $\sum c_{a^\tau}\le B$.
- Observational c-component estimator: $P_x(\boldsymbol{w}_i)=\sum_{x'}\prod_{V_j\in\boldsymbol{C}_i}P\big((x'\!\circ\boldsymbol{w}_i)_{V_j}\mid(x'\!\circ\boldsymbol{w}_i)_{\boldsymbol{Z}_j}\big)\prod_{V_j\notin\boldsymbol{C}_i}P\big((\boldsymbol{w}_i)_{V_j}\mid(x\circ\boldsymbol{w}_i)_{\boldsymbol{Z}_j}\big)$, with $\hat\mu_{i,x}$ unbiased (Lemma 4).
- UCB: $\bar\mu^t_{i,x}=\hat\mu^t_{i,x}+\sqrt{2\ln t/(N^t_{i,x}+S^t_{i,x})}$.
- Infrequent-arm threshold (simple regret): an arm is *infrequent* if $\hat q_{i,x}\le(1/n(\hat{\boldsymbol q}))^{1/k_i}$, where $n(\hat{\boldsymbol q}):=\min\{\tau \mid \sum_{i,x} c_{i,x}\,\mathbb{1}\{\hat q_{i,x}<(1/\tau)^{1/k_i}\}\le\tau\}$ — the cost-weighted generalization of Maiti's $m(\mathcal{C})$.

**Main results / theorems:** **Thm 5 (cumulative regret):** $\delta_0\Big(\tfrac{8\ln B}{\delta_0^2}+1+\tfrac{\pi^2}{3}\Big)+\sum_{\delta_{i,x}>0}\delta_{i,x}\Big(\tfrac{8\ln B}{\delta_{i,x}^2}+2-\tfrac{8p_{i,x}}{18\delta_0^2|\boldsymbol{V}|}\ln b_{i,x}\cdot\tau_{i,x,b}+\tfrac{\pi^2}{3}\Big)$, with the cost-normalized gaps $\delta_a$ replacing plain gaps $\Delta_a$ and $\ln B$ replacing $\ln T$. **Thm 7 (simple regret):** $R_s = \mathcal{O}\!\big(\sqrt{\tfrac{n(\hat{\boldsymbol q})}{B}\,\log\tfrac{NB}{n(\hat{\boldsymbol q})}}\big)$. **Lower bounds:** simple regret $\Omega(\sqrt{n(\boldsymbol q)/B})$; cumulative $\min_{A_B}\max R_c \ge \Omega(\sqrt{\lceil B/c\rceil\,KN})$ for $N$ nodes of domain size $K$. **Corrections:** Remark 8 — $c_{i,x}=1$ in Thm 7 *recovers* Maiti et al. (2022)'s simple-regret bound. Remark 9 — Nair et al. (2021) use a *worse* exploration set; Appendix E shows $n(\boldsymbol q)\le c\,m'(\boldsymbol q)$ for all $c>1$, so Algorithm 2 beats Nair even in Nair's own setting. The paper states it "corrects the oversights and errors in the proofs of these papers which affect the validity of the bounds claimed therein."

**Algorithms:** **Algorithm 1 (Budgeted Cumulative Regret):** pull each arm once; while $B^t\ge 1$, if $N^{t-1}_0<\beta^2\log t$ or budget too small to afford any intervention then observe ($a^t=a_0$), else exploit $a^t=\arg\max_a\bar\mu^{t-1}_a$; update via c-component factorization; pick cost-adjusted best $\tilde a=\arg\max_a \hat\mu^t_a/c_a$; decrement budget $B^t=B^{t-1}-c_{a^{t-1}}$. **Algorithm 2 (Budgeted Simple Regret):** Phase 1 spends $B/2$ observing, estimating all $\hat\mu_{i,x}$; compute $n(\hat{\boldsymbol q})$ and the infrequent set $\mathcal{A}'$; Phase 2 spends the remaining $B/2$ pulling each infrequent arm $n=\tfrac{B}{2\sum_{i,x}c_{i,x}\mathbb{1}\{a_{i,x}\in\mathcal{A}'\}}$ times — the **cost-weighted budget allocation** — then returns $\arg\max_a\hat\mu_a$.

**Relevance to pgmpy:** The strongest fit of the three for an intervention-cost-aware design. It reuses pgmpy's ADMG bidirected-edge representation, c-component decomposition, and Tian–Pearl identifiability check, plus the do-from-observation factorization that pgmpy's identification machinery already produces. The novel pgmpy-facing surface is *per-action cost annotations* ($c_a$ on intervention nodes, a natural extension of the node-role annotation system) and a *budget* parameter, with the cost-normalized optimum $\arg\max_a\mu_a/c_a$ and cost-weighted infrequent-arm allocation as the budgeted policy.

**Relation to causal incentives / CIDs:** Bidirected edges, c-components, and Tian–Pearl identifiability are the latent-confounder + identification toolkit central to the causal-incentives review. Non-uniform costs $c_a$ formalize the *cost of intervention/instrumentation* in a CID, and the budget $B$ is a hard resource constraint on information gathering; the cost-normalized value-of-control $\mu_a/c_a$ is the natural decision-theoretic objective, and the cumulative-regret lower bound $\Omega(\sqrt{\lceil B/c\rceil KN})$ quantifies the unavoidable cost of identifying the best control under confounding. The identifiability requirement again marks the point-identification frontier; beyond it the relevant object is the min/max of $\mathbb{E}[Y\mid do(X_i=x)]$ over SCMs consistent with $P(\boldsymbol{V})$ — the partial-identification bound the "Limits of Predicting Agents" line computes.

## E. Linear-SEM & Combinatorial (scaling to structure)

### Causal Bandits for Linear Structural Equation Models (JMLR, 2023)
**Authors:** Burak Varici, Karthikeyan Shanmugam, Prasanna Sattigeri, Ali Tajer
**Link:** arXiv:2208.12764 ; JMLR vol. 24 (https://www.jmlr.org/papers/volume24/22-0969/22-0969.pdf)
**TL;DR:** Introduces causal bandits over a *linear* SEM with soft interventions and gives UCB / Thompson-Sampling algorithms whose cumulative regret is polynomial in the number of nodes $N$ (despite $2^N$ arms) and scales as $\tilde{\mathcal{O}}(d^{L+1/2}\sqrt{NT})$ with max in-degree $d$ and longest causal path $L$.

**Problem & motivation:** Prior causal-bandit work (Lattimore et al.) treats arms atomically and incurs regret that grows with the (exponential) number of interventions, ignoring the parametric structure of the data-generating process. This paper assumes a *known graph* but *unknown linear-SEM weights*, so that observing all nodes after each intervention lets the learner share statistical strength across arms. The goal is to exploit linear structure so regret depends on graph parameters ($N$, $d$, $L$) rather than the arm count $2^N$.

**Formal setup & key definitions:** The model is a **linear SEM** on a DAG with $N$ nodes (plus a virtual source node $0$ carrying the affine/noise-mean term), with a designated reward node $N$ (the only childless node). Actions are *soft* interventions on arbitrary node subsets, so the action space is $\mathcal{A} = 2^{\mathcal{V}}$ with $|\mathcal{A}| = 2^N$: intervening on node $i$ swaps its incoming-edge weight row from the observational mechanism to a (also unknown) interventional one. Reward is the expected value of the sink node; objective is cumulative regret over horizon $T$.

**Key equations:**
- Linear SEM: $\mathbf{X} = \mathbf{B}^\top \mathbf{X} + \boldsymbol{\varepsilon}$, with $\mathbf{B} \in \mathbb{R}^{(N+1)\times(N+1)}$ *strictly upper triangular* (encoding edge weights $\mathbf{H}$ and affine terms $\boldsymbol{\nu}$); noise $\boldsymbol{\varepsilon}$ independent, $1$-sub-Gaussian, $\|\boldsymbol{\varepsilon}\| \leq m_\varepsilon$.
- Intervention-dependent weights (row $i$): $[\mathbf{B}_a]_i = \mathbb{1}\{i \in a\}\,[\mathbf{B}^*]_i + \mathbb{1}\{i \notin a\}\,[\mathbf{B}]_i$, where $\mathbf{B}^*$ holds the (unknown) post-intervention weights.
- Reward as a **path-sum over the weight matrix powers**: $\mu_a = f(\mathbf{B}_a) \triangleq \sum_{\ell=1}^{L+1} [\mathbf{B}_a^{\ell}]_{(0,N)}$ — i.e. the reward at sink $N$ is a linear function of the noise vector, summing products of edge weights along every directed path from source $0$ to $N$ (equivalently $\mu_a = \langle f(\mathbf{B}_a), \boldsymbol{\nu}\rangle$).
- Regret: $\mathbb{E}[R(T)] = T\mu_{a^*} - \sum_{t=1}^T \mathbb{E}[X_N(t)]$, with $a^* = \arg\max_{a\in\mathcal{A}} \mu_a$.
- Least-squares (per node $i$, observational part): $[\mathbf{B}(t)]_i = [\mathbf{V}_i(t)]^{-1} \sum_{s\le t:\, i\notin a_s} \mathbf{X}_{\overline{\mathrm{pa}}(i)}(s)\,X_i(s)$, with regularized Gram matrix $\mathbf{V}_i(t)$ ($\lambda_{\min}\ge 1$); interventional weights $[\mathbf{B}^*(t)]_i$ estimated from rounds with $i \in a_s$.
- Confidence ellipsoid: $\mathcal{C}_i(t) = \{\boldsymbol{\theta}: \|\boldsymbol{\theta}\|\le m_B,\ \|\boldsymbol{\theta} - [\mathbf{B}(t-1)]_i\|_{\mathbf{V}_i(t-1)} \le \beta_T\}$.

**Main results / theorems:**
- **Upper bound (both algorithms):** $\mathbb{E}[R(T)] = \tilde{\mathcal{O}}\big(d^{\,L+1/2}\sqrt{NT}\big)$, where $d$ = max in-degree, $L$ = length of the longest causal path. Crucially the arm count $2^N$ does **not** appear — regret is polynomial in $N$.
- **Lower bound:** $\Omega\big(d^{\,L/2-2}\sqrt{T}\big)$, so the $\sqrt{T}$ rate and the qualitative $d,L$ scaling are near-optimal (gap is in the $d$-exponent).
- Thompson-Sampling variant attains matching *Bayesian* regret of the same order.

**Algorithms:**
- **LinSEM-UCB** (frequentist): each round, form per-node least-squares estimates and confidence ellipsoids, then play the intervention whose *optimistic* reward $\max_{\Theta_i \in \mathcal{C}_i(t)} f(\Theta)$ is largest (optimism in the face of uncertainty over the path-sum reward).
- **LinSEM-TS** (Bayesian): sample weight matrices from posterior ellipsoids and play the arm maximizing the sampled $f(\cdot)$.

**Relevance to pgmpy:** This is the most direct match to pgmpy's `LinearGaussianCPD` / `LinearGaussianBayesianNetwork`: the SEM $\mathbf{X} = \mathbf{B}^\top\mathbf{X}+\boldsymbol{\varepsilon}$ with per-node linear-Gaussian mechanisms is exactly what those classes represent, and soft interventions correspond to replacing a node's `LinearGaussianCPD`. The path-sum reward $f(\mathbf{B}) = \sum_\ell [\mathbf{B}^\ell]$ is computable from the weighted adjacency of a `LinearGaussianBayesianNetwork`; the per-node least-squares estimator mirrors pgmpy's existing linear-Gaussian fitting. A causal-bandit module would add the sequential-intervention loop and confidence ellipsoids on top.

**Relation to causal incentives / CIDs:** The linear SEM is precisely the SCM substrate of the causal-incentives framework; the reward node $N$ is a utility node and each soft intervention is a decision/policy choice. The result that regret scales with $d$ and $L$ rather than $2^N$ is the bandit analogue of the CID insight that graphical structure (here in-degree and path length) bounds the effective complexity of the decision problem — the $2^N$ arm explosion is tamed by the SEM's factorization.

---

### Robust Causal Bandits for Linear Models (2023)
**Authors:** Zirui Yan, Arpan Mukherjee, Burak Varici, Ali Tajer
**Link:** arXiv:2310.19794
**TL;DR:** Extends the linear-SEM causal bandit to *non-stationary* (time-varying) models with a deviation budget $C$, giving a robust weighted-least-squares UCB algorithm with regret $\tilde{\mathcal{O}}\big(d^{\,L-1/2}(\sqrt{NT}+NC)\big)$ — sublinear whenever the total model drift $C$ is sublinear.

**Problem & motivation:** The Varici et al. guarantees assume the SEM weights are *fixed* across rounds; the authors show that even $T^{1/(2L)}$ rounds of deviation make a non-robust algorithm (LinSEM-UCB) suffer *linear* regret. This paper makes the learner robust to bounded, adversary-agnostic temporal fluctuations of the causal mechanisms, where deviations are unknown but their cumulative size is budgeted by $C$.

**Formal setup & key definitions:** Same **linear SEM** $\mathbf{X} = \mathbf{B}^\top\mathbf{X} + \boldsymbol{\varepsilon}$ with strictly-upper-triangular $\mathbf{B}$, known topology, unknown weights, reward = childless node $N$, soft interventions over $\mathcal{A} = 2^{\mathcal{V}}$. At round $t$ the *actual* model $\mathbf{D}_{a}(t)$ may differ from the nominal $\mathbf{B}_{a}(t)$; robustness is measured against the aggregated deviation $C$.

**Key equations:**
- Per-round deviation: $\boldsymbol{\Delta}_a(t) \triangleq \mathbf{D}_a(t) - \mathbf{B}_a(t)$, with row-wise bound $\|[\boldsymbol{\Delta}_a(t)]_i\| \le m_c$.
- Deviation Frequency budget: $C_{\mathrm{DF}} \triangleq \max_{i\in[N]} \sum_{t=1}^T \max_{a(t)} \mathbb{1}\{\|[\boldsymbol{\Delta}_a(t)]_i\| \neq 0\}$.
- Aggregate Deviation budget: $C_{\mathrm{AD}} \triangleq \max_{i\in[N]} \sum_{t=1}^T \max_{a(t)} \|[\boldsymbol{\Delta}(t)]_i\|$ (results stated for a unified budget $C$).
- Reward decomposition unchanged: $\mu_a = \langle f(\mathbf{B}_a), \boldsymbol{\nu}\rangle$, $f(\mathbf{A}) = \sum_{\ell=0}^{L} [\mathbf{A}^\ell]_N$.
- **Robust weighted-OLS weights:** $w_i(t) = \min\!\big\{\tfrac{1}{C},\ \tfrac{1}{C\,\|\mathbf{X}_{\mathrm{pa}(i)}(t)\|_{[\tilde{\mathbf{V}}_{i,a(t)}(t)]^{-1}}}\big\}$ — downweights high-leverage (potentially corrupted) samples.
- Confidence radius: $\beta_t(\delta) = \sqrt{2\log(1/\delta) + d\log(1 + m^2 t/(dC^2))} + 1 + m$.
- Optimistic arm choice: $\mathrm{UCB}_a(t) = \max_{\forall i:\,[\Theta]_i \in \mathcal{C}_{i,a}(t)} \langle f(\Theta), \boldsymbol{\nu}\rangle$.

**Main results / theorems:**
- **Upper bound:** $\mathbb{E}[R(T)] = \tilde{\mathcal{O}}\big(d^{\,L-1/2}(\sqrt{NT} + NC)\big)$ — note the $d$-exponent is improved to $L-1/2$ (versus Varici's $L+1/2$), and the extra $NC$ term is the unavoidable cost of drift. When $C = o(\sqrt{T})$ the rate is the near-optimal $\sqrt{T}$; when $C = \Omega(\sqrt{T})$ regret stays sublinear as long as $C$ is sublinear.
- **Lower bound:** $\Omega\big(d^{\,L/2-2}\max\{\sqrt{T},\, d^2 C\}\big)$.
- **Non-robustness of prior work:** an existing fixed-model algorithm incurs *linear* regret after only $\sim T^{1/(2L)}$ deviating rounds.

**Algorithms:** A robust UCB-style algorithm (Algorithm 1; built on Varici's LinSEM-UCB) combining (i) **weighted ordinary least squares (W-OLS)** estimating both observational $\mathbf{B}(t)$ and interventional $\mathbf{B}^*(t)$ weights, (ii) drift-aware sample weights $w_i(t)$ that filter outliers without explicit change detection, (iii) time-uniform confidence ellipsoids in a re-weighted norm, and (iv) optimistic intervention selection over the path-sum reward.

**Relevance to pgmpy:** Same linear-Gaussian core as the Varici paper, plus a *non-stationarity* layer relevant to any pgmpy workflow where a `LinearGaussianBayesianNetwork` is re-fit over time / streaming data. The weighted-least-squares estimator with outlier-robust weights is a natural robust-fitting option that could complement pgmpy's existing parameter estimators for linear-Gaussian models.

**Relation to causal incentives / CIDs:** Models the realistic case where the SCM/utility structure underlying a CID drifts over time; robustness to a deviation budget $C$ is the bandit analogue of designing decision policies stable under mechanism shift. The persistence of the $d^{\Theta(L)}$, $\mathrm{poly}(N)$ scaling (rather than $2^N$) reinforces that graphical structure, not raw arm count, governs both learnability and robustness of multi-decision causal problems.

---

### Combinatorial Causal Bandits (AAAI, 2023)
**Authors:** Shi Feng, Wei Chen
**Link:** arXiv:2206.01995 (AAAI 2023)
**TL;DR:** Defines causal bandits where each round intervenes on *up to $K$* nodes (an exponentially large combinatorial action space) under a **binary generalized linear model (BGLM)**, and gives the BGLM-OFU algorithm with $\tilde{\mathcal{O}}(\sqrt{T})$ regret polynomial in the number of nodes $n$.

**Problem & motivation:** Earlier causal bandits intervene on a single node; many applications (marketing, biology) require *simultaneous* interventions on several variables, blowing the action space up to $\sim\sum_{k\le K}\binom{n}{k}2^k$ arms. The paper asks how to get regret polynomial in $n$ (not in the arm count) by exploiting a parametric BGLM propagation model, and addresses the resulting computational difficulty of the offline optimization.

**Formal setup & key definitions:** A DAG over binary nodes $\mathbf{X}\cup\{Y\}$ with reward = target node $Y$. **Combinatorial intervention:** select a subset $S\subseteq\mathbf{X}$ with $|S|\le K$ and set its values (do$(S=s)$), then observe *all* node values and the reward $Y$. The propagation follows a **BGLM**: each node's activation probability is a monotone link function of a linear combination of its (binary) parents. Objective is cumulative expected regret on $Y$.

**Key equations:**
- BGLM node model: $\Pr(X=1 \mid \boldsymbol{Pa}(X)=\boldsymbol{pa}(X)) = f_X(\boldsymbol{\theta}_X^* \cdot \boldsymbol{pa}(X)) + \varepsilon_X$, where $f_X$ is scalar, monotonically non-decreasing, twice-differentiable; $\boldsymbol{\theta}_X^* \in [0,1]^{|\boldsymbol{Pa}(X)|}$ unknown; $\varepsilon_X$ zero-mean sub-Gaussian.
- Link-derivative lower bound (Assumption 2): $\kappa = \inf_{X,\,\boldsymbol{v},\,\|\boldsymbol{\theta}-\boldsymbol{\theta}_X^*\|\le 1} \dot f_X(\boldsymbol{v}\cdot\boldsymbol{\theta}) > 0$.
- Reward objective: maximize $\mathbb{E}[Y \mid \mathrm{do}(S=s)]$ over $|S|\le K$.
- **MLE / pseudo-log-likelihood estimating equation:** solve $\sum_{i=1}^t \big(X^i - f_X(\boldsymbol{V}'_{i,X}\boldsymbol{\theta}_X)\big)\boldsymbol{V}_{i,X} = 0$ per node, then build confidence ellipsoids around $\hat{\boldsymbol{\theta}}_X$.

**Main results / theorems:**
- **Theorem 1 (BGLM-OFU, Markovian BGLM):** $R(T) = \mathcal{O}\!\big(\tfrac{1}{\kappa}\, n\, L_{\max}^{(1)}\, \sqrt{D\,T}\,\log T\big) = \tilde{\mathcal{O}}(\sqrt{T})$, where $n$ = number of nodes, $D$ = maximum in-degree, $L_{\max}^{(1)}$ = a bounded-smoothness (Lipschitz) constant of the reward in the parameters (adapted from online influence maximization), $\kappa$ = link-derivative lower bound. Regret is **polynomial in $n$**, never in the $2^{\Theta}$ arm count.
- **Linear / hidden-variable case (BLM-LR):** for binary linear models with hidden variables, a least-squares variant achieves $R(T) = \mathcal{O}(n\sqrt{D}\,\sqrt{T}\log T)$ (a higher-degree polynomial-in-$n$ variant arises under the looser assumptions).
- **Learning lemma:** once $\lambda_{\min}(M_{t,X})$ exceeds a threshold $\propto |\boldsymbol{Pa}(X)|L_{f_X}^2/\kappa^4\,(|\boldsymbol{Pa}(X)|^2 + \ln(1/\delta))$, the MLE concentrates inside the confidence region.
- **Computational note:** the per-round $\arg\max$ jointly over the intervention set $S$ and parameter vector $\boldsymbol{\theta}'$ is computationally hard at scale; the paper handles tractable special cases (e.g. linear models) rather than claiming a general efficient oracle.

**Algorithms:**
- **BGLM-OFU** (Algorithms 1+2): an initialization phase of $T_0$ pure-observation rounds, then optimism-in-the-face-of-uncertainty using MLE + confidence ellipsoids, selecting the optimistic intervention set each round.
- **BLM-LR** (Algorithm 3): a least-squares / linear-regression variant for binary linear models with hidden variables (drops the strict-monotone-link assumption).

**Relevance to pgmpy:** The BGLM is essentially a discrete Bayesian network with *parametric* (logit/probit-style) CPDs — close to pgmpy's discrete CPDs and `NoisyORCPD`; the monotone-link binary mechanism could be a new parametric CPD family. The combinatorial action space (choose $\le K$ nodes for do-interventions and observe all nodes) and the per-node MLE estimator are both new relative to pgmpy's current estimators, but the do$(S=s)$ semantics align with pgmpy's existing intervention/`CausalInference` machinery.

**Relation to causal incentives / CIDs:** Intervening on $\le K$ simultaneous nodes is exactly a **multi-decision** CID: each intervened node is a decision node and $Y$ is the utility node. The BGLM is a parametric SCM/SCIM. The headline that regret is $\mathrm{poly}(n)$ despite an exponential combinatorial arm space is the bandit-side statement of why structured (factorized) decision problems are tractable — graphical structure (in-degree $D$, smoothness $L_{\max}^{(1)}$) replaces brute-force enumeration over joint decisions.

---

### Combinatorial Pure Exploration of Causal Bandits (ICLR, 2023)
**Authors:** Nuoya Xiong, Wei Chen
**Link:** arXiv:2206.07883 (ICLR 2023)
**TL;DR:** Gives the first *gap-dependent, fully adaptive* fixed-confidence best-intervention-identification algorithms for combinatorial causal bandits on both BGLM and general (hidden-variable) graphs, with polynomial-in-graph-size sample complexity controlled by a gap-and-observation hardness quantity.

**Problem & motivation:** Pure exploration (best-arm identification) for causal bandits had only worst-case, two-stage (separate observe-then-explore) results with exponential dependence. This paper seeks $(\epsilon,\delta)$-PAC identification of the best combinatorial intervention whose sample complexity adapts simultaneously to (i) the reward gaps $\Delta_a$ and (ii) how informative passive observation is for each arm, removing the rigid two-stage split.

**Formal setup & key definitions:** Causal graph $G=(\mathbf{X}\cup\{Y\}\cup\mathbf{U},E)$ with all observable (and hidden $\mathbf{U}$) variables binary; $Y$ is the reward. Two regimes: **BGLM** (same monotone-link model as Feng & Chen, no hidden confounding) and **general graphs** (hidden variables, causal effects identified via *admissible sequences* from observational data). Actions are do$(S=s)$ interventions (including the null/observational action do()). **Objective ($(\epsilon,\delta)$-PAC):** with probability $\ge 1-\delta$, output $a^o$ with $\mu^* - \mu_{a^o} \le \epsilon$, minimizing the number of rounds.

**Key equations:**
- BGLM node model (as before): $\Pr(X=1\mid \boldsymbol{Pa}(X)=\boldsymbol{pa}(X)) = f_X(\boldsymbol{\theta}_X\cdot\boldsymbol{pa}(X)) + e_X$, $f_X$ strictly increasing, $e_X$ bounded zero-mean.
- Gap: $\Delta_a = \mu_{a^*} - \max_{a\neq a^*}\mu_a$ for $a=a^*$, and $\Delta_a = \mu_{a^*} - \mu_a$ for $a\neq a^*$.
- Hardness sum: $H_r = \sum_{i=1}^{r} \dfrac{1}{\max\{\Delta_{a_i},\,\epsilon/2\}^2}$ (arms ordered by increasing gap).
- **Gap-dependent observation threshold** (the key new quantity): $m_{\epsilon,\Delta} = \min\big\{\tau\in[|\mathbf{A}|] : \big|\{a\in\mathbf{A}\mid q_a\cdot\max\{\Delta_a,\epsilon/2\}^2 < 1/H_\tau\}\big| \le \tau\big\}$, where $q_a$ is the (graph-structure-dependent) probability that a passive observation yields usable information about arm $a$.

**Main results / theorems:**
- **Theorem 1 (BGLM):** sample complexity $T = \mathcal{O}\!\Big(H_{m^{(L)}_{\epsilon,\Delta}}\,\log\!\big(|\mathbf{A}|\,H_{m^{(L)}_{\epsilon,\Delta}}/\delta\big)\Big)$, where $m^{(L)}_{\epsilon,\Delta}$ is the BGLM gap-dependent observation threshold. This is **polynomial in graph size**, versus exponential bounds in prior pure-exploration causal-bandit work.
- **General graphs:** an analogous adaptive-threshold bound using admissible-sequence causal-effect estimation under hidden variables, giving a "significant improvement" over prior sample complexity.
- **Lower bound:** a matching gap-dependent lower bound is proven, so the algorithms "nearly match" the optimum.

**Algorithms:**
- **CCPE-BGLM** (Combinatorial Causal Pure Exploration for BGLM): interleaves, per round, (1) passive observation + MLE of BGLM parameters, (2) two LUCB-style interventional plays on the current best/contender arms, (3) confidence-interval merging — adapting to both gaps and observational feedback.
- **CCPE-General:** extends the same scheme to general graphs, replacing direct estimation with admissible-sequence-based causal-effect estimation from observational data.

**Relevance to pgmpy:** The BGLM regime maps to pgmpy discrete BNs with parametric CPDs; the general-graph regime with hidden variables and *admissible sequences* connects directly to pgmpy's `CausalInference`/identification machinery (`Adjustment`, frontdoor, ADMG/MAG handling). The $q_a$ observation-probability quantity is computed from the graph's identifiability structure — exactly the kind of quantity pgmpy's causal-identification code already reasons about. A pure-exploration/best-intervention API would be a new addition.

**Relation to causal incentives / CIDs:** Best-intervention identification is the "find the optimal policy" problem of a (multi-decision) CID solved by sampling rather than planning; the combinatorial action set $\mathbf{A}$ corresponds to joint settings of multiple decision nodes, and $Y$ is the utility node. The result that sample complexity is governed by gaps and the structure-dependent observation probabilities $q_a$ (polynomial in graph size, not in $|\mathbf{A}|$) is the pure-exploration counterpart of the CID principle that graphical structure — here both the reward-gap geometry and observational identifiability — determines how hard the decision problem is, taming the exponential arm/decision space.

## F. Contextual, Budgeted & General Models (+ survey)

### Causal Contextual Bandits with Targeted Interventions (ICLR, 2022)
**Authors:** Chandrasekar Subramanian, Balaraman Ravindran (IIT Madras / Robert Bosch Centre for DS&AI)
**Link:** https://openreview.net/pdf?id=F5Em8ASCosV
**TL;DR:** The first contextual bandit that combines causal-graph side-information with *targeted interventions* (the agent picks both an action and the sub-population, specified by context values, to experiment on) and learns a context→action policy minimizing simple regret, guided by a novel entropy-like information-leakage measure `Unc`.

**Problem & motivation:** Standard contextual bandits need many costly samples and ignore two real-world levers: (a) a known qualitative causal graph relating context, action, and reward, and (b) the ability during a "training phase" to run an experiment on a *chosen* sub-population rather than on a randomly-arriving user (e.g. software A/B tests on `os=iOS` users). The agent learns a *policy* (not a single best arm — the first causal-bandit work to do so), evaluated on a held-out round where only the main context is observed.

**Formal setup & key definitions:** Environment is a causal model $\mathcal{M}$ = DAG $G$ over the action variable $X$, reward $Y$, and context $\mathbf{C}=\mathbf{C}^{tar}\cup\mathbf{C}^{other}$ (main vs. auxiliary context), plus a factorizing joint $P$. All variables finite/categorical (so linearity-based generalization is invalid). The agent knows $G$ but not the CPDs. A **targeted intervention** $(x,\mathbf{c}^{tar})$ applies $do(X{=}x)$ conditioned on $\mathbf{C}^{tar}{=}\mathbf{c}^{tar}$, sampling $P(\cdot\mid do(x),\mathbf{C}^{tar}{=}\mathbf{c}^{tar})$; a **standard interaction** observes $\mathbf{c}^{tar}\sim P(\mathbf{C}^{tar})$ then samples $P(\cdot\mid do(x),\mathbf{c}^{tar})$. After $T$ training rounds the agent outputs $\hat\phi:\mathrm{val}(\mathbf{C}^{tar})\to\mathrm{val}(X)$; auxiliary context $\mathbf{C}^{other}$ is observable only in training. The graph induces *information leakage*: one targeted intervention updates beliefs about many others via shared CPDs.

**Key equations:**
- Simple-regret objective: $\mathrm{Regret} \triangleq \sum_{\mathbf{c}^{tar}} [\mu^*_{\mathbf{c}^{tar}} - \hat\mu_{\mathbf{c}^{tar}}]\,P(\mathbf{c}^{tar})$, with $\mu^*_{\mathbf{c}^{tar}}=E[Y\mid do(\phi^*(\mathbf{c}^{tar})),\mathbf{c}^{tar}]$.
- Reward factorization (Eq. 2, under A1): $E[Y\mid do(x),\mathbf{c}^{tar}] = \sum_{\mathbf{c}^{other}} P(Y{=}1\mid x,\mathbf{c}\langle PA_Y\rangle)\prod_{c\in\mathbf{c}^{other}} P(C{=}c\mid \mathbf{c}\langle PA_C\rangle)$.
- Knowledge measure: $\mathrm{Ent}(P(V|pa_V)) \triangleq -\sum_i \frac{\theta_{V|pa_V}[i]}{\sum_j\theta_{V|pa_V}[j]}\ln\frac{\theta_{V|pa_V}[i]}{\sum_j\theta_{V|pa_V}[j]}$ (entropy of normalized Dirichlet pseudo-counts); `Unc` aggregates the post-intervention change in `Ent` over all CPDs touched by $(x,\mathbf{c}^{tar})$, weighted by $\hat P(\mathbf{c}')\,\hat E[Y\mid \mathbf{c}',do(x')]$ — an expected information-gain about the *value* of every other targeted intervention.

**Main results / theorems:** Theorem 3.1: for any $0<\delta<1$, w.p. $\ge 1-\delta$, $\mathrm{Regret} \le 3\,E_{pa_Y,\mathbf{c}^{tar}}\!\sqrt{\tfrac{2}{(\alpha T/N_X)P(pa_Y,\mathbf{c}^{tar})-\epsilon^T_{X,PA_Y}}\ln\tfrac{2N_X(N_C+|\mathbf{C}|)}{\delta}}\; +\; 3\sum_{C\in\mathbf{C}^{other}} E_{pa_C,\mathbf{c}^{tar}}\!\sqrt{\tfrac{2}{\alpha T\,P(pa_C,\mathbf{c}^{tar})-\epsilon^T_{PA_C}}\ln\tfrac{2(N_C+|\mathbf{C}|)}{\delta}}$. Qualitatively $\mathrm{Regret}\to 0$ as $T\to\infty$, scales like $1/\sqrt{T}$ and (loosely) $\propto \sqrt{N_X N_C \ln(N_X N_C)}$. Proven only as a performance guard (no comparable contextual simple-regret baseline exists); empirical gains over six baselines are largest at small budget (~35% lower regret than next-best at $T{=}24$).

**Algorithms:** **Unc-CCB**. *Training Phase 1* (fraction $\alpha$ of $T$): observe context, pick $x$ uniformly at random, observe all variables, do Bayesian Dirichlet count updates on every CPD. *Phase 2* (remaining $1-\alpha$): each round pick $\arg\min_{x,\mathbf{c}^{tar}} \sum_{x',\mathbf{c}^{tar'}} \mathrm{Unc}\!\big(E[Y\mid do(x'),\mathbf{c}^{tar'}]\,\big|\,x,\mathbf{c}^{tar}\big)$, execute that targeted intervention, update beliefs. *Evaluation*: return $\hat\phi(\mathbf{c}^{tar})=\arg\max_x \hat E[Y\mid do(x),\mathbf{c}^{tar}]$ via Eq. 2. Assumptions: edge $X\to Y$ with no other directed $X\rightsquigarrow Y$ path and no context caused by $X$; (A1) any $C$ confounding some $C'\in\mathbf{C}^{tar}$ and $Y$ must itself be in $\mathbf{C}^{tar}$ (sufficient: $\mathbf{C}^{tar}$ ancestral).

**Relevance to pgmpy:** A near-perfect fit for a `DiscreteBayesianNetwork` + `CausalInference.query(do=...)` substrate: context = observed evidence (condition on $\mathbf{C}^{tar}$ before choosing $do(X)$); targeted intervention = `do(X=x)` *combined with* evidence on a context subset; the Dirichlet belief updates are exactly `BayesianEstimator`-style counting; Eq. 2 is variable elimination on the post-intervention BN. A library would need: node-role tags (action/reward/context-main/context-aux), a do+condition query, and an info-gain acquisition score (`Unc`) over the (action × context-subset) grid.

**Relation to causal incentives / CIDs:** This is the canonical *information-link* CID: the agent observes context (a parent of the reward) before its decision node, so the optimal object is a *policy* over contexts, not a single arm. `Unc` is a literal **value-of-information / expected-information-gain** acquisition criterion — choosing which $(x,\mathbf{c}^{tar})$ experiment maximally reduces uncertainty about the *reward-relevant* CPDs, mediated by the graph's d-separation structure. Targeted interventions enrich the decision domain beyond a CID's usual action set: the agent chooses both the intervention and the sub-population it acts on.

---

### Budgeted and Non-budgeted Causal Bandits (AISTATS, 2021)
**Authors:** Vineet Nair, Vishakha Patil, Gaurav Sinha
**Link:** https://arxiv.org/abs/2012.07058
**TL;DR:** Studies causal bandits where interventions cost more than observations: under a budget it minimizes simple regret by optimally trading off cheap observations against costly interventions (exploiting "no-backdoor" leakage), and in the non-budgeted/general-graph case achieves *constant* (horizon-independent) cumulative regret when the reward's parent distribution is known.

**Problem & motivation:** In practice intervening (running a real experiment) is far more expensive than passively observing. Prior causal bandits ignore this asymmetry. The paper introduces an explicit cost/budget model and asks: given a fixed budget, how should an agent split spend between observations and interventions to find the best intervention (simple regret), and separately, can causal structure give horizon-independent cumulative regret on general graphs?

**Formal setup & key definitions:** DAG $G$ over $\mathbf{X}=\{X_1,\dots,X_n\}$ with reward $Y$; arms are interventions $do(X_i{=}x)$ plus the observation/do-nothing arm $a_0$; $M$ = number of arms; $\mu_a=E[Y\mid do(X_i{=}x)]$. **Cost model:** each observation ($a_0$) costs $1$, each non-empty intervention costs $\gamma > 1$; total budget $B$. **No-backdoor assumption** (budgeted setting): no backdoor path from an intervenable node to $Y$, so $E[Y\mid do(X_i{=}x)] = E[Y\mid X_i{=}x]$ — hence pure observations also estimate interventional rewards (causal information leakage). $p=\min_{i,x} p_{i,x}$ is the minimum observational probability of any value.

**Key equations:**
- Budget recursion: $B_{t+1} = B_t - 1$ if $a_t = a_0$ (observe), $B_{t+1} = B_t - \gamma$ if $a_t \neq a_0$ (intervene).
- Simple regret: $r_{ALG}(B) = \max_{a\in\mathcal{A}} \mu_a - \mu_{a_B}$; cumulative regret: $R_{ALG}(B) = G_B - G_{ALG}(B)$, where $G_B$ is the optimal total reward attainable within budget.
- Effective sample count (leakage): $E^{i,x}_t = N^{i,x}_t + \sum_{s=1}^t \mathbf{1}\{a_s = a_0 \text{ and } X_i = x\}$ — observations augment the count for intervention $(i,x)$ under no-backdoor.
- UCB index: $\bar\mu_{i,x}(t) = \tfrac{1}{\gamma}\big(\hat\mu_{i,x}(t) + \sqrt{8\log t / E^{i,x}_t}\big)$; general-graph estimate $\hat\mu_a(t) = \sum_{\mathbf{y}} \hat\mu_{\mathbf{y}}(t)\,P(Pa(Y){=}\mathbf{y}\mid do(a))$.

**Main results / theorems:**
- **Theorem 1 (OBS-ALG, observation-only simple regret):** $O\!\big(\sqrt{\tfrac{1}{pB}\log(pMB)}\big)$.
- **Theorem 2 (γ-NB-ALG, budgeted trade-off):** if $\gamma \ge \tfrac{1}{p\cdot m(p)}$ then $O\!\big(\sqrt{\tfrac{1}{pB}\log(pMB)}\big)$ (observation-dominated); if $\gamma \le \tfrac{1}{p\cdot m(p)}$ then $O\!\big(\sqrt{\tfrac{\gamma\, m(p)}{B}\log\tfrac{MB}{\gamma\, m(p)}}\big)$ ($m(p)$ counts hard-to-observe arms).
- **Theorem 3 (CRM-NB-ALG, cumulative regret, no-backdoor):** if the optimal arm is the observation arm $a^*=a_0$, expected cumulative regret is $O(1)$ (constant); otherwise instance-dependent logarithmic regret.
- **Theorem 4 (C-UCB-2, general graphs):** when the parent distribution $P(Pa(Y)\mid do(a))$ is known for each arm, *constant* expected cumulative regret $\sum_{a}\Delta_a\big(L_a + \tfrac{2\pi^2}{3}\big)$, where $L_a$ depends only on instance parameters and **not** on the horizon $T$.

**Algorithms:** (1) **γ-NB-ALG** (budgeted simple regret): spend ~$B/2$ on observations to estimate $\hat p,\,\widehat{m(p)}$; if $\hat p\,\widehat{m(p)} \ge 1/\gamma$ keep observing, else allocate to low-observational-probability interventions; output highest empirical mean. (2) **CRM-NB-ALG** (cumulative regret, no-backdoor): pull each arm once; force $a_0$ to be played $\ge \beta^2\log t$ times (leakage exploration), else play highest weighted-UCB arm; adapt $\beta$ from the observed gap. (3) **C-UCB-2** (general graphs): round-robin init, maintain parent-config estimates $\hat\mu_{\mathbf{y}}$, compute $\hat\mu_a=\sum_{\mathbf y}\hat\mu_{\mathbf y}P(Pa(Y){=}\mathbf y\mid do(a))$, pull $\arg\max_a \bar\mu_a$. Assumptions: known graph, binary reward $Y\in\{0,1\}$, $p>0$, no-backdoor (NB algorithms), known $P(Pa(Y)\mid do(a))$ for the constant-regret Theorem 4.

**Relevance to pgmpy:** Cleanly maps onto a BN where each node carries **intervention-cost metadata** ($\gamma$) and the agent has a **budget** resource. The no-backdoor condition is checkable via `CausalInference`/backdoor-adjustment machinery, and the leakage identity $E[Y|do(X_i)]=E[Y|X_i]$ is exactly a "no adjustment needed" query. $\hat\mu_a=\sum_{\mathbf y}\hat\mu_{\mathbf y}P(Pa(Y)|do(a))$ is one-step variable elimination over $Pa(Y)$. A library needs: per-node cost annotation, a budget-tracking interaction loop, and an estimator pooling observational + interventional counts.

**Relation to causal incentives / CIDs:** This is the **cost-aware decision** axis of a CID — each decision (intervention) has a resource cost $\gamma$ and the agent optimizes utility under a budget constraint, while the cheap "observe" action is a degenerate decision that still yields information. The no-backdoor assumption is precisely the CID condition under which the information link $X_i\to Y$ is unconfounded, so observational data identifies the interventional value. Constant cumulative regret reflects that perfect structural knowledge collapses the exploration cost — a CID with a fully specified mechanism needs no exploration of $Pa(Y)$.

---

### Causal Bandits with General Causal Models and Interventions (AISTATS, 2024)
**Authors:** (AISTATS 2024; arXiv 2403.00233) — generalizes SCM classes and intervention types.
**Link:** https://arxiv.org/abs/2403.00233
**TL;DR:** Causal bandits where the SCM is *unknown* and drawn from a general class $\mathcal{F}$ of Lipschitz functions (not just linear) and interventions are *generalized soft* with arbitrary granularity (infinitely many), with matching UCB/Thompson upper bounds and a minimax lower bound expressed via the function class's eluder dimension and covering number.

**Problem & motivation:** Prior causal bandits assume restrictive mechanism families (linear / generalized-linear) and atomic/hard interventions, and pay $O(\sqrt N)$ or $O(N)$ in graph size $N$. This work asks for regret guarantees under *arbitrary* nonlinear (Lipschitz) mechanisms and *continuous soft* interventions, characterized only by general complexity measures, while reducing the dependence on $N$.

**Formal setup & key definitions:** Known DAG $\mathcal{L}(\mathcal V,\mathcal E)$ with $N$ nodes; SCM $X_i = f_i(X_{pa(i)}) + \varepsilon_i$, $f_i\in\mathcal F_i$ (unknown, Lipschitz), $\varepsilon_i$ independent sub-Gaussian. **Generalized soft interventions** at node $i$: $X_i = f_i(X_{pa(i)}; a_i)+\varepsilon_i$ with $a_i\in\mathcal A_i\subseteq\mathbb R$ (continuous, infinite intervention space; atomic interventions are a special case). Reward $\mu_{\mathbf a} \triangleq E_{\mathbf a}[X_N]$ at the sink node $X_N$; optimum $\mathbf a^* \triangleq \arg\max_{\mathbf a\in\mathcal A}\mu_{\mathbf a}$. Structural parameters: $d$ = max in-degree, $L$ = longest causal-path length, $K_i$ = Lipschitz constant, $C_i$ = boundedness constant. Complexity: **eluder dimension** $\dim(\mathcal F,\varepsilon)$ and **$\alpha$-covering number** $cn_\alpha(\mathcal F_i)$.

**Key equations:**
- Cumulative / Bayesian regret: $R(T)=\sum_{t=1}^T (\mu_{\mathbf a^*}-\mu_{\mathbf a(t)})$; $BR(T)=E_{\mathcal F}E[R_f(T)]$.
- Lipschitz propagation factor: $K=\prod_{\ell=2}^L K^{(\ell)}$ with $K^{(\ell)}=\max_{i:L_i=\ell}K_i$.
- Per-node estimation-error term: $\mathcal B_i(\mathcal F_i,\delta) = 1 + \min\{\dim(\mathcal F_i,\alpha_i),T\}\,C_i + 4\sqrt{\dim(\mathcal F_i,\alpha_i)\,\beta_T(\mathcal F_i,\alpha_i,\delta)\,T}$.

**Main results / theorems:**
- **Theorem 5.3 (GCB-UCB upper bound):** $E[R(T)] = O\!\big(K\, d^{\,L-1}\sqrt{T\,\dim(\mathcal F)\log(NT\, cn(\mathcal F))}\big)$.
- **Corollary 5.4 (Bayesian regret / GCB-TS):** same rate for Thompson Sampling.
- **Theorem 5.5 (linear SCMs, refined):** $\dim(\mathcal F_i),\log cn(\mathcal F_i)=\tilde O(d\log T)$ gives $O\!\big(K\,d^{\,L}\sqrt{T\log T\,\varpi_L}\big)$ with $\varpi_L=\log T\,[1+\tfrac{\log N}{d\log T}]$ — graph-size dependence shrinks to $\sqrt{\log N/\log T}$ (vs. $\sqrt N$ in Sussex et al. 2023, $N$ in Varici et al. 2023).
- **Theorem 5.6 (minimax lower bound, linear):** $E[R(T)] \ge \Omega\!\big(K\, d^{\,L/2-1}\sqrt{T}\big)$ — tight up to log factors and a $d$-gap.
- **Theorems 5.7 / 5.9:** degree-2 polynomial SCMs $O\!\big(K\,d^{\,L+1}\sqrt{T\log T\,\varpi_P}\big)$; width-$s$ neural-net SCMs $O\!\big(\bar K\,d^{\,L-1}\sqrt{T\log T\,\varpi_N}\big)$.

**Algorithms:** **GCB-UCB** (Alg. 1): per node, least-squares fit $\tilde f_{i,t}=\arg\min_{f_i\in\mathcal F_i}\sum_{s<t}[f_i(X_{pa(i)}(s);a_i(s))-X_i(s)]^2$; build confidence set $\mathcal C_{i,t}$; act optimistically $\mathbf a(t)=\arg\max_{\mathbf a}\max_{f\in\mathcal C_t}E_{\mathbf a}[X_N\mid f]$. **GCB-TS** (Alg. 2): sample $\tilde f$ from the posterior, then act $\mathbf a(t)=\arg\max_{\mathbf a}E_{\mathbf a}[X_N\mid\tilde f]$. Assumptions: known DAG; Lipschitz + bounded mechanisms; additive sub-Gaussian noise; soft/parametric interventions; finite eluder dim and covering number.

**Relevance to pgmpy:** This is the **SCM substrate** generalization — pgmpy's `FunctionalBayesianNetwork`/`LinearGaussianBayesianNetwork` are exactly the function classes $\mathcal F$ (linear → `LinearGaussianCPD`; arbitrary Lipschitz → `FunctionalCPD`). Generalized soft interventions correspond to parameterized `do` on continuous nodes (set/shift a mechanism rather than fix a value). A library needs: continuous + arbitrary-mechanism CPDs, a "soft do" that re-parameterizes a node's mechanism, ancestral sampling to evaluate $E_{\mathbf a}[X_N]$, and per-node residual fitting (least-squares / posterior) for the UCB/TS loops.

**Relation to causal incentives / CIDs:** Sits at the top (SCM) layer of the model-class hierarchy: arbitrary functional mechanisms = a fully specified structural CID. Soft interventions generalize a CID's discrete decision domain to *continuous action spaces that reshape mechanisms*. The eluder-dimension regret captures the value-of-information cost of learning an unknown mechanism; the $d^{L-1}$/$K$ factors quantify how an intervention's influence propagates along directed paths to the utility node — directly the "downstream incentive / control" notion in causal-incentives analysis.

---

### Causality in Bandits: A Survey (ACM Computing Surveys, 2025)
**Authors:** Chandrasekar Subramanian, Balaraman Ravindran (IIT Madras)
**Link:** https://dl.acm.org/doi/10.1145/3744917 (ACM Comput. Surv. 57(12), 2025)
> **Caveat:** no open/arXiv version exists and the ACM full text is bot-protected; the top-level taxonomy below is confirmed from the abstract/indexing, but the finer axes are reconstructed from these authors' companion paper (the ICLR 2022 entry above) and field consensus, **not quoted from the survey body**.

**TL;DR:** The first survey dedicated to the bandits ∩ causal-inference intersection; it gives a taxonomy that first splits the field by **learning objective** (find one best arm vs. learn a context→action policy) and then by the causal assumptions (graph knowledge, intervention type, confounding, cost), placing the major works in that structure and listing open directions.

**Problem & motivation:** Bandit and causal-inference literatures developed largely independently; recent work shows treating arms as interventions on a causal graph yields information leakage across arms and provably better algorithms. The survey organizes this young field.

**Taxonomy (the organizing content — list, not equations):**
- **Primary axis — learning objective (confirmed):**
  - **Best-arm / single-best-intervention identification:** Lattimore et al. 2016 (parallel, atomic, simple regret), Yabe et al. 2018 (propagating inference), Sen et al. 2017 (importance sampling, soft interventions), Lu et al. 2020 (causal background knowledge).
  - **Policy learning** (context → action): Subramanian & Ravindran 2022 — noted as essentially the only entry, marking a sparse, open quadrant.
- **Secondary axes used to sub-classify works:**
  - *Causal-graph knowledge*: known vs. unknown/partial (parent-of-reward discovery then standard bandit) vs. combinatorial structure.
  - *Intervention type*: atomic/hard $do(X{=}x)$ vs. soft/stochastic vs. generalized/continuous; single-node vs. combinatorial multi-node (Feng & Chen 2023, Xiong & Chen 2023).
  - *Confounding*: none vs. unobserved confounders (Bareinboim et al. 2015; Maiti et al. 2022 — first regret analysis under UCs).
  - *Cost / budget*: unconstrained vs. budgeted with intervention costlier than observation (Nair et al. 2021; confounded-budgeted variants).
  - *Regret notion*: simple (pure-exploration) vs. cumulative.
  - *Context*: non-contextual MAB vs. contextual (context as causal side-information; targeted interventions on sub-populations).

**Main organizing claims & open problems:** Central claim — modeling arms as causal interventions enables *information leakage* (one pull updates beliefs about many arms via the graph factorization), the source of causal bandits' improved sample efficiency. Highlighted open directions: (1) **policy/contextual** causal bandits are largely unexplored (only one prior work); (2) **unknown / partially-known graphs** (joint causal discovery + bandit learning); (3) robustness to **distribution shift** between training/deployment ($G$ fixed, $\mathcal M$ changes); (4) **unobserved confounders** in richer settings; (5) **budget/cost-aware** and combinatorial-intervention settings; (6) unifying simple- and cumulative-regret guarantees with general intervention/model classes.

**Relevance to pgmpy:** The taxonomy *is* the design map for a causal-bandit module: the axes correspond directly to API knobs — graph knowledge (known BN vs. structure learning), intervention type (`do` atomic vs. soft/continuous on `FunctionalCPD`), confounding (latent-variable / ADMG support and backdoor checks), cost/budget (per-node cost metadata + budget loop), objective (best-arm vs. policy), and regret notion (evaluation metric). A pgmpy causal-bandit component should let a user instantiate any cell of this taxonomy on top of a `DiscreteBayesianNetwork`/`LinearGaussianBayesianNetwork`/`FunctionalBayesianNetwork` with a `CausalInference`-backed interaction loop.

**Relation to causal incentives / CIDs:** The taxonomy's two main branches mirror the CID/agent hierarchy: *best-arm* = a single decision node with no information links (choose one $do$); *policy/contextual* = a decision node with **information links** from observed context (value-of-information, policy over evidence). The secondary axes map onto the model-class layers of the causal-incentives review — known-graph/atomic = CBN-level, unknown-graph and general/soft interventions = SCM-level; confounding ↔ latent-variable CIDs; budget/cost ↔ cost-annotated decision nodes. The emphasis on information leakage is exactly the CID claim that graph structure (d-separation) determines which observations and interventions are mutually informative for the utility node.

---

## Glossary of recurring concepts

### Objectives
- **Simple regret** $R_T=\mu^*-\mathbb{E}[\mu_{\hat a_T}]$ — quality of the single arm *recommended* after a budget $T$ (pure exploration / best-arm identification).
- **Cumulative regret** $R_T=T\mu^*-\sum_t \mu_{a_t}$ — total reward lost *while* learning (online exploit-explore).
- **Best-arm identification** — find one optimal intervention; **policy learning** — find a context→action map (the rarer, harder objective).
- **Regret Decision Criterion (RDC) / Effect of Treatment on the Treated (ETT)** $\mathbb{E}[Y_{X=a}\mid X=x]$ — the counterfactual objective an agent must optimize under unobserved confounding (MABUC), not the interventional $\mathbb{E}[Y\mid do(a)]$.

### Intervention types (the action space)
- **Atomic / hard** $do(X_i=x)$ — fix one variable to a value (Lattimore, Maiti, Nair).
- **Soft / stochastic** — replace a node's CPD $P(V\mid pa)$ with a chosen one (Sen; General-Models).
- **Non-atomic** $A\in\{0,1,*\}^N$ — fix an arbitrary *subset* of variables at once (Yabe).
- **Combinatorial** — choose $\le K$ nodes to intervene on; action space $\sim\binom{n}{\le K}$ (Feng–Chen, Xiong–Chen).
- **Generalized soft** $X_i=f_i(pa;a_i)$, $a_i\in\mathbb{R}$ — continuous, mechanism-reshaping interventions (General-Models).
- **Targeted intervention** $(x,\mathbf c^{tar})$ — $do(X=x)$ on a *chosen sub-population* (context conditioning) (Subramanian–Ravindran).
- **Observation / null arm** $do()$ — passive sampling; cheap, and under no-backdoor it still estimates interventional rewards.

### Information leakage & estimators
- **Information leakage** — pulling one arm reveals values of other graph variables, so one sample informs *many* arms' reward estimates (the source of causal bandits' edge over structure-blind MAB).
- **Importance-weighted / clipped-IS estimator** $\hat\mu_a=\frac1T\sum_t Y_t R_a(X_t)\mathbb 1\{R_a\le B_a\}$, $R_a=P\{Pa_Y\mid a\}/Q\{Pa_Y\}$ — reweights a shared sample pool to estimate any intervention's value (Lattimore, Sen).
- **Propagating inference** — estimate native CPD parameters and propagate marginals through topological order to score every (overlapping) arm (Yabe).
- **C-UCB / C-TS** — UCB / Thompson indices computed on the reward-parent decomposition $\mu_a=\sum_{\mathbf z}\hat\mu_{\mathbf z}P(Pa_Y=\mathbf z\mid a)$ (Lu, Bilodeau).
- **Separating-set / information-sharing estimator** $\hat\mu_{IS}(\boldsymbol\zeta)=\sum_{\mathbf s}\hat\mu(\mathbf s)\hat p(\mathbf s\mid\boldsymbol\zeta)$ for $\mathbf I\perp Y\mid \mathbf S$ — unbiased, variance $\le$ sample mean (de Kroon).

### Structural-reduction objects (the "which arms matter" core)
- **MIS (minimal intervention set)** — $\mathbf X\subseteq an(Y)_{G_{\overline{\mathbf X}}}$; dropping any variable changes the achievable reward (Lee–Bareinboim).
- **POMIS (possibly-optimal MIS)** — an MIS that is optimal in *some* SCM consistent with $G$; characterized by $\mathrm{IB}(G_{\overline{\mathbf X}},Y)=\mathbf X$. **MUCT** = minimal UC-territory (descendant- + c-component-closed set containing $Y$); **interventional border (IB)** = its parents outside it. Only POMIS arms can be optimal.
- **POMPS (possibly-optimal mixed policy scope)** — generalizes POMIS to jointly choose *intervene-set* and *observe-set*; non-redundancy = (i) action $\in an(Y)$ + (ii) observed context d-connected to $Y$ (Lee–Bareinboim 2020).
- **c-component / latent projection / no-backdoor / d-separator** — confounding-structure objects shared with the causal-incentives review (ADMG bidirected edges, Tian–Pearl identifiability).

### Complexity constants (what regret scales with, instead of arm count)
- **$m(\mathbf q)$** (Lattimore): smallest $\tau$ with $\le\tau$ "rare" variables; $2\le m(\mathbf q)\le N$.
- **$m(\eta)$ / $\gamma^*$** (Lattimore general / Yabe): min-max importance-weight variance over sampling designs.
- **$m(\mathcal C)$ / $n(\hat{\mathbf q})$** (Maiti / Confounded-Budgeted): count of arms whose do-effect is *not* observationally identifiable (confounding cost), optionally cost-weighted.
- **$d, L$** (linear-SEM/general): max in-degree and longest causal-path length — regret $\sim d^{\Theta(L)}\sqrt{NT}$, polynomial in $N$ despite $2^N$ arms.
- **eluder dimension / covering number** (General-Models): complexity of an arbitrary Lipschitz mechanism class.
- **$\kappa$, $L_{\max}$** (BGLM): link-derivative lower bound and reward-smoothness constants.

### Model / payoff families
- **CBN** (tabular), **linear SEM** $\mathbf X=\mathbf B^\top\mathbf X+\boldsymbol\varepsilon$, **BGLM** (binary generalized linear model: monotone link of a linear combination of binary parents), **general Lipschitz SCM** $X_i=f_i(pa;a_i)+\varepsilon_i$ — increasing parametric generality, each enabling polynomial-in-structure regret.

---

## Synthesis: mapping to pgmpy and to causal incentives

> Literature synthesis — input to the design proposal, not the proposal itself.

### 1. The one-line bridge

**A causal bandit is online intervention-selection on a CID.** The static object is identical
to the causal-incentives review's central data structure — a causal graph with a designated
**reward/utility node** $Y$, **intervention/decision** variables, and observed **context/chance**
variables. The bandit adds the *temporal/learning* dimension the CID literature leaves implicit:
the agent repeatedly chooses an intervention (a do/decision), receives the utility, and updates
beliefs, optimizing **regret** rather than assuming the mechanism is known. Where the
causal-incentives papers ask "*given* the model, what does an optimal agent observe/control?",
the causal-bandit papers ask "*learning* the model online, which interventions should the agent
try, and how fast can it converge?". They are the static and dynamic faces of the same object.

### 2. Correspondence: causal-bandit concept ↔ causal-incentives / CID concept

| Causal bandit | CID / incentives analogue (from the companion review) |
|---|---|
| Arm = intervention $do(X=x)$ | Setting of a **decision node** |
| Reward $Y$ | **Utility node** |
| Context observed before acting | **Information link** into the decision |
| Observe-vs-do (MABUC); pulling $do()$ to learn | **Value of information**; non-intervened observation |
| "Where to intervene" / **MIS** ($\mathbf X\subseteq an(Y)$) | **Value of control / instrumental control incentive** (a target is worth controlling iff on a directed path to utility) |
| **POMIS** (interventional border under confounding) | The control-incentive frontier once latent confounding is accounted for |
| "What to observe" / **POMPS** non-redundancy | **Requisite observations** / value-of-information criterion for information links |
| Information leakage across arms | d-separation structure determining which observations/controls are mutually informative |
| Unobserved confounders, c-components, no-backdoor | **Latent-variable CIDs**; Tian–Pearl identifiability |
| Do-effect not identifiable ($m(\mathcal C)$ arms) | **Partial identification** — min/max of the query over consistent SCMs ("Limits of Predicting Agents") |
| Soft / generalized interventions | Continuous **decision domain**; mechanism-reshaping policy |
| Intervention cost $\gamma$ / budget $B$ | **Cost-aware decision** under a resource constraint |
| Regret vs an oracle that knows the mechanism | Sub-optimality of a policy under model uncertainty |

The deepest shared insight, stated twice: **graphical structure, not raw action count, governs
the difficulty of a causal decision problem.** In the incentives review this is the
minimal-reduction / requisite-graph collapse; in causal bandits it is regret scaling with
$m(\mathbf q)$, POMIS-count, or $d^{L}\,\mathrm{poly}(N)$ instead of $|\mathcal A|=2^N$.

### 3. Mapping to pgmpy — what exists vs. what is new

Causal bandits reuse pgmpy's *modeling + inference + identification* stack almost wholesale; the
genuinely new surface is the **online decision loop** and a few estimator/enumeration primitives.

| Capability | In pgmpy today | Needed for causal bandits |
|---|---|---|
| Graph + typed reward/action/context nodes | `DiscreteBayesianNetwork`, `base/` role annotations | role tags (action/reward/context); shared with CID module |
| Interventions | `CausalInference` / `do()` / `simulate(do=...)` (atomic) | **soft / generalized / non-atomic / combinatorial** `do` (CPD/mechanism override) |
| Payoff families | `Discrete`, `LinearGaussianCPD`, `FunctionalCPD` | BGLM (monotone-link binary) CPD family; otherwise covered |
| Reward decomposition $\sum_{\mathbf z}\hat\mu_{\mathbf z}P(Pa_Y\!\mid\! a)$ | variable elimination / BP | — (reuse) |
| Confounding | `ADMG`/`MAG`, c-components, `Adjustment`, `Frontdoor`, ID-algorithm | POMIS/MUCT/interventional-border enumeration; latent projection |
| Structure knowledge | `PC`, `GES`, `FCI`, `ci_tests`, separating sets | **interventional / online** discovery; central-node experiments |
| Online estimation & decisions | **absent** (estimators are batch/offline) | UCB / kl-UCB / Thompson index objects; importance-sampling & info-sharing estimators; **regret accounting**; the interaction loop |
| Resources | absent | per-node **intervention-cost** metadata + **budget** loop |
| Acquisition / VoI scoring | partial (identification) | `Unc`-style expected-information-gain over (action × context) |

So a `pgmpy.causal_bandits` (or `causal_decision`) module is mostly a **thin sequential-decision
layer** over existing machinery, plus the POMIS/POMPS graph algorithms and a soft-intervention API.

### 4. The two reviews share a core — argue for one design, not two

The causal-incentives review converged on a layered stack (L0 typed graph · L1 CPDs/policies/
interventions · L2 graphical-criteria engine incl. requisite graph & d-separation · L4
counterfactual/partial-ID). **Causal bandits need exactly L0–L2 and L4, and add one new layer:**

- **L0–L1 (shared):** the typed causal graph + intervention/`do` machinery is identical. Soft and
  combinatorial interventions are the *same* extension both literatures want.
- **L2 (shared):** POMIS = the value-of-control criterion; POMPS = value-of-information + control
  criteria together. These are the *same* graph algorithms the CID incentive-detectors need
  (ancestors, directed paths to utility, requisite observations, c-components). Build once, use
  in both.
- **L4 (shared):** confounded bandits' fallback to causal-effect **bounds** is the same
  partial-identification machinery the "Limits of Predicting Agents" CID paper needs.
- **L7 — NEW (bandit-specific): an online decision/learning layer** — UCB/TS/kl-UCB index
  objects, importance-sampling / info-sharing estimators, regret accounting, budget tracking,
  and the experiment-selection (VoI acquisition) loop. This is what neither pgmpy nor the
  CID-incentives work currently has, and is the principal new contribution of a bandit module.

Implication for the proposal: a single **"causal decision-making" subsystem** is the natural home
for both — CIDs/incentives provide the static analysis (what's worth observing/controlling),
causal bandits provide the dynamic learning (how to find it online), over a shared graph +
intervention + identification core. PyCID (from the incentives review) shows the
subclass-`BayesianNetwork` integration pattern; a bandit loop sits on top of the same classes.

### 5. Natural phasing (for the proposal to refine)

1. **Phase 1 — known-graph, atomic, single-objective.** Reward/action/context node roles; an
   interaction loop with UCB/kl-UCB/Thompson; the reward-parent decomposition estimator; simple
   *and* cumulative regret accounting. Reproduces Lattimore, Lu (C-UCB/C-TS), MABUC (with the
   ETT/counterfactual estimator reusing pgmpy counterfactual queries).
2. **Phase 2 — structural arm reduction (L2).** POMIS / MUCT / interventional-border enumeration
   over ADMGs; run the bandit only over POMIS arms; POMPS for the contextual/policy objective.
   Directly shared with the CID incentive-criteria module.
3. **Phase 3 — confounding & partial identification (L4).** c-component estimators, identifiable
   do-from-observation reuse, and causal-effect **bounds** when unidentifiable; d-separator
   adaptivity (HAC-UCB), separating-set information-sharing.
4. **Phase 4 — costs, budgets, soft/combinatorial/continuous interventions, parametric payoffs.**
   Per-node cost + budget loop; soft-intervention (CPD override) API; BGLM/linear-SEM/`FunctionalCPD`
   estimators; combinatorial action spaces.
5. **Phase 5 — unknown graph / online discovery.** Interventional structure learning interleaved
   with the bandit loop (central-node experiments; "partial structure discovery is enough");
   reuse `PC`/`FCI`/`ci_tests`.

The cheapest, highest-value first step mirrors the incentives review's: a thin known-graph bandit
loop reusing pgmpy's do-operator and inference, plus the POMIS reduction (Phase 1+2).

### 6. Open questions to settle in brainstorming (not decided here)

- **One subsystem or two?** Build causal bandits inside a shared `causal_decision` package with the
  CID/incentives work (shared L0–L2/L4), or as a standalone `causal_bandits` module that imports
  the graph/identification core?
- **Scope of v1:** known-graph atomic best-arm/cumulative bandits only, or include POMIS reduction
  and the contextual/policy objective from the start?
- **Online vs. offline boundary:** pgmpy is batch/offline today; how invasive is adding a stateful
  interaction loop + regret accounting, and should it follow a scikit-learn-style `partial_fit`/
  environment API or a callback-driven simulator?
- **Soft / continuous interventions:** unify the "soft do" (CPD/mechanism override) API across the
  bandit and CID/SCM work, and decide tabular-first vs. `FunctionalCPD`-first.
- **Reuse PyCID / the incentives module** for POMIS-adjacent graph algorithms, or reimplement
  natively on pgmpy's ADMG base?
- **Environment/simulator contract:** does pgmpy ship a "causal environment" (a BN the agent
  queries) for benchmarking regret, alongside the learner?

These connect both literature reviews to a single causal-decision-making design; resolve before
drafting the proposal (which follows the proposal template, not an implementation plan).
