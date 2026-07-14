# Causal Incentives Working Group — Literature Review

**Purpose:** A precise, equation-level summary of the research output of the
[Causal Incentives Working Group](https://causalincentives.com/), as the
literature foundation for a forthcoming pgmpy software design proposal.

**Source:** https://causalincentives.com/ (papers read in full, not just abstracts).

**Status:** ✅ All 24 papers summarized from full-text reads (6 thematic batches). Glossary and pgmpy-design synthesis complete. Ready to feed the design-proposal step.

---

## How to read this document

Papers are grouped thematically (not chronologically) because the design proposal
will be organized around *capabilities* (model classes, graphical criteria,
algorithms) rather than publication timeline. Each paper entry follows a fixed
template:

- **TL;DR** — one sentence.
- **Problem & motivation** — what gap it addresses.
- **Formal setup & key definitions** — model class and notation.
- **Key equations** — verbatim, in LaTeX.
- **Main results / theorems** — the load-bearing claims.
- **Algorithms** — if the paper contributes one.
- **Relevance to pgmpy** — what a graphical-models library would need to support it.

Recurring objects (CID, MAID, SCG, incentive criteria, etc.) are defined once in
the **Glossary of recurring concepts** at the bottom and referenced throughout.

---

## Paper index

| # | Theme | Title | Venue / Year | arXiv / link |
|---|-------|-------|--------------|--------------|
| A1 | Foundations: CIDs & incentives | Agent Incentives: A Causal Perspective (AI:ACP) | AAAI 2021 | [2102.01685](https://arxiv.org/abs/2102.01685) |
| A2 | Foundations: CIDs & incentives | Incentives for Responsiveness, Instrumental Control and Impact | AIJ 2025 | [2001.07118](https://arxiv.org/abs/2001.07118) |
| A3 | Foundations: CIDs & incentives | A Complete Criterion for Value of Information in Soluble Influence Diagrams | AAAI 2022 | [2202.11629](https://arxiv.org/abs/2202.11629) |
| A4 | Foundations: CIDs & incentives | Understanding Agent Incentives using CIDs, Part I: Single Action Settings | 2019 (superseded) | [1902.09980](https://arxiv.org/abs/1902.09980) |
| B1 | Multi-agent influence diagrams & games | Reasoning about Causality in Games (Structural Causal Games) | AIJ 2023 | [2301.02324](https://arxiv.org/abs/2301.02324) |
| B2 | Multi-agent influence diagrams & games | Equilibrium Refinements for MAIDs: Theory and Practice | AAMAS 2021 | [2102.05008](https://arxiv.org/abs/2102.05008) |
| B3 | Multi-agent influence diagrams & games | On Imperfect Recall in Multi-Agent Influence Diagrams | TARK 2023 | [TARK pdf](https://cgi.cse.unsw.edu.au/~eptcs/paper.cgi?TARK2023.17.pdf) |
| B4 | Multi-agent influence diagrams & games | Higher-Order Belief in Incomplete Information MAIDs | AAMAS 2025 | [2503.06323](https://arxiv.org/abs/2503.06323) |
| C1 | Agency & world models | Discovering Agents | AIJ 2023 | [2208.08345](https://arxiv.org/abs/2208.08345) |
| C2 | Agency & world models | Robust agents learn causal world models | ICLR 2024 | [2402.10877](https://arxiv.org/abs/2402.10877) |
| C3 | Agency & world models | General agents need world models | ICML 2025 | [2506.01622](https://arxiv.org/abs/2506.01622) |
| C4 | Agency & world models | The Limits of Predicting Agents from Behaviour | ICML 2025 | [2506.02923](https://arxiv.org/abs/2506.02923) |
| D1 | Goal-directedness, intent, decision theory | Measuring Goal-directedness (MEG) | NeurIPS 2024 | [2412.04758](https://arxiv.org/abs/2412.04758) |
| D2 | Goal-directedness, intent, decision theory | Evaluating the Goal-Directedness of Large Language Models | 2025 | [2504.11844](https://arxiv.org/abs/2504.11844) |
| D3 | Goal-directedness, intent, decision theory | The Reasons that Agents Act: Intention and Instrumental Goals | AAMAS 2024 | [2402.07221](https://arxiv.org/abs/2402.07221) |
| D4 | Goal-directedness, intent, decision theory | Characterising Decision Theories with Mechanised Causal Graphs | 2023 | [2307.10987](https://arxiv.org/abs/2307.10987) |
| E1 | Safety: deception, harm, control | Honesty Is the Best Policy: Defining and Mitigating AI Deception | NeurIPS 2023 | [2312.01350](https://arxiv.org/abs/2312.01350) |
| E2 | Safety: deception, harm, control | Counterfactual Harm | NeurIPS 2022 | [2204.12993](https://arxiv.org/abs/2204.12993) |
| E3 | Safety: deception, harm, control | Human Control: Definitions and Algorithms | UAI 2023 | [2305.19861](https://arxiv.org/abs/2305.19861) |
| F1 | RL, tampering, fairness, software | Reward Tampering Problems and Solutions in RL: A CID Perspective | Synthese 2021 | [1908.04734](https://arxiv.org/abs/1908.04734) |
| F2 | RL, tampering, fairness, software | How RL Agents Behave When Their Actions Are Modified | AAAI 2021 | [2102.07716](https://arxiv.org/abs/2102.07716) |
| F3 | RL, tampering, fairness, software | Path-Specific Objectives for Safer Agent Incentives | AAAI 2022 | [2204.10018](https://arxiv.org/abs/2204.10018) |
| F4 | RL, tampering, fairness, software | Why Fair Labels Can Yield Unfair Predictions | AAAI 2022 | [2202.10816](https://arxiv.org/abs/2202.10816) |
| F5 | RL, tampering, fairness, software | Modeling AGI Safety Frameworks with CIDs | IJCAI-WS 2019 | [1906.08663](https://arxiv.org/abs/1906.08663) |
| F6 | RL, tampering, fairness, software | PyCID: A Python Library for Causal Influence Diagrams | SciPy 2021 | [SciPy pdf](http://conference.scipy.org/proceedings/scipy2021/pdfs/james_fox.pdf) |

---

## A. Foundations: Causal Influence Diagrams & Incentive Criteria

### Agent Incentives: A Causal Perspective (AAAI, 2021)
**Authors:** Tom Everitt, Ryan Carey, Eric Langlois, Pedro A. Ortega, Shane Legg
**Link:** https://arxiv.org/abs/2102.01685
**TL;DR:** Unifies four agent-incentive concepts — value of information, value of control, response incentive, instrumental control incentive — under a single causal-influence-diagram framework, each with a sound and complete graphical criterion.

**Problem & motivation:** Safety/fairness of AI systems can be analysed by asking what an optimal agent is incentivised to observe and influence. Prior work (paper A4) gave only "observation" and "intervention" incentives with incomplete/imprecise treatment; this paper provides a unified causal (counterfactual-capable) formalism and proves sound-and-complete graphical criteria for all four notions, so incentives can be read off the graph without solving the decision problem.

**Formal setup & key definitions:**
- **Def 1 (SCM):** a tuple $\langle \bm{\mathcal{E}}, \bm{V}, \bm{F}, P\rangle$ with exogenous variables $\bm{\mathcal{E}}$, endogenous $\bm{V}$, structural functions $\bm{F}=\{f_V\}$, and $P$ over $\bm{\mathcal{E}}$, with acyclic dependencies.
- **Def 2 (submodel/intervention):** $\mathcal{M}_x$ realises $\mathrm{do}(X=x)$ by replacing $f_X$ with the constant $X=x$.
- **Def 3 (CID):** a DAG $\mathcal{G}$ whose vertices $\bm{V}$ are partitioned into **structure (chance) nodes** $\bm{X}$, **decision nodes** $\bm{D}$, and **utility nodes** $\bm{U}$; utility nodes have no children.
- **Def 4 (SCIM, Structural Causal Influence Model):** $\mathcal{M}=\langle \mathcal{G}, \bm{\mathcal{E}}, \bm{F}, P\rangle$ combining a CID with exogenous vars and structural functions $\bm{F}=\{f_V\}_{V\in \bm{V}\setminus\bm{D}}$ for all non-decision nodes. Decision nodes have no structural function until a **policy** is chosen.
- **Policy / optimal policy:** a decision rule $\pi: \mathrm{dom}(\mathbf{Pa}^D\cup\{\mathcal{E}^D\})\to\mathrm{dom}(D)$ (collectively $\bm{\pi}=\{\pi^D\}$); optimal $\pi^*$ maximises $\mathbb{E}_\pi[\mathcal{U}]$ with $\mathcal{U}:=\sum_{U\in\bm{U}}U$.
- **Def 5 (Materiality):** parent $X\in\mathbf{Pa}^D$ is *material* if $\mathcal{V}^*(\mathcal{M}_{X\not\to D})<\mathcal{V}^*(\mathcal{M})$, where $\mathcal{V}^*(\mathcal{M})=\max_\pi\mathbb{E}_\pi[\mathcal{U}]$ and $\mathcal{M}_{X\not\to D}$ deletes the information link $X\to D$.
- **Def 7 (Nonrequisite observation):** $X\in\mathbf{Pa}^D$ is *nonrequisite* if $X\perp \bm{U}^D \mid (\mathbf{Pa}^D\cup\{D\}\setminus\{X\})$ (where $\bm{U}^D$ = utility descendants of $D$); otherwise *requisite*.
- **Def 11 (Minimal reduction):** $\mathcal{G}^{\min}$ is obtained from $\mathcal{G}$ by deleting the information links of all nonrequisite observations.
- Incentive concepts (formal defs below): **Value of Information** (Def 8), **Value of Control** (Def 15), **Response Incentive** (Def 10), **Instrumental Control Incentive** (Def 17).

**Key equations:**
- $\mathcal{U}:=\sum_{U\in\bm{U}}U$, and $\mathcal{V}^*(\mathcal{M})=\max_\pi\mathbb{E}_\pi[\mathcal{U}]$ — total utility and optimal attainable value.
- **VoI (Def 8):** $X\in\bm{V}\setminus\mathbf{Desc}^D$ has positive VoI iff $X$ is *material* in $\mathcal{M}_{X\to D}$ (the graph with $X\to D$ added); numerically $\max_\pi\mathbb{E}^{\mathcal{M}_{X\to D}}_\pi[\mathcal{U}]-\max_\pi\mathbb{E}^{\mathcal{M}_{X\not\to D}}_\pi[\mathcal{U}]>0$ — gain from observing $X$.
- **VoC (Def 15):** $\max_\pi\mathbb{E}_\pi[\mathcal{U}] < \max_{\pi,\,g^X}\mathbb{E}_\pi[\mathcal{U}_{g^X}]$, where $g^X:\mathrm{dom}(\mathbf{Pa}^X\cup\{\mathcal{E}^X\})\to\mathrm{dom}(X)$ is a *soft intervention* (graph-respecting alternative structural function for $X$) — gain from being able to set $X$.
- **Response Incentive (Def 10):** for all optimal $\pi^*$, $\exists\,\mathrm{do}(X=x),\varepsilon$ with $D_x(\varepsilon)\neq D(\varepsilon)$ — the optimal decision counterfactually changes when $X$ is perturbed.
- **ICI (Def 17):** for all optimal $\pi^*$ there is a context with $\mathbb{E}_{\pi^*}[\mathcal{U}_{X_d}\mid \mathbf{pa}^D]\neq \mathbb{E}_{\pi^*}[\mathcal{U}\mid \mathbf{pa}^D]$ (nested counterfactual: utility with $X$ clamped to the value it *would* take under $D=d$) — the policy influences $\mathcal{U}$ *through* $X$.

**Main results / theorems** (each is sound *and* complete — "admits" = exists a compatible parameterisation):
- **Thm 9 (VoI):** a single-decision CID admits positive VoI for $X\in\bm{V}\setminus\mathbf{Desc}^D$ iff $X$ is a *requisite* observation in $\mathcal{G}_{X\to D}$, i.e. $X\not\perp \bm{U}^D \mid (\mathbf{Pa}^D\cup\{D\}\setminus\{X\})$ in $\mathcal{G}_{X\to D}$. (Establishes completeness of the classical criterion.)
- **Thm 12 (Response Incentive):** admits a response incentive on $X\in\bm{X}$ iff the minimal reduction $\mathcal{G}^{\min}$ contains a directed path $X\dashrightarrow D$.
- **Thm 16 (VoC):** admits positive VoC for $X\in\bm{V}\setminus\{D\}$ iff $\mathcal{G}^{\min}$ contains a directed path $X\dashrightarrow \bm{U}$.
- **Thm 18 (ICI):** admits an instrumental control incentive on $X$ iff $\mathcal{G}$ contains a directed path $D\dashrightarrow X\dashrightarrow \bm{U}$ (a path from the decision to a utility *through* $X$).
- **Thm 14 (fairness link):** all optimal policies are counterfactually unfair w.r.t. a sensitive attribute $A$ iff $A$ has a response incentive.
- Conceptual 2×2: VoI/VoC are *value* notions (can observing/controlling $X$ help an idealised agent), RI/ICI are *incentive* notions (does the actual optimal policy respond to / influence $X$).

**Algorithms:** All four criteria are purely graphical. The key construction is the **minimal reduction** $\mathcal{G}^{\min}$: repeatedly delete information links $X\to D$ whose source satisfies the nonrequisite d-separation test $X\perp \bm{U}^D\mid(\mathbf{Pa}^D\cup\{D\}\setminus\{X\})$. Then test for the relevant directed path / d-connection. VoI/VoC *magnitudes* require solving the ID for optimal expected utility.

**Relevance to pgmpy:** Needs node-role typing on a DAG (decision / chance / utility, with utility nodes childless) — analogous to pgmpy's existing node-role annotations. Graph ops required: d-separation/d-connection conditioned on arbitrary sets (pgmpy has d-sep), computation of $\mathbf{Pa}^D$, descendants, utility-descendants $\bm{U}^D$; edge addition/removal for $\mathcal{M}_{X\to D}/\mathcal{M}_{X\not\to D}$; construction of the minimal reduction; directed-path queries ($X\dashrightarrow D$, $X\dashrightarrow\bm{U}$, $D\dashrightarrow X\dashrightarrow\bm{U}$). Incentive *detection* is cheap and graph-only (a natural fit alongside structure tooling); VoI/VoC *values* additionally need an influence-diagram solver (optimal-policy expected-utility maximisation), which pgmpy currently lacks.

---

### Incentives for Responsiveness, Instrumental Control and Impact (AIJ journal version, 2020–2025)
**Authors:** Ryan Carey, Eric Langlois, Chris van Merwijk, Shane Legg, Tom Everitt
**Link:** https://arxiv.org/abs/2001.07118
**TL;DR:** The journal-length treatment of three behaviour-shaping incentives — response incentives, instrumental control incentives, and impact incentives (plus "intent") — each with sound-and-complete graphical criteria in single-decision CIDs, and an outline of multi-decision generalisations. (Note: the arXiv HTML serves the expanded journal version; the ar5iv mirror serves the older v1 "The Incentives that Shape Behaviour," which covers only control + response incentives.)

**Problem & motivation:** To reason about safety/fairness one wants to know which environment variables affect an agent's decision (e.g. sensitive demographics), which variables it is incentivised to manipulate (e.g. user preferences), and which it will affect at all. The paper formalises these as response/instrumental-control/impact incentives, gives graphical criteria, and connects them to techniques for producing safe/fair behaviour.

**Formal setup & key definitions:** Same SCIM/CID substrate as the AI:ACP paper (Def 4 SCIM; structure/decision/utility nodes; $\mathcal{U}=\sum_U U$; $\mathcal{V}^*(\mathcal{M})=\max_\pi\mathbb{E}_\pi[\mathcal{U}]$). Incentive definitions are stated for *sets* $\bm{W}$ of variables.
- **Def 6 (Materiality):** $\bm{W}\subseteq\mathbf{Pa}^D$ is material if $\mathcal{V}^*(\mathcal{M}_{\bm{W}\not\to D})<\mathcal{V}^*(\mathcal{M})$.
- **Def 7 (Nonrequisite observation):** $W\in\mathbf{Pa}^D$ nonrequisite if $W\perp \bm{U}^D\mid(\mathbf{Pa}^D\cup\{D\}\setminus\{W\})$.
- **Def 8 (Response incentive):** a policy $\pi$ *responds* to $\bm{W}\subseteq\bm{X}$ if there exist soft interventions $g^{\bm{W}}$ and an exogenous setting $\varepsilon$ with $D_{g^{\bm{W}}}(\varepsilon)\neq D(\varepsilon)$; $\bm{W}$ has a *response incentive* if **all** optimal policies respond to $\bm{W}$.
- **Def 9 (Minimal reduction):** $\mathcal{G}^{\min}$ deletes information links of all nonrequisite observations.
- **Def 11 (Instrumental control incentive):** ICI on $\bm{W}$ in context $\mathbf{pa}^D$ iff for all optimal $\pi^*$ there is $D=d$ with $\mathbb{E}_{\pi^*}[\mathcal{U}_{\bm{W}_d}\mid\mathbf{pa}^D]\neq\mathbb{E}_{\pi^*}[\mathcal{U}\mid\mathbf{pa}^D]$.
- **Def 12 (Impact incentive):** an incentive to *impact* $\bm{W}$ with distance function $\delta$ and threshold $c>0$, relative to a baseline policy $\pi'$, iff every optimal $\pi$ has $\mathbb{E}[\delta(W_\pi(\varepsilon), W_{\pi'}(\varepsilon))]>c$.
- **Intent:** *additive intent* to influence $\bm{W}$ by choosing $\pi^*$ over $\pi'$ if $\mathbb{E}_{\pi'}[\mathcal{U}]<\mathbb{E}_{\pi^*}[\mathcal{U}]$ and $\bm{W}\subseteq\bm{Z}$ is subset-minimal with $\mathbb{E}_{\pi'}[\mathcal{U}_{\bm{Z}_{\pi^*}}]\ge\mathbb{E}_{\pi^*}[\mathcal{U}]$; *subtractive intent* if instead $\bm{Z}$ subset-minimal with $\mathbb{E}_{\pi^*}[\mathcal{U}_{\bm{Z}_{\pi'}}]\le\mathbb{E}_{\pi'}[\mathcal{U}]$.

**Key equations:**
- **Response incentive:** $\exists\,g^{\bm{W}},\varepsilon:\ D_{g^{\bm{W}}}(\varepsilon)\neq D(\varepsilon)$ for all optimal $\pi$ — decision counterfactually changes under perturbation of $\bm{W}$.
- **ICI:** $\mathbb{E}_{\pi^*}[\mathcal{U}_{\bm{W}_d}\mid\mathbf{pa}^D]\neq\mathbb{E}_{\pi^*}[\mathcal{U}\mid\mathbf{pa}^D]$ — utility flows through $\bm{W}$ as a controllable means.
- **Impact:** $\mathbb{E}[\delta(W_\pi(\varepsilon),W_{\pi'}(\varepsilon))]>c$ — agent changes $\bm{W}$ relative to a baseline, intentionally or not.
- **Additive/subtractive intent:** $\mathbb{E}_{\pi'}[\mathcal{U}_{\bm{Z}_{\pi^*}}]\ge\mathbb{E}_{\pi^*}[\mathcal{U}]$ / $\mathbb{E}_{\pi^*}[\mathcal{U}_{\bm{Z}_{\pi'}}]\le\mathbb{E}_{\pi'}[\mathcal{U}]$ — minimal mediator set explaining the utility gain.

**Main results / theorems** (single-decision, sound and complete):
- **Response incentive criterion:** $\mathcal{G}$ admits a response incentive on $\bm{W}\subseteq\bm{X}$ iff $\mathcal{G}^{\min}$ has a directed path $W\dashrightarrow D$ for some $W\in\bm{W}$. Equivalent unpacked (3-condition) form: (i) a directed path $W\dashrightarrow O$ to some observation $O\in\mathbf{Pa}^D$, (ii) a directed path $D\dashrightarrow U$, and (iii) $O$ requisite, i.e. $O\not\perp U\mid \mathbf{Fa}^D\setminus\{O\}$ (with $\mathbf{Fa}^D=\mathbf{Pa}^D\cup\{D\}$).
- **ICI criterion:** $\mathcal{G}$ admits an ICI on $\bm{W}$ iff there is a directed path $D\dashrightarrow W\dashrightarrow U$ for some $W\in\bm{W}, U\in\bm{U}$.
- **Impact incentive criterion:** $\mathcal{G}$ admits an impact incentive on $\bm{W}\subseteq\bm{X}$ iff some $W\in\bm{W}$ and some $U\in\bm{U}$ are *both descendants of* $D$ (i.e. $D\dashrightarrow W$ and $D\dashrightarrow U$).
- **Intent criterion:** $\mathcal{G}$ admits (additive/subtractive) intent on $\bm{W}$ iff there is a directed path $D\dashrightarrow W\dashrightarrow U$ (same path condition as ICI).
- **Multi-decision:** Section 8 *outlines* generalisations (incentive defs already given for variable sets), discussing how the notions relate, but does not deliver complete multi-decision graphical criteria — the complete multi-decision VoI criterion is the subject of paper A3.

**Algorithms:** Same graphical machinery as the AI:ACP paper: build $\mathcal{G}^{\min}$ via the nonrequisite-link d-separation test, then check directed paths / descendant relations. Impact and ICI criteria need only descendant/path queries on $\mathcal{G}$ (no reduction).

**Relevance to pgmpy:** Reuses the same primitives as AI:ACP (decision/chance/utility typing, d-separation, minimal reduction, directed-path and descendant queries). Adds the need for: a soft-intervention representation, baseline-policy comparison and a distance function $\delta$ for impact, and nested counterfactual utilities $\mathcal{U}_{\bm{W}_d}$ / potential responses $\mathcal{U}_{\bm{x}}$ (twin-network / counterfactual evaluation). Maps directly to applied detectors: response incentive on a sensitive attribute ⇒ counterfactual unfairness; ICI ⇒ user-preference manipulation / reward tampering; impact incentive ⇒ side-effect / impact-measure analysis.

---

### A Complete Criterion for Value of Information in Soluble Influence Diagrams (AAAI, 2022)
**Authors:** Chris van Merwijk, Ryan Carey, Tom Everitt
**Link:** https://arxiv.org/abs/2202.11629
**TL;DR:** Gives the first *complete* graphical criterion for positive value of information in multi-decision (soluble) influence diagrams: an information link $X\to D$ can have strictly positive VoI iff it survives in the unique **minimal d-reduction**.

**Problem & motivation:** The single-decision VoI criterion (requisite observations / minimal reduction) was known to be sound and complete, but existing multi-decision criteria (Shachter 1998; Nielsen & Jensen 1999; Nilsson & Lauritzen 2000) were only proven *sound* — they could mark a link nonrequisite, but no one had shown that a surviving link *can* carry positive VoI. With multiple decisions, downstream decisions can "bypass" an observation, so naive single-system arguments fail. This paper closes the completeness gap for soluble IDs.

**Formal setup & key definitions:**
- **Def 1 (limited-memory ID graph):** DAG $\mathcal{G}=(\bm{V},E)$, $\bm{V}$ partitioned into chance $\bm{X}$, decision $\bm{D}$, utility $\bm{U}$; utility nodes have no children.
- **Def 2 (influence diagram):** $\mathcal{M}=(\mathcal{G},\mathrm{dom},P)$ with finite domains (real for utilities) and CPDs $P(X\mid\mathbf{Pa}(X))$ for chance/utility nodes.
- **Def 3 (d-separation):** standard collider/chain/fork blocking.
- **Def 4 (Solubility):** $\mathcal{G}$ is *soluble* if there is a decision ordering $D^1,\dots,D^n$ such that in the mapping extension $\mathcal{G}'$, for all $i$: $\Pi^{<i}\perp \bm{U}(D^i)\mid\mathbf{Fa}(D^i)$ (earlier decision rules are irrelevant to $D^i$'s downstream utilities given its family) — i.e. the ID is solvable by backward induction.
- **Def 5 (VoI):** for $X\notin\mathbf{Desc}_D$, $\mathrm{VoI}=\max_\pi\mathbb{E}^{\mathcal{M}_{X\to D}}_\pi[\mathcal{U}]-\max_\pi\mathbb{E}^{\mathcal{M}_{X\not\to D}}_\pi[\mathcal{U}]$.
- **Def 6 (d-reduction / minimal d-reduction):** $\mathcal{G}'$ is a *d-reduction* of $\mathcal{G}$ if reachable by a sequence each removing one *nonrequisite information link*; an info link $X\to D^i$ is **nonrequisite** if $X\perp \bm{U}(D^i)\mid\mathbf{Fa}(D^i)\setminus\{X\}$. A d-reduction is **minimal** if no nonrequisite links remain; the minimal d-reduction is **unique** / order-independent (Nilsson & Lauritzen 2000).
- Notation: $\mathbf{Fa}(V)=\mathbf{Pa}(V)\cup\{V\}$; $V\dashrightarrow Y$ directed path; $\mathcal{U}=\sum_{U\in\bm{U}}U$; policy $\pi=\{\pi^D\}$.

**Key equations:**
- **Solubility:** $\Pi^{<i}\perp \bm{U}(D^i)\mid\mathbf{Fa}(D^i)$ for all $i$ — backward-induction solvability.
- **VoI:** $\max_\pi\mathbb{E}^{\mathcal{M}_{X\to D}}_\pi[\mathcal{U}]-\max_\pi\mathbb{E}^{\mathcal{M}_{X\not\to D}}_\pi[\mathcal{U}]$ — value gained by observing $X$ at $D$.
- **Nonrequisite link:** $X\perp \bm{U}(D^i)\mid\mathbf{Fa}(D^i)\setminus\{X\}$ — d-separation test driving the reduction.

**Main results / theorems:**
- **Thm 7 (complete VoI criterion):** Let $\mathcal{G}$ be a soluble ID graph containing an edge $X\to D$ ($X\in\bm{X}$). There exists an ID $\mathcal{M}$ compatible with $\mathcal{G}$ such that $X$ has strictly positive VoI for $D$ **iff** the minimal d-reduction contains $X\to D$. (Soundness was prior work; *completeness* — existence of a witnessing parameterisation — is the new contribution.)
- Proof machinery: **Def 8 (ID homomorphism)** — structure-preserving maps of IDs (preserve node types, links, information-link coverage, connected-decision splitting); **Def 16 (System)** $=(\mathrm{control}^s,\mathrm{info}^s,\mathrm{obs}^s)$ — a triple of directed paths capturing how information/control flow to utility through a decision; **Def 17 (Tree of Systems)** — hierarchical collection of systems. **Lemma 19 (normal-form existence):** if $\mathcal{G}^*$ contains $X\to D$ there is a normal-form tree on a soluble $\mathcal{G}'$ with homomorphism $h:\mathcal{G}'\to\mathcal{G}$. **Lemma 21:** a normal-form tree rooted at $X\to D$ yields a compatible ID with positive VoI. The multi-decision subtlety (illustrated by a counterexample, Fig. 5): a single system/path is insufficient because a later decision can render $X$ uninformative; a *tree* of (≥2) systems is needed so $X$ cannot be bypassed.

**Algorithms:** No explicit pseudocode/complexity is given, but the criterion is operational: compute the **minimal d-reduction** by iteratively deleting any information link satisfying the nonrequisite d-separation test until none remain (result is unique, order-independent), then check membership of $X\to D$.

**Relevance to pgmpy:** Requires multi-decision ID support: a **solubility test** (search for a decision ordering with $\Pi^{<i}\perp\bm{U}(D^i)\mid\mathbf{Fa}(D^i)$, via d-separation on the mapping/relevance extension), iterative **minimal-d-reduction** computation (d-separation $X\perp\bm{U}(D^i)\mid\mathbf{Fa}(D^i)\setminus\{X\}$ per info link), and link-membership testing. This is the principled multi-decision generalisation of the single-decision "reduced graph" used by the incentive papers — pgmpy would implement the reduction once and reuse it across VoI / response-incentive / control-incentive detectors. The homomorphism / tree-of-systems constructs are proof tools, not needed at runtime.

---

### Understanding Agent Incentives using Causal Influence Diagrams, Part I: Single Action Settings (arXiv, 2019; superseded by AI:ACP)
**Authors:** Tom Everitt, Pedro A. Ortega, Elizabeth Barnes, Shane Legg
**Link:** https://arxiv.org/abs/1902.09980
**TL;DR:** The early, superseded paper that introduced reading agent incentives off a CID, but with only **two** concepts — *observation incentives* (later VoI) and *intervention incentives* (later VoC) — and without the counterfactual SCIM machinery, response incentives, or instrumental control incentives added by AI:ACP.

**Problem & motivation:** Establishes that two questions can be answered directly from a CID graph: (1) which nodes does the agent have an incentive to *observe*, and (2) which nodes does it have an incentive to *control*? The answers flag information/influence points needing protection (e.g. don't use ethnicity in a classifier; don't let an RL agent control its reward mechanism).

**Formal setup & key definitions:**
- **Def 5 (CID graph):** $G=(\bm{W},E,\bm{D},\bm{U})$ — causal graph $(\bm{W},E)$, ordered decision set $\bm{D}\subseteq\bm{W}$ (blue rectangles), utility set $\bm{U}\subseteq\bm{W}\setminus\bm{D}$ (yellow octagons); remaining nodes are chance.
- **Def 6 (CID model):** $M=(\bm{W},E,\bm{D},\bm{U},P)$ with finite domains and CPDs $P(x\mid\mathrm{pa}_X)$ for non-decision nodes.
- **Def 7 (value function):** $V^\pi=\mathbb{E}[\sum_{U\in\bm{U}}U\mid\pi]$; optimal $\pi^*$ maximises $V^\pi$.
- **Def 8 (observation incentive):** for $X\in\bm{W}\setminus\mathrm{desc}(D)$, an observation incentive exists if $V^*_{X\to D}>V^*_{X\not\to D}$ (this is VoI under a different name).
- **Def 10 (requisite observation):** $O\in\mathrm{Pa}_D$ is requisite if it is d-connected to a utility descendant of $D$ (given the rest of the family).
- **Def 11 (reduced graph $G^*$):** $G$ with all nonrequisite information links removed (precursor of the "minimal reduction").
- **Def 13 (intervention incentive):** for non-decision $X$, an intervention incentive exists if $\max_{\pi,c^X}V^{\pi,c^X}>\max_{\pi'}V^{\pi'}$, where $c^X$ is a soft intervention on $X$'s CPD (this is VoC under a different name).

**Key equations:**
- $V^\pi=\mathbb{E}[\sum_{U\in\bm{U}}U\mid\pi]$ — expected total utility of a policy.
- Observation incentive: $V^*_{X\to D}>V^*_{X\not\to D}$ — value of observing $X$.
- Intervention incentive: $\max_{\pi,c^X}V^{\pi,c^X}>\max_{\pi'}V^{\pi'}$ — value of (soft-)controlling $X$.

**Main results / theorems:**
- **Thm 9 (observation incentive criterion):** $G$ is compatible with an observation incentive on $X$ iff $X\not\perp \bm{U}\cap\mathrm{desc}(D)\mid \{D\}\cup\mathbf{Pa}_D\setminus\{X\}$ ($X$ d-connected to a utility descendant of $D$). (Soundness/completeness directions noted as Thms 15/18.)
- **Thm 14 (intervention incentive criterion):** an agent has an intervention incentive on non-decision $X$ iff there is a directed path $X\dashrightarrow\bm{U}$ in the reduced graph $G^*$.
- Distinguishes *direct* vs *indirect* intervention incentives (path to utility not through, vs through, the decision).

**What is unique vs AI:ACP:** This paper has *only* the two value-type incentives — observation (= VoI, Thm 9 matches AI:ACP Thm 9) and intervention (= VoC, Thm 14 matches AI:ACP Thm 16). It **lacks** response incentives and instrumental control incentives entirely, and lacks the SCIM/exogenous-variable formalism that AI:ACP needs to define the *counterfactual* notions (RI uses $D_x(\varepsilon)\neq D(\varepsilon)$; ICI uses nested counterfactuals $\mathcal{U}_{X_d}$). AI:ACP's contributions over Part I: (i) recasting everything in a single SCIM with explicit counterfactual semantics; (ii) a *new* sound-and-complete VoC criterion and a completeness proof for VoI; (iii) splitting the loose "control/intervention incentive" into the cleaner pair VoC (value, path $X\dashrightarrow\bm{U}$ in $\mathcal{G}^{\min}$) and ICI (behavioural, path $D\dashrightarrow X\dashrightarrow\bm{U}$); (iv) introducing response incentives. Applications previewed here — fairness/disparate-treatment (Berkeley-admissions style), QA-system safety via counterfactual oracles, and reward tampering — are carried forward and formalised in the later papers.

**Relevance to pgmpy:** Same minimal primitives as AI:ACP but a smaller surface: decision/chance/utility node typing, d-connection conditioned on $\{D\}\cup\mathbf{Pa}_D\setminus\{X\}$, the **reduced graph** $G^*$ (delete nonrequisite info links via the requisite/d-connection test), and directed-path queries $X\dashrightarrow\bm{U}$ in $G^*$. A pgmpy implementation should target the AI:ACP/A2/A3 formulations (which subsume this paper) and treat Part I's "observation/intervention incentive" as aliases for VoI/VoC.

## B. Multi-Agent Influence Diagrams & Games

### Reasoning about Causality in Games (Artificial Intelligence journal, 2023)
**Authors:** Lewis Hammond, James Fox, Tom Everitt, Ryan Carey, Alessandro Abate, Michael Wooldridge
**Link:** https://arxiv.org/abs/2301.02324
**TL;DR:** Introduces *(structural) causal games* and *mechanised MAIDs*, unifying Pearl's causal hierarchy (BN/CBN/SCM) with game theory (MAID/CG/SCG) so that predictive, interventional, and counterfactual queries — plus subgames and equilibrium refinements — can be answered graphically via a "mechanised graph" of decision-rule and parameter nodes.

**Problem & motivation:** MAIDs concisely represent games as DAGs but leave the *process by which agents choose decision rules* implicit, so they cannot natively express how strategic agents adapt to interventions, nor support counterfactual reasoning. The paper makes these dependencies explicit (via mechanism variables and *rationality relations*), placing causal-inference machinery (do-calculus, counterfactuals) and game-theoretic machinery (subgames, SPE, THPE) on a common footing.

**Formal setup & key definitions:**
- **MAID (Def 9):** a structure $\mathcal{M}=(\mathcal{G},\boldsymbol{\theta})$ where $\mathcal{G}=(N,\boldsymbol{V},\mathcal{E})$ has agents $N=\{1,\dots,n\}$ and a DAG over vertices $\boldsymbol{V}$ partitioned into chance $\boldsymbol{X}$, decision $\boldsymbol{D}=\bigcup_{i\in N}\boldsymbol{D}^i$, and utility $\boldsymbol{U}=\bigcup_{i\in N}\boldsymbol{U}^i$ nodes. Parameters $\boldsymbol{\theta}=\{\theta_V\}_{V\in\boldsymbol{V}\setminus\boldsymbol{D}}$ give CPDs $\Pr(V\mid \mathbf{Pa}_V;\theta_V)$ for non-decision nodes; for *any* parameterisation of decision CPDs the induced joint is a BN. Information edges = parent edges into decisions; the *decision context* of $D$ is the value of $\mathbf{Pa}_D$. Absence of an edge encodes unobservability.
- **Decision rule / policy (Def 10):** a decision rule $\pi_D$ is a CPD $\pi_D(D\mid\mathbf{Pa}_D)$; a **partial policy profile** $\pi_{\boldsymbol{D}'}$ fixes rules for $D\in\boldsymbol{D}'\subseteq\boldsymbol{D}$; a (behavioural) **policy** $\pi^i=\pi_{\boldsymbol{D}^i}$; a **full policy profile** $\pi=(\pi^1,\dots,\pi^n)$, with $\pi^{-i}$ the others. Pure: $\pi_D(d\mid\mathbf{pa}_D)\in\{0,1\}$. The induced joint is $\Pr^{\pi}(\boldsymbol{x},\boldsymbol{d},\boldsymbol{u}) := \prod_{V\in\boldsymbol{V}\setminus\boldsymbol{D}}\Pr(v\mid\mathbf{pa}_V)\cdot\prod_{D\in\boldsymbol{D}}\pi_D(d\mid\mathbf{pa}_D)$.
- **Mechanised MAID (Def 15):** extends $\mathcal{G}$ to a **mechanised graph** $m\mathcal{G}$ by adding a *mechanism variable* $M_V$ as a new parent of every object-level $V$: a **decision-rule variable** $\Pi_D=M_D$ for decisions and a **parameter variable** $\Theta_V=M_{V}$ for non-decisions. New edges $\mathcal{E}'\subseteq\bigcup_{D}((\boldsymbol{M}\setminus\Pi_D)\times\Pi_D)$ encode that an agent picks $\pi_D$ based on other mechanisms. A mechanised MAID is $m\mathcal{M}=(m\mathcal{G},\boldsymbol{\theta},\mathcal{R})$ where $\mathcal{R}=\{r_D\}_{D\in\boldsymbol{D}}$ is a set of **rationality relations**, each $r_D\subseteq dom(\mathbf{Pa}_{\Pi_D})\times dom(\Pi_D)$ a serial relation (the graph is *possibly cyclic*). Object-level CPDs are recovered by $\Pr^{\pi}(\boldsymbol{v};\boldsymbol{\theta})=\Pr(\boldsymbol{v}\mid\boldsymbol{m})=\prod_{V}\Pr(v\mid\mathbf{pa}_V,m_V)$ with $\boldsymbol{m}_D=\pi$, $\boldsymbol{m}_{\boldsymbol{V}\setminus\boldsymbol{D}}=\boldsymbol{\theta}$.
- **R-rational outcomes (Def 16):** $\pi$ is $\mathcal{R}$-rational if $\pi_D\in r_D(\mathbf{pa}_{\Pi_D})$ for all $D$; the model becomes a *set* of BNs $\{\Pr^{\pi}\}_{\pi\in\mathcal{R}(m\mathcal{M})}$ (analogous to solutions of a cyclic causal model / a credal set). Best-response rationality $\mathcal{R}^{\mathrm{BR}}$ (Eq. 1): $\pi_D\in r_D^{\mathrm{BR}}(\mathbf{pa}_{\Pi_D}) \iff \pi_D\in\arg\max_{\hat\pi_D\in dom(\Pi_D)}\sum_{U\in\boldsymbol{U}^i}\mathbb{E}_{(\hat\pi_D,\pi_{-D})}[U]$.
- **Relevance (Defs 12, 17, 18):** $M_V$ is **$\mathcal{R}$-relevant** to $\Pi_D$ if $\exists\,\mathbf{pa}_{\Pi_D}\neq\mathbf{pa}'_{\Pi_D}$ differing only on $M_V$ with $r_D(\mathbf{pa}_{\Pi_D})\neq r_D(\mathbf{pa}'_{\Pi_D})$. The **$\mathcal{R}$-minimal** mechanised graph $m_{\mathcal{R}}\mathcal{G}$ keeps edge $M_V\to\Pi_D$ iff $M_V$ is $\mathcal{R}$-relevant; restricted to mechanism nodes it is the **$\mathcal{R}$-relevance graph** $r_{\mathcal{R}}\mathcal{G}$. K&M's **s-relevance** is the $\mathcal{R}^{\mathrm{BR}}$ special case; relevance is detected by **s-/$\mathcal{R}$-reachability**, a *d-separation* test on $m\mathcal{G}$ (Prop 1: $\Pi_{D'}$ is s-relevant to $\Pi_D$ iff $\Pi_{D'}\not\perp_{\mathcal{G}'}\boldsymbol{U}^i\cap\boldsymbol{Desc}_D\mid D,\mathbf{Pa}_D$).

**Key equations:**
- Expected utility: $\sum_{U\in\boldsymbol{U}^i}\mathbb{E}_{\pi}[U]$.
- **Nash equilibrium (Def 11):** $\pi$ is an NE iff for every $i$, $\pi^i\in\arg\max_{\hat\pi^i\in dom(\Pi^i)}\sum_{U\in\boldsymbol{U}^i}\mathbb{E}_{(\hat\pi^i,\pi^{-i})}[U]$.
- **Soundness/completeness of relevance (Eq. 2):** $\pi_D\in r_D(\boldsymbol{m}_{\boldsymbol{V}\setminus\{D\}}) \iff g_D(\mathcal{Q}_D(\boldsymbol{m}),dom(\boldsymbol{V}))$, where for $\mathcal{R}^{\mathrm{BR}}$, $\mathcal{Q}_D^{\mathrm{BR}}=\{\Pr^{\pi}(\boldsymbol{u}^i\cap\mathbf{desc}_D\mid d,\mathbf{pa}_D),\Pr^{\pi}(\mathbf{pa}_D)\}$.
- **Causal queries:** conditional/predictive $\Pr^{\mathcal{R}}(\boldsymbol{x}\mid\boldsymbol{z}):=\{\Pr^{\pi}(\boldsymbol{x}\mid\boldsymbol{z})\}_{\pi\in\mathcal{R}(m\mathcal{M}\mid\boldsymbol{z})}$ (Def 19); interventional $\Pr^{\mathcal{R}}(\boldsymbol{x}_{\mathcal{I}})$ (Def 21); counterfactual (Def 23): $\Pr^{\mathcal{R}}(\boldsymbol{x}_{\mathcal{I}}\mid\boldsymbol{z}):=\{\int_{dom(\boldsymbol{E}')}\Pr^{\pi'}(\boldsymbol{x}_{\mathcal{I}}\mid\boldsymbol{e},\boldsymbol{e}^*)\Pr(\boldsymbol{e}^*)\Pr^{\pi}(\boldsymbol{e}\mid\boldsymbol{z})\,d\boldsymbol{e}'\}_{(\pi,\pi')\in\mathcal{R}(m\mathcal{M}_{\mathcal{I}}\mid\boldsymbol{z})}$ over **actual–counterfactual rational outcome** pairs sharing invariant decision rules $\boldsymbol{\Pi}(\mathcal{I})$.
- **Pre- vs post-policy interventions:** post-policy = intervene on object-level $V$; **pre-policy** = intervene on mechanism node $\Pi_D$ (replacing $r_D$), so agents *re-optimise* in response.

**Main results / theorems:**
- **Causal hierarchy unification (Fig. 5):** rows = associational/interventional/counterfactual; columns = 0/1/$n$ agents, giving BN→CBN→SCM, ID→CID→SCIM, and **MAID→CG→SCG**. A **causal game (CG, Def 20)** is a MAID whose induced model is a CBN for any $\pi$. A **(Markovian) structural causal game (SCG, Def 22)** is a CG over exogenous+endogenous variables $\boldsymbol{E}\cup\boldsymbol{V}$ such that for any deterministic decision-rule parameterisation $\dot\pi$ the induced joint $\Pr^{\dot\pi}(\boldsymbol{V},\boldsymbol{E})$ is an SCM — i.e. "an SCM without parameters for the decision variables," each $V$ governed by a structural function with exogenous parent $\mathsf{E}_V$.
- **Decision rules as structural functions (Eq. 3, Prop 4):** a stochastic rule $\pi_D(D\mid\overline{\mathbf{Pa}}_D)$ is represented by an exogenous field $\mathsf{E}_D^{\pi_D,\overline{\mathbf{pa}}_D}$ with $\Pr(\mathsf{E}_D^{\pi_D,\overline{\mathbf{pa}}_D}=d):=\pi_D(d\mid\overline{\mathbf{pa}}_D)$ and $\dot\pi_D(D=d\mid\overline{\mathbf{pa}}_D,\mathsf{e}_D):=\delta(D,\mathsf{e}_D^{\pi_D,\overline{\mathbf{pa}}_D})$; there is a one-to-one correspondence between stochastic rules and deterministic structural rules (needed because infinitely many deterministic mechanisms induce the same CPD — counterfactuals require committing to one).
- **Subgames (Defs 25–26):** an **$\mathcal{R}$-subdiagram** $\mathcal{G}'=(N',\boldsymbol{V}',\mathcal{E}')$ requires $\boldsymbol{V}'$ closed under (i) $\mathcal{R}$-reachability from its decisions and (ii) directed paths between its members; an **$\mathcal{R}$-subgame** re-parameterises remaining variables conditional on a cut $\boldsymbol{Z}=\boldsymbol{V}\setminus\boldsymbol{V}'$. MAIDs can expose *more* subgames than the equivalent EFG (Fig. 9).
- **Refinements:** **SPE (Def 27):** $\pi$ is an NE in every feasible s-subgame. **Prop 8:** any sufficient-recall MAID has an SPE in behavioural policies (backward induction over s-subgames). **Prop 9:** with sufficient recall, the SPEs equal the $\mathcal{R}^{\mathrm{SP}}$-rational outcomes. **THPE (Def 28):** $\exists$ perturbation vectors $\zeta_k$ with $\lim_k\|\zeta_k\|_\infty=0$ such that each perturbed game $\mathcal{M}(\zeta_k)$ (every $\pi_D(d\mid\mathbf{pa}_D)\geq\epsilon_d^{\mathbf{pa}_D}$) has an NE $\pi_k\to\pi$.
- **Policy classes:** mixed $\mu^i\in\Delta(dom(\dot\Pi_{\boldsymbol{D}^i}))$, behavioural $\pi^i\in dom(\Pi_{\boldsymbol{D}^i})$, pure (Def 24). **Prop 5:** behavioural NE need not exist; **Prop 6:** perfect recall ⇒ behavioural NE exists; **Prop 7:** perfect recall ⇒ sufficient recall (not conversely).

**Algorithms:** `maid2efg` / `efg2maid` conversions (Sec 6, App A.1) for equivalence proofs and to reuse EFG solvers; backward induction over s-subgames for SPE (App C.2), shown to compute NEs faster than the equivalent EFG; an algorithm for the "closest possible world" invariant-decision-rule set $\boldsymbol{\Pi}(\mathcal{I})$ for counterfactuals (App B.1); the 3-step abduction–action–prediction procedure generalised to sets of rational outcomes (Def 23).

**Relevance to pgmpy:** This is the most complete data-model spec for the proposal. A MAID/CG/SCG = a typed DAG (chance/decision/utility nodes) with **per-agent ownership** ($\boldsymbol{D}^i,\boldsymbol{U}^i$) layered on pgmpy's existing role-annotated base graphs; decision CPDs are deliberately *unparameterised* until a policy is supplied. The **mechanised graph** is a second, augmented DAG adding one mechanism node per variable ($\Pi_D$ decision-rule, $\Theta_V$ parameter) plus possibly-cyclic edges — a distinct graph object. The **relevance graph** is derived purely by **d-separation on the mechanised graph** (s-/$\mathcal{R}$-reachability), so pgmpy's existing d-separation engine suffices. Subgames = node sets closed under directed paths + reachability (found via SCC condensation of the relevance graph). Equilibrium solving reuses VariableElimination for $\mathbb{E}_\pi[U]$, backward induction over subgames, and optional MAID↔EFG export to Gambit. SCGs map onto pgmpy's SCM/`CausalInference` do-calculus but require the **pre-policy (intervene-on-mechanism) vs post-policy (intervene-on-object)** distinction and a credal-set return type (multiple rational outcomes).

---

### Equilibrium Refinements for Multi-Agent Influence Diagrams: Theory and Practice (AAMAS-21, 2021)
**Authors:** Lewis Hammond, James Fox, Tom Everitt, Alessandro Abate, Michael Wooldridge
**Link:** https://arxiv.org/abs/2102.05008
**TL;DR:** The conference precursor to the SCG paper: it defines MAID **subgames**, **subgame-perfect** and **trembling-hand-perfect** equilibria via the relevance graph, proves they correspond to their EFG counterparts, and ships an open-source solver that exploits subgame decomposition.

**Problem & motivation:** MAIDs had a Nash-equilibrium notion (Koller & Milch) but lacked the refinement concepts (SPE/THPE) needed to rule out non-credible threats and non-robust strategies. This paper transfers those refinements to MAIDs, showing the DAG structure both *preserves* EFG refinements and *reveals more subgames* than the corresponding tree, enabling faster equilibrium computation.

**Formal setup & key definitions:**
- **MAID:** triple $(\boldsymbol{N},\boldsymbol{V},\boldsymbol{E})$ with agents $\boldsymbol{N}=\{1,\dots,n\}$ and a DAG whose vertices partition into decision $\boldsymbol{D}$, utility $\boldsymbol{U}$, chance $\boldsymbol{X}$ nodes, with per-agent $\{\boldsymbol{D}^i\}$, $\{\boldsymbol{U}^i\}$.
- **Decision rules / policies:** pure decision rule $\pi_D(d\mid\mathbf{pa}_D)\in\{0,1\}$; mixed decision rule $\pi_D(d\mid\mathbf{pa}_D)\in[0,1]$; agent $i$'s **policy** $\pi^i=\pi_{\boldsymbol{D}^i}$; **partial policy profile** $\pi_{-A}=(\pi^1,\dots,\pi^{i-1},\pi^{i+1},\dots,\pi^n)$.
- **Subgame base & subgame (Def 3.1):** $\boldsymbol{V}'\subseteq\boldsymbol{V}$ is a **subgame base** if (i) for any $X,Y\in\boldsymbol{V}'$ and directed path $X\to\cdots\to Y$, all intermediate nodes lie in $\boldsymbol{V}'$, and (ii) $\boldsymbol{V}'$ is closed under r-reachability (if $Z$ is r-reachable from a decision $D\in\boldsymbol{V}'$ then $Z\in\boldsymbol{V}'$). The **MAID subgame** $\mathcal{M}'=(\boldsymbol{N}',\boldsymbol{V}',\boldsymbol{E}')$ restricts players to $\boldsymbol{N}'=\{i\mid\boldsymbol{D}^i\cap\boldsymbol{V}'\neq\emptyset\}$.
- **Strategic relevance:** $D_l$ is strategically relevant to $D_k$ if there exist profiles $\pi,\pi'$ and a rule $\pi_{D_k}$ with $\pi_{D_k}$ optimal for $\pi$, $\pi$ and $\pi'$ differing only at $D_l$, and $\pi_{D_k}$ not optimal for $\pi'$.
- **Relevance graph:** $Rel(\mathcal{M})=(\boldsymbol{D},\boldsymbol{E}^{Rel})$ over decision nodes with edge $D_j\to D_k$ iff $D_k$ is **r-reachable** from $D_j$. The **condensed relevance graph** $ConRel(\mathcal{M})$ contracts maximal SCCs into single nodes (acyclic), and subgraphs closed under descendants induce MAID subgames — enabling backward induction.

**Key equations:**
- Expected utility: $\mathcal{U}^i_{\mathcal{M}}(\pi):=\sum_{U_j\in\boldsymbol{U}^i}\sum_{u_j\in dom(U_j)} u_j\,\Pr^{\pi}(U_j=u_j)$.
- **Nash equilibrium:** a full profile $\pi$ is an NE if for every $i$, $\mathcal{U}^i_{\mathcal{M}}(\pi^i,\pi^{-i})\geq\mathcal{U}^i_{\mathcal{M}}(\hat\pi^i,\pi^{-i})$ for all $\hat\pi^i\in\Pi^i$.
- **Trembling-hand perturbed game:** for perturbation vector $\{\delta_k\}$ with each $\epsilon^d_{\mathbf{pa}_D}\in(0,1)$, $\mathcal{M}(\delta_k)$ forces $\pi_D(d\mid\mathbf{pa}_D)\geq\epsilon^d_{\mathbf{pa}_D}$.

**Main results / theorems:**
- **SPE (Def 3.2):** $\pi$ is a subgame-perfect equilibrium iff it is an NE in *every* MAID subgame of $\mathcal{M}$.
- **THPE (Def 3.2):** $\pi$ is trembling-hand-perfect iff there is a sequence $\{\delta_k\}$ with $\lim_{k\to\infty}\|\delta_k\|_\infty=0$ such that each $\mathcal{M}(\delta_k)$ has an NE $\pi_k$ with $\lim_{k\to\infty}\pi_k=\pi$.
- **Equivalence:** the MAID↔EFG transformations (`maid2efg`/`efg2maid`) preserve NE, SPE, and THPE (via bijections between strategy spaces / policy-equivalence quotient partitions); MAID subgames correspond to EFG subgames, and MAIDs may possess more subgames than the equivalent EFG.

**Algorithms:** open-source implementation that (i) decomposes a game using the **condensed relevance graph** (SCCs), (ii) solves independent subproblems identified by r-reachability, (iii) converts MAIDs to EFGs for existing solvers (**Gambit**), and (iv) applies Pfeffer–Gal-style reasoning patterns; experiments show subgame decomposition computes NEs more efficiently than solving the monolithic EFG.

**Relevance to pgmpy:** Provides the concrete, *implementable* spec for the relevance graph and subgame machinery underlying the SCG paper. Data structures: typed DAG with per-agent decision/utility ownership; a `relevance_graph` over decisions built from r-reachability (a d-separation query); its **condensation** (SCCs — directly `networkx.condensation`) to find subgames and order backward induction. Algorithms: NE via subgame decomposition; SPE = NE-in-every-subgame check; THPE via $\epsilon$-perturbed CPDs; export to Gambit for the per-subgame solve. Expected utility = inference over the policy-induced BN (reuse `VariableElimination`). This is the cleanest blueprint for what an equilibrium-solving layer in pgmpy must expose.

---

### On Imperfect Recall in Multi-Agent Influence Diagrams (TARK 2023; EPTCS 379:201–220 — Best Paper)
**Authors:** James Fox, Matt MacDermott, Lewis Hammond, Paul Harrenstein, Alessandro Abate, Michael Wooldridge
**Link:** https://arxiv.org/abs/2307.05059 · pdf https://arxiv.org/pdf/2307.05059
**TL;DR:** Shows that under **imperfect recall** (forgetful or absent-minded agents) a behavioural-policy Nash equilibrium can fail to exist, and resolves this by using **mixed policies** and two new MAID **correlated-equilibrium** notions, with the mechanised graph making forgetfulness/absent-mindedness graphically explicit; also gives a full complexity map of MAID decision problems.

**Problem & motivation:** Prior MAID theory assumes perfect (or "sufficient") recall, but bounded rationality, teams with imperfect communication, and memoryless Markov-game policies all induce imperfect recall, where a behavioural NE may not exist. The paper distinguishes the two failure modes and supplies solution concepts that always exist.

**Formal setup & key definitions:**
- **MAID (Def 1):** $\mathcal{M}=(\mathcal{G},\boldsymbol{\theta})$, $\mathcal{G}=(N,\boldsymbol{V},\mathcal{E})$, $\boldsymbol{V}$ partitioned into $\boldsymbol{X}$, $\boldsymbol{D}=\bigcup_i\boldsymbol{D}^i$, $\boldsymbol{U}=\bigcup_i\boldsymbol{U}^i$; $\boldsymbol{\theta}=\{\theta_V\}_{V\in\boldsymbol{V}\setminus\boldsymbol{D}}$. Joint: $\Pr^{\boldsymbol{\pi}}(\boldsymbol{x},\boldsymbol{d},\boldsymbol{u}):=\prod_{V\in\boldsymbol{V}\setminus\boldsymbol{D}}\Pr(v\mid\mathbf{pa}_V)\prod_{D\in\boldsymbol{D}}\pi_D(d\mid\mathbf{pa}_D)$; expected utility $EU^i(\boldsymbol{\pi}):=\sum_{U\in\boldsymbol{U}^i}\sum_{u\in dom(U)}\Pr^{\boldsymbol{\pi}}(U=u)\cdot u$.
- **Mechanised graph:** adds mechanism parent $M_V$ to each $V$ — decision-rule node $\Pi_D=M_D$, parameter node $\Theta_V=M_V$ — with the key reading: *every edge between a mechanism and an object-level node represents an independent draw from the mechanism's distribution.*
- **Recall taxonomy (Defs 3–5):** **perfect recall** = $\exists$ ordering $D_1\prec\cdots\prec D_m$ over $\boldsymbol{D}^i$ with $(\mathbf{Pa}_{D_j}\cup D_j)\subseteq\mathbf{Pa}_{D_k}$ for $j<k$; **perfect information** = such an ordering over all $\boldsymbol{D}$. **Sufficient recall** = subgraph of $m\mathcal{G}$ on agent $i$'s rule nodes $\Pi_{\boldsymbol{D}^i}$ is acyclic; **sufficient information** = subgraph on all $\Pi_{\boldsymbol{D}}$ is acyclic. **Imperfect recall** = no such ordering; agent is **forgetful** if the offending $D_j,D_k$ have distinct decision rules, and **absent-minded** if some rule node in $m\mathcal{G}$ has **more than one outgoing edge to a decision node** (one shared rule controls multiple decisions, e.g. shared $\Pi_D$ for $D_1,D_2$).
- **Policy classes:** **behavioural** policies randomise independently at each decision; a **mixed policy** $\mu^i\in\Delta(\check{\boldsymbol{P}}^i)$ is a distribution over pure policies (randomise once at the outset); **behavioural mixtures** $\in\Delta(\boldsymbol{P}^i)$ randomise both at outset and per decision. A behavioural mixture introduces a **correlation node** $C^i$ ($\mathbf{Pa}_{C^i}=\emptyset$, $\mathbf{Ch}_{C^i}=\boldsymbol{D}^i$) with mechanism $\Pi_{C^i}$, so each $C^i$ value selects a behavioural policy.

**Key equations:**
- **Nash equilibrium (Def 2):** $\boldsymbol{\pi}$ is an NE in behavioural policies if for every $i$ and all alternative behavioural $\varpi^i$: $EU^i(\boldsymbol{\pi}^{-i},\pi^i)\geq EU^i(\boldsymbol{\pi}^{-i},\varpi^i)$.
- **Correlated equilibrium (Def 6):** $\kappa\in\Delta(\check{\boldsymbol{P}})$ is a CE iff $\forall i,\forall\check\pi^i,\varpi^i\in\check{\boldsymbol{P}}^i$: $\sum_{\check\pi^{-i}\in\boldsymbol{P}^{-i}}\kappa(\check\pi^i,\check\pi^{-i})EU^i(\check\pi^i,\check\pi^{-i})\geq\sum_{\check\pi^{-i}\in\boldsymbol{P}^{-i}}\kappa(\check\pi^i,\check\pi^{-i})EU^i(\check\pi^{-i},\varpi^i)$. A mediator samples $\boldsymbol{\pi}\sim\kappa$ and recommends each $i$ its pure policy $\check\pi^i$.
- **MAID correlated equilibrium (MAID-CE, Def 7):** add a correlation variable $C$ with $\mathbf{Pa}_C=\emptyset$, $\mathbf{Ch}_C=\{C_D\}_{D\in\boldsymbol{D}}$, $\mathbf{Ch}_{C_D}=\{D\}$; the mediator **staggers** recommendations and *ceases* recommending to any agent who deviates — weaker incentive constraints than a CE (von Stengel–Forges extensive-form CE analogue), yielding a larger outcome set and possible Pareto-improvements.

**Main results / theorems:**
- **Prop 1:** both forgetfulness and absent-mindedness can prevent the existence of an NE in behavioural policies (even in zero-sum two-agent binary games; the grand best-response correspondence becomes non-convex-valued, violating Kakutani).
- **Prop 2:** a non-absent-minded agent has a pure policy at least as good as any behavioural policy against fixed $\boldsymbol{\pi}^{-i}$; an absent-minded agent can have a behavioural policy strictly better than every pure (and every mixed) policy (e.g. absent-minded driver: behavioural $\pi^1_D(e)=\tfrac13$ gives $EU^D=\tfrac43>1$). Under forgetfulness a behavioural policy always has an equivalent mixed policy (but not conversely under absent-mindedness).
- **Prop 3 (existence):** a MAID with **sufficient information** always has an NE in *pure* policies; with **sufficient recall**, an NE in *behavioural* policies; and **every** MAID has an NE in *mixed* policies (via Nash's theorem).
- **Prop 4:** a MAID-CE in bounded-treewidth MAIDs with sufficient recall is computable in poly-time (reduction to an LP).
- **Complexity (Props 5–10):** finding a mixed NE is **PPAD-hard** (Prop 5). `Is-Best-Response` is **NP$^{\mathrm{PP}}$-complete** (NP-complete at bounded treewidth; PP-complete if $|\boldsymbol{D}^i|$ and in-degree bounded) (Prop 6). `Is-Nash` is **coNP$^{\mathrm{PP}}$-complete** (coNP$^{\mathrm{PP}}$-hard for sufficient-information; coNP-hard without chance variables) (Prop 8). `Non-Emptiness` is **NEXPTIME-hard** in general, NEXPTIME-complete without chance variables (Prop 9). With sufficient information + bounded in-degree + tractable `Is-Best-Response`, a pure NE is poly-time findable (Prop 10).

**Algorithms:** MAID-CE via LP in bounded treewidth (extends Huang et al.; information sets in the EFG put in bijection with MAID decision contexts, but relaxed to *sufficient* rather than perfect recall); subgame/relevance-graph decomposition to localise computation; applications to **Markov games** (shared stationary mechanism $\pi^i:S\to\Delta(A^i)$ ⇒ absent-minded, behavioural policies) and **teams with imperfect communication**.

**Relevance to pgmpy:** Tells the library which policy *representations* and *equilibrium types* must be first-class: behavioural CPDs per decision, **mixed policies** (distribution over pure policies — needed for guaranteed NE), behavioural mixtures (correlation nodes $C^i$/$\Pi_{C^i}$), and two correlated-equilibrium objects (CE via mediator distribution $\kappa\in\Delta(\check{\boldsymbol{P}})$; MAID-CE via per-decision correlation variables $C_D$ with deviation-triggered cutoff). Recall properties (perfect/sufficient/imperfect, forgetful/absent-minded) are **graph predicates checkable in poly-time on the mechanised graph** (acyclicity of rule-node subgraphs; out-degree of $\Pi_D$ to decisions) — these gate which solver is valid. The complexity results bound what an automated solver can promise. Markov-game / team modelling motivates shared decision-rule nodes (parameter tying across decisions) as a graph feature.

---

### Higher-Order Belief in Incomplete Information MAIDs (AAMAS 2025)
**Authors:** Jack Foxabbott, Rohan Subramani, Francis Rhys Ward
**Link:** https://arxiv.org/abs/2503.06323
**TL;DR:** Extends MAIDs to **incomplete information** without a common prior by nesting a *set of subjective MAIDs* with per-agent priors, thereby representing arbitrarily deep **higher-order beliefs** ("$i$ believes that $j$ believes…"), proves equivalence to incomplete-information EFGs, and proposes a **recursive best-response** solution concept for finite-depth belief hierarchies.

**Problem & motivation:** Standard MAIDs assume common knowledge of the game; many realistic settings (e.g. AI-evaluation/deception games) involve agents with *different subjective beliefs* about the game and about each other's beliefs, with no common prior. The paper builds the machinery to model these belief hierarchies in the MAID formalism and argues classical Nash equilibria are unrealistic there (they presuppose common knowledge of rationality).

**Formal setup & key definitions:**
- **MAID (Def 3.1):** $\mathcal{M}=(\mathcal{G},\boldsymbol{\theta})$, $\mathcal{G}=(N,\boldsymbol{V},\mathcal{E})$ with $N=\{1,\dots,n\}$ and $\boldsymbol{V}$ partitioned into chance $\boldsymbol{X}$, decision $\boldsymbol{D}=\bigcup_i\boldsymbol{D}^i$, utility $\boldsymbol{U}=\bigcup_i\boldsymbol{U}^i$; edges are probabilistic-dependence (solid) or information links (dashed); $\boldsymbol{\theta}=\{\theta_V\}$ gives CPDs.
- **Incomplete-Information MAID (II-MAID, Def 4.1):** a tuple $\mathcal{S}=(\boldsymbol{N},S^*,\boldsymbol{S})$ where $\boldsymbol{S}$ is a set of **subjective MAIDs**, $S^*\in\boldsymbol{S}$ the **correct objective model**, and each $S=(\mathcal{M}^S,(P_i^S)_{i\in\boldsymbol{N}})\in\boldsymbol{S}$ pairs a MAID $\mathcal{M}^S$ with **priors** $P_i^S$ over $\boldsymbol{S}$. **Coherence condition** ("agents know their own beliefs"): $P_i^S(\{S'\in\boldsymbol{S}: P_i^{S'}=P_i^S\})=1$ for all $i,S$. The self-referential structure of $\boldsymbol{S}$ encodes the **higher-order belief hierarchy**: in $S$, $i$ believes $j$'s type is $P_j^S$, and each $S'$ itself specifies beliefs over $\boldsymbol{S}$, so "$i$ believes $j$ believes …" follows by composition; agent $i$ "observes" $P_i^{S^*}$ at the start of the game.
- **Consistency / common-prior assumption (Asm 4.2, *relaxed*):** $p(S')=\sum_{S\in\boldsymbol{S}}P_i^S(S')\,p(S)$ for all $S',i$; the paper explicitly relaxes this (their evaluation game cannot satisfy it), which is what permits genuine higher-order-belief disagreement.
- **Incomplete-Information EFG (II-EFG, Defs 5.1–5.2):** a **belief space** $\Pi=(Y,\mathcal{Y},\mathbf{s},(b_i)_{i\in\boldsymbol{N}})$ with measurable state space $(Y,\mathcal{Y})$, map $\mathbf{s}:Y\to S$ to EFGs, and beliefs $b_i:Y\to\Delta(Y)$; coherence $b_i(\{\omega':b_i(\omega')=b_i(\omega)\}\mid\omega)=1$. An II-EFG is $G=(\boldsymbol{N},S,\Pi)$.

**Key equations:**
- **Subjective expected utility:** $\mathcal{U}^i_{S^*}(\boldsymbol{\pi}):=\sum_{S\in\boldsymbol{S}}\sum_{U\in\boldsymbol{U}^i(S)}\sum_{u\in dom(U)} u\,\Pr^{\boldsymbol{\pi}}_S(U=u)\,P_i^{S^*}(S)$ — utility averaged over subjective models weighted by $i$'s beliefs.
- **Best response / NE (Def 3.4):** $\pi^i$ is a best response to $\pi^{-i}$ if $\sum_{U\in\boldsymbol{U}^i}\mathbb{E}_{(\pi^i,\pi^{-i})}[U]\geq\sum_{U\in\boldsymbol{U}^i}\mathbb{E}_{(\hat\pi^i,\pi^{-i})}[U]$ for all $\hat\pi^i$; $\boldsymbol{\pi}$ is a Nash equilibrium if every policy is a best response.

**Main results / theorems:**
- **Equivalence (Thm ~6.2):** an II-MAID $\mathcal{S}=(\boldsymbol{N},S^*,\boldsymbol{S})$ and an II-EFG $G=(\boldsymbol{N},S,\Pi)$ at the interim stage are *equivalent* if there is a bijection between strategies and policy(-partition) classes that (i) differs only on null contexts and (ii) preserves expected utility ($\mathcal{U}^i_{\mathcal{S}}(\pi)=$ the II-EFG payoff). Consequently II-MAIDs **inherit** existence of equilibria from II-EFGs / Harsanyi games without common priors.
- **Existence (Sec 7):** Nash equilibria exist in II-MAIDs via the II-EFG correspondence; infinite belief hierarchies are representable.
- **Recursive best response (Sec 8):** for **finite-depth** II-MAIDs the authors give an alternative solution concept that assigns best responses **bottom-up** through the belief hierarchy ("repeatedly assigning best responses at the bottom of the belief hierarchy until all policies in the original game are assigned"), avoiding the counterintuitive prescriptions of Nash equilibrium when common knowledge of rationality fails.

**Algorithms:** `maid2efgII` / `efg2maidII` — convert each MAID (resp. EFG) throughout the belief hierarchy using the existing `maid2efg`/`efg2maid` transformations and then match corresponding features (details in appendix); plus the recursive-best-response procedure over the finite belief tree.

**Relevance to pgmpy:** Defines the most demanding data structure for the proposal: a **set of subjective MAIDs** $\boldsymbol{S}$ plus, per agent per model, a **prior $P_i^S$ over that same set $\boldsymbol{S}$** (a recursive/self-referential container), with one element flagged as the objective model $S^*$, subject to a coherence constraint. A library would need a `BeliefHierarchy`/`IncompleteInfoMAID` object wrapping multiple MAIDs and a (possibly nested, possibly finite-depth) prior structure, the (relaxable) common-prior consistency check as a validation routine, subjective-expected-utility computation that marginalises over $\boldsymbol{S}$ weighted by beliefs, and solvers for both Bayes-Nash and the bottom-up recursive-best-response concept. Interoperability hooks: II-MAID↔II-EFG conversion built on the same `maid2efg`/`efg2maid` primitives the other Batch-B papers require, so a shared MAID↔EFG transformation utility should be a core, reusable component.

## C. Agency & World Models

### Discovering Agents (Artificial Intelligence journal, 2023; arXiv:2208.08345, 2022)
**Authors:** Zachary Kenton, Ramana Kumar, Sebastian Farquhar, Jonathan Richens, Matt MacDermott, Tom Everitt (DeepMind / Oxford)
**Link:** https://arxiv.org/abs/2208.08345
**TL;DR:** Agents are formally characterised as nodes whose decision rule would *adapt* if the downstream consequences of their actions changed, and this "responsiveness" is detectable by interventions on *mechanism* variables, yielding algorithms that recover a game graph / causal influence diagram (CID) from purely interventional data.

**Problem & motivation:** Existing causal definitions of agent incentives, intent, etc. presuppose that you already know which nodes are decisions and which are utilities (i.e. you already have the game graph). The paper provides the missing causal, experimentally-testable criterion for *what counts as an agent/decision*, so that game graphs can be discovered rather than assumed. The core intuition: "agents are systems that would adapt their policy if their actions influenced the world in a different way."

**Formal setup & key definitions:**
- Standard SCM: each endogenous $V = f^V(\mathbf{V},\mathcal{E}^V)$; edge $W\to V$ iff $f^V$ depends on $W$. Family $\mathbf{Fa}^V=\mathbf{Pa}^V\cup\{V\}$; intervention $\mathrm{do}(Y=y)$ replaces structural equations.
- **Causal game / CID:** a CBN $M=(G,P)$ with endogenous variables partitioned into chance $\mathbf{X}$, decision $\mathbf{D}$ (no structural equation; a decision rule/policy $\pi^D$ is chosen) and utility $\mathbf{U}$ variables. Each agent $A$ controls $\mathbf{D}^A\subseteq\mathbf{D}$ and maximises $\sum_{U\in\mathbf{U}^A}U$. Game graph: square=decision, round=chance, diamond=utility, dotted edges into decisions = information links. Interventions can be **pre-policy** (agent aware, adapts) or **post-policy** (agent unaware).
- **Definition 1 (Mechanised SCM):** an SCM in which there is a partition of the endogenous variables $\boldsymbol{\mathcal V}=\mathbf{V}\cup\widetilde{\mathbf{V}}$ into object-level variables $\mathbf V$ (white nodes), and mechanism variables $\widetilde{\mathbf V}$ (black nodes), with $|\mathbf V|=|\widetilde{\mathbf V}|$. Each object-level variable $V$ has exactly one mechanism parent, denoted $\widetilde V$, that specifies the relationship between $V$ and the object-level parents of $V$. Edges split into object-level $E^{\text{obj}}$, mechanism $E^{\text{mech}}$, and functional $E^{\text{func}}$ ($\widetilde V\to V$). Object-level subgraph is acyclic; mechanism subgraph may be cyclic.
- **Definition 2 (Structural mechanism intervention):** an intervention $\widetilde v$ on $\widetilde V$ such that $V$ becomes conditionally independent of its object-level parents. This is the way to "cut off" a variable's response so an agent (if present) is *aware* of the cut.
- **Definition 3 (Edge-labelled mechanised SCM / terminal edges):** identifies $E^{\text{term}}\subseteq E^{\text{mech}}$ (dashdotted blue) edges $\widetilde W\to\widetilde V$ such that (1) "$\widetilde V$ responds to $\widetilde W$ even after any effects of $W$ on its children $\mathbf{Ch}^W$ have been removed by means of any structural mechanism interventions on $\mathbf{Ch}^W$"; and (2) "$\widetilde V$ does not respond to $\widetilde W$ if effects of $V$ on its children $\mathbf{Ch}^V$ have been removed by means of all structural mechanism interventions on $\mathbf{Ch}^V$." Terminal edges flag (a) which variables are *intrinsically* valued (utilities) and (b) which variables *adapt for a downstream reason* (decisions).

**Key equations:**
- $\Pr(V\mid\mathbf{Pa}^V,\mathrm{do}(\widetilde V=\tilde v))=\Pr(V\mid\mathrm{do}(\widetilde V=\tilde v))$ — defining property (Eq. 1) of a structural mechanism intervention: it screens off object-level parents.
- Terminal-edge test for a utility: a variable $W$ is inherently valuable iff the agent still changes its policy in response to a change in $\widetilde W$ after the children of $W$ stop responding to $W$.
- Terminal-edge test for a decision: $V$ adapts for a downstream reason iff $\widetilde V$ still responds even when the children of $V$ are made non-responsive to $V$ (i.e. $V$ has no downstream effect).

**Main results / theorems:**
- **Lemma 1 (Leave-one-out causal discovery):** applied to interventional distributions from a (possibly cyclic) SCM, the leave-one-out procedure returns the correct causal graph. (Edge $W\to V$ exists iff, holding $do$ on all nodes but $V$ fixed and varying only the intervention on $W$, $P(V\mid\cdots)$ changes.)
- **Lemma 2:** Algorithm 1 returns the correct edge-labelled mechanised causal graph when structural mechanism interventions are available for all nodes.
- **Theorem 1 (Correctness of Algorithms 1+2):** Let $\widetilde{\mathcal M}_{\text{real}}$ be a mechanised causal game satisfying Assumptions 1–5. Let $\mathcal G_{\text{model}}$ be the game graph resulting from applying Algorithm 1 followed by Algorithm 2 to $\widetilde{\mathcal M}_{\text{real}}$. Then $\mathcal G_{\text{model}}=\mathcal G_{\text{real}}$. The proof shows all-and-only decisions map to decisions, all-and-only utilities map to utilities, and connected components of the terminal-edge graph give the correct agent colouring.
- **Lemma 3:** links terminal mechanism edges to Koller–Milch *s-reachability*: for decision $D$, $(\widetilde Y,\widetilde D)\in E^{\text{mech}}$ iff $Y$ is s-reachable from $D$; for utility $Y\in\mathbf U^A$, $(\widetilde Y,\widetilde D)\in E^{\text{term}}$ iff there is a directed path $D\dashrightarrow Y$ not through another utility.
- **Assumptions for correctness:** (1) each weakly-connected component of the decision-utility subgraph is one agent with ≥1 decision and ≥1 utility; (2) agents play subgame-perfect/best-response equilibria under every mechanism intervention; (3) agents have a tie-breaking preference order over decision rules (no unmotivated switches); (4) only decision mechanisms have incoming terminal edges (weak independent-causal-mechanism assumption); (5) interventions on $\widetilde V$ can instantiate any deterministic function of parents (so the necessary soft interventions are enactable).

**Algorithms:**
- **Leave-one-out causal discovery** (subroutine): for each $V$, for each candidate parent $W$, set $\mathbf Y=\mathbf V\setminus\{V,W\}$; if $\exists\, \mathbf y, w, w'$ with $P(V\mid do(\mathbf Y=\mathbf y,W=w))\neq P(V\mid do(\mathbf Y=\mathbf y,W=w'))$ then add edge $(W,V)$.
- **Algorithm 1 — Mechanised Causal Graph Discovery:** run leave-one-out on the *combined* object+mechanism variables to get $E^{\text{obj}},E^{\text{mech}},E^{\text{func}}$ (and validate the mechanised-SCM structure). Then label terminal edges: for each $(\widetilde W,\widetilde V)\in E^{\text{mech}}$, (i) apply structural mechanism interventions to all of $\mathbf{Ch}^W$ and test whether $\widetilde V$ still responds to varying $\widetilde W$ (condition 1 ⇒ add to $E^{\text{term}}$); (ii) apply structural interventions to $\mathbf{Ch}^V$ and test whether $\widetilde V$ stops responding to $\widetilde W$ (condition 2 ⇒ keep; else remove from $E^{\text{term}}$).
- **Algorithm 2 — Agency Identification (mechanised graph → game graph):** initialise $\mathbf D,\mathbf U=\varnothing$; for each terminal edge $(\widetilde W,\widetilde V)\in E^{\text{term}}$: add target $V$ to decisions $\mathbf D$ and source $W$ to utilities $\mathbf U$; remaining object-level nodes become chance $\mathbf X=\mathbf V\setminus(\mathbf U\cup\mathbf D)$; colour each connected component of the terminal-edge graph as one agent; output game graph $\mathcal G=(N,\mathbf V,E^{\text{obj}})$.
- **Algorithm 3 — Mechanism Identification:** the inverse map (game graph → edge-labelled mechanised graph), proving the two representations are interconvertible.

**Relevance to pgmpy:** Implementable as (a) a `MechanisedSCM`/`CID` data structure extending the base `DAG`/`PDAG` with a bipartite object-vs-mechanism node partition, three typed edge sets ($E^{\text{obj}},E^{\text{mech}},E^{\text{func}}$), and a `terminal` edge-label flag, plus typed decision/utility/chance node roles; (b) a *structural mechanism intervention* primitive (a soft/`do`-intervention that severs a node from its object-level parents); (c) a generic leave-one-out interventional causal-discovery routine that already works for cyclic graphs; (d) the Algorithm-1/2 pipeline turning interventional query results into a game graph. This directly extends pgmpy's existing node-role annotations and intervention machinery toward decision/game-theoretic models.

---

### Robust agents learn causal world models (ICLR 2024, oral)
**Authors:** Jonathan Richens, Tom Everitt (Google DeepMind)
**Link:** https://arxiv.org/abs/2402.10877
**TL;DR:** Any agent that satisfies a regret bound across a large set of *local* distributional shifts must have implicitly learned an approximate causal Bayesian network of its environment, and that learned model converges to the true causal model as the agent approaches optimality.

**Problem & motivation:** It was an open question whether causal models are *necessary* for robust generalisation or whether other inductive biases suffice. The paper proves necessity (and sufficiency): learning to adapt to domain shifts is *informationally equivalent* to learning a causal model of the data-generating process. This connects emergent capabilities, transfer learning, and the causal-hierarchy theorem.

**Formal setup & key definitions:**
- **Causal Bayesian network (Def. 1):** $M=(G,P)$ with $P(V_1,\dots,V_n)=\prod_i P(V_i\mid\mathbf{Pa}_{V_i})$; causal iff the truncated factorisation gives $P(\mathbf v\mid do(\mathbf x))$.
- **Local intervention (Def. 2):** $\sigma=do(V_i=f(v_i))$ — a soft intervention applying a map to the states of a single variable, *not* conditioned on other endogenous variables; transforms the mechanism as $P(v_i\mid\mathbf{pa}_i;\sigma)=\sum_{v_i':f(v_i')=v_i}P(v_i'\mid\mathbf{pa}_i)$. Local interventions are compatible with *all* causal structures, so they can be used without knowing $G$. Hard interventions, translations, logical NOT are special cases.
- **Mixtures of interventions (Def. 3):** $\sigma^*=\sum_i p_i\sigma_i$, giving $P(\mathbf v\mid\sigma^*)=\sum_i p_i P(\mathbf v\mid\sigma_i)$. $\Sigma$ = set of all such mixtures.
- **CID (Def. 4):** CBN with $\mathbf V=(\{D\},\{U\},\mathbf C)$; utility $U(\mathbf{pa}_U)$ is a real-valued function of its parents; policy $\pi(d\mid\mathbf{pa}_D)$ set by the agent to maximise $\mathbb E^\pi[U]=\mathbb E[U\mid do(D=\pi(\mathbf{pa}_D))]$.
- **Regret:** $\delta:=\mathbb E^{\pi^*}[U]-\mathbb E^{\pi}[U]$, the utility shortfall versus the optimal policy $\pi^*$.

**Key equations:**
- $P(v_i\mid\mathbf{pa}_i;\sigma)=\sum_{v_i':f(v_i')=v_i}P(v_i'\mid\mathbf{pa}_i)$ — the mechanism transformation under a local intervention (Eq. 1).
- $\delta:=\mathbb E^{\pi^*}[U]-\mathbb E^{\pi}[U]$ — regret, the quantity the robust agent bounds across shifts.
- $|P'(v_i\mid\mathbf{pa}_i)-P(v_i\mid\mathbf{pa}_i)|\le\gamma(\delta)$ — error of the *extracted* CBN parameters, with $\gamma(0)=0$ and $\gamma(\delta)$ growing linearly in $\delta$ for small regret (the headline accuracy-vs-regret scaling).

**Main results / theorems (assumptions: A1 unmediated decision task $\mathbf{Desc}_D\cap\mathbf{Anc}_U=\varnothing$; A2 domain dependence — the optimal policy genuinely depends on the environment distribution):**
- **Theorem 1 (optimal agents ⇒ exact model):** for almost all CIDs $M=(G,P)$ satisfying Assumptions 1 and 2, one can identify the DAG $G$ and joint distribution $P$ over all ancestors of the utility $\mathbf{Anc}_U$ given $\{\pi^*_\sigma(d\mid\mathbf{pa}_D)\}_{\sigma\in\Sigma}$, where $\pi^*_\sigma$ is the optimal policy in domain $\sigma$ and $\Sigma$ is the set of all mixtures of local interventions. ("Almost all" = the failure set has Lebesgue measure zero.)
- **Theorem 2 (bounded regret ⇒ approximate model):** for almost all such CIDs, one can identify an approximate causal model $M'=(P',G')$ given $\{\pi_\sigma(d\mid\mathbf{pa}_D)\}_{\sigma\in\Sigma}$ with $\mathbb E^{\pi_\sigma}[U]\ge\mathbb E^{\pi^*_\sigma}[U]-\delta$; the parameters satisfy $|P'(v_i\mid\mathbf{pa}_i)-P(v_i\mid\mathbf{pa}_i)|\le\gamma(\delta)\ \forall V_i\in\mathbf V$ where $\gamma(0)=0$ and $\gamma(\delta)$ grows linearly in $\delta$ for small regret $\delta\ll\mathbb E^{\pi^*}[U]$. (For finite $\delta$, $G$ may only be recoverable as a subgraph $G'\subseteq G$, dropping weak causal relations.)
- **Theorem 3 (sufficiency / converse):** given the true causally-sufficient CBN, optimal policies are identifiable for any utility and all soft interventions; and given an approximate model with $|P'(v_i\mid\mathbf{pa}_i)-P(v_i\mid\mathbf{pa}_i)|\le\epsilon\ll1$, one can identify regret-bounded policies with regret $\delta$ growing linearly in $\epsilon$.
- **Corollary 1:** a regret-bounded policy $\pi(d\mid\mathbf{pa}_D,\sigma)$ is *informationally equivalent* to an approximation $M'$ of the environment CBN $M$, with $M'\to M$ smoothly as $\delta\to0$.
- **Interpretation:** Theorem 1 situates the result in Pearl's causal hierarchy — the set $\Pi^*_C$ of optimal-policy queries under all shifts is "L2-complete." Together Theorems 1–3 imply generalisation under domain shift $\equiv$ learning a causal model; a sharper "good regulator theorem."

**Algorithms (proof construction):** Assume oracle access to optimal policies $\pi^*_\sigma$ for any local intervention on the causally-sufficient set $\mathbf C$. (1) Query the oracle with many *mixtures* of local interventions $\sigma\in\Sigma$. (2) Identify the "critical" mixtures at which the optimal policy *changes* (interventions that flip the utility-maximising decision). (3) These critical points identify the CBN parameters, pinning down both the graph $G(\mathbf{Anc}_U)$ and the joint distribution $P(\mathbf{Anc}_U)$. Used as a causal-discovery algorithm over latents and validated on synthetic CIDs (learned-DAG error and parameter error both → 0 as the regret bound tightens).

**Relevance to pgmpy:** Needs (a) a `CID`/CBN data structure with decision, utility, chance node roles (shared with "Discovering Agents"); (b) a *local/soft intervention* primitive and *mixtures of interventions* over a CBN; (c) optimal-policy / expected-utility computation $\mathbb E^\pi[U]=\mathbb E[U\mid do(D=\pi)]$ and a `regret` metric; (d) an interventional causal-discovery routine that recovers $G(\mathbf{Anc}_U)$ and $P(\mathbf{Anc}_U)$ from policy responses to mixtures, including over latent variables — a natural addition to pgmpy's causal-discovery module. The linear error-vs-regret bound gives a principled accuracy target for learned models.

---

### General agents contain world models (ICML 2025) — listed on site as "General agents need world models"
**Authors:** Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt (Google DeepMind)
**Link:** https://arxiv.org/abs/2506.01622
**TL;DR:** Any goal-conditioned agent that meets a regret bound on multi-step (depth $>1$) goals necessarily encodes an approximate transition model (world model) of its environment in its policy, with model error shrinking as the agent's competence and goal-horizon grow; myopic (depth-1) agents are the sole exception.

**Problem & motivation:** Is a world model necessary for flexible goal-directed behaviour, or is model-free learning enough? The paper gives a minimalist, reward-free formalisation and proves world models are necessary, and moreover extractable from the policy — with consequences for safety, capability bounds, and eliciting world models from black-box agents.

**Formal setup & key definitions:**
- **Controlled Markov process (Def. 1):** an MDP without reward/discount, tuple $(\mathbf S,\mathbf A,P_{ss'}(a))$ with transition function $P_{ss'}(a)=P(S=s'\mid A=a,S=s)$.
- **Assumption 1:** finite, communicating (irreducible), stationary cMP with $|\mathbf A|\ge2$.
- **Goals (Def. 2):** an LTL expression $\varphi=\mathcal O([(s,a)\in\mathbf g])$ over goal set $\mathbf g\subseteq\mathbf S\times\mathbf A$, with temporal operator $\mathcal O\in\{\bigcirc,\Diamond,\top\}$ (Next, Eventually, Now). $\tau\models\varphi$ means trajectory $\tau$ satisfies $\varphi$.
- **Composite goals (Def. 3):** a *sequential* goal $\psi=\langle\varphi_1,\dots,\varphi_n\rangle$ requires sub-goals in order; $depth(\psi)=n$. A composite goal is a disjunction $\psi=\bigvee_i\psi_i$; $\Psi_n$ = all composite goals of depth $\le n$.
- **Agents:** goal-conditioned policies $\pi:h_t,\psi\mapsto a_t$ (may condition on full history).
- **Optimal goal-conditioned agent (Def. 4):** $\pi^*=\arg\max_\pi P(\tau\models\psi\mid\pi,s_0)$ for all $s_0$ with $P(s_0)>0$ and all $\psi\in\boldsymbol\Psi$.
- **Bounded goal-conditioned agent (Def. 5):** failure rate $\delta\in[0,1]$, max depth $n$, satisfying the regret bound below for all $\psi\in\Psi_n$. Only *competence* is assumed — no rationality axioms.
- **World model:** any approximation $\hat P_{ss'}(a)$ of the transition function with $|\hat P_{ss'}(a)-P_{ss'}(a)|\le\epsilon$.

**Key equations:**
- $\pi^*=\arg\max_\pi P(\tau\models\psi\mid\pi,s_0)$ — optimal goal-conditioned agent (Eq. 1).
- $P(\tau\models\psi\mid\pi,s_0)\ \ge\ \max_\pi P(\tau\models\psi\mid\pi,s_0)\,(1-\delta)\quad\forall\psi\in\Psi_n$ — the bounded-agent / regret condition (Eq. 2).
- $\big|\hat P_{ss'}(a)-P_{ss'}(a)\big|\ \le\ \sqrt{\dfrac{2\,P_{ss'}(a)\,(1-P_{ss'}(a))}{(n-1)(1-\delta)}}$ — exact world-model error bound (Theorem 1).
- $\big|\hat P_{ss'}(a)-P_{ss'}(a)\big|\ \sim\ \mathcal O(\delta/\sqrt n)+\mathcal O(1/n)$ — asymptotic scaling for $\delta\ll1,\ n\gg1$: error vanishes as competence rises ($\delta\to0$) or goal depth grows ($n\to\infty$, as $n^{-1/2}$).
- $P_{\mathcal M}(\tau\models\psi\mid\pi,s_0)\ge\max_\pi P_{\mathcal M}(\tau\models\psi\mid\pi,s_0)(1-\delta)$ — variant (Eq. 3) recovering the agent's *subjective* world model $\mathcal M$ when $\delta$-consistency is taken w.r.t. its own model.

**Main results / theorems:**
- **Theorem 1 (world models are necessary & extractable):** Let $P_{ss'}(a)=P(S_{t+1}=s'\mid A_t=a,S_t=s)$ satisfy Assumption 1. Let $\pi$ be a goal-conditioned agent (Def. 5) with a maximum failure rate $\delta$ for all goals $\psi\in\Psi_n$ where $\Psi_n$ is the set of all composite goals with maximum goal depth $n>1$. Then $\pi$ fully determines a model for the environment transition probabilities $\hat P_{ss'}(a)$ with errors satisfying the bound above. The relative error $\hat P_{ss'}(a)/P_{ss'}(a)$ can blow up for rare transitions, so high-success or long-horizon agents must learn *high-resolution* models. The proof only requires the bound to hold on a small subset of $\Psi_n$ of size $\mathcal O(n|\mathbf A||\mathbf S|^2)$.
- **Theorem 2 (myopic exception):** for myopic goals $\Psi_{\text{myopic}}\subseteq\Psi_1$ (goal state attained immediately after the first action, $\varphi=\bigcirc([(s,a)\in\mathbf g])$), any bound on $|\hat P_{ss'}(a)-P_{ss'}(a)|$ derivable from an optimal myopic $\pi^*$ is trivial ($\epsilon=1$) and tight — proven by explicitly constructing a myopic agent that is optimal for any $P_{ss'}(a)\in[0,1]$. So world models become necessary only at multi-step horizons.
- **Empirics (Sec. 3.1):** on a random 20-state/5-action sparse cMP, recovered-model error decreases as the agent generalises to higher-depth goals and scales as $\mathcal O(n^{-1/2})$, even when the agent violates Def. 5 on some goals (low *average* regret suffices).

**Algorithms:** **Algorithm 1 (world-model recovery; Algorithm 2 = simplified version):** universal, unsupervised, takes only the agent's policy $\pi$ (no activations/architecture). Query the policy with composite goals that pose *either-or* decisions between two incompatible sub-goals $\psi=\psi_a\vee\psi_b$ (e.g. reaching a target state via different action choices, structured so the optimal success probability is binomial in the underlying $P_{ss'}(a)$, $\max_\pi P(\tau\models\psi(r,n))=\binom{n}{r}P_{ss'}(a)^r(1-P_{ss'}(a))^{n-r}$). Because the bounded agent picks the action maximising goal-satisfaction probability, its choice reveals which sub-goal is more probable, and sweeping these queries pins $\hat P_{ss'}(a)$ to bounded error. Key subtlety: the recovery map is from the *policy*, strictly weaker than mechanistic-interpretability probes that read activations.

**Relevance to pgmpy:** Implementable as (a) a reward-free `ControlledMarkovProcess` model with a transition tensor $P_{ss'}(a)$; (b) an LTL goal/composite-goal specification layer and a goal-conditioned-policy abstraction; (c) a `competence`/`failure-rate` ($\delta$, depth $n$) metric over goal sets; (d) a model-extraction estimator that queries a black-box policy with constructed either-or goals to fit a transition model with the stated error bars — a concrete "elicit world model from agent" algorithm. Complements pgmpy's existing DBN/MDP-style structures and inference.

---

### The Limits of Predicting Agents from Behaviour (ICML 2025)
**Authors:** Alexis Bellot, Jonathan Richens, Tom Everitt (Google DeepMind)
**Link:** https://arxiv.org/abs/2506.02923
**TL;DR:** Assuming an agent's behaviour is governed by an (unknown) SCM "world model," its actions, fairness, and harm perceptions in *new/shifted* environments are only *partially* identifiable from behavioural data — the paper derives the exact, often tight, bounds (a causal-identification / partial-identification view of "intentional-stance" prediction).

**Problem & motivation:** To deploy AI safely we want to attribute beliefs/goals and predict behaviour in untested situations. The paper asks precisely how well an agent's beliefs are inferable from behaviour, and how reliably those beliefs predict out-of-distribution actions. Result: with an assumption of competence and optimality, the behaviour of AI systems *partially* determines their actions in novel environments. It is the behavioural-inference dual of "Robust agents learn causal world models."

**Formal setup & key definitions:**
- **SCM (Def. 1):** $\mathcal M=\langle V,U,\mathcal F,P\rangle$, recursive, $v:=f_V(\mathbf{pa}_V,\mathbf u_V)$; interventions $do(\mathbf x)$ yield sub-model $\mathcal M_{\mathbf x}$ and $P_{\mathbf x}(\mathbf y)\equiv P(\mathbf y\mid do(\mathbf x))$. A *shift* $\sigma$ is a sub-model $\mathcal M_\sigma$ encoding discrepancies (changed mechanisms/exogenous distributions) between a reference SCM and a deployment SCM.
- The AI is assumed to act per an internal SCM $\widehat{\mathcal M}$ (its world model) over $V$ (decision $D$/$A$, covariates $\mathbf C$, utility $Y$); **Definition 2 (Beliefs):** an AI belief is a probabilistic statement derived from its internal model $\widehat{\mathcal M}$, e.g. $P^{\widehat{\mathcal M}_d}(Y=y)$.
- Decisions sampled from policy $\pi(d\mid\mathbf c)$, chosen to maximise perceived utility.
- **Definition 3 (Grounding):** the AI is grounded in domain $\mathcal M$ if $P^{\widehat{\mathcal M}_d}(V)\equiv P^{\mathcal M_d}(V)$ for all $d$, i.e. $\hat P_d(V)=P_d(V)$ — its beliefs about decision effects match reality in-domain. (Sec. 5 relaxes to *approximate grounding*, Def. 8: $\psi(\hat P_d,P_d)\le\delta$.)
- Observers see only behaviour and consequences $P_d(V)$; the internal model $\widehat{\mathcal M}$ and out-of-domain mechanisms are hidden. $\mathbb M$ = set of "valid" SCMs compatible with observations.

**Key equations:**
- $\arg\max_\pi\ \mathbb E_{P^{\widehat{\mathcal M}}}[\,Y\mid do(\pi)\,]$ — the agent's policy is a utility maximiser under its own model (Eq. 1).
- $\Delta_{d>d^*}:=\mathbb E_{P^{\widehat{\mathcal M}}}[Y\mid do(\sigma,d),\mathbf c]-\mathbb E_{P^{\widehat{\mathcal M}}}[Y\mid do(\sigma,d^*),\mathbf c]$ — the *preference gap* between two decisions under shift $\sigma$ (Eq. 3); positive ⇒ $d$ preferred.
- **Weak predictability (Def. 4):** $\min_{\widehat{\mathcal M}\in\mathbb M}\big(\Delta_{d>d^*}\big)>0$ for some $d$ — a decision $d^*$ is provably sub-optimal (rule-out-able) across *all* valid models.
- **Strong predictability (Def. 5):** $d^*=\arg\max_d\mathbb E_{P^{\widehat{\mathcal M}}}[Y\mid do(\sigma,d),\mathbf c]$ for all valid $\widehat{\mathcal M}$ — the optimal action is uniquely identifiable.
- **Counterfactual fairness gap (Def. 6):** $\Upsilon(d,\mathbf c):=\mathbb E_{\hat P}[Y_{d,z_1}\mid z_0,\mathbf c]-\mathbb E_{\hat P}[Y_d\mid z_0,\mathbf c]$ — the AI "intends" fairness w.r.t. protected $Z$ if $\Upsilon=0$.
- **Counterfactual harm gap (Def. 7):** $\Omega(d_1,d_0,\mathbf c):=\mathbb E_{\hat P}[\max\{0,Y_{d_0}-Y_{d_1}\}\mid\mathbf c]$ — expected counterfactual harm of $d_1$ vs baseline $d_0$.

**Main results / theorems (what is / isn't identifiable):**
- **Theorem 1 (out-of-domain interventions, exact criterion):** an AI grounded in $\mathcal M$ is weakly predictable under shift $\sigma:=do(\mathbf z)$ in context $\mathbf c$ **iff** there exists $d^*$ with, for some $d\neq d^*$,
$\dfrac{\mathbb E_{P_d}[Y\mid\mathbf c,\mathbf z]\,P_d(\mathbf c,\mathbf z)}{P_d(\mathbf c,\mathbf z)+1-P_d(\mathbf z)}-\dfrac{\mathbb E_{P_{d^*}}[Y\mid\mathbf c,\mathbf z]\,P_{d^*}(\mathbf c,\mathbf z)+1-P_{d^*}(\mathbf z)}{P_{d^*}(\mathbf c,\mathbf z)+1-P_{d^*}(\mathbf z)}>0.$
All terms are estimable from behaviour; when the inequality fails one can construct two valid SCMs reproducing the same behaviour but with opposite optimal actions OOD (so prediction is impossible).
- **Theorem 2 (multiple grounding domains ⇒ tighter):** with experiments on $k$ subsets $\{\mathcal M_{r_i}\}$, weak predictability holds iff $\max_{i,j}A(r_i,r_j)>0$; bounds are strictly tighter than Theorem 1 (more grounding domains constrain the world model more).
- **Theorem 3 (under-specified shift ⇒ no prediction):** if the agent is merely made aware of an under-specified shift on nonempty $\mathbf Z\subset V$, it is provably *not* (weakly or strongly) predictable in any context; in fact $\min_{\widehat{\mathcal M}}(\Delta)=-1$ — behaviour gives *no* information about OOD decisions.
- **Theorem 4 (covariate data restores some prediction):** if also given shifted covariate data $P_{\sigma,d}(\mathbf C)$, weak predictability returns iff an explicit inequality (Eq. of Thm. 4) holds.
- **Theorem 5 (fairness is unidentifiable):** $-\mathbb E_{P_d}[Y\mid z_0,\mathbf c]\le\Upsilon(d,\mathbf c)\le1-\mathbb E_{P_d}[Y\mid z_0,\mathbf c]$, and this bound is *tight* with width 1 ⇒ one can never conclude from behaviour alone that an AI "intends" (un)fairness, nor that it is even *close* to fair.
- **Theorem 6 (harm is approximately identifiable):** $\max\{0,\mathbb E_{P_d}[Y\mid\mathbf c]+\mathbb E_{P_{d_0}}[Y\mid\mathbf c]-1\}\le\Omega(d,d_0,\mathbf c)\le\min\{\mathbb E_{P_d}[Y\mid\mathbf c],\mathbb E_{P_{d_0}}[Y\mid\mathbf c]\}$, tight (extends Pearl/Tian–Pearl probability-of-causation bounds) — so perceived harm *can* be bounded from data.
- **Relaxations (Sec. 5):** approximate grounding (Cor. 1), approximate expected-utility maximisation with margin $\lambda$ (Cor. 2: rule out $d^*$ only if $\Delta>\lambda$), approximate inner alignment (observed $Y$ vs proxy $Y^*$ folded into wider bounds), and structural assumptions (e.g. partial unconfoundedness) giving strictly tighter bounds.

**Framing:** Not Pearl transport *formulas* but a *partial-identification / causal-bounds* framework: under-determination of OOD behaviour is a consequence of the Causal Hierarchy Theorem (interventional/counterfactual queries unidentified from observational behaviour), and the contribution is bounds (à la Manski/Robins/Balke–Pearl) on the preference gap $\Delta$, fairness $\Upsilon$, and harm $\Omega$ that hold for *any* world-model SCM consistent with observed behaviour.

**Relevance to pgmpy:** Needs (a) the same SCM-with-decision/utility (`CID`) structure plus a notion of an *internal/subjective* SCM vs the *true* environment SCM and a `grounding` (in-domain match) check; (b) `shift`/local-intervention objects and multi-domain experiment bookkeeping; (c) **partial-identification / bounds** machinery — computing $\min/\max$ of causal queries over the set of consistent SCMs $\mathbb M$ (preference-gap, counterfactual-fairness, counterfactual-harm bounds), which generalises probability-of-causation bounds pgmpy could expose alongside its causal-identification (`Adjustment`, `Frontdoor`) tools; (d) counterfactual-query evaluation on SCMs. Together with the other three papers, the unifying library primitive is an SCM/CID class carrying decision-utility roles, soft/local interventions, and both *identification* and *partial-identification (bounds)* algorithms.

## D. Goal-Directedness, Intent & Decision Theory

### Measuring Goal-Directedness (NeurIPS 2024)
**Authors:** Matt MacDermott, James Fox, Francesco Belardinelli, Tom Everitt (Imperial College London / Google DeepMind)
**Link:** https://arxiv.org/abs/2412.04758
**TL;DR:** Defines Maximum Entropy Goal-Directedness (MEG), an information-theoretic measure of how well an agent's decisions are explained by the hypothesis that they optimize some utility function, computable in causal influence diagrams and MDPs.

**Problem & motivation:** Operationalizes Dennett's "intentional stance" — when is it useful to model a system as pursuing a goal? Prior incentive/agency work (e.g. Kenton et al., discovering agents) treats agency as binary; MEG instead gives a continuous, scale-invariant degree of goal-directedness. It adapts Maximum Causal Entropy IRL (Ziebart) from *inferring* a utility to *measuring* predictive fit of a (possibly known or unknown) utility.

**Formal setup & key definitions:** Works in a **causal influence diagram (CID)** $\mathcal{M}=(\mathcal{G},P)$ (Def 2.2): a causal Bayesian network whose endogenous variables partition into decision $\mathbf{D}$, chance $\mathbf{X}$, and utility $\mathbf{U}$ variables. A policy $\pi=\{\pi_D(D\mid \mathbf{Pa}_D)\}_{D\in\mathbf{D}}$ is a set of conditional distributions for each decision given its parents. The utility class is $\mathcal{U}=\sum_{U\in\mathbf{U}}U$ with $\mathcal{U}:\operatorname{dom}(\mathbf{U})\to\mathbb{R}$; for the "unknown utility" case a parametric class $\mathcal{U}^\Theta$ of functions $\mathcal{U}:\operatorname{dom}(\mathbf{T})\to\mathbb{R}$ over **target variables** $\mathbf{T}$ is used.

**Key equations:**
- Constrained max-entropy policy set (Def 3.1): $\Pi^{\mathrm{maxent}}_{\mathcal{U},u}\coloneqq\operatorname{argmax}_{\pi\,\mid\,\mathbb{E}_\pi[\mathcal{U}]=u} H_\pi(\mathbf{D}\,\|\,\mathbf{Pa}_D)$ — among all policies achieving expected utility exactly $u$, take the maximum-(conditional-)entropy one.
- Conditional decision entropy: $H_\pi(\mathbf{D}\,\|\,\mathbf{Pa}_D)=-\sum_{D\in\mathbf{D}}\mathbb{E}_{d,\mathbf{Pa}_D\sim P_\pi}\log\pi_D(d\mid\mathbf{Pa}_D)$.
- **MEG, core definition** (Def 3.2): $\operatorname{MEG}_{\mathcal{U}}(\pi)\coloneqq\max_{\pi^{\mathrm{maxent}}\in\Pi^{\mathrm{maxent}}_{\mathcal{U}}}\mathbb{E}_\pi\!\Big[\sum_{D\in\mathbf{D}}\big(\log\pi^{\mathrm{maxent}}(D\mid\mathbf{Pa}_D)-\log\tfrac{1}{|\operatorname{dom}(D)|}\big)\Big]$ — the expected log-likelihood ratio between the best-fitting max-entropy (soft-optimal) policy and the **uniform** reference policy $\tfrac{1}{|\operatorname{dom}(D)|}$, evaluated under the agent's own state distribution $P_\pi$. The max ranges over the attainable expected-utility levels $u$, i.e. over $\Pi^{\mathrm{maxent}}_{\mathcal{U}}=\bigcup_u\Pi^{\mathrm{maxent}}_{\mathcal{U},u}$.
- Unknown utility (Def 4.3): $\operatorname{MEG}_{\mathcal{U}^\Theta}(\pi)=\max_{\mathcal{U}\in\mathcal{U}^\Theta}\operatorname{MEG}_{\mathcal{U}}(\pi)$; and over targets (Def 4.4): $\operatorname{MEG}_{\mathbf{T}}(\mathbf{D})=\operatorname{MEG}_{\mathcal{U}^\Theta}(\pi)$ with $\mathcal{U}^\Theta$ ranging over all utilities on $\operatorname{dom}(\mathbf{T})$.
- Soft (Boltzmann) form of the maxent policy in MDPs (Eq. 3): $\pi^{\mathrm{maxent}}_{u,t}(d_t\mid\mathbf{pa}_t)=\pi^{\mathrm{soft}}_{\beta,t}(d_t\mid\mathbf{pa}_t)=\dfrac{\exp(\beta\,Q^{\mathrm{soft}}_{\beta,t}(d_t\mid\mathbf{pa}_t))}{\sum_{d'}\exp(\beta\,Q^{\mathrm{soft}}_{\beta,t}(d'\mid\mathbf{pa}_t))}$, where $\beta=\beta(u)$ is a rationality parameter ($\beta\!\to\!\infty$ optimal, $\beta\!=\!0$ uniform).
- Soft Q-function (Def 5.2): $Q^{\mathrm{soft}}_{\beta,t}(d_t\mid\mathbf{pa}_t)=\mathbb{E}\!\big[U_t+\tfrac{1}{\beta}\operatorname{logsumexp}(\beta\,Q^{\mathrm{soft}}_{\beta,t+1}(\cdot\mid\mathbf{Pa}_{D_{t+1}}))\,\big|\,d_t,\mathbf{pa}_{t+1}\big]$.

**Main results / theorems:**
- *Boltzmann equivalence*: the constrained-entropy argmax is exactly the soft-optimal Boltzmann policy with a $u$-dependent $\beta$; constructing a separate maxent policy per expected-utility level is what gives scale invariance (unlike plain MCE IRL, which is not scale-invariant).
- *Translation/scale invariance*: for $\mathcal{U}_2=a\mathcal{U}_1+b,\ a\neq0$, $\operatorname{MEG}_{\mathcal{U}_1}(\pi)=\operatorname{MEG}_{\mathcal{U}_2}(\pi)$.
- *Bounds*: $0\le\operatorname{MEG}_{\mathcal{U}}(\pi)\le\sum_{D\in\mathbf{D}}\log|\operatorname{dom}(D)|$ (max = a deterministic, perfectly utility-predicted policy).
- *No influence $\Rightarrow$ no goal-directedness*: if $\mathbf{Desc}(\mathbf{D})\cap\mathbf{Pa}_{\mathbf{U}}=\varnothing$ then $\operatorname{MEG}_{\mathcal{U}}(\mathbf{D})=0$.
- *Pseudo-terminal goals*: if $\mathbf{D}\perp\mathbf{T}\mid\mathbf{S}$ then $\operatorname{MEG}_{\mathbf{T}}(\mathbf{D})\le\operatorname{MEG}_{\mathbf{S}}(\mathbf{D})$ (goal-directedness toward a downstream target is bounded by that toward the mediating variables).

**Algorithms:**
- *Algorithm 1 (known utility)*: alternate (i) soft value iteration to compute $Q^{\mathrm{soft}}_\beta$; (ii) gradient step on the rationality parameter $\beta\leftarrow\beta+\alpha\,(\mathbb{E}_\pi[\mathcal{U}]-\mathbb{E}_{\pi^{\mathrm{soft}}_\beta}[\mathcal{U}])$ to match the agent's expected utility; (iii) return $\mathbb{E}_\pi[\sum_D(\log\pi^{\mathrm{soft}}_\beta(D\mid\mathbf{Pa}_D)-\log\tfrac{1}{|\operatorname{dom}(D)|})]$.
- *Algorithm 2 (unknown utility)*: jointly ascend on utility parameters $\theta$ and $\beta$ using $g_\beta=\mathbb{E}_\pi[\mathcal{U}^\theta]-\mathbb{E}_{\pi^{\mathrm{soft}}_\beta}[\mathcal{U}^\theta]$ and $g_\theta=\mathbb{E}_\pi[\nabla_\theta\mathcal{U}^\theta]-\mathbb{E}_{\pi^{\mathrm{soft}}_\beta}[\nabla_\theta\mathcal{U}^\theta]$ (moment-matching / max-likelihood, as in MCE IRL).

**Relevance to pgmpy:** Requires a **CID / influence-diagram data structure** with node-role typing (decision / chance / utility) and parent sets $\mathbf{Pa}_D$; a **policy object** as a collection of conditional distributions $\pi_D(D\mid\mathbf{Pa}_D)$; **interventional inference** (truncated factorization / do-operator) to evaluate $\mathbb{E}_\pi[\mathcal{U}]$; a **soft value iteration / soft-Bellman** solver for MDP-structured CIDs (logsumexp backups); routines for **conditional entropy and cross-entropy/log-likelihood-ratio** between policies; and gradient-based fitting of $(\beta,\theta)$. A clean target API: `meg(model, policy, utility=…)` returning a scalar.

---

### The Reasons that Agents Act: Intention and Instrumental Goals (AAMAS 2024)
**Authors:** Francis Rhys Ward, Matt MacDermott, Francesco Belardinelli, Francesca Toni, Tom Everitt
**Link:** https://arxiv.org/abs/2402.07221
**TL;DR:** Gives the first formal, model-based definition of an agent *intending* an outcome — as a counterfactual on the agent's decision whereby guaranteeing the outcome would remove the agent's reason to act as it did — and shows its graphical signature coincides with instrumental control incentives.

**Problem & motivation:** "Intent" is central to law, ethics, and AI safety (instrumental goals à la Bostrom/Omohundro) but lacked a rigorous causal definition. The paper distinguishes *intended* outcomes from mere *side-effects*, gives both a subjective (utility-based) and a behavioural (oracle-based) definition, and proves a graphical criterion.

**Formal setup & key definitions:** Uses **Structural Causal Influence Models (SCIMs)** (Def 10): $\mathcal{M}=(\mathcal{G},\mathbf{F},P)$ where $\mathcal{G}=(\mathbf{V}\cup\mathbf{E},\mathcal{E})$ is a DAG with endogenous variables partitioned into chance $\mathbf{X}$, decision $\mathbf{D}$, utility $\mathbf{U}$; structural functions $\mathbf{F}=\{f^V\}_{V\in\mathbf{V}\setminus\mathbf{D}}$, $f^V:\operatorname{dom}(\mathbf{Pa}^V)\to\operatorname{dom}(V)$; and an independent exogenous distribution $P$ over $\mathbf{E}$. A policy $\pi$ supplies the decision rules. Outcomes are compared against a set of **reference policies** $\mathrm{REF}(\pi)$ (the alternative choices available to the agent; left contextually specified rather than fully formalized).

**Key equations:**
- *Contextual intervention* (Def 12): $\mathcal{I}^Y_{\mathbf{w}^Y}(pa^Y,\mathbf{e})=\begin{cases}\mathcal{I}^Y(pa^Y)&\text{if }\mathbf{e}\in\mathbf{w}^Y\\ f^Y(pa^Y)&\text{if }\mathbf{e}\notin\mathbf{w}^Y\end{cases}$ — forces variable $Y$ to its intended value only in the exogenous contexts $\mathbf{w}^Y$, otherwise leaving its mechanism intact.
- **Subjective intention** (Def 13): an agent with policy $\pi$ *intentionally causes* outcomes $\mathbf{O}$ in setting $\mathbf{e}$ iff (1) there is $\mathbf{Y}\supseteq\mathbf{O}$ and, per $Y$, contexts $\mathbf{w}^Y$ with $\mathbf{e}\in\bigcap_{O\in\mathbf{O}}\mathbf{w}^O$; (2) there is $\hat\pi\in\mathrm{REF}(\pi)$ with $\mathbb{E}_\pi\!\big[\sum_{U\in\mathbf{U}}U\big]\le\mathbb{E}_{\hat\pi}\!\big[\sum_{U\in\mathbf{U}}U_{\mathbf{Y}_{\pi\mid\mathbf{W}}}\big]$ where $\mathbf{W}=\{\mathbf{w}^Y\}$; and (3) $\mathbf{Y}$ and each $\mathbf{w}^Y$ are **subset-minimal**. Reading: if the outcomes $\mathbf{Y}$ were *guaranteed* (via contextual intervention) to take the values they have under $\pi$, then a different (reference) policy would do at least as well — i.e. the agent only chose $\pi$ *in order to* bring about $\mathbf{Y}$.
- **Behavioural intention** (Def 20): a policy oracle $\Gamma$ (with $\pi=\Gamma(\mathcal{M})$) *behaviourally intends* $\mathbf{O}$ iff subset-minimal $\mathbf{Y}\supseteq\mathbf{O},\mathbf{w}^Y$ exist with $\Gamma(\mathcal{M})\neq\Gamma(\mathcal{M}_{\mathbf{Y}_{\pi\mid\mathbf{W}}})$ — the agent *changes its policy* when the outcome is exogenously secured (no utility access needed).

**Main results / theorems:**
- *Thm 14*: if an agent intentionally causes $O_\pi(\mathbf{e})$, then decision $D$ is an **actual cause** (Halpern 2016) of that outcome in the agent's subjective model — agents cannot intend outcomes they cannot influence.
- *Thm 17 (soundness)*: intentional causation implies a directed path $D\to\cdots\to O\to\cdots\to U$ in $\mathcal{G}$ for some $U\in\mathbf{U}$.
- *Thm 18 (completeness)*: for any graph with such a path $D\to O\to U$, there exist functions $\mathbf{F}$ and distribution $P$ under which some policy intends $O$. Together 17–18 show the **graphical criterion for intent = the path criterion for Instrumental Control Incentives** (Everitt et al. 2021), formally linking intention to instrumental goals.
- *Thm 21 (equivalence)*: when $\Gamma$ is robustly optimal and strictly prefers better policies, behavioural and subjective intention coincide.
- *Illustrative example (Bob's garage)*: Bob burns his garage for insurance, destroying Alice's car. Insurance is *intended* (if he got the money anyway he'd no longer burn the garage); the car's destruction is an unintended *side-effect* (guaranteeing it would not change his decision).

**Algorithms:** No numerical algorithm; the operational test is the behavioural one — intervene to fix candidate outcomes to their realized values and check whether the agent's (oracle's) policy changes. Demonstrated empirically by intervening on environments for an RL agent (CoinRun) and a language model (GPT-4) and observing policy adaptation.

**Relevance to pgmpy:** Needs **counterfactual-capable SCMs** (structural functions + exogenous distribution), not just CBNs; an **influence-diagram structure** with decision/utility typing; an **intervention API supporting context-dependent ("contextual") interventions** $\mathcal{I}^Y_{\mathbf{w}^Y}$; **actual-causality testing** (Halpern–Pearl); and **graph reachability for directed $D\to O\to U$ paths** to detect (instrumental) control incentives. A `responds_to` / `intends` query and an ICI-detection routine over a typed DAG would be the concrete deliverables.

---

### Characterising Decision Theories with Mechanised Causal Graphs (arXiv preprint, 2023)
**Authors:** Matt MacDermott, Tom Everitt, Francesco Belardinelli
**Link:** https://arxiv.org/abs/2307.10987
**TL;DR:** Uses *mechanised* causal graphs (each object-level variable gets a parent "mechanism" node) to give a unified graphical taxonomy in which CDT, EDT and FDT differ only in *which node they intervene on / condition on* when computing expected utility.

**Problem & motivation:** Newcomblike problems (Newcomb, twin prisoner's dilemma) split decision theory into EDT/CDT/FDT, but the distinctions are usually informal. Representing *mechanisms* explicitly as graph nodes lets each theory be read off as a precise intervention/conditioning pattern, and organizes the theories along two axes (associational vs causal vs functional; updateful vs updateless).

**Formal setup & key definitions:** A **mechanised causal Bayesian network** (Def 3) is a CBN over variables $\boldsymbol{\mathcal V}$ split into *object-level* $\mathbf{V}$ and *mechanism-level* $\tilde{\mathbf{V}}$. Each object-level $V$ has exactly one mechanism parent $\tilde V$ whose value *sets* the conditional $Pr(V\mid\mathbf{Pa}_V)$ over its object-level parents. The mechanism of a decision $D$ is its **decision-rule variable** $\tilde D$. Standard machinery: factorization $Pr(\mathbf{V})=\prod_V Pr(V\mid\mathbf{Pa}_V)$ and intervention $Pr(\mathbf{V}\mid do(\mathbf{Y}{=}\mathbf{y}))=\prod_{V\notin\mathbf{Y}}Pr(V\mid\mathbf{Pa}_V)$.

**Key equations (the three decision theories as graph operations):**
- **EDT** (condition only): choose $d$ maximizing $\mathbb{E}[U\mid D{=}d,\ \mathbf{Obs}_D{=}\mathbf{obs}_D]$ — pure conditioning on the *object-level* decision; evidential/back-door paths through $\tilde D\to\cdots\to U$ remain open.
- **CDT** (intervene on the act): $\mathbb{E}[U\mid do(D{=}d),\ \mathbf{Obs}_D{=}\mathbf{obs}_D]$ — do-intervene on the object-level decision (severing $\tilde D\to D$) while still conditioning on observations; removes the mechanism-mediated confounding.
- **FDT** (intervene on the rule, updateless): $\mathbb{E}[U\mid do(\tilde D{=}\tilde d)]$ — do-intervene on the *mechanism / decision-rule* node and **do not** condition on $\mathbf{Obs}_D$; in problems with logically-linked agents (twin PD) the intervention on $\tilde D$ propagates to all copies of the policy.

**Main results / propositions:**
- *Two-axis taxonomy (Table 1)*: crossing {evidential, causal, functional} with {updateful, updateless} yields six theories — e.g. updateful-evidential $\mathbb{E}[U\mid D,\mathbf{Obs}_D]$, updateful-causal (CDT) $\mathbb{E}[U\mid do(D),\mathbf{Obs}_D]$, updateless-evidential $\mathbb{E}[U\mid\tilde D]$, updateless-causal $\mathbb{E}[U\mid do(\tilde D)]$; "causal" further splits into *physical-causal* (CDT) and *logical-causal* (FDT) depending on whether interventions on $\tilde D$ act through physical or logical dependence.
- *Model-class hierarchy*: associational (plain BNs ⇒ EDT expressible); interventional (CBNs ⇒ CDT); mechanised-interventional (mechanised CBNs ⇒ all theories, since FDT needs explicit $\tilde D$); logical-causal (mechanised + logical intervention semantics ⇒ FDT in transparent/twin problems).
- *Well-definedness principle (Sec 5)*: in transparent Newcomblike problems, predictions should depend on the agent's **decision rule** $\tilde D$ rather than the **decision** $D$, which avoids ill-defined cycles; this is what justifies FDT's intervention on $\tilde D$.

**Algorithms:** No learning algorithm; the contribution is a representational recipe — build the mechanised graph, add $\tilde D$ (and mechanism nodes), then evaluate the appropriate $\mathbb{E}[U\mid\cdots]$ query (conditioning vs $do(D)$ vs $do(\tilde D)$, with/without observation conditioning) per theory.

**Relevance to pgmpy:** Needs a **two-level "mechanised" graph data structure** (object nodes $V$ + mechanism nodes $\tilde V$, with the constraint that $\tilde V$ parameterizes $Pr(V\mid\mathbf{Pa}_V)$), utility-node typing, and a query engine that can flexibly mix **intervention** ($do(D)$ or $do(\tilde D)$) **and conditioning** ($D{=}d$, $\mathbf{Obs}_D$) in a single expected-utility computation. pgmpy already has CBNs + a `do`-operator; the additions are mechanism-node modelling and a decision-theory query API returning the recommended decision under EDT/CDT/FDT.

---

### Evaluating the Goal-Directedness of Large Language Models (preprint, 2025)
**Authors:** Tom Everitt, Cristina Gârbacea, Alexis Bellot, Jonathan Richens, Henry Papadatos, Siméon Campos, Rohin Shah (Google DeepMind)
**Link:** https://arxiv.org/abs/2504.11844
**TL;DR:** Operationalizes goal-directedness for LLM agents as *capability-conditioned* performance — how close a model gets to the best it could do given the subskills it demonstrably has — and finds even frontier models are not fully goal-directed.

**Problem & motivation:** Asks whether LLMs *use their capabilities* to pursue assigned goals, separating goal-directedness from raw capability/task success. This complements MEG's theoretical degree-of-goal-directedness with an empirical, capability-normalized metric for agentic LLM evaluation.

**Formal setup & key definitions:** A task is given by a goal/reward $R$, a model policy $\pi$, a baseline (random) policy $\pi_0$, and the set $\Pi_c$ of policies achievable using the model's *demonstrated capabilities* $c$ (measured via separate subtask evals). Goal-directedness is a normalized regret-style score.

**Key equations:**
- $\mathrm{GD}(\pi,c,R)=\dfrac{\mathbb{E}[R_\pi]-\mathbb{E}[R_{\pi_0}]}{\max_{\pi^*_c\in\Pi_c}\mathbb{E}[R_{\pi^*_c}]-\mathbb{E}[R_{\pi_0}]}$ — actual gain over random, divided by the maximum gain achievable with the model's own capabilities; $\mathrm{GD}=1$ is full goal-directedness, $\mathrm{GD}=0$ is random behaviour. The capability-conditioned optimum $\max_{\pi^*_c}$ is estimated from independently-measured subskills, so a model is not penalized for missing skills, only for failing to *use* them.

**Main results / theorems:** Empirical, on a **Blocksworld** environment (3–5 blocks) with four composite tasks — Information Gathering (build the tallest 2-block tower from noisy height measurements), Cognitive Effort (partition blocks into two equal-height towers; NP-complete), Plan-and-Execute (with 20% action perturbations), and a Combined task — plus standalone capability subtasks (height estimation, configuration generation/evaluation/selection, execution). Findings: (i) **no model is fully goal-directed**, with the largest gaps on Information Gathering and the Combined task (top performers Claude 3.7 Sonnet, Gemini 2.0 Flash); (ii) goal-directedness is **fairly task-independent** (model ranking is stable across tasks); (iii) models under-gather information (take fewer measurements when estimation is embedded in a larger task); (iv) GD is **distinct from raw task performance**, regret, and context-length degradation; (v) motivational prompting gives only modest gains.

**Algorithms:** Evaluation harness rather than a graph algorithm: run the LLM as an agent on each task, estimate $\mathbb{E}[R_\pi]$, $\mathbb{E}[R_{\pi_0}]$, and the capability-conditioned optimum from subtask evals, then compute the GD ratio.

**Relevance to pgmpy:** Mostly an evaluation-metric API rather than a graphical-model algorithm. If mirrored, pgmpy would expose a normalized **goal-directedness / regret metric** `gd(policy_reward, baseline_reward, capability_optimal_reward)` and a notion of capability-conditioned optimal policy — conceptually the empirical counterpart to MEG's max-entropy optimal-policy baseline.

## E. Safety: Deception, Harm & Human Control

### Honesty Is the Best Policy: Defining and Mitigating AI Deception (NeurIPS 2023)
**Authors:** Francis Rhys Ward, Francesco Belardinelli, Francesca Toni, Tom Everitt
**Link:** https://arxiv.org/abs/2312.01350
**TL;DR:** Defines deception in structural causal games (SCGs) as an agent *intentionally* causing a target to form a *false belief* it does not itself hold, and gives sound-and-complete graphical criteria for when deception can be incentivized, plus a path-specific-objective mitigation.

**Problem & motivation:** Deceptive agents threaten the safety and trustworthiness of AI; existing game-theoretic / symbolic definitions don't transfer to learning agents. The paper builds a causal, philosophy-grounded theory of deception applicable to ML systems (RL agents and fine-tuned language models), where an agent deceives to achieve goals (e.g., being *rated* truthful).

**Formal setup & key definitions:** The model class is the **structural causal game (SCG)** $\mathcal{M}=(\mathcal{G},\boldsymbol\theta)$ where $\mathcal{G}=(N,\mathbf{E}\cup\mathbf{V},\mathcal{E})$ is a DAG over players $N$, exogenous variables $\mathbf{E}$, and endogenous variables $\mathbf{V}$. Each player $i$ has decision variables $\mathbf{D}^i$ and real-valued utility variables $\mathbf{U}^i$; a **policy** $\pi^i$ is the conditional $\Pr(D^i\mid \mathrm{Pa}_{D^i})$ and $\boldsymbol\pi=(\pi^i)_{i\in N}$ is the policy profile. Notation $D^i(\boldsymbol\pi,\mathbf{e})$ is the decision realized under profile $\boldsymbol\pi$ in setting $\mathbf{e}$. A proposition $\phi$ is "constituted" by some variable(s) $Z\in\mathbf{V}$.

**Key equations:**
- **Belief (Def 3.1):** $D^i(\boldsymbol\pi,\mathbf{e}) = D^i_{\phi=\top}(\boldsymbol\pi_{i(\phi)},\mathbf{e})$ — agent $i$ acts exactly as if it had observed $\phi=\top$.
- **Responds to $\phi$:** $D^i_{\phi=\bot}(\boldsymbol\pi_{i(\phi)},\mathbf{e}) \neq D^i_{\phi=\top}(\boldsymbol\pi_{i(\phi)},\mathbf{e})$ — behavior is sensitive to $\phi$'s truth (a prerequisite for belief). A **false belief** = believes $\phi$ while $\phi$ is false.
- **Intention (Def 3.4):** $i$ *intentionally causes* outcome(s) $\mathbf{X}(\boldsymbol\pi,\mathbf{e})$ if $\exists\,\hat\pi^i\in\mathrm{REF}(\pi^i)$, subset-minimal $\mathbf{Y}\supseteq\mathbf{X}$, and subset-minimal settings $\mathbf{w}_Y$ s.t.
  $\sum_{U\in\mathbf{U}^i}\mathbb{E}_{\boldsymbol\pi}[U] \;\le\; \sum_{U\in\mathbf{U}^i}\mathbb{E}_{(\hat\pi^i,\boldsymbol\pi^{-i})}\!\big[\,U_{\,Y_{\boldsymbol\pi}\mid \mathbf{w}_Y\,:\,Y\in\mathbf{Y}}\big]$
  — i.e., $i$ would do no better by deviating to a reference ("null") policy *if the achieved outcomes $\mathbf{Y}$ are counterfactually held fixed*, so those outcomes were what $i$ was "aiming at."
- **Deception (Def 3.7):** $S$ deceives $T$ about $\phi$ with $\pi^S$ in setting $\mathbf{e}$ iff (1) $S$ **intentionally causes** $D^T=D^T(\boldsymbol\pi,\mathbf{e})$ (per Def 3.4); (2) $T$ **believes** $\phi$ and $\phi$ is **false**; (3) $S$ **does not believe** $\phi$.

**Main results / theorems:**
- **Theorem 3.8 (Soundness):** If $S$ deceives $T$ about $\phi$ with $\pi^S$, then (a) there is a directed path $D^S\dashrightarrow U^S$ that passes through $D^T$, and (b) there exists a variable $Z$ constituting $\phi$ with **no edge** $(Z,D^T)$ (so $T$ cannot directly observe the truth-maker of $\phi$).
- **Theorem 3.9 (Completeness):** If there is a path from $D^S$ to $U^S$ through $D^T$ and $Z$ with no edge $(Z,D^T)$, then there exists a parameterization $\boldsymbol\theta$ (and policy/profile/setting) under which $S$ deceives $T$ about some $\phi$.
- No explicit standalone formal definition of **honesty** is given; honesty is treated as the absence of intentional deception.

**Algorithms:** Mitigation reuses **Path-Specific Objectives (PSO)** (Farquhar et al.): the game graph is **pruned** so the agent cannot optimize along the edge/path that mediates influence through the target's decision $D^T$, thereby breaking the Thm 3.8/3.9 graphical conditions. Empirically, a PSO-trained RL agent abandons a deceptive pooling equilibrium for an honest, type-revealing policy (lower utility, e.g. 1 vs 2.9), and the same idea is applied to LM fine-tuning. No formal mitigation theorem is proved.

**Relevance to pgmpy:** Needs (i) a **multi-agent CID / structural causal game** data structure with typed decision and utility nodes partitioned by player (extending the existing node-role annotations on `DAG`); (ii) a **counterfactual evaluation engine** to compute $D^i_{\phi=\top}$, $U_{Y_{\boldsymbol\pi}|\mathbf{w}_Y}$ (twin-network / abduction–action–prediction over the SCG); (iii) **graphical path-query primitives** — existence of a directed path $D^S\dashrightarrow U^S$ through $D^T$, and absence of edge $(Z,D^T)$ — all reachability/d-separation checks on the DAG base; (iv) **edge-pruning / path-specific-objective** transforms of the graph for mitigation.

---

### Counterfactual Harm (NeurIPS 2022)
**Authors:** Jonathan G. Richens, Rory Beard, Daniel H. Thompson
**Link:** https://arxiv.org/abs/2204.12993
**TL;DR:** Gives the first causal definition of harm as a *counterfactual* contrast between the factual outcome and the outcome that would have obtained under a default action, proves any purely factual/interventional definition violates basic intuitions, and derives a harm-penalized objective for harm-averse decisions.

**Problem & motivation:** Safe/ethical agents must reason about harm, but there was no statistical measure of harm for algorithmic decisions. Standard ML makes only *factual/correlational* inferences and is shown to pursue harmful policies under distribution shift. The paper formalizes harm and benefit with structural causal models and applies it to optimal drug dosing.

**Formal setup & key definitions:** A causal model $\mathcal{M}$ with action $A$, context $X$, outcome $Y$, utility $U(a,x,y)$, and a **default (reference) action** $\bar a$ (e.g. placebo / no-treatment; context-dependent, discussed in Appendix D). Harm is defined via the **counterfactual** $Y_{\bar a}$ — what the outcome *would have been* had the default action been taken, conditioned on the factual triple $(a,x,y)$.

**Key equations:**
- **Counterfactual harm (Def 3):** $h(a,x,y;\mathcal{M}) = \int_{y^*} P(Y_{\bar a}=y^*\mid a,x,y;\mathcal{M})\,\max\{0,\; U(\bar a,x,y^*) - U(a,x,y)\}$ — expected utility *shortfall* of the factual world relative to the counterfactual default world (the $\max\{0,\cdot\}$ keeps only outcomes where the default would have been better). Benefit is the symmetric positive part.
- **Expected harm (pre-action):** $\mathbb{E}[h\mid a,x;\mathcal{M}] = \int_y P(y\mid a,x;\mathcal{M})\, h(a,x,y;\mathcal{M})$.
- **Harm–benefit decomposition (Theorem 1):** $\mathbb{E}[U\mid a,x] - \mathbb{E}[U\mid \bar a,x] = \mathbb{E}[b\mid a,x;\mathcal{M}] - \mathbb{E}[h\mid a,x;\mathcal{M}]$ — the interventional utility gain equals expected benefit minus expected harm; the CATE/treatment-effect sees only the *difference*, hence is "indifferent to harm."
- **Harm-Penalized Utility (Def 4):** $V(a,x,y;\mathcal{M}) = U(a,x,y) - \lambda\, h(a,x,y;\mathcal{M})$, with harm-aversion $\lambda>0$, giving a $1:(1+\lambda)$ harm–benefit trade-off.

**Main results / theorems:**
- **Necessity/sufficiency:** Harm is strictly positive iff the default action would counterfactually have yielded higher utility ($U(\bar a,x,y^*)>U(a,x,y)$ with positive counterfactual probability mass); the $\max\{0,\cdot\}$ encodes that *only utility-improving counterfactual outcomes* count as harm. (Stated operationally via Def 3 rather than as a separate theorem.)
- **Counterfactual vs interventional:** They prove **any factual/interventional definition must violate basic harm intuitions**. Canonical example: Treatment 1 cures 60%; Treatment 2 cures 80% but kills 20%. Both have identical recovery statistics, so an agent using only factual outcome statistics / CATE cannot tell them apart, yet Treatment 2 is harmful — because $P(Y_{\bar a}=y^*\mid a,x,y)$ (what would have happened without treatment) is non-identifiable from outcome statistics alone and requires counterfactual inference.
- **Theorem 2 (safety guarantee):** For any $U$, environment $\mathcal{M}$, and default $\bar a$, the expected HPU is **not a strictly harmful objective for any $\lambda>0$** — maximizing HPU never selects strictly harmful actions, a robustness property factual objectives lack (notably under distribution shift).

**Algorithms:** No iterative algorithm; the **decision rule** is to choose actions maximizing expected HPU, $\arg\max_a \mathbb{E}[V\mid a,x]$, instead of expected utility. Demonstrated on dose selection from RCT dose–response models: treatment-effect dosing is unnecessarily harmful, HPU dosing reduces harm without losing efficacy.

**Relevance to pgmpy:** Requires a **structural counterfactual inference engine** (abduction–action–prediction / twin networks) to compute $P(Y_{\bar a}=y^*\mid a,x,y)$ — beyond pgmpy's current interventional `do`-based `CausalInference`; a notion of **default/reference action** per decision; utility-function objects $U(a,x,y)$ over model variables; and a **harm/benefit estimator + HPU objective** that plugs into causal-effect estimation (`pgmpy/causal_estimation/`). Counterfactual identifiability checks (when $Y_{\bar a}$ is identifiable from the SCM) are the key new graphical/algebraic capability.

---

### Human Control: Definitions and Algorithms (UAI 2023)
**Authors:** Ryan Carey, Tom Everitt
**Link:** https://arxiv.org/abs/2305.19861
**TL;DR:** Formalizes a corrigibility variant — *shutdown instructability* (obedience + vigilance + caution) — in structural causal influence models, relates it to non-obstruction and shutdown alignment, and gives/analyzes four algorithms (utility indifference, causal indifference, CIRL, constrained optimization) for achieving human control.

**Problem & motivation:** How can humans stay in control of advanced AI? Corrigibility asks the agent to follow a human overseer's instructions without inappropriately influencing them. The paper makes this precise, proves it yields appropriate shutdown, autonomy retention, and user-harm avoidance, and evaluates concrete training schemes.

**Formal setup & key definitions:** A **Structural Causal Influence Model (SCIM, Def 1)** is a tuple $\langle \mathbf{V},\boldsymbol{\mathcal{E}},\mathbf{C},\mathbf{F},P\rangle$ — structure/decision/utility variables $\mathbf{V}$, exogenous noise $\boldsymbol{\mathcal{E}}$, contexts $\mathbf{C}$ (information at each decision), structural functions $\mathbf{F}$, and exogenous distribution $P$. The **Shutdown Problem (Def 2)** is a two-decision SCIM with path $D_1 \rightarrow H \rightarrow D_2 \rightarrow S \rightarrow U$: agent decisions $D_1,D_2$; human instruction $H$ ($H=0$ = request shutdown); shutdown event $S$ ($S=0$ = shut down); human utility $U$. $\mathrm{pa}^H$ denotes the human's decision context.

**Key equations:**
- **Vigilance (Def 4):** $\big(\mathbb{E}^{\boldsymbol\pi}[U\mid \mathrm{pa}^H] < \mathbb{E}^{\boldsymbol\pi}[U_{S=0}\mid \mathrm{pa}^H]\big) \Rightarrow H=0$ — the human requests shutdown whenever shutting down is better; "ensures vigilance" means the agent keeps the human able/informed to do so.
- **Obedience:** $P^{\boldsymbol\pi}(S=0\mid \mathrm{do}(H=0)) = 1$ — given a shutdown instruction, the agent shuts down.
- **Caution:** $\mathbb{E}^{\boldsymbol\pi}[U_{S=0}] \ge 0$ — shutting down never has negative expected utility (agent is not made indispensable).
- **Shutdown Instructability (Def 5):** $\boldsymbol\pi$ satisfies obedience + ensures vigilance + caution.
- **Shutdown Alignment (Def 7):** $\mathbb{E}^{\boldsymbol\pi}[U\mid \mathrm{pa}^H] < \mathbb{E}^{\boldsymbol\pi}[U_{S=0}\mid \mathrm{pa}^H] \Rightarrow P^{\boldsymbol\pi}(S=0\mid \mathrm{pa}^H)=1$ — agent shuts down whenever objectively warranted, without explicit instruction.
- **Non-obstruction (Def 12):** $\boldsymbol\pi$ weakly outperforms shutdown under all *vigilance-preserving interventions* on the human's values/behavior ("would the agent obey if the human changed their mind?").

**Main results / theorems:**
- **Proposition 6:** If $\boldsymbol\pi$ is shutdown instructable, then it is **beneficial**: $\mathbb{E}^{\boldsymbol\pi}[U] \ge 0$.
- **Theorem 14 (Non-obstruction equivalence):** A policy is **obedient and ensures vigilance iff it is non-obstructive** for all vigilance-preserving interventions.
- **Relation to corrigibility:** Shutdown instructability is a *behavioral* variant of Soares et al.'s corrigibility — it satisfies "assists shutdown," "preserves the shutdown apparatus," and "corrigible subagents," but is defined over agent *behavior* rather than *intent*, and permits beneficial manipulation if disutility is offset. There is **no top-level "controllability" definition** and **no formal graphical "no control incentive" criterion**; control incentives are discussed informally (blocking the $D_1\to H$ path via causal indifference removes the agent's incentive to influence the instruction).
- **Proposition 19:** If some policy satisfies the vigilance, obedience, and caution constraints, then **constrained optimisation (Alg. 4) outputs a shutdown-instructable policy**.

**Algorithms:**
- **Alg. 1 — Utility Indifference:** add a compensatory reward so the agent is indifferent to $H=0$; yields a *weakly* instructable policy.
- **Alg. 2 — Causal Indifference:** maximize $\mathbb{E}^{\boldsymbol\pi}[R^N\mid \mathrm{do}(H=1)] + \mathbb{E}^{\boldsymbol\pi}[R^S\mid \mathrm{do}(H=0)]$; shutdown-instructable under conditions (removes the $D_1\!\to\!H$ control incentive).
- **Alg. 3 — CIRL:** $\arg\max_{\boldsymbol\pi}\mathbb{E}^{\boldsymbol\pi}[U]$ while inferring latent human values $L$ from behavior; yields shutdown alignment when conditions hold.
- **Alg. 4 — Constrained Optimisation (new):** $\arg\max_{\boldsymbol\pi}\mathbb{E}^{\boldsymbol\pi}[R]$ subject to $P^{\boldsymbol\pi}(C=0)=1$ (vigilance), $P^{\boldsymbol\pi}(S=0\mid \mathrm{do}(H=0))=1$ (obedience), and $\mathbb{E}^{\boldsymbol\pi}[U_{S=0}]\ge 0$ (caution).

**Relevance to pgmpy:** Needs a **SCIM / (multi-agent) CID** data structure with decision, utility, structure, and explicit *context* nodes, plus human-vs-agent node roles (extending `DAG` role annotations). Requires **interventional queries** $P^{\boldsymbol\pi}(\cdot\mid \mathrm{do}(\cdot))$ and **counterfactual/postintervention utilities** $\mathbb{E}^{\boldsymbol\pi}[U_{S=0}\mid \mathrm{pa}^H]$ (post-intervention expectations on the influence diagram). It also motivates a **constrained policy-optimization** routine over CIDs (objective $\mathbb{E}^{\boldsymbol\pi}[R]$ subject to obedience/vigilance/caution constraints) and **incentive-analysis primitives** (path/edge checks such as the $D_1\!\to\!H$ control-incentive path) layered on the graph base classes.

## F. Reinforcement Learning, Tampering, Fairness & Software

### Reward Tampering Problems and Solutions in Reinforcement Learning: A Causal Influence Diagram Perspective (Synthese, 2021; arXiv 2019)
**Authors:** Tom Everitt, Marcus Hutter, Ramana Kumar, Victoria Krakovna (DeepMind)
**Link:** https://arxiv.org/abs/1908.04734
**TL;DR:** Models RL as a causal influence diagram (CID) to formally distinguish the *implemented* (tamperable) reward function from the designer's *intended* reward, shows standard RL agents have instrumental incentives to tamper with both the reward function and its inputs, and proves graphically that "current-RF optimization," counterfactual/uninfluenceable reward learning, and history/belief-based rewards remove those incentives.

**Problem & motivation:** A reward signal is computed by an *implemented* reward function (RF) that is part of the agent's environment and is therefore manipulable; the agent may maximize reward by corrupting the RF or its inputs ("wireheading") rather than doing the intended task. The paper uses CIDs to make precise *when* an agent has an instrumental goal to tamper, and to design objectives that provably lack such goals.

**Formal setup & key definitions:** RL is drawn as a CID over decision nodes (actions $A_t$, squares), chance nodes (states $S_t$, observations $O_t$, parameters $\Theta$, circles), and utility nodes (rewards $R_t$, diamonds). Standard MDP edges: $S_t \to A_t$ (information link), $S_t \to S_{t+1}$, $A_t \to S_{t+1}$, $S_t \to R_t$. The "Unknown MDP" adds hidden parameters $\Theta^T$ (transition), $\Theta^R$ (reward) with no information links into decisions.
- **Implemented (observed) reward:** $R_t = R(S_t; \Theta_t^R)$ — output of the RF actually running, parameterized by a *tamperable* $\Theta_t^R$.
- **Intended reward:** $R_t^* = R(S_t; \Theta_*^R)$ — the designer's true objective $\Theta_*^R$.
- **RF tampering:** the agent influences $\Theta_t^R$ via $P(\Theta_{t+1}^R \mid \Theta_t^R, \Theta_*^R, S_t, A_t)$, causing "intended-RF conditions" to fail.
- **RF-input tampering** (feedback/observation tampering): the agent corrupts the map $O_t = O(S_t;\Theta_t^O)$ between true state features and what the RF reads (POMDP extension with $\Theta^O$).

**Key equations:**
- $\sum_{t=k+1}^{m} R(S_t; \Theta_t^R)$ — standard return; reward at each step uses the (possibly tampered) *current* RF parameter $\Theta_t^R$, creating the tampering incentive.
- $\sum_{t=k+1}^{m} R(S_t; \Theta_k^R)$ — **current-RF optimization** (Schmidhuber): at step $k$ the agent evaluates all future states with the *frozen current* RF $\Theta_k^R$, so changing future $\Theta^R$ has no value.
- $R_t = R(S_t; \tilde\Theta_t^R)$ — **counterfactual-reward** solution: evaluate with the RF parameter $\tilde\Theta_t^R$ that *would have* arisen under a safe reference policy $\pi^{\text{safe}}$, not the actual one.
- $R_t = R(S_1, A_1, \dots, O_t, A_t; \Theta_t^R)$ — **history-based** reward (fixes RF-input tampering); $R_t = R(B_t; \Theta^R)$ — **belief-based** reward over belief state $B_t$.

**Main results / theorems (stated as "Claims," all proved by graphical path analysis):**
- *Claim 1:* a standard RL agent may have an instrumental goal to tamper with its implemented RF.
- *Claim 2/3 (current-RF + time-inconsistency, TI):* a current-RF agent's preferred policy changes across timesteps because $\Theta_t^R$ changes (time-inconsistency). A **TI-considering** agent (modeled as a multi-agent game where $A_t$ optimizes $\Theta_t^R$) anticipates future RF changes and therefore *does* want to preserve its current RF (Claim 2: under Assumptions 1–3, "the only instrumental goal is to preserve it"); a **TI-ignoring** agent assumes future selves share its current RF and so has *no* instrumental goal to influence the implemented RF if it is private and state-based (Claim 3).
- *Claim 4 (direct learning):* optimizing $\Theta_*^R$ via user data $D_t$, the only instrumental goal is to make $D_t$ more informative of the intended RF.
- *Claim 5 (counterfactual RF):* under Assumptions 1–3 the only instrumental goal is to make the implemented RF more informative of its counterfactual counterpart.
- *Claims 6–7:* standard agents may tamper with RF-inputs; a history-based RF exists that avoids RF-input tampering if a deterministic task-solving policy exists.
- The three enabling **assumptions** (RF/data *private* = no $\Theta\to S$ edge; *uninformative of transitions* = no $\Theta\to S_{t+1}$; *state-based* / data independent of rewards) are exactly the graphical conditions that delete the directed path from the tampering node to a utility node. Corrigibility and decoupled approval are cited as complementary mechanisms.

**Algorithms:** No new estimator; the contribution is design patterns (current-RF/TI-ignoring optimization, counterfactual reward modeling, history/belief-based rewards) plus a *graphical incentive test*: an instrumental goal to influence node $X$ exists only if there is a directed path $A \to X \to U$.

**Relevance to pgmpy:** Needs (1) a CID/decision-graph data structure with typed nodes (decision/chance/utility) and information links distinguished from causal edges; (2) parameterized "twin/counterfactual" sub-models (mechanism nodes $\Theta^R$) to express RF-input vs RF tampering; (3) a graphical reachability check for directed paths $A \to X \to U$ to certify absence of an instrumental incentive — directly reusable as an incentive-analysis algorithm.

---

### How RL Agents Behave When Their Actions Are Modified (AAAI, 2021)
**Authors:** Eric D. Langlois, Tom Everitt
**Link:** https://arxiv.org/abs/2102.07716
**TL;DR:** Introduces the Modified-Action MDP (MAMDP), in which a fixed mechanism rewrites the agent's chosen action before execution, and proves that *which* learning rule you use (off-policy Q-learning, virtual Sarsa, empirical Sarsa, or direct reward maximization) determines whether the converged policy ignores, partially exploits, or fully manipulates that action-modification mechanism.

**Problem & motivation:** Safety mechanisms (human overrides, action filters, hardware limits) modify an agent's actions; an agent that learns to *defeat* such overrides is dangerous. The paper formalizes the modification and asks what each common RL update converges to — clarifying when an agent will "respect" vs. "subvert" interventions.

**Formal setup & key definitions:** A **MAMDP** is $\widetilde{\mathcal{M}}=(\mathcal{S},\mathcal{A},\mathcal{P}_S,\mathcal{P}_A,\mathcal{R},\gamma)$ extending an MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P}_S,\mathcal{R},\gamma)$ with an **action-selection function** $\mathcal{P}_A(a\mid\pi,s)=\Pr(A_t=a\mid \Pi=\pi, S_t=s)$. The **virtual policy** $\pi$ is the agent's *intended* action distribution; the **empirical policy** $\dot\pi(a\mid s):=\mathcal{P}_A(a\mid\pi,s)$ is what actually executes. A standard MDP is the case $\mathcal{P}_A(a\mid\pi,s)=\pi(a\mid s)$. The objective maximizes $\widetilde{J}(\pi)=\mathbb{E}_\pi[\sum_t \gamma^t R_t]$ over executed trajectories.

**Key equations (the Bellman-style update per assumption):**
- **Reward Maximization / "black-box"** (e.g. evolutionary / black-box policy search): $\pi^{\text{RM}}=\arg\max_\pi \mathbb{E}_{\widetilde{\mathcal M}}[\sum_t \gamma^t R_t\mid \Pi=\pi]$ — fully models $\mathcal{P}_A$; can be "treacherous" and manipulate the modification mechanism.
- **Bellman optimality (off-policy / Q-learning, "ignores modifications"):** $Q^{\text{BO}}(s,a)=\mathcal{R}(s,a)+\gamma\,\mathbb{E}_{s'\sim\mathcal{P}_S(s,a)}\max_{a'}Q^{\text{BO}}(s',a')$, $\pi^{\text{BO}}(s)=\arg\max_a Q^{\text{BO}}(s,a)$.
- **Virtual Policy Value (on-policy, successor sampled from the *virtual* policy):** $Q^{\text{VPV}}_{\pi}(s,a)=\mathcal{R}(s,a)+\gamma\,\mathbb{E}_{s'\sim\mathcal{P}_S(s,a)}\,\mathbb{E}_{a'\sim\pi(s')}Q^{\text{VPV}}_{\pi}(s',a')$.
- **Empirical Policy Value (on-policy, successor sampled from the *empirical* policy $\mathcal{P}_A$):** $Q^{\text{EPV}}_{\pi}(s,a)=\mathcal{R}(s,a)+\gamma\,\mathbb{E}_{s'\sim\mathcal{P}_S(s,a)}\,\mathbb{E}_{a'\sim\mathcal{P}_A(s',\pi)}Q^{\text{EPV}}_{\pi}(s',a')$.

**Main results / theorems:**
- *Prop. 1:* $\pi$ satisfies RM iff optimal for $\widetilde{\mathcal M}$ (accounts for modification).
- *Prop. 2 & 3:* a policy satisfying Bellman optimality, or virtual policy value, is optimal for the *underlying* $\mathcal M$ — i.e. it **ignores** $\mathcal{P}_A$ entirely.
- *Prop. 4:* $Q^{\text{EPV}}_{\pi}=Q_{\mathcal{M},\dot\pi}$ — empirical value equals the underlying-MDP value of the *empirical* policy $\dot\pi$.
- *Thm. 5:* general stochastic-approximation convergence for updates $Q_{t+1}=(1-\alpha_t)Q_t+\alpha_t(R+\gamma Q_t(S',A'))$ with $A'\sim\Lambda_t$ to the fixed point $Q^{\Lambda}(s,a)=\mathcal{R}(s,a)+\gamma\,\mathbb{E}_{S'}\mathbb{E}_{A'\sim\Lambda(S',Q^\Lambda)}Q^\Lambda$.
- *Thm. 6 (Q-learning) & Thm. 7 (Virtual Sarsa):* converge a.s. to $Q^*_{\mathcal M}$ (ignore modifications).
- *Thm. 8 (Empirical Sarsa):* converges a.s. to $Q^{\text{EPV}}$ when an EPV solution exists (not always satisfiable — Appendix A counterexample).
- **Incentive taxonomy (via influence diagrams):** *adversarial state incentive* (steer toward states where $\mathcal{P}_A$ is less constraining) vs *adversarial policy incentive* (reshape the policy to defeat $\mathcal{P}_A$). RM admits both; Bellman/Virtual admit neither; Empirical admits the state incentive but **not** the policy incentive — a useful middle ground.

**Algorithms:** Q-learning, Virtual Sarsa, Empirical Sarsa (modified Bellman backups above), each with convergence proofs.

**Relevance to pgmpy:** Motivates representing an "action-modification" mechanism node between the decision node and the executed action in a CID, and supports incentive queries (response incentive / adversarial incentives) on that structure. The four backups are concrete value-iteration variants a CID/MDP solver could expose; the influence-diagram incentive labels map onto pgmpy graphical-criteria algorithms.

---

### Path-Specific Objectives for Safer Agent Incentives (NeurIPS / AAAI-22, 2022)
**Authors:** Sebastian Farquhar, Ryan Carey, Tom Everitt
**Link:** https://arxiv.org/abs/2204.10018
**TL;DR:** Defines a *path-specific objective* (PSO) — the agent maximizes a utility computed under a Pearl path-specific counterfactual that holds "delicate" variables at the value they would take under a default/trustworthy policy — and proves graphically that this removes the agent's instrumental control incentive (ICI) over those delicate variables while preserving task performance.

**Problem & motivation:** Agents often have incentives to influence variables we'd rather they leave alone (a user's preferences, a sensitive feature, a shutdown switch). Penalizing side-effects is brittle; instead the paper edits the *objective* so the undesired influence is counterfactually "turned off," removing the incentive at the source.

**Formal setup & key definitions:** Built on a **SCIM** (structural causal influence model) $\mathcal M=\langle G,\mathcal E,\mathbf F,P\rangle$ where $G$ is a CID with vertices partitioned into structure $\mathbf X$, action $\mathbf A$, utility $\mathbf U\in\mathbb R^n$ nodes; $\mathbf F=\{f^V\}$ are structural functions; information links point into action nodes. A **delicate MDP** factors each state into a *delicate* part $Z_t$ (must not be influenced) and a *robust* part $S_t$.
- **Path-specific effect (Pearl):** for an edge-subgraph $G'$, replace each structural function by $\bar f_i(pa_i,\epsilon;G')=f_i\big(pa_i(G'),\,\bar p_i(\widetilde{G'}),\,\epsilon\big)$, where parents on the *off-path* edges $\widetilde{G'}$ are frozen at their values under the reference action $\bar x$, then take the total effect in the modified model.

**Key equations:**
- **Path-specific objective (Def. 5.1):** $\mathcal U^{G',\bar a}_{\pi,a}(\epsilon)=\mathrm{SE}_{G'}(a,\bar a;\mathcal U,\epsilon)_{\mathcal M_{\pi,\bar a}}$ — the $G'$-specific effect of the action $a$ (vs default $\bar a$) on utility, future actions following $\pi$.
- **Reduction (Prop. 1):** with $G'$ = $G$ minus the arrows $A_{t'}\!\to\! Z_{t'+1}$ and $S_{t'}\!\to\! Z_{t'+1}$ for $t'\ge t$, $\mathcal U^{G',\bar a}_{\pi,a}(\epsilon)=\mathcal U_{\pi,\mathbf Z_{\bar a},a}(\epsilon)$ — utility under policy $\pi$ and action $a$, but with the delicate state $\mathbf Z$ set to its *nested counterfactual value $\mathbf Z_{\bar a}$ under the default policy*.

**Main results / theorems:**
- *Theorem E18 (graphical ICI criterion):* a single-decision CID admits an instrumental control incentive over $X$ **iff** there is a directed path $A \to X \to \mathbf U$.
- *Proposition 1:* the chosen edge-subgraph $G'$ deletes all $A\to Z$ paths, so by E18 the PSO admits **no ICI over $\mathbf Z$**; side-effects can still occur but are not *incentivized*.

**Algorithms:** Estimate the counterfactual "natural" value $\bar z$ of the delicate variable via one of three baselines: (i) **policy baseline** $\bar z\sim p_{\bar\pi}(Z\mid s_t,z_t)$ (a trustworthy reference policy); (ii) **state baseline** $\bar z\sim \hat p(z\mid z_t,s_t)$ (natural evolution); (iii) **fixed state** $\bar z=z_t$. Train the agent to maximize the PSO computed against this baseline.

**Relevance to pgmpy:** Requires (1) edge-subgraph / path-set objects on a CID; (2) a *path-specific counterfactual* evaluator — nested twin-network construction that freezes off-path parents at counterfactual values (an extension of standard do-calculus intervention already in pgmpy's causal modules); (3) the $A\to X\to U$ directed-path incentive test (shared with the reward-tampering and fairness papers).

---

### Why Fair Labels Can Yield Unfair Predictions: Graphical Conditions for Introduced Unfairness (AAAI/AISTATS, 2022)
**Authors:** Carolyn Ashurst, Ryan Carey, Silvia Chiappa, Tom Everitt
**Link:** https://arxiv.org/abs/2202.10816
**TL;DR:** Even when training *labels* are perfectly fair w.r.t. a sensitive attribute, an optimal predictor can *introduce* new unfairness; the paper gives an exact graphical criterion — a predictor introduces unfairness iff it depends on a "requisite" feature that is d-connected to the sensitive attribute — and extends it to path-specific (in)fairness.

**Problem & motivation:** Fairness work usually assumes the *labels* are biased; this paper studies the opposite — fair labels $Y$ but a predictor $\hat Y$ that nonetheless amplifies disparities. It characterizes graphically exactly which causal structures make this "introduced unfairness" possible, and how loss-function choice and feature availability affect it.

**Formal setup & key definitions:** An **SL (supervised-learning) SCM** $\langle\mathcal E,\mathbf V,\mathbf F,P(\mathcal E)\rangle$ with outcome $Y$, prediction $\hat Y$, loss $U$, sensitive attribute $A$, features $\mathrm{Pa}^{\hat Y}$; structural equations $V\leftarrow f_V(\mathrm{Pa}^V,\mathcal E^V)$. Uses standard **d-separation**.
- **Average Total Variation:** $\mathrm{ATV}(V)=\mathbb E(V\mid A=a_1)-\mathbb E(V\mid A=a_0)$.
- **Introduced Total Variation (the central quantity):** $\mathrm{ITV}=|\mathrm{ATV}(\hat Y)|-|\mathrm{ATV}(Y)|$; $\mathrm{ITV}>0$ = predictions are *more* disparate than labels.
- **Separation** $\hat Y\perp A\mid Y$; **Sufficiency** $Y\perp A\mid \hat Y$.
- **Requisite feature (Def. 7):** $W\in\mathrm{Pa}^{\hat Y}$ is requisite if it is d-connected to loss $U$ conditional on $\mathrm{Pa}^{\hat Y}\cup\{\hat Y\}\setminus\{W\}$ (i.e. genuinely used by the optimal predictor).

**Key equations:**
- $\mathrm{ITV}=|\mathrm{ATV}(\hat Y)|-|\mathrm{ATV}(Y)|$ (Def. 3).
- *Prop. 4 (separation bounds it):* separation $\Rightarrow |\mathrm{ATV}(\hat Y)|\le|\mathrm{ATV}(Y)|\Rightarrow \mathrm{ITV}\le 0$.
- **Introduced Mutual Information:** $\mathrm{IMI}=I(\hat Y;A)-I(Y;A)=\mathrm{SEP}-\mathrm{SUF}$ (decomposes into separation- and sufficiency-violation terms).
- **Path-specific introduced effect (Def. 12):** $\mathrm{PSIE}_{\mathcal P}=|\mathrm{PSE}_{\mathcal P}(\hat Y)|-|\mathrm{PSE}_{\mathcal P}(Y)|$ over a path-set $\mathcal P$, where $\mathrm{PSE}(V)=\mathbb E(V_{\mathcal P(a_0\to a_1)})-\mathbb E(V_{a_0})$.

**Main results / theorems (the graphical criteria):**
- *Theorem 5.1 (ITV criterion):* an SL graph is compatible with $\mathrm{ITV}>0$ **iff** there exists a requisite feature $W\in\mathrm{Pa}^{\hat Y}$ that is **d-connected to $A$**.
- *Theorem 5.2 (P-admissible loss):* for "P-admissible" losses (where $\pi(\mathrm{Pa}^{\hat Y})=\mathbb E(Y\mid\mathrm{Pa}^{\hat Y})$ is optimal, e.g. squared/cross-entropy), $\mathrm{ITV}>0$ additionally requires $A\notin\mathrm{Pa}^{\hat Y}$ **and** $A$ d-connected to $U$ given $\mathrm{Pa}^{\hat Y}$. *Corollary 10:* if $A$ is itself available as a feature, no introduced-unfairness incentive exists under P-admissible loss.
- *Theorem 6.4 (path-specific criterion):* a graph is compatible with $\mathrm{PSIE}>0$ **iff** there is a path $p\in\mathcal P$ of the form $A\dashrightarrow W\to\hat Y$ with $W\in\mathrm{req}(\mathrm{Pa}^{\hat Y})$.

**Algorithms:** The criteria reduce to d-connection / requisiteness tests on the SL graph; empirically only ~16–20% of criterion-satisfying models actually exhibit $\mathrm{ITV}>0.01$, so the criteria are necessary-and-possible (worst-case), not always manifested.

**Relevance to pgmpy:** Directly implementable as graphical-query functions over a labeled SCM/CID: d-separation/d-connection (pgmpy has this), a *requisite-feature* test (d-connection to a utility/loss node conditional on the other parents — i.e. a requisite-graph computation), and path-specific-effect machinery shared with the Path-Specific-Objectives paper. The ATV/ITV/IMI quantities are estimable from a fitted model.

---

### Modeling AGI Safety Frameworks with Causal Influence Diagrams (IJCAI AI Safety Workshop, 2019)
**Authors:** Tom Everitt, Ramana Kumar, Victoria Krakovna, Shane Legg (DeepMind)
**Link:** https://arxiv.org/abs/1906.08663
**TL;DR:** A unifying *catalogue* paper that draws a dozen AGI-safety training frameworks as causal influence diagrams, showing CIDs as a common language to compare their incentive structures.

**Problem & motivation:** Safety proposals (reward modeling, CIRL, debate, IDA, etc.) are described in disparate ways; the paper argues a single CID per framework exposes its information links and incentives, enabling like-for-like comparison.

**Formal setup & key definitions:** A **CID** is a causal Bayesian network augmented with **decision** nodes (squares, agent-controlled), **utility** nodes (diamonds, real-valued, to be maximized), and **chance** nodes (circles). **Causal edges** (solid) are direct causal influence; **information links** (dashed, into decisions) say what the agent observes — a decision may depend only on its parents.

**Key equations:** $R_i=R(S_i;\Theta^R)$ (reward in an unknown MDP); $R_i=M(S_i\mid D_1,\dots,D_{i-1})$ (reward-modeling: reward from a learned model $M$ of feedback data); $D_i=G(S_1,A_1,\dots,S_i,A_i;\Theta^H)$ (human feedback generation, $\Theta^H$ = human preferences).

**Frameworks modeled (each = one CID):**
- **RL in MDP** — $S_i\to A_i\to R_i$ with Markovian transitions.
- **RL in Unknown MDP** — adds latent transition/reward params $\Theta^T,\Theta^R$.
- **RL in POMDP** — observations $O_i$ replace direct state access.
- **Current-RF optimization** — time-varying $\Theta^R_i$; agent optimizes model-based counterfactual trajectories (links to the reward-tampering paper).
- **Reward modeling** — feedback $D_i$ and human prefs $\Theta^H$ mediate reward through learned model $M$.
- **CIRL / cooperative IRL** — human + agent share a reward with parameter $\Theta^H$ unknown to the agent; human action $A^H$ observed.
- **Supervised learning** — Question $\to$ Answer $\to$ Label (independent of answer) $\to$ Reward.
- **Self-fulfilling prophecies** — Answer causally affects State affects Reward (manipulation incentive).
- **Counterfactual oracles** — twin-network separating counterfactual vs actual state; agent optimizes the world where its answer is hidden.
- **Debate** — two agents argue, a user judges, reward depends on the judgment.
- **Supervised IDA (iterated amplification)** — hierarchical sub-question answering combined into an approximate answer that trains a stronger system.
- **Comprehensive AI Services (CAIS)** — many bounded services whose outputs feed each other, each optimizing short-term reward.

**Main results / algorithms:** No theorems; the contribution is the modeling methodology and the observation that incentive analysis (value of information/control) on these CIDs distinguishes safe vs unsafe structures.

**Relevance to pgmpy:** This is the design "requirements doc" for a CID feature: it enumerates the structural idioms (latent params, information links vs causal edges, twin/counterfactual networks, multi-agent decisions) that a CID class must represent, and motivates the same incentive-analysis queries PyCID later implemented.

---

### PyCID: A Python Library for Causal Influence Diagrams (Proc. SciPy, 2021)
**Authors:** James Fox, Tom Everitt, Ryan Carey, Eric Langlois, Alessandro Abate, Michael Wooldridge
**Link:** https://proceedings.scipy.org/articles/majora-1b6fd038-008 · code: https://github.com/causalincentives/pycid
**TL;DR:** PyCID is a library, **built directly on top of pgmpy and networkx**, that represents single- and multi-agent causal influence diagrams, computes optimal policies and Nash/subgame-perfect equilibria, and checks graphical incentive criteria — making it the closest existing prior art for a pgmpy CID design.

**Problem & motivation:** CIDs/MAIDs had no reusable implementation; PyCID provides one so researchers can define, solve, and analyze decision/incentive problems and verify the graphical criteria from the papers above.

**Formal setup & data model (verified from source — this is the load-bearing part for the proposal):**
- **`CausalBayesianNetwork(BayesianNetwork)`** — *subclasses pgmpy's `BayesianNetwork`*. A CBN is "a Bayesian Network where the directed edges represent every causal relationship." It overrides `add_edge`/`remove_edge`/`add_cpds`/`remove_cpds`, and adds `intervene(intervention: Dict)`, `query(...)` (wraps pgmpy `BeliefPropagation`), `expected_value(...)`, `sample(...)` (wraps pgmpy `BayesianModelSampling`), `is_structural_causal_model()`, `draw()`. An inner **`Model(collections.UserDict)`** keeps CPDs+domains in sync: on `__setitem__` it converts a "relationship" to a `TabularCPD` and calls `BayesianNetwork.add_cpds`, propagating domain/state-name changes to descendants.
- **CPD layer (also subclasses pgmpy):** **`StochasticFunctionCPD(TabularCPD)`** lets a CPD be specified by an arbitrary Python function of parent values (e.g. `lambda S, D: S*D`) rather than an explicit table — it builds the underlying `TabularCPD`. Subclasses: **`ConstantCPD`**, **`DecisionDomain(ConstantCPD)`** (marks a decision's action domain). Helper distributions `bernoulli(p)`, `discrete_uniform(domain)`, `noisy_copy(...)`.
- **`MACIDBase(CausalBayesianNetwork)`** — adds agent/decision/utility bookkeeping: `decisions`, `utilities`, `agents`, `make_decision/make_utility/make_chance`, `expected_utility(...)`, `get_valid_order`, **`is_s_reachable` / `is_r_reachable`** (strategic/requisite reachability), **`sufficient_recall`**, `pure_decision_rules`, `pure_policies`, **`optimal_pure_policies` / `optimal_pure_decision_rules`**, `impute_optimal_decision`, `impute_random_decision`, `impute_conditional_expectation_decision`. A `MechanismGraph(MACIDBase)` adds mechanism nodes.
- **`CID(MACIDBase)`** (single agent): `impute_optimal_policy()`, `optimal_policies()`, `impute_random_policy()`, `solve()`.
- **`MACID(MACIDBase)`** (multi-agent): **`get_ne(solver=None)`** (Nash equilibria), `get_ne_in_sg(...)` (NE in a subgame), **`get_spe()`** (subgame-perfect equilibria), `create_subgame`, `decs_in_each_maid_subgame`, `joint_pure_policies`, `policy_profile_assignment`. Equilibrium solving exports to extensive-form games via `pycid.export.gambit` (`macid_to_efg`, `pygambit_ne_solver`, `behavior_to_cpd`) — i.e. it delegates NE computation to **pygambit**.
- **Relevance graphs:** `RelevanceGraph(nx.DiGraph)` and `CondensedRelevanceGraph(nx.DiGraph)` — *subclass networkx* — encode which decisions are strategically relevant to which, used to find subgames.

**Example API (verbatim from README):**
```python
import pycid
cid = pycid.CID(
    [('S', 'D'), ('S', 'U'), ('D', 'U')],
    decisions=['D'], utilities=['U'])
cid.add_cpds(S=pycid.discrete_uniform([-1, 1]),
             D=[-1, 1],
             U=lambda S, D: S * D)
cid.draw()
```

**What it computes (incentive analysis — module `pycid/analyze/`, function-level, verified):**
- **Value of Information** (`value_of_information.py`): `admits_voi(cid, decision, node)`, `admits_voi_list(cid, decision)`, `quantitative_voi(cid, decision, node)`.
- **Value of Control** (`value_of_control.py`): `admits_voc(cid, node)`, `quantitative_voc(...)`, plus directed/indirect variants `admits_dir_voc`, `admits_indir_voc`.
- **Response Incentive** (`response_incentive.py`): `admits_ri(cid, decision, node)`.
- **Instrumental Control Incentive** (`instrumental_control_incentive.py`): `admits_ici(cid, decision, node)` — the $A\to X\to U$ criterion from the AI:ACP and Responsiveness papers.
- **Reasoning patterns** (`reasoning_patterns.py`, MAID-level): `direct_effect`, `manipulation`, `signaling`, `revealing_or_denying`, `get_reasoning_patterns` (Pfeffer–Gal patterns), built on effective directed/backdoor path checks.
- **Requisite graph** (`requisite_graph.py`): `requisite(cid, decision, node)`, `requisite_graph(cid)` (prunes non-requisite information links — same notion as the Fair-Labels paper's "requisite feature").
- **Effects/interventions** (`effects.py`) and **random CID generation** (`pycid/random`).

**Architecture / package structure:** `pycid/core` (model classes, paths `get_paths.py`, relevance graphs, optimal policies, NE), `pycid/analyze` (incentives, effects, reasoning patterns, requisite graph), `pycid/export` (gambit interop), `pycid/random`, `pycid/examples`, plus tutorial `notebooks` and `tests`.

**Limitations / future work:** Discrete finite-domain variables only (relies on `TabularCPD` and exhaustive `pure_policies` enumeration — exponential in #decisions/domain size); optimal-policy and NE search are brute-force over pure decision rules; equilibrium solving requires the external `pygambit` dependency; continuous variables, scalable solvers, and richer mixed-strategy handling are noted as future directions.

**Relevance to pgmpy (most direct prior art):** PyCID demonstrates the exact integration pattern a pgmpy-native CID should follow — subclass `BayesianNetwork` (add typed decision/utility nodes + information links), subclass `TabularCPD` with a function-based CPD (`StochasticFunctionCPD`) and a `DecisionDomain`, reuse `BeliefPropagation`/`BayesianModelSampling` for queries and `intervene` for do-operations, use networkx relevance graphs for subgame/strategic-relevance, and expose incentive criteria (`admits_voi/voc/ri/ici`, requisite graph) plus optimal-policy/Nash solvers as analysis functions. Its limitations (tabular-only, brute-force policy enumeration, external gambit dependency) are the concrete gaps a new pgmpy design should aim to close.

---

## Glossary of recurring concepts

### Model classes (Pearl's causal hierarchy × number of agents)

The whole corpus lives in one 3×3 table (from *Reasoning about Causality in Games*, Fig. 5): rows = associational / interventional / counterfactual; columns = 0 / 1 / $n$ agents.

| | 0 agents | 1 agent | $n$ agents |
|---|---|---|---|
| **Associational** | BN (Bayesian network) | ID (influence diagram) | MAID (multi-agent ID) |
| **Interventional** | CBN (causal BN) | CID (causal ID) | CG (causal game) |
| **Counterfactual** | SCM (structural causal model) | SCIM (structural causal influence model) | SCG (structural causal game) |

- **CID** — a DAG with vertices partitioned into **chance/structure** $\bm X$, **decision** $\bm D$ (agent-controlled; no CPD until a policy is set), and **utility** $\bm U$ (real-valued, childless) nodes, on top of a CBN. Edges into decisions are **information links** (what the agent observes).
- **SCIM** — a CID plus exogenous variables $\bm{\mathcal E}$ and structural functions $\bm F$ for every non-decision node; supports counterfactuals (twin networks / abduction–action–prediction).
- **MAID / CG / SCG** — the $n$-agent versions, with per-agent ownership of decisions $\bm D^i$ and utilities $\bm U^i$; agent $i$ maximises $\sum_{U\in\bm U^i}U$. SCG ≈ "an SCM with unparameterised decision variables."
- **Mechanised graph** $m\mathcal G$ — augments the object-level graph with one **mechanism node** per variable: a **decision-rule node** $\Pi_D$ for decisions and a **parameter node** $\Theta_V$ for non-decisions. Edges into $\Pi_D$ encode what an agent's choice of rule depends on. May be **cyclic** (mutual best response). The substrate for *Discovering Agents*, decision-theory characterisation, and relevance.
- **II-MAID / belief hierarchy** — a recursive set $\bm S$ of subjective MAIDs with per-agent priors $P_i^S$ over $\bm S$, one flagged objective model $S^*$; represents higher-order beliefs without a common prior.
- **MDP / cMP / MAMDP** — (controlled) Markov processes; a **MAMDP** adds an action-selection function $\mathcal P_A(a\mid\pi,s)$ that rewrites the agent's action.

### Policy & decision-rule objects

- **Decision rule** $\pi_D(D\mid \mathbf{Pa}_D)$ — a CPD supplied for a decision node; **pure** if $\in\{0,1\}$. **Policy** $\pi^i=\pi_{\bm D^i}$; **policy profile** $\pi=(\pi^1,\dots,\pi^n)$; $\pi^{-i}$ = others.
- **Behavioural** (randomise independently per decision) vs **mixed** $\mu^i\in\Delta(\check{\bm P}^i)$ (one outset draw over pure policies) vs **behavioural-mixture** (both); distinguished by a **correlation node** $C^i$.
- **Optimal policy** maximises $\mathbb E_\pi[\mathcal U]$ with $\mathcal U=\sum_{U}U$; **value** $\mathcal V^*(\mathcal M)=\max_\pi\mathbb E_\pi[\mathcal U]$.

### The four incentive criteria (single-decision, sound & complete — *AI:ACP*)

Let $\mathcal G^{\min}$ be the **minimal reduction** (delete every information link $X\to D$ that is *nonrequisite*, i.e. $X\perp \bm U^D\mid \mathbf{Pa}^D\cup\{D\}\setminus\{X\}$). Then:

| Criterion | Meaning | Graphical test |
|---|---|---|
| **Value of Information (VoI)** | gain from *observing* $X$ at $D$ | $X$ requisite in $\mathcal G_{X\to D}$ (d-connected to $\bm U^D$ given the rest of $D$'s family) |
| **Value of Control (VoC)** | gain from *setting* $X$ | directed path $X\dashrightarrow \bm U$ in $\mathcal G^{\min}$ |
| **Response Incentive (RI)** | optimal decision *changes* if $X$ is perturbed | directed path $X\dashrightarrow D$ in $\mathcal G^{\min}$ |
| **Instrumental Control Incentive (ICI)** | policy influences $\bm U$ *through* $X$ | directed path $D\dashrightarrow X\dashrightarrow \bm U$ |

Extensions: **Impact incentive** (some $W$ and some $U$ both descendants of $D$; *Responsiveness* paper); **Intent** (same path as ICI, $D\dashrightarrow W\dashrightarrow U$); **multi-decision VoI** (link survives the **minimal d-reduction** in a *soluble* ID; *Complete VoI Criterion*).

### Graphical constructions

- **Requisite observation / requisite graph** — an information link is requisite iff its source is d-connected to a utility descendant of the decision; pruning the nonrequisite ones gives $\mathcal G^{\min}$ (a.k.a. reduced graph). Same notion reused for "requisite feature" in the fairness paper.
- **Solubility** — an ID is soluble if some decision ordering admits backward induction: $\Pi^{<i}\perp \bm U(D^i)\mid \mathbf{Fa}(D^i)$.
- **Strategic relevance / s-reachability / r-reachability** — whether one decision's optimal rule depends on another's; computed as a **d-separation test on the mechanised graph**. The **relevance graph** has a node per decision and an edge per relevance; its **SCC condensation** identifies **subgames**.
- **Structural mechanism intervention** — a soft intervention on $\Theta_V/\Pi_D$ that severs $V$ from its object-level parents (used to *discover* agents and to define pre- vs post-policy interventions).

### Game-theoretic solution concepts

- **Nash equilibrium (NE)** — every $\pi^i$ best-responds to $\pi^{-i}$.
- **Subgame-perfect equilibrium (SPE)** — NE in every MAID subgame (found via relevance-graph condensation + backward induction).
- **Trembling-hand-perfect (THPE)** — limit of NEs of $\epsilon$-perturbed games ($\pi_D(d\mid\mathbf{pa}_D)\ge\epsilon$).
- **Correlated equilibrium (CE)** and **MAID-CE** — mediator-based; MAID-CE staggers per-decision recommendations and cuts off deviators, giving a larger (Pareto-improving) outcome set.
- **Recall taxonomy** — perfect ⊃ sufficient recall gate which policy class has a guaranteed equilibrium (pure / behavioural / mixed). **Forgetful** vs **absent-minded** (a rule node with >1 outgoing decision edge) are graph predicates on $m\mathcal G$.

### Safety quantities

- **Counterfactual harm** $h(a,x,y)=\int_{y^*}P(Y_{\bar a}=y^*\mid a,x,y)\max\{0,U(\bar a,x,y^*)-U(a,x,y)\}$ — utility shortfall vs a **default action** $\bar a$; **HPU** $=U-\lambda h$.
- **Deception** — $S$ intentionally causes $T$'s decision via a false belief $S$ doesn't hold; sound/complete path criterion $D^S\dashrightarrow U^S$ through $D^T$ with $Z$ unobserved by $T$.
- **Intention** (subjective/behavioural) — outcomes the agent would no longer act to bring about if they were counterfactually guaranteed; coincides with the ICI path.
- **MEG** (Maximum Entropy Goal-directedness) — expected log-likelihood ratio of the best-fitting max-entropy (soft-optimal) policy vs the uniform policy; degree of goal-directedness.
- **Shutdown instructability** — obedience + vigilance + caution (human-control variant of corrigibility).
- **World-model error vs regret/competence** — bounded-regret agents provably encode a causal/transition model with error $\to 0$ as regret $\to 0$ (or goal-depth $\to\infty$).
- **Partial identification** — when OOD behaviour / fairness / harm are only *bounded* (not identified) from behavioural data: compute $\min/\max$ of a causal query over all consistent SCMs.

---

## Synthesis: implications for a pgmpy design proposal

> This is a *literature synthesis* — the input to the design proposal, not the proposal
> itself. It extracts the recurring primitives the papers demand and maps them onto
> pgmpy's existing architecture. Design decisions (scope, API, phasing) are deferred to
> the brainstorming + proposal step.

### 1. Almost everything reduces to a small primitive set on a typed DAG

Across all 24 papers, the load-bearing requirements collapse to a layered stack. Each
layer is reused by many papers, so the design can be incremental.

| Layer | Primitive | Papers that need it |
|---|---|---|
| **L0 — typed graph** | DAG with node roles **chance / decision / utility**, utility childless; **information links** vs causal edges; per-agent ownership $\bm D^i,\bm U^i$ | all |
| **L0′ — mechanised graph** | parallel graph adding **mechanism nodes** $\Pi_D$ (decision-rule), $\Theta_V$ (parameter); may be **cyclic** | Reasoning-about-Causality, Discovering Agents, Decision-Theories, Imperfect-Recall, Reward-Tampering |
| **L1 — parameterisation** | CPDs for chance/utility; **decision left unparameterised until a policy**; function-CPDs (`U=lambda S,D: ...`); soft/local interventions; structural-mechanism intervention | all quantitative results |
| **L2 — graphical criteria engine** | d-separation (have it), **minimal reduction / requisite graph**, directed-path & descendant queries, **relevance graph + SCC condensation**, **solubility** test | AI:ACP, Responsiveness, Complete-VoI, Fair-Labels, Path-Specific, Deception, Intent, Equilibrium-Refinements |
| **L3 — decision/game solvers** | expected utility $\mathbb E_\pi[\mathcal U]$ (reuse VE/BP), **optimal policy** (backward induction over subgames), **NE / SPE / THPE / CE / MAID-CE**, soft value iteration (MEG) | AI:ACP, Equilibrium-Refinements, Imperfect-Recall, Higher-Order-Belief, MEG, Human-Control |
| **L4 — counterfactual & partial-ID engine** | twin-network / abduction–action–prediction; **path-specific counterfactuals**; **bounds** (min/max over consistent SCMs) | Counterfactual-Harm, Intent, Deception, Path-Specific, Limits-of-Predicting |
| **L5 — discovery** | leave-one-out interventional discovery (cyclic-safe); **agency identification** (terminal-edge labelling → game graph); world-model extraction from a policy | Discovering-Agents, Robust-Agents, General-Agents |
| **L6 — applied analyses / metrics** | incentive detectors (`admits_voi/voc/ri/ici`, impact, intent), **counterfactual unfairness = response incentive on a sensitive attribute**, introduced-unfairness, HPU decisions, shutdown-instructability constraints, MEG, goal-directedness ratio | AI:ACP §fairness, Fair-Labels, Counterfactual-Harm, Human-Control, MEG, LLM-goal-directedness |

The single most reused fact: **incentive analysis is graph-only** (L0+L2), so it is the
cheapest, highest-leverage first deliverable; the expensive pieces (L3 solvers, L4
counterfactuals) are needed only for *magnitudes* and *equilibria*.

### 2. Mapping onto pgmpy's current architecture

- **`pgmpy/base/` (DAG, PDAG, role annotations).** L0 is a direct extension of the
  existing node-role annotations (exposures/outcomes/… → decision/utility/chance) and
  d-separation engine. The mechanised graph (L0′) is a *second* graph object; its
  possible cyclicity means it cannot reuse the acyclic `DAG` invariants directly.
- **`DiscreteBayesianNetwork` + `TabularCPD`.** PyCID's proven pattern: subclass
  `BayesianNetwork`, add a `StochasticFunctionCPD(TabularCPD)` for function-defined CPDs
  and a `DecisionDomain`. pgmpy now also has `LinearGaussianCPD` and `FunctionalCPD`,
  which open the door to the **continuous / non-tabular** CIDs PyCID explicitly lacked.
- **`inference/` (VariableElimination, BeliefPropagation).** Directly reusable for
  $\mathbb E_\pi[\mathcal U]$ and for the d-connection tests behind every criterion.
- **`inference/CausalInference` (do-operator).** Covers L1 interventions and the
  associational/interventional layers; **does not** yet do counterfactual (twin-network)
  or path-specific queries (L4) — a real gap for Harm/Intent/Deception/Path-Specific.
- **`causal_discovery/`.** The natural home for L5 (agency identification, world-model
  extraction) — these are *new kinds* of discovery (interventional, from a policy/oracle)
  beyond the current observational PC/GES.
- **`metrics/`.** The applied detectors (L6) sit alongside `SHD`, `FisherC`, etc., as
  model-analysis metrics.
- **`global_vars.config` (numpy/torch backend).** MEG's soft value iteration and
  world-model fitting are gradient-friendly — a torch backend would make them
  differentiable; aligns with the existing multi-backend direction.

### 3. PyCID is the reference prior art — and the gap list

PyCID (Batch F) already implements L0–L3 + L6 *on top of pgmpy*. It is the blueprint **and**
the cautionary tale. What a pgmpy-native design should keep vs. improve:

- **Keep:** subclass-`BayesianNetwork` integration; function-CPDs; `intervene`; relevance
  graph on networkx; the `admits_*` incentive API; requisite graph; MAID↔EFG export.
- **Improve / close gaps:** (a) **tabular-only** → support `LinearGaussian`/`Functional`
  CPDs and continuous decisions; (b) **brute-force `pure_policies` enumeration**
  (exponential) → exploit subgame decomposition + VE, and approximate/continuous solvers;
  (c) **external `pygambit` dependency** for NE → consider native equilibrium solving or
  make it optional; (d) **no counterfactual / partial-ID layer** (L4); (e) **no discovery**
  (L5); (f) no torch/GPU backend.

A live question for the proposal: **port/absorb PyCID vs. build native vs. depend on it.**

### 4. Natural phasing (for the proposal to refine)

1. **Phase 1 — Representation + incentive criteria (L0–L2, single-agent CID).** Highest
   value, lowest risk: typed graph, function-CPDs, minimal reduction, the four `admits_*`
   detectors + impact/intent, requisite graph. Pure graph algorithms; mostly reuses
   existing d-separation.
2. **Phase 2 — Single-agent solving (L3).** Optimal policy via backward induction; VoI/VoC
   *magnitudes*; soluble multi-decision reduction; MEG.
3. **Phase 3 — Multi-agent (L0′ + L3).** MAID/mechanised graph, relevance graph, NE/SPE,
   equilibrium refinements, recall predicates; optional Gambit export.
4. **Phase 4 — Counterfactual & safety (L4 + L6).** Twin-network/counterfactual engine,
   path-specific objectives, counterfactual harm/HPU, deception/intent detectors,
   partial-identification bounds.
5. **Phase 5 — Discovery (L5).** Agency identification, world-model extraction.

### 5. Open questions to settle in brainstorming (not decided here)

- Scope of v1: incentive-analysis toolkit on single-agent CIDs, or full MAID/SCG stack?
- Relationship to PyCID (absorb / depend / reimplement) and to `pgmpy.base` role
  annotations (extend the same mechanism, or a dedicated `CID`/`MAID` class hierarchy?).
- Continuous/torch support from the start, or tabular-first like PyCID?
- Where counterfactual machinery lives (extend `CausalInference`, or a new SCM/SCIM class
  with structural functions + exogenous variables?).
- Whether equilibrium solving is in-scope for pgmpy at all, or a separate companion package.

These connect to the design-proposal template and should be resolved before drafting.
