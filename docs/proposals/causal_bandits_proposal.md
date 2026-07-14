# Proposal: `pgmpy.causal_bandits` - Structure-Aware Online Intervention Selection (v1)

## Status

Revised after an audit of the current pgmpy codebase and the assumptions of the algorithms cited in
`causal_bandits_literature_review.md`. This proposal is authoritative for v1 where it differs from the broader
roadmap in `causal_bandits_field_and_roadmap.md`.

## Contributors

- @ankurankan
- _(add reviewers / co-authors)_

## Introduction

pgmpy has a strong static causal stack: graph classes (`DAG`, `ADMG`, `MAG`, and `PDAG`), hard interventions,
tabular and continuous probabilistic models, exact and approximate inference, identification criteria, node roles,
and simulation. It does not have an online decision layer that repeatedly selects an intervention, receives
post-intervention feedback, and updates a learning strategy.

A causal bandit adds that sequential layer. An action is an intervention such as `do(X=x)`, the reward is a numeric
utility derived from a designated outcome node, and post-action observations can share information across actions.
For example, if all actions reveal the parents of the reward, a learner can estimate reward means for parent
configurations and use known `P(Pa(Y) | a)` distributions to score many actions from each observation.

The phrase "known causal model" is ambiguous in this literature. The distinction is essential for the API:

- The **environment** owns the complete, unknown data-generating process. A simulator may represent it with a fully
  parameterized `DiscreteBayesianNetwork`.
- The **learner** receives only the information assumed public by its algorithm: the graph, feasible actions,
  feedback contract, reward definition, and any explicitly known side distributions.
- The **benchmark evaluator** may additionally know each action's expected reward, but that oracle information is
  never exposed to the learner.

Without this separation, passing the same fully parameterized Bayesian network to the environment and learner makes
the learning problem trivial: the learner can inspect the reward CPD and compute the optimal intervention directly.

### Why pgmpy

The opportunity is not an empty software niche. As of July 2026, `causalrl` has a stable 1.x API and includes POMIS,
MABUC examples, environments, and broader causal-RL functionality. The case for pgmpy is instead:

1. A native online layer that composes with pgmpy's graph, factor, inference, and simulation objects.
2. Explicit, validated algorithm assumptions rather than accepting statistically invalid estimator-policy
   combinations.
3. Reference implementations and benchmarks that make the learner's information boundary auditable.
4. A foundation on which later POMIS, confounded, continuous, and active-discovery work can be implemented without
   changing the interaction contract.

`causalrl`, `sanghack81/SCMMAB-NIPS2018`, and the authors' reference implementations should be audited as comparison
and parity sources before implementing overlapping algorithms. They are not proposed as runtime dependencies.

## Scope

### Included in v1

V1 implements one coherent causal-bandit problem class rather than attempting to cover every paper behind a common
policy string:

- A fully known, causally sufficient `DAG` over discrete variables.
- A finite, explicit action set consisting of hard atomic interventions `do(X=x)` and optionally the null
  observational action `do()`.
- A single discrete reward node with an explicit state-to-utility mapping into `[0, 1]`.
- Post-action observation of the reward and all parents of the reward.
- Known action-specific parent distributions `P(Pa(Y) | a)`.
- Cumulative-regret learning with **C-UCB** (Lu et al., 2020).
- A structure-blind empirical UCB baseline over the same action set.
- An `ask()` / `tell()` learner API, a tabular simulation environment, experiment history, and separate oracle-based
  benchmark evaluation.

The public problem representation may support an explicit finite list of multi-node hard interventions if doing so
does not complicate the implementation, because C-UCB itself is defined for a supplied finite action set. Atomic
actions plus `do()` are the required and documented v1 surface.

### Explicitly out of scope for v1

- POMIS/MIS/POMPS structural action-space reduction.
- ADMG-backed or otherwise confounded environments.
- Causal Thompson Sampling and MABUC.
- Lattimore's fixed-design simple-regret algorithm and its truncated importance-weighted estimator.
- A generic estimator x policy product API.
- Simple-regret or fixed-confidence best-arm-identification guarantees.
- Soft, stochastic, continuous, targeted, or combinatorial interventions.
- Unknown-graph or interventional structure discovery.
- Costs, budgets, delayed feedback, and batched actions.
- Logged-bandit/off-policy evaluation on historical interventional datasets.
- Integration with the existing `pgmpy.metrics` registry.

These are not small switches on C-UCB. They use different action spaces, feedback, public knowledge, statistical
state, objectives, or causal representations and should receive separate proposals.

### Algorithm capability matrix

This matrix is part of the design contract. An implementation must not make combinations available unless their row's
requirements are satisfied.

| Algorithm | Objective | Learner-visible causal information | Per-round feedback | Reward model | Status |
|---|---|---|---|---|---|
| Empirical UCB | Cumulative regret | Explicit action set only | Selected action and reward | Bounded numeric | v1 |
| C-UCB | Cumulative regret | `Pa(Y)` and known `P(Pa(Y) \| a)` | Realized `Pa(Y)` and reward | `[0, 1]`, bounded/sub-Gaussian | v1 |
| C-TS | Bayesian cumulative regret | `Pa(Y)` and known `P(Pa(Y) \| a)` | Realized `Pa(Y)` and reward | Declared Beta-Bernoulli or Gaussian model | Deferred |
| Lattimore general algorithm | Simple regret | Known parent kernels and sampling design | Post-action variables and reward | Binary/bounded, estimator-specific | Deferred |
| POMIS-kl-UCB | Cumulative regret | Causal diagram and valid manipulable set | Selected intervention and reward | Bernoulli for kl-UCB variant | Deferred |
| MABUC Causal TS | Cumulative contextual objective | Observational distribution and current intention | Intention, selected action, and reward | Binary | Deferred |

## Proposed Solution

### Design principles

1. **Keep the hidden model hidden.** Learners receive `CausalBanditProblem`, never the environment's parameterized
   Bayesian network or benchmark oracle.
2. **Represent assumptions in types and validation.** Reward support, observed variables, feasible actions, and known
   parent kernels are part of the problem object.
3. **Expose algorithms, not arbitrary combinations.** `CUCB` owns the sufficient statistics and action-index formula
   required by the paper. Internal helpers may be shared, but unsupported combinations are not public API.
4. **Use `ask()` / `tell()` as the primary online contract.** A synchronous runner is a convenience for simulation,
   not the only way to use a learner.
5. **Separate learning from evaluation.** Regret requires oracle action means and is therefore a benchmark concern.
6. **Make actions canonical and hashable.** Pull counts and histories must not use mutable dictionaries as keys.

### Module layout

```text
pgmpy/causal_bandits/
  __init__.py
  types.py          Intervention, Interaction
  problem.py        CausalBanditProblem, atomic_interventions
  environment.py    BanditEnvironment, DiscreteBayesianNetworkEnvironment, CallbackEnvironment
  learners.py       BaseBanditLearner, EmpiricalUCB, CUCB
  experiment.py     BanditHistory, run_bandit
  evaluation.py     BenchmarkResult, evaluate_regret
  benchmarks.py     expected_rewards_from_model, parallel_bandit, reward_parent_bandit, sachs_bandit
pgmpy/tests/test_causal_bandits/
  test_types.py
  test_problem.py
  test_environment.py
  test_learners.py
  test_evaluation.py
  test_benchmarks.py
```

POMIS and other structural action-reduction functions are deliberately absent from the v1 layout.

## Public Contracts

### `Intervention`

Actions need value semantics because they are dictionary keys in learner state and result objects.

```python
@dataclass(frozen=True)
class Intervention:
    assignments: frozenset[tuple[Hashable, Hashable]]

    @classmethod
    def from_dict(cls, assignments: Mapping[Hashable, Hashable]) -> "Intervention": ...

    @classmethod
    def observe(cls) -> "Intervention": ...

    def as_dict(self) -> dict[Hashable, Hashable]: ...
```

Validation rejects duplicate variables, intervention on the reward node, unknown variables, and states outside the
model/problem domains. `Intervention.observe()` represents `do()` and is not conflated with an intervention that sets
a variable to its naturally likely value.

### `Interaction`

```python
@dataclass(frozen=True)
class Interaction:
    action: Intervention
    reward: float
    observed: Mapping[Hashable, Hashable]
```

`observed` contains post-action values. V1 problem validation requires it to contain every reward parent. The raw
outcome state may also be retained in `observed`; `reward` is the numeric value after applying the problem's utility
mapping.

### `CausalBanditProblem`

`CausalBanditProblem` is the learner-visible, immutable problem description. It does not store a Bayesian network,
CPDs, a simulator callback, or oracle expected rewards.

```python
@dataclass(frozen=True)
class CausalBanditProblem:
    graph: DAG
    actions: tuple[Intervention, ...]
    reward_node: Hashable
    reward_values: Mapping[Hashable, float]
    state_names: Mapping[Hashable, tuple[Hashable, ...]]
    observed_nodes: frozenset[Hashable]
    parent_distributions: Mapping[Intervention, DiscreteFactor]

    @classmethod
    def from_model(
        cls,
        model: DiscreteBayesianNetwork,
        *,
        actions: Sequence[Intervention],
        reward_node: Hashable | None = None,
        reward_values: Mapping[Hashable, float],
    ) -> "CausalBanditProblem": ...
```

`from_model` is a benchmark/convenience constructor. It computes `P(Pa(Y) | a)` for every action using exact
inference, copies only the DAG structure and state domains into the returned problem, and discards all CPDs. In real
applications, callers can construct the problem directly from a graph and parent-kernel estimates obtained from
prior data or domain knowledge. If an action directly fixes a reward parent, the resulting factor retains that parent
as a degenerate dimension so every action kernel has the same variable schema.

Immutability is logical as well as dataclass-level: construction defensively copies the graph, factors, state-name
collections, actions, and mappings. Public accessors do not return mutable references to internal state.

Validation includes:

- `reward_values` covers every reward state and maps into `[0, 1]`.
- Every action is feasible and does not intervene on the reward.
- The action set is non-empty and has no duplicates.
- `observed_nodes` includes the reward parents.
- Every parent distribution has exactly `Pa(Y)` as its variables, uses the declared state names, is normalized, and
  has support compatible with the problem.
- The v1 graph is acyclic and contains no declared latent nodes. Causal sufficiency remains an explicit problem
  assumption because it cannot be established from a graph object alone.

The model's existing `outcomes` role can provide a convenience default for `reward_node` when it contains exactly one
node. Actions remain explicit because an `exposures` role describes a causal analysis question, not necessarily the
set of operationally feasible interventions. No new role is added in v1.

### Environments

```python
class BanditEnvironment(Protocol):
    def step(self, action: Intervention) -> Interaction: ...


class DiscreteBayesianNetworkEnvironment:
    def __init__(
        self,
        model: DiscreteBayesianNetwork,
        problem: CausalBanditProblem,
        *,
        seed: int | None = None,
    ): ...

    def step(self, action: Intervention) -> Interaction: ...


class CallbackEnvironment:
    def __init__(
        self,
        problem: CausalBanditProblem,
        callback: Callable[[Intervention], Interaction],
    ): ...
```

The tabular environment owns the full Bayesian network. It validates that the model agrees with the public problem
but does not expose the model through the learner API. Reproducibility is sequence-based: two environments initialized
with the same seed produce the same interaction sequence, while repeated calls on one environment do not reset to the
same random draw.

Calling `DiscreteBayesianNetwork.simulate(n_samples=1)` currently checks and copies a model on every call. The first
implementation may reuse it for correctness, but the benchmark suite must measure this path. If per-round overhead is
material, the environment should maintain a sampler or implement a one-row ancestral sampling path rather than
prematurely changing the model API.

Physical experiments do not need an environment object. Users can call `learner.ask()`, execute the intervention in
their own system, and pass the resulting `Interaction` to `learner.tell()`.

### Learners

```python
class BaseBanditLearner:
    def ask(self) -> Intervention: ...
    def tell(self, interaction: Interaction) -> None: ...
    def recommend(self) -> Intervention: ...


class EmpiricalUCB(BaseBanditLearner):
    """Structure-blind UCB using reward counts and means per action."""


class CUCB(BaseBanditLearner):
    """Causal UCB using shared reward-parent statistics and known parent kernels."""
```

`EmpiricalUCB` updates only the selected action. It is the v1 baseline and also validates the common interaction
machinery without relying on causal side information.

For `CUCB`, let `z` index configurations of `Pa(Y)`. The learner maintains a count `N_z` and empirical reward mean
`mu_hat_z` for each configuration. At round `t`, it constructs a confidence bound `U_z(t)` and scores action `a` by

```text
score_t(a) = sum_z U_z(t) * P(Pa(Y)=z | a).
```

After playing `a`, it observes the realized parent configuration and reward and updates that configuration's
statistics. This is the required information-sharing mechanism: one observation can change the index of every action.

The confidence schedule, initialization, tie-breaking, and clipping to the reward range must follow the cited C-UCB
algorithm and be parameters only where the theory permits. Learners accept `random_state` for deterministic
tie-breaking; environments independently accept `seed` for data generation.

A public `OnlineEstimator.estimate(action) -> (mean, count)` abstraction is intentionally rejected. A scalar effective
count cannot represent the parent-configuration confidence vector used by C-UCB, and it cannot later represent the
posterior state used by Causal Thompson Sampling.

### Experiment runner and history

```python
@dataclass(frozen=True)
class BanditHistory:
    interactions: tuple[Interaction, ...]
    recommendation: Intervention
    action_counts: Mapping[Intervention, int]


def run_bandit(
    learner: BaseBanditLearner,
    environment: BanditEnvironment,
    horizon: int,
) -> BanditHistory: ...
```

`run_bandit` is a synchronous convenience loop. The same history can be built by a user-owned `ask()` / `tell()`
loop. It does not query an oracle or compute regret.

### Benchmark evaluation

```python
@dataclass(frozen=True)
class BenchmarkResult:
    history: BanditHistory
    cumulative_pseudo_regret: np.ndarray
    recommendation_regret: float
    expected_rewards: Mapping[Intervention, float]


def evaluate_regret(
    history: BanditHistory,
    expected_rewards: Mapping[Intervention, float],
) -> BenchmarkResult: ...


def expected_rewards_from_model(
    model: DiscreteBayesianNetwork,
    problem: CausalBanditProblem,
) -> Mapping[Intervention, float]: ...
```

V1 reports cumulative pseudo-regret
`sum_t (mu_star - mu(action_t))` and terminal recommendation regret
`mu_star - mu(history.recommendation)`. These are only available when an evaluator has oracle expected rewards.
Realized reward shortfall, fixed-confidence stopping, and simple-regret learning algorithms are separate concepts and
are not labeled as v1 guarantees.

`expected_rewards_from_model` is a benchmark-only helper that computes each action's exact expected utility from the
fully parameterized simulator. It is kept out of `CausalBanditProblem`, learner constructors, and `run_bandit`; callers
must pass its result explicitly to `evaluate_regret` after an experiment.

Regret data remains in `BenchmarkResult`; v1 does not add a new metric class to `pgmpy.metrics`, whose current base
classes are designed for graph comparison and graph-versus-data evaluation.

## Alternatives Considered

| Decision | Chosen for v1 | Rejected alternative | Reason |
|---|---|---|---|
| Learner knowledge | Immutable public problem separate from hidden environment | Pass one parameterized BN to both | Prevents reward-CPD and oracle leakage. |
| Online API | `ask()` / `tell()` with an optional runner | Agent that can only call `run(env, horizon)` | Supports external and physical experiments without coupling learning to simulation. |
| Algorithm surface | `EmpiricalUCB` and `CUCB` classes | Arbitrary estimator x policy strings | The cited methods require incompatible sufficient statistics and selection rules. |
| Actions | Explicit, hashable finite interventions | Infer all actions from node roles | Feasibility is operational metadata, not the same as exposure annotation. |
| Reward | Explicit utility map and `[0, 1]` validation | Cast a tabular outcome state to `float` | Tabular state labels are often non-numeric; UCB assumptions must be visible. |
| Regret | Separate oracle evaluator | Environment oracle called by learner/runner | Real environments have no oracle, and evaluation data must not affect action selection. |
| V1 objective | Cumulative regret | One `objective` flag for cumulative and simple regret | Pure exploration requires a different allocation algorithm, not different accounting. |
| POMIS | Deferred to a separate proposal | Default action reducer in v1 | POMIS requires multivariate interventions and a confounded/non-manipulable graph story absent from v1. |
| Existing packages | Audit for parity and possible attributed ports | Claim first-mover status or add a runtime dependency | Current competitors overlap substantially, but native pgmpy integration remains useful. |

## Testing Strategy

### Contract and validation tests

- `Intervention` is hashable, mapping-order independent, round-trips through `as_dict`, and represents `do()`
  distinctly.
- Problem validation rejects invalid states, reward interventions, missing utility values, malformed parent kernels,
  and missing parent feedback.
- `CausalBanditProblem.from_model` returns no reference to the source model or its CPDs.
- A learner cannot access environment model or oracle fields through any declared interface.

### Environment tests

- On a small BN, a forced intervention fixes its variable and produces the correct downstream distribution.
- Outcome states are converted through `reward_values`, including non-numeric state names.
- Equal seeds reproduce equal sequences across environments; successive calls do not restart the RNG.
- Callback interactions are validated against the problem feedback and reward contracts.

### Learner tests

- `EmpiricalUCB` updates only the selected action and matches hand-computed indices.
- `CUCB` matches hand-computed parent counts, confidence bounds, and weighted action scores on a two-parent toy graph.
- A single parent-configuration observation updates the scores of every action with mass on that configuration.
- Initialization, unobserved configurations, deterministic tie-breaking, and `recommend()` are covered explicitly.
- Invalid or mismatched interactions fail before mutating learner state.

### Evaluation tests

- Cumulative pseudo-regret and recommendation regret match exact values for a handcrafted history.
- The evaluator rejects missing or extra oracle action means.
- Running a callback environment without oracle means succeeds and returns a normal `BanditHistory`.

### Statistical benchmarks

Aggregate performance is not asserted from one stochastic trajectory. Slow benchmark tests use fixed seed sets and
report means and uncertainty across repetitions. Their purpose is to detect large regressions and reproduce qualitative
information-sharing behavior, not to encode a universal claim that C-UCB beats UCB on every finite sample path.

The initial benchmark suite contains:

- A parallel-bandit-style graph with known parent distributions.
- A small reward-parent graph where multiple actions share parent configurations.
- A fitted Sachs BN simulator, clearly labeled synthetic rather than real online evaluation.

Docstring examples run under the repository's existing doctest job. The smallest relevant pytest target and
`pre-commit` are run before merging.

## User Journeys

### 1. Simulated C-UCB benchmark without model leakage

```python
from pgmpy.causal_bandits import (
    CUCB,
    CausalBanditProblem,
    DiscreteBayesianNetworkEnvironment,
    atomic_interventions,
    evaluate_regret,
    expected_rewards_from_model,
    run_bandit,
)

actions = atomic_interventions(model, variables=["X1", "X2"], include_observe=True)
problem = CausalBanditProblem.from_model(
    model,
    actions=actions,
    reward_node="Y",
    reward_values={"failure": 0.0, "success": 1.0},
)

environment = DiscreteBayesianNetworkEnvironment(model, problem, seed=42)
learner = CUCB(problem, random_state=42)  # learner receives no CPDs
history = run_bandit(learner, environment, horizon=5_000)

# Oracle means are computed only for evaluation, after the learner has finished.
oracle_means = expected_rewards_from_model(model, problem)
result = evaluate_regret(history, expected_rewards=oracle_means)
```

### 2. Structure-aware learner against the structure-blind baseline

```python
from pgmpy.causal_bandits import CUCB, EmpiricalUCB

histories = {}
for learner_cls in (CUCB, EmpiricalUCB):
    environment = DiscreteBayesianNetworkEnvironment(model, problem, seed=42)
    learner = learner_cls(problem, random_state=42)
    histories[learner_cls.__name__] = run_bandit(learner, environment, horizon=5_000)
```

Benchmark reports compare aggregate pseudo-regret across a fixed seed set. They do not assert ordering from these two
individual histories.

### 3. User-owned physical experiment loop

```python
learner = CUCB(problem, random_state=42)

for _ in range(100):
    action = learner.ask()
    interaction = run_external_experiment(action)
    learner.tell(interaction)

recommended_action = learner.recommend()
```

No environment or oracle is required when the user owns experiment execution.

### 4. Sachs fitted-simulator demonstration

```python
from pgmpy.example_models import load_model

model = load_model("bnlearn/sachs")
actions = atomic_interventions(
    model,
    variables=["PKA", "PKC", "Mek"],
    include_observe=True,
)
problem = CausalBanditProblem.from_model(
    model,
    actions=actions,
    reward_node="Akt",
    reward_values={"LOW": 0.0, "AVG": 0.5, "HIGH": 1.0},
)
```

This demonstrates the API on a familiar parameterized network. It does not claim an online evaluation on the real
Sachs interventional dataset. Supporting historical interventional data requires logged-action metadata, coverage or
propensity assumptions, and a separate off-policy evaluation design.

## Deferred Work

### POMIS and structural action reduction

A POMIS proposal must address all of the following together:

- POMIS arms are assignments to possibly multivariate intervention sets, not atomic actions.
- The original result assumes all non-reward variables are manipulable.
- Restricting feasible actions requires the non-manipulable extension and latent projection; filtering unconstrained
  POMISs is not sound.
- POMIS is most informative on causal diagrams with bidirected confounding, while pgmpy's `ADMG` is a structural graph
  and not a parameterized simulation model.
- `get_district` is available on `_CoreGraph` classes but not `DAG`; a structural implementation must either normalize
  DAGs to an ADMG representation or use a singleton-district adapter.

The pure POMIS graph functions may still live in `pgmpy.causal_bandits.structural` or a future shared
`causal_decision` package. They should be tested against the maintained brute-force oracle and published fixtures from
`SCMMAB-NIPS2018`. Integration with a learner follows only after the action and executable-SCM contracts exist.

### Lattimore simple-regret algorithms

The parallel-bandit and general-graph algorithms should be implemented as dedicated pure-exploration learners. The
general algorithm requires a sampling-design optimization over `eta` and a custom cross-action importance-weighted
estimator. `BayesianModelSampling.likelihood_weighted_sample` conditions samples on evidence and is not that estimator.
The truncated estimator is intentionally biased, so its tests must verify the bias/variance construction and resulting
formula rather than assert unbiasedness.

### MABUC

MABUC needs a separate contextual/counterfactual contract: observational initialization, a pre-action intention value,
intention-specific randomization, binary rewards, and an ETT-based decision target. pgmpy currently has no general ETT
query implementation, and an ADMG alone cannot generate interactions. It must not be represented as
`CounterfactualEstimator + ThompsonSampling` on the v1 environment.

### Causal Thompson Sampling

C-TS can reuse the v1 problem and interaction contracts, but it needs parent-configuration posterior state and a
declared Beta-Bernoulli or Gaussian model. It should be added as an algorithm class after C-UCB, not as a generic policy
over scalar estimates.

## Implementation Sequence

1. Add the immutable action, interaction, and problem contracts with validation tests.
2. Add the tabular and callback environments, `ask()` / `tell()` base contract, and history runner.
3. Add `EmpiricalUCB` and exact deterministic unit tests.
4. Add `CUCB`, parent-kernel construction, and formula-level tests against the paper.
5. Add oracle-separated evaluation and synthetic benchmark factories.
6. Add documentation, the fitted Sachs demonstration, slow aggregate benchmark checks, and an ecosystem comparison
   note covering `causalrl` and reference repositories.

Each step is independently reviewable. POMIS, Lattimore simple-regret algorithms, MABUC, and C-TS begin only after v1
contracts are stable and each receives its own algorithm-specific design review.

## Acceptance Criteria

- No public learner constructor accepts a parameterized Bayesian network or oracle action means.
- V1 examples cannot obtain the reward CPD through the problem object.
- C-UCB action scores and updates match hand calculations and the cited algorithm.
- The reward and feedback assumptions are validated at construction and update time.
- Real callback use works without an oracle or regret calculation.
- Simulation is reproducible as a sequence and meets an agreed per-round benchmark threshold.
- Statistical comparison uses repeated seeded runs and reports uncertainty.
- The fitted Sachs example is labeled as simulation, not real online evidence.
- Relevant pytest targets and pre-commit checks pass.

## References

- Bareinboim, Forney, and Pearl. *Bandits with Unobserved Confounders: A Causal Approach*. NeurIPS 2015.
- Lattimore, Lattimore, and Reid. *Causal Bandits: Learning Good Interventions via Causal Inference*. NeurIPS 2016.
- Lee and Bareinboim. *Structural Causal Bandits: Where to Intervene?*. NeurIPS 2018.
- Lee and Bareinboim. *Structural Causal Bandits with Non-Manipulable Variables*. AAAI 2019.
- Lu, Meisami, Tewari, and Yan. *Regret Analysis of Bandit Problems with Causal Background Knowledge*. UAI 2020.
- `sanghack81/SCMMAB-NIPS2018`: https://github.com/sanghack81/SCMMAB-NIPS2018
- `causalrl`: https://github.com/raphaelrrcoelho/causalrl
