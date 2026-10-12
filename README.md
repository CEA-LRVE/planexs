# planexs

[![CI](https://github.com/CEA-LRVE/planexs/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/CEA-LRVE/planexs/actions/workflows/ci.yaml)

A general-purpose, domain-agnostic Bayesian Optimization library for Python, built on top of
[Ax](https://ax.dev) and [BoTorch](https://botorch.org).

`planexs` is equally comfortable with a campaign of 4-5 extremely costly physical-experiment
evaluations, a few hundred ML hyperparameter-tuning trials, or thousands of cheap, fast
evaluations, without the caller having to pre-classify which case they are in. See
[`PROJECT_SPECIFICATION.md`](./PROJECT_SPECIFICATION.md) for the full design.

The name stands for *PLANification EXpérimentale Séquentielle*, the French for sequential experimental design.

## Status

**Phase 2 of 4 (the exploration/exploitation acquisition layer)**: see the
[roadmap](./PROJECT_SPECIFICATION.md#6-roadmap).

Phase 1 implemented:

- The full domain model: `Range`/`Real`/`Integer`, `Choice`/`Categorical`/`Boolean`/`Fixed`/
  `Derived` (with `dependent_parameters`, FR14's definition side), `LinearConstraint` (FR9),
  `Objective` with `N >= 1` metrics and optional weights, `OutcomeConstraint` (FR15).
- `AxBackend`: search space and optimization-config mapping, manual trial injection (FR3),
  `ask()` via Ax's own no-manual-tuning default generation strategy (FR4), Ax's built-in
  sensitivity analysis (FR6), prediction with uncertainty (FR7, FR8), Pareto-frontier/
  best-trial lookup, and the `axClient`/`fitModel()` escape hatches (FR16, ADR-0005).
- `JsonStudyRepository`: save/load a study to/from a single JSON file, with reproducibility
  metadata (FR11, FR12).

Phase 2 adds:

- `ask(n_points=..., alpha=...)` (FR5, FR10): planexs' own acquisition layer, a blend of
  exploitation and exploration weighted by `alpha`, with batches chosen sequentially. It fits
  its own surrogate model from the trial history (ADR-0007) and handles log scales, integer
  grids, ordered and categorical choices, and conditional parameters.
- `NonlinearConstraint` (FR17), e.g. `NonlinearConstraint("var * x ** z", "<=", 50.0)`, written
  as an expression string so a study still saves and reloads. Ax cannot enforce it, so
  `ask()` with no `alpha` switches to the custom layer and logs a warning (ADR-0006). The same
  goes for an equality `LinearConstraint` such as a mixture summing to one, which Ax cannot
  enforce either (ADR-0008).
- `predict()` works from the second trial on: it uses planexs' own surrogate while Ax, still in
  its initial space-filling phase, has no model yet.

Known limits of the custom layer in this phase: it does not enforce `OutcomeConstraint`s (a
warning is logged), it rejects `LinearConstraint`s on log-scaled parameters, and its
`alpha` has not been tuned: on the problems measured so far the best value is
problem-dependent and Ax's own default (`ask()` without `alpha`) is a strong baseline. See
ADR-0006 for the numbers. **Not yet implemented** (Phase 3 onward): the small-sample
robustness work and an acquisition layer that reasons about `dependent_parameters`. See
`PROJECT_SPECIFICATION.md` section 6.

> **Verification status.** Run against the locked versions of the real libraries
> (`pydantic` 2.13.5, `ax-platform` 1.3.1, `botorch` 0.18.1, `torch` 2.14.0): all 425 tests pass
> (399 unit, 26 integration), and `ruff check`, `ruff format --check`, `mypy` (strict) and
> `scripts/check_naming_convention.py` all pass. `ruff` also enforces that every public class,
> method and function has a docstring (see `docs/contributing.md`).

## Install

```bash
uv sync
```

This installs `planexs` and its pinned dependencies (`ax-platform`, `botorch`, `gpytorch`,
`torch`, `pydantic`) into a local virtual environment. For development (tests, lint, type
checking, docs):

```bash
uv sync --extra dev --extra docs
```

## Quickstart

```python
from planexs import BayesianOptimizer, Real, Integer, Categorical, Boolean

bo = BayesianOptimizer(
    parameters=[
        Real("temperature", 20.0, 120.0),
        Integer("num_layers", 1, 8),
        Categorical("solvent", ["water", "ethanol", "toluene"]),
        Boolean("use_catalyst", default=True),
    ],
    objective="maximize",
)

# seed with a point you already ran, if you have one (FR3)
bo.tell(
    {"temperature": 80.0, "num_layers": 3, "solvent": "water", "use_catalyst": True},
    {"objective": 0.62},
)

# ask for the next point, using Ax's own no-manual-tuning default strategy (FR4)
x = bo.ask()[0]
y = run_my_experiment(**x)  # your own code: a real experiment or a training run
bo.tell(x, {"objective": y})

# what does the model currently believe, with uncertainty? (FR7, FR8)
prediction = bo.predict(
    {"temperature": 100.0, "num_layers": 4, "solvent": "ethanol", "use_catalyst": False}
)
print(prediction.mean, prediction.sem)

bo.parameterImportance()  # FR6
bo.save("study.json")  # FR11, FR12
bo_reloaded = BayesianOptimizer.load("study.json")
```

Pass `alpha` to choose the exploration/exploitation trade-off yourself (`0.0` favors the best
predicted point, `1.0` the least-known region), and `n_points` for a batch:

```python
batch = bo.ask(n_points=3, alpha=0.3)
```

See [`examples/`](./examples) for complete, runnable scripts:
[`lab_experiment.py`](./examples/lab_experiment.py) (a small-budget, high-cost-per-evaluation
campaign, seeded via `tell()`), [`escape_hatch.py`](./examples/escape_hatch.py)
(`axClient`/`fitModel()` used directly, ADR-0005),
[`ml_hyperparameter_search.py`](./examples/ml_hyperparameter_search.py) (batches, a mixed
and conditional search space, `alpha` per round) and
[`nonlinear_constraint.py`](./examples/nonlinear_constraint.py) (`NonlinearConstraint` and the
automatic switch, ADR-0006).

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/check_naming_convention.py
```

## Documentation

Architectural decisions are recorded as ADRs under [`docs/adr/`](./docs/adr); see in
particular [ADR-0001](./docs/adr/0001-hybrid-ax-botorch-architecture.md) (why Ax + BoTorch,
not either alone) and [ADR-0005](./docs/adr/0005-escape-hatches-to-ax-and-botorch.md) (why
`axClient`/`fitModel()` exist). The naming-convention rationale (camelCase methods, `CamelCase`
file names) lives in [`docs/contributing.md`](./docs/contributing.md).

## License

TBD (section 1.4): not yet decided whether this stays internal or gets published.
