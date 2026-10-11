"""A physical-experiment campaign: a small, extremely costly budget.

This is section 1.1's flagship scenario: a handful of evaluations total (think
4-5), each one so expensive (on the order of EUR 1,000,000) that no single
suggested point can be "wasted" on uninformed exploration.

Demonstrates:

- `initialization_budget=0` on `AxBackend` (section 4.4): no evaluation is spent
  on a random, space-filling design; the study is seeded entirely through
  `tell()` instead.
- Manual injection of prior data via `tell()` (FR3).
- `ask()` using Ax's own no-manual-tuning default strategy (FR4), called here with
  no `alpha` argument. See `ml_hyperparameter_search.py` for `alpha`-controlled
  exploration/exploitation (FR5).
- `predict()` with calibrated uncertainty (FR7, FR8).
- `parameterImportance()` (FR6).
- `save()`/`load()` (FR11, FR12).

Run with:
    uv run python examples/lab_experiment.py
"""

from __future__ import annotations

import logging
import os

from planexs import AxBackend, BayesianOptimizer, Categorical, Real
from planexs.logging_config import configureLogging

logger = logging.getLogger(__name__)


def runReactorExperiment(temperature: float, pressure: float, catalyst: str) -> float:
    """Stand-in for a real, expensive physical experiment: replace this with
    whatever actually runs your lab's reactor, rig, or instrument, and have it
    return the measured yield (or whatever quantity `objective` represents).
    """
    # A toy response surface, only so this script produces plausible-looking
    # numbers when run standalone; it has no bearing on planexs' own logic.
    catalyst_bonus = {"A": 0.00, "B": 0.05, "C": -0.02}[catalyst]
    return max(
        0.0,
        min(
            1.0,
            0.9
            - ((temperature - 235.0) / 150.0) ** 2
            - ((pressure - 6.0) / 9.0) ** 2
            + catalyst_bonus,
        ),
    )


def main() -> None:
    """Runs the example: a small-budget campaign seeded with prior data through `tell()`."""
    configureLogging(level=logging.INFO)

    # A custom AxBackend, injected rather than left as BayesianOptimizer's own
    # default, specifically to set initialization_budget=0. Note: because the
    # backend is constructed here rather than by BayesianOptimizer itself, the
    # random seed BayesianOptimizer records in its own ReproducibilityMetadata
    # is not necessarily the exact seed this backend ends up using internally;
    # pass the same `random_seed=` to both if exact reproducibility matters.
    backend = AxBackend(initialization_budget=0)

    bo = BayesianOptimizer(
        parameters=[
            Real("temperature", 150.0, 300.0),
            Real("pressure", 1.0, 10.0),
            Categorical("catalyst", ["A", "B", "C"]),
        ],
        objective="maximize",
        name="reactor-yield-campaign",
        backend=backend,
    )

    # Seed the study with runs already carried out before planexs was involved
    # (FR3): a real campaign almost always starts with some prior knowledge.
    known_runs = [
        ({"temperature": 180.0, "pressure": 3.0, "catalyst": "A"}, 0.41),
        ({"temperature": 220.0, "pressure": 5.0, "catalyst": "B"}, 0.53),
    ]
    for parameters, observed_yield in known_runs:
        bo.tell(parameters, {"objective": observed_yield})
    print(f"Seeded with {bo.n_trials} known run(s).")

    # With a handful of extremely costly evaluations left, ask for exactly one
    # carefully chosen point rather than a speculative batch.
    next_point = bo.ask(n_points=1)[0]
    print(f"Next experiment to run: {next_point}")

    observed_yield = runReactorExperiment(**next_point)  # a real experiment, in practice
    recorded_trial = bo.tell(next_point, {"objective": observed_yield})
    print(
        f"Recorded trial #{recorded_trial.trial_index}: observed yield = {observed_yield:.3f}"
    )

    # What does the model now believe about a point we have not run? (FR7, FR8)
    candidate = {"temperature": 250.0, "pressure": 7.0, "catalyst": "C"}
    prediction = bo.predict(candidate)
    print(
        f"Model belief for {candidate}: "
        f"mean={prediction.mean['objective']:.3f}, sem={prediction.sem['objective']:.3f}"
    )

    # Which parameters matter most? (FR6) With only three trials so far, Ax may
    # not yet have enough data for a reliable answer; computeSensitivity() logs
    # a warning and reports {} for a metric it cannot assess, rather than
    # raising, so the rest of the script still runs.
    print(f"Parameter importance: {bo.parameterImportance()}")

    output_dir = os.path.join(os.path.split(__file__)[0], "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "reactor_yield_campaign.json")
    bo.save(output_path)
    print(f"Saved to {output_path}")

    reloaded = BayesianOptimizer.load(output_path)
    n_best = len(reloaded.paretoFront)
    print(f"Reloaded study with {reloaded.n_trials} trial(s) and {n_best} best trial(s).")


if __name__ == "__main__":
    main()
