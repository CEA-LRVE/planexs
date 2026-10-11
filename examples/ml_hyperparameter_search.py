"""Hyperparameter search for a model, with the exploration/exploitation dial.

This is the opposite of `lab_experiment.py`: evaluations are cheap enough to run
several in parallel (a batch per round), and there are many more of them, so the
interesting question is how to *spend* them. Demonstrates:

- `ask(n_points=..., alpha=...)` (FR5, FR10): a batch of candidates chosen
  jointly, where `alpha` is the exploration/exploitation trade-off. Here it is
  lowered round by round, one common pattern. The best setting is problem
  dependent: measure before trusting any schedule, including this one. On this
  toy surface, low `alpha` worked best, and `ask()` without `alpha` (Ax's own
  strategy) is a strong baseline to compare against.
- A mixed search space: a log-scaled real (learning rate), an integer (depth),
  an ordered choice (batch size), a categorical (optimizer) and a *conditional*
  parameter (momentum only exists for SGD, through `dependent_parameters`, FR14).
- `ask()` before there is enough data for a model: the first rounds are
  space-filling, then the surrogate model takes over automatically.

Run with:
    uv run python examples/ml_hyperparameter_search.py
"""

from __future__ import annotations

import logging
import math

from planexs import BayesianOptimizer, Categorical, Choice, Integer, Real
from planexs.logging_config import configureLogging

N_ROUNDS = 5
BATCH_SIZE = 3
ALPHA_FIRST_ROUND = 0.3  # some exploration in the first model-based round
ALPHA_LAST_ROUND = 0.0  # then pure exploitation of what the model has learned


def trainAndEvaluate(parameters: dict[str, float | int | str | bool]) -> float:
    """Stand-in for training a model and returning its validation loss: replace
    this with your own training run. A toy response surface, only so this script
    produces plausible numbers when run standalone.
    """
    optimizer = parameters["optimizer"]
    loss = 0.25
    loss += 0.4 * (math.log10(float(parameters["learning_rate"])) + 2.5) ** 2  # best near 3e-3
    loss += 0.08 * abs(int(parameters["num_layers"]) - 4)
    loss += 2.0 * (float(parameters["dropout"]) - 0.2) ** 2
    loss += {16: 0.06, 32: 0.02, 64: 0.0, 128: 0.05}[int(parameters["batch_size"])]
    loss += {"adam": 0.0, "rmsprop": 0.05, "sgd": 0.12}[str(optimizer)]
    if optimizer == "sgd":
        # momentum only exists for SGD, so it is only present in the parameters then
        loss -= 0.10 * float(parameters["momentum"])
    return loss


def main() -> None:
    """Runs the example: batches of hyperparameters, with `alpha` lowered round by round."""
    configureLogging(level=logging.INFO)

    bo = BayesianOptimizer(
        parameters=[
            Real("learning_rate", 1e-5, 1e-1, scaling="log"),
            Integer("num_layers", 1, 8),
            Real("dropout", 0.0, 0.6),
            Choice("batch_size", [16, 32, 64, 128], parameter_type="int"),
            Categorical(
                "optimizer",
                ["adam", "sgd", "rmsprop"],
                dependent_parameters={"sgd": ["momentum"]},
            ),
            Real("momentum", 0.0, 0.99),
        ],
        objective="minimize",
        name="hyperparameter-search",
        random_seed=0,
    )

    best_loss = math.inf
    for round_index in range(N_ROUNDS):
        progress = round_index / (N_ROUNDS - 1)
        alpha = ALPHA_FIRST_ROUND + progress * (ALPHA_LAST_ROUND - ALPHA_FIRST_ROUND)

        # A batch: these candidates are chosen together, so they complement each
        # other rather than being three copies of one guess (FR10).
        batch = bo.ask(n_points=BATCH_SIZE, alpha=alpha)
        print(f"\nRound {round_index + 1}/{N_ROUNDS}  (alpha={alpha:.2f})")

        # In practice these would run in parallel, on separate workers.
        for candidate in batch:
            validation_loss = trainAndEvaluate(candidate)
            bo.tell(candidate, {"objective": validation_loss})
            best_loss = min(best_loss, validation_loss)
            print(f"  loss={validation_loss:.3f}  {_describe(candidate)}")
        print(f"  best so far: {best_loss:.3f}")

    print(f"\nDone: {bo.n_trials} evaluations, best validation loss {best_loss:.3f}.")


def _describe(parameters: dict[str, float | int | str | bool]) -> str:
    shown = {
        name: (f"{value:.4g}" if isinstance(value, float) else value)
        for name, value in parameters.items()
    }
    return ", ".join(f"{name}={value}" for name, value in shown.items())


if __name__ == "__main__":
    main()
