"""A feasibility constraint that is not linear: `NonlinearConstraint` (FR17).

Sizing a pipe: choose a flow rate and a diameter to maximize throughput per unit
of pipe cost, subject to a limit on the pressure drop. The pressure drop grows
with the square of the flow and falls steeply with the diameter, which no linear
constraint can express:

    flow ** 2 / diameter ** 4 <= 2.0

Ax's own `parameter_constraints` are linear only, so planexs enforces this one in
its own acquisition layer (ADR-0006). Demonstrates:

- Declaring the constraint as an expression string, so the study still saves and
  reloads like any other (FR11).
- The automatic switch: `ask()` with no `alpha` would normally use Ax's own
  strategy, which cannot enforce this constraint, so planexs switches to its own
  layer and logs a warning saying so. Passing `alpha` explicitly is the way to
  choose the exploration/exploitation trade-off yourself, and it silences the
  warning.
- That every suggested point satisfies the constraint.

Run with:
    uv run python examples/nonlinear_constraint.py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from planexs import BayesianOptimizer, NonlinearConstraint, Real
from planexs.logging_config import configureLogging

PRESSURE_DROP_LIMIT = 2.0


def throughputPerCost(flow: float, diameter: float) -> float:
    """Stand-in for the real evaluation: how much flow each unit of pipe cost buys."""
    return flow / (5.0 + 2.0 * diameter**2)


def pressureDrop(flow: float, diameter: float) -> float:
    """The pressure drop of the pipe, the quantity the `NonlinearConstraint` limits.

    Args:
        flow: The flow rate.
        diameter: The pipe diameter.

    Returns:
        `flow ** 2 / diameter ** 4`, the same expression the constraint is written with.
    """
    return flow**2 / diameter**4


def main() -> None:
    """Runs the example: a nonlinear constraint, the automatic switch, and a reload."""
    configureLogging(level=logging.INFO)

    bo = BayesianOptimizer(
        parameters=[Real("flow", 1.0, 20.0), Real("diameter", 1.0, 10.0)],
        objective="maximize",
        constraints=[
            NonlinearConstraint("flow ** 2 / diameter ** 4", "<=", PRESSURE_DROP_LIMIT),
        ],
        name="pipe-sizing",
        random_seed=0,
    )

    # 1) No alpha: planexs switches to its own layer and warns (ADR-0006). Look for
    #    the WARNING line in the output below.
    print("ask() with no alpha (watch for the warning):")
    (first,) = bo.ask()
    results = [_evaluate(bo, first)]

    # 2) An explicit alpha: no switch to announce, and a mostly-exploiting trade-off.
    print("\nask(alpha=0.3), no warning:")
    for _ in range(9):
        (candidate,) = bo.ask(alpha=0.3)
        results.append(_evaluate(bo, candidate))

    value, flow, diameter = max(results)
    (pred_best_candidate,) = bo.ask(alpha=0.0)
    pred_best_value, pred_best_flow, pred_best_diameter = _evaluate(bo, pred_best_candidate)
    print(
        f"\nBest: flow={flow:.2f}, diameter={diameter:.2f}, "
        f"throughput per cost={value:.3f}, "
        f"pressure drop={pressureDrop(flow, diameter):.3f} (limit {PRESSURE_DROP_LIMIT})"
        f"\nBest predicted: flow={pred_best_flow:.2f}, diameter={pred_best_diameter:.2f}, "
        f"throughput per cost={pred_best_value:.3f}, "
        f"pressure drop={pressureDrop(pred_best_flow, pred_best_diameter):.3f} "
        f"(limit {PRESSURE_DROP_LIMIT})"
    )

    # 3) The constraint is part of the saved study and is still enforced after a reload.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pipe_sizing.json"
        bo.save(path)
        reloaded = BayesianOptimizer.load(path)
        (after_reload,) = reloaded.ask(alpha=0.3)
        drop = pressureDrop(float(after_reload["flow"]), float(after_reload["diameter"]))
        print(f"After reload, a new suggestion has pressure drop {drop:.3f}.")


def _evaluate(
    bo: BayesianOptimizer, candidate: dict[str, float | int | str | bool]
) -> tuple[float, float, float]:
    """Evaluates and tells one candidate; returns `(value, flow, diameter)`."""
    flow, diameter = float(candidate["flow"]), float(candidate["diameter"])
    drop = pressureDrop(flow, diameter)
    assert drop <= PRESSURE_DROP_LIMIT * (1 + 1e-5), "planexs returned an infeasible point"
    value = throughputPerCost(flow, diameter)
    bo.tell(candidate, {"objective": value})
    print(
        f"  flow={flow:6.2f} diameter={diameter:5.2f}  "
        f"pressure drop={drop:5.3f}  throughput/cost={value:.3f}"
    )
    return value, flow, diameter


if __name__ == "__main__":
    main()
