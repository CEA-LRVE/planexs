"""`axClient` and `fitModel()` used directly (FR16, ADR-0005).

Demonstrates the two escape hatches `BayesianOptimizer` exposes for anything
its high-level facade does not cover (design philosophy #2, section 1.3):

- `axClient`: the live `ax.api.client.Client` instance backing this optimizer.
  Section 4.7 lists several real Ax features planexs' facade does not wrap yet
  (closed-loop `run_trials`, `attach_baseline`, `StorageConfig`, ...); any of
  them is reachable through this property. This script calls
  `get_next_trials` directly on it, chosen because its signature is
  well-confirmed against Ax's own API reference, so this example stays
  correct and runnable regardless of which section-4.7 feature Phase 2/3
  eventually wrap natively.
- `fitModel()`: the current fitted BoTorch surrogate model, for writing and
  optimizing a custom acquisition function directly with BoTorch and feeding
  the result back through `tell()`, without needing Phase 2's `alpha` dial.

Run with:
    uv run python examples/escape_hatch.py
"""

from __future__ import annotations

import logging

from planexs import BayesianOptimizer, Real
from planexs.logging_config import configureLogging

logger = logging.getLogger(__name__)


def sphere(x: float, y: float) -> float:
    """A toy objective (the classic sphere function) so this script has
    something cheap and deterministic to optimize.
    """
    return x**2 + y**2


def main() -> None:
    """Runs the example: reaches Ax's client and the surrogate model directly (ADR-0005)."""
    configureLogging(level=logging.INFO)

    bo = BayesianOptimizer(
        parameters=[Real("x", -5.0, 5.0), Real("y", -5.0, 5.0)],
        objective="minimize",
        name="escape-hatch-demo",
    )

    # A handful of trials, told through planexs' own ask()/tell(), so Ax has
    # enough data for both escape hatches below to have something to show.
    for point in bo.ask(n_points=5):
        bo.tell(point, {"objective": sphere(**point)})
    print(f"Told {bo.n_trials} trial(s) via planexs' own ask()/tell().")

    # --- axClient: escape hatch #1 ------------------------------------------
    # Anything reached through axClient is outside what planexs validates or
    # keeps in sync with its own trial history (ADR-0005's own caveat).
    # get_next_trials, like BayesianOptimizer.ask(), attaches a new RUNNING
    # trial in Ax's own bookkeeping; a real caller mixing axClient calls with
    # planexs' own ask()/tell() should eventually complete or abandon it, the
    # same way planexs' own ask()/tell() pairing does internally
    # (AxBackend.attachTrial).
    ax_suggestion = bo.axClient.get_next_trials(max_trials=1)
    print(f"Ax suggested directly via axClient.get_next_trials(): {ax_suggestion}")

    # --- fitModel(): escape hatch #2 ----------------------------------------
    try:
        model = bo.fitModel()
        train_inputs = model.train_inputs[0]
        print(
            f"Fitted BoTorch model (via fitModel()): {type(model).__name__}, "
            f"trained on {train_inputs.shape[0]} point(s) in "
            f"{train_inputs.shape[-1]} dimension(s)."
        )
        print(
            "From here, a caller can build any botorch.acquisition.* "
            "AcquisitionFunction against `model` directly, run optimize_acqf "
            "themselves, and feed the winning point back through bo.tell() "
            "-- exactly the kind of power-user path design philosophy #2 "
            "(section 1.3) says the facade should never block."
        )
    except RuntimeError as error:
        # fitModel() walks part of Ax's internal object graph (see its own
        # docstring in AxBackend.py); this is the documented, expected
        # failure mode if that internal shape has moved for the pinned Ax
        # version, or if there are not yet enough trials for a model to exist.
        print(f"fitModel() could not extract a model: {error}")


if __name__ == "__main__":
    main()
