"""End-to-end integration test: the full `tell`/`ask`/`save`/`load` loop on a
dummy objective, against a real `AxBackend`.

Requires a real `ax-platform` (and therefore `botorch`/`gpytorch`/`torch`)
install; skipped otherwise, so the rest of the suite still runs without the
full ML stack. Exercises real Ax generation-strategy fitting, so it is
slower than the unit tests, which mostly use `FakeBackend` or avoid the Ax
import entirely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ax")

from planexs import BayesianOptimizer, Real


def dummySphere(x: float, y: float) -> float:
    """A cheap, deterministic stand-in for a real objective: the classic
    sphere function, minimized at `(0, 0)`.
    """
    return x**2 + y**2


class TestFullTellAskSaveLoadLoop:
    def test_full_loop(self, tmp_path: Path) -> None:
        bo = BayesianOptimizer(
            parameters=[Real("x", -5.0, 5.0), Real("y", -5.0, 5.0)],
            objective="minimize",
            name="dummy-sphere",
            random_seed=0,
        )

        # Seed with a couple of known points (FR3), bypassing ask().
        bo.tell({"x": 1.0, "y": 1.0}, {"objective": dummySphere(1.0, 1.0)})
        bo.tell({"x": -2.0, "y": 0.5}, {"objective": dummySphere(-2.0, 0.5)})
        assert bo.n_trials == 2

        # Ask for a batch of points via Ax's own no-manual-tuning default
        # strategy (FR4), jointly (FR10), and tell the results back.
        batch = bo.ask(n_points=2)
        assert len(batch) == 2
        for point in batch:
            bo.tell(point, {"objective": dummySphere(point["x"], point["y"])})
        assert bo.n_trials == 4

        # Prediction with calibrated uncertainty for an unevaluated point
        # (FR7, FR8): a plain shape/sanity check, not a numerical-accuracy one.
        prediction = bo.predict({"x": 0.0, "y": 0.0})
        assert "objective" in prediction.mean
        assert prediction.sem["objective"] >= 0.0

        # Single-objective: paretoFront collapses to the one best trial found
        # so far.
        assert len(bo.paretoFront) == 1

        # Save and reload (FR11, FR12): the reloaded study resumes exactly
        # where it left off, not just with the same final numbers.
        path = tmp_path / "dummy_sphere.json"
        bo.save(path)
        reloaded = BayesianOptimizer.load(path)
        assert reloaded.n_trials == bo.n_trials

        # A reloaded study keeps taking ask()/tell() calls like any other.
        next_point = reloaded.ask(n_points=1)[0]
        reloaded.tell(next_point, {"objective": dummySphere(next_point["x"], next_point["y"])})
        assert reloaded.n_trials == 5
