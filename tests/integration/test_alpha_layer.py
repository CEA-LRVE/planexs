"""End-to-end tests of the alpha layer against a real Ax backend (FR5, FR10, FR17).

The unit tests use a fake backend; what only a real Ax can show is whether the
parameterizations the custom layer suggests are ones Ax accepts back through
`tell()`, including for conditional and derived parameters.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from planexs import (
    BayesianOptimizer,
    Categorical,
    Choice,
    Derived,
    Integer,
    LegacyMetric,
    LegacyObjective,
    LinearConstraint,
    NonlinearConstraint,
    Real,
    SearchSpace,
)


def _bowl(p: dict[str, float | int | str | bool]) -> float:
    return (p["x"] - 0.7) ** 2 + (p["y"] - 0.2) ** 2  # type: ignore[operator]


class TestAlphaLoop:
    def test_ax_accepts_every_suggestion_and_the_search_converges(self) -> None:
        bo = BayesianOptimizer(
            parameters=[Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)], random_seed=0
        )
        for point in bo.ask(n_points=4):  # Ax's own default strategy seeds the study
            bo.tell(point, {"objective": _bowl(point)})

        for _ in range(6):
            (point,) = bo.ask(alpha=0.2)
            bo.tell(point, {"objective": _bowl(point)})

        assert bo.n_trials == 10
        best = min(t.results["objective"] for t in bo._trials)  # type: ignore[attr-defined]
        assert best < 0.02  # within ~0.14 of the optimum

    def test_a_batch_is_accepted_and_told_back(self) -> None:
        bo = BayesianOptimizer(
            parameters=[Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)], random_seed=1
        )
        for point in bo.ask(n_points=3, alpha=0.5):  # cold start: space-filling
            bo.tell(point, {"objective": _bowl(point)})
        batch = bo.ask(n_points=3, alpha=0.5)  # now model-based

        assert len({tuple(p.items()) for p in batch}) == 3
        for point in batch:
            bo.tell(point, {"objective": _bowl(point)})
        assert bo.n_trials == 6

    def test_a_mixed_space_of_every_parameter_kind(self) -> None:
        bo = BayesianOptimizer(
            parameters=[
                Real("lr", 1e-4, 1e-1, scaling="log"),
                Integer("layers", 1, 6),
                Real("temp", 20.0, 120.0, step_size=5.0),
                Categorical("solvent", ["water", "ethanol", "toluene"]),
                Choice("batch", [16, 32, 64, 128], parameter_type="int"),
            ],
            random_seed=2,
        )

        def loss(p: dict[str, float | int | str | bool]) -> float:
            return (
                (math.log10(p["lr"]) + 2.5) ** 2
                + abs(p["layers"] - 3)
                + (  # type: ignore[arg-type,operator]
                    0.0 if p["solvent"] == "ethanol" else 1.0
                )
            )

        for _ in range(3):
            for point in bo.ask(n_points=2, alpha=0.5):
                bo.tell(point, {"objective": loss(point)})

        assert bo.n_trials == 6

    def test_conditional_parameters_are_accepted_by_ax(self) -> None:
        # Ax rejects a parameterization that includes a parameter switched off by
        # `dependent_parameters`: the custom layer must leave it out.
        bo = BayesianOptimizer(
            parameters=[
                Categorical(
                    "model",
                    ["mlp", "cnn"],
                    dependent_parameters={"mlp": ["hidden_units"], "cnn": ["num_filters"]},
                ),
                Integer("hidden_units", 8, 512),
                Integer("num_filters", 8, 256),
            ],
            random_seed=3,
        )

        for _ in range(3):
            for point in bo.ask(n_points=3, alpha=0.6):
                assert set(point) == (
                    {"model", "hidden_units"}
                    if point["model"] == "mlp"
                    else {"model", "num_filters"}
                )
                bo.tell(point, {"objective": float(point.get("hidden_units", 100)) / 50.0})  # type: ignore[arg-type]

        assert bo.n_trials == 9

    def test_derived_parameters_are_accepted_by_ax(self) -> None:
        bo = BayesianOptimizer(
            parameters=[
                Real("a", 0.0, 1.0),
                Real("b", 0.0, 1.0),
                Derived("total", "a + b", "float"),
            ],
            random_seed=4,
        )

        for _ in range(2):
            for point in bo.ask(n_points=2, alpha=0.5):
                assert point["total"] == pytest.approx(point["a"] + point["b"])  # type: ignore[operator]
                bo.tell(point, {"objective": point["total"]})  # type: ignore[dict-item]

    def test_multi_objective_suggestions_are_accepted_by_ax(self) -> None:
        objective = LegacyObjective(
            [LegacyMetric("cost", minimize=True), LegacyMetric("util", minimize=False)]
        )
        bo = BayesianOptimizer(
            parameters=SearchSpace([Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)]),
            objective=objective,
            random_seed=5,
        )

        for _ in range(3):
            for point in bo.ask(n_points=2, alpha=0.4):
                x, y = point["x"], point["y"]
                bo.tell(point, {"cost": x**2 + y, "util": math.sqrt(x) + 0.5 * y})  # type: ignore[operator]

        assert bo.n_trials == 6
        assert len(bo.paretoFront) >= 1


class TestNonlinearConstraint:
    @staticmethod
    def _optimizer(seed: int = 0) -> BayesianOptimizer:
        return BayesianOptimizer(
            parameters=[Real("x", 0.1, 10.0), Real("y", 0.1, 10.0)],
            objective="maximize",
            constraints=[NonlinearConstraint("x * y ** 2", "<=", 50.0)],
            random_seed=seed,
        )

    def test_plain_ask_switches_layer_warns_and_still_satisfies_the_constraint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bo = self._optimizer()

        with caplog.at_level(logging.WARNING, logger="planexs"):
            for _ in range(4):
                (point,) = bo.ask()
                assert point["x"] * point["y"] ** 2 <= 50.0 * (1 + 1e-5)  # type: ignore[operator]
                bo.tell(point, {"objective": point["x"] + point["y"]})  # type: ignore[operator]

        assert len([r for r in caplog.records if "NonlinearConstraint" in r.getMessage()]) == 4

    def test_the_optimum_is_found_on_the_constraint_boundary(self) -> None:
        # Maximize x + y subject to x * y**2 <= 50: the optimum (10, sqrt(5)) has the
        # constraint active. A balanced alpha is used on purpose: at alpha near 0 the
        # search can stall on an already-evaluated point (see the summary of ADR-0006's
        # forced default).
        bo = self._optimizer(seed=1)
        for _ in range(10):
            (point,) = bo.ask(alpha=0.5)
            bo.tell(point, {"objective": point["x"] + point["y"]})  # type: ignore[operator]

        best = max(bo._trials, key=lambda t: t.results["objective"])  # type: ignore[attr-defined]
        x, y = best.parameters["x"], best.parameters["y"]
        assert x * y**2 <= 50.0 * (1 + 1e-5)  # type: ignore[operator]
        assert best.results["objective"] > 12.0  # true optimum 12.236

    def test_survives_save_and_load_and_keeps_being_enforced(self, tmp_path: Path) -> None:
        bo = self._optimizer(seed=2)
        for _ in range(3):
            (point,) = bo.ask(alpha=0.3)
            bo.tell(point, {"objective": point["x"] + point["y"]})  # type: ignore[operator]
        path = tmp_path / "study.json"
        bo.save(path)

        reloaded = BayesianOptimizer.load(path)

        constraints = reloaded._search_space.constraints  # type: ignore[attr-defined]
        assert [type(c) for c in constraints] == [NonlinearConstraint]
        assert constraints[0] == NonlinearConstraint("x * y ** 2", "<=", 50.0)
        for point in reloaded.ask(n_points=3, alpha=0.3):
            assert point["x"] * point["y"] ** 2 <= 50.0 * (1 + 1e-5)  # type: ignore[operator]

    def test_a_linear_and_a_nonlinear_constraint_together(self) -> None:
        bo = BayesianOptimizer(
            parameters=[Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[
                LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.5, "<="),
                NonlinearConstraint("a * b", ">=", 0.05),
            ],
            random_seed=3,
        )

        for _ in range(3):
            for point in bo.ask(n_points=2, alpha=0.4):
                assert point["a"] + point["b"] + point["c"] <= 1.5 + 1e-5  # type: ignore[operator]
                assert point["a"] * point["b"] >= 0.05 * (1 - 1e-4)  # type: ignore[operator]
                bo.tell(point, {"objective": point["c"]})  # type: ignore[dict-item]


class TestEqualityConstraint:
    """`LinearConstraint(..., "=")`: Ax cannot build a study with one in its own constraints,
    so planexs keeps it out of Ax and enforces it in its own acquisition layer (ADR-0008).
    """

    @staticmethod
    def _mixture(seed: int = 0) -> BayesianOptimizer:
        return BayesianOptimizer(
            parameters=[Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")],
            random_seed=seed,
        )

    def test_a_study_with_a_mixture_constraint_can_be_built(self) -> None:
        # This used to raise ax's UserInputError ("Expected an inequality").
        assert self._mixture().n_trials == 0

    def test_ax_is_only_given_the_constraints_it_can_enforce(self) -> None:
        bo = BayesianOptimizer(
            parameters=[Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[
                LinearConstraint({"a": 1.0, "b": 1.0}, 1.5, "<="),
                LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "="),
            ],
            random_seed=0,
        )
        in_ax = bo.axClient._experiment.search_space.parameter_constraints
        assert len(in_ax) == 1  # only the inequality

    def test_every_suggestion_sums_to_one_and_ax_accepts_it_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bo = self._mixture(seed=1)

        with caplog.at_level(logging.WARNING, logger="planexs"):
            for i in range(6):
                (point,) = bo.ask() if i == 0 else bo.ask(alpha=0.3)
                assert point["a"] + point["b"] + point["c"] == pytest.approx(1.0, abs=1e-5)  # type: ignore[operator]
                bo.tell(
                    point, {"objective": (point["a"] - 0.2) ** 2 + (point["b"] - 0.5) ** 2}
                )  # type: ignore[operator]

        assert bo.n_trials == 6
        # only the first, alpha-less call announces the switch
        assert (
            len([r for r in caplog.records if "equality LinearConstraint" in r.getMessage()])
            == 1
        )

    def test_the_equality_survives_save_and_load(self, tmp_path: Path) -> None:
        bo = self._mixture(seed=2)
        for _ in range(3):
            (point,) = bo.ask(alpha=0.3)
            bo.tell(point, {"objective": point["a"]})  # type: ignore[dict-item]
        path = tmp_path / "mixture.json"
        bo.save(path)

        reloaded = BayesianOptimizer.load(path)

        constraints = reloaded._search_space.constraints  # type: ignore[attr-defined]
        assert constraints == [LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")]
        for point in reloaded.ask(n_points=3, alpha=0.3):
            assert point["a"] + point["b"] + point["c"] == pytest.approx(1.0, abs=1e-5)  # type: ignore[operator]

    def test_a_mixture_with_a_nonlinear_constraint_too(self) -> None:
        bo = BayesianOptimizer(
            parameters=[Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[
                LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "="),
                NonlinearConstraint("a * b", ">=", 0.05),
            ],
            random_seed=3,
        )
        for _ in range(3):
            for point in bo.ask(n_points=2, alpha=0.4):
                assert point["a"] + point["b"] + point["c"] == pytest.approx(1.0, abs=1e-5)  # type: ignore[operator]
                assert point["a"] * point["b"] >= 0.05 * (1 - 1e-4)  # type: ignore[operator]
                bo.tell(point, {"objective": point["c"]})  # type: ignore[dict-item]
