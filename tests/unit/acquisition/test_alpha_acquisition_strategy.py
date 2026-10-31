"""Unit tests for `AlphaAcquisitionStrategy` (FR5, FR10, FR17)."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import pytest
import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.model import Model

from planexs import (
    AlphaAcquisitionStrategy,
    Boolean,
    Categorical,
    Choice,
    ExplorationExploitationAcquisition,
    Integer,
    LegacyMetric,
    LegacyObjective,
    LegacyTrial,
    LinearConstraint,
    MultiObjectiveExplorationExploitationAcquisition,
    NonlinearConstraint,
    Real,
    SearchSpace,
)
from planexs.models.buildSurrogateModel import buildSurrogateModel
from planexs.models.encodeTrials import encodeTrials
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder

Parameterization = dict[str, float | int | str | bool]


def _strategy() -> AlphaAcquisitionStrategy:
    # Small budgets: these tests check behavior, not solution quality.
    return AlphaAcquisitionStrategy(
        num_restarts=4, raw_samples=128, n_reference_points=128, n_mc_samples=64
    )


def _fit(
    space: SearchSpace,
    objective: LegacyObjective,
    function: Callable[[Parameterization], dict[str, float]],
    n: int = 8,
    strategy: AlphaAcquisitionStrategy | None = None,
) -> tuple[Model, list[LegacyTrial]]:
    strategy = strategy or _strategy()
    points = strategy.suggestSpaceFilling(space, n, seed=0)
    trials = [LegacyTrial(parameters=point, results=function(point)) for point in points]
    train_x, train_y, train_yvar = encodeTrials(SearchSpaceEncoder(space), objective, trials)
    return buildSurrogateModel(train_x, train_y, train_yvar), trials


_MINIMIZE_F = LegacyObjective([LegacyMetric("f", minimize=True)])


def _bowl(p: Parameterization) -> dict[str, float]:
    return {"f": (p["x"] - 0.7) ** 2 + (p["y"] - 0.2) ** 2}  # type: ignore[operator]


@pytest.fixture(scope="module")
def unit_square() -> SearchSpace:
    return SearchSpace([Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)])


@pytest.fixture(scope="module")
def bowl_model(unit_square: SearchSpace):  # type: ignore[no-untyped-def]
    return _fit(unit_square, _MINIMIZE_F, _bowl)


class TestSuggest:
    def test_returns_the_requested_number_of_distinct_valid_points(
        self,
        unit_square,
        bowl_model,  # type: ignore[no-untyped-def]
    ) -> None:
        model, _ = bowl_model

        points = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.5, 4, seed=0)

        assert len(points) == 4
        assert len({tuple(p.items()) for p in points}) == 4
        for point in points:
            assert set(point) == {"x", "y"}
            assert all(0.0 <= point[k] <= 1.0 for k in point)  # type: ignore[operator]

    def test_alpha_zero_lands_nearer_the_optimum_than_alpha_one(
        self,
        unit_square,
        bowl_model,  # type: ignore[no-untyped-def]
    ) -> None:
        model, _trials = bowl_model

        def distance_to_optimum(p: Parameterization) -> float:
            return math.hypot(p["x"] - 0.7, p["y"] - 0.2)  # type: ignore[operator]

        exploit = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.0, 1, seed=1)[0]
        explore = _strategy().suggest(model, _MINIMIZE_F, unit_square, 1.0, 1, seed=1)[0]

        assert distance_to_optimum(exploit) < 0.25
        assert distance_to_optimum(exploit) < distance_to_optimum(explore)

    def test_alpha_one_goes_where_the_data_is_sparse(
        self,
        unit_square,
        bowl_model,  # type: ignore[no-untyped-def]
    ) -> None:
        model, trials = bowl_model

        def gap_to_data(p: Parameterization) -> float:
            return min(
                math.hypot(p["x"] - t.parameters["x"], p["y"] - t.parameters["y"])
                for t in trials
            )  # type: ignore[operator]

        exploit = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.0, 1, seed=1)[0]
        explore = _strategy().suggest(model, _MINIMIZE_F, unit_square, 1.0, 1, seed=1)[0]

        assert gap_to_data(explore) > gap_to_data(exploit)

    def test_maximizing_flips_the_direction_of_exploitation(
        self,
        unit_square,
        bowl_model,  # type: ignore[no-untyped-def]
    ) -> None:
        model, _ = bowl_model
        maximize = LegacyObjective([LegacyMetric("f", minimize=False)])

        best_low = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.0, 1, seed=1)[0]
        best_high = _strategy().suggest(model, maximize, unit_square, 0.0, 1, seed=1)[0]

        # minimizing the bowl heads for (0.7, 0.2); maximizing it for the far corner (0, 1)
        assert math.hypot(best_low["x"] - 0.7, best_low["y"] - 0.2) < 0.25  # type: ignore[operator]
        assert best_high["x"] < 0.35 and best_high["y"] > 0.65  # type: ignore[operator]

    def test_is_reproducible_for_a_seed(self, unit_square, bowl_model) -> None:  # type: ignore[no-untyped-def]
        model, _ = bowl_model
        first = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.5, 3, seed=11)
        second = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.5, 3, seed=11)
        other = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.5, 3, seed=12)

        assert first == second
        assert first != other

    def test_the_first_point_of_a_batch_does_not_depend_on_the_batch_size(
        self,
        unit_square,
        bowl_model,  # type: ignore[no-untyped-def]
    ) -> None:
        model, _ = bowl_model
        single = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.3, 1, seed=5)
        batch = _strategy().suggest(model, _MINIMIZE_F, unit_square, 0.3, 3, seed=5)
        assert batch[0] == single[0]

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
    def test_rejects_alpha_outside_the_unit_interval(
        self,
        unit_square,
        bowl_model,
        alpha,  # type: ignore[no-untyped-def]
    ) -> None:
        with pytest.raises(ValueError, match="alpha"):
            _strategy().suggest(bowl_model[0], _MINIMIZE_F, unit_square, alpha, 1)

    def test_rejects_a_non_positive_batch_size(self, unit_square, bowl_model) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="n_points"):
            _strategy().suggest(bowl_model[0], _MINIMIZE_F, unit_square, 0.5, 0)

    def test_rejects_a_model_whose_outputs_do_not_match_the_objective(
        self,
        unit_square,
        bowl_model,  # type: ignore[no-untyped-def]
    ) -> None:
        two_metrics = LegacyObjective(
            [LegacyMetric("f", minimize=True), LegacyMetric("g", minimize=False)]
        )
        with pytest.raises(ValueError, match=r"1 output.*2 metric"):
            _strategy().suggest(bowl_model[0], two_metrics, unit_square, 0.5, 1)


class TestAcquisitionRouting:
    """The objective, not the caller, decides which acquisition function is used."""

    @staticmethod
    def _model(n_outputs: int):  # type: ignore[no-untyped-def]
        x = torch.rand(6, 2, dtype=torch.double, generator=torch.Generator().manual_seed(0))
        y = torch.stack([x[:, 0] * (j + 1) + x[:, 1] ** 2 for j in range(n_outputs)], dim=-1)
        return buildSurrogateModel(x, y), torch.rand(20, 2, dtype=torch.double)

    def test_one_metric_uses_the_single_objective_function_in_its_direction(self) -> None:
        model, reference = self._model(1)
        for minimize in (True, False):
            acquisition = _strategy().buildAcquisitionFunction(
                model, LegacyObjective([LegacyMetric("f", minimize=minimize)]), 0.3, reference
            )
            assert isinstance(acquisition, ExplorationExploitationAcquisition)
            assert acquisition.minimize is minimize
            assert acquisition.alpha == 0.3

    def test_several_metrics_without_weights_use_the_multi_objective_function(self) -> None:
        model, reference = self._model(2)
        objective = LegacyObjective(
            [LegacyMetric("a", minimize=True), LegacyMetric("b", minimize=False)]
        )

        acquisition = _strategy().buildAcquisitionFunction(model, objective, 0.5, reference)

        assert isinstance(acquisition, MultiObjectiveExplorationExploitationAcquisition)

    def test_weights_scalarize_into_the_single_objective_function(self) -> None:
        model, reference = self._model(2)
        objective = LegacyObjective(
            [LegacyMetric("cost", minimize=True), LegacyMetric("util", minimize=False)],
            weights=[2.0, 1.0],
        )

        acquisition = _strategy().buildAcquisitionFunction(model, objective, 0.5, reference)

        assert isinstance(acquisition, ExplorationExploitationAcquisition)
        transform = acquisition.posterior_transform
        assert isinstance(transform, ScalarizedPosteriorTransform)
        # cost is minimized, so its signed weight is negative: -2 * cost + 1 * util,
        # the same scalarization toAxOptimizationConfig gives Ax.
        assert transform.weights.tolist() == [-2.0, 1.0]
        assert acquisition.minimize is False

    def test_a_model_and_objective_of_different_widths_are_rejected(self) -> None:
        model, reference = self._model(2)
        with pytest.raises(ValueError, match=r"2 output.*1 metric"):
            _strategy().buildAcquisitionFunction(model, _MINIMIZE_F, 0.5, reference)


class TestMultiObjective:
    def test_suggests_valid_points_and_favors_uncovered_parts_of_the_front(
        self, unit_square
    ) -> None:  # type: ignore[no-untyped-def]
        objective = LegacyObjective(
            [LegacyMetric("cost", minimize=True), LegacyMetric("util", minimize=False)]
        )

        def function(p: Parameterization) -> dict[str, float]:
            return {"cost": p["x"] ** 2 + p["y"], "util": math.sqrt(p["x"]) + 0.5 * p["y"]}  # type: ignore[operator]

        model, _ = _fit(unit_square, objective, function)

        points = _strategy().suggest(model, objective, unit_square, 0.2, 3, seed=0)

        assert len(points) == 3
        assert len({tuple(p.items()) for p in points}) == 3
        assert all(0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0 for p in points)  # type: ignore[operator]

    def test_weighted_objective_optimizes_the_weighted_sum(self, unit_square) -> None:  # type: ignore[no-untyped-def]
        # minimize 2*cost - util  with  cost = x, util = y  ->  the corner (x=0, y=1)
        objective = LegacyObjective(
            [LegacyMetric("cost", minimize=True), LegacyMetric("util", minimize=False)],
            weights=[2.0, 1.0],
        )
        model, _ = _fit(unit_square, objective, lambda p: {"cost": p["x"], "util": p["y"]})  # type: ignore[arg-type,return-value]

        point = _strategy().suggest(model, objective, unit_square, 0.0, 1, seed=0)[0]

        assert point["x"] < 0.15 and point["y"] > 0.85  # type: ignore[operator]


class TestMixedSpaces:
    @staticmethod
    def _space() -> SearchSpace:
        return SearchSpace(
            [
                Real("lr", 1e-4, 1e-1, scaling="log"),
                Integer("layers", 1, 6),
                Real("temp", 20.0, 120.0, step_size=5.0),
                Categorical("solvent", ["water", "ethanol", "toluene"]),
                Boolean("flag"),
                Choice("batch", [16, 32, 64, 128], parameter_type="int"),
            ]
        )

    @staticmethod
    def _function(p: Parameterization) -> dict[str, float]:
        return {
            "f": (math.log10(p["lr"]) + 2.5) ** 2  # type: ignore[arg-type]
            + 0.2 * abs(p["layers"] - 3)  # type: ignore[operator]
            + (0.0 if p["solvent"] == "ethanol" else 1.0)
            + (0.0 if p["flag"] else 0.3)
        }

    def test_every_suggestion_is_a_legal_typed_parameterization(self) -> None:
        space = self._space()
        model, _ = _fit(space, _MINIMIZE_F, self._function, n=10)

        points = _strategy().suggest(model, _MINIMIZE_F, space, 0.5, 3, seed=0)

        for p in points:
            assert 1e-4 <= p["lr"] <= 1e-1  # type: ignore[operator]
            assert type(p["layers"]) is int and 1 <= p["layers"] <= 6
            assert 20.0 <= p["temp"] <= 120.0 and (p["temp"] - 20.0) % 5.0 == pytest.approx(
                0.0
            )  # type: ignore[operator]
            assert p["solvent"] in {"water", "ethanol", "toluene"}
            assert type(p["flag"]) is bool
            assert p["batch"] in {16, 32, 64, 128}

    def test_exploitation_finds_the_best_category(self) -> None:
        space = self._space()
        model, _ = _fit(space, _MINIMIZE_F, self._function, n=14)

        point = _strategy().suggest(model, _MINIMIZE_F, space, 0.0, 1, seed=0)[0]

        # Only the dominant effect (1.0 for the solvent) is asserted: recovering the
        # 0.3 effect of `flag` from 14 random points in 6 dimensions is a question
        # of GP accuracy, not of the code under test.
        assert point["solvent"] == "ethanol"

    def test_more_categorical_combinations_than_the_cap_still_work(self) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 1.0)] + [Categorical(f"c{i}", ["a", "b", "c"]) for i in range(4)]
        )
        strategy = AlphaAcquisitionStrategy(
            num_restarts=2, raw_samples=64, n_reference_points=64, n_mc_samples=32,
            max_categorical_combinations=5,
        )  # fmt: skip
        model, _ = _fit(space, _MINIMIZE_F, lambda p: {"f": p["x"]}, n=8, strategy=strategy)  # type: ignore[arg-type,return-value]

        points = strategy.suggest(model, _MINIMIZE_F, space, 0.5, 2, seed=0)

        assert len(points) == 2 and all(f"c{i}" in points[0] for i in range(4))

    def test_conditional_parameters_are_pruned_from_each_suggestion(self) -> None:
        space = SearchSpace(
            [
                Categorical(
                    "model", ["mlp", "cnn"],
                    dependent_parameters={"mlp": ["hidden"], "cnn": ["filters"]},
                ),
                Integer("hidden", 8, 512),
                Integer("filters", 8, 256),
            ]
        )  # fmt: skip
        model, _ = _fit(
            space, _MINIMIZE_F, lambda p: {"f": float(p.get("hidden", 300)) / 100.0}, n=8
        )  # type: ignore[arg-type]

        points = _strategy().suggest(model, _MINIMIZE_F, space, 0.7, 6, seed=0)

        for p in points:
            assert set(p) == (
                {"model", "hidden"} if p["model"] == "mlp" else {"model", "filters"}
            )


class TestConstraints:
    def test_a_nonlinear_constraint_is_honored_and_the_optimum_sits_on_its_boundary(
        self,
    ) -> None:
        space = SearchSpace(
            [Real("x", 0.1, 10.0), Real("y", 0.1, 10.0)],
            constraints=[NonlinearConstraint("x * y ** 2", "<=", 50.0)],
        )
        objective = LegacyObjective([LegacyMetric("f", minimize=False)])
        model, _ = _fit(space, objective, lambda p: {"f": p["x"] + p["y"]})  # type: ignore[operator]

        points = _strategy().suggest(model, objective, space, 0.0, 3, seed=0)

        for p in points:
            assert p["x"] * p["y"] ** 2 <= 50.0 * (1 + 1e-5)  # type: ignore[operator]
        # maximizing x + y under x * y^2 <= 50: the best point is on the boundary
        assert max(p["x"] * p["y"] ** 2 for p in points) > 0.8 * 50.0  # type: ignore[operator]

    def test_a_mixture_constraint_and_a_nonlinear_one_hold_together(self) -> None:
        space = SearchSpace(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[
                LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "="),
                NonlinearConstraint("a * b", ">=", 0.05),
            ],
        )
        model, _ = _fit(space, _MINIMIZE_F, lambda p: {"f": p["c"]})  # type: ignore[arg-type,return-value]

        for p in _strategy().suggest(model, _MINIMIZE_F, space, 0.3, 3, seed=0):
            assert p["a"] + p["b"] + p["c"] == pytest.approx(1.0, abs=1e-5)  # type: ignore[operator]
            assert p["a"] * p["b"] >= 0.05 * (1 - 1e-4)  # type: ignore[operator]

    def test_integer_rounding_never_breaks_a_linear_constraint(self) -> None:
        space = SearchSpace(
            [Integer("a", 0, 10), Integer("b", 0, 10)],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0}, 7.0, "<=")],
        )
        objective = LegacyObjective([LegacyMetric("f", minimize=False)])
        model, _ = _fit(space, objective, lambda p: {"f": p["a"] + 2.0 * p["b"]})  # type: ignore[operator]

        for alpha in (0.0, 0.5, 1.0):
            for p in _strategy().suggest(model, objective, space, alpha, 3, seed=0):
                assert p["a"] + p["b"] <= 7  # type: ignore[operator]

    def test_a_rounding_that_breaks_a_constraint_is_repaired_to_the_nearest_feasible_neighbor(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from planexs.acquisition.EncodedConstraints import EncodedConstraints

        space = SearchSpace(
            [Integer("a", 0, 10), Integer("b", 0, 10)],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0}, 7.0, "<=")],
        )
        encoder = SearchSpaceEncoder(space)
        constraints = EncodedConstraints(space, encoder)
        # In practice the trigger is an optimum sitting on the constraint boundary, such as
        # (3.5, 3.5): rounding both coordinates up gives (4, 4), sum 8. The point below is a
        # deterministic stand-in with the same property: its nearest rounding is infeasible.
        relaxed = torch.tensor([0.36, 0.36], dtype=torch.double)
        assert not bool(constraints.isFeasible(encoder.snapColumns(relaxed)))

        with caplog.at_level(
            logging.WARNING, logger="planexs.acquisition.AlphaAcquisitionStrategy"
        ):
            point = _strategy()._resolveCandidate(None, encoder, constraints, relaxed, seed=0)  # type: ignore[arg-type]

        parameters = encoder.decode(point)
        assert parameters["a"] + parameters["b"] <= 7  # type: ignore[operator]
        # the closest feasible roundings to (3.6, 3.6) are (3, 4) and (4, 3), not (3, 3)
        assert {parameters["a"], parameters["b"]} == {3, 4}
        assert caplog.text == ""  # repaired locally, no random-pool fallback needed

    def test_a_nonlinear_constraint_over_integers_is_honored(self) -> None:
        space = SearchSpace(
            [Integer("a", 1, 10), Integer("b", 1, 10)],
            constraints=[NonlinearConstraint("a * b", "<=", 20.0)],
        )
        objective = LegacyObjective([LegacyMetric("f", minimize=False)])
        model, _ = _fit(space, objective, lambda p: {"f": p["a"] + p["b"]})  # type: ignore[operator]

        for p in _strategy().suggest(model, objective, space, 0.2, 3, seed=0):
            assert p["a"] * p["b"] <= 20  # type: ignore[operator]

    def test_a_constraint_on_a_categorical_space_is_honored(self) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 10.0), Real("y", 0.0, 10.0), Categorical("c", ["u", "v"])],
            constraints=[NonlinearConstraint("x + y", "<=", 6.0)],
        )
        objective = LegacyObjective([LegacyMetric("f", minimize=False)])
        model, _ = _fit(space, objective, lambda p: {"f": p["x"] + p["y"]})  # type: ignore[operator]

        for p in _strategy().suggest(model, objective, space, 0.0, 2, seed=0):
            assert p["x"] + p["y"] <= 6.0 + 1e-5  # type: ignore[operator]

    def test_an_infeasible_optimizer_result_falls_back_to_a_feasible_point_with_a_warning(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        space = SearchSpace(
            [Real("x", 0.1, 10.0), Real("y", 0.1, 10.0)],
            constraints=[NonlinearConstraint("x * y", "<=", 5.0)],
        )
        objective = LegacyObjective([LegacyMetric("f", minimize=False)])
        model, _ = _fit(space, objective, lambda p: {"f": p["x"] + p["y"]})  # type: ignore[operator]
        strategy = _strategy()
        # Simulate a solver that gives up at a corner far outside the constraint.
        monkeypatch.setattr(
            strategy, "_optimizeOne", lambda *a, **k: torch.ones(2, dtype=torch.double)
        )

        with caplog.at_level(
            logging.WARNING, logger="planexs.acquisition.AlphaAcquisitionStrategy"
        ):
            points = strategy.suggest(model, objective, space, 0.0, 2, seed=0)

        assert "does not satisfy the search space's constraints" in caplog.text
        assert all(p["x"] * p["y"] <= 5.0 + 1e-5 for p in points)  # type: ignore[operator]

    def test_infeasible_constraints_raise_instead_of_returning_violating_points(self) -> None:
        space = SearchSpace(
            [Real("x", 0.1, 1.0), Real("y", 0.1, 1.0)],
            constraints=[NonlinearConstraint("x * y", "<=", -1.0)],
        )
        with pytest.raises(RuntimeError, match="no point satisfying"):
            _strategy().suggestSpaceFilling(space, 3, seed=0)


class TestSuggestSpaceFilling:
    def test_returns_distinct_points_and_needs_no_model(self, unit_square) -> None:  # type: ignore[no-untyped-def]
        points = _strategy().suggestSpaceFilling(unit_square, 5, seed=0)
        assert len(points) == 5 and len({tuple(p.items()) for p in points}) == 5

    def test_is_reproducible_for_a_seed(self, unit_square) -> None:  # type: ignore[no-untyped-def]
        strategy = _strategy()
        assert strategy.suggestSpaceFilling(
            unit_square, 4, seed=3
        ) == strategy.suggestSpaceFilling(unit_square, 4, seed=3)

    def test_rejects_a_non_positive_batch_size(self, unit_square) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="n_points"):
            _strategy().suggestSpaceFilling(unit_square, 0)
