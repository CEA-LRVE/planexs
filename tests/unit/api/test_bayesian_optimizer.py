"""Unit tests for `BayesianOptimizer`'s facade logic.

Uses `FakeBackend` (an in-memory `OptimizationBackend`) throughout, so these
tests exercise `BayesianOptimizer`'s own logic (constructor-case resolution,
`tell`/`ask` bookkeeping, property behavior, error handling) without needing
a real Ax install.
"""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any

import pytest
from fake_backend import FakeBackend  # local test double, tests/unit/api/fake_backend.py

from planexs import (
    AcquisitionStrategy,
    BayesianOptimizer,
    Categorical,
    LegacyMetric,
    LegacyObjective,
    LegacyOutcomeConstraint,
    LinearConstraint,
    NonlinearConstraint,
    PredictionUnavailableError,
    Real,
    SearchSpace,
)
from planexs.api.BayesianOptimizer import (
    DEFAULT_ALPHA_WHEN_FORCED,
    MIN_TRIALS_FOR_MODEL,
)


def _searchSpace() -> SearchSpace:
    return SearchSpace(parameters=[Real("x", 0.0, 1.0)])


class TestConstructorCaseResolution:
    def test_common_case_builds_implicit_single_metric_objective(self) -> None:
        backend = FakeBackend()
        BayesianOptimizer(
            parameters=[Real("x", 0.0, 1.0)], objective="minimize", backend=backend
        )

        assert backend.created_objective is not None
        assert [m.name for m in backend.created_objective.metrics] == ["objective"]
        assert backend.created_objective.metrics[0].minimize is True

    def test_common_case_maximize(self) -> None:
        backend = FakeBackend()
        BayesianOptimizer(
            parameters=[Real("x", 0.0, 1.0)], objective="maximize", backend=backend
        )

        assert backend.created_objective is not None
        assert backend.created_objective.metrics[0].minimize is False

    def test_common_case_rejects_other_objective_strings(self) -> None:
        with pytest.raises(ValueError, match=r"minimize.*maximize"):
            BayesianOptimizer(
                parameters=[Real("x", 0.0, 1.0)],
                objective="maximise",  # a plausible typo, deliberately not accepted
                backend=FakeBackend(),
            )

    def test_advanced_case_uses_search_space_and_objective_as_is(self) -> None:
        search_space = _searchSpace()
        objective = LegacyObjective(metrics=[LegacyMetric(name="loss", minimize=True)])
        backend = FakeBackend()

        BayesianOptimizer(parameters=search_space, objective=objective, backend=backend)

        assert backend.created_search_space is search_space
        assert backend.created_objective is objective

    def test_advanced_case_rejects_extra_constraints(self) -> None:
        search_space = _searchSpace()
        objective = LegacyObjective(metrics=[LegacyMetric(name="loss", minimize=True)])

        with pytest.raises(ValueError, match="ambiguous"):
            BayesianOptimizer(
                parameters=search_space,
                objective=objective,
                outcome_constraints=[LegacyOutcomeConstraint("loss", 1.0, "<=")],
                backend=FakeBackend(),
            )

    def test_rejects_search_space_with_string_objective(self) -> None:
        with pytest.raises(TypeError):
            BayesianOptimizer(
                parameters=_searchSpace(), objective="minimize", backend=FakeBackend()
            )

    def test_rejects_parameter_list_with_objective_instance(self) -> None:
        with pytest.raises(TypeError):
            BayesianOptimizer(
                parameters=[Real("x", 0.0, 1.0)],
                objective=LegacyObjective(metrics=[LegacyMetric(name="loss", minimize=True)]),
                backend=FakeBackend(),
            )


class TestTellAndAsk:
    def test_tell_returns_trial_with_backend_assigned_index(self) -> None:
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)

        trial = bo.tell({"x": 0.5}, {"objective": 1.0})

        assert trial.trial_index == 0
        assert trial.parameters == {"x": 0.5}
        assert trial.results == {"objective": 1.0}
        assert bo.n_trials == 1

    def test_n_trials_increments_across_multiple_tells(self) -> None:
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)

        bo.tell({"x": 0.1}, {"objective": 1.0})
        bo.tell({"x": 0.2}, {"objective": 2.0})

        assert bo.n_trials == 2

    def test_ask_delegates_to_backend_suggest_default(self) -> None:
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)

        points = bo.ask(n_points=3)

        assert points == [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}]

    def test_ask_rejects_non_positive_n_points(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(ValueError, match="n_points"):
            bo.ask(n_points=0)


class TestPredictAndImportance:
    def test_predict_reshapes_backend_mean_sem_tuples(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())

        prediction = bo.predict({"x": 0.5})

        assert prediction.mean == {"objective": 0.5}
        assert prediction.sem == {"objective": 0.1}
        assert prediction.variance == pytest.approx({"objective": 0.01})

    def test_parameter_importance_delegates_to_backend(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        assert bo.parameterImportance() == {"objective": {"x": 1.0}}


class TestParetoFront:
    def test_single_objective_collapses_to_best_trial(self) -> None:
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)
        bo.tell({"x": 0.5}, {"objective": 1.0})

        front = bo.paretoFront

        assert len(front) == 1
        assert front[0] is backend.attached_trials[-1]

    def test_single_objective_empty_before_any_trial(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        assert bo.paretoFront == []

    def test_multi_objective_uses_pareto_frontier(self) -> None:
        search_space = _searchSpace()
        objective = LegacyObjective(
            metrics=[
                LegacyMetric(name="a", minimize=True),
                LegacyMetric(name="b", minimize=False),
            ]
        )
        backend = FakeBackend()
        bo = BayesianOptimizer(parameters=search_space, objective=objective, backend=backend)

        bo.tell({"x": 0.1}, {"a": 1.0, "b": 2.0})
        bo.tell({"x": 0.2}, {"a": 0.5, "b": 1.0})

        assert bo.paretoFront == backend.attached_trials


class TestEscapeHatches:
    def test_ax_client_raises_for_non_ax_backend(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(TypeError, match="AxBackend"):
            _ = bo.axClient

    def test_fit_model_propagates_backend_error(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(RuntimeError):
            bo.fitModel()


class TestSave:
    def test_save_rejects_non_ax_backend(self) -> None:
        # Fails before any file I/O happens (LegacyStudySnapshot's backend_kind
        # argument is evaluated, and raises, before save() ever calls
        # JsonStudyRepository().save()), so no path/tmp_path is needed here.
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(TypeError, match="AxBackend"):
            bo.save("unused-path.json")


class _CountingBackend(FakeBackend):
    """A `FakeBackend` that counts how often the default strategy is consulted."""

    def __init__(self) -> None:
        super().__init__()
        self.suggest_default_calls = 0

    def suggestDefault(self, n_points: int) -> list[dict[str, float | int | str | bool]]:
        self.suggest_default_calls += 1
        return super().suggestDefault(n_points)


class _SpyStrategy(AcquisitionStrategy):
    """Records how it is called, and returns fixed points."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def buildAcquisitionFunction(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def suggest(
        self, model: Any, objective: Any, search_space: Any, alpha: float, n_points: int,
        seed: int | None = None,
    ) -> list[dict[str, float | int | str | bool]]:  # fmt: skip
        self.calls.append(
            {
                "kind": "suggest",
                "alpha": alpha,
                "n_points": n_points,
                "seed": seed,
                "model": model,
            }
        )
        return [{"x": 0.5} for _ in range(n_points)]

    def suggestSpaceFilling(
        self, search_space: Any, n_points: int, seed: int | None = None
    ) -> list[dict[str, float | int | str | bool]]:
        self.calls.append(
            {
                "kind": "filling",
                "alpha": None,
                "n_points": n_points,
                "seed": seed,
                "model": None,
            }
        )
        return [{"x": 0.25} for _ in range(n_points)]


def _told(bo: BayesianOptimizer, n: int) -> None:
    for i in range(n):
        x = (i + 1) / (n + 1)
        bo.tell({"x": x}, {"objective": (x - 0.4) ** 2})


class TestAskWithAlpha:
    def test_alpha_uses_the_custom_layer_and_never_the_backends_default(self) -> None:
        backend = _CountingBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)
        _told(bo, 4)

        points = bo.ask(n_points=3, alpha=0.5)

        assert backend.suggest_default_calls == 0
        assert len(points) == 3 and all(0.0 <= p["x"] <= 1.0 for p in points)  # type: ignore[operator]
        assert len({p["x"] for p in points}) == 3

    def test_alpha_none_still_delegates_to_the_backend(self) -> None:
        backend = _CountingBackend()
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=backend)

        assert bo.ask(n_points=2) == [{"x": 0.0}, {"x": 1.0}]
        assert backend.suggest_default_calls == 1

    @pytest.mark.parametrize("alpha", [-0.1, 1.5, float("nan")])
    def test_rejects_alpha_outside_the_unit_interval(self, alpha: float) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(ValueError, match="alpha"):
            bo.ask(alpha=alpha)

    @pytest.mark.parametrize("alpha", [0.0, 1.0])
    def test_accepts_the_endpoints_of_the_unit_interval(self, alpha: float) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        _told(bo, 3)
        assert len(bo.ask(alpha=alpha)) == 1

    def test_rejects_a_non_positive_batch_size_with_alpha_too(self) -> None:
        bo = BayesianOptimizer(parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend())
        with pytest.raises(ValueError, match="n_points"):
            bo.ask(n_points=0, alpha=0.5)

    def test_passes_alpha_batch_size_and_a_fitted_model_to_the_strategy(self) -> None:
        spy = _SpyStrategy()
        bo = BayesianOptimizer(
            parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend(), acquisition_strategy=spy
        )
        _told(bo, MIN_TRIALS_FOR_MODEL)

        bo.ask(n_points=4, alpha=0.3)

        [call] = spy.calls
        assert (call["kind"], call["alpha"], call["n_points"]) == ("suggest", 0.3, 4)
        assert call["model"].num_outputs == 1

    def test_suggests_space_filling_points_until_there_is_data_for_a_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        spy = _SpyStrategy()
        bo = BayesianOptimizer(
            parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend(), acquisition_strategy=spy
        )
        kinds = []
        with caplog.at_level(logging.INFO, logger="planexs.api.BayesianOptimizer"):
            for _ in range(MIN_TRIALS_FOR_MODEL + 1):
                bo.ask(alpha=0.5)
                kinds.append(spy.calls[-1]["kind"])
                _told(bo, 1)

        assert kinds == ["filling"] * MIN_TRIALS_FOR_MODEL + ["suggest"]
        assert "space-filling" in caplog.text

    def test_seed_is_reproducible_and_differs_between_asks_and_tells(self) -> None:
        def seeds(random_seed: int) -> list[int | None]:
            spy = _SpyStrategy()
            bo = BayesianOptimizer(
                parameters=[Real("x", 0.0, 1.0)], backend=FakeBackend(),
                acquisition_strategy=spy, random_seed=random_seed,
            )  # fmt: skip
            bo.ask(alpha=0.5)
            bo.ask(alpha=0.5)  # same history, second ask
            _told(bo, 1)
            bo.ask(alpha=0.5)  # new history
            return [call["seed"] for call in spy.calls]

        first = seeds(7)

        assert first == seeds(7)
        assert len(set(first)) == 3  # never the same seed twice
        assert first != seeds(8)

    def test_the_real_layer_is_reproducible_for_a_study_seed_and_history(self) -> None:
        def suggestion() -> list[dict[str, float | int | str | bool]]:
            bo = BayesianOptimizer(
                parameters=[Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)],
                backend=FakeBackend(),
                random_seed=3,
            )
            for x, y in [(0.1, 0.9), (0.5, 0.5), (0.9, 0.2), (0.3, 0.3)]:
                bo.tell({"x": x, "y": y}, {"objective": (x - 0.6) ** 2 + y})
            return bo.ask(n_points=2, alpha=0.4)

        assert suggestion() == suggestion()

    def test_warns_that_outcome_constraints_are_not_enforced_by_the_custom_layer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        objective = LegacyObjective(
            [LegacyMetric("loss", minimize=True), LegacyMetric("qps", minimize=False)],
            outcome_constraints=[LegacyOutcomeConstraint("qps", 100.0, ">=")],
        )
        bo = BayesianOptimizer(
            parameters=SearchSpace([Real("x", 0.0, 1.0)]),
            objective=objective,
            backend=FakeBackend(),
        )

        with caplog.at_level(logging.WARNING, logger="planexs.api.BayesianOptimizer"):
            bo.ask(alpha=0.5)
            assert "OutcomeConstraints" in caplog.text
            caplog.clear()
            bo.ask()  # the default path lets Ax enforce them
            assert caplog.text == ""


class TestNonlinearConstraintSwitch:
    """ADR-0006: a `NonlinearConstraint` forces the custom layer, with a warning."""

    @staticmethod
    def _space(*, nonlinear: bool) -> SearchSpace:
        constraints = [NonlinearConstraint("x * y ** 2", "<=", 5.0)] if nonlinear else []
        return SearchSpace(
            [Real("x", 0.1, 10.0), Real("y", 0.1, 10.0)], constraints=constraints
        )

    def _optimizer(
        self, *, nonlinear: bool, backend: FakeBackend | None = None, strategy: Any = None
    ) -> BayesianOptimizer:
        return BayesianOptimizer(
            parameters=self._space(nonlinear=nonlinear),
            objective=LegacyObjective([LegacyMetric("objective", minimize=True)]),
            backend=backend or FakeBackend(),
            acquisition_strategy=strategy,
        )

    @pytest.mark.parametrize(
        ("alpha", "nonlinear", "warns"),
        [
            (None, True, True),  # the one case that switches
            (0.5, True, False),  # alpha given: already the custom layer, nothing to announce
            (None, False, False),  # nothing to enforce: Ax's default, no warning
            (0.5, False, False),
        ],
    )
    def test_the_warning_fires_exactly_when_alpha_is_none_and_a_nonlinear_constraint_exists(
        self,
        caplog: pytest.LogCaptureFixture,
        alpha: float | None,
        nonlinear: bool,
        warns: bool,
    ) -> None:
        bo = self._optimizer(nonlinear=nonlinear)

        with caplog.at_level(logging.WARNING, logger="planexs"):
            bo.ask(alpha=alpha)

        switch_warnings = [
            r for r in caplog.records if "NonlinearConstraint" in r.getMessage()
        ]
        assert len(switch_warnings) == (1 if warns else 0)
        if warns:
            assert switch_warnings[0].levelno == logging.WARNING
            assert "DEFAULT_ALPHA_WHEN_FORCED" in switch_warnings[0].getMessage()

    def test_a_linear_constraint_alone_does_not_switch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _CountingBackend()
        space = SearchSpace(
            [Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)],
            constraints=[LinearConstraint({"x": 1.0, "y": 1.0}, 1.0, "<=")],
        )
        bo = BayesianOptimizer(
            parameters=space,
            objective=LegacyObjective([LegacyMetric("objective", minimize=True)]),
            backend=backend,
        )

        with caplog.at_level(logging.WARNING, logger="planexs"):
            bo.ask()

        assert backend.suggest_default_calls == 1 and caplog.text == ""

    def test_every_forced_call_warns_again(self, caplog: pytest.LogCaptureFixture) -> None:
        bo = self._optimizer(nonlinear=True)
        with caplog.at_level(logging.WARNING, logger="planexs"):
            bo.ask()
            bo.ask()
        assert len([r for r in caplog.records if "NonlinearConstraint" in r.getMessage()]) == 2

    def test_the_switch_uses_the_named_default_alpha_and_never_asks_the_backend(self) -> None:
        backend, spy = _CountingBackend(), _SpyStrategy()
        bo = self._optimizer(nonlinear=True, backend=backend, strategy=spy)
        _told_2d(bo, MIN_TRIALS_FOR_MODEL)

        bo.ask()

        assert DEFAULT_ALPHA_WHEN_FORCED == 0.0
        assert spy.calls[-1]["alpha"] == DEFAULT_ALPHA_WHEN_FORCED
        assert backend.suggest_default_calls == 0

    def test_an_explicit_alpha_is_left_alone(self) -> None:
        spy = _SpyStrategy()
        bo = self._optimizer(nonlinear=True, strategy=spy)
        _told_2d(bo, MIN_TRIALS_FOR_MODEL)

        bo.ask(alpha=0.8)

        assert spy.calls[-1]["alpha"] == 0.8

    @pytest.mark.parametrize("n_told", [0, 1, 4])
    def test_suggestions_satisfy_the_constraint_with_or_without_data(
        self, n_told: int
    ) -> None:
        bo = self._optimizer(nonlinear=True)
        _told_2d(bo, n_told)

        for point in bo.ask(n_points=4):
            assert point["x"] * point["y"] ** 2 <= 5.0 * (1 + 1e-5)  # type: ignore[operator]


def _told_2d(bo: BayesianOptimizer, n: int) -> None:
    for i in range(n):
        x, y = 0.5 + i, 1.0 + 0.3 * i  # x * y ** 2 stays below 5
        bo.tell({"x": x, "y": y}, {"objective": math.hypot(x - 2.0, y - 1.5)})


class TestEqualityConstraintSwitch:
    """An equality `LinearConstraint` is something Ax cannot enforce either (ADR-0008)."""

    @staticmethod
    def _mixture(
        backend: FakeBackend | None = None, strategy: Any = None
    ) -> BayesianOptimizer:
        return BayesianOptimizer(
            parameters=[Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")],
            backend=backend or FakeBackend(),
            acquisition_strategy=strategy,
        )

    def test_a_plain_ask_switches_layer_and_warns_naming_the_equality(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _CountingBackend()
        bo = self._mixture(backend)

        with caplog.at_level(logging.WARNING, logger="planexs"):
            bo.ask()

        [record] = [r for r in caplog.records if "equality LinearConstraint" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "DEFAULT_ALPHA_WHEN_FORCED" in record.getMessage()
        assert backend.suggest_default_calls == 0

    def test_an_explicit_alpha_is_left_alone_and_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bo = self._mixture()
        with caplog.at_level(logging.WARNING, logger="planexs"):
            bo.ask(alpha=0.4)
        assert caplog.text == ""

    def test_suggestions_meet_the_equality_with_or_without_data(self) -> None:
        for n_told in (0, 1, 4):
            bo = self._mixture()
            for i in range(n_told):
                bo.tell(
                    {"a": 0.2 + 0.1 * i, "b": 0.3, "c": 0.5 - 0.1 * i}, {"objective": float(i)}
                )
            for point in bo.ask(n_points=3):
                assert point["a"] + point["b"] + point["c"] == pytest.approx(1.0, abs=1e-5)  # type: ignore[operator]

    def test_the_forced_alpha_is_the_named_default(self) -> None:
        spy = _SpyStrategy()
        bo = self._mixture(strategy=spy)
        bo.tell({"a": 0.2, "b": 0.3, "c": 0.5}, {"objective": 1.0})
        bo.tell({"a": 0.5, "b": 0.3, "c": 0.2}, {"objective": 2.0})

        bo.ask()

        assert spy.calls[-1]["alpha"] == DEFAULT_ALPHA_WHEN_FORCED

    def test_both_kinds_are_named_in_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bo = BayesianOptimizer(
            parameters=[Real("a", 0.1, 1.0), Real("b", 0.1, 1.0), Real("c", 0.1, 1.0)],
            constraints=[
                LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "="),
                NonlinearConstraint("a * b", ">=", 0.01),
            ],
            backend=FakeBackend(),
        )
        with caplog.at_level(logging.WARNING, logger="planexs"):
            bo.ask()
        [record] = [r for r in caplog.records if "DEFAULT_ALPHA_WHEN_FORCED" in r.getMessage()]
        # named once each, in the order they were declared
        assert "an equality LinearConstraint and a NonlinearConstraint" in record.getMessage()


class _NoModelBackend(FakeBackend):
    """A backend that, like Ax early in a study, has no model to predict with."""

    def __init__(self) -> None:
        super().__init__()
        self.predict_calls = 0

    def predict(
        self, x: dict[str, float | int | str | bool]
    ) -> dict[str, tuple[float, float]]:
        self.predict_calls += 1
        raise PredictionUnavailableError("no model yet")


def _countFits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    module = importlib.import_module("planexs.api.BayesianOptimizer")

    fits: list[int] = []
    original = module.buildSurrogateModel

    def counting(*args: Any, **kwargs: Any) -> Any:
        fits.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "buildSurrogateModel", counting)
    return fits


class TestPredictWithoutABackendModel:
    """`predict()` falls back to planexs' own surrogate while the backend has none."""

    @staticmethod
    def _optimizer(backend: FakeBackend | None = None) -> BayesianOptimizer:
        return BayesianOptimizer(
            parameters=[Real("x", 0.0, 10.0)], backend=backend or _NoModelBackend()
        )

    @staticmethod
    def _tell(bo: BayesianOptimizer, points: list[float]) -> None:
        for x in points:
            bo.tell({"x": x}, {"objective": (x - 4.0) ** 2})

    def test_predicts_from_the_trial_history(self) -> None:
        bo = self._optimizer()
        self._tell(bo, [0.0, 2.0, 4.0, 6.0])

        near = bo.predict({"x": 4.0})
        far = bo.predict({"x": 50.0 / 5})  # 10.0, the unexplored end

        assert near.mean["objective"] == pytest.approx(0.0, abs=2.0)
        assert far.sem["objective"] > near.sem["objective"] >= 0.0
        assert near.variance["objective"] == pytest.approx(near.sem["objective"] ** 2)

    def test_the_backends_own_model_is_used_whenever_it_has_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fits = _countFits(monkeypatch)
        bo = self._optimizer(FakeBackend())  # FakeBackend.predict answers (0.5, 0.1)
        self._tell(bo, [0.0, 2.0, 4.0])

        prediction = bo.predict({"x": 3.0})

        assert (prediction.mean["objective"], prediction.sem["objective"]) == (0.5, 0.1)
        assert fits == []

    def test_needs_at_least_two_trials(self) -> None:
        bo = self._optimizer()
        with pytest.raises(PredictionUnavailableError, match=r"at least 2.*0 told"):
            bo.predict({"x": 1.0})
        self._tell(bo, [1.0])
        with pytest.raises(PredictionUnavailableError, match=r"at least 2.*1 told"):
            bo.predict({"x": 1.0})

    def test_the_failure_is_a_runtime_error_callers_can_catch_generically(self) -> None:
        with pytest.raises(RuntimeError):
            self._optimizer().predict({"x": 1.0})

    def test_refits_only_when_the_history_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fits = _countFits(monkeypatch)
        bo = self._optimizer()
        self._tell(bo, [0.0, 2.0, 4.0])

        first = bo.predict({"x": 1.0})
        bo.predict({"x": 3.0})
        bo.predict({"x": 5.0})
        assert len(fits) == 1  # three predictions, one fit

        bo.tell({"x": 8.0}, {"objective": 16.0})
        after = bo.predict({"x": 1.0})
        assert len(fits) == 2
        assert after != first

    def test_logs_once_per_fit_that_it_is_using_its_own_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bo = self._optimizer()
        self._tell(bo, [0.0, 2.0, 4.0])

        with caplog.at_level(logging.INFO, logger="planexs.api.BayesianOptimizer"):
            bo.predict({"x": 1.0})
            bo.predict({"x": 2.0})

        assert caplog.text.count("predicting with planexs' own surrogate") == 1

    def test_every_metric_of_a_multi_objective_study_is_predicted(self) -> None:
        objective = LegacyObjective(
            [LegacyMetric("cost", minimize=True), LegacyMetric("util", minimize=False)]
        )
        bo = BayesianOptimizer(
            parameters=SearchSpace([Real("x", 0.0, 10.0)]),
            objective=objective,
            backend=_NoModelBackend(),
        )
        for x in (0.0, 3.0, 6.0, 9.0):
            bo.tell({"x": x}, {"cost": x, "util": 100.0 - x})

        prediction = bo.predict({"x": 6.0})

        assert set(prediction.mean) == {"cost", "util"}
        assert prediction.mean["cost"] == pytest.approx(6.0, abs=1.5)
        assert prediction.mean["util"] == pytest.approx(94.0, abs=1.5)

    def test_a_conditional_parameter_may_be_left_out_of_the_point(self) -> None:
        bo = BayesianOptimizer(
            parameters=[
                Categorical("kind", ["a", "b"], dependent_parameters={"a": ["size"]}),
                Real("size", 0.0, 1.0),
            ],
            backend=_NoModelBackend(),
        )
        bo.tell({"kind": "a", "size": 0.2}, {"objective": 1.0})
        bo.tell({"kind": "b"}, {"objective": 2.0})
        bo.tell({"kind": "a", "size": 0.8}, {"objective": 3.0})

        assert bo.predict({"kind": "b"}).sem["objective"] >= 0.0
