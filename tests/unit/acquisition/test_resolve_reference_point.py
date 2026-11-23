"""Unit tests for `resolveReferencePoint`, the reference point of the hypervolume from its
configuration (roadmap task P5-05, specification 5.8)."""

from __future__ import annotations

import pytest
import torch
from toy_pareto_surrogates import F1, F2, G, makeParetoSurrogate

from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.ExplicitReferencePointConfig import ExplicitReferencePointConfig
from planexs.acquisition.InferredReferencePointConfig import InferredReferencePointConfig
from planexs.acquisition.outcomeConstraintCallables import outcomeConstraintCallables
from planexs.acquisition.ParetoObjectives import ParetoObjectives
from planexs.acquisition.resolveReferencePoint import resolveReferencePoint
from planexs.core.ConfigError import ConfigError
from planexs.goals.OutcomeConstraint import OutcomeConstraint

COMPONENT = "acquisition 'test_v1'"


def resolve(
    config: ExplicitReferencePointConfig | InferredReferencePointConfig,
    signs: tuple[float, float] = (1.0, 1.0),
    *,
    outcomes: tuple[str, ...] = (F1, F2, G),
    constraints: list[OutcomeConstraint] | None = None,
) -> torch.Tensor:
    """The reference point for the toy surrogate, whose objectives are `f1` and `f2`."""
    model = makeParetoSurrogate(outcomes).model
    baseline = baselinePoints(model)
    # The objectives are in the order of the outputs of the model; the signs are those of
    # (f1, f2), whichever comes first.
    names = tuple(name for name in outcomes if name in (F1, F2))
    objectives = ParetoObjectives(
        names=names,
        indices=tuple(outcomes.index(name) for name in names),
        signs=tuple(signs[(F1, F2).index(name)] for name in names),
        num_outputs=len(outcomes),
    )
    callables = (
        outcomeConstraintCallables(model, baseline, constraints or [], outcomes) or None
    )
    return resolveReferencePoint(
        config,
        objectives,
        model=model,
        baseline=baseline,
        constraints=callables,
        component=COMPONENT,
    )


def predictedFront(signs: tuple[float, float]) -> list[tuple[float, float]]:
    """The front of the posterior means at the observed points, worked out here from the means
    and nothing else, with every objective maximized."""
    model = makeParetoSurrogate((F1, F2, G)).model
    with torch.no_grad():
        mean = model.posterior(baselinePoints(model)).mean
    points = [(float(row[0]) * signs[0], float(row[1]) * signs[1]) for row in mean]
    return [
        point
        for point in points
        if not any(
            other[0] >= point[0] and other[1] >= point[1] and other != point
            for other in points
        )
    ]


class TestAnExplicitReferencePoint:
    def test_the_values_are_taken_as_they_are_for_maximized_objectives(self) -> None:
        point = resolve(ExplicitReferencePointConfig(values={F1: 0.25, F2: -0.5}))

        assert point.tolist() == [0.25, -0.5]

    def test_the_value_of_a_minimized_objective_is_negated(self) -> None:
        # The highest result that still counts is -0.5 in the space where it is maximized.
        point = resolve(
            ExplicitReferencePointConfig(values={F1: 0.5, F2: 0.25}), signs=(-1.0, 1.0)
        )

        assert point.tolist() == [-0.5, 0.25]

    def test_the_values_are_placed_by_name_on_the_objectives_in_the_order_of_the_outputs(
        self,
    ) -> None:
        # The model has f2 before f1, and so do the objectives; the configuration lists f1
        # first.
        point = resolve(
            ExplicitReferencePointConfig(values={F1: 0.2, F2: 0.7}), outcomes=(F2, F1, G)
        )

        assert point.tolist() == [0.7, 0.2]

    def test_it_has_the_precision_of_the_model(self) -> None:
        point = resolve(ExplicitReferencePointConfig(values={F1: 0.0, F2: 0.0}))

        assert point.dtype == baselinePoints(makeParetoSurrogate().model).dtype

    def test_a_missing_objective_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="Missing: \\['f2'\\]") as caught:
            resolve(ExplicitReferencePointConfig(values={F1: 0.0, G: 0.0}))

        assert caught.value.field_path == "reference_point.values"
        assert caught.value.component == COMPONENT

    def test_a_value_for_an_outcome_that_is_not_an_objective_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="Not objectives of the goal: \\['g'\\]"):
            resolve(ExplicitReferencePointConfig(values={F1: 0.0, F2: 0.0, G: 0.0}))


class TestAnInferredReferencePoint:
    @pytest.mark.parametrize("margin", [0.0, 0.1, 0.5])
    @pytest.mark.parametrize("signs", [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)])
    def test_it_is_the_nadir_of_the_predicted_front_moved_out_by_the_margin(
        self, margin: float, signs: tuple[float, float]
    ) -> None:
        front = predictedFront(signs)
        nadir = [min(point[axis] for point in front) for axis in (0, 1)]
        ideal = [max(point[axis] for point in front) for axis in (0, 1)]
        expected = [nadir[axis] - margin * (ideal[axis] - nadir[axis]) for axis in (0, 1)]

        point = resolve(InferredReferencePointConfig(margin=margin), signs=signs)

        assert point.tolist() == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_it_is_worse_than_every_point_of_the_front_in_every_objective(self) -> None:
        # With every objective maximized, so also for a minimized one.
        for signs in [(1.0, 1.0), (-1.0, 1.0)]:
            point = resolve(InferredReferencePointConfig(), signs=signs)

            for member in predictedFront(signs):
                assert member[0] > float(point[0])
                assert member[1] > float(point[1])

    def test_a_larger_margin_puts_it_further_out(self) -> None:
        near = resolve(InferredReferencePointConfig(margin=0.1))
        far = resolve(InferredReferencePointConfig(margin=0.5))

        assert (far < near).all()

    def test_a_margin_of_zero_puts_it_on_the_nadir(self) -> None:
        point = resolve(InferredReferencePointConfig(margin=0.0))

        front = predictedFront((1.0, 1.0))
        assert float(point[0]) == pytest.approx(min(member[0] for member in front))
        assert float(point[1]) == pytest.approx(min(member[1] for member in front))

    def test_it_follows_the_data(self) -> None:
        # Only the margin is configured: the point differs because the data does.
        spread = makeParetoSurrogate((F1, F2), observed_x=(0.55, 0.7, 0.85, 0.95)).model
        near = makeParetoSurrogate((F1, F2), observed_x=(0.6, 0.65, 0.7, 0.75)).model

        def inferred(model: object) -> list[float]:
            baseline = baselinePoints(model)  # type: ignore[arg-type]
            objectives = ParetoObjectives(
                names=(F1, F2), indices=(0, 1), signs=(1.0, 1.0), num_outputs=2
            )
            return resolveReferencePoint(
                InferredReferencePointConfig(),
                objectives,
                model=model,  # type: ignore[arg-type]
                baseline=baseline,
                constraints=None,
                component=COMPONENT,
            ).tolist()

        assert inferred(spread) != pytest.approx(inferred(near))

    def test_the_front_it_reads_leaves_out_the_points_predicted_to_violate_a_constraint(
        self,
    ) -> None:
        constraint = OutcomeConstraint(outcome=G, comparator="<=", bound=0.5)

        free = resolve(InferredReferencePointConfig())
        constrained = resolve(InferredReferencePointConfig(), constraints=[constraint])

        # One feasible point is left on the front, at x = 0.436: the point is at 90 % of it,
        # the margin being taken of its own absolute value.
        assert not torch.allclose(free, constrained)
        with torch.no_grad():
            model = makeParetoSurrogate((F1, F2, G)).model
            mean = model.posterior(baselinePoints(model)).mean
        feasible = mean[:, 2] <= 0.5
        best = mean[feasible][:, :2]
        survivor = best[(best[:, 1]).argmax()]
        assert constrained.tolist() == pytest.approx((survivor * 0.9).tolist(), rel=1e-9)
