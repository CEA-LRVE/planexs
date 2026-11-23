"""Unit tests for `predictedParetoFront`, the front that a model predicts at the observed
points (roadmap task P5-05, specification 5.8)."""

from __future__ import annotations

import torch
from toy_pareto_surrogates import F1, F2, G, makeParetoSurrogate

from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.outcomeConstraintCallables import outcomeConstraintCallables
from planexs.acquisition.ParetoObjectives import ParetoObjectives
from planexs.acquisition.predictedParetoFront import predictedParetoFront
from planexs.goals.OutcomeConstraint import OutcomeConstraint

NAMES = (F1, F2, G)


def front(
    signs: tuple[float, float],
    *,
    constraints: list[OutcomeConstraint] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The front, and the posterior means of the objectives at the observed points."""
    model = makeParetoSurrogate(NAMES).model
    baseline = baselinePoints(model)
    objectives = ParetoObjectives(names=(F1, F2), indices=(0, 1), signs=signs, num_outputs=3)
    callables = outcomeConstraintCallables(model, baseline, constraints or [], NAMES) or None
    result = predictedParetoFront(model, baseline, objectives, callables)
    with torch.no_grad():
        mean = model.posterior(baseline).mean
    return result, mean[:, :2] * torch.tensor(signs, dtype=mean.dtype)


class TestPredictedParetoFront:
    def test_it_holds_the_means_that_nothing_dominates(self) -> None:
        result, means = front((1.0, 1.0))

        # f1 = x grows over the space and f2 = sin(3x) peaks at 0.52: the points to the right
        # of the peak trade one against the other, and the ones to its left are beaten by them.
        assert 0 < result.shape[0] < means.shape[0]
        for point in result:
            assert not any(
                bool((other >= point).all() and (other > point).any()) for other in means
            )
        for point in means:
            if not any(torch.equal(point, kept) for kept in result):
                assert any(
                    bool((other >= point).all() and (other > point).any()) for other in means
                )

    def test_it_is_made_of_predicted_means_and_not_of_observations(self) -> None:
        result, means = front((1.0, 1.0))

        for point in result:
            assert any(torch.equal(point, mean) for mean in means)

    def test_a_minimized_objective_is_negated(self) -> None:
        maximized, _ = front((1.0, 1.0))
        minimized, means = front((-1.0, 1.0))

        assert (minimized[:, 0] <= 0.0).all()
        assert not torch.equal(maximized, minimized)
        # With f1 minimized the points on the left of the peak are the front.
        assert minimized.shape[0] < means.shape[0]

    def test_a_point_predicted_to_violate_a_constraint_is_left_out(self) -> None:
        constraint = OutcomeConstraint(outcome=G, comparator="<=", bound=0.5)

        free, _ = front((1.0, 1.0))
        constrained, _ = front((1.0, 1.0), constraints=[constraint])

        # Only the observed point at x = 0.436 is both feasible and on the front.
        assert free.shape[0] > constrained.shape[0] == 1

    def test_when_every_point_violates_a_constraint_the_front_is_that_of_all_of_them(
        self,
    ) -> None:
        # A front of nothing has no range to infer a reference point from.
        impossible = OutcomeConstraint(outcome=G, comparator=">=", bound=100.0)

        free, _ = front((1.0, 1.0))
        constrained, _ = front((1.0, 1.0), constraints=[impossible])

        assert torch.equal(free, constrained)

    def test_it_builds_no_graph(self) -> None:
        result, _ = front((1.0, 1.0))

        assert not result.requires_grad
