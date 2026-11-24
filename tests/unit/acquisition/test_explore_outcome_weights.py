"""Unit tests for `exploreOutcomeWeights`, the weight of each output of a model in an
exploration (roadmap task P5-06, specification 5.8, FR21)."""

from __future__ import annotations

import pytest
import torch
from toy_pareto_surrogates import F1, F2, G, makeParetoSurrogate

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.exploreOutcomeWeights import exploreOutcomeWeights
from planexs.acquisition.outcomeSpreads import outcomeSpreads


def weights(explored: tuple[str, ...], outcomes: tuple[str, ...] = (F1, F2, G)) -> list[float]:
    model = makeParetoSurrogate(outcomes).model
    result = exploreOutcomeWeights(explored, outcomes, model, baselinePoints(model))
    return [float(value) for value in result]


class TestExploreOutcomeWeights:
    def test_an_explored_outcome_weighs_the_reciprocal_of_its_spread(self) -> None:
        model = makeParetoSurrogate((F1, F2, G)).model
        spreads = outcomeSpreads(model, baselinePoints(model))

        result = weights((F1, F2))

        assert result[0] == pytest.approx(1.0 / spreads[0])
        assert result[1] == pytest.approx(1.0 / spreads[1])

    def test_an_output_that_is_not_explored_weighs_nothing(self) -> None:
        assert weights((F1, F2))[2] == 0.0
        assert weights((G,))[:2] == [0.0, 0.0]

    def test_the_weights_have_no_sign(self) -> None:
        assert all(value >= 0.0 for value in weights((F1, F2, G)))

    def test_the_weights_are_placed_by_name_whatever_the_order_of_the_outputs(self) -> None:
        in_order = weights((F1, F2), (F1, F2, G))
        reversed_ = weights((F1, F2), (G, F2, F1))

        assert reversed_ == pytest.approx(list(reversed(in_order)))

    def test_the_unit_of_an_outcome_does_not_change_what_it_weighs_in_its_own_unit(
        self,
    ) -> None:
        # g measured a thousand times larger: its weight is a thousand times smaller, so that
        # weight times spread is the same, and the outcome counts the same.
        model = makeParetoSurrogate((F1, F2, G), constraint_scale=1000.0).model

        large = exploreOutcomeWeights((G,), (F1, F2, G), model, baselinePoints(model))

        assert float(large[2]) == pytest.approx(weights((G,))[2] / 1000.0, rel=1e-6)

    def test_it_has_the_precision_of_the_observed_points(self) -> None:
        model = makeParetoSurrogate((F1, F2, G)).model
        points = baselinePoints(model)

        result = exploreOutcomeWeights((F1,), (F1, F2, G), model, points)

        assert result.dtype == points.dtype
        assert result.shape == (3,)

    def test_an_outcome_that_the_surrogate_does_not_model_is_refused(self) -> None:
        with pytest.raises(AcquisitionContextError, match="'g'") as caught:
            weights((F1, G), (F1, F2))

        assert caught.value.field_path == "goal.outcomes.g"

    def test_an_outcome_with_no_spread_weighs_one(self) -> None:
        model = makeParetoSurrogate((F1, F2), observed_x=(0.5,)).model

        result = exploreOutcomeWeights((F1, F2), (F1, F2), model, baselinePoints(model))

        assert torch.equal(result, torch.ones(2, dtype=result.dtype))
