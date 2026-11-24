"""Unit tests for `outcomeSpreads`, the scale of each outcome of a fitted model (roadmap task
P5-06, specification 5.8)."""

from __future__ import annotations

import pytest
import torch
from toy_surrogates import makeToySurrogate

from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.outcomeSpreads import outcomeSpreads


class TestOutcomeSpreads:
    def test_it_is_the_standard_deviation_of_the_posterior_mean_over_the_observed_points(
        self,
    ) -> None:
        model = makeToySurrogate().model
        points = baselinePoints(model)
        with torch.no_grad():
            mean = model.posterior(points).mean

        spreads = outcomeSpreads(model, points)

        assert spreads == pytest.approx(
            [mean[:, 0].std(correction=0).item(), mean[:, 1].std(correction=0).item()]
        )

    def test_there_is_one_for_each_output_of_the_model(self) -> None:
        both = makeToySurrogate().model
        alone = makeToySurrogate(objective_only=True).model

        assert len(outcomeSpreads(both, baselinePoints(both))) == 2
        assert len(outcomeSpreads(alone, baselinePoints(alone))) == 1

    def test_an_outcome_measured_in_a_larger_unit_has_a_proportionally_larger_spread(
        self,
    ) -> None:
        small = makeToySurrogate(constraint_scale=1.0).model
        large = makeToySurrogate(constraint_scale=1000.0).model

        (_, small_spread) = outcomeSpreads(small, baselinePoints(small))
        (_, large_spread) = outcomeSpreads(large, baselinePoints(large))

        assert large_spread == pytest.approx(1000.0 * small_spread, rel=1e-6)

    def test_an_outcome_with_no_spread_is_left_in_its_own_unit(self) -> None:
        # One observed point: all the means are the same.
        model = makeToySurrogate(observed_x=(0.5,)).model

        assert outcomeSpreads(model, baselinePoints(model)) == [1.0, 1.0]

    def test_it_builds_no_graph_and_changes_nothing(self) -> None:
        model = makeToySurrogate().model
        before = [parameter.detach().clone() for parameter in model.parameters()]

        outcomeSpreads(model, baselinePoints(model))

        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
