"""Unit tests for `outcomeConstraintCallables`, the outcome constraints of a goal as the
feasibility constraints of a BoTorch acquisition (roadmap task P5-03, specification 5.8,
FR15).

BoTorch reads a constraint as a callable on posterior samples that is negative where the
constraint is met, so the tests check the sign, the column it reads, the scale it divides by
and what it refuses.
"""

from __future__ import annotations

import pytest
import torch
from toy_surrogates import CONSTRAINED, OBJECTIVE, makeToySurrogate

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.outcomeConstraintCallables import outcomeConstraintCallables
from planexs.goals.OutcomeConstraint import OutcomeConstraint

NAMES = (OBJECTIVE, CONSTRAINED)


def samples() -> torch.Tensor:
    """Posterior samples of shape `4 x 1 x 2 x 2`: sample, batch, q, outcome (`y`, `c`).

    `y` is always 100, so a callable that reads the wrong column is not near the bound of `c`.
    """
    c = torch.tensor(
        [[[0.2, 0.5]], [[0.5, 0.8]], [[0.0, 0.1]], [[0.9, 0.5]]], dtype=torch.float64
    )
    y = torch.full_like(c, 100.0)
    return torch.stack([y, c], dim=-1)


def callables(*constraints: OutcomeConstraint, scale: float = 1.0) -> list:  # type: ignore[type-arg]
    """The callables for the toy surrogate, with the constrained outcome in `scale` units."""
    surrogate = makeToySurrogate(constraint_scale=scale)
    return outcomeConstraintCallables(
        surrogate.model, baselinePoints(surrogate.model), constraints, surrogate.outcome_names
    )


class TestTheSign:
    def test_an_upper_bound_is_negative_where_the_outcome_is_below_it(self) -> None:
        (violation,) = callables(
            OutcomeConstraint(outcome=CONSTRAINED, comparator="<=", bound=0.5)
        )

        values = violation(samples())

        assert values.shape == (4, 1, 2)
        assert (values[samples()[..., 1] < 0.5] < 0).all()
        assert (values[samples()[..., 1] > 0.5] > 0).all()
        assert (values[samples()[..., 1] == 0.5] == 0).all()

    def test_a_lower_bound_is_the_other_way_round(self) -> None:
        (violation,) = callables(
            OutcomeConstraint(outcome=CONSTRAINED, comparator=">=", bound=0.5)
        )

        values = violation(samples())

        assert (values[samples()[..., 1] > 0.5] < 0).all()
        assert (values[samples()[..., 1] < 0.5] > 0).all()

    def test_it_reads_the_column_of_the_constrained_outcome(self) -> None:
        # y is 100 everywhere: if the callable read it, a bound of 0.5 would always be
        # violated.
        (violation,) = callables(
            OutcomeConstraint(outcome=CONSTRAINED, comparator="<=", bound=0.5)
        )

        assert (violation(samples()) < 0).any()

    def test_it_reads_the_column_by_the_name_and_not_by_the_position(self) -> None:
        surrogate = makeToySurrogate()
        constraint = OutcomeConstraint(outcome=OBJECTIVE, comparator="<=", bound=50.0)

        (violation,) = outcomeConstraintCallables(
            surrogate.model, baselinePoints(surrogate.model), [constraint], NAMES
        )

        assert (violation(samples()) > 0).all()  # y is 100 everywhere: above the bound


class TestTheScale:
    def test_the_violation_is_in_units_of_the_spread_of_the_outcome(self) -> None:
        constraint = OutcomeConstraint(outcome=CONSTRAINED, comparator="<=", bound=0.5)
        (metre,) = callables(constraint)
        (micrometre,) = callables(
            OutcomeConstraint(outcome=CONSTRAINED, comparator="<=", bound=0.5e6), scale=1.0e6
        )
        in_micrometres = samples().clone()
        in_micrometres[..., 1] *= 1.0e6

        assert torch.allclose(metre(samples()), micrometre(in_micrometres), rtol=1e-6)

    def test_the_spread_is_that_of_the_posterior_mean_over_the_observed_points(self) -> None:
        surrogate = makeToySurrogate()
        points = baselinePoints(surrogate.model)
        with torch.no_grad():
            spread = surrogate.model.posterior(points).mean[:, 1].std(correction=0).item()
        (violation,) = callables(
            OutcomeConstraint(outcome=CONSTRAINED, comparator="<=", bound=0.5)
        )

        value = violation(samples())[0, 0, 0].item()  # c = 0.2

        assert value == pytest.approx((0.2 - 0.5) / spread)

    def test_an_outcome_with_no_spread_is_left_in_its_own_unit(self) -> None:
        surrogate = makeToySurrogate(objective_only=True)
        one_point = surrogate.model.train_inputs[0][:1]
        (violation,) = outcomeConstraintCallables(
            surrogate.model,
            one_point,
            [OutcomeConstraint(outcome=OBJECTIVE, comparator="<=", bound=0.5)],
            [OBJECTIVE],
        )
        values = torch.tensor([[[[0.2]]]], dtype=torch.float64)

        assert violation(values).item() == pytest.approx(0.2 - 0.5)


class TestTheCallables:
    def test_there_is_one_callable_for_each_constraint_in_order(self) -> None:
        first = OutcomeConstraint(outcome=CONSTRAINED, comparator="<=", bound=0.5)
        second = OutcomeConstraint(outcome=CONSTRAINED, comparator=">=", bound=0.2)

        below, above = callables(first, second)

        c = samples()[..., 1]
        # The first reads `c <= 0.5` and rises with c, the second reads `c >= 0.2` and falls
        # with it: so the first is the second's mirror, by a constant that is the gap of the
        # bounds in units of the spread.
        total = below(samples()) + above(samples())
        assert (below(samples())[c == 0.5] == 0).all()
        assert (above(samples())[c == 0.0] > 0).all()
        assert torch.allclose(total, total.flatten()[0].expand_as(total))
        assert total.flatten()[0] < 0

    def test_no_constraint_gives_no_callable(self) -> None:
        assert callables() == []

    def test_an_outcome_the_surrogate_does_not_model_is_refused(self) -> None:
        surrogate = makeToySurrogate()
        constraint = OutcomeConstraint(outcome="purity", comparator=">=", bound=0.9)

        with pytest.raises(AcquisitionContextError, match="'purity'") as caught:
            outcomeConstraintCallables(
                surrogate.model, baselinePoints(surrogate.model), [constraint], NAMES
            )

        assert caught.value.field_path == "goal.outcome_constraints[0].outcome"
        assert "'y', 'c'" in str(caught.value)

    def test_the_position_of_the_refused_constraint_is_named(self) -> None:
        surrogate = makeToySurrogate()
        constraints = [
            OutcomeConstraint(outcome=CONSTRAINED, comparator="<=", bound=0.5),
            OutcomeConstraint(outcome="purity", comparator=">=", bound=0.9),
        ]

        with pytest.raises(AcquisitionContextError) as caught:
            outcomeConstraintCallables(
                surrogate.model, baselinePoints(surrogate.model), constraints, NAMES
            )

        assert caught.value.field_path == "goal.outcome_constraints[1].outcome"
