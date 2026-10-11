"""Unit tests for `EncodedConstraints` and `sampleFeasibleEncoded`."""

from __future__ import annotations

import math

import pytest
import torch

from planexs import (
    Categorical,
    Choice,
    Fixed,
    Integer,
    LinearConstraint,
    NonlinearConstraint,
    Real,
    SearchSpace,
)
from planexs.acquisition.EncodedConstraints import EncodedConstraints
from planexs.acquisition.sampleFeasibleEncoded import sampleFeasibleEncoded
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder


def _build(space: SearchSpace) -> tuple[SearchSpaceEncoder, EncodedConstraints]:
    encoder = SearchSpaceEncoder(space)
    return encoder, EncodedConstraints(space, encoder)


class TestLinearEncoding:
    @pytest.mark.parametrize("comparator", ["<=", ">=", "="])
    def test_feasibility_in_unit_space_matches_natural_values(self, comparator: str) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 10.0), Real("y", 2.0, 6.0)],
            constraints=[LinearConstraint({"x": 2.0, "y": 3.0}, 30.0, comparator)],  # type: ignore[arg-type]
        )
        encoder, constraints = _build(space)
        points = torch.rand(
            400, 2, dtype=torch.double, generator=torch.Generator().manual_seed(1)
        )

        feasible = constraints.isFeasible(points, tolerance=1e-9)

        raw = encoder.rawRangeValues(points)
        value = 2.0 * raw[:, 0] + 3.0 * raw[:, 1]
        expected = {
            "<=": value <= 30.0 + 1e-9,
            ">=": value >= 30.0 - 1e-9,
            "=": (value - 30.0).abs() <= 1e-9,
        }[comparator]
        assert torch.equal(feasible, expected)

    def test_botorch_formats(self) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 10.0), Real("y", 2.0, 6.0)],
            constraints=[
                LinearConstraint({"x": 1.0, "y": 1.0}, 8.0, "<="),
                LinearConstraint({"x": 1.0, "y": -1.0}, 0.0, ">="),
                LinearConstraint({"x": 1.0, "y": 1.0}, 9.0, "="),
            ],
        )
        _, constraints = _build(space)

        assert constraints.inequality is not None and len(constraints.inequality) == 2
        assert constraints.equality is not None and len(constraints.equality) == 1
        assert constraints.nonlinear is None
        indices, coefficients, rhs = constraints.equality[0]
        # x + y = 9, x = 10 u0, y = 2 + 4 u1  ->  10 u0 + 4 u1 = 7
        assert indices.tolist() == [0, 1]
        assert coefficients.tolist() == [10.0, 4.0]
        assert rhs == pytest.approx(7.0)

    def test_a_mixture_equality_is_satisfied_by_the_point_it_describes(self) -> None:
        space = SearchSpace(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")],
        )
        _, constraints = _build(space)
        good = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)
        bad = torch.tensor([[0.2, 0.3, 0.6]], dtype=torch.double)
        assert bool(constraints.isFeasible(good)) and not bool(constraints.isFeasible(bad))

    def test_rejects_a_log_scaled_parameter(self) -> None:
        space = SearchSpace(
            [Real("x", 1.0, 10.0, scaling="log"), Real("y", 0.0, 1.0)],
            constraints=[LinearConstraint({"x": 1.0, "y": 1.0}, 5.0, "<=")],
        )
        with pytest.raises(ValueError, match="log-scaled parameter 'x'"):
            _build(space)

    @pytest.mark.parametrize(
        "other",
        [Choice("k", [1, 2, 3], "int"), Categorical("k", ["a", "b"]), Fixed("k", 2.0)],
    )
    def test_rejects_non_range_parameters(self, other: Choice) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 1.0), other],
            constraints=[LinearConstraint({"x": 1.0, "k": 1.0}, 2.0, "<=")],
        )
        with pytest.raises(ValueError, match="not a searched Range parameter"):
            _build(space)

    def test_integer_ranges_are_supported(self) -> None:
        space = SearchSpace(
            [Integer("a", 0, 10), Integer("b", 0, 10)],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0}, 7.0, "<=")],
        )
        encoder, constraints = _build(space)
        inside = encoder.encodeParameters({"a": 3, "b": 4})
        outside = encoder.encodeParameters({"a": 4, "b": 4})
        assert bool(constraints.isFeasible(inside)) and not bool(
            constraints.isFeasible(outside)
        )


class TestNonlinearEncoding:
    @staticmethod
    def _space() -> SearchSpace:
        return SearchSpace(
            [Real("x", 0.1, 10.0), Real("y", 0.1, 10.0)],
            constraints=[NonlinearConstraint("x * y ** 2", "<=", 50.0)],
        )

    def test_evaluates_on_natural_values_not_on_the_unit_cube(self) -> None:
        encoder, constraints = _build(self._space())
        inside = encoder.encodeParameters({"x": 2.0, "y": 4.0})  # 32 <= 50
        outside = encoder.encodeParameters({"x": 4.0, "y": 4.0})  # 64 > 50

        assert bool(constraints.isFeasible(inside)) and not bool(
            constraints.isFeasible(outside)
        )

    def test_botorch_callables_are_intra_point_and_work_on_a_single_point(self) -> None:
        encoder, constraints = _build(self._space())
        assert constraints.nonlinear is not None
        [(function, intra_point)] = constraints.nonlinear
        point = encoder.encodeParameters({"x": 2.0, "y": 4.0}).requires_grad_()

        value = function(point)
        value.backward()

        assert intra_point is True
        assert float(value) == pytest.approx(50.0 - 32.0)
        assert point.grad is not None and bool(torch.isfinite(point.grad).all())

    def test_violation_is_zero_when_feasible_and_relative_to_the_bound_when_not(self) -> None:
        encoder, constraints = _build(self._space())
        feasible = encoder.encodeParameters({"x": 2.0, "y": 4.0})
        violated = encoder.encodeParameters({"x": 10.0, "y": 10.0})  # 1000 vs 50

        assert float(constraints.violation(feasible)) == 0.0
        assert float(constraints.violation(violated)) == pytest.approx((1000.0 - 50.0) / 50.0)

    def test_an_undefined_expression_is_infeasible(self) -> None:
        space = SearchSpace(
            [Real("x", -1.0, 1.0)], constraints=[NonlinearConstraint("sqrt(x)", ">=", 0.0)]
        )
        encoder, constraints = _build(space)
        negative = encoder.encodeParameters({"x": -0.5})
        assert math.isinf(float(constraints.violation(negative)))
        assert not bool(constraints.isFeasible(negative))

    def test_fixed_parameters_can_be_used_as_constants(self) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 10.0), Fixed("k", 3.0)],
            constraints=[NonlinearConstraint("k * x", "<=", 15.0)],
        )
        encoder, constraints = _build(space)
        assert bool(constraints.isFeasible(encoder.encodeParameters({"x": 5.0})))
        assert not bool(constraints.isFeasible(encoder.encodeParameters({"x": 6.0})))

    def test_rejects_a_categorical_parameter(self) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 1.0), Categorical("c", ["a", "b"])],
            constraints=[NonlinearConstraint("x * c", "<=", 1.0)],
        )
        with pytest.raises(ValueError, match="Range parameters and numeric Fixed"):
            _build(space)


class TestNoConstraints:
    def test_everything_is_feasible(self) -> None:
        _encoder, constraints = _build(SearchSpace([Real("x", 0.0, 1.0)]))
        assert not constraints.has_constraints
        assert constraints.inequality is None and constraints.equality is None
        assert constraints.nonlinear is None
        assert bool(constraints.isFeasible(torch.rand(5, 1, dtype=torch.double)).all())


class TestSampleFeasibleEncoded:
    def test_without_constraints_it_is_plain_sobol(self) -> None:
        encoder, constraints = _build(SearchSpace([Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)]))
        assert torch.equal(
            sampleFeasibleEncoded(encoder, constraints, 8, seed=2),
            encoder.sampleEncoded(8, seed=2),
        )

    def test_every_point_satisfies_a_nonlinear_constraint(self) -> None:
        space = SearchSpace(
            [Real("x", 0.1, 10.0), Real("y", 0.1, 10.0)],
            constraints=[NonlinearConstraint("x * y ** 2", "<=", 50.0)],
        )
        encoder, constraints = _build(space)

        points = sampleFeasibleEncoded(encoder, constraints, 64, seed=0)

        assert points.shape == (64, 2)
        assert bool(constraints.isFeasible(points).all())
        assert len({tuple(p.tolist()) for p in points}) == 64  # distinct

    def test_a_mixture_equality_is_met_exactly(self) -> None:
        space = SearchSpace(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")],
        )
        encoder, constraints = _build(space)

        points = sampleFeasibleEncoded(encoder, constraints, 40, seed=0)

        assert torch.allclose(
            points.sum(dim=-1), torch.ones(40, dtype=torch.double), atol=1e-6
        )

    def test_linear_and_nonlinear_constraints_hold_together(self) -> None:
        space = SearchSpace(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            constraints=[
                LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "="),
                NonlinearConstraint("a * b", ">=", 0.05),
            ],
        )
        encoder, constraints = _build(space)

        points = sampleFeasibleEncoded(encoder, constraints, 30, seed=0)

        assert bool(constraints.isFeasible(points).all())

    def test_snapped_points_are_valid_and_still_feasible(self) -> None:
        space = SearchSpace(
            [Integer("a", 0, 10), Integer("b", 0, 10), Categorical("c", ["u", "v", "w"])],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0}, 7.0, "<=")],
        )
        encoder, constraints = _build(space)

        points = sampleFeasibleEncoded(encoder, constraints, 40, seed=0, snap=True)

        assert torch.equal(encoder.snapColumns(points), points)
        assert bool(constraints.isFeasible(points).all())
        for point in points:
            parameters = encoder.decode(point)
            assert parameters["a"] + parameters["b"] <= 7  # type: ignore[operator]

    def test_is_reproducible_for_a_seed(self) -> None:
        space = SearchSpace(
            [Real("x", 0.1, 10.0), Real("y", 0.1, 10.0)],
            constraints=[NonlinearConstraint("x * y", "<=", 5.0)],
        )
        encoder, constraints = _build(space)
        first = sampleFeasibleEncoded(encoder, constraints, 16, seed=7)
        second = sampleFeasibleEncoded(encoder, constraints, 16, seed=7)
        assert torch.equal(first, second)

    def test_an_empty_feasible_region_is_an_error_not_an_infinite_loop(self) -> None:
        space = SearchSpace(
            [Real("x", 0.1, 1.0), Real("y", 0.1, 1.0)],
            constraints=[NonlinearConstraint("x * y", "<=", -1.0)],
        )
        encoder, constraints = _build(space)
        with pytest.raises(RuntimeError, match="no point satisfying"):
            sampleFeasibleEncoded(encoder, constraints, 8, seed=0)

    def test_a_tiny_feasible_region_is_repeated_to_fill_the_request(self) -> None:
        space = SearchSpace(
            [Integer("a", 0, 10), Integer("b", 0, 10)],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0}, 1.0, "<=")],
        )
        encoder, constraints = _build(space)

        points = sampleFeasibleEncoded(encoder, constraints, 50, seed=0, snap=True)

        assert points.shape == (50, 2)
        assert bool(constraints.isFeasible(points).all())
