"""`Problem` accepts every parameterization that the encoder decodes (roadmap task P2-10).

Specification 5.5 says that decoding any point of the unit cube returns a legal
parameterization: rounded onto its grids, with the parameters that a choice switches off
removed and the derived ones computed, and that a constraint is never violated by a point
returned to the caller. `Problem.validateParameterization` is the domain's own statement of
what legal means, in plain numbers and with no torch. If the two disagreed, a study would
refuse the points of its own optimizer. These tests make them meet, on a space with every
kind of parameter and constraint.

They live here, not under `tests/unit/domain`, because they need the encoding layer and torch.
"""

from __future__ import annotations

import functools
from typing import Any

import pytest

from planexs import (
    Boolean,
    Categorical,
    Choice,
    Derived,
    Fixed,
    Integer,
    LinearConstraint,
    NonlinearConstraint,
    Real,
    SearchSpace,
)
from planexs.domain.Outcome import Outcome
from planexs.domain.ParameterizationError import ParameterizationError
from planexs.domain.Problem import Problem
from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.sampleFeasibleEncoded import sampleFeasibleEncoded
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder

#: Points decoded per seed. Enough to reach both sides of the conditional branch and most of
#: the grid, few enough to keep the polytope sampler quick.
POINTS = 60
SEEDS = [0, 1, 2]


def mixedSpace() -> SearchSpace:
    """One of every kind of parameter, a conditional branch, and the constraints that the
    encoder can enforce: a mixture that sums to one, a limit on two ranges and a nonlinear one.
    """
    return SearchSpace(
        [
            Real("temperature", 20.0, 80.0),
            Real("rate", 1e-3, 1.0, scaling="log"),
            Integer("cycles", 1, 10),
            Real("ratio", 0.0, 1.0, step_size=0.1),
            Integer("batch", 2, 20, step_size=2),
            Categorical("solvent", ["water", "ethanol", "acetone"]),
            Choice("level", [1, 2, 3], "int", is_ordered=True),
            Boolean("stirred"),
            Fixed("scale", 3),
            Categorical("model", ["linear", "tree"], dependent_parameters={"tree": ["depth"]}),
            Integer("depth", 1, 8),
            Derived("total", "temperature + cycles", "float"),
            Real("a", 0.0, 1.0),
            Real("b", 0.0, 1.0),
            Real("c", 0.0, 1.0),
        ],
        [
            LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, bound=1.0, comparator="="),
            LinearConstraint(
                {"temperature": 1.0, "cycles": 5.0}, bound=110.0, comparator="<="
            ),
            NonlinearConstraint("rate * temperature", "<=", 30.0),
        ],
    )


def decodedPoints(seed: int) -> list[dict[str, Any]]:
    """Feasible encoded points of `mixedSpace`, decoded the way a policy's suggestions are.

    Sampling the polytope is the slow part, so it is done once per seed. Every call returns
    its own copies: a test may change the points it gets.
    """
    return [dict(point) for point in _sampled(seed)]


@functools.cache
def _sampled(seed: int) -> tuple[dict[str, Any], ...]:
    space = mixedSpace()
    encoder = SearchSpaceEncoder(space)
    constraints = EncodedConstraints(space, encoder)
    points = sampleFeasibleEncoded(encoder, constraints, POINTS, seed=seed, snap=True)
    return tuple(dict(encoder.decode(point)) for point in points)


def problemOf(space: SearchSpace) -> Problem:
    return Problem(search_space=space, outcomes=[Outcome(name="y", type="continuous")])


class TestEveryDecodedPointIsLegal:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_space_with_every_kind_of_parameter_and_constraint(self, seed: int) -> None:
        problem = problemOf(mixedSpace())

        decoded = decodedPoints(seed)

        for parameterization in decoded:
            problem.validateParameterization(parameterization)

    def test_the_sample_reaches_both_sides_of_the_branch_and_the_grids(self) -> None:
        # A sample that never takes the `tree` branch, or never lands on a step, would show
        # nothing about them.
        decoded = [p for seed in SEEDS for p in decodedPoints(seed)]

        assert any("depth" in p for p in decoded) and any("depth" not in p for p in decoded)
        assert {p["solvent"] for p in decoded} == {"water", "ethanol", "acetone"}
        assert {p["stirred"] for p in decoded} == {True, False}
        assert {p["level"] for p in decoded} == {1, 2, 3}
        assert len({p["ratio"] for p in decoded}) > 5
        assert all(p["scale"] == 3 and "total" in p for p in decoded)

    def test_a_space_without_constraints(self) -> None:
        space = SearchSpace(mixedSpace().parameters)
        problem = problemOf(space)
        encoder = SearchSpaceEncoder(space)

        for point in encoder.sampleEncoded(150, seed=7, snap=True):
            problem.validateParameterization(encoder.decode(point))

    def test_the_decoded_points_are_also_legal_after_a_save_and_a_reload(self) -> None:
        problem = problemOf(mixedSpace())
        rebuilt = Problem.model_validate_json(problem.model_dump_json())

        for parameterization in decodedPoints(5):
            rebuilt.validateParameterization(parameterization)

    def test_a_legal_parameterization_can_be_encoded_again(self) -> None:
        space = mixedSpace()
        problem = problemOf(space)
        encoder = SearchSpaceEncoder(space)

        for parameterization in decodedPoints(3):
            problem.validateParameterization(parameterization)
            assert (
                encoder.decode(encoder.encodeParameters(parameterization)) == parameterization
            )


class TestAPerturbedPointIsRefused:
    """The same points, changed in one way each: the check is not accepting everything."""

    def example(self, model: str) -> dict[str, Any]:
        for parameterization in decodedPoints(0):
            if parameterization["model"] == model:
                return parameterization
        raise AssertionError(f"no decoded point with model={model!r}")

    @pytest.mark.parametrize(
        ("change", "fragment"),
        [
            ({"temperature": 95.0}, "outside the bounds"),
            ({"cycles": 2.5}, "not a whole number"),
            ({"ratio": 0.35}, "not on the grid"),
            ({"batch": 3}, "not on the grid"),
            ({"solvent": "methanol"}, "not one of its values"),
            ({"level": 4}, "not one of its values"),
            ({"stirred": 1}, "not one of its values"),
            ({"scale": 4}, "not one of its values"),
            ({"colour": "red"}, "not a parameter"),
            ({"a": 0.5, "b": 0.5, "c": 0.5}, "does not hold"),
        ],
    )
    def test_one_change(self, change: dict[str, Any], fragment: str) -> None:
        problem = problemOf(mixedSpace())
        point = {**self.example("linear"), **change}

        with pytest.raises(ParameterizationError, match=fragment):
            problem.validateParameterization(point)

    def test_a_required_parameter_removed(self) -> None:
        point = self.example("linear")
        del point["scale"]

        with pytest.raises(ParameterizationError, match="'scale' has no value"):
            problemOf(mixedSpace()).validateParameterization(point)

    def test_a_parameter_of_the_branch_that_is_off(self) -> None:
        point = {**self.example("linear"), "depth": 3}

        with pytest.raises(
            ParameterizationError, match="'depth' is given, but it only exists"
        ):
            problemOf(mixedSpace()).validateParameterization(point)

    def test_a_parameter_of_the_branch_that_is_on_removed(self) -> None:
        point = self.example("tree")
        del point["depth"]

        with pytest.raises(ParameterizationError, match="'depth' has no value"):
            problemOf(mixedSpace()).validateParameterization(point)

    def test_the_nonlinear_constraint_is_enforced(self) -> None:
        point = {**self.example("linear"), "temperature": 80.0, "rate": 0.9, "cycles": 1}

        with pytest.raises(ParameterizationError, match="rate \\* temperature"):
            problemOf(mixedSpace()).validateParameterization(point)
