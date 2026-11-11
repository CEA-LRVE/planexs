"""Unit tests for `ProblemFacts`: what a problem contains, as plain data (roadmap P2-10)."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any, Literal

import pytest
from pydantic import Field, ValidationError

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
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.registerConstraint import registerConstraint
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.domain.ProblemFacts import ProblemFacts


def problemOf(
    parameters: list[Any],
    constraints: list[Any] | None = None,
    outcomes: list[Outcome] | None = None,
) -> Problem:
    """A problem of the given parts, with one continuous outcome unless told otherwise."""
    return Problem(
        search_space=SearchSpace(parameters, constraints),
        outcomes=outcomes or [Outcome(name="y", type="continuous")],
    )


class TestOutcomes:
    def test_the_number_of_outcomes(self) -> None:
        one = problemOf([Real("x", 0.0, 1.0)])
        three = problemOf(
            [Real("x", 0.0, 1.0)],
            outcomes=[Outcome(name=n, type="continuous") for n in ("a", "b", "c")],
        )

        assert one.facts().n_outcomes == 1
        assert three.facts().n_outcomes == 3

    def test_the_outcome_types_are_distinct_and_sorted(self) -> None:
        problem = problemOf(
            [Real("x", 0.0, 1.0)],
            outcomes=[
                Outcome(name="a", type="continuous"),
                Outcome(name="b", type="binary"),
                Outcome(name="c", type="continuous"),
            ],
        )

        assert problem.facts().outcome_types == ("binary", "continuous")

    def test_a_single_kind(self) -> None:
        assert problemOf([Real("x", 0.0, 1.0)]).facts().outcome_types == ("continuous",)


class TestParameterTypes:
    def test_the_sugar_constructors_are_not_types(self) -> None:
        real_and_integer = problemOf([Real("x", 0.0, 1.0), Integer("n", 1, 5)])
        every_choice = problemOf([Categorical("c", ["a", "b"]), Boolean("f"), Fixed("k", 3)])

        assert real_and_integer.facts().parameter_types == ("range",)
        assert every_choice.facts().parameter_types == ("choice",)

    def test_all_three_are_listed_sorted_and_once(self) -> None:
        problem = problemOf(
            [
                Real("x", 0.0, 1.0),
                Categorical("c", ["a", "b"]),
                Derived("d", "x * 2", "float"),
                Real("y", 0.0, 1.0),
                Boolean("f"),
            ]
        )

        assert problem.facts().parameter_types == ("choice", "derived", "range")


class TestConstraintTypes:
    def test_none_when_the_space_has_no_constraint(self) -> None:
        assert problemOf([Real("x", 0.0, 1.0)]).facts().constraint_types == ()

    def test_each_type_that_occurs(self) -> None:
        linear = LinearConstraint({"x": 1.0}, 1.0, "<=")
        nonlinear = NonlinearConstraint("x ** 2", "<=", 1.0)
        parameters = [Real("x", 0.0, 1.0)]

        assert problemOf(parameters, [linear]).facts().constraint_types == ("linear",)
        assert problemOf(parameters, [nonlinear]).facts().constraint_types == ("nonlinear",)
        both = problemOf(parameters, [nonlinear, linear, linear]).facts()
        assert both.constraint_types == ("linear", "nonlinear")


class TestConditionalParameters:
    def test_a_choice_that_switches_parameters_on(self) -> None:
        problem = problemOf(
            [
                Categorical("m", ["a", "b"], dependent_parameters={"b": ["depth"]}),
                Integer("depth", 1, 5),
            ]
        )

        assert problem.facts().has_conditional_parameters is True

    def test_no_choice_has_dependents(self) -> None:
        problem = problemOf([Categorical("m", ["a", "b"]), Integer("depth", 1, 5)])

        assert problem.facts().has_conditional_parameters is False

    def test_a_map_of_empty_lists_switches_nothing_on(self) -> None:
        problem = problemOf(
            [Categorical("m", ["a", "b"], dependent_parameters={"a": [], "b": []})]
        )

        assert problem.facts().has_conditional_parameters is False

    def test_one_value_with_dependents_among_values_without(self) -> None:
        problem = problemOf(
            [
                Categorical("m", ["a", "b", "c"], dependent_parameters={"a": [], "c": ["z"]}),
                Real("z", 0.0, 1.0),
            ]
        )

        assert problem.facts().has_conditional_parameters is True


class TestEqualityConstraints:
    @property
    def parameters(self) -> list[Any]:
        return [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)]

    def test_a_linear_equality(self) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 1.0}, 1.0, "=")

        assert (
            problemOf(self.parameters, [constraint]).facts().has_equality_constraints is True
        )

    @pytest.mark.parametrize("comparator", ["<=", ">="])
    def test_a_linear_inequality_is_not_one(self, comparator: Any) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 1.0}, 1.0, comparator)

        assert (
            problemOf(self.parameters, [constraint]).facts().has_equality_constraints is False
        )

    def test_a_nonlinear_constraint_is_not_one(self) -> None:
        constraint = NonlinearConstraint("a * b", "<=", 1.0)

        assert (
            problemOf(self.parameters, [constraint]).facts().has_equality_constraints is False
        )

    def test_no_constraint(self) -> None:
        assert problemOf(self.parameters).facts().has_equality_constraints is False

    def test_one_equality_among_inequalities(self) -> None:
        constraints = [
            LinearConstraint({"a": 1.0}, 0.5, "<="),
            LinearConstraint({"a": 1.0, "b": 1.0}, 1.0, "="),
            NonlinearConstraint("a * b", "<=", 1.0),
        ]

        assert problemOf(self.parameters, constraints).facts().has_equality_constraints is True


class TestCategoricalParameters:
    """A choice is categorical if it is declared unordered. A choice that leaves the order
    open is for the encoding configuration to settle (specification 5.3), and is a fact of its
    own: the domain does not decide it.
    """

    @pytest.mark.parametrize(
        "parameter", [Categorical("c", ["a", "b", "c"]), Boolean("flag")], ids=["cat", "bool"]
    )
    def test_a_declared_unordered_choice(self, parameter: Any) -> None:
        facts = problemOf([parameter]).facts()

        assert facts.has_categorical_parameters is True
        assert facts.has_choices_with_undeclared_order is False

    def test_a_declared_ordered_choice(self) -> None:
        facts = problemOf([Choice("n", [1, 2, 3], "int", is_ordered=True)]).facts()

        assert facts.has_categorical_parameters is False
        assert facts.has_choices_with_undeclared_order is False

    @pytest.mark.parametrize(
        "parameter",
        [
            Choice("n", [1, 2, 3], "int"),
            Choice("s", ["lo", "mid", "hi"], "str"),
            Choice("two", [1, 2], "int"),
        ],
        ids=["numbers", "strings", "two"],
    )
    def test_a_choice_that_leaves_the_order_open(self, parameter: Any) -> None:
        facts = problemOf([parameter]).facts()

        assert facts.has_categorical_parameters is False
        assert facts.has_choices_with_undeclared_order is True

    def test_a_fixed_parameter_is_a_constant_not_a_choice(self) -> None:
        facts = problemOf([Fixed("k", 3), Real("x", 0.0, 1.0)]).facts()

        assert facts.has_categorical_parameters is False
        assert facts.has_choices_with_undeclared_order is False

    def test_a_categorical_with_one_value_is_not_searched(self) -> None:
        facts = problemOf([Categorical("c", ["only"]), Real("x", 0.0, 1.0)]).facts()

        assert facts.has_categorical_parameters is False

    def test_no_choice_at_all(self) -> None:
        facts = problemOf([Real("x", 0.0, 1.0), Derived("d", "x * 2", "float")]).facts()

        assert facts.has_categorical_parameters is False
        assert facts.has_choices_with_undeclared_order is False

    def test_both_kinds_can_occur_together(self) -> None:
        facts = problemOf(
            [Categorical("c", ["a", "b"]), Choice("n", [1, 2, 3], "int")]
        ).facts()

        assert facts.has_categorical_parameters is True
        assert facts.has_choices_with_undeclared_order is True


class TestTheWholeSummary:
    def test_every_fact_of_a_rich_problem(self) -> None:
        problem = problemOf(
            [
                Real("t", 0.0, 1.0),
                Integer("n", 1, 5),
                Categorical("m", ["a", "b"], dependent_parameters={"b": ["depth"]}),
                Integer("depth", 1, 5),
                Boolean("f"),
                Fixed("k", 3),
                Derived("d", "t * 2", "float"),
                Real("a", 0.0, 1.0),
                Real("b", 0.0, 1.0),
            ],
            [
                LinearConstraint({"a": 1.0, "b": 1.0}, 1.0, "="),
                NonlinearConstraint("t * n", "<=", 4.0),
            ],
            [Outcome(name="yield", type="continuous"), Outcome(name="phase", type="binary")],
        )

        assert problem.facts() == ProblemFacts(
            n_outcomes=2,
            outcome_types=("binary", "continuous"),
            parameter_types=("choice", "derived", "range"),
            constraint_types=("linear", "nonlinear"),
            has_conditional_parameters=True,
            has_equality_constraints=True,
            has_categorical_parameters=True,
            has_choices_with_undeclared_order=False,
        )

    def test_the_facts_do_not_depend_on_the_order_of_the_parts(self) -> None:
        parameters = [
            Real("a", 0.0, 1.0),
            Categorical("m", ["x", "y"], dependent_parameters={"y": ["a"]}),
            Derived("d", "a * 2", "float"),
        ]
        constraints = [
            LinearConstraint({"a": 1.0}, 1.0, "="),
            NonlinearConstraint("a ** 2", "<=", 1.0),
        ]
        outcomes = [Outcome(name="p", type="binary"), Outcome(name="q", type="continuous")]
        reference = problemOf(parameters, constraints, outcomes).facts()

        for ps, cs, os_ in itertools.product(
            itertools.permutations(parameters),
            itertools.permutations(constraints),
            itertools.permutations(outcomes),
        ):
            # `a` is switched on by `m`: a space where `m` comes later is still a legal one.
            assert problemOf(list(ps), list(cs), list(os_)).facts() == reference

    def test_the_method_and_the_constructor_agree(self) -> None:
        problem = problemOf([Real("x", 0.0, 1.0)])

        assert problem.facts() == ProblemFacts.of(problem)

    def test_facts_are_computed_again_each_time_from_the_problem(self) -> None:
        problem = problemOf([Real("x", 0.0, 1.0)])

        assert problem.facts() is not problem.facts()
        assert problem.facts() == problem.facts()


class TestAsPureData:
    def sample(self) -> ProblemFacts:
        return problemOf(
            [Categorical("c", ["a", "b"]), Real("x", 0.0, 1.0)],
            outcomes=[Outcome(name="y", type="binary")],
        ).facts()

    def test_it_is_frozen(self) -> None:
        with pytest.raises(ValidationError, match="frozen"):
            self.sample().n_outcomes = 5  # type: ignore[misc]

    def test_an_unknown_field_is_refused(self) -> None:
        data = self.sample().model_dump()
        data["has_gpu"] = True

        with pytest.raises(ValidationError, match="has_gpu"):
            ProblemFacts.model_validate(data)

    def test_it_round_trips_through_json(self) -> None:
        facts = self.sample()

        assert ProblemFacts.model_validate_json(facts.model_dump_json()) == facts

    def test_the_json_form_holds_lists_of_names(self) -> None:
        data = self.sample().model_dump(mode="json")

        assert data["parameter_types"] == ["choice", "range"]
        assert data["outcome_types"] == ["binary"]
        assert data["constraint_types"] == []
        assert data["has_categorical_parameters"] is True

    def test_it_is_hashable_data_that_can_key_a_cache(self) -> None:
        assert hash(self.sample()) == hash(self.sample())


class TestAConstraintTypeThatThePackageDoesNotKnow:
    def test_its_registered_name_is_among_the_constraint_types(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        class OrderingConfig(ComponentConfig):
            type: Literal["ordering"] = Field(
                default="ordering", description="Always 'ordering'."
            )
            first: str = Field(description="The parameter that must not exceed the second.")
            second: str = Field(description="The parameter that the first must not exceed.")

        class Ordering(Constraint):
            first: str
            second: str

            def referencedNames(self) -> frozenset[str]:
                return frozenset({self.first, self.second})

            def violation(self, parameters: Mapping[str, object]) -> float:
                return 0.0

            def toDict(self) -> dict[str, Any]:
                return {"type": "ordering", "first": self.first, "second": self.second}

        registerConstraint("ordering", config=OrderingConfig, description="Order.")(Ordering)
        problem = problemOf(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)],
            [Ordering(first="a", second="b"), LinearConstraint({"a": 1.0}, 1.0, "<=")],
        )

        assert problem.facts().constraint_types == ("linear", "ordering")
        assert problem.facts().has_equality_constraints is False
