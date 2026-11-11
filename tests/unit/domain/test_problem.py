"""Unit tests for `Problem`: its construction, its serialization and the check that a
parameterization is legal for it (roadmap task P2-10, specification 5.3).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal

import pytest
from pydantic import Field, ValidationError

from planexs import (
    Boolean,
    Categorical,
    Derived,
    Fixed,
    Integer,
    LinearConstraint,
    NonlinearConstraint,
    Real,
    SearchSpace,
)
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.PlanexsError import PlanexsError
from planexs.domain.Outcome import Outcome
from planexs.domain.ParameterizationCheckConfig import ParameterizationCheckConfig
from planexs.domain.ParameterizationError import ParameterizationError
from planexs.domain.parameters.Choice import Choice
from planexs.domain.parameters.Derived import Derived as DerivedType
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.ParameterValueError import ParameterValueError
from planexs.domain.parameters.Range import Range
from planexs.domain.parameters.registerParameter import registerParameter
from planexs.domain.Problem import Problem


def outcomes() -> list[Outcome]:
    """One continuous outcome."""
    return [Outcome(name="yield", type="continuous")]


def richSpace() -> SearchSpace:
    """Every kind of parameter and both kinds of constraint, one conditional branch."""
    return SearchSpace(
        [
            Real("temperature", 20.0, 80.0),
            Real("rate", 1e-3, 1.0, scaling="log"),
            Integer("cycles", 1, 10),
            Real("ratio", 0.0, 1.0, step_size=0.1),
            Categorical("model", ["linear", "tree"], dependent_parameters={"tree": ["depth"]}),
            Integer("depth", 1, 8),
            Boolean("stirred"),
            Fixed("scale", 3),
            Derived("total", "temperature + cycles", "float"),
            Real("a", 0.0, 1.0),
            Real("b", 0.0, 1.0),
            Real("c", 0.0, 1.0),
        ],
        [
            LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, bound=1.0, comparator="="),
            LinearConstraint(
                {"temperature": 1.0, "cycles": 5.0}, bound=100.0, comparator="<="
            ),
            NonlinearConstraint("rate * temperature", "<=", 30.0),
        ],
    )


def richProblem() -> Problem:
    """The rich space with a continuous and a binary outcome."""
    return Problem(
        search_space=richSpace(),
        outcomes=[
            Outcome(name="yield", type="continuous", description="Mass fraction."),
            Outcome(name="phase", type="binary"),
        ],
    )


#: A legal parameterization of the rich problem, in the branch where `depth` does not exist.
LEGAL: dict[str, Any] = {
    "temperature": 50.0,
    "rate": 0.1,
    "cycles": 5,
    "ratio": 0.3,
    "model": "linear",
    "stirred": True,
    "scale": 3,
    "a": 0.2,
    "b": 0.3,
    "c": 0.5,
}


def refused(problem: Problem, parameters: Any, **options: float) -> ParameterizationError:
    """The error that the check raises for `parameters`; fails if there is none."""
    configuration = ParameterizationCheckConfig(**options) if options else None
    with pytest.raises(ParameterizationError) as caught:
        problem.validateParameterization(parameters, configuration)
    return caught.value


class TestConstruction:
    def test_it_holds_a_search_space_and_outcomes(self) -> None:
        problem = richProblem()

        assert problem.search_space.parameter_names[0] == "temperature"
        assert [outcome.name for outcome in problem.outcomes] == ["yield", "phase"]
        assert isinstance(problem.outcomes, tuple)

    def test_outcomes_may_be_given_as_any_sequence(self) -> None:
        as_tuple = Problem(search_space=richSpace(), outcomes=tuple(outcomes()))

        assert as_tuple.outcomes == tuple(outcomes())

    def test_it_is_frozen(self) -> None:
        problem = richProblem()

        with pytest.raises(ValidationError, match="frozen"):
            problem.outcomes = ()  # type: ignore[misc]

    def test_it_holds_its_own_copy_of_the_search_space(self) -> None:
        space = richSpace()
        problem = Problem(search_space=space, outcomes=outcomes())

        space.addParameter(Real("late", 0.0, 1.0))

        assert "late" not in problem.search_space.parameter_names
        assert problem.search_space is not space
        assert problem.validateParameterization(LEGAL) is None  # "late" is not required

    def test_an_unknown_field_is_refused(self) -> None:
        data = richProblem().model_dump(mode="json")
        data["colour"] = "red"

        with pytest.raises(ValidationError, match="colour"):
            Problem.model_validate(data)


class TestInvariants:
    def test_without_an_outcome_it_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="at least one outcome") as caught:
            Problem(search_space=richSpace(), outcomes=[])

        assert caught.value.component == "problem"
        assert caught.value.field_path == "outcomes"
        assert caught.value.suggestion is not None

    def test_two_outcomes_with_one_name_are_refused(self) -> None:
        both = [Outcome(name="y", type="continuous"), Outcome(name="y", type="binary")]

        with pytest.raises(ConfigError, match=r"duplicate outcome names: \['y'\]"):
            Problem(search_space=richSpace(), outcomes=both)

    def test_every_duplicated_name_is_reported_once(self) -> None:
        names = ["y", "z", "y", "z", "w"]
        both = [Outcome(name=name, type="continuous") for name in names]

        with pytest.raises(ConfigError, match=r"\['y', 'z'\]"):
            Problem(search_space=richSpace(), outcomes=both)

    @pytest.mark.parametrize(
        "constraint",
        [
            LinearConstraint({"temperature": 1.0, "temprature": 1.0}, 90.0, "<="),
            NonlinearConstraint("temperature * pressure", "<=", 90.0),
        ],
        ids=["linear", "nonlinear"],
    )
    def test_a_constraint_about_a_parameter_that_does_not_exist_is_refused(
        self, constraint: Any
    ) -> None:
        space = SearchSpace([Real("temperature", 20.0, 80.0)], [constraint])

        with pytest.raises(ConfigError, match=r"not among the parameters") as caught:
            Problem(search_space=space, outcomes=outcomes())

        assert caught.value.field_path == "search_space.constraints"
        assert "temprature" in caught.value.message or "pressure" in caught.value.message

    def test_a_constraint_may_name_a_fixed_or_a_derived_parameter(self) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 1.0), Fixed("scale", 3), Derived("twice", "x * 2", "float")],
            [
                NonlinearConstraint("x * scale", "<=", 2.0),
                NonlinearConstraint("twice", "<=", 1.5),
            ],
        )

        Problem(search_space=space, outcomes=outcomes())

    def test_the_names_are_checked_when_a_document_is_loaded_too(self) -> None:
        data = richProblem().model_dump(mode="json")
        data["search_space"]["constraints"].append(
            {
                "type": "linear",
                "coefficients": {"nowhere": 1.0},
                "bound": 1.0,
                "comparator": "<=",
            }
        )

        with pytest.raises(ConfigError, match="nowhere"):
            Problem.model_validate(data)


class TestSerialization:
    def test_a_rich_problem_round_trips_through_json(self) -> None:
        problem = richProblem()

        rebuilt = Problem.model_validate_json(problem.model_dump_json())

        # `Real` and `Fixed` are sugar: they come back as the `Range` and `Choice` they stand
        # for (specification 5.3), so the dump is equal from the start and the object once it
        # has been loaded.
        assert rebuilt.model_dump() == problem.model_dump()
        assert Problem.model_validate_json(rebuilt.model_dump_json()) == rebuilt

    def test_it_round_trips_through_a_dictionary(self) -> None:
        problem = richProblem()

        rebuilt = Problem.model_validate(problem.model_dump())

        assert rebuilt.model_dump() == problem.model_dump()
        assert Problem.model_validate(rebuilt.model_dump()) == rebuilt

    def test_the_json_form_is_plain_data_that_survives_json_dumps(self) -> None:
        problem = richProblem()
        data = problem.model_dump(mode="json")

        assert json.loads(json.dumps(data)) == data
        rebuilt = Problem.model_validate(json.loads(json.dumps(data)))
        assert rebuilt.model_dump(mode="json") == data

    def test_the_parts_come_back_as_their_registered_types(self) -> None:
        rebuilt = Problem.model_validate_json(richProblem().model_dump_json())

        kinds = {type(parameter) for parameter in rebuilt.search_space.parameters}
        assert kinds == {Range, Choice, DerivedType}
        assert [type(c) for c in rebuilt.search_space.constraints] == [
            LinearConstraint,
            LinearConstraint,
            NonlinearConstraint,
        ]
        assert all(isinstance(outcome, Outcome) for outcome in rebuilt.outcomes)
        assert rebuilt.outcomes[0].description == "Mass fraction."

    def test_a_reloaded_problem_judges_parameterizations_as_the_original_does(self) -> None:
        original = richProblem()
        rebuilt = Problem.model_validate_json(original.model_dump_json())
        illegal = {**LEGAL, "temperature": 95.0, "a": 0.9}

        for problem in (original, rebuilt):
            problem.validateParameterization(LEGAL)
            assert "temperature" in str(refused(problem, illegal))
        assert str(refused(original, illegal)) == str(refused(rebuilt, illegal))

    def test_the_facts_survive_the_round_trip(self) -> None:
        original = richProblem()

        rebuilt = Problem.model_validate_json(original.model_dump_json())

        assert rebuilt.facts() == original.facts()

    def test_an_outcome_kind_that_is_not_registered_is_refused_with_the_list(self) -> None:
        data = richProblem().model_dump(mode="json")
        data["outcomes"][0]["type"] = "ordinal"

        with pytest.raises(ConfigError, match=r"ordinal.*binary, continuous"):
            Problem.model_validate(data)

    def test_a_parameter_type_that_is_not_registered_is_refused_with_the_list(self) -> None:
        data = richProblem().model_dump(mode="json")
        data["search_space"]["parameters"][0]["type"] = "spiral"

        with pytest.raises(ConfigError, match=r"spiral.*choice, derived, range"):
            Problem.model_validate(data)

    def test_a_constraint_type_that_is_not_registered_is_refused_with_the_list(self) -> None:
        data = richProblem().model_dump(mode="json")
        data["search_space"]["constraints"][0]["type"] = "quadratic"

        with pytest.raises(ConfigError, match=r"quadratic.*linear, nonlinear"):
            Problem.model_validate(data)

    def test_a_document_without_outcomes_is_refused(self) -> None:
        data = richProblem().model_dump(mode="json")
        data["outcomes"] = []

        with pytest.raises(ConfigError, match="at least one outcome"):
            Problem.model_validate(data)


class TestALegalParameterization:
    def test_the_example_is_accepted(self) -> None:
        assert richProblem().validateParameterization(LEGAL) is None

    def test_the_branch_that_a_choice_switches_on_is_accepted_with_its_parameter(self) -> None:
        richProblem().validateParameterization({**LEGAL, "model": "tree", "depth": 4})

    def test_a_computed_parameter_may_be_left_out_or_given(self) -> None:
        problem = richProblem()

        problem.validateParameterization(LEGAL)
        problem.validateParameterization({**LEGAL, "total": 55.0})

    def test_whole_floats_are_accepted_for_integers_and_integers_for_floats(self) -> None:
        richProblem().validateParameterization({**LEGAL, "cycles": 5.0, "temperature": 50})

    def test_a_value_on_a_bound_is_accepted(self) -> None:
        richProblem().validateParameterization({**LEGAL, "temperature": 20.0, "cycles": 1})

    def test_the_parameterization_is_not_changed(self) -> None:
        before = copy.deepcopy(LEGAL)

        richProblem().validateParameterization(LEGAL)

        assert before == LEGAL

    def test_any_mapping_is_accepted(self) -> None:
        from types import MappingProxyType

        richProblem().validateParameterization(MappingProxyType(LEGAL))

    def test_a_mixture_that_sums_to_one_up_to_rounding(self) -> None:
        third = 1.0 / 3.0

        richProblem().validateParameterization({**LEGAL, "a": third, "b": third, "c": third})


class TestAnIllegalParameterization:
    """One test per kind of fault. The error names the parameter and says how to fix it."""

    @pytest.mark.parametrize("value", [None, [("a", 1)], "a=1", 3, ["a"]])
    def test_it_is_not_a_mapping(self, value: object) -> None:
        error = refused(richProblem(), value)

        assert "must be a mapping" in error.message
        assert error.component == "problem"
        assert error.suggestion is not None

    def test_a_name_that_is_not_a_parameter(self) -> None:
        error = refused(richProblem(), {**LEGAL, "tempature": 50.0})

        assert "'tempature' is not a parameter of the search space" in error.message
        assert error.field_path == "tempature"
        assert error.component == "problem"
        assert "spelling" in str(error.suggestion)

    def test_a_name_that_is_not_a_string(self) -> None:
        error = refused(richProblem(), {**LEGAL, 7: 1.0})

        assert "7" in error.message

    def test_a_value_above_a_bound(self) -> None:
        error = refused(richProblem(), {**LEGAL, "temperature": 95.0})

        assert "95.0" in error.message and "range 'temperature'" in error.message
        assert "outside the bounds 20.0 to 80.0" in error.message
        assert error.field_path == "temperature"
        assert "from 20.0 to 80.0" in str(error.suggestion)

    @pytest.mark.parametrize(
        ("name", "value", "fragment"),
        [
            ("temperature", "50", "not a finite number"),
            ("temperature", True, "not a finite number"),
            ("temperature", float("nan"), "not a finite number"),
            ("cycles", 2.5, "not a whole number"),
            ("ratio", 0.35, "not on the grid"),
            ("model", "forest", "not one of its values"),
            ("stirred", 1, "not one of its values"),
            ("scale", 4, "not one of its values"),
            ("total", "lots", "derived parameter 'total'"),
        ],
    )
    def test_a_value_the_parameter_does_not_accept(
        self, name: str, value: object, fragment: str
    ) -> None:
        error = refused(richProblem(), {**LEGAL, name: value})

        assert fragment in error.message
        assert error.field_path == name

    def test_a_missing_parameter(self) -> None:
        legal = {k: v for k, v in LEGAL.items() if k != "temperature"}

        error = refused(richProblem(), legal)

        assert "The parameter 'temperature' has no value" in error.message
        assert error.field_path == "temperature"
        assert error.suggestion == "give it a value"

    def test_a_missing_fixed_parameter(self) -> None:
        legal = {k: v for k, v in LEGAL.items() if k != "scale"}

        assert "'scale' has no value" in refused(richProblem(), legal).message

    def test_a_missing_parameter_of_a_branch_that_is_on(self) -> None:
        error = refused(richProblem(), {**LEGAL, "model": "tree"})  # no depth

        assert "The parameter 'depth' has no value" in error.message

    def test_a_parameter_of_a_branch_that_is_off(self) -> None:
        error = refused(richProblem(), {**LEGAL, "depth": 4})  # model is 'linear'

        assert "The parameter 'depth' is given, but it only exists when 'model' is 'tree'" in (
            error.message
        )
        assert "leave it out" in str(error.suggestion)

    def test_every_problem_of_a_stage_is_reported_together(self) -> None:
        legal = {k: v for k, v in LEGAL.items() if k != "temperature"}
        bad = {**legal, "cycles": 2.5, "ratio": 0.35, "model": "forest", "q": 1}

        error = refused(richProblem(), bad)

        assert "4 problems" in error.message or "5 problems" in error.message
        for fragment in ("'q'", "cycles", "ratio", "model"):
            assert fragment in error.message
        assert error.field_path == "q"  # the unknown names come first
        assert error.suggestion == "fix the problems listed above"

    def test_a_value_fault_is_reported_before_a_missing_parameter(self) -> None:
        # Which parameters are missing is not known until the given ones are legal: the choice
        # that switches a branch on may be the one with the bad value.
        legal = {k: v for k, v in LEGAL.items() if k != "temperature"}

        error = refused(richProblem(), {**legal, "cycles": 2.5})

        assert "cycles" in error.message
        assert "has no value" not in error.message

    def test_a_missing_parameter_is_reported_before_a_violated_constraint(self) -> None:
        legal = {k: v for k, v in LEGAL.items() if k != "stirred"}

        error = refused(richProblem(), {**legal, "a": 0.9})  # the mixture would not sum to one

        assert "'stirred' has no value" in error.message
        assert "does not hold" not in error.message

    def test_the_error_is_a_planexs_error_and_a_value_error(self) -> None:
        error = refused(richProblem(), {**LEGAL, "temperature": 95.0})

        assert isinstance(error, PlanexsError)
        assert isinstance(error, ValueError)


class TestConditionalParameters:
    def chain(self) -> Problem:
        space = SearchSpace(
            [
                Categorical("a", ["off", "on"], dependent_parameters={"on": ["b"]}),
                Categorical("b", ["x", "y"], dependent_parameters={"y": ["c"]}),
                Real("c", 0.0, 1.0),
            ]
        )
        return Problem(search_space=space, outcomes=outcomes())

    @pytest.mark.parametrize(
        "legal",
        [
            {"a": "off"},
            {"a": "on", "b": "x"},
            {"a": "on", "b": "y", "c": 0.5},
        ],
    )
    def test_each_level_of_a_chain_of_choices(self, legal: dict[str, Any]) -> None:
        self.chain().validateParameterization(legal)

    @pytest.mark.parametrize(
        ("illegal", "fragment"),
        [
            ({"a": "off", "b": "x"}, "'b' is given, but it only exists when 'a' is 'on'"),
            (
                {"a": "on", "b": "x", "c": 0.5},
                "'c' is given, but it only exists when 'b' is 'y'",
            ),
            ({"a": "on", "b": "y"}, "'c' has no value"),
            ({"a": "off", "c": 0.5}, "'c' is given"),
            ({"a": "on"}, "'b' has no value"),
        ],
    )
    def test_a_branch_that_is_on_when_it_should_be_off_or_the_other_way(
        self, illegal: dict[str, Any], fragment: str
    ) -> None:
        assert fragment in refused(self.chain(), illegal).message

    def test_a_chain_declared_in_the_reverse_order_is_still_followed_to_its_end(self) -> None:
        # The parameters are declared dependents first, so that one pass over them in
        # declaration order cannot find what the last choice switches on: the search for the
        # active ones has to go on until nothing changes.
        space = SearchSpace(
            [
                Real("c", 0.0, 1.0),
                Categorical("b", ["x", "y"], dependent_parameters={"y": ["c"]}),
                Categorical("a", ["off", "on"], dependent_parameters={"on": ["b"]}),
            ]
        )
        problem = Problem(search_space=space, outcomes=outcomes())

        problem.validateParameterization({"a": "on", "b": "y", "c": 0.5})
        problem.validateParameterization({"a": "on", "b": "x"})
        assert "'c' has no value" in refused(problem, {"a": "on", "b": "y"}).message
        assert "'c' is given" in refused(problem, {"a": "on", "b": "x", "c": 0.5}).message

    def test_a_parameter_that_two_choices_switch_on_is_active_if_either_does(self) -> None:
        space = SearchSpace(
            [
                Categorical("p", ["u", "v"], dependent_parameters={"u": ["shared"]}),
                Categorical("q", ["u", "v"], dependent_parameters={"u": ["shared"]}),
                Real("shared", 0.0, 1.0),
            ]
        )
        problem = Problem(search_space=space, outcomes=outcomes())

        problem.validateParameterization({"p": "u", "q": "v", "shared": 0.5})
        problem.validateParameterization({"p": "v", "q": "u", "shared": 0.5})
        problem.validateParameterization({"p": "u", "q": "u", "shared": 0.5})
        problem.validateParameterization({"p": "v", "q": "v"})
        assert "'shared' has no value" in refused(problem, {"p": "u", "q": "v"}).message
        assert (
            "'shared' is given"
            in refused(problem, {"p": "v", "q": "v", "shared": 0.5}).message
        )

    def test_the_message_names_every_value_that_switches_a_parameter_on(self) -> None:
        space = SearchSpace(
            [
                Categorical("p", ["u", "v"], dependent_parameters={"u": ["shared"]}),
                Categorical("q", ["u", "v"], dependent_parameters={"v": ["shared"]}),
                Real("shared", 0.0, 1.0),
            ]
        )
        problem = Problem(search_space=space, outcomes=outcomes())

        message = refused(problem, {"p": "v", "q": "u", "shared": 0.5}).message

        assert "when 'p' is 'u' or when 'q' is 'v'" in message

    def test_a_choice_whose_value_is_illegal_does_not_decide_what_is_missing(self) -> None:
        error = refused(self.chain(), {"a": "maybe"})

        assert "not one of its values" in error.message
        assert "has no value" not in error.message


class TestConstraints:
    @pytest.mark.parametrize(
        ("change", "fragment"),
        [
            ({"a": 0.5, "b": 0.5, "c": 0.5}, "violated by"),  # the mixture sums to 1.5
            ({"temperature": 80.0, "cycles": 10}, "violated by"),  # 80 + 50 > 100
            ({"temperature": 80.0, "cycles": 1, "rate": 0.9}, "violated by"),  # 72 > 30
        ],
        ids=["linear equality", "linear inequality", "nonlinear"],
    )
    def test_a_constraint_that_does_not_hold(
        self, change: dict[str, Any], fragment: str
    ) -> None:
        error = refused(richProblem(), {**LEGAL, **change})

        assert "does not hold" in error.message and fragment in error.message
        assert error.field_path is not None
        assert (
            error.suggestion is not None
            and "change the parameters it names" in error.suggestion
        )

    def test_the_message_shows_the_constraint(self) -> None:
        error = refused(
            richProblem(), {**LEGAL, "temperature": 80.0, "cycles": 1, "rate": 0.9}
        )

        assert "NonlinearConstraint(expression='rate * temperature'" in error.message
        assert error.field_path == "rate"

    def test_every_violated_constraint_is_reported_together(self) -> None:
        bad = {**LEGAL, "a": 0.5, "b": 0.5, "temperature": 80.0, "cycles": 1, "rate": 0.9}

        error = refused(richProblem(), bad)

        assert error.message.count("does not hold") == 2
        assert "2 problems" in error.message

    def test_a_point_on_a_bound_holds(self) -> None:
        # 50 + 5 * 10 = 100, exactly on the bound of the linear constraint.
        richProblem().validateParameterization({**LEGAL, "temperature": 50.0, "cycles": 10})

    def test_an_undefined_expression_is_reported_as_undefined(self) -> None:
        space = SearchSpace(
            [Real("x", -1.0, 1.0)], [NonlinearConstraint("sqrt(x)", ">=", 0.0)]
        )
        problem = Problem(search_space=space, outcomes=outcomes())

        problem.validateParameterization({"x": 0.25})
        error = refused(problem, {"x": -0.5})

        assert "its value is undefined here" in error.message

    def test_a_constraint_about_an_inactive_parameter_does_not_apply(self) -> None:
        space = SearchSpace(
            [
                Categorical(
                    "model", ["linear", "tree"], dependent_parameters={"tree": ["depth"]}
                ),
                Integer("depth", 1, 8),
            ],
            [NonlinearConstraint("depth", "<=", 4.0)],
        )
        problem = Problem(search_space=space, outcomes=outcomes())

        problem.validateParameterization({"model": "linear"})  # depth does not exist
        problem.validateParameterization({"model": "tree", "depth": 4})
        assert "does not hold" in refused(problem, {"model": "tree", "depth": 6}).message

    def test_a_constraint_about_a_computed_parameter_that_was_not_given_cannot_be_judged(
        self,
    ) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 10.0), Derived("twice", "x * 2", "float")],
            [NonlinearConstraint("twice", "<=", 12.0)],
        )
        problem = Problem(search_space=space, outcomes=outcomes())

        problem.validateParameterization({"x": 5.0, "twice": 10.0})
        assert "cannot be judged" in refused(problem, {"x": 5.0}).message
        assert "does not hold" in refused(problem, {"x": 5.0, "twice": 14.0}).message

    def test_a_constraint_about_a_parameter_that_is_not_a_number_cannot_be_evaluated(
        self,
    ) -> None:
        space = SearchSpace(
            [Categorical("solvent", ["water", "ethanol"])],
            [LinearConstraint({"solvent": 1.0}, bound=1.0, comparator="<=")],
        )
        problem = Problem(search_space=space, outcomes=outcomes())

        error = refused(problem, {"solvent": "water"})

        assert "cannot be evaluated" in error.message
        assert "needs a number for 'solvent'" in error.message


class TestTheTolerances:
    def space(self) -> Problem:
        space = SearchSpace(
            [Real("a", 0.0, 2.0), Real("ratio", 0.0, 1.0, step_size=0.1)],
            [LinearConstraint({"a": 1.0}, bound=1.0, comparator="<=")],
        )
        return Problem(search_space=space, outcomes=outcomes())

    def test_a_violation_within_the_default_tolerance_is_accepted(self) -> None:
        self.space().validateParameterization(
            {"a": 1.0000005, "ratio": 0.3}
        )  # 5e-7 of the scale

    def test_a_violation_above_it_is_refused(self) -> None:
        assert "does not hold" in refused(self.space(), {"a": 1.00001, "ratio": 0.3}).message

    def test_a_point_that_holds_is_accepted_even_with_no_tolerance_at_all(self) -> None:
        exact = ParameterizationCheckConfig(constraint_tolerance=0.0, grid_tolerance=0.0)

        self.space().validateParameterization({"a": 1.0, "ratio": 0.5}, exact)  # on the bound
        self.space().validateParameterization({"a": 0.25, "ratio": 0.0}, exact)

    def test_the_constraint_tolerance_is_a_setting(self) -> None:
        problem = self.space()
        slightly_over = {"a": 1.0000005, "ratio": 0.3}

        assert (
            "does not hold"
            in refused(problem, slightly_over, constraint_tolerance=0.0).message
        )
        problem.validateParameterization(
            {"a": 1.01, "ratio": 0.3}, ParameterizationCheckConfig(constraint_tolerance=0.1)
        )

    def test_the_grid_tolerance_reaches_the_parameters(self) -> None:
        problem = self.space()
        rounded = {"a": 0.5, "ratio": 0.30000000000000004}  # 3 * 0.1 in floating point

        problem.validateParameterization(rounded)
        assert "not on the grid" in refused(problem, rounded, grid_tolerance=0.0).message

    def test_the_options_default_to_the_defaults_of_the_config(self) -> None:
        problem = self.space()
        parameterization = {"a": 1.0000005, "ratio": 0.3}

        problem.validateParameterization(parameterization)
        problem.validateParameterization(parameterization, ParameterizationCheckConfig())


class ToyTypes:
    """A parameter type that the domain has never heard of, defined as a plugin would."""

    @staticmethod
    def defineCyclic() -> type[Parameter]:
        class CyclicConfig(ComponentConfig):
            type: Literal["cyclic"] = Field(default="cyclic", description="Always 'cyclic'.")
            name: str = Field(description="The parameter's name.")
            period: float = Field(description="The length of one turn.")
            default: float | None = Field(default=None, description="The optional default.")

        class Cyclic(Parameter):
            period: float

            def __init__(self, name: str, period: float, default: float | None = None) -> None:
                super().__init__(name=name, period=period, default=default)

            def _validate_default(self) -> None:
                return None

            def validateValue(self, value: object, options: Any = None) -> None:
                if not isinstance(value, (int, float)) or not 0.0 <= value < self.period:
                    raise ParameterValueError(
                        f"The angle {value!r} of {self.name!r} is not within one turn of "
                        f"{self.period!r}.",
                        component=f"parameter {self.name!r}",
                        field_path=self.name,
                    )

            def toDict(self) -> dict[str, Any]:
                return {
                    "type": "cyclic",
                    "name": self.name,
                    "period": self.period,
                    "default": self.default,
                }

        registerParameter("cyclic", config=CyclicConfig, description="A wrapping angle.")(
            Cyclic
        )
        return Cyclic


class TestAParameterTypeThatThePackageDoesNotKnow:
    def test_it_is_checked_through_its_own_rule_and_saved_and_reloaded(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        Cyclic = ToyTypes.defineCyclic()
        space = SearchSpace([Cyclic("angle", 360.0), Real("x", 0.0, 1.0)])
        problem = Problem(search_space=space, outcomes=outcomes())

        problem.validateParameterization({"angle": 90.0, "x": 0.5})
        error = refused(problem, {"angle": 360.0, "x": 0.5})
        assert "is not within one turn of 360.0" in error.message
        assert error.field_path == "angle"
        assert problem.facts().parameter_types == ("cyclic", "range")
        rebuilt = Problem.model_validate_json(problem.model_dump_json())
        assert rebuilt.model_dump() == problem.model_dump()
        assert type(rebuilt.search_space.parameters[0]) is Cyclic
        rebuilt.validateParameterization({"angle": 90.0, "x": 0.5})
