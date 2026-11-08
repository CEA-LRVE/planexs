"""Unit tests for the check that `SearchSpace` makes of each tag when it is written.

Parameters and constraints are checked by the same code, so every case here runs for both: a
document whose parameter or constraint has a type that nobody registered, or whose form has
no `type` at all, could be written but never read back, and is refused when it is written
(specification 3.6, 5.12). Pydantic reports the error as a `PydanticSerializationError` that
carries the text of the original, hints included.

The types below are never registered: they only write a form that the registry does not know.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_core import PydanticSerializationError

from planexs import LinearConstraint, Range, SearchSpace
from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.parameters.Parameter import Parameter


class UnknownParameter(Parameter):
    def _validate_default(self) -> None:
        return None

    def toDict(self) -> dict[str, Any]:
        return {"type": "cyclic", "name": self.name}


class UntaggedParameter(Parameter):
    def _validate_default(self) -> None:
        return None

    def toDict(self) -> dict[str, Any]:
        return {"name": self.name}


class NonStringTagParameter(Parameter):
    def _validate_default(self) -> None:
        return None

    def toDict(self) -> dict[str, Any]:
        return {"type": 3, "name": self.name}


class UnknownConstraint(Constraint):
    def toDict(self) -> dict[str, Any]:
        return {"type": "ordering"}


class UntaggedConstraint(Constraint):
    def toDict(self) -> dict[str, Any]:
        return {}


class NonStringTagConstraint(Constraint):
    def toDict(self) -> dict[str, Any]:
        return {"type": None}


GOOD_PARAMETER = Range("ok", (0.0, 1.0))
OTHER_GOOD_PARAMETER = Range("fine", (0.0, 1.0))
GOOD_CONSTRAINT = LinearConstraint({"ok": 1.0}, bound=1.0, comparator="<=")
OTHER_GOOD_CONSTRAINT = LinearConstraint({"ok": 1.0}, bound=2.0, comparator=">=")


def withParameters(*parameters: Parameter) -> SearchSpace:
    return SearchSpace(parameters=list(parameters))


def withConstraints(*constraints: Constraint) -> SearchSpace:
    return SearchSpace(parameters=[GOOD_PARAMETER], constraints=list(constraints))


class Case:
    """What differs between the two kinds: the noun, the space builder and the faulty items."""

    def __init__(
        self,
        kind: str,
        make: Callable[..., SearchSpace],
        good: Parameter | Constraint,
        other_good: Parameter | Constraint,
        unknown: Parameter | Constraint,
        untagged: Parameter | Constraint,
        non_string: Parameter | Constraint,
        unknown_label: str,
        registered: str,
    ) -> None:
        self.kind = kind
        self.make = make
        self.good = good
        self.other_good = other_good
        self.unknown = unknown
        self.untagged = untagged
        self.non_string = non_string
        self.unknown_label = unknown_label
        self.registered = registered


CASES = {
    "parameter": Case(
        "parameter",
        withParameters,
        GOOD_PARAMETER,
        OTHER_GOOD_PARAMETER,
        UnknownParameter(name="angle"),
        UntaggedParameter(name="tilt"),
        NonStringTagParameter(name="twist"),
        "Unknown parameter 'cyclic'",
        "choice, derived, range",
    ),
    "constraint": Case(
        "constraint",
        withConstraints,
        GOOD_CONSTRAINT,
        OTHER_GOOD_CONSTRAINT,
        UnknownConstraint(),
        UntaggedConstraint(),
        NonStringTagConstraint(),
        "Unknown constraint 'ordering'",
        "linear, nonlinear",
    ),
}
EVERY_KIND = pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
BOTH_WAYS = pytest.mark.parametrize("how", ["model_dump", "model_dump_json"])


class TestATagThatNobodyRegistered:
    @EVERY_KIND
    @BOTH_WAYS
    def test_it_is_refused_and_the_registered_types_are_listed(
        self, case: Case, how: str
    ) -> None:
        space = case.make(case.unknown)

        with pytest.raises(PydanticSerializationError) as caught:
            getattr(space, how)()

        text = str(caught.value)
        assert case.unknown_label in text
        assert f"Registered names: {case.registered}." in text
        assert f"Component: {case.kind}" in text
        assert "import the module or plugin that registers it" in text

    @EVERY_KIND
    def test_a_tag_that_is_not_a_string_is_refused_the_same_way(self, case: Case) -> None:
        space = case.make(case.non_string)

        with pytest.raises(PydanticSerializationError, match=f"Unknown {case.kind}"):
            space.model_dump()


class TestAFormWithNoTypeEntry:
    @EVERY_KIND
    @BOTH_WAYS
    def test_it_is_refused_with_the_hints_of_the_error(self, case: Case, how: str) -> None:
        space = case.make(case.untagged)

        with pytest.raises(PydanticSerializationError) as caught:
            getattr(space, how)()

        text = str(caught.value)
        assert f"{type(case.untagged).__name__}.toDict() wrote no 'type' entry" in text
        assert "so it could not be read back" in text
        assert f"Component: {case.kind}" in text
        assert "Field: type" in text
        assert f"write the registered name of the {case.kind} type under 'type'" in text

    def test_a_parameter_is_named_in_the_message(self) -> None:
        space = withParameters(UntaggedParameter(name="angle"))

        with pytest.raises(PydanticSerializationError, match="for the parameter 'angle'"):
            space.model_dump()

    def test_a_constraint_has_no_name_and_the_message_says_what_it_is(self) -> None:
        space = withConstraints(UntaggedConstraint())

        with pytest.raises(
            PydanticSerializationError, match="for the constraint, so it could"
        ):
            space.model_dump()


class TestWhichFaultIsReported:
    @EVERY_KIND
    def test_the_first_fault_is_the_one_that_is_raised(self, case: Case) -> None:
        space = case.make(case.good, case.unknown, case.untagged)

        with pytest.raises(PydanticSerializationError) as caught:
            space.model_dump()

        assert case.unknown_label in str(caught.value)
        assert "wrote no 'type' entry" not in str(caught.value)

    @EVERY_KIND
    def test_what_precedes_the_fault_does_not_hide_it(self, case: Case) -> None:
        space = case.make(case.good, case.other_good, case.untagged)

        with pytest.raises(PydanticSerializationError, match="wrote no 'type' entry"):
            space.model_dump()

    def test_the_parameters_are_written_before_the_constraints(self) -> None:
        space = SearchSpace(
            parameters=[UnknownParameter(name="angle")], constraints=[UnknownConstraint()]
        )

        with pytest.raises(PydanticSerializationError, match="Unknown parameter 'cyclic'"):
            space.model_dump()


class TestWhatIsValid:
    @EVERY_KIND
    def test_a_valid_space_is_written_with_the_tags_of_its_types(self, case: Case) -> None:
        space = case.make(case.good)

        dumped = space.model_dump(mode="json")

        assert dumped["parameters"][0]["type"] == "range"
        if case.kind == "constraint":
            assert dumped["constraints"][0]["type"] == "linear"

    def test_an_empty_space_is_written(self) -> None:
        assert SearchSpace(parameters=[]).model_dump() == {"parameters": [], "constraints": []}

    def test_a_space_that_was_refused_can_be_written_once_the_fault_is_gone(self) -> None:
        space = withParameters(GOOD_PARAMETER)
        space.parameters.append(UnknownParameter(name="angle"))
        with pytest.raises(PydanticSerializationError):
            space.model_dump()

        space.parameters.pop()

        assert [p["name"] for p in space.model_dump()["parameters"]] == ["ok"]
