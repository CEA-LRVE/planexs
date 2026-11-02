"""Unit tests for `Parameter.toDict` and `parameterFromDict`.

Guards the bug that motivated explicit serialization: `SearchSpace` holds a
list of the abstract `Parameter` type, so Pydantic's own handling dropped the
subclass fields on save and could not rebuild an abstract class on load.
"""

from __future__ import annotations

import json

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
    Parameter,
    Range,
    Real,
    SearchSpace,
)
from planexs.core.ConfigError import ConfigError
from planexs.domain.parameters.parameterFromDict import parameterFromDict


def _roundTrip(parameter):  # type: ignore[no-untyped-def]
    # Through real JSON text, so tuples become lists and dict keys become strings,
    # exactly as after a save()/load().
    return parameterFromDict(json.loads(json.dumps(parameter.toDict())))


class TestRangeRoundTrip:
    def test_real_comes_back_as_an_equivalent_range(self) -> None:
        original = Real("temperature", 20.0, 120.0, default=50.0)
        reloaded = _roundTrip(original)

        assert type(reloaded) is Range
        assert reloaded.name == "temperature"
        assert reloaded.bounds == (20.0, 120.0)
        assert reloaded.parameter_type == "float"
        assert reloaded.default == 50.0

    def test_integer_keeps_its_int_type_and_step(self) -> None:
        reloaded = _roundTrip(Integer("n", 0, 100, step_size=5, default=10))
        assert reloaded.parameter_type == "int"
        assert reloaded.step_size == 5.0
        assert reloaded.default == 10

    def test_log_scaling_survives(self) -> None:
        reloaded = _roundTrip(Real("lr", 1e-5, 1e-1, scaling="log"))
        assert reloaded.scaling == "log"


class TestChoiceRoundTrip:
    def test_categorical_with_dependent_parameters(self) -> None:
        original = Categorical(
            "model_type",
            ["mlp", "cnn"],
            dependent_parameters={"mlp": ["hidden_units"], "cnn": ["num_filters"]},
            default="mlp",
        )
        reloaded = _roundTrip(original)

        assert type(reloaded) is Choice
        assert reloaded.values == ["mlp", "cnn"]
        assert reloaded.is_ordered is False
        assert reloaded.dependent_parameters == {
            "mlp": ["hidden_units"],
            "cnn": ["num_filters"],
        }
        assert reloaded.default == "mlp"

    def test_non_string_dependent_keys_keep_their_type(self) -> None:
        # JSON object keys are always strings; the pair encoding must keep
        # an int key an int, or it would no longer be among `values`.
        original = Choice(
            "depth", [1, 2, 3], parameter_type="int", dependent_parameters={2: ["skip"]}
        )
        reloaded = _roundTrip(original)
        assert reloaded.dependent_parameters == {2: ["skip"]}

    def test_boolean_keeps_bool_values(self) -> None:
        reloaded = _roundTrip(Boolean("use_catalyst", default=True))
        assert reloaded.values == [True, False]
        assert reloaded.parameter_type == "bool"
        assert reloaded.default is True

    def test_fixed_comes_back_as_a_single_value_choice(self) -> None:
        reloaded = _roundTrip(Fixed("device", "cpu"))
        assert type(reloaded) is Choice
        assert reloaded.values == ["cpu"]
        assert reloaded.default == "cpu"


class TestDerivedRoundTrip:
    def test_derived(self) -> None:
        reloaded = _roundTrip(Derived("total", "a + b", parameter_type="float"))
        assert type(reloaded) is Derived
        assert reloaded.expression == "a + b"
        assert reloaded.default is None


class TestTypeTag:
    def test_every_type_is_tagged(self) -> None:
        assert Real("x", 0.0, 1.0).toDict()["type"] == "range"
        assert Categorical("c", ["a", "b"]).toDict()["type"] == "choice"
        assert Derived("d", "x + 1", parameter_type="float").toDict()["type"] == "derived"

    def test_unknown_type_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="type"):
            parameterFromDict({"type": "mystery", "name": "x"})

    def test_missing_type_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="type"):
            parameterFromDict({"name": "x"})


# One example of every shape of every registered type, written so that a lossy round trip
# would show: floats that have no short decimal form, an int default on a float range, keys
# of mixed types among the conditional parameters, a log scale and a step.
EXACT_EXAMPLES: list[Parameter] = [
    Range("x", (0.0, 1.0)),
    Range("x", (0.1 + 0.2, 1e-9 + 1.0), default=0.5),
    Range("x", (-5.0, 5.0), step_size=0.25, default=-1.25),
    Range("x", (1e-5, 1e5), scaling="log", default=1.0),
    Range("n", (1.0, 64.0), parameter_type="int", step_size=1.0, scaling="log", default=8),
    Choice("c", ["a", "b"], parameter_type="str"),
    Choice("c", ["low", "mid", "high"], parameter_type="str", is_ordered=True, default="mid"),
    Choice("c", [1, 2, 3], parameter_type="int", is_ordered=None),
    Choice("c", [0.1, 0.2 + 0.1], parameter_type="float", is_ordered=False, default=0.1),
    Choice("c", [True, False], parameter_type="bool", is_ordered=False, default=True),
    Choice(
        "c",
        [1, 2, 3],
        parameter_type="int",
        is_ordered=True,
        dependent_parameters={1: ["a"], 2: ["a", "b"], 3: []},
    ),
    Choice(
        "c",
        [True, False],
        parameter_type="bool",
        dependent_parameters={True: ["on_only"]},
    ),
    Choice(
        "c",
        [0.5, 1.5],
        parameter_type="float",
        dependent_parameters={0.5: ["half"], 1.5: ["x", "y"]},
    ),
    Derived("d", "a + b", parameter_type="float"),
    Derived("d", "a > 0", parameter_type="bool"),
    Derived("d", "round(a)", parameter_type="int"),
]


class TestTheRoundTripIsExact:
    @pytest.mark.parametrize(
        "original", EXACT_EXAMPLES, ids=lambda p: f"{p.toDict()['type']}-{p.name}"
    )
    def test_a_parameter_comes_back_equal_and_of_the_same_type(
        self, original: Parameter
    ) -> None:
        reloaded = _roundTrip(original)

        assert type(reloaded) is type(original)
        assert reloaded == original
        assert reloaded.toDict() == original.toDict()

    @pytest.mark.parametrize(
        "original", EXACT_EXAMPLES, ids=lambda p: f"{p.toDict()['type']}-{p.name}"
    )
    def test_a_second_round_trip_changes_nothing(self, original: Parameter) -> None:
        once = _roundTrip(original)
        twice = _roundTrip(once)

        assert json.dumps(twice.toDict(), sort_keys=True) == json.dumps(
            original.toDict(), sort_keys=True
        )

    def test_a_whole_search_space_comes_back_equal(self) -> None:
        space = SearchSpace(
            parameters=[
                Range("a", (0.0, 1.0), default=0.25),
                Range("b", (1e-3, 1e3), scaling="log"),
                Range("n", (1.0, 8.0), parameter_type="int", step_size=1.0),
                Choice(
                    "m",
                    ["mlp", "cnn"],
                    parameter_type="str",
                    dependent_parameters={"mlp": ["n"]},
                ),
                Choice("flag", [True, False], parameter_type="bool", is_ordered=False),
                Derived("total", "a + b", parameter_type="float"),
            ],
            constraints=[
                LinearConstraint({"a": 1.0, "b": 2.0}, bound=3.0, comparator="<="),
                NonlinearConstraint("a * b", "<=", 50.0),
            ],
        )

        reloaded = SearchSpace.model_validate_json(space.model_dump_json())

        assert reloaded == space

    def test_every_convenience_constructor_reloads_as_its_base_type(self) -> None:
        sugar: list[Parameter] = [
            Real("r", 0.0, 1.0, default=0.5),
            Integer("i", 1, 9, step_size=2, default=3),
            Categorical("k", ["u", "v"], default="v"),
            Boolean("flag", default=False),
            Fixed("f", 3),
        ]
        for original in sugar:
            reloaded = _roundTrip(original)
            assert type(reloaded) in (Range, Choice)
            assert reloaded.toDict() == original.toDict()
