"""Unit tests for `Parameter.toDict` and `parameterFromDict`.

Guards the bug that motivated explicit serialization: `SearchSpace` holds a
list of the abstract `Parameter` type, so Pydantic's own handling dropped the
subclass fields on save and could not rebuild an abstract class on load.
"""

from __future__ import annotations

import json

import pytest

from planexs import Boolean, Categorical, Choice, Derived, Fixed, Integer, Range, Real
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


class TestKindTag:
    def test_every_kind_is_tagged(self) -> None:
        assert Real("x", 0.0, 1.0).toDict()["kind"] == "range"
        assert Categorical("c", ["a", "b"]).toDict()["kind"] == "choice"
        assert Derived("d", "x + 1", parameter_type="float").toDict()["kind"] == "derived"

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            parameterFromDict({"kind": "mystery", "name": "x"})

    def test_missing_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            parameterFromDict({"name": "x"})
