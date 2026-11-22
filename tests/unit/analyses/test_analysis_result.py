"""Unit tests for `AnalysisResult`, the typed, serializable result of an analysis (roadmap task
P7-01, specification 5.10, 6).

What is tested: what every result says about itself, that it is immutable at any depth, and
the serialization round trip: to plain JSON-compatible data and back, exactly. The results of
the real analyses (the best point, the Pareto front, a prediction grid) arrive with them; a
fake result with the kinds of field they will have stands for them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from analysis_fakes import FakeBand, FakeResult
from pydantic import BaseModel, ValidationError

from planexs.analyses.AnalysisResult import AnalysisResult


def aResult(**fields: Any) -> FakeResult:
    """A result with every kind of field, and what is not given as the defaults say."""
    defaults: dict[str, Any] = {
        "analysis": "fake_analysis_v1",
        "history_version": 4,
        "best_id": 2,
        "counts": (3, 1, 0, 2),
        "values": (0.5, 1.25, -3.0),
        "samples": (0.1, 0.2),
        "band": FakeBand(low=-3.0, high=1.25),
        "outcomes": ("yield",),
        "fit_observations": 5,
    }
    return FakeResult(**{**defaults, **fields})


class Loose(BaseModel):
    """A nested model that is not frozen: what a result must refuse to hold."""

    low: float


class FrozenWithList(BaseModel, frozen=True):
    """A nested model that is frozen but holds a list: still something that can be changed."""

    items: list[float]


class ListResult(AnalysisResult):
    """A result whose field is a list."""

    xs: list[float]


class DictResult(AnalysisResult):
    """A result whose field is a dictionary."""

    table: dict[str, float]


class SetResult(AnalysisResult):
    """A result whose field is a set."""

    tags: set[str]


class LooseResult(AnalysisResult):
    """A result that holds a model that is not frozen."""

    band: Loose


class NestedListResult(AnalysisResult):
    """A result that holds a list inside a tuple."""

    rows: tuple[list[float], ...]


class DeepResult(AnalysisResult):
    """A result that holds a frozen model that holds a list."""

    inner: FrozenWithList


class TestWhatEveryResultSays:
    def test_it_names_the_analysis_and_the_version_of_the_history(self) -> None:
        result = aResult()

        assert result.analysis == "fake_analysis_v1"
        assert result.history_version == 4

    def test_the_base_class_is_a_result_on_its_own(self) -> None:
        result = AnalysisResult(analysis="bare_v1", history_version=0)

        assert result.toDict() == {"analysis": "bare_v1", "history_version": 0}

    def test_a_subclass_adds_typed_fields(self) -> None:
        result = aResult(best_id=None, counts=(), values=(), samples=(), band=None)

        assert result.best_id is None
        assert result.band is None
        assert isinstance(result, AnalysisResult)

    def test_the_fields_of_the_subclass_are_validated_by_type(self) -> None:
        with pytest.raises(ValidationError, match="best_id"):
            aResult(best_id="the second one")

    @pytest.mark.parametrize("missing", ["analysis", "history_version"])
    def test_the_two_fields_of_the_base_are_required(self, missing: str) -> None:
        fields: dict[str, Any] = {"analysis": "bare_v1", "history_version": 1}
        del fields[missing]

        with pytest.raises(ValidationError, match=missing):
            AnalysisResult(**fields)

    @pytest.mark.parametrize("version", [-1, 1.5, "3", True, None])
    def test_a_history_version_is_a_whole_number_from_zero(self, version: Any) -> None:
        with pytest.raises(ValidationError, match="history_version"):
            AnalysisResult(analysis="bare_v1", history_version=version)

    def test_an_unknown_field_is_refused_with_its_name(self) -> None:
        with pytest.raises(ValidationError, match="extra_field"):
            AnalysisResult(analysis="bare_v1", history_version=0, extra_field=1)

    def test_every_field_is_described(self) -> None:
        assert all(field.description for field in AnalysisResult.model_fields.values())

    def test_results_are_compared_by_value(self) -> None:
        assert aResult() == aResult()
        assert aResult() != aResult(history_version=5)
        assert aResult() != aResult(values=(0.5,))


class TestItIsImmutable:
    def test_a_field_cannot_be_assigned(self) -> None:
        result = aResult()

        with pytest.raises(ValidationError, match="frozen"):
            result.best_id = 9  # type: ignore[misc]

    def test_a_field_cannot_be_added(self) -> None:
        result = aResult()

        with pytest.raises(ValidationError):
            result.extra = 1  # type: ignore[attr-defined]

    def test_a_field_cannot_be_deleted(self) -> None:
        result = aResult()

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            del result.best_id

    def test_a_sequence_is_read_as_a_tuple_that_cannot_grow(self) -> None:
        result = aResult(values=[1.0, 2.0])  # a list is accepted for a tuple field

        assert result.values == (1.0, 2.0)
        assert isinstance(result.values, tuple)

    def test_the_nested_model_is_frozen_too(self) -> None:
        result = aResult()

        assert result.band is not None
        with pytest.raises(ValidationError, match="frozen"):
            result.band.low = 0.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("cls", "fields", "what"),
        [
            (ListResult, {"xs": [1.0]}, "a list"),
            (DictResult, {"table": {"a": 1.0}}, "a dict"),
            (SetResult, {"tags": {"a"}}, "a set"),
            (LooseResult, {"band": Loose(low=1.0)}, "a Loose that is not frozen"),
            (NestedListResult, {"rows": ([1.0], [2.0])}, "a list"),
            (DeepResult, {"inner": FrozenWithList(items=[1.0])}, "a list"),
        ],
    )
    def test_a_field_that_holds_something_that_can_be_changed_is_refused(
        self, cls: type[AnalysisResult], fields: dict[str, Any], what: str
    ) -> None:
        with pytest.raises(ValidationError, match="can be changed") as caught:
            cls(analysis="bad_v1", history_version=0, **fields)

        message = str(caught.value)
        assert cls.__name__ in message
        assert what in message
        assert "tuple" in message

    def test_the_refusal_names_the_field(self) -> None:
        with pytest.raises(ValidationError, match="field 'rows'"):
            NestedListResult(analysis="bad_v1", history_version=0, rows=([1.0],))

    def test_what_is_refused_cannot_be_read_back_either(self) -> None:
        with pytest.raises(ValidationError, match="can be changed"):
            ListResult.fromDict({"analysis": "bad_v1", "history_version": 0, "xs": [1.0]})


class TestSerialization:
    def test_to_dict_gives_plain_data_that_json_writes(self) -> None:
        data = aResult().toDict()

        assert json.loads(json.dumps(data, allow_nan=False)) == data

    def test_to_dict_holds_only_json_types(self) -> None:
        def check(value: object) -> None:
            if isinstance(value, dict):
                assert all(isinstance(key, str) for key in value)
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            else:
                assert value is None or isinstance(value, (str, int, float, bool))

        check(aResult().toDict())

    def test_a_tuple_is_written_as_a_list_and_a_nested_model_as_a_dictionary(self) -> None:
        data = aResult().toDict()

        assert data["counts"] == [3, 1, 0, 2]
        assert data["band"] == {"low": -3.0, "high": 1.25}

    def test_each_call_gives_a_new_dictionary(self) -> None:
        result = aResult()

        first = result.toDict()
        first["best_id"] = 99
        first["values"].append(7.0)  # type: ignore[union-attr]

        assert result.toDict() == aResult().toDict()

    def test_the_round_trip_is_exact(self) -> None:
        result = aResult()

        assert FakeResult.fromDict(result.toDict()) == result

    def test_the_round_trip_through_a_json_string_is_exact(self) -> None:
        result = aResult()

        stored = json.loads(json.dumps(result.toDict(), allow_nan=False))

        again = FakeResult.fromDict(stored)
        assert again == result
        assert isinstance(again.values, tuple)
        assert isinstance(again.band, FakeBand)

    def test_the_round_trip_keeps_what_is_absent_absent(self) -> None:
        result = aResult(best_id=None, band=None, fit_observations=None, counts=(), values=())

        again = FakeResult.fromDict(json.loads(json.dumps(result.toDict())))

        assert again == result
        assert again.best_id is None
        assert again.band is None

    def test_the_round_trip_keeps_every_float_to_the_last_digit(self) -> None:
        values = (0.1 + 0.2, 1 / 3, 1e-300, 1.7976931348623157e308, -0.0)
        result = aResult(values=values)

        again = FakeResult.fromDict(json.loads(json.dumps(result.toDict())))

        assert again.values == values

    def test_the_round_trip_of_the_base_class(self) -> None:
        result = AnalysisResult(analysis="bare_v1", history_version=12)

        assert AnalysisResult.fromDict(result.toDict()) == result

    def test_a_result_read_back_is_a_result_of_the_class_it_is_read_with(self) -> None:
        data = aResult().toDict()

        assert type(FakeResult.fromDict(data)) is FakeResult

    def test_data_that_is_not_a_result_of_the_class_is_refused_with_the_field(self) -> None:
        data = aResult().toDict()
        del data["history_version"]

        with pytest.raises(ValidationError, match="history_version"):
            FakeResult.fromDict(data)

    def test_a_field_of_the_wrong_type_is_refused_with_its_name(self) -> None:
        data = aResult().toDict()
        data["counts"] = "many"

        with pytest.raises(ValidationError, match="counts"):
            FakeResult.fromDict(data)

    def test_a_field_the_class_does_not_have_is_refused_with_its_name(self) -> None:
        data = aResult().toDict()
        data["surprise"] = 1

        with pytest.raises(ValidationError, match="surprise"):
            FakeResult.fromDict(data)

    def test_a_nested_field_is_reported_with_its_path(self) -> None:
        data = aResult().toDict()
        data["band"] = {"low": "wide", "high": 1.0}

        with pytest.raises(ValidationError, match=r"band\.low"):
            FakeResult.fromDict(data)

    def test_from_dict_reads_any_mapping_and_leaves_it_alone(self) -> None:
        data = aResult().toDict()
        before = json.dumps(data, sort_keys=True)

        FakeResult.fromDict(data)

        assert json.dumps(data, sort_keys=True) == before


class TestNumbersMustBeFinite:
    """JSON has no nan and no infinity: a result that held one would not be the one stored."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_number_that_is_not_finite_is_refused_when_the_result_is_built(
        self, bad: float
    ) -> None:
        with pytest.raises(ValidationError, match="values"):
            aResult(values=(1.0, bad))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_number_that_is_not_finite_in_a_nested_model_is_refused(
        self, bad: float
    ) -> None:
        with pytest.raises(ValidationError):
            aResult(band=FakeBand(low=bad, high=1.0))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_number_that_is_not_finite_is_refused_when_the_result_is_read_back(
        self, bad: float
    ) -> None:
        data = aResult().toDict()
        data["samples"] = [0.5, bad]

        with pytest.raises(ValidationError, match="samples"):
            FakeResult.fromDict(data)
