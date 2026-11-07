"""Unit tests for `outcomeFromDict`: reading a stored outcome through the registry."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from planexs.core.ConfigError import ConfigError
from planexs.domain.Outcome import Outcome
from planexs.domain.outcomeFromDict import outcomeFromDict

EXAMPLES = [
    Outcome(name="yield", type="continuous"),
    Outcome(name="yield", type="continuous", description="Mass yield, in %."),
    Outcome(name="phase", type="binary"),
    Outcome(name="phase", type="binary", description="Liquid (0) or solid (1)."),
    Outcome(name="température", type="continuous", description="Accents and 温度 survive."),
    Outcome(name="a", type="continuous", description=""),
]


class TestTheRoundTripIsExact:
    @pytest.mark.parametrize("original", EXAMPLES, ids=lambda o: f"{o.type}-{o.name}")
    def test_an_outcome_comes_back_equal(self, original: Outcome) -> None:
        reloaded = outcomeFromDict(json.loads(json.dumps(original.toDict())))

        assert type(reloaded) is Outcome
        assert reloaded == original
        assert reloaded.toDict() == original.toDict()

    @pytest.mark.parametrize("original", EXAMPLES, ids=lambda o: f"{o.type}-{o.name}")
    def test_a_second_round_trip_changes_nothing(self, original: Outcome) -> None:
        once = outcomeFromDict(json.loads(json.dumps(original.toDict())))
        twice = outcomeFromDict(json.loads(json.dumps(once.toDict())))

        assert json.dumps(twice.toDict(), sort_keys=True) == json.dumps(
            original.toDict(), sort_keys=True
        )

    def test_the_form_a_person_writes_by_hand_is_read(self) -> None:
        # The form of specification 9.2: the description may be left out.
        assert outcomeFromDict({"name": "yield", "type": "continuous"}) == Outcome(
            name="yield", type="continuous"
        )

    def test_the_rebuilt_outcome_validates_values_as_its_kind_says(self) -> None:
        outcome = outcomeFromDict({"name": "phase", "type": "binary"})

        assert outcome.validateValue(True) == 1.0
        assert outcome.kind().allows_noise is False


class TestReadingErrors:
    def test_an_unknown_type_is_an_error_that_lists_the_registered_kinds(self) -> None:
        with pytest.raises(ConfigError) as caught:
            outcomeFromDict({"name": "grade", "type": "ordinal"})

        error = caught.value
        assert "Unknown outcome_kind type 'ordinal'" in error.message
        assert "Registered names: binary, continuous" in error.message
        assert error.field_path == "type"

    def test_a_missing_type_is_an_error_that_lists_the_registered_kinds(self) -> None:
        with pytest.raises(ConfigError, match="no 'type' field") as caught:
            outcomeFromDict({"name": "yield"})

        assert "Registered names: binary, continuous" in caught.value.message

    def test_a_form_that_names_the_kind_under_another_key_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="no 'type' field"):
            outcomeFromDict({"name": "yield", "kind": "continuous"})

    def test_a_type_that_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a string"):
            outcomeFromDict({"name": "yield", "type": 3})

    def test_what_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a mapping"):
            outcomeFromDict(["yield", "continuous"])  # type: ignore[arg-type]

    def test_an_unknown_field_is_refused_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            outcomeFromDict({"name": "yield", "type": "continuous", "minimize": True})

        assert caught.value.field_path == "minimize"
        assert "Unknown field" in caught.value.message

    def test_a_missing_name_is_refused_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            outcomeFromDict({"type": "continuous"})

        assert caught.value.field_path == "name"

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_a_field_of_the_wrong_type_is_refused_with_its_path(self, field: str) -> None:
        data: dict[str, Any] = {"name": "yield", "type": "continuous", "description": "d"}
        data[field] = ["not", "a", "string"]

        with pytest.raises(ConfigError) as caught:
            outcomeFromDict(data)

        assert caught.value.field_path == field

    def test_the_error_of_an_unknown_type_comes_before_the_errors_of_the_fields(self) -> None:
        # A mapping with several faults reports the one that tells what to register first.
        with pytest.raises(ConfigError, match="Unknown outcome_kind type 'ordinal'"):
            outcomeFromDict({"type": "ordinal", "unit": "%"})

    def test_the_mapping_that_is_read_is_not_modified(self) -> None:
        data = {"name": "yield", "type": "continuous", "description": "Mass yield."}
        before = copy.deepcopy(data)

        outcomeFromDict(data)

        assert data == before
