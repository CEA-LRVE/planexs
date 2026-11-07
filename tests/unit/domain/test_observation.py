"""Unit tests for `planexs.domain.Observation` (specification 5.4, ADR-0012 decision 5).

The rules of an observation against a list of outcomes are each tested on their own: every
outcome present, no extra name, each value valid for its kind, noise only where the kind
allows it, and noise finite and not negative. The coupling of an observation to the state of
a trial is tested in `test_trial.py`.
"""

from __future__ import annotations

import json
import math
from typing import Any, ClassVar, Literal

import numpy as np
import pytest
from pydantic import Field, ValidationError

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.core.PlanexsError import PlanexsError
from planexs.domain.Observation import Observation
from planexs.domain.ObservationError import ObservationError
from planexs.domain.Outcome import Outcome
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.OutcomeValueError import OutcomeValueError
from planexs.domain.outcome_kinds.registerOutcomeKind import registerOutcomeKind

YIELD = Outcome(name="yield", type="continuous")
COST = Outcome(name="cost", type="continuous")
PHASE = Outcome(name="phase", type="binary")
OUTCOMES = [YIELD, COST, PHASE]
VALUES: dict[str, object] = {"yield": 0.75, "cost": 12, "phase": True}


def make(
    values: dict[str, object] | None = None,
    noise_std: dict[str, object] | None = None,
    outcomes: list[Outcome] | None = None,
) -> Observation:
    """An observation of the three outcomes above, built the way a caller builds one."""
    return Observation.create(
        OUTCOMES if outcomes is None else outcomes,
        VALUES if values is None else values,
        noise_std,
    )


class TestCreate:
    def test_every_value_is_stored_as_its_kind_returns_it(self) -> None:
        observation = make()
        assert observation.values == {"yield": 0.75, "cost": 12.0, "phase": 1.0}
        assert all(type(value) is float for value in observation.values.values())

    @pytest.mark.parametrize(
        ("given", "stored"),
        [
            (True, 1.0),
            (False, 0.0),
            (1, 1.0),
            (0, 0.0),
            (np.bool_(True), 1.0),
            (np.int64(0), 0.0),
        ],
    )
    def test_a_label_is_stored_as_zero_or_one(self, given: object, stored: float) -> None:
        observation = make({**VALUES, "phase": given})
        assert observation.values["phase"] == stored
        assert type(observation.values["phase"]) is float

    def test_numpy_numbers_are_accepted(self) -> None:
        observation = make({**VALUES, "yield": np.float64(0.5)}, {"yield": np.float32(0.25)})
        assert observation.values["yield"] == 0.5
        assert observation.noise_std == {"yield": 0.25}
        assert type(observation.noise_std["yield"]) is float

    def test_the_order_of_the_given_mapping_does_not_matter(self) -> None:
        shuffled = {"phase": True, "cost": 12, "yield": 0.75}
        assert make(shuffled) == make()

    def test_noise_is_optional_and_defaults_to_none_known(self) -> None:
        assert make().noise_std == {}
        assert make(noise_std={}).noise_std == {}

    def test_noise_may_be_known_for_some_outcomes_only(self) -> None:
        observation = make(noise_std={"yield": 0.05})
        assert observation.noise_std == {"yield": 0.05}
        assert "cost" not in observation.noise_std

    def test_the_mappings_given_are_not_kept(self) -> None:
        values = dict(VALUES)
        noise: dict[str, object] = {"yield": 0.05}
        observation = make(values, noise)
        values["yield"] = 99.0
        noise["yield"] = 99.0
        noise["cost"] = 1.0
        assert observation.values["yield"] == 0.75
        assert observation.noise_std == {"yield": 0.05}


class TestEveryOutcomeIsPresentAndNothingElse:
    def test_an_outcome_without_a_value_is_refused(self) -> None:
        with pytest.raises(ObservationError, match="no value for 'cost'") as caught:
            make({"yield": 0.75, "phase": True})
        assert caught.value.field_path == "values"
        assert "['yield', 'cost', 'phase']" in str(caught.value)

    def test_every_missing_outcome_is_listed_in_one_message(self) -> None:
        with pytest.raises(ObservationError, match="no value for 'yield', 'cost'"):
            make({"phase": False})

    def test_an_empty_mapping_misses_every_outcome(self) -> None:
        with pytest.raises(ObservationError, match="no value for 'yield', 'cost', 'phase'"):
            make({})

    def test_a_value_for_a_name_that_is_no_outcome_is_refused(self) -> None:
        with pytest.raises(
            ObservationError, match="a value for 'extra', which is no outcome"
        ) as caught:
            make({**VALUES, "extra": 1.0})
        assert caught.value.field_path == "values"

    def test_a_misspelled_name_is_reported_as_missing_and_as_unknown_together(self) -> None:
        values = {"yeld": 0.75, "cost": 12, "phase": True}
        with pytest.raises(ObservationError) as caught:
            make(values)
        message = str(caught.value)
        assert "no value for 'yield'" in message
        assert "a value for 'yeld', which is no outcome" in message

    def test_a_noise_standard_deviation_for_a_name_that_is_no_outcome_is_refused(self) -> None:
        with pytest.raises(
            ObservationError,
            match="a noise standard deviation for 'extra', which is no outcome",
        ) as caught:
            make(noise_std={"extra": 0.1})
        assert caught.value.field_path == "noise_std"

    def test_the_names_are_checked_before_the_values(self) -> None:
        # `yield` is not a number either, but the missing outcome is what is reported.
        with pytest.raises(ObservationError, match="no value for 'cost'"):
            make({"yield": "bad", "phase": True})

    def test_the_outcomes_may_be_a_list_of_one(self) -> None:
        observation = make({"yield": 1.5}, outcomes=[YIELD])
        assert observation.values == {"yield": 1.5}


class TestEachValueIsValidForItsKind:
    @pytest.mark.parametrize(
        "bad", [math.nan, math.inf, -math.inf, "0.5", None, True, [1.0], 1 + 2j]
    )
    def test_a_continuous_value_must_be_a_finite_number(self, bad: object) -> None:
        with pytest.raises(OutcomeValueError, match="'yield'"):
            make({**VALUES, "yield": bad})

    @pytest.mark.parametrize("bad", [2, -1, 0.5, 1.0, "1", "yes", None, math.nan])
    def test_a_binary_value_must_be_a_label(self, bad: object) -> None:
        with pytest.raises(OutcomeValueError, match="'phase'"):
            make({**VALUES, "phase": bad})

    def test_the_error_comes_from_the_kind_and_names_the_outcome(self) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            make({**VALUES, "cost": math.nan})
        assert isinstance(caught.value, PlanexsError)
        assert isinstance(caught.value, ValueError)
        assert caught.value.component == "outcome 'cost'"

    def test_one_bad_value_refuses_the_whole_observation(self) -> None:
        with pytest.raises(OutcomeValueError):
            make({"yield": 0.75, "cost": 12, "phase": 2})


class TestNoiseOnlyWhereTheKindAllowsIt:
    def test_a_continuous_outcome_may_have_a_noise_standard_deviation(self) -> None:
        assert make(noise_std={"yield": 0.05, "cost": 1.5}).noise_std == {
            "yield": 0.05,
            "cost": 1.5,
        }

    def test_a_binary_outcome_may_not(self) -> None:
        with pytest.raises(
            ObservationError, match="outcome 'phase', whose kind 'binary' has none"
        ) as caught:
            make(noise_std={"phase": 0.1})
        assert caught.value.component == "outcome 'phase'"
        assert caught.value.field_path == "noise_std"
        assert caught.value.suggestion is not None

    def test_the_rule_is_asked_of_the_kind_not_of_its_name(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        # A kind that the core has never heard of, registered the way a plugin would. With
        # `allows_noise` set, the observation accepts a deviation for it, and refuses it
        # where it is not: no branch on `continuous` or `binary` is involved.
        noisy, silent = defineToyKinds()
        count = Outcome(name="count", type=noisy)
        flag = Outcome(name="flag", type=silent)
        observation = Observation.create(
            [count, flag], {"count": 3, "flag": 1}, {"count": 0.5}
        )
        assert observation.values == {"count": 3.0, "flag": 1.0}
        assert observation.noise_std == {"count": 0.5}
        with pytest.raises(ObservationError, match="whose kind 'silent_toy' has none"):
            Observation.create([count, flag], {"count": 3, "flag": 1}, {"flag": 0.5})
        with pytest.raises(OutcomeValueError, match="'count'"):
            Observation.create([count, flag], {"count": -2, "flag": 1})


class TestNoiseIsFiniteAndNotNegative:
    def test_zero_is_a_legal_noise_standard_deviation(self) -> None:
        # A deterministic evaluation: the surrogate floors it, the observation does not.
        assert make(noise_std={"yield": 0.0}).noise_std == {"yield": 0.0}

    @pytest.mark.parametrize("bad", [-0.1, -1, -1e-12])
    def test_a_negative_noise_standard_deviation_is_refused(self, bad: float) -> None:
        with pytest.raises(ObservationError, match="is negative") as caught:
            make(noise_std={"yield": bad})
        assert caught.value.field_path == "noise_std"
        assert caught.value.component == "outcome 'yield'"

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_noise_standard_deviation_is_refused(self, bad: float) -> None:
        with pytest.raises(ObservationError, match="not finite"):
            make(noise_std={"yield": bad})

    @pytest.mark.parametrize("bad", ["0.1", None, True, [0.1]])
    def test_a_noise_standard_deviation_that_is_not_a_number_is_refused(
        self, bad: object
    ) -> None:
        with pytest.raises(ObservationError, match="not a number"):
            make(noise_std={"yield": bad})

    def test_the_kind_is_checked_before_the_number(self) -> None:
        with pytest.raises(ObservationError, match="has none"):
            make(noise_std={"phase": math.nan})


class TestBuiltDirectly:
    """`Observation(values=..., noise_std=...)` checks what needs no outcomes."""

    def test_numbers_are_read_as_floats_and_noise_defaults_to_empty(self) -> None:
        observation = Observation(values={"y": 1, "z": 2.5})
        assert observation.values == {"y": 1.0, "z": 2.5}
        assert all(type(value) is float for value in observation.values.values())
        assert observation.noise_std == {}

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_value_is_refused(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="not finite"):
            Observation(values={"y": bad})

    @pytest.mark.parametrize("bad", ["0.5", None, True, [1]])
    def test_a_value_that_is_not_a_number_is_refused(self, bad: Any) -> None:
        # A string that looks like a number is not coerced, and neither is a bool.
        with pytest.raises(ValidationError, match="not a number"):
            Observation(values={"y": bad})

    @pytest.mark.parametrize("bad", [-0.5, math.nan, math.inf, "0.1", True])
    def test_a_bad_noise_standard_deviation_is_refused(self, bad: Any) -> None:
        with pytest.raises(ValidationError):
            Observation(values={"y": 1.0}, noise_std={"y": bad})

    def test_a_noise_standard_deviation_needs_a_value(self) -> None:
        with pytest.raises(ValidationError, match="noise standard deviation for 'z'"):
            Observation(values={"y": 1.0}, noise_std={"z": 0.1})

    def test_the_values_are_required(self) -> None:
        with pytest.raises(ValidationError):
            Observation()  # type: ignore[call-arg]

    def test_values_that_are_not_a_mapping_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            Observation(values=[1.0, 2.0])  # type: ignore[arg-type]

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Observation(values={"y": 1.0}, noise="0.1")  # type: ignore[call-arg]

    def test_a_refusal_is_a_value_error_that_names_the_outcome(self) -> None:
        with pytest.raises(ValueError, match="outcome 'y'"):
            Observation(values={"y": math.nan})


class TestImmutability:
    def test_no_attribute_can_be_assigned(self) -> None:
        observation = make()
        with pytest.raises(ValidationError):
            observation.values = {}  # type: ignore[misc]
        with pytest.raises(ValidationError):
            observation.noise_std = {}  # type: ignore[misc]

    def test_the_mappings_are_read_only(self) -> None:
        observation = make(noise_std={"yield": 0.05})
        with pytest.raises(TypeError):
            observation.values["yield"] = 1.0  # type: ignore[index]
        with pytest.raises(TypeError):
            del observation.values["yield"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            observation.noise_std["yield"] = 1.0  # type: ignore[index]

    def test_the_default_noise_is_read_only_too(self) -> None:
        with pytest.raises(TypeError):
            make().noise_std["yield"] = 1.0  # type: ignore[index]

    def test_changing_the_dictionaries_given_to_the_constructor_changes_nothing(self) -> None:
        values = {"y": 1.0}
        noise = {"y": 0.1}
        observation = Observation(values=values, noise_std=noise)
        values["y"] = 9.0
        noise["y"] = 9.0
        assert observation.values == {"y": 1.0}
        assert observation.noise_std == {"y": 0.1}

    def test_an_observation_built_from_another_ones_mappings_is_equal_to_it(self) -> None:
        observation = make(noise_std={"yield": 0.05})
        again = Observation(values=observation.values, noise_std=observation.noise_std)
        assert again == observation


class TestEqualityAndHash:
    def test_equal_observations_are_equal_and_hash_alike(self) -> None:
        first = Observation(values={"a": 1.0, "b": 2.0}, noise_std={"a": 0.1})
        second = Observation(values={"b": 2.0, "a": 1.0}, noise_std={"a": 0.1})
        assert first == second
        assert hash(first) == hash(second)

    def test_different_values_or_noise_make_different_observations(self) -> None:
        base = Observation(values={"a": 1.0}, noise_std={"a": 0.1})
        assert base != Observation(values={"a": 2.0}, noise_std={"a": 0.1})
        assert base != Observation(values={"a": 1.0}, noise_std={"a": 0.2})
        assert base != Observation(values={"a": 1.0})

    def test_an_observation_can_be_kept_in_a_set(self) -> None:
        first = Observation(values={"a": 1.0})
        assert (
            len({first, Observation(values={"a": 1.0}), Observation(values={"a": 2.0})}) == 2
        )


class TestSerialization:
    def test_the_exact_shape_of_a_dump(self) -> None:
        observation = make(noise_std={"yield": 0.05})
        assert observation.model_dump(mode="json") == {
            "values": {"yield": 0.75, "cost": 12.0, "phase": 1.0},
            "noise_std": {"yield": 0.05},
        }

    def test_the_noise_of_an_observation_with_none_is_an_empty_mapping(self) -> None:
        assert make().model_dump(mode="json")["noise_std"] == {}

    def test_the_python_dump_holds_no_read_only_container(self) -> None:
        dumped = make(noise_std={"yield": 0.05}).model_dump()
        assert type(dumped["values"]) is dict
        assert type(dumped["noise_std"]) is dict

    def test_the_dump_is_plain_json(self) -> None:
        dumped = make(noise_std={"yield": 0.05}).model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped

    @pytest.mark.parametrize("noise", [None, {"yield": 0.05}, {"yield": 0.0, "cost": 2.0}])
    def test_an_observation_survives_a_round_trip_through_a_mapping(
        self, noise: dict[str, object] | None
    ) -> None:
        observation = make(noise_std=noise)
        assert Observation.model_validate(observation.model_dump(mode="json")) == observation

    @pytest.mark.parametrize("noise", [None, {"yield": 0.05}])
    def test_an_observation_survives_a_round_trip_through_json_text(
        self, noise: dict[str, object] | None
    ) -> None:
        observation = make(noise_std=noise)
        again = Observation.model_validate_json(observation.model_dump_json())
        assert again == observation
        assert hash(again) == hash(observation)

    def test_the_noise_may_be_left_out_of_a_stored_document(self) -> None:
        assert Observation.model_validate({"values": {"y": 1.0}}) == Observation(
            values={"y": 1.0}
        )

    def test_a_document_with_a_non_finite_value_is_refused_on_load(self) -> None:
        with pytest.raises(ValidationError, match="not finite"):
            Observation.model_validate_json('{"values": {"y": NaN}}')


class TestObservationError:
    def test_it_is_a_planexs_error_and_a_value_error(self) -> None:
        assert issubclass(ObservationError, PlanexsError)
        assert issubclass(ObservationError, ValueError)

    def test_it_renders_the_message_and_every_hint(self) -> None:
        with pytest.raises(ObservationError) as caught:
            make(noise_std={"phase": 0.1})
        lines = str(caught.value).splitlines()
        assert lines[0].startswith(
            "A noise standard deviation was given for the outcome 'phase'"
        )
        assert lines[1:] == [
            "Component: outcome 'phase'",
            "Field: noise_std",
            "Suggestion: leave the noise standard deviation of this outcome out",
        ]


# ---------------------------------------------------------------------------------------
# A toy plugin: two outcome kinds the core has never heard of.
# ---------------------------------------------------------------------------------------


def defineToyKinds() -> tuple[str, str]:
    """Registers two toy kinds, `noisy_toy` (noise allowed) and `silent_toy` (none), and
    returns their names. Must run with the `isolatedDomainKinds` fixture, so that nothing
    reaches the real registry or the real vocabulary.
    """

    class NoisyToyConfig(ComponentConfig):
        type: Literal["noisy_toy"] = Field(default="noisy_toy", description="Always so.")

    class SilentToyConfig(ComponentConfig):
        type: Literal["silent_toy"] = Field(default="silent_toy", description="Always so.")

    class NoisyToy(OutcomeKind):
        capability_tag: ClassVar[str] = "outcome_kind:noisy_toy"
        allows_noise: ClassVar[bool] = True

        def validateValue(self, value: object, outcome: str) -> float:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise OutcomeValueError(
                    f"The value {value!r} of the toy outcome {outcome!r} is not a count.",
                    component=f"outcome {outcome!r}",
                )
            return float(value)

    class SilentToy(OutcomeKind):
        capability_tag: ClassVar[str] = "outcome_kind:silent_toy"
        allows_noise: ClassVar[bool] = False

        def validateValue(self, value: object, outcome: str) -> float:
            if value not in (0, 1):
                raise OutcomeValueError(
                    f"The value {value!r} of the toy outcome {outcome!r} is not a flag.",
                    component=f"outcome {outcome!r}",
                )
            return float(value)  # type: ignore[arg-type]

    registerOutcomeKind("noisy_toy", config=NoisyToyConfig, description="A toy count.")(
        NoisyToy
    )
    registerOutcomeKind("silent_toy", config=SilentToyConfig, description="A toy flag.")(
        SilentToy
    )
    return "noisy_toy", "silent_toy"
