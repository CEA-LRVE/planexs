"""Unit tests for `planexs.domain.Trial` (specification 5.4, ADR-0012)."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from planexs.core.PlanexsError import PlanexsError
from planexs.core.TrialStateError import TrialStateError
from planexs.domain.Observation import Observation
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState

PENDING = TrialState.pending
COMPLETED = TrialState.completed
FAILED = TrialState.failed
ABANDONED = TrialState.abandoned
TERMINAL_STATES = (COMPLETED, FAILED, ABANDONED)

#: The three moves the table of ADR-0012 allows, written out here independently of the code.
LEGAL_MOVES = {(PENDING, COMPLETED), (PENDING, FAILED), (PENDING, ABANDONED)}
ALL_PAIRS = list(itertools.product(TrialState, repeat=2))
PAIR_IDS = [f"{origin}->{target}" for origin, target in ALL_PAIRS]

PARAMETERS: dict[str, bool | int | float | str] = {
    "temperature": 25.5,
    "cycles": 3,
    "stirred": True,
    "catalyst": "A",
}
OBSERVATION = Observation(values={"yield": 0.75, "cost": 12.0}, noise_std={"yield": 0.05})
METADATA: dict[str, Any] = {"operator": "ada", "batch": 7, "tags": ["a", {"b": [1, 2]}]}
REASON = "the furnace tripped"


def build(state: TrialState, trial_id: int = 4) -> Trial:
    """A trial built directly in `state`: creating a trial is not a transition."""
    if state is COMPLETED:
        return Trial(
            id=trial_id,
            parameters=PARAMETERS,
            state=state,
            observation=OBSERVATION,
            metadata=METADATA,
        )
    if state is FAILED:
        return Trial(
            id=trial_id, parameters=PARAMETERS, state=state, metadata=METADATA, reason=REASON
        )
    return Trial(id=trial_id, parameters=PARAMETERS, state=state, metadata=METADATA)


def move(trial: Trial, target: TrialState) -> Trial:
    """Moves `trial` to `target` with the extra argument that target needs."""
    if target is COMPLETED:
        return trial.transitionTo(target, observation=OBSERVATION)
    if target is FAILED:
        return trial.transitionTo(target, reason=REASON)
    return trial.transitionTo(target)


class TestCreation:
    def test_a_new_trial_is_pending_with_no_observation_no_metadata_and_no_reason(
        self,
    ) -> None:
        trial = Trial(id=0, parameters=PARAMETERS)
        assert trial.state is PENDING
        assert trial.observation is None
        assert trial.metadata == {}
        assert trial.reason is None

    @pytest.mark.parametrize("state", list(TrialState))
    def test_a_trial_can_be_built_in_any_state(self, state: TrialState) -> None:
        # Creation is not a transition: attach builds completed and failed trials, and a
        # snapshot is read back into trials in every state.
        trial = build(state)
        assert trial.state is state
        assert trial.id == 4
        assert trial.parameters == PARAMETERS

    def test_the_values_keep_their_python_types(self) -> None:
        trial = Trial(id=1, parameters={"a": True, "b": 3, "c": 1.0, "d": "x"})
        assert [type(value) for value in trial.parameters.values()] == [bool, int, float, str]

    def test_the_trial_does_not_check_the_parameterization_against_a_problem(self) -> None:
        # That is the study's job (ADR-0012): the record accepts any mapping of scalars.
        assert Trial(id=1, parameters={}).parameters == {}

    @pytest.mark.parametrize("trial_id", [0, 1, 7, 1_000_000])
    def test_the_id_is_a_non_negative_integer(self, trial_id: int) -> None:
        assert Trial(id=trial_id, parameters=PARAMETERS).id == trial_id

    @pytest.mark.parametrize("bad_id", [-1, True, False, 1.0, "3", None])
    def test_an_id_that_is_not_a_non_negative_integer_is_rejected(self, bad_id: Any) -> None:
        with pytest.raises(ValidationError):
            Trial(id=bad_id, parameters=PARAMETERS)

    @pytest.mark.parametrize("bad_value", [[1], {"a": 1}, None, (1, 2)])
    def test_a_parameter_value_must_be_a_scalar(self, bad_value: Any) -> None:
        with pytest.raises(ValidationError):
            Trial(id=1, parameters={"x": bad_value})

    def test_a_parameter_name_must_be_a_string(self) -> None:
        with pytest.raises(ValidationError):
            Trial(id=1, parameters={1: 2.0})

    def test_an_unknown_field_is_rejected_on_validation(self) -> None:
        data = {"id": 1, "parameters": {"x": 1.0}, "colour": "red"}
        with pytest.raises(ValidationError, match="colour"):
            Trial.model_validate(data)

    def test_parameters_are_required(self) -> None:
        with pytest.raises(ValidationError, match="parameters"):
            Trial.model_validate({"id": 1})

    def test_a_state_is_accepted_as_its_string_value_on_validation(self) -> None:
        trial = Trial.model_validate(
            {"id": 1, "parameters": {"x": 1.0}, "state": "failed", "reason": "no signal"}
        )
        assert trial.state is FAILED

    def test_an_unknown_state_is_rejected_on_validation(self) -> None:
        with pytest.raises(ValidationError, match="state"):
            Trial.model_validate({"id": 1, "parameters": {"x": 1.0}, "state": "running"})


class TestReason:
    @pytest.mark.parametrize("blank", [None, "", "   ", "\n\t"])
    def test_a_failed_trial_requires_a_non_blank_reason(self, blank: str | None) -> None:
        with pytest.raises(ValidationError, match="needs a non-blank reason"):
            Trial(id=1, parameters=PARAMETERS, state=FAILED, reason=blank)

    def test_a_failed_trial_keeps_its_reason_exactly(self) -> None:
        assert Trial(
            id=1, parameters=PARAMETERS, state=FAILED, reason="  oven door open "
        ).reason == ("  oven door open ")

    @pytest.mark.parametrize("state", [PENDING, COMPLETED, ABANDONED])
    def test_only_a_failed_trial_may_carry_a_reason(self, state: TrialState) -> None:
        with pytest.raises(ValidationError, match="only a failed trial carries one"):
            Trial(id=1, parameters=PARAMETERS, state=state, reason=REASON)

    @pytest.mark.parametrize("blank", ["", "  "])
    def test_failing_a_pending_trial_with_a_blank_reason_is_rejected(self, blank: str) -> None:
        trial = Trial(id=1, parameters=PARAMETERS)
        with pytest.raises(ValueError, match="non-blank reason"):
            trial.fail(blank)
        assert trial.state is PENDING

    def test_a_reason_that_is_not_a_string_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Trial(id=1, parameters=PARAMETERS, state=FAILED, reason=3)


class TestImmutability:
    @pytest.mark.parametrize(
        "field", ["id", "parameters", "state", "observation", "metadata", "reason"]
    )
    def test_no_attribute_can_be_assigned(self, field: str) -> None:
        trial = build(COMPLETED)
        with pytest.raises(ValidationError, match="frozen"):
            setattr(trial, field, getattr(trial, field))

    def test_the_parameters_cannot_be_changed_in_place(self) -> None:
        trial = build(PENDING)
        with pytest.raises(TypeError):
            trial.parameters["temperature"] = 99.0
        with pytest.raises(TypeError):
            del trial.parameters["temperature"]
        assert trial.parameters == PARAMETERS

    def test_the_observation_cannot_be_changed_in_place(self) -> None:
        trial = build(COMPLETED)
        assert trial.observation is not None
        with pytest.raises(ValidationError):
            trial.observation.values = {}  # type: ignore[misc]
        with pytest.raises(TypeError):
            trial.observation.values["yield"] = 1.0  # type: ignore[index]
        with pytest.raises(TypeError):
            trial.observation.noise_std["yield"] = 1.0  # type: ignore[index]
        assert trial.observation == OBSERVATION

    def test_the_metadata_is_read_only_at_every_depth(self) -> None:
        trial = build(PENDING)
        with pytest.raises(TypeError):
            trial.metadata["operator"] = "bob"
        tags = trial.metadata["tags"]
        with pytest.raises(AttributeError):
            tags.append("c")
        with pytest.raises(TypeError):
            tags[1]["b"] = []
        assert trial.metadata["tags"] == ("a", {"b": (1, 2)})

    def test_changing_the_dictionaries_given_to_the_constructor_does_not_change_the_trial(
        self,
    ) -> None:
        parameters = {"x": 1.0}
        metadata: dict[str, Any] = {"tags": ["a"], "nested": {"k": [1]}}
        trial = Trial(id=1, parameters=parameters, metadata=metadata)
        parameters["x"] = 2.0
        metadata["tags"].append("b")
        metadata["nested"]["k"].append(2)
        assert trial.parameters == {"x": 1.0}
        assert trial.metadata == {"tags": ("a",), "nested": {"k": (1,)}}

    def test_changing_the_observation_given_to_tell_does_not_change_the_trial(self) -> None:
        values = {"yield": 0.5}
        done = Trial(id=1, parameters=PARAMETERS).tell(Observation(values=values))
        values["yield"] = 9.0
        assert done.observation == Observation(values={"yield": 0.5})

    def test_a_json_array_is_read_back_as_a_tuple(self) -> None:
        trial = Trial(id=1, parameters=PARAMETERS, metadata={"k": [1]})
        assert trial.metadata["k"] == (1,)
        assert trial.metadata != {"k": [1]}

    def test_equal_trials_are_equal_and_hash_alike(self) -> None:
        assert build(COMPLETED) == build(COMPLETED)
        assert hash(build(COMPLETED)) == hash(build(COMPLETED))
        assert len({build(PENDING), build(PENDING)}) == 1

    def test_trials_that_differ_in_any_field_are_not_equal(self) -> None:
        base = build(PENDING)
        assert base != build(PENDING, trial_id=5)
        assert base != build(ABANDONED)
        assert base != Trial(id=4, parameters={**PARAMETERS, "cycles": 4}, metadata=METADATA)
        assert base != Trial(id=4, parameters=PARAMETERS, metadata={"other": 1})


class TestMetadata:
    def test_the_default_is_empty_and_read_only(self) -> None:
        trial = Trial(id=1, parameters=PARAMETERS)
        assert trial.metadata == {}
        with pytest.raises(TypeError):
            trial.metadata["k"] = 1

    def test_the_default_is_read_only_also_when_validated_from_a_mapping(self) -> None:
        trial = Trial.model_validate({"id": 1, "parameters": {"x": 1.0}})
        with pytest.raises(TypeError):
            trial.metadata["k"] = 1

    def test_any_json_value_is_accepted(self) -> None:
        metadata = {
            "n": None,
            "b": True,
            "i": 1,
            "f": 1.5,
            "s": "x",
            "l": [None, [1]],
            "d": {},
        }
        assert Trial(id=1, parameters=PARAMETERS, metadata=metadata).metadata["f"] == 1.5

    @pytest.mark.parametrize("bad", [{1, 2}, b"bytes", object(), 1 + 2j])
    def test_a_value_that_is_not_json_is_rejected(self, bad: Any) -> None:
        with pytest.raises(ValidationError):
            Trial(id=1, parameters=PARAMETERS, metadata={"k": bad})

    def test_a_key_that_is_not_a_string_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Trial(id=1, parameters=PARAMETERS, metadata={1: "x"})

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_number_is_rejected_wherever_it_is(self, bad: float) -> None:
        # Written as null on save, it would change the history silently.
        with pytest.raises(ValidationError, match="not JSON-compatible"):
            Trial(id=1, parameters=PARAMETERS, metadata={"k": bad})
        with pytest.raises(ValidationError, match="not JSON-compatible"):
            Trial(id=1, parameters=PARAMETERS, metadata={"k": [{"deep": bad}]})
        # The observation is an `Observation`, which refuses it with its own message.
        with pytest.raises(ValidationError, match="not finite"):
            Trial(
                id=1,
                parameters=PARAMETERS,
                state=COMPLETED,
                observation={"values": {"yield": bad}},  # type: ignore[arg-type]
            )
        with pytest.raises(ValidationError, match="not JSON-compatible"):
            Trial(id=1, parameters={"x": bad})

    def test_the_metadata_goes_through_every_transition_unchanged(self) -> None:
        trial = build(PENDING)
        for done in (trial.tell(OBSERVATION), trial.fail(REASON), trial.abandon()):
            assert done.metadata == trial.metadata
            assert done.parameters == trial.parameters


class TestTransitions:
    def test_tell_completes_a_pending_trial_with_the_observation(self) -> None:
        pending = build(PENDING)
        done = pending.tell(OBSERVATION)
        assert done is not pending
        assert done.state is COMPLETED
        assert done.observation == OBSERVATION
        assert done.reason is None

    def test_fail_fails_a_pending_trial_with_the_reason(self) -> None:
        pending = build(PENDING)
        failed = pending.fail(REASON)
        assert failed is not pending
        assert failed.state is FAILED
        assert failed.reason == REASON
        assert failed.observation is None

    def test_abandon_abandons_a_pending_trial(self) -> None:
        pending = build(PENDING)
        abandoned = pending.abandon()
        assert abandoned is not pending
        assert abandoned.state is ABANDONED
        assert abandoned.observation is None
        assert abandoned.reason is None

    def test_a_transition_leaves_the_original_trial_untouched(self) -> None:
        pending = build(PENDING)
        before = pending.model_dump()
        pending.tell(OBSERVATION)
        pending.fail(REASON)
        pending.abandon()
        assert pending.state is PENDING
        assert pending.model_dump() == before

    @pytest.mark.parametrize("trial_id", [0, 7, 1_000_000])
    @pytest.mark.parametrize("target", TERMINAL_STATES)
    def test_a_transition_never_changes_the_id(
        self, trial_id: int, target: TrialState
    ) -> None:
        pending = build(PENDING, trial_id=trial_id)
        assert move(pending, target).id == trial_id

    @pytest.mark.parametrize(("origin", "target"), ALL_PAIRS, ids=PAIR_IDS)
    def test_every_pair_of_states_is_a_transition_only_if_it_is_legal(
        self, origin: TrialState, target: TrialState
    ) -> None:
        trial = build(origin)
        if (origin, target) in LEGAL_MOVES:
            moved = move(trial, target)
            assert moved.state is target
            assert moved.id == trial.id
            assert trial.state is origin
        else:
            with pytest.raises(TrialStateError):
                move(trial, target)
            assert trial.state is origin

    @pytest.mark.parametrize("origin", TERMINAL_STATES)
    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(lambda trial: trial.tell(OBSERVATION), id="tell"),
            pytest.param(lambda trial: trial.fail(REASON), id="fail"),
            pytest.param(lambda trial: trial.abandon(), id="abandon"),
        ],
    )
    def test_no_named_operation_works_on_a_terminal_trial(
        self, origin: TrialState, operation: Callable[[Trial], Trial]
    ) -> None:
        with pytest.raises(TrialStateError):
            operation(build(origin))

    def test_a_legal_move_is_never_to_pending(self) -> None:
        for origin in TrialState:
            with pytest.raises(TrialStateError):
                build(origin).transitionTo(PENDING)

    def test_a_transition_is_checked_before_its_arguments(self) -> None:
        # A blank reason on a terminal trial is a lifecycle error, not a validation error.
        with pytest.raises(TrialStateError):
            build(COMPLETED).fail("")


class TestTrialStateError:
    def test_it_is_an_error_of_the_planexs_family(self) -> None:
        with pytest.raises(PlanexsError):
            build(COMPLETED).abandon()

    def test_the_error_names_the_trial_the_field_and_both_states(self) -> None:
        with pytest.raises(TrialStateError) as caught:
            build(COMPLETED, trial_id=12).abandon()
        error = caught.value
        assert error.component == "trial 12"
        assert error.field_path == "state"
        assert "Trial 12 is completed and cannot become abandoned" in error.message
        assert "terminal state" in error.message

    def test_the_suggestion_for_a_terminal_trial_is_to_record_a_new_trial(self) -> None:
        with pytest.raises(TrialStateError) as caught:
            build(FAILED).tell(OBSERVATION)
        assert caught.value.suggestion is not None
        assert "new trial" in caught.value.suggestion

    def test_the_suggestion_for_a_move_from_pending_to_pending_is_to_report_it(self) -> None:
        with pytest.raises(TrialStateError) as caught:
            build(PENDING).transitionTo(PENDING)
        assert "stays pending until it is reported" in caught.value.message
        assert caught.value.suggestion == "tell, fail or abandon the trial"

    def test_the_rendered_error_has_the_three_hints(self) -> None:
        with pytest.raises(TrialStateError) as caught:
            build(ABANDONED, trial_id=3).tell(OBSERVATION)
        lines = str(caught.value).splitlines()
        assert lines[1:] == [
            "Component: trial 3",
            "Field: state",
            "Suggestion: record a repeated evaluation, a retry or a late result "
            "as a new trial",
        ]


class TestObservationCoupling:
    """A completed trial requires an observation, and the other states forbid one."""

    def test_a_completed_trial_without_an_observation_is_rejected(self) -> None:
        with pytest.raises(
            ValidationError, match="Trial 4 is completed and needs an observation"
        ):
            Trial(id=4, parameters=PARAMETERS, state=COMPLETED)

    @pytest.mark.parametrize("state", [PENDING, ABANDONED])
    def test_a_trial_that_is_not_completed_cannot_carry_an_observation(
        self, state: TrialState
    ) -> None:
        with pytest.raises(ValidationError, match=f"is {state.value} but has an observation"):
            Trial(id=4, parameters=PARAMETERS, state=state, observation=OBSERVATION)

    def test_a_failed_trial_cannot_carry_an_observation(self) -> None:
        with pytest.raises(ValidationError, match="is failed but has an observation"):
            Trial(
                id=4,
                parameters=PARAMETERS,
                state=FAILED,
                observation=OBSERVATION,
                reason=REASON,
            )

    def test_a_move_to_completed_needs_an_observation(self) -> None:
        with pytest.raises(ValidationError, match="needs an observation"):
            build(PENDING).transitionTo(COMPLETED)

    @pytest.mark.parametrize("target", [FAILED, ABANDONED])
    def test_a_move_to_another_state_cannot_bring_an_observation(
        self, target: TrialState
    ) -> None:
        reason = REASON if target is FAILED else None
        with pytest.raises(ValidationError, match="has an observation"):
            build(PENDING).transitionTo(target, observation=OBSERVATION, reason=reason)

    def test_an_illegal_move_is_still_reported_as_one_whatever_the_observation(self) -> None:
        with pytest.raises(TrialStateError):
            build(COMPLETED).tell(OBSERVATION)

    def test_tell_stores_the_very_observation_it_is_given(self) -> None:
        done = build(PENDING).tell(OBSERVATION)
        assert done.observation is OBSERVATION

    def test_a_mapping_in_the_dump_form_is_read_as_an_observation(self) -> None:
        document: Any = {"values": {"yield": 0.75}, "noise_std": {"yield": 0.05}}
        trial = Trial(id=4, parameters=PARAMETERS, state=COMPLETED, observation=document)
        assert isinstance(trial.observation, Observation)
        assert trial.observation == Observation(
            values={"yield": 0.75}, noise_std={"yield": 0.05}
        )

    def test_a_bare_mapping_of_values_is_not_an_observation(self) -> None:
        # The 0.1.0 shape, `{outcome: value}`: refused, so it cannot be mistaken for one.
        with pytest.raises(ValidationError):
            Trial(
                id=4,
                parameters=PARAMETERS,
                state=COMPLETED,
                observation={"yield": 0.75},  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("state", list(TrialState))
    def test_the_observation_is_present_exactly_when_the_trial_is_completed(
        self, state: TrialState
    ) -> None:
        assert (build(state).observation is not None) is (state is COMPLETED)

    def test_the_observation_survives_a_json_round_trip_inside_a_trial(self) -> None:
        trial = build(COMPLETED)
        text = trial.model_dump_json()
        again = Trial.model_validate_json(text)
        assert again == trial
        assert again.observation == OBSERVATION
        assert json.loads(text)["observation"] == {
            "values": {"yield": 0.75, "cost": 12.0},
            "noise_std": {"yield": 0.05},
        }


class TestSerialization:
    @pytest.mark.parametrize("state", list(TrialState))
    def test_the_dump_is_plain_json_for_every_state(self, state: TrialState) -> None:
        dumped = build(state).model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped
        assert dumped["state"] == state.value
        assert dumped["metadata"]["tags"] == ["a", {"b": [1, 2]}]

    def test_the_python_dump_holds_no_read_only_container(self) -> None:
        dumped = build(COMPLETED).model_dump()
        assert json.loads(json.dumps(dumped)) == json.loads(json.dumps(dumped))
        assert type(dumped["parameters"]) is dict
        assert type(dumped["observation"]) is dict
        assert type(dumped["observation"]["values"]) is dict
        assert type(dumped["observation"]["noise_std"]) is dict
        assert type(dumped["metadata"]["tags"]) is list

    def test_the_exact_shape_of_a_dump(self) -> None:
        trial = Trial(
            id=3,
            parameters={"x": 1.5, "n": 2},
            state=FAILED,
            metadata={"k": [1]},
            reason="no signal",
        )
        assert trial.model_dump(mode="json") == {
            "id": 3,
            "parameters": {"x": 1.5, "n": 2},
            "state": "failed",
            "observation": None,
            "metadata": {"k": [1]},
            "reason": "no signal",
        }

    @pytest.mark.parametrize("state", list(TrialState))
    def test_a_trial_survives_a_round_trip_through_a_mapping(self, state: TrialState) -> None:
        trial = build(state)
        assert Trial.model_validate(trial.model_dump(mode="json")) == trial

    @pytest.mark.parametrize("state", list(TrialState))
    def test_a_trial_survives_a_round_trip_through_json_text(self, state: TrialState) -> None:
        trial = build(state)
        reloaded = Trial.model_validate_json(trial.model_dump_json())
        assert reloaded == trial
        assert reloaded.model_dump_json() == trial.model_dump_json()

    def test_the_types_of_the_values_survive_a_round_trip(self) -> None:
        trial = Trial(
            id=1,
            parameters={"a": True, "b": 3, "c": 1.0, "d": "x"},
            metadata={"f": 1.0, "i": 1},
        )
        reloaded = Trial.model_validate_json(trial.model_dump_json())
        assert [type(value) for value in reloaded.parameters.values()] == [
            bool,
            int,
            float,
            str,
        ]
        assert type(reloaded.metadata["f"]) is float
        assert type(reloaded.metadata["i"]) is int

    def test_a_trial_that_went_through_a_transition_survives_a_round_trip(self) -> None:
        done = build(PENDING).tell(OBSERVATION)
        assert Trial.model_validate_json(done.model_dump_json()) == done

    def test_a_reloaded_trial_is_as_read_only_as_a_new_one(self) -> None:
        reloaded = Trial.model_validate_json(build(PENDING).model_dump_json())
        with pytest.raises(TypeError):
            reloaded.parameters["temperature"] = 1.0
        with pytest.raises(TypeError):
            reloaded.metadata["operator"] = "bob"

    def test_a_document_that_breaks_the_reason_rule_is_rejected_on_reload(self) -> None:
        failed_without_reason = '{"id": 1, "parameters": {"x": 1.0}, "state": "failed"}'
        with pytest.raises(ValidationError, match="needs a non-blank reason"):
            Trial.model_validate_json(failed_without_reason)
        completed_with_reason = (
            '{"id": 1, "parameters": {"x": 1.0}, "state": "completed", "reason": "x"}'
        )
        with pytest.raises(ValidationError, match="only a failed trial carries one"):
            Trial.model_validate_json(completed_with_reason)

    def test_a_document_with_a_bad_id_or_an_unknown_field_is_rejected_on_reload(self) -> None:
        with pytest.raises(ValidationError):
            Trial.model_validate_json('{"id": -1, "parameters": {}}')
        with pytest.raises(ValidationError, match="extra"):
            Trial.model_validate_json('{"id": 1, "parameters": {}, "extra": 1}')

    def test_a_large_id_is_written_and_read_exactly(self) -> None:
        trial = Trial(id=2**53 + 1, parameters=PARAMETERS)
        assert Trial.model_validate_json(trial.model_dump_json()).id == 2**53 + 1
