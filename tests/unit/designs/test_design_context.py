"""Unit tests for `DesignContext` (roadmap task P4-06, specification 5.8, 6).

The context is the read-only information a design is given. Tested: that it holds what it was
given, that it is read-only (and that the history cannot be changed through it), and that it
counts observations as the specification says (completed trials only).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from design_fakes import makeContext, makeSpace, makeTrials

from planexs.designs.DesignContext import DesignContext
from planexs.domain.TrialState import TrialState

FIELDS = ["problem", "encoder", "constraints", "trials", "generator"]


class TestAContext:
    def test_it_holds_what_it_was_given(self) -> None:
        generator = torch.Generator().manual_seed(3)
        base = makeContext()

        context = DesignContext(
            problem=base.problem,
            encoder=base.encoder,
            constraints=base.constraints,
            trials=makeTrials(completed=2),
            generator=generator,
        )

        assert context.problem is base.problem
        assert context.encoder is base.encoder
        assert context.constraints is base.constraints
        assert context.generator is generator
        assert len(context.trials) == 2

    def test_it_is_built_with_keywords(self) -> None:
        base = makeContext()

        with pytest.raises(TypeError):
            DesignContext(  # type: ignore[misc]
                base.problem, base.encoder, base.constraints, [], base.generator
            )

    def test_a_missing_field_is_a_type_error(self) -> None:
        base = makeContext()

        with pytest.raises(TypeError, match="generator"):
            DesignContext(  # type: ignore[call-arg]
                problem=base.problem,
                encoder=base.encoder,
                constraints=base.constraints,
                trials=[],
            )

    def test_contexts_are_compared_by_identity(self) -> None:
        first = makeContext()
        second = makeContext()

        assert first != second
        assert first == first
        assert len({first, second}) == 2

    def test_the_generator_is_the_one_member_that_advances(self) -> None:
        context = makeContext()
        generator = context.generator

        first = torch.rand(1, generator=generator)
        second = torch.rand(1, generator=generator)

        assert context.generator is generator
        assert not torch.equal(first, second)


class TestTheContextIsReadOnly:
    @pytest.mark.parametrize("field", FIELDS)
    def test_no_field_can_be_assigned(self, field: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(makeContext(), field, None)

    def test_no_field_can_be_added(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            makeContext().extra = 1  # type: ignore[attr-defined]

    @pytest.mark.parametrize("field", FIELDS)
    def test_no_field_can_be_deleted(self, field: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            delattr(makeContext(), field)

    def test_the_trials_are_stored_as_a_tuple(self) -> None:
        context = makeContext(trials=makeTrials(completed=2))

        assert isinstance(context.trials, tuple)

    def test_the_history_cannot_be_changed_through_the_context(self) -> None:
        trials = makeTrials(completed=2)
        context = makeContext(trials=trials)

        with pytest.raises(AttributeError):
            context.trials.append(trials[0])  # type: ignore[attr-defined]

    def test_changing_the_list_that_was_given_does_not_change_the_context(self) -> None:
        trials = makeTrials(completed=2)
        context = makeContext(trials=trials)

        trials.append(makeTrials(pending=1)[0])
        trials.clear()

        assert len(context.trials) == 2

    def test_any_sequence_of_trials_is_accepted(self) -> None:
        base = makeContext()
        trials = tuple(makeTrials(completed=3))

        context = DesignContext(
            problem=base.problem,
            encoder=base.encoder,
            constraints=base.constraints,
            trials=trials,
            generator=base.generator,
        )

        assert context.trials == trials


class TestCountingObservations:
    def test_an_empty_history_has_none(self) -> None:
        assert makeContext().countObservations() == 0

    def test_a_completed_trial_is_an_observation(self) -> None:
        assert makeContext(trials=makeTrials(completed=3)).countObservations() == 3

    @pytest.mark.parametrize("state", ["pending", "failed", "abandoned"])
    def test_a_trial_in_any_other_state_has_no_observation(self, state: str) -> None:
        # Pending: not reported yet. Failed: it never produced one. Abandoned: withdrawn.
        assert makeContext(trials=makeTrials(**{state: 4})).countObservations() == 0

    def test_only_the_completed_trials_are_counted_in_a_mixed_history(self) -> None:
        trials = makeTrials(completed=2, pending=3, failed=1, abandoned=2)

        assert makeContext(trials=trials).countObservations() == 2

    def test_every_state_is_covered_by_the_count(self) -> None:
        # If a state were added to the lifecycle, this fails until the rule is decided for it.
        assert {state.value for state in TrialState} == {
            "pending",
            "completed",
            "failed",
            "abandoned",
        }

    def test_the_count_follows_the_history_it_was_given(self) -> None:
        small = makeContext(space=makeSpace(), trials=makeTrials(completed=1))
        large = makeContext(space=makeSpace(), trials=makeTrials(completed=9))

        assert (small.countObservations(), large.countObservations()) == (1, 9)
