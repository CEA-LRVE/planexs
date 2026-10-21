"""Unit tests for `Participant` and `DeclaredParticipant` (specification 5.12)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.PlanexsError import PlanexsError

MakeEntry = Callable[..., ComponentEntry]


class TestFields:
    def test_every_field_is_kept(self) -> None:
        participant = DeclaredParticipant(
            "problem", needs=frozenset({"a:x"}), supports=frozenset({"b:y"})
        )
        assert participant.label == "problem"
        assert participant.needs == frozenset({"a:x"})
        assert participant.supports == frozenset({"b:y"})

    def test_needs_and_supports_default_to_empty_frozen_sets(self) -> None:
        participant = DeclaredParticipant("goal")
        assert participant.needs == frozenset()
        assert participant.supports == frozenset()
        assert isinstance(participant.needs, frozenset)
        assert isinstance(participant.supports, frozenset)

    @pytest.mark.parametrize(
        "given",
        [
            pytest.param(["b:x", "a:y"], id="list"),
            pytest.param(("b:x", "a:y"), id="tuple"),
            pytest.param({"b:x", "a:y"}, id="set"),
            pytest.param({"b:x": 1, "a:y": 2}, id="dict-keys"),
            pytest.param((tag for tag in ["b:x", "a:y"]), id="generator"),
            pytest.param(["b:x", "a:y", "b:x"], id="duplicates-collapse"),
        ],
    )
    def test_any_collection_of_tags_becomes_a_frozen_set(self, given: Any) -> None:
        participant = DeclaredParticipant("problem", needs=given)
        assert participant.needs == frozenset({"a:y", "b:x"})
        assert isinstance(participant.needs, frozenset)
        assert participant.supports == frozenset()

    def test_supports_are_frozen_too(self) -> None:
        assert isinstance(DeclaredParticipant("x", supports=["a:b"]).supports, frozenset)

    @pytest.mark.parametrize("field", ["label", "needs", "supports"])
    def test_a_field_cannot_be_reassigned(self, field: str) -> None:
        participant = DeclaredParticipant("problem")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(participant, field, "other")

    def test_equal_participants_are_equal_and_hash_alike(self) -> None:
        first = DeclaredParticipant("p", needs=["a:x"])
        second = DeclaredParticipant("p", needs=("a:x",))
        assert first == second
        assert hash(first) == hash(second)
        assert first != DeclaredParticipant("q", needs=["a:x"])
        assert first != DeclaredParticipant("p", supports=["a:x"])


class TestBareStrings:
    @pytest.mark.parametrize("field", ["needs", "supports"])
    def test_a_single_string_is_refused_not_split_into_characters(self, field: str) -> None:
        with pytest.raises(CompatibilityError, match="single string") as caught:
            DeclaredParticipant("problem", **{field: "space:categorical"})
        assert caught.value.component == "problem"
        assert caught.value.field_path == field
        assert caught.value.suggestion is not None

    def test_the_refusal_is_a_planexs_error(self) -> None:
        with pytest.raises(PlanexsError):
            DeclaredParticipant("problem", needs="a:b")

    def test_the_tags_themselves_are_not_judged_here(self) -> None:
        # Whether a tag is well formed and known is `checkCompatibility`'s call, against the
        # vocabulary of the kind table it is given.
        assert DeclaredParticipant("p", needs={"Not A Tag"}).needs == frozenset({"Not A Tag"})


class TestFromEntry:
    def test_the_label_names_the_kind_and_the_component(self, makeEntry: MakeEntry) -> None:
        participant = DeclaredParticipant.fromEntry(
            "surrogate", makeEntry("single_task_gp_v1")
        )
        assert participant.label == "surrogate 'single_task_gp_v1'"

    def test_the_label_matches_the_one_registry_errors_use(self, makeEntry: MakeEntry) -> None:
        # `Registry.register` hints at `f"{kind} {name!r}"`: one convention, two places.
        participant = DeclaredParticipant.fromEntry("acquisition", makeEntry("ucb_v1"))
        assert participant.label == f"{'acquisition'} {'ucb_v1'!r}"

    def test_the_tags_are_the_ones_the_entry_declares(self, makeEntry: MakeEntry) -> None:
        entry = makeEntry("gp_v1", needs={"a:x", "b:y"}, supports={"c:z"})
        participant = DeclaredParticipant.fromEntry("surrogate", entry)
        assert participant.needs == frozenset({"a:x", "b:y"})
        assert participant.supports == frozenset({"c:z"})

    def test_an_entry_without_tags_gives_a_participant_without_tags(
        self, makeEntry: MakeEntry
    ) -> None:
        participant = DeclaredParticipant.fromEntry("kernel", makeEntry("rbf_v1"))
        assert participant.needs == frozenset()
        assert participant.supports == frozenset()

    def test_it_returns_a_declared_participant(self, makeEntry: MakeEntry) -> None:
        assert isinstance(
            DeclaredParticipant.fromEntry("kernel", makeEntry("rbf_v1")), DeclaredParticipant
        )
