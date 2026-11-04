"""Unit tests for `planexs.domain.TrialState` (specification 5.4, ADR-0012)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from planexs.domain.TrialState import TrialState

ADR_0012 = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "adr"
    / "0012-outcomes-trials-and-observations.md"
)

PENDING = TrialState.pending
COMPLETED = TrialState.completed
FAILED = TrialState.failed
ABANDONED = TrialState.abandoned

#: The three moves the table of ADR-0012 allows, written out here independently of the code.
LEGAL_MOVES = {(PENDING, COMPLETED), (PENDING, FAILED), (PENDING, ABANDONED)}

ALL_PAIRS = list(itertools.product(TrialState, repeat=2))


def adrTable() -> dict[tuple[str, str], str]:
    """Reads the 16 rows of the transition table of ADR-0012: (from, to) to its Legal cell."""
    names = {state.value for state in TrialState}
    rows: dict[tuple[str, str], str] = {}
    for line in ADR_0012.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] in names and cells[1] in names:
            rows[(cells[0], cells[1])] = cells[2]
    return rows


class TestMembers:
    def test_there_are_exactly_the_four_states_of_the_specification(self) -> None:
        assert [state.value for state in TrialState] == [
            "pending",
            "completed",
            "failed",
            "abandoned",
        ]

    def test_a_state_is_its_own_name_as_a_string(self) -> None:
        # This is how a state is written in a snapshot.
        assert all(state.value == state.name for state in TrialState)
        assert TrialState.failed == "failed"

    def test_a_state_is_read_back_from_its_value(self) -> None:
        assert TrialState("abandoned") is ABANDONED

    def test_an_unknown_value_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TrialState("running")


class TestIsTerminal:
    def test_only_pending_is_not_terminal(self) -> None:
        assert {state: state.is_terminal for state in TrialState} == {
            PENDING: False,
            COMPLETED: True,
            FAILED: True,
            ABANDONED: True,
        }


class TestCanMoveTo:
    @pytest.mark.parametrize(
        ("origin", "target"), ALL_PAIRS, ids=[f"{a}->{b}" for a, b in ALL_PAIRS]
    )
    def test_every_pair_of_states_is_legal_only_if_it_is_one_of_the_three_moves(
        self, origin: TrialState, target: TrialState
    ) -> None:
        assert origin.canMoveTo(target) is ((origin, target) in LEGAL_MOVES)

    def test_there_are_sixteen_pairs_and_three_of_them_are_legal(self) -> None:
        assert len(ALL_PAIRS) == 16
        assert sum(origin.canMoveTo(target) for origin, target in ALL_PAIRS) == 3

    def test_nothing_moves_to_pending_and_nothing_moves_to_itself(self) -> None:
        assert not any(origin.canMoveTo(PENDING) for origin in TrialState)
        assert not any(state.canMoveTo(state) for state in TrialState)

    def test_a_terminal_state_never_moves_anywhere(self) -> None:
        for origin in (COMPLETED, FAILED, ABANDONED):
            assert not any(origin.canMoveTo(target) for target in TrialState)

    def test_the_code_agrees_with_the_table_written_in_the_adr(self) -> None:
        # The table of ADR-0012 is the reference (specification 5.4): read it from the file,
        # so that a change to either the table or the code that leaves the other behind fails.
        table = adrTable()
        assert len(table) == 16, "the table of ADR-0012 must list all sixteen pairs"
        for (origin, target), legal in table.items():
            assert legal in ("yes", "no")
            assert TrialState(origin).canMoveTo(TrialState(target)) is (legal == "yes")
