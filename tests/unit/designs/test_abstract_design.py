"""Unit tests for `AbstractDesign`, the contract of the `design` kind (roadmap task P4-06,
specification 5.8, 6).

What is tested is what every design shares and no design redefines: that the class is abstract
and configured once, and how `propose` reads a count. The fake design records what it is asked,
so that the tests can see which of its two methods `propose` called, and with what.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
import torch
from design_fakes import RecordingDesign, fakeConfig, makeContext, makeTrials
from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.DesignContext import DesignContext

CONFIG = fakeConfig("recording_v1")


def design(n_points: int = 4) -> RecordingDesign:
    """A recording design whose size is `n_points`, minus one per trial in the history."""
    return RecordingDesign(CONFIG(n_points=n_points))


class TestTheContract:
    def test_the_abstract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractDesign(CONFIG())  # type: ignore[abstract]

    @pytest.mark.parametrize("missing", ["pointsNeeded", "drawPoints"])
    def test_a_subclass_must_define_both_abstract_methods(self, missing: str) -> None:
        methods: dict[str, Any] = {
            "pointsNeeded": lambda self, trials: 0,
            "drawPoints": lambda self, context, n: torch.empty(0),
        }
        del methods[missing]
        Incomplete = type("Incomplete", (AbstractDesign,), methods)

        with pytest.raises(TypeError, match=missing):
            Incomplete(CONFIG())

    def test_propose_is_the_entry_point_and_is_not_abstract(self) -> None:
        assert "propose" not in AbstractDesign.__abstractmethods__


class TestTheConfiguration:
    def test_a_design_keeps_the_configuration_it_was_made_with(self) -> None:
        config = CONFIG(n_points=9)

        assert RecordingDesign(config).config is config

    def test_the_configuration_is_read_only(self) -> None:
        with pytest.raises(AttributeError):
            design().config = CONFIG()  # type: ignore[misc]

    def test_from_config_makes_the_class_it_is_called_on(self) -> None:
        config = CONFIG()

        built = RecordingDesign.fromConfig(config)

        assert type(built) is RecordingDesign
        assert built.config is config

    def test_from_config_makes_the_subclass_and_not_the_base(self) -> None:
        class Sub(RecordingDesign):
            """A subclass of a design."""

        assert type(Sub.fromConfig(CONFIG())) is Sub


class TestProposeWithoutACount:
    """No count means the points the design still needs."""

    def test_it_asks_the_design_how_many_it_needs_and_draws_that_many(self) -> None:
        recording = design(n_points=4)

        points = recording.propose(makeContext())

        assert points.shape[0] == 4
        assert recording.drawn == [4]

    def test_it_reads_the_history_of_the_context(self) -> None:
        # The fake needs `n_points` less one per trial, whatever its state.
        recording = design(n_points=4)

        points = recording.propose(makeContext(trials=makeTrials(completed=1, pending=1)))

        assert points.shape[0] == 2
        assert recording.drawn == [2]

    def test_a_design_that_needs_nothing_more_gives_no_point_and_draws_none(self) -> None:
        recording = design(n_points=2)

        points = recording.propose(makeContext(trials=makeTrials(completed=5)))

        assert points.shape[0] == 0
        assert recording.drawn == []


class TestProposeWithACount:
    def test_an_explicit_count_is_honoured_whatever_the_size_of_the_design(self) -> None:
        recording = design(n_points=4)

        points = recording.propose(makeContext(), count=7)

        assert points.shape[0] == 7
        assert recording.drawn == [7]

    def test_an_explicit_count_ignores_what_the_history_says_the_design_still_needs(
        self,
    ) -> None:
        # A batch that a user asked for is the caller's: it is not cut to what is left.
        recording = design(n_points=2)

        points = recording.propose(makeContext(trials=makeTrials(completed=5)), count=3)

        assert points.shape[0] == 3

    def test_zero_is_a_count_like_another_and_gives_no_point(self) -> None:
        recording = design(n_points=4)

        points = recording.propose(makeContext(), count=0)

        assert points.shape[0] == 0
        assert recording.drawn == []

    def test_a_count_of_one_is_a_single_point(self) -> None:
        assert design().propose(makeContext(), count=1).shape[0] == 1

    @pytest.mark.parametrize("count", [-1, -100])
    def test_a_negative_count_is_refused_and_names_the_field(self, count: int) -> None:
        recording = design()

        with pytest.raises(ConfigError) as caught:
            recording.propose(makeContext(), count=count)

        error = caught.value
        assert error.field_path == "count"
        assert error.component == "design 'recording_v1'"
        assert "whole number of zero or more" in error.message
        assert repr(count) in error.message
        assert error.suggestion
        assert isinstance(error, PlanexsError)
        assert recording.drawn == []

    @pytest.mark.parametrize("count", [2.0, 2.5, "2", True, False])
    def test_a_count_that_is_not_a_whole_number_is_refused(self, count: Any) -> None:
        # A bool is an int in Python and is not a number of points.
        with pytest.raises(ConfigError, match="whole number"):
            design().propose(makeContext(), count=count)


class TestTheEmptyResult:
    def test_it_has_the_width_and_the_dtype_of_the_encoding(self) -> None:
        context = makeContext()

        points = design().propose(context, count=0)

        assert tuple(points.shape) == (0, context.encoder.n_columns)
        assert points.dtype == context.encoder.dtype

    def test_single_precision_is_kept(self) -> None:
        context = makeContext(runtime=RuntimeConfig(dtype="float32"))

        assert design().propose(context, count=0).dtype == torch.float32


class TestWhatTheDesignIsGiven:
    def test_draw_points_receives_the_context_and_the_resolved_count(self) -> None:
        received: list[tuple[DesignContext, int]] = []

        class Spy(RecordingDesign):
            def drawPoints(self, context: DesignContext, n: int) -> torch.Tensor:
                received.append((context, n))
                return super().drawPoints(context, n)

        context = makeContext()
        Spy(CONFIG(n_points=3)).propose(context)
        Spy(CONFIG(n_points=3)).propose(context, count=5)

        assert [n for _, n in received] == [3, 5]
        assert all(given is context for given, _ in received)

    def test_points_needed_receives_the_trials_of_the_context(self) -> None:
        seen: list[tuple[Any, ...]] = []

        class Spy(RecordingDesign):
            def pointsNeeded(self, trials: Any) -> int:
                seen.append(tuple(trials))
                return 1

        trials = makeTrials(completed=2, failed=1)
        Spy(CONFIG()).propose(makeContext(trials=trials))

        assert seen == [tuple(trials)]


def test_a_configuration_of_another_name_is_still_a_component_config() -> None:
    # The contract is generic in the schema: any `ComponentConfig` will do.
    class OtherConfig(ComponentConfig):
        """A schema."""

        type: Literal["other_v1"] = Field(default="other_v1", description="The name.")
        n_points: int = Field(default=1, description="The size.")

    assert RecordingDesign(OtherConfig()).propose(makeContext()).shape[0] == 1
