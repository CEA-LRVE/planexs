"""Unit tests for `AbstractOutcomeTransformBuilder`, the contract of the `outcome_transform`
kind (roadmap task P3-12, specification 5.6, 6).
"""

from __future__ import annotations

from typing import Any

import pytest
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from outcome_transform_fakes import RecordingTransformBuilder, fakeConfig

from planexs.core.ConfigError import ConfigError
from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.LogOutcomeTransformBuilder import (
    LogOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.NoneOutcomeTransformBuilder import (
    NoneOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.OutcomeTransformError import OutcomeTransformError
from planexs.surrogates.outcome_transforms.StandardizeOutcomeTransformBuilder import (
    StandardizeOutcomeTransformBuilder,
)

CONFIG = fakeConfig("fake_transform_v1")


class NoTransformBuilder(RecordingTransformBuilder):
    """A transform that is no transform: it leaves the outcomes as they are."""

    def makeTransform(self, n_outcomes: int) -> OutcomeTransform | None:
        self.received.append(n_outcomes)
        return None


class NotATransformBuilder(RecordingTransformBuilder):
    """A faulty transform: what it makes is not an outcome transform."""

    def makeTransform(self, n_outcomes: int) -> Any:
        self.received.append(n_outcomes)
        return "not a transform"


class TestTheContract:
    def test_the_contract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractOutcomeTransformBuilder(CONFIG())  # type: ignore[abstract]

    def test_a_transform_that_does_not_define_makeTransform_cannot_be_instantiated(
        self,
    ) -> None:
        class Incomplete(AbstractOutcomeTransformBuilder[Any]):
            pass

        with pytest.raises(TypeError, match="makeTransform"):
            Incomplete(CONFIG())  # type: ignore[abstract]

    def test_fromConfig_makes_an_instance_of_the_class_it_is_called_on(self) -> None:
        config = CONFIG(scale=2.0)

        builder = RecordingTransformBuilder.fromConfig(config)

        assert type(builder) is RecordingTransformBuilder
        assert builder.config is config

    def test_the_configuration_is_a_read_only_attribute(self) -> None:
        builder = RecordingTransformBuilder(CONFIG())

        with pytest.raises(AttributeError):
            builder.config = CONFIG(scale=3.0)  # type: ignore[misc]

    @pytest.mark.parametrize(
        "builder",
        [
            StandardizeOutcomeTransformBuilder,
            LogOutcomeTransformBuilder,
            NoneOutcomeTransformBuilder,
        ],
    )
    def test_no_real_transform_overrides_the_entry_point(self, builder: type) -> None:
        # `build` checks `n_outcomes` and the result for every transform in the same way: it
        # is not overridden.
        assert builder.build is AbstractOutcomeTransformBuilder.build


class TestWhatBuildDoes:
    def test_it_gives_makeTransform_the_number_of_outcomes_and_returns_its_transform(
        self,
    ) -> None:
        builder = RecordingTransformBuilder(CONFIG())

        transform = builder.build(3)

        assert isinstance(transform, Standardize)
        assert builder.received == [3]

    def test_no_transform_is_a_legal_answer_and_is_passed_on_as_it_is(self) -> None:
        builder = NoTransformBuilder(CONFIG())

        assert builder.build(3) is None
        assert builder.received == [3]

    def test_every_call_makes_a_new_transform(self) -> None:
        # A transform is stateful: two models must never share one.
        builder = RecordingTransformBuilder(CONFIG())

        first, second = builder.build(3), builder.build(3)

        assert first is not second

    @pytest.mark.parametrize("n_outcomes", [1, 3, 40])
    def test_any_number_of_outcomes_of_one_or_more_is_accepted(self, n_outcomes: int) -> None:
        builder = RecordingTransformBuilder(CONFIG())

        builder.build(n_outcomes)

        assert builder.received == [n_outcomes]


class TestABadNumberOfOutcomes:
    @pytest.mark.parametrize("bad", [0, -1, -40, True, False, 2.5, 3.0, "3", None])
    def test_it_is_refused_before_the_transform_is_made(self, bad: Any) -> None:
        builder = RecordingTransformBuilder(CONFIG())

        with pytest.raises(ConfigError) as caught:
            builder.build(bad)

        error = caught.value
        assert "whole number of one or more" in error.message
        assert repr(bad) in error.message
        assert error.component == "outcome_transform 'fake_transform_v1'"
        assert error.field_path == "n_outcomes"
        assert "outcomes the surrogate models" in (error.suggestion or "")
        assert builder.received == []  # nothing was made


class TestAFaultyTransform:
    def test_something_that_is_not_an_outcome_transform_is_refused_with_a_named_error(
        self,
    ) -> None:
        builder = NotATransformBuilder(CONFIG())

        with pytest.raises(OutcomeTransformError) as caught:
            builder.build(1)

        error = caught.value
        assert "made a str" in error.message
        assert "not a BoTorch OutcomeTransform" in error.message
        assert error.component == "outcome_transform 'fake_transform_v1'"
        assert "None for no transform" in (error.suggestion or "")
