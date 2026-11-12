"""Unit tests for `AbstractSurrogateBuilder`, the contract of the `surrogate` kind (roadmap
task P3-07, specification 5.6, 6).
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import Field
from surrogate_fakes import FakeModel, fakeConfig, makeContext

from planexs.core.ComponentConfig import ComponentConfig
from planexs.surrogates.AbstractSurrogateBuilder import AbstractSurrogateBuilder
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext


class CountingConfig(ComponentConfig):
    """A schema with a typed field, for a builder that reads it."""

    type: Literal["counting_v1"] = Field(default="counting_v1", description="The name.")
    minimum: int = Field(default=2, description="The least number of observations, a count.")


class CountingBuilder(AbstractSurrogateBuilder[CountingConfig]):
    """A builder that reads a field of its own configuration."""

    def build(self, context: SurrogateBuildContext) -> BuiltSurrogate:
        """Reports whether the context holds enough observations."""
        return BuiltSurrogate(
            model=FakeModel(),  # type: ignore[arg-type]
            status=FitStatus(
                n_observations=context.n_observations,
                is_low_data=context.n_observations < self.config.minimum,
            ),
        )


class TestTheContract:
    def test_the_abstract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractSurrogateBuilder(fakeConfig("a_v1")())  # type: ignore[abstract]

    def test_a_subclass_without_build_cannot_be_instantiated(self) -> None:
        class NoBuild(AbstractSurrogateBuilder[Any]):
            """Forgot `build`."""

        with pytest.raises(TypeError, match="build"):
            NoBuild(fakeConfig("a_v1")())  # type: ignore[abstract]

    def test_a_subclass_that_defines_build_can_be_instantiated(self) -> None:
        assert isinstance(CountingBuilder(CountingConfig()), AbstractSurrogateBuilder)

    def test_build_takes_a_context_and_returns_a_built_surrogate(self) -> None:
        built = CountingBuilder(CountingConfig(minimum=10)).build(makeContext())

        assert isinstance(built, BuiltSurrogate)
        assert built.status == FitStatus(n_observations=4, is_low_data=True)


class TestTheConfiguration:
    def test_a_builder_keeps_the_configuration_it_was_made_with(self) -> None:
        config = CountingConfig(minimum=7)

        assert CountingBuilder(config).config is config

    def test_the_configuration_is_read_only(self) -> None:
        builder = CountingBuilder(CountingConfig())

        with pytest.raises(AttributeError):
            builder.config = CountingConfig(minimum=9)  # type: ignore[misc]

    def test_the_builder_reads_its_settings_from_it(self) -> None:
        # Four observations: enough for a minimum of 4, not for a minimum of 5.
        assert (
            CountingBuilder(CountingConfig(minimum=4)).build(makeContext()).status.is_low_data
            is False
        )
        assert (
            CountingBuilder(CountingConfig(minimum=5)).build(makeContext()).status.is_low_data
            is True
        )

    def test_a_builder_builds_as_often_as_it_is_asked(self) -> None:
        builder = CountingBuilder(CountingConfig())

        first = builder.build(makeContext())
        second = builder.build(makeContext())

        assert first.status == second.status


class TestFromConfig:
    def test_it_makes_the_class_it_is_called_on(self) -> None:
        config = CountingConfig(minimum=3)

        builder = CountingBuilder.fromConfig(config)

        assert type(builder) is CountingBuilder
        assert builder.config is config

    def test_it_makes_the_subclass_and_not_the_base(self) -> None:
        class Sub(CountingBuilder):
            """A subclass of a builder."""

        assert type(Sub.fromConfig(CountingConfig())) is Sub

    def test_it_is_the_same_as_calling_the_class(self) -> None:
        config = CountingConfig()

        assert CountingBuilder.fromConfig(config).config is CountingBuilder(config).config

    def test_the_configuration_it_receives_is_not_modified(self) -> None:
        config = CountingConfig(minimum=3)
        before = config.model_dump()

        CountingBuilder.fromConfig(config).build(makeContext())

        assert config.model_dump() == before
