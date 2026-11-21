"""Unit tests for `registerAcquisition` (roadmap task P5-01, specification 5.12, 6, 10.6).

What is tested: what a registration stores, everything that makes it fail (and that a
failure leaves nothing behind), and the consistency of the reference-set declaration of a
builder, which the decorator checks when the module is imported.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pytest
from acquisition_fakes import defineFake, fakeConfig
from botorch.acquisition.acquisition import AcquisitionFunction
from pydantic import Field

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.acquisition.registerAcquisition import registerAcquisition
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent


class Config(ComponentConfig):
    """A configuration whose `type` is pinned to `plain_v1`."""

    type: Literal["plain_v1"] = Field(default="plain_v1", description="The name, plain_v1.")


class TestWhatARegistrationStores:
    def test_the_entry_holds_the_name_the_schema_the_description_and_the_tags(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        cls = defineFake(
            "stored_v1",
            description="Scores a point.",
            needs=frozenset({"goal:optimize", "posterior:gaussian"}),
            supports=frozenset({"pending:supported", "acquisition:differentiable"}),
        )

        entry = acquisitionRegistry().lookup("stored_v1")

        assert entry.name == "stored_v1"
        assert entry.description == "Scores a point."
        assert entry.needs == {"goal:optimize", "posterior:gaussian"}
        assert entry.supports == {"pending:supported", "acquisition:differentiable"}
        assert issubclass(entry.config_schema, ComponentConfig)
        assert entry.config_schema.model_fields["type"].default == "stored_v1"
        assert entry.builder(entry.config_schema()).__class__ is cls

    def test_the_decorator_returns_the_class_unchanged(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        class Plain(AbstractAcquisitionBuilder[Config]):
            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                raise AssertionError("not called")

        decorated = registerAcquisition("plain_v1", config=Config, description="Plain.")(Plain)

        assert decorated is Plain

    def test_the_builder_of_the_entry_is_from_config(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("made_v1")

        resolved = resolveComponent(acquisitionRegistry(), {"type": "made_v1", "beta": 3.5})
        builder = resolved.entry.builder(resolved.config)

        assert isinstance(builder, AbstractAcquisitionBuilder)
        assert builder.config is resolved.config
        assert builder.config.beta == 3.5

    def test_a_name_with_a_version_suffix_is_valid(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("log_ei_v2")

        assert "log_ei_v2" in acquisitionRegistry()

    def test_two_acquisitions_live_side_by_side(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("first_v1")
        defineFake("second_v1")

        assert acquisitionRegistry().names() == ["first_v1", "second_v1"]

    def test_needs_and_supports_default_to_nothing(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        class Plain(AbstractAcquisitionBuilder[Config]):
            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                raise AssertionError("not called")

        registerAcquisition("plain_v1", config=Config, description="Plain.")(Plain)

        entry = acquisitionRegistry().lookup("plain_v1")
        assert entry.needs == frozenset()
        assert entry.supports == frozenset()


class TestWhatMakesARegistrationFail:
    def test_a_name_that_is_taken_is_refused_and_the_first_entry_stays(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        first = defineFake("taken_v1", description="The first.")

        with pytest.raises(RegistryError, match="already registered"):
            defineFake("taken_v1", description="The second.")

        entry = acquisitionRegistry().lookup("taken_v1")
        assert entry.description == "The first."
        assert entry.builder(entry.config_schema()).__class__ is first

    @pytest.mark.parametrize(
        "name", ["Upper_v1", "with space", "1leading", "", "trailing_", "a__b"]
    )
    def test_an_invalid_name_is_refused(
        self, name: str, isolatedAcquisitionKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="not a valid component name"):
            defineFake(name)

        assert acquisitionRegistry().names() == []

    def test_a_class_that_is_not_a_builder_is_refused(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        class NotABuilder:
            pass

        with pytest.raises(
            RegistryError, match="not a subclass of AbstractAcquisitionBuilder"
        ):
            registerAcquisition("nope_v1", config=Config, description="No.")(NotABuilder)  # type: ignore[type-var]

        assert acquisitionRegistry().names() == []

    def test_a_schema_whose_type_is_not_pinned_to_the_name_is_refused(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        class Plain(AbstractAcquisitionBuilder[Config]):
            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                raise AssertionError("not called")

        with pytest.raises(
            RegistryError, match="must be pinned to the registered name"
        ) as caught:
            registerAcquisition("other_v1", config=Config, description="Other.")(Plain)

        assert caught.value.field_path == "type"
        assert acquisitionRegistry().names() == []

    def test_a_description_over_several_lines_is_refused(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="description"):
            defineFake("lines_v1", description="One line.\nAnd another.")

        assert acquisitionRegistry().names() == []

    def test_a_blank_description_is_refused(self, isolatedAcquisitionKind: KindTable) -> None:
        with pytest.raises(RegistryError, match="description"):
            defineFake("blank_v1", description="   ")


class TestTheReferenceSetDeclaration:
    """`needs_reference_points` and `referencePointCount` are tied: the flag is what the policy
    reads to know whether to draw a set, and the method says how many, so one without the other
    is a builder that asks for nothing or is given what it ignores.
    """

    def test_a_builder_that_needs_no_set_registers_with_the_defaults(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        cls = defineFake("none_v1")

        assert cls.needs_reference_points is False
        assert cls.referencePointCount is AbstractAcquisitionBuilder.referencePointCount

    def test_a_builder_that_needs_a_set_and_says_how_many_registers(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        cls = defineFake("sized_v1", with_reference_set=True)

        assert cls.needs_reference_points is True
        assert "sized_v1" in acquisitionRegistry()

    def test_the_flag_without_a_size_is_refused(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        class Asks(AbstractAcquisitionBuilder[Config]):
            needs_reference_points: ClassVar[bool] = True

            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                raise AssertionError("not called")

        with pytest.raises(
            RegistryError, match="does not override referencePointCount"
        ) as caught:
            registerAcquisition("plain_v1", config=Config, description="Plain.")(Asks)

        assert caught.value.component == "acquisition 'plain_v1'"
        assert caught.value.field_path == "needs_reference_points"
        assert caught.value.suggestion is not None
        assert "referencePointCount" in caught.value.suggestion
        assert acquisitionRegistry().names() == []

    def test_a_size_without_the_flag_is_refused(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        class Sizes(AbstractAcquisitionBuilder[Config]):
            def referencePointCount(self) -> int:
                return 8

            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                raise AssertionError("not called")

        with pytest.raises(RegistryError, match="never draw one") as caught:
            registerAcquisition("plain_v1", config=Config, description="Plain.")(Sizes)

        assert caught.value.field_path == "needs_reference_points"
        assert acquisitionRegistry().names() == []

    @pytest.mark.parametrize("flag", ["yes", 1, None])
    def test_a_flag_that_is_not_a_bool_is_refused(
        self, flag: Any, isolatedAcquisitionKind: KindTable
    ) -> None:
        class Odd(AbstractAcquisitionBuilder[Config]):
            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                raise AssertionError("not called")

        Odd.needs_reference_points = flag

        with pytest.raises(RegistryError, match="not a bool") as caught:
            registerAcquisition("plain_v1", config=Config, description="Plain.")(Odd)

        assert caught.value.field_path == "needs_reference_points"

    def test_a_size_that_a_subclass_inherits_counts_as_overridden(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        class Base(AbstractAcquisitionBuilder[Config]):
            needs_reference_points: ClassVar[bool] = True

            def referencePointCount(self) -> int:
                return 8

            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                raise AssertionError("not called")

        class Child(Base):
            pass

        registerAcquisition("plain_v1", config=Config, description="Plain.")(Child)

        assert "plain_v1" in acquisitionRegistry()

    def test_a_refusal_comes_before_the_registry_is_touched(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        class Asks(AbstractAcquisitionBuilder[Config]):
            needs_reference_points: ClassVar[bool] = True

            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                raise AssertionError("not called")

        with pytest.raises(RegistryError):
            registerAcquisition("plain_v1", config=Config, description="Plain.")(Asks)

        # The name is still free: the same name registers once the class is consistent.
        defineFake("plain_v1")
        assert "plain_v1" in acquisitionRegistry()

    def test_the_fake_schema_carries_the_size_as_a_described_field(self) -> None:
        schema = fakeConfig("sized_v1", with_size=True)

        assert schema.model_fields["n_reference_points"].description
        assert schema().n_reference_points == 16  # type: ignore[attr-defined]
