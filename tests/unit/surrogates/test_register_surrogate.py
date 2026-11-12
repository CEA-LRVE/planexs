"""Unit tests for `registerSurrogate` (roadmap task P3-07, specification 5.12, 6, 10.6).

A fake builder stands for a real surrogate: the contract is tested before any surrogate exists.
Every test that registers one uses `isolatedSurrogateKind`, because a registered name is
permanent and must never reach the real registry.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import Field
from surrogate_fakes import FakeModel, defineFake, fakeConfig, makeContext

import planexs.surrogates  # noqa: F401
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.surrogates.AbstractSurrogateBuilder import AbstractSurrogateBuilder
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.registerSurrogate import registerSurrogate
from planexs.surrogates.surrogateRegistry import surrogateRegistry


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        builder = defineFake(
            "gp_v1",
            needs=frozenset({"noise:known"}),
            supports=frozenset({"posterior:gaussian", "outputs:multi"}),
            description="A GP for tests.",
        )

        entry = surrogateRegistry().lookup("gp_v1")

        assert entry.name == "gp_v1"
        assert entry.description == "A GP for tests."
        assert entry.needs == {"noise:known"}
        assert entry.supports == {"posterior:gaussian", "outputs:multi"}
        assert entry.config_schema.model_fields["type"].default == "gp_v1"
        assert entry.builder == builder.fromConfig

    def test_the_class_is_returned_unchanged(self, isolatedSurrogateKind: KindTable) -> None:
        builder = defineFake()

        assert issubclass(builder, AbstractSurrogateBuilder)
        assert builder.__name__ == "FakeBuilder"

    def test_a_surrogate_that_declares_nothing_is_legal(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake(supports=frozenset())

        entry = surrogateRegistry().lookup("fake_v1")

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_several_surrogates_live_side_by_side(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("a_v1")
        defineFake("b_v1")

        assert surrogateRegistry().names() == ["a_v1", "b_v1"]

    def test_registering_builds_nothing(self, isolatedSurrogateKind: KindTable) -> None:
        # A builder is free of side effects at import and registration time (specification 6).
        calls: list[str] = []

        @registerSurrogate("lazy_v1", config=fakeConfig("lazy_v1"), description="Lazy.")
        class Lazy(AbstractSurrogateBuilder[Any]):
            def __init__(self, config: Any) -> None:
                calls.append("init")
                super().__init__(config)

            def build(self, context: Any) -> BuiltSurrogate:
                calls.append("build")
                return BuiltSurrogate(
                    FakeModel(), FitStatus(n_observations=0, is_low_data=True)
                )  # type: ignore[arg-type]

        assert calls == []

    def test_the_registry_entry_passes_the_contract_of_specification_6(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1", supports=frozenset({"posterior:gaussian"}))
        entry = surrogateRegistry().lookup("gp_v1")
        vocabulary = isolatedSurrogateKind.vocabulary

        assert surrogateRegistry().isValidName(entry.name)
        assert issubclass(entry.config_schema, ComponentConfig)
        assert all(
            field.description and field.description.strip()
            for field in entry.config_schema.model_fields.values()
        )
        assert all(tag in vocabulary for tag in entry.needs | entry.supports)
        assert callable(entry.builder)
        assert entry.description.splitlines() == [entry.description]


class TestFromAConfigurationToAFittedModel:
    """The path that a study will take: a mapping, the registry, a builder, a context."""

    def test_a_mapping_resolves_to_the_entry_and_a_validated_configuration(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1")

        resolved = resolveComponent(
            surrogateRegistry(), {"type": "gp_v1", "min_observations": 9}
        )

        assert resolved.entry.name == "gp_v1"
        assert resolved.config.model_dump() == {"type": "gp_v1", "min_observations": 9}

    def test_the_entry_builder_makes_the_builder_and_the_builder_fits(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1")
        resolved = resolveComponent(
            surrogateRegistry(), {"type": "gp_v1", "min_observations": 9}
        )

        builder = resolved.entry.builder(resolved.config)
        built = builder.build(makeContext())

        assert isinstance(builder, AbstractSurrogateBuilder)
        assert builder.config is resolved.config
        assert isinstance(built, BuiltSurrogate)
        # Four observations against a minimum of nine: the builder says it is low-data.
        assert built.status == FitStatus(n_observations=4, is_low_data=True)

    def test_a_setting_changes_what_the_builder_does(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1")
        registry = surrogateRegistry()

        def statusFor(minimum: int) -> FitStatus:
            resolved = resolveComponent(
                registry, {"type": "gp_v1", "min_observations": minimum}
            )
            return resolved.entry.builder(resolved.config).build(makeContext()).status

        assert statusFor(4).is_low_data is False
        assert statusFor(5).is_low_data is True

    def test_the_resolved_configuration_has_every_default_written_out(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1")

        resolved = resolveComponent(surrogateRegistry(), {"type": "gp_v1"})

        assert toResolvedConfig(resolved.config) == {"type": "gp_v1", "min_observations": 3}

    def test_an_unknown_field_is_refused_with_its_name(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1")

        with pytest.raises(ConfigError, match="min_observation") as caught:
            resolveComponent(surrogateRegistry(), {"type": "gp_v1", "min_observation": 9})

        assert "surrogate 'gp_v1'" in caught.value.message

    def test_an_unknown_surrogate_lists_the_registered_ones(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("a_v1")
        defineFake("b_v1")

        with pytest.raises(ConfigError, match="Registered names: a_v1, b_v1") as caught:
            resolveComponent(surrogateRegistry(), {"type": "c_v1"})

        assert caught.value.field_path == "type"


class TestARegistrationThatIsRefused:
    def assertNothingWasRegistered(self, table: KindTable) -> None:
        assert table.registry("surrogate").names() == []

    @pytest.mark.parametrize(
        "name", ["Upper", "1st_v1", "with space", "trailing_", "", "a__b"]
    )
    def test_an_invalid_name(self, isolatedSurrogateKind: KindTable, name: str) -> None:
        with pytest.raises(RegistryError, match="not a valid component name"):
            defineFake(name)

        self.assertNothingWasRegistered(isolatedSurrogateKind)

    def test_a_name_that_is_taken_is_never_replaced(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1", description="The first one.")

        with pytest.raises(RegistryError, match="already registered") as caught:
            defineFake("gp_v1", description="The second one.")

        assert "permanent identifier" in str(caught.value)
        assert surrogateRegistry().lookup("gp_v1").description == "The first one."

    def test_a_class_that_is_not_a_surrogate_builder(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        class NotABuilder:
            """Not a subclass of the contract."""

        with pytest.raises(
            RegistryError, match="not a subclass of AbstractSurrogateBuilder"
        ) as caught:
            registerSurrogate("plain_v1", config=fakeConfig("plain_v1"), description="Plain.")(
                NotABuilder  # type: ignore[type-var]
            )

        assert caught.value.component == "surrogate 'plain_v1'"
        self.assertNothingWasRegistered(isolatedSurrogateKind)

    def test_a_schema_whose_type_is_not_pinned_to_the_name(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        class OtherConfig(ComponentConfig):
            """A schema pinned to another name."""

            type: Literal["other_v1"] = Field(default="other_v1", description="The name.")

        with pytest.raises(RegistryError, match="pinned to the registered name") as caught:

            @registerSurrogate("gp_v1", config=OtherConfig, description="Mismatch.")
            class Mismatch(AbstractSurrogateBuilder[OtherConfig]):
                def build(self, context: Any) -> BuiltSurrogate:
                    raise NotImplementedError

        assert caught.value.field_path == "type"
        self.assertNothingWasRegistered(isolatedSurrogateKind)

    @pytest.mark.parametrize("description", ["", "   ", "Two\nlines."])
    def test_a_description_that_is_not_one_line(
        self, isolatedSurrogateKind: KindTable, description: str
    ) -> None:
        with pytest.raises(RegistryError, match="description"):
            defineFake(description=description)

        self.assertNothingWasRegistered(isolatedSurrogateKind)

    def test_a_needs_that_is_a_single_string_and_not_a_collection(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="needs"):
            defineFake(needs="posterior:gaussian")  # type: ignore[arg-type]

        self.assertNothingWasRegistered(isolatedSurrogateKind)


class TestTheCompatibilityCheck:
    """Needs and supports are what the compatibility check reads (specification 5.12)."""

    def test_an_entry_is_a_participant_with_the_tags_it_declared(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake(
            "gp_v1",
            needs=frozenset({"noise:known"}),
            supports=frozenset({"posterior:gaussian"}),
        )

        participant = DeclaredParticipant.fromEntry(
            "surrogate", surrogateRegistry().lookup("gp_v1")
        )

        assert participant.label == "surrogate 'gp_v1'"
        assert participant.needs == {"noise:known"}
        assert participant.supports == {"posterior:gaussian"}

    def test_a_need_that_a_surrogate_supports_is_met(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1", supports=frozenset({"posterior:gaussian"}))
        surrogate = DeclaredParticipant.fromEntry(
            "surrogate", surrogateRegistry().lookup("gp_v1")
        )
        acquisition = DeclaredParticipant(
            label="acquisition 'log_ei_v1'", needs=frozenset({"posterior:gaussian"})
        )

        checkCompatibility([acquisition, surrogate], isolatedSurrogateKind)

    def test_a_need_that_no_participant_supports_is_reported_with_the_surrogates_that_would(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("classifier_v1", supports=frozenset({"posterior:latent_gaussian"}))
        defineFake("gp_v1", supports=frozenset({"posterior:gaussian"}))
        classifier = DeclaredParticipant.fromEntry(
            "surrogate", surrogateRegistry().lookup("classifier_v1")
        )
        acquisition = DeclaredParticipant(
            label="acquisition 'log_ei_v1'", needs=frozenset({"posterior:gaussian"})
        )

        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([acquisition, classifier], isolatedSurrogateKind)

        message = caught.value.message
        assert "Need 'posterior:gaussian'" in message
        assert "needed by: acquisition 'log_ei_v1'" in message
        assert "supported by: surrogate 'gp_v1'" in message
        assert "classifier_v1" not in message.split("supported by:")[1]

    def test_a_surrogate_that_needs_known_noise_is_met_by_another_participant(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("gp_v1", needs=frozenset({"noise:known"}), supports=frozenset())
        gp = DeclaredParticipant.fromEntry("surrogate", surrogateRegistry().lookup("gp_v1"))

        with pytest.raises(CompatibilityError, match="Need 'noise:known'"):
            checkCompatibility([gp], isolatedSurrogateKind)

        provider = DeclaredParticipant(label="problem", supports=frozenset({"noise:known"}))
        checkCompatibility([gp, provider], isolatedSurrogateKind)
