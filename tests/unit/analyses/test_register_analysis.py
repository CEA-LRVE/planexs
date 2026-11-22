"""Unit tests for `registerAnalysis` (roadmap task P7-01, specification 5.12, 6, 10.6).

A fake analysis stands for a real one: the contract is tested before any analysis exists. Every
test that registers one uses `isolatedAnalysisKind`, because a registered name is permanent and
must never reach the real registry.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from analysis_fakes import (
    CountingProvider,
    FakeBehaviour,
    buildFake,
    defineFake,
    defineRefining,
    fakeConfig,
    makeGoal,
    makeProblem,
    makeState,
    registerToyTags,
)
from pydantic import Field

import planexs.analyses  # noqa: F401
from planexs.analyses.AbstractAnalysis import AbstractAnalysis
from planexs.analyses.analysisRegistry import analysisRegistry
from planexs.analyses.registerAnalysis import registerAnalysis
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.core.toResolvedConfig import toResolvedConfig


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        registerToyTags(isolatedAnalysisKind)
        analysis = defineFake(
            "batch_like_v1",
            needs=frozenset({"toy:gaussian"}),
            supports=frozenset({"toy:grid", "toy:optimize"}),
            description="A batch-like analysis for tests.",
        )

        entry = analysisRegistry().lookup("batch_like_v1")

        assert entry.name == "batch_like_v1"
        assert entry.description == "A batch-like analysis for tests."
        assert entry.needs == {"toy:gaussian"}
        assert entry.supports == {"toy:grid", "toy:optimize"}
        assert entry.config_schema.model_fields["type"].default == "batch_like_v1"
        assert entry.builder == analysis.fromConfig

    def test_the_class_is_returned_unchanged(self, isolatedAnalysisKind: KindTable) -> None:
        analysis = defineFake()

        assert issubclass(analysis, AbstractAnalysis)
        assert analysis.__name__ == "FakeAnalysis"

    def test_a_analysis_that_declares_nothing_is_legal(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()

        entry = analysisRegistry().lookup("fake_analysis_v1")

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_several_analyses_live_side_by_side(self, isolatedAnalysisKind: KindTable) -> None:
        defineFake("a_v1")
        defineFake("b_v1")

        assert analysisRegistry().names() == ["a_v1", "b_v1"]

    def test_registering_builds_nothing(self, isolatedAnalysisKind: KindTable) -> None:
        # A builder is free of side effects at import and registration time (specification 6).
        calls: list[str] = []

        @registerAnalysis("lazy_v1", config=fakeConfig("lazy_v1"), description="Lazy.")
        class Lazy(FakeBehaviour):
            def __init__(self, config: Any) -> None:
                calls.append("init")
                super().__init__(config)

            @classmethod
            def fromConfig(cls, config: Any, **inputs: Any) -> Any:
                calls.append("fromConfig")
                return super().fromConfig(config, **inputs)

        assert calls == []

    def test_the_registry_entry_passes_the_contract_of_specification_6(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        registerToyTags(isolatedAnalysisKind)
        defineFake("analysis_v1", supports=frozenset({"toy:grid"}))
        entry = analysisRegistry().lookup("analysis_v1")
        vocabulary = isolatedAnalysisKind.vocabulary

        assert analysisRegistry().isValidName(entry.name)
        assert issubclass(entry.config_schema, ComponentConfig)
        assert all(
            field.description and field.description.strip()
            for field in entry.config_schema.model_fields.values()
        )
        assert all(tag in vocabulary for tag in entry.needs | entry.supports)
        assert callable(entry.builder)
        assert entry.description.splitlines() == [entry.description]


class TestFromAConfigurationToAnAnalysis:
    """The path that a study will take: a mapping, the registry, an analysis, a result."""

    def test_a_mapping_resolves_to_the_entry_and_a_validated_configuration(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("analysis_v1")

        resolved = resolveComponent(
            analysisRegistry(), {"type": "analysis_v1", "use_model": True}
        )

        assert resolved.entry.name == "analysis_v1"
        assert resolved.config.model_dump() == {
            "type": "analysis_v1",
            "use_model": True,
            "draws": 3,
        }

    def test_the_entry_builder_makes_an_analysis_that_computes_a_result(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("analysis_v1")
        resolved = resolveComponent(analysisRegistry(), {"type": "analysis_v1"})

        analysis = resolved.entry.builder(
            resolved.config, problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
        )

        assert isinstance(analysis, AbstractAnalysis)
        assert analysis.config is resolved.config
        result = analysis.compute(makeState(), CountingProvider())
        assert result.analysis == "analysis_v1"
        assert result.history_version == 0

    def test_the_builder_takes_the_problem_the_goal_and_the_runtime_by_keyword(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("analysis_v1")
        resolved = resolveComponent(analysisRegistry(), {"type": "analysis_v1"})

        with pytest.raises(TypeError):
            resolved.entry.builder(resolved.config)
        with pytest.raises(TypeError):
            resolved.entry.builder(resolved.config, makeProblem(), makeGoal(), RuntimeConfig())

    def test_each_build_is_a_new_analysis_with_a_configuration_of_its_own(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("analysis_v1")

        first = buildFake("analysis_v1", draws=2)
        second = buildFake("analysis_v1", draws=5)

        assert first is not second
        assert first.config.draws == 2
        assert second.config.draws == 5
        assert len(first.compute(makeState(), CountingProvider()).samples) == 2
        assert len(second.compute(makeState(), CountingProvider()).samples) == 5

    def test_the_resolved_configuration_has_every_default_written_out(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("analysis_v1")

        resolved = resolveComponent(analysisRegistry(), {"type": "analysis_v1"})

        assert toResolvedConfig(resolved.config) == {
            "type": "analysis_v1",
            "use_model": False,
            "draws": 3,
        }

    def test_the_resolved_configuration_resolves_to_the_same_analysis_again(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # What a study stores is what it reads back (specification 5.11): an analysis built
        # from the stored configuration is the one that was built from the first.
        defineFake("analysis_v1")
        first = resolveComponent(
            analysisRegistry(), {"type": "analysis_v1", "use_model": True}
        )

        again = resolveComponent(analysisRegistry(), toResolvedConfig(first.config))

        assert again.config == first.config
        assert again.entry is first.entry

    def test_an_unknown_field_is_refused_with_its_name(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("analysis_v1")

        with pytest.raises(ConfigError, match="use_modell") as caught:
            resolveComponent(analysisRegistry(), {"type": "analysis_v1", "use_modell": True})

        assert "analysis 'analysis_v1'" in caught.value.message

    def test_a_value_of_the_wrong_type_is_refused_with_the_field(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("analysis_v1")

        with pytest.raises(ConfigError, match="use_model") as caught:
            resolveComponent(analysisRegistry(), {"type": "analysis_v1", "use_model": "maybe"})

        assert caught.value.field_path == "use_model"

    def test_an_unknown_analysis_lists_the_registered_ones(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("a_v1")
        defineFake("b_v1")

        with pytest.raises(ConfigError, match="Registered names: a_v1, b_v1") as caught:
            resolveComponent(analysisRegistry(), {"type": "c_v1"})

        assert caught.value.field_path == "type"


class TestARegistrationThatIsRefused:
    def assertNothingWasRegistered(self, table: KindTable) -> None:
        assert table.registry("analysis").names() == []

    @pytest.mark.parametrize(
        "name", ["Upper", "1st_v1", "with space", "trailing_", "", "a__b"]
    )
    def test_an_invalid_name(self, isolatedAnalysisKind: KindTable, name: str) -> None:
        with pytest.raises(RegistryError, match="not a valid component name"):
            defineFake(name)

        self.assertNothingWasRegistered(isolatedAnalysisKind)

    def test_a_name_that_is_taken_is_never_replaced(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("analysis_v1", description="The first one.")

        with pytest.raises(RegistryError, match="already registered") as caught:
            defineFake("analysis_v1", description="The second one.")

        assert "permanent identifier" in str(caught.value)
        assert analysisRegistry().lookup("analysis_v1").description == "The first one."

    def test_a_class_that_is_not_a_analysis(self, isolatedAnalysisKind: KindTable) -> None:
        class NotAAnalysis:
            """Not a subclass of the contract."""

        with pytest.raises(
            RegistryError, match="not a subclass of AbstractAnalysis"
        ) as caught:
            registerAnalysis("plain_v1", config=fakeConfig("plain_v1"), description="Plain.")(
                NotAAnalysis  # type: ignore[type-var]
            )

        assert caught.value.component == "analysis 'plain_v1'"
        self.assertNothingWasRegistered(isolatedAnalysisKind)

    def test_a_goal_is_not_a_analysis(self, isolatedAnalysisKind: KindTable) -> None:
        # The kinds have contracts of their own: a class of another kind is refused here.
        from planexs.goals.ExploreGoal import ExploreGoal

        with pytest.raises(RegistryError, match="not a subclass of AbstractAnalysis"):
            registerAnalysis("goal_v1", config=fakeConfig("goal_v1"), description="A goal.")(
                ExploreGoal  # type: ignore[type-var]
            )

        self.assertNothingWasRegistered(isolatedAnalysisKind)

    def test_a_schema_whose_type_is_not_pinned_to_the_name(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        class OtherConfig(ComponentConfig):
            """A schema pinned to another name."""

            type: Literal["other_v1"] = Field(default="other_v1", description="The name.")

        with pytest.raises(RegistryError, match="pinned to the registered name") as caught:

            @registerAnalysis("analysis_v1", config=OtherConfig, description="Mismatch.")
            class Mismatch(FakeBehaviour):
                pass

        assert caught.value.field_path == "type"
        self.assertNothingWasRegistered(isolatedAnalysisKind)

    @pytest.mark.parametrize("description", ["", "   ", "Two\nlines."])
    def test_a_description_that_is_not_one_line(
        self, isolatedAnalysisKind: KindTable, description: str
    ) -> None:
        with pytest.raises(RegistryError, match="description"):
            defineFake(description=description)

        self.assertNothingWasRegistered(isolatedAnalysisKind)

    def test_a_needs_that_is_a_single_string_and_not_a_collection(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="needs"):
            defineFake(needs="toy:gaussian")  # type: ignore[arg-type]

        self.assertNothingWasRegistered(isolatedAnalysisKind)

    @pytest.mark.parametrize("side", ["needs", "supports"])
    def test_a_tag_that_is_not_in_the_vocabulary(
        self, isolatedAnalysisKind: KindTable, side: str
    ) -> None:
        # A typo in a declaration is an error where it is made, not a need nobody can meet.
        with pytest.raises(RegistryError, match="toy:batsh"):
            defineFake(**{side: frozenset({"toy:batsh"})})

        self.assertNothingWasRegistered(isolatedAnalysisKind)


class TestTheCompatibilityCheck:
    """An analysis is a participant: its tags are what the compatibility check reads (5.12)."""

    def study(self, *needs: str, supports: tuple[str, ...] = ()) -> DeclaredParticipant:
        return DeclaredParticipant(
            label="study", needs=frozenset(needs), supports=frozenset(supports)
        )

    def test_an_analysis_is_a_participant_with_a_label_and_tags(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        registerToyTags(isolatedAnalysisKind)
        defineFake(
            "analysis_v1", needs=frozenset({"toy:gaussian"}), supports=frozenset({"toy:grid"})
        )

        analysis = buildFake("analysis_v1")

        assert analysis.label == "analysis 'analysis_v1'"
        assert analysis.needs == {"toy:gaussian"}
        assert analysis.supports == {"toy:grid"}

    def test_a_need_that_the_analysis_supports_is_met(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        registerToyTags(isolatedAnalysisKind)
        defineFake("analysis_v1", supports=frozenset({"toy:grid"}))
        analysis = buildFake("analysis_v1")

        checkCompatibility([analysis, self.study("toy:grid")], isolatedAnalysisKind)

    def test_a_need_that_no_participant_supports_names_the_analyses_that_would(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        registerToyTags(isolatedAnalysisKind)
        defineFake("pointwise_v1", supports=frozenset({"toy:optimize"}))
        defineFake("grid_v1", supports=frozenset({"toy:grid"}))
        analysis = buildFake("pointwise_v1")

        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([analysis, self.study("toy:grid")], isolatedAnalysisKind)

        message = caught.value.message
        assert "Need 'toy:grid'" in message
        assert "needed by: study" in message
        assert "supported by: analysis 'grid_v1'" in message
        assert "pointwise_v1" not in message.split("supported by:")[1]

    def test_what_an_analysis_needs_of_the_model_is_met_by_the_surrogate(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # The analysis of a Gaussian posterior declares it: a study whose surrogate does not
        # support it is refused before anything is computed.
        registerToyTags(isolatedAnalysisKind)
        defineFake("analysis_v1", needs=frozenset({"toy:gaussian"}))
        analysis = buildFake("analysis_v1")

        with pytest.raises(CompatibilityError, match="Need 'toy:gaussian'"):
            checkCompatibility([analysis], isolatedAnalysisKind)

        surrogate = DeclaredParticipant(
            label="surrogate 'toy_v1'", supports=frozenset({"toy:gaussian"})
        )
        checkCompatibility([analysis, surrogate], isolatedAnalysisKind)

    def test_what_an_analysis_needs_of_the_goal_is_met_by_the_goal(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        registerToyTags(isolatedAnalysisKind)
        defineFake("analysis_v1", needs=frozenset({"toy:optimize"}))
        analysis = buildFake("analysis_v1")
        goal = DeclaredParticipant(label="goal 'explore_v1'")

        with pytest.raises(CompatibilityError, match="Need 'toy:optimize'"):
            checkCompatibility([analysis, goal], isolatedAnalysisKind)

        optimizing = DeclaredParticipant(
            label="goal 'optimize_v1'", supports=frozenset({"toy:optimize"})
        )
        checkCompatibility([analysis, optimizing], isolatedAnalysisKind)

    def test_the_tags_narrowed_to_a_configuration_decide_the_outcome_of_the_check(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # The entry needs a Gaussian posterior, but an instance that does not ask for the
        # surrogate needs nothing, and one that does ask for it needs the tag.
        registerToyTags(isolatedAnalysisKind)
        defineRefining()
        without_model = buildFake("refining_v1")
        with_model = buildFake("refining_v1", use_model=True)

        checkCompatibility([without_model], isolatedAnalysisKind)
        with pytest.raises(CompatibilityError, match="needed by: analysis 'refining_v1'"):
            checkCompatibility([with_model], isolatedAnalysisKind)

    def test_the_support_narrowed_to_a_configuration_decides_too(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        registerToyTags(isolatedAnalysisKind)
        defineRefining()
        without_model = buildFake("refining_v1")
        with_model = buildFake("refining_v1", use_model=True)
        surrogate = DeclaredParticipant(
            label="surrogate", supports=frozenset({"toy:gaussian"})
        )

        checkCompatibility([without_model, self.study("toy:grid")], isolatedAnalysisKind)
        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility(
                [with_model, surrogate, self.study("toy:grid")], isolatedAnalysisKind
            )

        assert "supported by: analysis 'refining_v1'" in caught.value.message

    def test_an_analysis_whose_tag_is_not_in_the_vocabulary_is_reported_as_a_typo(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # The check reads the tags of the analysis object, and a typo there is not an unmet
        # need that nobody could ever meet: it is an error that names the participant.
        defineFake("analysis_v1")
        analysis = buildFake("analysis_v1")

        class Typo(type(analysis)):  # type: ignore[misc]
            @property
            def supports(self) -> frozenset[str]:
                return frozenset({"toy:gridd"})

        with pytest.raises(RegistryError, match="toy:gridd") as caught:
            checkCompatibility([Typo(analysis.config)], isolatedAnalysisKind)

        assert caught.value.component == "analysis 'analysis_v1'"
        assert caught.value.field_path == "supports"
