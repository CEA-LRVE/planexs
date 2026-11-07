"""Unit tests for `registerOutcomeKind`, with a toy outcome kind registered in the tests.

The toy kind is a count (a non-negative integer): a kind that the two built-in ones cannot
express, defined here the way a plugin would define it (specification 1.3). It is registered
in a copy of the domain's kinds (the `isolatedDomainKinds` fixture), whose capability
vocabulary is a copy too, because a registered name and a capability tag are permanent and
must not outlive the test that registers them.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import Field

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.domain.Outcome import Outcome
from planexs.domain.outcome_kinds.ContinuousKindConfig import ContinuousKindConfig
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.outcomeKindRegistry import outcomeKindRegistry
from planexs.domain.outcome_kinds.OutcomeValueError import OutcomeValueError
from planexs.domain.outcome_kinds.registerOutcomeKind import registerOutcomeKind
from planexs.domain.outcomeFromDict import outcomeFromDict

#: What the real registry and the real vocabulary hold, taken when this module is collected:
#: no test in this file may leave anything in them.
REAL_REGISTRY = outcomeKindRegistry()
REAL_NAMES = ["binary", "continuous"]
REAL_TAGS = ["outcome_kind:binary", "outcome_kind:continuous"]
MISSING = object()
DESCRIPTION = "A count: a non-negative integer."


def defineCount(
    register: bool = True, **attributes: Any
) -> tuple[type[OutcomeKind], type[ComponentConfig]]:
    """Defines the toy kind (a schema and a class), and registers it unless told not to.

    The class declares the tag and the noise rule that `attributes` override (`MISSING` leaves
    one out). It is defined inside a function so that it is registered when the test calls it,
    with the isolated table already swapped in, and not when the module is imported.
    """

    class CountKindConfig(ComponentConfig):
        type: Literal["count"] = Field(default="count", description="Always 'count'.")

    class CountKind(OutcomeKind):
        def validateValue(self, value: object, outcome: str) -> float:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise OutcomeValueError(
                    f"The value {value!r} of the count outcome {outcome!r} is not a "
                    "non-negative integer.",
                    component=f"outcome {outcome!r}",
                )
            return float(value)

    declared: dict[str, Any] = {"capability_tag": "outcome_kind:count", "allows_noise": False}
    declared.update(attributes)
    for key, value in declared.items():
        if value is not MISSING:
            setattr(CountKind, key, value)
    if register:
        registerOutcomeKind("count", config=CountKindConfig, description=DESCRIPTION)(
            CountKind
        )
    return CountKind, CountKindConfig


def assertNothingWasRegistered(table: KindTable, name: str = "count") -> None:
    assert name not in outcomeKindRegistry()
    assert outcomeKindRegistry().names() == REAL_NAMES
    assert table.vocabulary.tags("outcome_kind") == REAL_TAGS


class TestAToyKindRegisteredInTheTest:
    def test_it_is_registered_next_to_the_built_in_kinds(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineCount()

        assert outcomeKindRegistry().names() == ["binary", "continuous", "count"]
        assert outcomeKindRegistry() is not REAL_REGISTRY

    def test_its_tag_is_registered_with_the_description_of_the_kind(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineCount()

        vocabulary = isolatedDomainKinds.vocabulary
        assert "outcome_kind:count" in vocabulary
        assert vocabulary.describe("outcome_kind:count") == DESCRIPTION
        assert vocabulary.tags("outcome_kind") == [*REAL_TAGS, "outcome_kind:count"]

    def test_its_tag_does_not_reach_the_real_vocabulary(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineCount()

        assert "outcome_kind:count" not in CapabilityVocabulary.default()
        assert isolatedDomainKinds.vocabulary is not CapabilityVocabulary.default()

    def test_the_entry_declares_no_capability_tag_of_its_own(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineCount()
        entry = outcomeKindRegistry().lookup("count")

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_an_outcome_of_the_toy_kind_validates_its_values(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        kind, _ = defineCount()
        outcome = Outcome(name="defects", type="count")

        assert type(outcome.kind()) is kind
        assert outcome.kind().allows_noise is False
        assert outcome.kind().capability_tag == "outcome_kind:count"
        assert outcome.validateValue(3) == 3.0
        with pytest.raises(OutcomeValueError, match="'defects'"):
            outcome.validateValue(-1)

    def test_an_outcome_of_the_toy_kind_round_trips(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineCount()
        original = Outcome(name="defects", type="count", description="Defects per wafer.")

        reloaded = outcomeFromDict(original.toDict())

        assert reloaded == original

    def test_a_kind_that_is_not_registered_cannot_be_named(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineCount(register=False)

        with pytest.raises(ConfigError) as caught:
            Outcome(name="defects", type="count")

        assert "Unknown outcome_kind type 'count'" in caught.value.message
        assert "Registered names: binary, continuous." in caught.value.message

    def test_it_does_not_outlive_the_test(self) -> None:
        # Runs after the tests above (same file) and uses no fixture: the real registry and
        # the real vocabulary must hold exactly the built-in kinds and their tags.
        assert REAL_REGISTRY.names() == REAL_NAMES
        assert outcomeKindRegistry() is REAL_REGISTRY
        assert CapabilityVocabulary.default().tags("outcome_kind") == REAL_TAGS


class TestTheDecorator:
    def test_it_returns_the_class_unchanged_and_registers_its_builder(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        kind, config = defineCount()
        entry = outcomeKindRegistry().lookup("count")

        assert entry.builder == kind.fromConfig
        assert entry.config_schema is config
        assert entry.description == DESCRIPTION

    def test_the_default_builder_makes_the_kind(self, isolatedDomainKinds: KindTable) -> None:
        kind, config = defineCount()

        assert type(kind.fromConfig(config())) is kind

    def test_a_class_that_is_not_an_outcome_kind_is_refused(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        _, config = defineCount(register=False)

        with pytest.raises(RegistryError, match="not a subclass of OutcomeKind"):
            registerOutcomeKind("count", config=config, description="d")(int)  # type: ignore[type-var]

        assertNothingWasRegistered(isolatedDomainKinds)

    def test_a_schema_that_is_not_pinned_to_the_name_is_refused(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        kind, config = defineCount(register=False)

        with pytest.raises(RegistryError) as caught:
            registerOutcomeKind("tally", config=config, description="d")(kind)

        assert caught.value.field_path == "type"
        assert "pinned" in caught.value.message
        assertNothingWasRegistered(isolatedDomainKinds, "tally")

    def test_a_name_that_is_taken_is_refused_and_its_tag_is_not_registered(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        kind, _ = defineCount(register=False)

        with pytest.raises(RegistryError, match="already registered"):
            registerOutcomeKind("continuous", config=ContinuousKindConfig, description="d")(
                kind
            )

        assert outcomeKindRegistry().lookup("continuous").description != "d"
        assert isolatedDomainKinds.vocabulary.tags("outcome_kind") == REAL_TAGS

    def test_a_name_that_is_not_snake_case_is_refused_and_its_tag_is_not_registered(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        kind, _ = defineCount(register=False)

        class CapitalConfig(ComponentConfig):
            type: Literal["Count"] = Field(default="Count", description="Always 'Count'.")

        with pytest.raises(RegistryError, match="not a valid component name"):
            registerOutcomeKind("Count", config=CapitalConfig, description="d")(kind)

        assertNothingWasRegistered(isolatedDomainKinds, "Count")

    def test_a_capability_tag_in_needs_that_is_unknown_is_refused_and_nothing_is_registered(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        kind, config = defineCount(register=False)

        with pytest.raises(RegistryError, match="outcome_kind:unknown"):
            registerOutcomeKind(
                "count",
                config=config,
                description="d",
                needs=frozenset({"outcome_kind:unknown"}),
            )(kind)

        assertNothingWasRegistered(isolatedDomainKinds)

    def test_a_builder_that_does_not_return_a_kind_is_caught_when_the_outcome_builds_it(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        kind, config = defineCount(register=False)

        class Broken(kind):  # type: ignore[valid-type, misc]
            @classmethod
            def fromConfig(cls, config: ComponentConfig) -> Any:
                return 3

        registerOutcomeKind("count", config=config, description="d")(Broken)

        with pytest.raises(RegistryError, match="which is not an OutcomeKind"):
            Outcome(name="defects", type="count").kind()


class TestTheCapabilityTagOfAKind:
    @pytest.mark.parametrize(
        ("attributes", "field"),
        [
            ({"capability_tag": MISSING}, "capability_tag"),
            ({"capability_tag": 3}, "capability_tag"),
            ({"capability_tag": None}, "capability_tag"),
            ({"capability_tag": "Count"}, "capability_tag"),
            ({"capability_tag": "outcome_kind:Count"}, "capability_tag"),
            ({"capability_tag": "outcome_kind:"}, "capability_tag"),
            ({"capability_tag": "count"}, "capability_tag"),
        ],
        ids=["missing", "int", "none", "no-colon", "uppercase", "empty-name", "no-group"],
    )
    def test_a_tag_that_is_missing_or_malformed_is_refused(
        self, isolatedDomainKinds: KindTable, attributes: dict[str, Any], field: str
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            defineCount(**attributes)

        assert caught.value.field_path == field
        assert "no usable capability_tag" in caught.value.message
        assertNothingWasRegistered(isolatedDomainKinds)

    @pytest.mark.parametrize("tag", ["surrogate:count", "posterior:gaussian", "goal:count"])
    def test_a_tag_outside_the_group_of_the_outcome_kinds_is_refused(
        self, isolatedDomainKinds: KindTable, tag: str
    ) -> None:
        with pytest.raises(RegistryError, match="not in the group 'outcome_kind'") as caught:
            defineCount(capability_tag=tag)

        assert caught.value.field_path == "capability_tag"
        assertNothingWasRegistered(isolatedDomainKinds)

    def test_a_tag_that_another_kind_has_is_refused(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="already registered") as caught:
            defineCount(capability_tag="outcome_kind:continuous")

        assert caught.value.field_path == "capability_tag"
        assertNothingWasRegistered(isolatedDomainKinds)

    @pytest.mark.parametrize("value", [MISSING, None, "yes", 1, 0])
    def test_a_kind_that_does_not_say_whether_noise_is_meaningful_is_refused(
        self, isolatedDomainKinds: KindTable, value: Any
    ) -> None:
        with pytest.raises(RegistryError, match="noise standard deviation") as caught:
            defineCount(allows_noise=value)

        assert caught.value.field_path == "allows_noise"
        assertNothingWasRegistered(isolatedDomainKinds)

    @pytest.mark.parametrize("allows", [True, False])
    def test_a_kind_may_allow_a_noise_or_not(
        self, isolatedDomainKinds: KindTable, allows: bool
    ) -> None:
        kind, _ = defineCount(allows_noise=allows)

        assert kind.allows_noise is allows
        assert Outcome(name="n", type="count").kind().allows_noise is allows
