"""Unit tests for the `likelihood` kind: its contract, its registry and its decorator (roadmap
task P3-11, specification 5.12, 6, 10.6).

A fake likelihood stands for one that a plugin registers: the contract is tested against code
that planexs does not ship. Every test that registers something uses `isolatedSurrogateKind`,
because a registered name is permanent and must never reach the real registry.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, ClassVar, Literal

import pytest
from gpytorch.likelihoods import GaussianLikelihood
from likelihood_fakes import makeContext

import planexs.surrogates  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.registerOutcomeKind import registerOutcomeKind
from planexs.surrogates.likelihoods.AbstractLikelihoodBuilder import AbstractLikelihoodBuilder
from planexs.surrogates.likelihoods.LikelihoodBuildContext import LikelihoodBuildContext
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry
from planexs.surrogates.likelihoods.registerLikelihood import registerLikelihood

CONTINUOUS = "outcome_kind:continuous"
BINARY = "outcome_kind:binary"


class StudentTConfig(ComponentConfig):
    """The configuration of the fake likelihood, with a setting of its own."""

    type: Literal["student_t_v1"] = "student_t_v1"
    degrees: float = 4.0


def defineFake(
    name: str = "student_t_v1",
    *,
    supports: frozenset[str] = frozenset({CONTINUOUS}),
    needs: frozenset[str] = frozenset(),
    config: type[ComponentConfig] = StudentTConfig,
) -> type[AbstractLikelihoodBuilder[Any]]:
    """Registers a fake likelihood in the table that is current, and returns its class.

    The class records the contexts it is asked to build in `built`, which is the way a test
    sees whether `build` called `createLikelihood` at all.
    """

    @registerLikelihood(
        name, config=config, description="A fake likelihood.", needs=needs, supports=supports
    )
    class FakeLikelihoodBuilder(AbstractLikelihoodBuilder[Any]):
        built: ClassVar[list[LikelihoodBuildContext]] = []

        def createLikelihood(self, context: LikelihoodBuildContext) -> GaussianLikelihood:
            self.built.append(context)
            return GaussianLikelihood()

    FakeLikelihoodBuilder.built = []
    return FakeLikelihoodBuilder


class TestTheKind:
    def test_it_holds_the_two_builtin_likelihoods(self) -> None:
        assert likelihoodRegistry().kind == "likelihood"
        assert likelihoodRegistry().names() == ["bernoulli_v1", "gaussian_v1"]
        assert "likelihood" in KindTable.default().kinds()

    def test_every_entry_declares_the_outcome_kinds_it_supports(self) -> None:
        registry = likelihoodRegistry()

        for name in registry.names():
            kinds = {
                tag
                for tag in registry.lookup(name).supports
                if tag.startswith("outcome_kind:")
            }
            assert kinds, f"{name} declares no outcome kind"

    def test_the_two_entries_support_different_kinds(self) -> None:
        registry = likelihoodRegistry()

        assert registry.lookup("gaussian_v1").supports >= {CONTINUOUS}
        assert registry.lookup("bernoulli_v1").supports >= {BINARY}
        assert BINARY not in registry.lookup("gaussian_v1").supports
        assert CONTINUOUS not in registry.lookup("bernoulli_v1").supports

    def test_every_outcome_kind_has_a_likelihood_that_can_model_it(self) -> None:
        registry = likelihoodRegistry()
        tags = KindTable.default().vocabulary.tags("outcome_kind")

        assert tags
        for tag in tags:
            assert any(tag in registry.lookup(name).supports for name in registry.names()), tag

    def test_a_builder_is_made_from_its_resolved_configuration(self) -> None:
        resolved = resolveComponent(likelihoodRegistry(), {"type": "gaussian_v1"})

        builder = resolved.entry.builder(resolved.config)

        assert builder.config is resolved.config
        assert toResolvedConfig(builder.config) == {
            "type": "gaussian_v1",
            "noise_lower_bound": None,
            "min_observation_variance": 1e-10,
        }


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        builder = defineFake(
            needs=frozenset({"posterior:latent_gaussian"}),
            supports=frozenset({CONTINUOUS, "noise:inferred"}),
        )

        entry = likelihoodRegistry().lookup("student_t_v1")

        assert entry.description == "A fake likelihood."
        assert entry.needs == {"posterior:latent_gaussian"}
        assert entry.supports == {CONTINUOUS, "noise:inferred"}
        assert entry.config_schema is StudentTConfig
        assert entry.builder == builder.fromConfig

    def test_the_class_is_returned_unchanged(self, isolatedSurrogateKind: KindTable) -> None:
        builder = defineFake()

        assert issubclass(builder, AbstractLikelihoodBuilder)
        assert builder.__name__ == "FakeLikelihoodBuilder"

    def test_a_registered_name_is_permanent(self, isolatedSurrogateKind: KindTable) -> None:
        with pytest.raises(RegistryError, match="already registered"):
            defineFake("gaussian_v1", config=_pinnedTo("gaussian_v1"))

    def test_the_kind_of_the_class_is_checked(self, isolatedSurrogateKind: KindTable) -> None:
        class NotALikelihood:
            pass

        with pytest.raises(RegistryError, match="not a subclass of AbstractLikelihoodBuilder"):
            registerLikelihood("odd_v1", config=_pinnedTo("odd_v1"), description="Odd.")(
                NotALikelihood  # type: ignore[type-var]
            )
        assert "odd_v1" not in likelihoodRegistry()

    def test_the_type_of_the_schema_must_be_pinned_to_the_name(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="must be pinned"):
            defineFake("other_v1")  # StudentTConfig is pinned to 'student_t_v1'

        assert "other_v1" not in likelihoodRegistry()

    def test_a_tag_that_is_not_in_the_vocabulary_is_refused(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="outcome_kind:count"):
            defineFake(supports=frozenset({"outcome_kind:count"}))

        assert "student_t_v1" not in likelihoodRegistry()


class TestTheDeclaredOutcomeKindsAreEnforced:
    def test_a_likelihood_builds_for_an_outcome_of_a_kind_it_supports(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        builder = defineFake()
        context = makeContext("continuous")

        likelihood = _builderOf(builder).build(context)

        assert isinstance(likelihood, GaussianLikelihood)
        assert builder.built == [context]

    def test_it_is_refused_for_an_outcome_of_another_kind_before_anything_is_built(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        # The check is in `build`, which a plugin does not override: a likelihood that a
        # plugin registers is held to the rule without writing it.
        builder = defineFake()

        with pytest.raises(ConfigError, match="cannot model the outcome 'y0'") as error:
            _builderOf(builder).build(makeContext("binary"))

        assert error.value.component == "likelihood 'student_t_v1'"
        assert error.value.field_path == "outcomes[0]"
        assert "continuous" in error.value.message
        assert builder.built == []

    def test_every_outcome_is_checked_and_the_first_that_fails_is_named(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        builder = defineFake(supports=frozenset({CONTINUOUS, BINARY}))
        assert _builderOf(builder).build(makeContext("binary", "continuous", "binary"))

        narrow = defineFake(
            "narrow_v1", supports=frozenset({CONTINUOUS}), config=_pinnedTo("narrow_v1")
        )
        with pytest.raises(ConfigError) as error:
            _builderOf(narrow, "narrow_v1").build(
                makeContext("continuous", "binary", "binary")
            )

        assert error.value.field_path == "outcomes[1]"

    def test_the_error_names_the_registered_likelihoods_that_would_do(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake(
            "plugin_label_v1",
            supports=frozenset({BINARY}),
            config=_pinnedTo("plugin_label_v1"),
        )
        gaussian = resolveComponent(likelihoodRegistry(), {"type": "gaussian_v1"})

        with pytest.raises(ConfigError) as error:
            gaussian.entry.builder(gaussian.config).build(makeContext("binary"))

        assert error.value.suggestion is not None
        assert "bernoulli_v1" in error.value.suggestion
        assert "plugin_label_v1" in error.value.suggestion
        assert "gaussian_v1" not in error.value.suggestion

    def test_a_kind_that_a_plugin_registers_is_judged_like_the_builtin_ones(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        # The check names no kind: it reads the tag that the kind declares. A count outcome,
        # registered here, has no likelihood until one declares it.
        _defineCountKind()
        counts = makeContext("count")
        gaussian = resolveComponent(likelihoodRegistry(), {"type": "gaussian_v1"})

        with pytest.raises(ConfigError, match="'count'") as error:
            gaussian.entry.builder(gaussian.config).build(counts)

        assert error.value.suggestion is not None
        assert "register a likelihood that supports the kind 'count'" in error.value.suggestion

        defineFake(
            "poisson_v1",
            supports=frozenset({"outcome_kind:count"}),
            config=_pinnedTo("poisson_v1"),
        )
        assert _builderOf(None, "poisson_v1").build(counts) is not None


class TestImportingTheKind:
    """The registry checks the tags of an entry when it is registered, and the tags of the
    surrogate layer are registered by another module. Python always runs the `__init__` of
    `planexs.surrogates` before any module below it, so the chain is the same whichever
    module a program imports first, and two starting points, the package and a module deep in
    the kind, stand for all of them. In a fresh interpreter, both likelihoods must end up
    registered.
    """

    @pytest.mark.parametrize(
        "module",
        ["planexs.surrogates", "planexs.surrogates.likelihoods.registerLikelihood"],
    )
    def test_both_likelihoods_are_registered_whichever_module_comes_first(
        self, module: str
    ) -> None:
        program = (
            f"import {module}\n"
            "import planexs.surrogates\n"
            "from planexs.surrogates.likelihoods import likelihoodRegistry as module\n"
            "print(module.likelihoodRegistry().names())\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == "['bernoulli_v1', 'gaussian_v1']"


def _pinnedTo(name: str) -> type[ComponentConfig]:
    """A configuration schema whose `type` is pinned to `name`."""
    return type(
        f"{name.title().replace('_', '')}Config",
        (ComponentConfig,),
        {"__annotations__": {"type": str}, "type": name},
    )


def _builderOf(
    builder_class: type[AbstractLikelihoodBuilder[Any]] | None, name: str = "student_t_v1"
) -> AbstractLikelihoodBuilder[Any]:
    """The builder object of a registered likelihood, made through the registry."""
    resolved = resolveComponent(likelihoodRegistry(), {"type": name})
    built: AbstractLikelihoodBuilder[Any] = resolved.entry.builder(resolved.config)
    return built


def _defineCountKind() -> None:
    """Registers a `count` outcome kind, as a plugin would, in the table that is current."""

    class CountKindConfig(ComponentConfig):
        type: Literal["count"] = "count"

    @registerOutcomeKind("count", config=CountKindConfig, description="A count.")
    class CountKind(OutcomeKind):
        capability_tag: ClassVar[str] = "outcome_kind:count"
        allows_noise: ClassVar[bool] = False

        def validateValue(self, value: object, outcome: str) -> float:
            return float(value)  # type: ignore[arg-type]
