"""Unit tests for the `fitter` kind and its registry (roadmap tasks P3-13 and P3-15,
specification 5.12, 10.6).

The registry contract test (`tests/unit/core/test_registry_contract.py`) runs over every kind
of the process-wide table, so the kind is covered by it as soon as it is in the table, and
`map_v1` and `variational_elbo_v1` with it: what is tested here is that they are, and what
makes it so.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest
from fitter_fakes import defineFake

import planexs.surrogates  # importing the package defines the kind, as the contract test does
import planexs.surrogates.fitters
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.fitters.MapFitter import MapFitter
from planexs.surrogates.fitters.VariationalElboFitter import VariationalElboFitter

#: The registry of the real table, taken before any test swaps it.
_REAL_REGISTRY = fitterRegistry()

#: The fields of the configuration of each fitter, with the default of each.
EXPECTED_FIELDS = {
    "map_v1": {
        "type": "map_v1",
        "max_attempts": 5,
        "pick_best_of_all_attempts": False,
        "maxiter": None,
    },
    "variational_elbo_v1": {
        "type": "variational_elbo_v1",
        "n_iterations": 100,
        "learning_rate": 0.1,
        "optimizer": "adam",
    },
}


class TestTheKind:
    def test_the_fitter_kind_is_in_the_process_wide_table(self) -> None:
        # The contract test parametrizes over `KindTable.default().kinds()`: this is what puts
        # the fitter under it, and under the completeness test beside it.
        assert "fitter" in KindTable.default()
        assert "fitter" in KindTable.default().kinds()

    def test_the_registry_is_the_one_of_the_table(self) -> None:
        assert fitterRegistry() is KindTable.default().registry("fitter")
        assert fitterRegistry().kind == "fitter"

    def test_the_registry_checks_tags_against_the_process_wide_vocabulary(self) -> None:
        assert fitterRegistry().vocabulary is CapabilityVocabulary.default()

    def test_an_unknown_fitter_is_an_error_that_names_the_kind_and_the_known_fitters(
        self,
    ) -> None:
        with pytest.raises(RegistryError, match=r"Unknown fitter 'nonexistent_v1'") as caught:
            fitterRegistry().lookup("nonexistent_v1")

        assert caught.value.component == "fitter"
        for name in EXPECTED_FIELDS:
            assert name in caught.value.message


class TestTheFitters:
    def test_exactly_the_fitters_of_the_cards_are_registered(self) -> None:
        assert fitterRegistry().names() == ["map_v1", "variational_elbo_v1"]

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_it_has_a_one_line_description(self, name: str) -> None:
        description = fitterRegistry().lookup(name).description

        assert description.strip() and description.splitlines() == [description]

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_its_configuration_describes_every_field_and_pins_its_type(
        self, name: str
    ) -> None:
        schema = fitterRegistry().lookup(name).config_schema

        assert schema.model_fields["type"].default == name
        for field, info in schema.model_fields.items():
            assert info.description and len(info.description.split()) >= 10, field

    @pytest.mark.parametrize(("name", "fields"), EXPECTED_FIELDS.items())
    def test_its_configuration_has_exactly_its_fields_and_their_defaults(
        self, name: str, fields: dict[str, object]
    ) -> None:
        schema = fitterRegistry().lookup(name).config_schema

        assert set(schema.model_fields) == set(fields)
        assert schema().model_dump() == fields

    def test_the_map_fitter_declares_that_it_needs_an_exact_gaussian_posterior(self) -> None:
        entry = fitterRegistry().lookup("map_v1")

        assert entry.needs == frozenset({"posterior:gaussian"})
        assert entry.supports == frozenset()

    def test_the_variational_fitter_declares_that_it_needs_a_latent_gaussian_posterior(
        self,
    ) -> None:
        entry = fitterRegistry().lookup("variational_elbo_v1")

        assert entry.needs == frozenset({"posterior:latent_gaussian"})
        assert entry.supports == frozenset()

    @pytest.mark.parametrize(
        ("name", "fitter_class"),
        [("map_v1", MapFitter), ("variational_elbo_v1", VariationalElboFitter)],
    )
    def test_the_registered_builder_makes_the_fitter_of_the_name(
        self, name: str, fitter_class: type
    ) -> None:
        entry = fitterRegistry().lookup(name)

        built = entry.builder(entry.config_schema())

        assert type(built) is fitter_class


class TestTheDeclarationIsTruthful:
    """A fitter that needs a tag is refused beside a participant that does not support it, when
    a study is built (specification 3.6, 10.6): `map_v1` fits exact models, so it needs
    `posterior:gaussian`, which an exact-GP surrogate supports and a classifier does not, and
    `variational_elbo_v1` fits approximate ones, so it needs `posterior:latent_gaussian`,
    which a classifier supports and an exact-GP surrogate does not.
    """

    @staticmethod
    def _fitter(name: str = "map_v1") -> DeclaredParticipant:
        return DeclaredParticipant.fromEntry("fitter", fitterRegistry().lookup(name))

    def test_it_is_accepted_beside_a_surrogate_with_an_exact_gaussian_posterior(self) -> None:
        surrogate = DeclaredParticipant(
            "surrogate 'exact_v1'", supports=frozenset({"posterior:gaussian"})
        )

        checkCompatibility([self._fitter(), surrogate])

    def test_it_is_refused_beside_a_surrogate_with_a_latent_gaussian_posterior(self) -> None:
        classifier = DeclaredParticipant(
            "surrogate 'classifier_v1'", supports=frozenset({"posterior:latent_gaussian"})
        )

        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([self._fitter(), classifier])

        assert "posterior:gaussian" in caught.value.message
        assert "fitter 'map_v1'" in caught.value.message

    def test_the_variational_fitter_is_accepted_beside_a_latent_gaussian_posterior(
        self,
    ) -> None:
        classifier = DeclaredParticipant(
            "surrogate 'classifier_v1'", supports=frozenset({"posterior:latent_gaussian"})
        )

        checkCompatibility([self._fitter("variational_elbo_v1"), classifier])

    def test_the_variational_fitter_is_refused_beside_an_exact_gaussian_posterior(
        self,
    ) -> None:
        exact = DeclaredParticipant(
            "surrogate 'exact_v1'", supports=frozenset({"posterior:gaussian"})
        )

        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([self._fitter("variational_elbo_v1"), exact])

        assert "posterior:latent_gaussian" in caught.value.message
        assert "fitter 'variational_elbo_v1'" in caught.value.message


class TestTheRegistryIsLookedUpAtEachCall:
    def test_a_swapped_table_is_followed(self, isolatedFitterKind: KindTable) -> None:
        swapped = isolatedFitterKind.registry("fitter")

        assert fitterRegistry() is swapped
        assert swapped is not _REAL_REGISTRY

    def test_the_swapped_table_holds_the_real_fitters_and_what_a_test_adds(
        self, isolatedFitterKind: KindTable
    ) -> None:
        defineFake("only_in_the_test_v1")

        assert fitterRegistry().names() == [
            "map_v1",
            "only_in_the_test_v1",
            "variational_elbo_v1",
        ]

    def test_the_real_registry_is_back_after_the_test(self) -> None:
        # Runs after the test above in file order, and fails if the swap leaked.
        assert "only_in_the_test_v1" not in fitterRegistry()
        assert "fake_fitter_v1" not in fitterRegistry()


def _run(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )


class TestImportingThePackage:
    """The kind and the fitter exist as soon as the package is imported, in a fresh interpreter
    that has imported nothing else: the population rule (specification 5.12).
    """

    @pytest.mark.parametrize("package", ["planexs.surrogates", "planexs.surrogates.fitters"])
    def test_it_defines_the_kind_and_registers_the_fitters(self, package: str) -> None:
        result = _run(
            f"import {package}\n"
            "from planexs.surrogates.fitters.fitterRegistry import fitterRegistry\n"
            "print(fitterRegistry().names())\n"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "['map_v1', 'variational_elbo_v1']"

    def test_importing_the_registration_decorator_alone_defines_the_kind(self) -> None:
        # A plugin imports `registerFitter` and nothing else of the layer.
        result = _run(
            "from planexs.surrogates.fitters.registerFitter import registerFitter\n"
            "from planexs.core.KindTable import KindTable\n"
            "print('fitter' in KindTable.default())\n"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"

    @pytest.mark.parametrize(
        ("module", "tag"),
        [
            ("MapFitter", "posterior:gaussian"),
            ("VariationalElboFitter", "posterior:latent_gaussian"),
        ],
    )
    def test_the_capability_tag_that_the_fitter_needs_exists_whichever_module_comes_first(
        self, module: str, tag: str
    ) -> None:
        # `needs` is checked against the vocabulary when the fitter is registered, so the
        # tags of the surrogate layer must exist by then even when the fitters come first.
        result = _run(
            f"import planexs.surrogates.fitters.{module}\n"
            "from planexs.core.CapabilityVocabulary import CapabilityVocabulary\n"
            f"print({tag!r} in CapabilityVocabulary.default())\n"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"


class TestTheLayerKeepsToItsDependencies:
    def test_no_module_of_the_fitters_imports_a_layer_above_encoding(self) -> None:
        # The layering table says `surrogates` may import `core`, `domain` and `encoding`.
        # Read from the source, as the layering check does.
        allowed = {"core", "domain", "encoding", "surrogates"}
        found: list[str] = []
        for path in sorted(
            pathlib.Path(planexs.surrogates.fitters.__file__).parent.glob("*.py")
        ):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                found += [
                    f"{path.name}: {module}"
                    for module in modules
                    if module.startswith("planexs.") and module.split(".")[1] not in allowed
                ]

        assert found == []
