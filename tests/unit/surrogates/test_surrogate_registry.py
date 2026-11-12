"""Unit tests for the `surrogate` kind and its registry (roadmap task P3-07, specification
5.12, 10.6).

The registry contract test (`tests/unit/core/test_registry_contract.py`) runs over every kind
of the process-wide table, so the kind is covered by it as soon as it is in the table: what is
tested here is that it is, and what makes it so.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from surrogate_fakes import defineFake

import planexs.surrogates  # noqa: F401  (defines the kind, as the contract test's import does)
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.surrogates.surrogateRegistry import surrogateRegistry

#: The registry of the real table, taken before any test swaps it.
_REAL_REGISTRY = surrogateRegistry()


class TestTheKind:
    def test_the_surrogate_kind_is_in_the_process_wide_table(self) -> None:
        # The contract test parametrizes over `KindTable.default().kinds()`: this is what puts
        # the surrogates under it, and under the completeness test beside it.
        assert "surrogate" in KindTable.default()
        assert "surrogate" in KindTable.default().kinds()

    def test_the_registry_is_the_one_of_the_table(self) -> None:
        assert surrogateRegistry() is KindTable.default().registry("surrogate")
        assert surrogateRegistry().kind == "surrogate"

    def test_the_registry_checks_tags_against_the_process_wide_vocabulary(self) -> None:
        assert surrogateRegistry().vocabulary is CapabilityVocabulary.default()

    def test_it_is_the_same_registry_at_every_call(self) -> None:
        assert surrogateRegistry() is surrogateRegistry()

    def test_an_unknown_surrogate_is_an_error_that_names_the_kind(self) -> None:
        with pytest.raises(
            RegistryError, match=r"Unknown surrogate 'nonexistent_v1'"
        ) as caught:
            surrogateRegistry().lookup("nonexistent_v1")

        assert caught.value.component == "surrogate"


class TestTheRegistryIsLookedUpAtEachCall:
    def test_a_swapped_table_is_followed(self, isolatedSurrogateKind: KindTable) -> None:
        swapped = isolatedSurrogateKind.registry("surrogate")

        assert surrogateRegistry() is swapped
        assert swapped is not _REAL_REGISTRY

    def test_what_a_test_registers_in_a_swapped_table_never_reaches_the_real_one(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        defineFake("only_in_the_test_v1")

        assert "only_in_the_test_v1" in surrogateRegistry()
        assert "only_in_the_test_v1" in isolatedSurrogateKind.registry("surrogate")

    def test_the_real_registry_is_back_after_the_test(self) -> None:
        # Runs after the test above in file order, and fails if the swap leaked.
        assert "only_in_the_test_v1" not in surrogateRegistry()
        assert "fake_v1" not in surrogateRegistry()


class TestImportingThePackage:
    """The kind and its tags exist as soon as `planexs.surrogates` is imported, in a fresh
    interpreter that has imported nothing else: the population rule (specification 5.12).
    """

    def test_it_defines_the_kind_and_registers_the_tags(self) -> None:
        program = (
            "import planexs.surrogates\n"
            "from planexs.core.KindTable import KindTable\n"
            "from planexs.core.CapabilityVocabulary import CapabilityVocabulary\n"
            "print('surrogate' in KindTable.default())\n"
            "print('posterior:gaussian' in CapabilityVocabulary.default())\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["True", "True"]

    def test_importing_the_registration_decorator_alone_is_enough_to_have_the_tags(
        self,
    ) -> None:
        # A plugin imports `registerSurrogate` and nothing else of the layer: the tags it
        # declares must exist by the time its decorator runs.
        program = (
            "from planexs.surrogates.registerSurrogate import registerSurrogate\n"
            "from planexs.core.CapabilityVocabulary import CapabilityVocabulary\n"
            "print(CapabilityVocabulary.default().tags('noise'))\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "['noise:inferred', 'noise:known']"
