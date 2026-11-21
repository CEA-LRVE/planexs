"""Unit tests for the `acquisition` kind and its registry (roadmap task P5-01, specification
5.12, 10.6).

The registry contract test (`tests/unit/core/test_registry_contract.py`) runs over every kind
of the process-wide table, so the kind is covered by it as soon as it is in the table: what is
tested here is that it is, and what makes it so.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from acquisition_fakes import defineFake

import planexs.acquisition  # noqa: F401  (defines the kind, as the contract test's import does)
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError

#: The registry of the real table, taken before any test swaps it.
_REAL_REGISTRY = acquisitionRegistry()


class TestTheKind:
    def test_the_acquisition_kind_is_in_the_process_wide_table(self) -> None:
        # The contract test parametrizes over `KindTable.default().kinds()`: this is what puts
        # the acquisitions under it, and under the completeness test beside it.
        assert "acquisition" in KindTable.default()
        assert "acquisition" in KindTable.default().kinds()

    def test_the_registry_is_the_one_of_the_table(self) -> None:
        assert acquisitionRegistry() is KindTable.default().registry("acquisition")
        assert acquisitionRegistry().kind == "acquisition"

    def test_the_registry_checks_tags_against_the_process_wide_vocabulary(self) -> None:
        assert acquisitionRegistry().vocabulary is CapabilityVocabulary.default()

    def test_it_is_the_same_registry_at_every_call(self) -> None:
        assert acquisitionRegistry() is acquisitionRegistry()

    def test_an_unknown_acquisition_is_an_error_that_names_the_kind(self) -> None:
        with pytest.raises(
            RegistryError, match=r"Unknown acquisition 'nonexistent_v1'"
        ) as caught:
            acquisitionRegistry().lookup("nonexistent_v1")

        assert caught.value.component == "acquisition"


class TestTheRegistryIsLookedUpAtEachCall:
    def test_a_swapped_table_is_followed(self, isolatedAcquisitionKind: KindTable) -> None:
        swapped = isolatedAcquisitionKind.registry("acquisition")

        assert acquisitionRegistry() is swapped
        assert swapped is not _REAL_REGISTRY

    def test_the_isolated_kind_starts_empty(self, isolatedAcquisitionKind: KindTable) -> None:
        # What the registration tests count and list must not depend on the real acquisitions
        # that the package registers by then.
        assert isolatedAcquisitionKind.registry("acquisition").names() == []

    def test_what_a_test_registers_in_a_swapped_table_never_reaches_the_real_one(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("only_in_the_test_v1")

        assert "only_in_the_test_v1" in acquisitionRegistry()
        assert "only_in_the_test_v1" in isolatedAcquisitionKind.registry("acquisition")

    def test_the_real_registry_is_back_after_the_test(self) -> None:
        # Runs after the test above in file order, and fails if the swap leaked.
        assert "only_in_the_test_v1" not in acquisitionRegistry()
        assert "fake_v1" not in acquisitionRegistry()


class TestImportingThePackage:
    """The kind and its three tags exist as soon as `planexs.acquisition` is imported, in a
    fresh interpreter that has imported nothing else: the population rule (specification
    5.12). The third tag is registered by the goal layer, which the registry imports for it.
    """

    def test_it_defines_the_kind_and_has_the_three_tags(self) -> None:
        program = (
            "import planexs.acquisition\n"
            "from planexs.core.CapabilityVocabulary import CapabilityVocabulary\n"
            "from planexs.core.KindTable import KindTable\n"
            "vocabulary = CapabilityVocabulary.default()\n"
            "print('acquisition' in KindTable.default())\n"
            "for tag in ('acquisition:differentiable', 'pending:supported',\n"
            "            'outcome_constraints:supported'):\n"
            "    print(tag in vocabulary)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["True", "True", "True", "True"]

    def test_importing_the_registry_module_alone_is_enough_to_define_the_kind(self) -> None:
        program = (
            "from planexs.acquisition.acquisitionRegistry import acquisitionRegistry\n"
            "print(acquisitionRegistry().kind)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["acquisition"]

    def test_the_package_pulls_in_no_engine_it_does_not_need(self) -> None:
        # The layer may import BoTorch, but never Ax: only `adapters` may (specification 5.1).
        program = (
            "import sys\n"
            "import planexs.acquisition\n"
            "from planexs.acquisition.registerAcquisition import registerAcquisition\n"
            "print('ax' in sys.modules)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["False"]
