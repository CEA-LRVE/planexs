"""Unit tests for the `policy` kind and its registry (roadmap task P6-01, specification 5.12,
10.6).

The registry contract test (`tests/unit/core/test_registry_contract.py`) runs over every kind
of the process-wide table, so the kind is covered by it as soon as it is in the table: what is
tested here is that it is, and what makes it so.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest
from policy_fakes import defineFake

import planexs.policies  # importing the package defines the kind, as the contract test's does
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.policies.policyRegistry import policyRegistry

#: The registry of the real table, taken before any test swaps it.
_REAL_REGISTRY = policyRegistry()


class TestTheKind:
    def test_the_policy_kind_is_in_the_process_wide_table(self) -> None:
        # The contract test parametrizes over `KindTable.default().kinds()`: this is what puts
        # the policies under it, and under the completeness test beside it.
        assert "policy" in KindTable.default()
        assert "policy" in KindTable.default().kinds()

    def test_the_registry_is_the_one_of_the_table(self) -> None:
        assert policyRegistry() is KindTable.default().registry("policy")
        assert policyRegistry().kind == "policy"

    def test_the_registry_checks_tags_against_the_process_wide_vocabulary(self) -> None:
        assert policyRegistry().vocabulary is CapabilityVocabulary.default()

    def test_an_unknown_policy_is_an_error_that_names_the_kind(self) -> None:
        with pytest.raises(RegistryError, match=r"Unknown policy 'nonexistent_v1'") as caught:
            policyRegistry().lookup("nonexistent_v1")

        assert caught.value.component == "policy"

    def test_the_kind_adds_no_tag_to_the_vocabulary_of_its_own(self) -> None:
        # What a policy needs and supports is said with the tags of the layers whose objects
        # it uses, registered where they are defined: no tag starts with the name of the kind.
        assert not [
            tag for tag in CapabilityVocabulary.default().tags() if tag.startswith("policy:")
        ]


class TestTheRegistryIsLookedUpAtEachCall:
    def test_a_swapped_table_is_followed(self, isolatedPolicyKind: KindTable) -> None:
        swapped = isolatedPolicyKind.registry("policy")

        assert policyRegistry() is swapped
        assert swapped is not _REAL_REGISTRY

    def test_the_swapped_table_starts_with_no_policy_and_holds_what_a_test_adds(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        assert policyRegistry().names() == []

        defineFake("only_in_the_test_v1")

        assert policyRegistry().names() == ["only_in_the_test_v1"]

    def test_the_swapped_table_holds_the_real_kinds_the_policies_depend_on(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # A problem looks its outcome kinds up in the table, and a goal its own entry.
        assert "continuous" in isolatedPolicyKind.registry("outcome_kind")
        assert "explore_v1" in isolatedPolicyKind.registry("goal")

    def test_the_real_registry_is_back_after_the_test(self) -> None:
        # Runs after the tests above in file order, and fails if the swap leaked.
        assert "only_in_the_test_v1" not in policyRegistry()
        assert "fake_policy_v1" not in policyRegistry()
        assert policyRegistry() is _REAL_REGISTRY


class TestImportingThePackage:
    """The kind exists as soon as `planexs.policies` is imported, in a fresh interpreter that
    has imported nothing else: the population rule (specification 5.12).
    """

    def test_it_defines_the_kind(self) -> None:
        program = (
            "import planexs.policies\n"
            "from planexs.core.KindTable import KindTable\n"
            "print('policy' in KindTable.default().kinds())\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"

    def test_importing_the_registration_decorator_alone_defines_the_kind(self) -> None:
        # A plugin imports `registerPolicy` and nothing else of the layer.
        program = (
            "from planexs.policies.registerPolicy import registerPolicy\n"
            "from planexs.policies.policyRegistry import policyRegistry\n"
            "print(policyRegistry().kind)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "policy"


class TestThePackageHonoursItsLayer:
    """The layer rule, read from the source: a policy sees the study through the view and does
    not import the layers that drive or store it (specification 5.1).
    """

    HIGHER = ("study", "config", "persistence", "adapters")

    def imports(self) -> set[str]:
        root = pathlib.Path(planexs.policies.__file__).parent
        found: set[str] = set()
        for path in root.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    found.add(node.module)
                elif isinstance(node, ast.Import):
                    found.update(alias.name for alias in node.names)
        return found

    @pytest.mark.parametrize("layer", HIGHER)
    def test_it_imports_no_higher_layer(self, layer: str) -> None:
        offenders = {
            name for name in self.imports() if name.split(".")[:2] == ["planexs", layer]
        }

        assert not offenders

    def test_it_imports_the_contracts_it_is_built_on(self) -> None:
        imported = self.imports()

        assert {"planexs.goals.AbstractGoal", "planexs.domain.Trial"} <= imported
