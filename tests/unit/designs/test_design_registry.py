"""Unit tests for the `design` kind and its registry (roadmap task P4-06, specification 5.12,
10.6).

The registry contract test (`tests/unit/core/test_registry_contract.py`) runs over every kind
of the process-wide table, so the kind is covered by it as soon as it is in the table, and
`sobol_v1` with it: what is tested here is that they are, and what makes it so.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest
from design_fakes import defineFake

import planexs.designs  # importing the package defines the kind, as the contract test's does
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.designs.designRegistry import designRegistry

#: The registry of the real table, taken before any test swaps it.
_REAL_REGISTRY = designRegistry()


class TestTheKind:
    def test_the_design_kind_is_in_the_process_wide_table(self) -> None:
        # The contract test parametrizes over `KindTable.default().kinds()`: this is what puts
        # the designs under it, and under the completeness test beside it.
        assert "design" in KindTable.default()
        assert "design" in KindTable.default().kinds()

    def test_the_registry_is_the_one_of_the_table(self) -> None:
        assert designRegistry() is KindTable.default().registry("design")
        assert designRegistry().kind == "design"

    def test_the_registry_checks_tags_against_the_process_wide_vocabulary(self) -> None:
        assert designRegistry().vocabulary is CapabilityVocabulary.default()

    def test_sobol_v1_is_registered(self) -> None:
        assert "sobol_v1" in designRegistry()
        assert designRegistry().lookup("sobol_v1").description.splitlines() == [
            designRegistry().lookup("sobol_v1").description
        ]

    def test_an_unknown_design_is_an_error_that_names_the_kind_and_the_known_designs(
        self,
    ) -> None:
        with pytest.raises(RegistryError, match=r"Unknown design 'nonexistent_v1'") as caught:
            designRegistry().lookup("nonexistent_v1")

        assert caught.value.component == "design"
        assert "sobol_v1" in caught.value.message


class TestTheRegistryIsLookedUpAtEachCall:
    def test_a_swapped_table_is_followed(self, isolatedDesignKind: KindTable) -> None:
        swapped = isolatedDesignKind.registry("design")

        assert designRegistry() is swapped
        assert swapped is not _REAL_REGISTRY

    def test_the_swapped_table_holds_the_real_designs_and_what_a_test_adds(
        self, isolatedDesignKind: KindTable
    ) -> None:
        defineFake("only_in_the_test_v1")

        assert designRegistry().names() == ["only_in_the_test_v1", "sobol_v1"]

    def test_the_real_registry_is_back_after_the_test(self) -> None:
        # Runs after the test above in file order, and fails if the swap leaked.
        assert "only_in_the_test_v1" not in designRegistry()
        assert "fake_design_v1" not in designRegistry()


class TestImportingThePackage:
    """The kind and the design exist as soon as `planexs.designs` is imported, in a fresh
    interpreter that has imported nothing else: the population rule (specification 5.12).
    """

    def test_it_defines_the_kind_and_registers_sobol_v1(self) -> None:
        program = (
            "import planexs.designs\n"
            "from planexs.designs.designRegistry import designRegistry\n"
            "print(designRegistry().names())\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "['sobol_v1']"

    def test_importing_the_registration_decorator_alone_defines_the_kind(self) -> None:
        # A plugin imports `registerDesign` and nothing else of the layer.
        program = (
            "from planexs.designs.registerDesign import registerDesign\n"
            "from planexs.core.KindTable import KindTable\n"
            "print('design' in KindTable.default())\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"


class TestTheLayerKeepsToItsDependencies:
    def test_no_module_of_the_layer_imports_a_layer_above_encoding(self) -> None:
        # The layering table says `designs` may import `core`, `domain` and `encoding`. Read
        # from the source, as the layering check does.
        allowed = {"core", "domain", "encoding", "designs"}
        found: list[str] = []
        for path in sorted(pathlib.Path(planexs.designs.__file__).parent.glob("*.py")):
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
