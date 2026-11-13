"""Unit tests for the `goal` kind and its registry (roadmap task P4-02, specification 5.12,
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
from goal_fakes import defineFake

import planexs.goals  # importing the package defines the kind, as the contract test's does
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.goals.goalRegistry import goalRegistry

#: The registry of the real table, taken before any test swaps it.
_REAL_REGISTRY = goalRegistry()


class TestTheKind:
    def test_the_goal_kind_is_in_the_process_wide_table(self) -> None:
        # The contract test parametrizes over `KindTable.default().kinds()`: this is what puts
        # the goals under it, and under the completeness test beside it.
        assert "goal" in KindTable.default()
        assert "goal" in KindTable.default().kinds()

    def test_the_registry_is_the_one_of_the_table(self) -> None:
        assert goalRegistry() is KindTable.default().registry("goal")
        assert goalRegistry().kind == "goal"

    def test_the_registry_checks_tags_against_the_process_wide_vocabulary(self) -> None:
        assert goalRegistry().vocabulary is CapabilityVocabulary.default()

    def test_it_is_the_same_registry_at_every_call(self) -> None:
        assert goalRegistry() is goalRegistry()

    def test_an_unknown_goal_is_an_error_that_names_the_kind(self) -> None:
        with pytest.raises(RegistryError, match=r"Unknown goal 'nonexistent_v1'") as caught:
            goalRegistry().lookup("nonexistent_v1")

        assert caught.value.component == "goal"


class TestTheRegistryIsLookedUpAtEachCall:
    def test_a_swapped_table_is_followed(self, isolatedGoalKind: KindTable) -> None:
        swapped = isolatedGoalKind.registry("goal")

        assert goalRegistry() is swapped
        assert swapped is not _REAL_REGISTRY

    def test_what_a_test_registers_in_a_swapped_table_never_reaches_the_real_one(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("only_in_the_test_v1")

        assert "only_in_the_test_v1" in goalRegistry()
        assert "only_in_the_test_v1" in isolatedGoalKind.registry("goal")

    def test_the_real_registry_is_back_after_the_test(self) -> None:
        # Runs after the test above in file order, and fails if the swap leaked.
        assert "only_in_the_test_v1" not in goalRegistry()
        assert "fake_goal_v1" not in goalRegistry()


class TestImportingThePackage:
    """The kind and its tags exist as soon as `planexs.goals` is imported, in a fresh
    interpreter that has imported nothing else: the population rule (specification 5.12).
    """

    def test_it_defines_the_kind_and_registers_the_tags(self) -> None:
        program = (
            "import planexs.goals\n"
            "from planexs.core.KindTable import KindTable\n"
            "from planexs.core.CapabilityVocabulary import CapabilityVocabulary\n"
            "print('goal' in KindTable.default())\n"
            "print('goal:optimize' in CapabilityVocabulary.default())\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["True", "True"]

    def test_importing_the_registration_decorator_alone_is_enough_to_have_the_tags(
        self,
    ) -> None:
        # A plugin imports `registerGoal` and nothing else of the layer: the tags it declares
        # must exist by the time its decorator runs.
        program = (
            "from planexs.goals.registerGoal import registerGoal\n"
            "from planexs.core.CapabilityVocabulary import CapabilityVocabulary\n"
            "print(CapabilityVocabulary.default().tags('objectives'))\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "['objectives:multi', 'objectives:single']"


class TestTheLayerStaysConfiguration:
    def test_no_module_of_the_layer_imports_an_engine(self) -> None:
        # Not a rule of the layering table, but what follows from its dependencies (a goal is
        # configuration, and the layer may import only `core` and `domain`). Read from the
        # source, because importing `planexs.goals` runs `planexs/__init__.py` first, which
        # still imports every layer.
        engines = {"torch", "botorch", "gpytorch", "ax"}
        found: list[str] = []
        for path in sorted(pathlib.Path(planexs.goals.__file__).parent.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                found += [
                    f"{path.name}: {module}"
                    for module in modules
                    if module.split(".")[0] in engines
                ]

        assert found == []
