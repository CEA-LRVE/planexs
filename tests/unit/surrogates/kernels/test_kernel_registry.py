"""Unit tests for the `kernel` kind and its registry (roadmap task P3-10, specification 5.12,
10.6).

The registry contract test (`tests/unit/core/test_registry_contract.py`) runs over every kind
of the process-wide table, so the kind is covered by it as soon as it is in the table, and the
three kernels with it: what is tested here is that they are, and what makes it so.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest
from kernel_fakes import defineFake

import planexs.surrogates  # importing the package defines the kind, as the contract test does
import planexs.surrogates.kernels
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry

#: The registry of the real table, taken before any test swaps it.
_REAL_REGISTRY = kernelRegistry()

#: The fields of the configuration of each kernel, with the default of each.
EXPECTED_FIELDS = {
    "botorch_default_v1": {"type": "botorch_default_v1"},
    "matern_v1": {"type": "matern_v1", "nu": 2.5, "ard": True},
    "rbf_v1": {"type": "rbf_v1", "ard": True},
}


class TestTheKind:
    def test_the_kernel_kind_is_in_the_process_wide_table(self) -> None:
        # The contract test parametrizes over `KindTable.default().kinds()`: this is what puts
        # the kernels under it, and under the completeness test beside it.
        assert "kernel" in KindTable.default()
        assert "kernel" in KindTable.default().kinds()

    def test_the_registry_is_the_one_of_the_table(self) -> None:
        assert kernelRegistry() is KindTable.default().registry("kernel")
        assert kernelRegistry().kind == "kernel"

    def test_the_registry_checks_tags_against_the_process_wide_vocabulary(self) -> None:
        assert kernelRegistry().vocabulary is CapabilityVocabulary.default()

    def test_an_unknown_kernel_is_an_error_that_names_the_kind_and_the_known_kernels(
        self,
    ) -> None:
        with pytest.raises(RegistryError, match=r"Unknown kernel 'nonexistent_v1'") as caught:
            kernelRegistry().lookup("nonexistent_v1")

        assert caught.value.component == "kernel"
        for name in EXPECTED_FIELDS:
            assert name in caught.value.message


class TestTheThreeKernels:
    def test_exactly_the_three_kernels_of_the_card_are_registered(self) -> None:
        assert kernelRegistry().names() == ["botorch_default_v1", "matern_v1", "rbf_v1"]

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_each_has_a_one_line_description(self, name: str) -> None:
        description = kernelRegistry().lookup(name).description

        assert description.strip() and description.splitlines() == [description]

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_each_configuration_describes_every_field_and_pins_its_type(
        self, name: str
    ) -> None:
        schema = kernelRegistry().lookup(name).config_schema

        assert schema.model_fields["type"].default == name
        for field, info in schema.model_fields.items():
            assert info.description and len(info.description.split()) >= 10, (name, field)

    @pytest.mark.parametrize(("name", "fields"), EXPECTED_FIELDS.items())
    def test_each_configuration_has_exactly_its_fields_and_their_defaults(
        self, name: str, fields: dict[str, object]
    ) -> None:
        schema = kernelRegistry().lookup(name).config_schema

        assert set(schema.model_fields) == set(fields)
        assert schema().model_dump() == fields

    @pytest.mark.parametrize("name", EXPECTED_FIELDS)
    def test_each_declares_no_capability_tag(self, name: str) -> None:
        # A kernel is a function of the encoded columns alone: it needs nothing from the study
        # and supports nothing for it, and the declaration is truthful (specification 10.6).
        entry = kernelRegistry().lookup(name)

        assert entry.needs == frozenset() and entry.supports == frozenset()


class TestTheRegistryIsLookedUpAtEachCall:
    def test_a_swapped_table_is_followed(self, isolatedKernelKind: KindTable) -> None:
        swapped = isolatedKernelKind.registry("kernel")

        assert kernelRegistry() is swapped
        assert swapped is not _REAL_REGISTRY

    def test_the_swapped_table_holds_the_real_kernels_and_what_a_test_adds(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake("only_in_the_test_v1")

        assert kernelRegistry().names() == [
            "botorch_default_v1",
            "matern_v1",
            "only_in_the_test_v1",
            "rbf_v1",
        ]

    def test_the_real_registry_is_back_after_the_test(self) -> None:
        # Runs after the test above in file order, and fails if the swap leaked.
        assert "only_in_the_test_v1" not in kernelRegistry()
        assert "fake_kernel_v1" not in kernelRegistry()


def _run(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )


class TestImportingThePackage:
    """The kind and the kernels exist as soon as the package is imported, in a fresh
    interpreter that has imported nothing else: the population rule (specification 5.12).
    """

    @pytest.mark.parametrize("package", ["planexs.surrogates", "planexs.surrogates.kernels"])
    def test_it_defines_the_kind_and_registers_the_three_kernels(self, package: str) -> None:
        result = _run(
            f"import {package}\n"
            "from planexs.surrogates.kernels.kernelRegistry import kernelRegistry\n"
            "print(kernelRegistry().names())\n"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "['botorch_default_v1', 'matern_v1', 'rbf_v1']"

    def test_importing_the_registration_decorator_alone_defines_the_kind(self) -> None:
        # A plugin imports `registerKernel` and nothing else of the layer.
        result = _run(
            "from planexs.surrogates.kernels.registerKernel import registerKernel\n"
            "from planexs.core.KindTable import KindTable\n"
            "print('kernel' in KindTable.default())\n"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"


class TestTheLayerKeepsToItsDependencies:
    def test_no_module_of_the_kernels_imports_a_layer_above_encoding(self) -> None:
        # The layering table says `surrogates` may import `core`, `domain` and `encoding`.
        # Read from the source, as the layering check does.
        allowed = {"core", "domain", "encoding", "surrogates"}
        found: list[str] = []
        for path in sorted(
            pathlib.Path(planexs.surrogates.kernels.__file__).parent.glob("*.py")
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
