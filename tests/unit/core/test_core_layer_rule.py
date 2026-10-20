"""Tests of the rule of the `core` layer (specification 5.1, roadmap task P1-05).

`core` imports nothing from the rest of planexs and no machine-learning library. The rule is
stated in the docstring of `planexs/core/__init__.py` and checked here on the source of every
module under `src/planexs/core`, so that a later task that adds a module to `core` is held to
it without anyone remembering to.

The check is static (it reads imports from the syntax tree) on purpose: importing any
`planexs.core` module first runs `planexs/__init__.py`, which imports the whole library, so a
runtime check of `sys.modules` cannot see `core` alone yet. Importing in a subprocess and
looking at `sys.modules` is the job of the purity tests of roadmap task P2-12.

Two layers are tested: the scanner on strings, and the real tree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import planexs.core

CORE_DIRECTORY = Path(planexs.core.__file__).resolve().parent

#: Top-level packages that `core` must never import (specification 5.1).
FORBIDDEN_LIBRARIES = ("torch", "botorch", "gpytorch", "ax")

#: The only part of planexs that `core` may import: itself.
CORE_PACKAGE = "planexs.core"


def forbiddenImports(source: str, package: str = CORE_PACKAGE) -> list[str]:
    """The imports in `source` that break the rule of the `core` layer.

    Every import statement counts, wherever it sits: at module level, inside a function, or
    under `TYPE_CHECKING`.

    Args:
        source: The Python source of one module of `core`.
        package: The package the module belongs to, used to resolve relative imports.

    Returns:
        The offending imported module names, in source order.
    """
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            offenders.extend(a.name for a in node.names if _breaksRule(a.name))
        elif isinstance(node, ast.ImportFrom):
            module = _resolve(node, package)
            if module is not None and _breaksRule(module):
                offenders.append(module)
    return offenders


def _resolve(node: ast.ImportFrom, package: str) -> str | None:
    """The absolute name of the module a `from ... import` statement reads from."""
    if node.level == 0:
        return node.module
    parts = package.split(".")
    if node.level > len(parts):
        return None
    base = parts[: len(parts) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _breaksRule(module: str) -> bool:
    """Whether importing `module` from `core` breaks the rule."""
    top_level = module.split(".")[0]
    if top_level in FORBIDDEN_LIBRARIES:
        return True
    if top_level == "planexs":
        return module != CORE_PACKAGE and not module.startswith(CORE_PACKAGE + ".")
    return False


class TestScanner:
    @pytest.mark.parametrize(
        "source",
        [
            "import json\nfrom pathlib import Path\n",
            "from planexs.core.PlanexsError import PlanexsError\n",
            "import planexs.core.Registry\n",
            "from planexs.core import PlanexsError\n",
            "from . import PlanexsError\nfrom .Registry import Registry\n",
            "import numpy\nimport pydantic\n",
        ],
    )
    def test_allowed_imports_are_not_reported(self, source: str) -> None:
        assert forbiddenImports(source) == []

    @pytest.mark.parametrize("library", FORBIDDEN_LIBRARIES)
    def test_a_machine_learning_library_is_reported(self, library: str) -> None:
        assert forbiddenImports(f"import {library}\n") == [library]
        assert forbiddenImports(f"from {library}.sub import thing\n") == [f"{library}.sub"]

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("import planexs\n", ["planexs"]),
            # Even `from planexs import core` is reported: it runs the top-level package, which
            # imports the whole library. `core` names its own modules instead.
            ("from planexs import core\n", ["planexs"]),
            ("from planexs.domain.Trial import Trial\n", ["planexs.domain.Trial"]),
            ("import planexs.api.BayesianOptimizer\n", ["planexs.api.BayesianOptimizer"]),
            # `planexs.corex` is another package that merely starts with the same letters.
            ("import planexs.corex\n", ["planexs.corex"]),
            # One level up from `planexs.core` is `planexs` itself.
            ("from .. import domain\n", ["planexs"]),
            ("from ..domain import Trial\n", ["planexs.domain"]),
        ],
    )
    def test_the_rest_of_planexs_is_reported(self, source: str, expected: list[str]) -> None:
        assert forbiddenImports(source) == expected

    def test_a_deferred_import_inside_a_function_is_reported(self) -> None:
        source = "def build():\n    import torch\n    return torch\n"
        assert forbiddenImports(source) == ["torch"]

    def test_an_import_under_type_checking_is_reported(self) -> None:
        source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import botorch\n"
        assert forbiddenImports(source) == ["botorch"]

    def test_offenders_come_in_source_order(self) -> None:
        source = "import torch\nimport json\nfrom planexs.domain import Trial\nimport ax\n"
        assert forbiddenImports(source) == ["torch", "planexs.domain", "ax"]


class TestRealTree:
    def test_core_has_source_files_to_check(self) -> None:
        # Guards against the glob silently matching nothing.
        assert (CORE_DIRECTORY / "__init__.py").is_file()
        assert (CORE_DIRECTORY / "PlanexsError.py").is_file()

    def test_no_module_of_core_breaks_the_rule(self) -> None:
        offences: dict[str, list[str]] = {}
        for path in sorted(CORE_DIRECTORY.rglob("*.py")):
            relative = path.relative_to(CORE_DIRECTORY)
            package = ".".join([CORE_PACKAGE, *relative.parent.parts])
            found = forbiddenImports(path.read_text(encoding="utf-8"), package)
            if found:
                offences[str(relative)] = found
        assert offences == {}

    def test_the_package_docstring_states_the_rule(self) -> None:
        docstring = " ".join((planexs.core.__doc__ or "").split()).lower()
        assert docstring, "planexs/core/__init__.py has no docstring"
        assert "imports nothing from the rest of planexs" in docstring
        assert "no machine-learning library" in docstring
        # Each forbidden library is named, as a whole word (so "ax" is not found in "relax").
        words = set(re.findall(r"[a-z]+", docstring))
        assert set(FORBIDDEN_LIBRARIES) <= words
