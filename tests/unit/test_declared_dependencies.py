"""The dependencies that `pyproject.toml` declares for what the code imports."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def pyproject() -> dict[str, Any]:
    """The parsed `pyproject.toml` of the repository."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def packageNames(requirements: Iterable[str]) -> set[str]:
    """The normalized names of the packages that requirement strings name."""
    names = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~\[\s;]", requirement.strip(), maxsplit=1)[0]
        names.add(re.sub(r"[-_.]+", "-", name).lower())
    return names


class TestYamlDependency:
    """Configuration files in YAML (decision D-4) need PyYAML and, for mypy, its stubs."""

    def test_pyyaml_is_a_dependency_of_the_package(self) -> None:
        assert "pyyaml" in packageNames(pyproject()["project"]["dependencies"])

    def test_it_is_bounded_below_and_above(self) -> None:
        (requirement,) = [
            r for r in pyproject()["project"]["dependencies"] if "yaml" in r.lower()
        ]
        assert re.search(r">=\s*6", requirement), requirement
        assert re.search(r"<\s*7", requirement), requirement

    def test_it_is_not_an_optional_extra(self) -> None:
        # The core reads YAML: it must not need an extra to import.
        extras = pyproject()["project"]["optional-dependencies"]
        for name, requirements in extras.items():
            if name != "dev":
                assert "pyyaml" not in packageNames(requirements), name

    def test_the_stubs_for_mypy_are_in_both_lists_of_development_tools(self) -> None:
        project = pyproject()
        extra = project["project"]["optional-dependencies"]["dev"]
        group = project["dependency-groups"]["dev"]
        assert "types-pyyaml" in packageNames(extra)
        assert "types-pyyaml" in packageNames(group)

    def test_the_stubs_are_not_a_runtime_dependency(self) -> None:
        assert "types-pyyaml" not in packageNames(pyproject()["project"]["dependencies"])

    def test_the_lock_file_records_the_requirement(self) -> None:
        lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
        package = lock.split('\nname = "planexs"\n', 1)[1].split("\n[[package]]\n", 1)[0]
        assert '{ name = "pyyaml", specifier = ">=6.0,<7.0" }' in package
        assert 'name = "types-pyyaml"\nversion = ' in lock


class TestNumpyDependency:
    """The random-number tree (`core/RngTree.py`) uses numpy directly: it is declared."""

    def test_numpy_is_a_dependency_of_the_package(self) -> None:
        assert "numpy" in packageNames(pyproject()["project"]["dependencies"])

    def test_it_is_bounded_below_and_above(self) -> None:
        (requirement,) = [
            r for r in pyproject()["project"]["dependencies"] if r.lower().startswith("numpy")
        ]
        # `SeedSequence` and `Generator` need 1.17; 1.26 is the first with Python 3.12 wheels.
        assert re.search(r">=\s*1\.26", requirement), requirement
        assert re.search(r"<\s*3", requirement), requirement

    def test_it_is_not_an_optional_extra(self) -> None:
        extras = pyproject()["project"]["optional-dependencies"]
        for name, requirements in extras.items():
            assert "numpy" not in packageNames(requirements), name

    def test_the_lock_file_records_the_requirement(self) -> None:
        lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
        package = lock.split('\nname = "planexs"\n', 1)[1].split("\n[[package]]\n", 1)[0]
        assert '{ name = "numpy", specifier = ">=1.26,<3.0" }' in package
