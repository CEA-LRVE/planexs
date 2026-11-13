"""The core and the domain are pure (roadmap task P2-12, specification 3.9, 5.1, 10.11).

`core` and `domain` import no machine-learning library (torch, BoTorch, GPyTorch, Ax), so that
a description of a problem can be read, validated and saved by a process that has none of
them. `core` imports no other layer of planexs at all, and `domain` imports only `core`.

The check is made at run time, in a fresh interpreter that imports every module of the
package, and then looks at what the process has loaded. That sees what
`scripts/check_layering.py` cannot, an import that is built from a string or that happens
through another library, and it sees what a reader of the source misses, a chain of modules
that ends in torch. What it cannot see is an import inside a function that nothing calls:
the static check reads those.

The check is a helper that takes the directory to read, and the tests of it run on throwaway
package trees that hold a module with a forbidden import. So the evidence that it fails when it
should is part of the suite, and it is not a claim about one afternoon.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: The libraries that `core` and `domain` must never import (specification 5.1).
MACHINE_LEARNING_LIBRARIES = ("torch", "botorch", "gpytorch", "ax")

#: Imports every module of a package, then reports what the process holds. `walk_packages`
#: imports the sub-packages itself, and each module is imported by name, as a user would.
_SCRIPT = """
import importlib, json, pkgutil, sys

package = importlib.import_module(sys.argv[1])
names = [package.__name__]
names += [m.name for m in pkgutil.walk_packages(package.__path__, package.__name__ + ".")]
for name in names:
    importlib.import_module(name)
print(json.dumps({"imported": sorted(names), "loaded": sorted(sys.modules)}))
"""


@dataclass(frozen=True)
class Imported:
    """What a fresh interpreter holds after importing every module of one package.

    Attributes:
        imported: The modules of the package that were imported, the package included.
        loaded: Every module in `sys.modules` afterwards.
    """

    imported: frozenset[str]
    loaded: frozenset[str]

    def machineLearningLibraries(self) -> list[str]:
        """The machine-learning libraries that were loaded, by their top-level name."""
        roots = {name.split(".")[0] for name in self.loaded}
        return sorted(roots & set(MACHINE_LEARNING_LIBRARIES))

    def planexsLayers(self) -> set[str]:
        """The layers of planexs that have a module loaded (`core`, `domain`, ...)."""
        return {
            name.split(".")[1]
            for name in self.loaded
            if name.startswith("planexs.") and name.count(".") >= 1
        }


def importEverythingUnder(package: str, source: Path) -> Imported:
    """Imports every module of `package`, found under `source`, in a fresh interpreter.

    Only `source` is on the interpreter's path ahead of the installed packages, so the planexs
    that is read is the one there, whatever else is installed.

    Args:
        package: The dotted name of the package, for example `planexs.core`.
        source: The directory that holds the top-level `planexs` directory.

    Returns:
        What the interpreter held afterwards.
    """
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _SCRIPT, package],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(source)},
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    return Imported(frozenset(data["imported"]), frozenset(data["loaded"]))


def modulesOnDisk(package: str, source: Path) -> set[str]:
    """The dotted name of every module of `package`, from the files under `source`."""
    directory = source.joinpath(*package.split("."))
    names = set()
    for path in directory.rglob("*.py"):
        parts = list(path.relative_to(source).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.add(".".join(parts))
    return names


# Module-level, and not methods of the classes: a class-scoped fixture that is an instance
# method is deprecated (it runs once per class but would be bound to one test's instance).
@pytest.fixture(scope="module")
def core() -> Imported:
    return importEverythingUnder("planexs.core", SRC)


@pytest.fixture(scope="module")
def domain() -> Imported:
    return importEverythingUnder("planexs.domain", SRC)


class TestTheCore:
    def test_every_module_of_the_core_was_imported(self, core: Imported) -> None:
        # Without this, a check that imported only the package, which holds no code, would
        # pass for any core.
        assert core.imported == modulesOnDisk("planexs.core", SRC)
        assert "planexs.core.Registry" in core.imported

    def test_it_loads_no_machine_learning_library(self, core: Imported) -> None:
        assert core.machineLearningLibraries() == []

    def test_it_imports_no_other_layer_of_planexs(self, core: Imported) -> None:
        assert core.planexsLayers() == {"core"}


class TestTheDomain:
    def test_every_module_of_the_domain_was_imported(self, domain: Imported) -> None:
        assert domain.imported == modulesOnDisk("planexs.domain", SRC)
        assert "planexs.domain.SearchSpace" in domain.imported

    def test_it_loads_no_machine_learning_library(self, domain: Imported) -> None:
        assert domain.machineLearningLibraries() == []

    def test_it_imports_only_the_core_besides_itself(self, domain: Imported) -> None:
        # Specification 5.1: the domain may depend on the core, and on nothing else.
        assert domain.planexsLayers() == {"core", "domain"}


def writeTree(root: Path, files: dict[str, str]) -> Path:
    """Writes a throwaway source tree: `files` maps a path under `root` to its source.

    The tree always holds a `planexs` package whose `core` and `domain` are clean, and a stub
    module for each machine-learning library, so that a scratch module can import one without
    the real library being loaded. The stubs come first on the path, in front of any real one.
    """
    clean = {
        "planexs/__init__.py": "",
        "planexs/core/__init__.py": "",
        "planexs/core/clean.py": "VALUE = 1\n",
        "planexs/domain/__init__.py": "",
        "planexs/domain/clean.py": "from planexs.core import clean\n",
        "planexs/encoding/__init__.py": "",
        **{f"{library}.py": "" for library in MACHINE_LEARNING_LIBRARIES},
        **files,
    }
    for relative, source in clean.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


class TestTheChecksCanFail:
    """The checks above, run on trees with a bad module in them: they must notice.

    Each test also checks that the interpreter read the scratch tree, by looking for a module
    that exists only there.
    """

    def test_a_clean_tree_passes_both_checks(self, tmp_path: Path) -> None:
        tree = writeTree(tmp_path, {})

        core = importEverythingUnder("planexs.core", tree)
        domain = importEverythingUnder("planexs.domain", tree)

        assert core.imported == {"planexs.core", "planexs.core.clean"}
        assert core.machineLearningLibraries() == [] and core.planexsLayers() == {"core"}
        assert domain.machineLearningLibraries() == []
        assert domain.planexsLayers() == {"core", "domain"}

    @pytest.mark.parametrize("package", ["core", "domain"])
    @pytest.mark.parametrize("library", MACHINE_LEARNING_LIBRARIES)
    def test_a_module_that_imports_a_machine_learning_library(
        self, tmp_path: Path, package: str, library: str
    ) -> None:
        tree = writeTree(tmp_path, {f"planexs/{package}/leaky.py": f"import {library}\n"})

        found = importEverythingUnder(f"planexs.{package}", tree)

        assert f"planexs.{package}.leaky" in found.imported  # the scratch tree was read
        assert found.machineLearningLibraries() == [library]

    def test_a_module_that_imports_a_submodule_of_a_library(self, tmp_path: Path) -> None:
        # A directory with an `__init__.py` wins over a `torch.py` beside it, so this `torch`
        # is a package with a `nn` in it, as the real one is.
        tree = writeTree(
            tmp_path,
            {
                "planexs/core/leaky.py": "import torch.nn\n",
                "torch/__init__.py": "",
                "torch/nn.py": "",
            },
        )

        found = importEverythingUnder("planexs.core", tree)

        assert "planexs.core.leaky" in found.imported
        assert found.machineLearningLibraries() == ["torch"]

    def test_a_chain_of_modules_that_ends_in_a_library(self, tmp_path: Path) -> None:
        tree = writeTree(
            tmp_path,
            {
                "planexs/core/front.py": "from planexs.core import middle\n",
                "planexs/core/middle.py": "from planexs.core import back\n",
                "planexs/core/back.py": "import botorch\n",
            },
        )

        found = importEverythingUnder("planexs.core", tree)

        assert found.machineLearningLibraries() == ["botorch"]

    def test_a_core_module_that_imports_another_layer(self, tmp_path: Path) -> None:
        tree = writeTree(
            tmp_path, {"planexs/core/reaches_up.py": "import planexs.domain.clean\n"}
        )

        found = importEverythingUnder("planexs.core", tree)

        assert "planexs.core.reaches_up" in found.imported
        assert found.planexsLayers() == {"core", "domain"}

    def test_a_core_module_that_imports_a_layer_through_a_from_import(
        self, tmp_path: Path
    ) -> None:
        tree = writeTree(
            tmp_path, {"planexs/core/reaches_up.py": "from planexs import encoding\n"}
        )

        found = importEverythingUnder("planexs.core", tree)

        assert found.planexsLayers() == {"core", "encoding"}

    def test_a_domain_module_that_imports_a_layer_other_than_the_core(
        self, tmp_path: Path
    ) -> None:
        tree = writeTree(
            tmp_path, {"planexs/domain/reaches_up.py": "import planexs.encoding\n"}
        )

        found = importEverythingUnder("planexs.domain", tree)

        assert "planexs.domain.reaches_up" in found.imported
        assert found.planexsLayers() == {"core", "domain", "encoding"}

    def test_the_check_reads_the_tree_it_is_given_and_not_the_installed_package(
        self, tmp_path: Path
    ) -> None:
        # The real core has a module called `Registry`. The scratch core does not: if the
        # interpreter read the installed package, it would find one.
        tree = writeTree(tmp_path, {})

        found = importEverythingUnder("planexs.core", tree)

        assert "planexs.core.Registry" not in found.loaded
