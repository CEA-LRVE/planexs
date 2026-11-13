"""The top-level package exports its names lazily (roadmap P2-11, specification 3.9, 8).

`import planexs`, `import planexs.core` and `import planexs.domain` must load no
machine-learning library: the core and the domain are pure, and a sibling name exported
from the top-level package is no reason to pull torch in. A name is imported when it is
first asked for. These tests check that, that every exported name still imports, that the
three places that list the names agree, and that a type checker still sees exactly those
names.

Most of them run in a fresh interpreter: what has been imported is state of the process, and
this one has long since imported torch.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import planexs

ROOT = Path(__file__).resolve().parents[2]
INIT = ROOT / "src" / "planexs" / "__init__.py"

#: The libraries that `core` and `domain` must never import (specification 5.1).
MACHINE_LEARNING_LIBRARIES = ("torch", "botorch", "gpytorch", "ax")


def run(code: str) -> str:
    """Runs `code` in a fresh interpreter and returns what it printed.

    Warnings are off: importing torch writes some, and they are not what is under test.
    """
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def loadedLibraries(statements: str) -> list[str]:
    """The machine-learning libraries loaded after `statements`, in a fresh process."""
    code = (
        f"import json, sys\n{statements}\n"
        f"libraries = {MACHINE_LEARNING_LIBRARIES!r}\n"
        "print(json.dumps(sorted(m for m in libraries if m in sys.modules)))"
    )
    return list(json.loads(run(code)))


def exportsDefinedUnder(*prefixes: str) -> list[str]:
    """The exported names whose defining module is under one of the packages."""
    return sorted(
        name
        for name, module in planexs._EXPORTS.items()  # type: ignore[attr-defined]
        if module.startswith(prefixes)
    )


class TestImportingLoadsNoMachineLearningLibrary:
    @pytest.mark.parametrize("module", ["planexs", "planexs.core", "planexs.domain"])
    def test_importing_the_package(self, module: str) -> None:
        assert loadedLibraries(f"import {module}") == []

    def test_the_check_would_notice_a_library(self) -> None:
        # Without this, the tests above would pass if the detector never fired. The encoder
        # works on tensors, so importing it does load torch.
        assert "torch" in loadedLibraries("import planexs.encoding.SearchSpaceEncoder")

    def test_importing_the_package_loads_none_of_its_own_modules(self) -> None:
        loaded = run(
            "import sys, planexs\n"
            "print(sorted(m for m in sys.modules if m.startswith('planexs.')))"
        )

        assert loaded == "[]"

    def test_asking_for_a_name_of_the_domain_or_the_core_loads_none_either(self) -> None:
        names = exportsDefinedUnder("planexs.domain.", "planexs.core.")
        assert names, "no export is defined in the domain or the core"
        statements = (
            "import sys, planexs\n"
            f"for name in {names!r}:\n"
            "    getattr(planexs, name)\n"
            f"    loaded = [m for m in {MACHINE_LEARNING_LIBRARIES!r} if m in sys.modules]\n"
            "    assert not loaded, f'{name} loaded {loaded}'"
        )

        assert loadedLibraries(statements) == []

    def test_a_from_import_of_such_names(self) -> None:
        statements = "from planexs import Real, SearchSpace, LinearConstraint, Choice"

        assert loadedLibraries(statements) == []


class TestLoadingOnDemand:
    @pytest.fixture
    def probe(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
        """A module of its own, exported as `Probe`, that has not been imported yet."""
        (tmp_path / "lazy_probe_module.py").write_text(
            "class Probe:\n    pass\n", encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setitem(planexs._EXPORTS, "Probe", "lazy_probe_module")  # type: ignore[attr-defined]
        yield "Probe"
        vars(planexs).pop("Probe", None)
        sys.modules.pop("lazy_probe_module", None)

    def test_a_name_is_imported_when_it_is_asked_for_and_not_before(self, probe: str) -> None:
        assert "lazy_probe_module" not in sys.modules

        value = planexs.Probe  # type: ignore[attr-defined]

        assert "lazy_probe_module" in sys.modules
        assert value is sys.modules["lazy_probe_module"].Probe

    def test_a_name_that_was_asked_for_is_kept_in_the_module(
        self, probe: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = planexs.Probe  # type: ignore[attr-defined]
        assert "Probe" in vars(planexs)

        # If the lookup came back to `__getattr__`, this would now fail to import.
        monkeypatch.setitem(planexs._EXPORTS, "Probe", "no_such_module")  # type: ignore[attr-defined]

        assert planexs.Probe is first  # type: ignore[attr-defined]

    def test_a_name_that_does_not_exist_is_an_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="module 'planexs' has no attribute 'Realy'"):
            planexs.Realy  # type: ignore[attr-defined]  # noqa: B018

        assert not hasattr(planexs, "Realy")
        assert getattr(planexs, "Realy", 5) == 5

    def test_a_module_that_cannot_be_imported_is_an_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A missing optional dependency (Ax is meant to become one) must say so, and not
        # look like a name that was never exported.
        monkeypatch.setitem(planexs._EXPORTS, "Ghost", "planexs.no_such_module")  # type: ignore[attr-defined]

        with pytest.raises(ImportError, match="no_such_module"):
            planexs.Ghost  # type: ignore[attr-defined]  # noqa: B018

    def test_dir_lists_the_names_that_are_not_loaded_yet(self) -> None:
        listed = run(
            "import planexs\n"
            "print(all(name in dir(planexs) for name in planexs.__all__),"
            " 'BayesianOptimizer' in vars(planexs))"
        )

        assert listed == "True False"


class TestEveryExportedName:
    def test_every_name_in_all_imports_and_is_what_the_table_says(self) -> None:
        for name in planexs.__all__:
            value = getattr(planexs, name)
            if name == "__version__":
                assert isinstance(value, str) and value
                continue
            assert value.__name__ == name, name
            assert value.__module__ == planexs._EXPORTS[name], name  # type: ignore[attr-defined]

    def test_a_from_import_statement_of_every_name(self) -> None:
        names = ", ".join(planexs.__all__)

        assert run(f"from planexs import {names}\nprint('ok')") == "ok"

    def test_a_star_import_binds_every_name(self) -> None:
        listed = run(
            "namespace = {}\n"
            "exec('from planexs import *', namespace)\n"
            "import planexs\n"
            "print(sorted(set(planexs.__all__) - set(namespace)))"
        )

        assert listed == "[]"

    def test_all_has_no_duplicate(self) -> None:
        assert len(planexs.__all__) == len(set(planexs.__all__))

    def test_the_version_is_a_string_from_the_package_metadata(self) -> None:
        assert re.fullmatch(r"\d+\.\d+\.\d+.*", planexs.__version__) or (
            planexs.__version__ == "0.0.0+unknown"
        )


def initTree() -> ast.Module:
    return ast.parse(INIT.read_text(encoding="utf-8"))


def isTypeCheckingTest(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "TYPE_CHECKING"


class TestTheThreePlacesThatListTheNamesAgree:
    """The names are listed under `TYPE_CHECKING` (for type checkers), in `_EXPORTS` (for the
    run time) and in `__all__`. A name in only some of them would work for one reader and not
    another.
    """

    def typeCheckingImports(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for node in initTree().body:
            if isinstance(node, ast.If) and isTypeCheckingTest(node.test):
                for statement in node.body:
                    assert isinstance(statement, ast.ImportFrom) and statement.module
                    for alias in statement.names:
                        assert alias.asname is None, alias.name
                        found[alias.name] = statement.module
        return found

    def test_the_imports_for_type_checkers_are_the_table(self) -> None:
        imports = self.typeCheckingImports()

        assert imports, "the scan found no import under TYPE_CHECKING"
        assert imports == planexs._EXPORTS  # type: ignore[attr-defined]

    def test_all_is_the_table_and_the_version(self) -> None:
        expected = {*planexs._EXPORTS, "__version__"}  # type: ignore[attr-defined]

        assert set(planexs.__all__) == expected

    def test_the_lookup_is_hidden_from_type_checkers(self) -> None:
        # In the open, `__getattr__` would make every attribute of the module valid for a type
        # checker, so that a misspelled name would not be reported.
        top_level = [n.name for n in initTree().body if isinstance(n, ast.FunctionDef)]
        hidden = [
            statement.name
            for node in initTree().body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isTypeCheckingTest(node.test.operand)
            for statement in node.body
            if isinstance(statement, ast.FunctionDef)
        ]

        assert "__getattr__" not in top_level
        assert "__getattr__" in hidden


class TestTypeCheckers:
    """A consumer of the package, type-checked in strict mode as the project is."""

    def test_mypy_strict_sees_the_names_and_only_those(self, tmp_path: Path) -> None:
        pytest.importorskip("mypy")
        names = ", ".join(name for name in planexs.__all__ if name != "__version__")
        # mypy's incremental cache is keyed by module name and checked by the hash of the
        # source, so a file with the same content under another path is reported under the
        # first path. The names are distinctive, and the output is matched by file name.
        (tmp_path / "lazy_consumer_good.py").write_text(
            f"import planexs\n"
            f"from planexs import {names}\n"
            f"version: str = planexs.__version__\n"
            f"parameter = planexs.Real('x', 0.0, 1.0)\n"
            f"reveal_type(planexs.Real)\n",
            encoding="utf-8",
        )
        (tmp_path / "lazy_consumer_bad.py").write_text(
            "import planexs\nfrom planexs import Realy\n\nplanexs.DoesNotExist\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                str(ROOT / "pyproject.toml"),
                str(tmp_path / "lazy_consumer_good.py"),
                str(tmp_path / "lazy_consumer_bad.py"),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )

        lines = result.stdout.splitlines()
        good = [line for line in lines if "lazy_consumer_good.py:" in line]
        bad = [line for line in lines if "lazy_consumer_bad.py:" in line]
        errors = [line for line in good if ": error:" in line]
        assert errors == [], errors  # every name is known, `__version__` is a `str`
        revealed = [line for line in good if "Revealed type" in line]
        assert len(revealed) == 1
        assert "planexs.domain.parameters.Real.Real" in revealed[0]  # the class, not `Any`
        assert "Any" not in revealed[0]
        assert sum('has no attribute "Realy"' in line for line in bad) == 1  # the import
        assert sum('has no attribute "DoesNotExist"' in line for line in bad) == 1
        assert result.returncode == 1  # the errors of bad.py, and only those
