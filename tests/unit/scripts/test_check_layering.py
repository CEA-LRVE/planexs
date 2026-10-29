"""Tests of `scripts/check_layering.py` and `scripts/layers.toml` (specification 5.1, 10.5).

The script is not an importable package, so it is loaded from its path. The tests are grouped
by what they pin: the rules file (and that it says what specification 5.1 says), how imports
are read, each rule on fixture trees (layer direction, legacy, the top-level package,
libraries, unclassified packages), the known-violation ratchet with its stale-entry failure,
the repository itself, and the command line and CI wiring.

Most tests run on a small rules file of their own (`SMALL_RULES`), so that what each one
expects can be read off the test. The real rules are tested separately, against the repository
and against the specification.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_layering.py"
RULES_FILE = REPO_ROOT / "scripts" / "layers.toml"
SPECIFICATION = REPO_ROOT / "docs" / "PROJECT_SPECIFICATION.md"
ROADMAP = REPO_ROOT / "docs" / "ROADMAP.md"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yaml"


def _loadChecker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_layering", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `dataclass` looks the module up while it builds the class.
    sys.modules["check_layering"] = module
    spec.loader.exec_module(module)
    return module


checker = _loadChecker()

#: A rules file small enough to read at a glance. `acquisition` and `adapters` are layers that
#: the fixture trees mostly do not contain: a layer may be listed before it exists.
SMALL_RULES = """
[layers]
core = []
domain = ["core"]
persistence = ["core", "domain"]
acquisition = ["core", "domain"]
adapters = ["core", "domain", "acquisition"]

[legacy]
packages = ["models", "backends"]

[[library_rules]]
libraries = ["torch", "ax"]
forbidden_in = ["core", "domain"]
reason = "these layers stay pure"

[[library_rules]]
libraries = ["ax"]
allowed_only_in = ["adapters", "backends"]
reason = "only adapters import Ax"
"""


def known(importer: str, imported: str, fixed_by: str = "P9-99") -> str:
    """A `[[known_violations]]` block for a rules file."""
    return textwrap.dedent(
        f"""
        [[known_violations]]
        importer = "{importer}"
        imported = "{imported}"
        fixed_by = "{fixed_by}"
        reason = "a test violation"
        """
    )


def makeRoot(tmp_path: Path, files: dict[str, str]) -> Path:
    """Writes `files` (path under `src/planexs` to source) and returns the repository root."""
    for relative, source in files.items():
        path = tmp_path / "src" / "planexs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return tmp_path


def reportFor(tmp_path: Path, files: dict[str, str], rules: str = SMALL_RULES) -> Any:
    """The report of checking a tree of `files` against `rules`."""
    root = makeRoot(tmp_path, files)
    return checker.checkTree(root / "src" / "planexs", checker.parseRules(rules), root)


def keysOf(report: Any) -> set[tuple[str, str]]:
    """The `(importer, imported)` of each violation of a report."""
    return {violation.key for violation in report.violations}


def reasonsOf(report: Any) -> list[str]:
    """Every reason of every violation of a report, in order."""
    return [reason for violation in report.violations for reason in violation.reasons]


def run(*arguments: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Runs `main` with the arguments and returns the status and what it printed."""
    code = checker.main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------------------------
# The rules file
# --------------------------------------------------------------------------------------------


class TestParsingRules:
    def test_a_valid_file_is_read_into_its_parts(self) -> None:
        rules = checker.parseRules(SMALL_RULES + known("planexs.domain.A", "persistence"))
        assert rules.layers["domain"] == frozenset({"core"})
        assert rules.layers["core"] == frozenset()
        assert rules.legacy == frozenset({"models", "backends"})
        assert [rule.libraries for rule in rules.library_rules] == [
            frozenset({"torch", "ax"}),
            frozenset({"ax"}),
        ]
        assert rules.library_rules[1].allowed_only_in == frozenset({"adapters", "backends"})
        assert rules.library_rules[0].allowed_only_in is None
        (entry,) = rules.known_violations
        assert entry.key == ("planexs.domain.A", "persistence")
        assert entry.fixed_by == "P9-99"

    def test_legacy_and_known_violations_may_be_left_out(self) -> None:
        rules = checker.parseRules("[layers]\ncore = []\n")
        assert rules.legacy == frozenset()
        assert rules.library_rules == ()
        assert rules.known_violations == ()

    @pytest.mark.parametrize(
        "text, message",
        [
            ("[layers\ncore = []", "not valid TOML"),
            ("", r"non-empty \[layers\] table"),
            ("[layers]\n", r"non-empty \[layers\] table"),
            ("layers = 3", r"non-empty \[layers\] table"),
            ("[layers]\ncore = []\n[layer]\nx = 1\n", "unknown key"),
            ("[layers]\ncore = 'none'\n", "list of non-empty strings"),
            ("[layers]\ncore = [1]\n", "list of non-empty strings"),
            ("[layers]\ncore = ['']\n", "list of non-empty strings"),
            ("[layers]\ndomain = ['core', 'core']\ncore = []\n", "lists a name twice"),
            ("[layers]\ncore = ['core']\n", "lists itself"),
            ("[layers]\ndomain = ['core']\n", "not layers"),
            ("[layers]\ncore = []\n[legacy]\npackages = ['core']\n", "both layers and legacy"),
            ("[layers]\ncore = []\n[legacy]\npackage = ['x']\n", "unknown key"),
            ("legacy = 3\n[layers]\ncore = []\n", r"\[legacy\] must be a table"),
            ("[layers]\ncore = []\nlegacy = 3\n", "layers.legacy must be a list"),
            ("[layers]\ncore = []\n[legacy]\npackages = 'x'\n", "list of non-empty strings"),
        ],
    )
    def test_an_incoherent_file_is_refused_with_a_message(
        self, text: str, message: str
    ) -> None:
        with pytest.raises(checker.LayeringCheckError, match=message):
            checker.parseRules(text)

    def test_layers_that_depend_on_each_other_are_refused_with_the_cycle(self) -> None:
        with pytest.raises(checker.LayeringCheckError, match="a -> b -> a"):
            checker.parseRules('[layers]\na = ["b"]\nb = ["a"]\n')

    def test_a_longer_cycle_is_found_and_named(self) -> None:
        with pytest.raises(checker.LayeringCheckError, match="a -> b -> c -> a"):
            checker.parseRules('[layers]\na = ["b"]\nb = ["c"]\nc = ["a"]\n')

    def test_a_diamond_is_not_a_cycle(self) -> None:
        rules = checker.parseRules(
            '[layers]\ntop = ["left", "right"]\nleft = ["base"]\nright = ["base"]\nbase = []\n'
        )
        assert set(rules.layers) == {"top", "left", "right", "base"}

    @pytest.mark.parametrize(
        "rule, message",
        [
            ('libraries = ["torch"]\nreason = "r"', "forbidden_in.*allowed_only_in"),
            ('forbidden_in = ["core"]\nreason = "r"', "libraries"),
            ('libraries = []\nforbidden_in = ["core"]\nreason = "r"', "must not be empty"),
            ('libraries = ["torch"]\nforbidden_in = ["core"]', "non-empty string `reason`"),
            ('libraries = ["torch"]\nforbidden_in = ["core"]\nreason = " "', "`reason`"),
            ('libraries = ["torch"]\nforbidden_in = ["nowhere"]\nreason = "r"', "nowhere"),
            ('libraries = ["torch"]\nallowed_only_in = ["nowhere"]\nreason = "r"', "nowhere"),
            (
                'libraries = ["torch"]\nforbidden_in = ["core"]\nreason = "r"\nx = 1',
                "unknown key",
            ),
        ],
    )
    def test_a_library_rule_must_be_complete_and_name_real_packages(
        self, rule: str, message: str
    ) -> None:
        with pytest.raises(checker.LayeringCheckError, match=message):
            checker.parseRules(f"[layers]\ncore = []\n[[library_rules]]\n{rule}\n")

    @pytest.mark.parametrize("missing", ["importer", "imported", "fixed_by", "reason"])
    def test_a_known_violation_needs_every_field(self, missing: str) -> None:
        fields = {
            "importer": "planexs.a.B",
            "imported": "core",
            "fixed_by": "P9-99",
            "reason": "r",
        }
        del fields[missing]
        body = "\n".join(f'{key} = "{value}"' for key, value in fields.items())
        with pytest.raises(checker.LayeringCheckError, match=missing):
            checker.parseRules(f"[layers]\ncore = []\n[[known_violations]]\n{body}\n")

    def test_a_known_violation_with_an_unknown_key_is_refused(self) -> None:
        text = "[layers]\ncore = []\n" + known("planexs.a.B", "core") + "line = 3\n"
        with pytest.raises(checker.LayeringCheckError, match="unknown key"):
            checker.parseRules(text)

    def test_a_known_violation_listed_twice_is_refused(self) -> None:
        text = "[layers]\ncore = []\n" + known("planexs.a.B", "core") * 2
        with pytest.raises(checker.LayeringCheckError, match="repeats the known violation"):
            checker.parseRules(text)

    def test_two_entries_for_one_importer_and_different_imports_are_fine(self) -> None:
        text = (
            "[layers]\ncore = []\n" + known("planexs.a.B", "core") + known("planexs.a.B", "x")
        )
        assert len(checker.parseRules(text).known_violations) == 2

    def test_a_rules_file_that_is_missing_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(checker.LayeringCheckError, match="cannot read the rules file"):
            checker.loadRules(tmp_path / "nowhere.toml")

    def test_the_name_of_the_file_is_in_the_messages(self, tmp_path: Path) -> None:
        path = tmp_path / "my_rules.toml"
        path.write_text("[layers\n", encoding="utf-8")
        with pytest.raises(checker.LayeringCheckError, match=r"my_rules\.toml"):
            checker.loadRules(path)


class TestTheRulesFileOfTheRepository:
    """`scripts/layers.toml` says what specification 5.1 and 10.5 say."""

    @staticmethod
    def specificationTable() -> dict[str, str]:
        """Each layer of the table of specification 5.1, with its 'May depend on' cell."""
        text = SPECIFICATION.read_text(encoding="utf-8")
        section = text.split("### 5.1 ", 1)[1].split("\n### ", 1)[0]
        rows: dict[str, str] = {}
        for line in section.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            match = re.fullmatch(r"`(\w+)`", cells[0]) if line.startswith("| `") else None
            if match:
                rows[match.group(1)] = cells[-1]
        return rows

    def test_the_layers_are_those_of_the_specification_table(self) -> None:
        rules = checker.loadRules(RULES_FILE)
        assert set(rules.layers) == set(self.specificationTable())

    def test_every_explicit_row_of_the_table_is_the_same_list(self) -> None:
        rules = checker.loadRules(RULES_FILE)
        explicit = {
            name: cell
            for name, cell in self.specificationTable().items()
            if cell == "nothing in planexs" or "`" in cell
        }
        assert len(explicit) >= 10
        for name, cell in explicit.items():
            wanted = frozenset(re.findall(r"`(\w+)`", cell))
            assert rules.layers[name] == wanted, name

    def test_the_rows_that_say_every_layer_above_depend_on_all_the_layers_above(self) -> None:
        # `policies` is "every layer above" and `config` is "every registry layer": both
        # read as every layer listed before them in the table.
        rules = checker.loadRules(RULES_FILE)
        table = self.specificationTable()
        order = list(table)
        assert table["policies"] == "every layer above"
        assert table["config"] == "every registry layer"
        for name in ("policies", "config"):
            assert rules.layers[name] == frozenset(order[: order.index(name)]), name

    def test_adapters_depend_on_contracts_and_not_on_the_facades(self) -> None:
        # Specification 5.1 says only "planexs' public contracts, plus the third-party
        # package". The rules read that as the contracts of every layer that defines one
        # and not the two facades above the plugins.
        rules = checker.loadRules(RULES_FILE)
        assert "planexs' public contracts" in self.specificationTable()["adapters"]
        assert not rules.layers["adapters"] & {"study", "config", "adapters"}
        assert {"policies", "analyses", "persistence", "core", "domain"} <= rules.layers[
            "adapters"
        ]

    def test_the_layers_never_depend_on_a_layer_that_depends_on_them(self) -> None:
        rules = checker.loadRules(RULES_FILE)
        for name, allowed in rules.layers.items():
            for dependency in allowed:
                assert name not in rules.layers[dependency], (name, dependency)

    def test_persistence_does_not_depend_on_config(self) -> None:
        # Specification 5.1: "Persistence holds configuration as plain JSON-compatible data,
        # which is why it does not depend on `config`."
        assert "config" not in checker.loadRules(RULES_FILE).layers["persistence"]

    def test_the_legacy_group_is_the_one_of_the_roadmap(self) -> None:
        assert checker.loadRules(RULES_FILE).legacy == frozenset(
            {"api", "backends", "models", "analysis"}
        )

    @pytest.mark.parametrize("library", ["ax", "torch", "botorch", "gpytorch"])
    @pytest.mark.parametrize("layer", ["core", "domain"])
    def test_no_machine_learning_library_is_allowed_in_core_and_domain(
        self, library: str, layer: str
    ) -> None:
        rules = checker.loadRules(RULES_FILE)
        assert checker.judgeLibrary(layer, library, rules)

    def test_ax_is_allowed_in_adapters_and_the_legacy_backends_only(self) -> None:
        rules = checker.loadRules(RULES_FILE)
        for unit in ("adapters", "backends"):
            assert checker.judgeLibrary(unit, "ax", rules) == []
        for unit in [*(set(rules.layers) - {"adapters"}), "api", "models", "analysis", None]:
            assert checker.judgeLibrary(unit, "ax", rules), unit

    def test_the_other_layers_may_import_the_engines(self) -> None:
        rules = checker.loadRules(RULES_FILE)
        for unit in ("encoding", "surrogates", "acquisition", "optimizers"):
            for library in ("torch", "botorch", "gpytorch"):
                assert checker.judgeLibrary(unit, library, rules) == [], (unit, library)

    @pytest.mark.parametrize("importer", ["core", "domain", "encoding", "policies", "study"])
    def test_every_pair_of_layers_is_judged_by_the_table(self, importer: str) -> None:
        rules = checker.loadRules(RULES_FILE)
        for imported in rules.layers:
            reasons = checker.judgePlanexs(importer, checker.Target("unit", imported), rules)
            allowed = imported == importer or imported in rules.layers[importer]
            assert (reasons == []) is allowed, (importer, imported, reasons)

    def test_every_known_violation_names_a_task_of_the_roadmap(self) -> None:
        tasks = set(re.findall(r"^### (P\d+-\d+)\.", ROADMAP.read_text("utf-8"), re.MULTILINE))
        assert len(tasks) > 100
        for entry in checker.loadRules(RULES_FILE).known_violations:
            assert entry.fixed_by in tasks, entry

    def test_every_known_violation_is_one_that_exists_in_the_repository(self) -> None:
        # The rules and the code agree both ways: the check passes (no new violation) and no
        # entry is stale.
        rules = checker.loadRules(RULES_FILE)
        report = checker.checkTree(REPO_ROOT / "src" / "planexs", rules, REPO_ROOT)
        assert report.ok, "\n".join(checker.formatReport(report))
        assert {v.key for v in report.violations} == {e.key for e in rules.known_violations}


# --------------------------------------------------------------------------------------------
# Reading imports
# --------------------------------------------------------------------------------------------

KNOWN_NAMES = frozenset({"core", "domain", "persistence", "models", "logging_config"})


def namesIn(source: str, package: str = "planexs.domain") -> list[str]:
    """The modules named by the imports of a snippet, in order."""
    tree = ast.parse(textwrap.dedent(source))
    return [name for name, _ in checker.importedNames(tree, package, KNOWN_NAMES)]


class TestReadingImports:
    @pytest.mark.parametrize(
        "source, expected",
        [
            ("import planexs.core.Registry", ["planexs.core.Registry"]),
            ("import planexs.core.Registry as registry", ["planexs.core.Registry"]),
            ("import os, sys", ["os", "sys"]),
            ("import os.path", ["os.path"]),
            ("from planexs.core.Registry import Registry", ["planexs.core.Registry"]),
            ("from planexs.core import Registry, Other", ["planexs.core"]),
            ("from collections.abc import Callable", ["collections.abc"]),
            ("from ax.api.client import Client", ["ax.api.client"]),
            ("import torch.nn as nn", ["torch.nn"]),
        ],
    )
    def test_absolute_imports(self, source: str, expected: list[str]) -> None:
        assert namesIn(source) == expected

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("from . import Trial", ["planexs.domain"]),
            ("from .Trial import Trial", ["planexs.domain.Trial"]),
            ("from .parameters.Range import Range", ["planexs.domain.parameters.Range"]),
            ("from ..core import Registry", ["planexs.core"]),
            ("from ..core.Registry import Registry", ["planexs.core.Registry"]),
            ("from .. import core", ["planexs.core"]),
            ("from .. import core, persistence", ["planexs.core", "planexs.persistence"]),
            ("from .. import Study", ["planexs"]),
            ("from ... import core", []),
            ("from .... import core", []),
        ],
    )
    def test_relative_imports_are_resolved_from_the_package(
        self, source: str, expected: list[str]
    ) -> None:
        assert namesIn(source) == expected

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("import planexs", ["planexs"]),
            ("from planexs import core", ["planexs.core"]),
            ("from planexs import core, persistence", ["planexs.core", "planexs.persistence"]),
            ("from planexs import logging_config", ["planexs.logging_config"]),
            ("from planexs import BayesianOptimizer", ["planexs"]),
            ("from planexs import *", ["planexs"]),
            ("from planexs import core as c, Study", ["planexs.core", "planexs"]),
        ],
    )
    def test_the_top_level_package_names_a_layer_or_a_re_exported_symbol(
        self, source: str, expected: list[str]
    ) -> None:
        assert namesIn(source) == expected

    def test_a_relative_import_from_an_init_starts_at_the_package_itself(self) -> None:
        # In `domain/__init__.py` the package is `planexs.domain`, so `.` is that package.
        assert namesIn("from . import Trial", package="planexs.domain") == ["planexs.domain"]
        assert namesIn("from .Trial import T", package="planexs.domain") == [
            "planexs.domain.Trial"
        ]

    def test_an_import_deeper_in_the_tree_resolves_from_its_own_package(self) -> None:
        assert namesIn("from ..Trial import T", package="planexs.domain.parameters") == [
            "planexs.domain.Trial"
        ]
        assert namesIn("from ...core import R", package="planexs.domain.parameters") == [
            "planexs.core"
        ]

    @pytest.mark.parametrize(
        "source",
        [
            "def late():\n    import torch\n",
            "class C:\n    import torch\n",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n",
            "try:\n    import torch\nexcept ImportError:\n    pass\n",
            "import sys\nif sys.platform:\n    import torch\nelse:\n    pass\n",
            "with open('f'):\n    import torch\n",
            "async def late():\n    import torch\n",
            "lambda_ = lambda: 1\nimport torch\n",
        ],
        ids=[
            "function",
            "class",
            "type-checking",
            "try",
            "if",
            "with",
            "async-function",
            "after-code",
        ],
    )
    def test_an_import_counts_wherever_it_is(self, source: str) -> None:
        assert "torch" in namesIn(source)

    def test_the_lines_are_those_of_the_statements_and_the_names_come_in_source_order(
        self,
    ) -> None:
        tree = ast.parse(
            "import os\n\n\ndef f():\n    from planexs.core import X\nimport sys\n"
        )
        found = checker.importedNames(tree, "planexs.domain", KNOWN_NAMES)
        assert found == [("os", 1), ("planexs.core", 5), ("sys", 6)]

    def test_a_snippet_with_no_import_names_nothing(self) -> None:
        assert namesIn("x = 1\n") == []
        assert namesIn("# import torch\ns = 'import torch'\n") == []

    @pytest.mark.parametrize(
        "name, kind, target",
        [
            ("planexs.core.Registry", "unit", "core"),
            ("planexs.core", "unit", "core"),
            ("planexs.backends.ax.AxBackend", "unit", "backends"),
            ("planexs.mystery.X", "unit", "mystery"),
            ("planexs", "top", "planexs"),
            ("planexs.logging_config", "top", "planexs"),
            ("torch.nn.functional", "library", "torch"),
            ("ax.api.client", "library", "ax"),
            ("os", "library", "os"),
        ],
    )
    def test_a_name_is_a_unit_the_top_level_package_or_a_library(
        self, name: str, kind: str, target: str
    ) -> None:
        found = checker.classify(name, frozenset({"logging_config"}))
        assert (found.kind, found.name) == (kind, target)

    def test_a_package_called_ax_inside_planexs_is_not_the_library(self) -> None:
        assert (
            checker.classify("planexs.backends.ax.fromAxSearchSpace", frozenset()).kind
            == "unit"
        )
        assert checker.classify("ax.core.parameter", frozenset()).kind == "library"

    @pytest.mark.parametrize(
        "relative, module, package",
        [
            ("core/Registry.py", "planexs.core.Registry", "planexs.core"),
            ("core/__init__.py", "planexs.core", "planexs.core"),
            (
                "domain/parameters/Real.py",
                "planexs.domain.parameters.Real",
                "planexs.domain.parameters",
            ),
            (
                "domain/parameters/__init__.py",
                "planexs.domain.parameters",
                "planexs.domain.parameters",
            ),
            ("logging_config.py", "planexs.logging_config", "planexs"),
            ("__init__.py", "planexs", "planexs"),
        ],
    )
    def test_the_module_and_package_of_a_file(
        self, relative: str, module: str, package: str
    ) -> None:
        assert checker._moduleName(Path(relative)) == (module, package)


# --------------------------------------------------------------------------------------------
# Layer direction
# --------------------------------------------------------------------------------------------


class TestLayerDirection:
    def test_a_layer_may_import_a_layer_that_its_entry_lists(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, {"domain/A.py": "from planexs.core.X import Y\n"})
        assert report.violations == ()

    def test_a_layer_may_import_itself_in_every_spelling(self, tmp_path: Path) -> None:
        source = (
            "from planexs.domain.B import X\n"
            "from .B import X\n"
            "from . import B\n"
            "import planexs.domain.B\n"
            "from planexs import domain\n"
        )
        assert reportFor(tmp_path, {"domain/A.py": source}).violations == ()

    def test_a_layer_may_not_import_a_layer_that_its_entry_does_not_list(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(tmp_path, {"core/A.py": "from planexs.domain.X import Y\n"})
        assert keysOf(report) == {("planexs.core.A", "domain")}
        (violation,) = report.violations
        assert violation.path == "src/planexs/core/A.py"
        assert violation.line == 1
        (reason,) = violation.reasons
        assert "layer 'core' may not depend on layer 'domain'" in reason
        assert "specification 5.1" in reason
        assert "may depend on no layer" in reason

    def test_the_reason_lists_the_layers_that_would_be_allowed(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, {"persistence/A.py": "import planexs.acquisition\n"})
        (reason,) = reasonsOf(report)
        assert "may depend on core, domain" in reason

    def test_the_direction_matters(self, tmp_path: Path) -> None:
        downward = reportFor(
            tmp_path / "down", {"persistence/A.py": "import planexs.domain.X\n"}
        )
        upward = reportFor(tmp_path / "up", {"domain/A.py": "import planexs.persistence.X\n"})
        assert downward.violations == ()
        assert keysOf(upward) == {("planexs.domain.A", "persistence")}

    def test_two_layers_that_do_not_list_each_other_may_not_import_each_other(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(
            tmp_path,
            {
                "persistence/A.py": "import planexs.acquisition.X\n",
                "acquisition/B.py": "import planexs.persistence.X\n",
            },
        )
        assert keysOf(report) == {
            ("planexs.persistence.A", "acquisition"),
            ("planexs.acquisition.B", "persistence"),
        }

    @pytest.mark.parametrize(
        "statement",
        [
            "import planexs.persistence",
            "import planexs.persistence.JsonStudyRepository as repository",
            "from planexs.persistence import JsonStudyRepository",
            "from planexs.persistence.Json import JsonStudyRepository",
            "from ..persistence import JsonStudyRepository",
            "from .. import persistence",
            "from planexs import persistence",
        ],
    )
    def test_each_way_of_importing_an_upper_layer_is_caught(
        self, tmp_path: Path, statement: str
    ) -> None:
        report = reportFor(tmp_path, {"domain/A.py": statement + "\n"})
        assert keysOf(report) == {("planexs.domain.A", "persistence")}

    @pytest.mark.parametrize(
        "source",
        [
            "def late():\n    import planexs.persistence\n",
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    import planexs.persistence\n",
        ],
        ids=["inside-a-function", "type-checking-only"],
    )
    def test_a_dependency_that_is_deferred_or_only_for_the_type_checker_still_counts(
        self, tmp_path: Path, source: str
    ) -> None:
        report = reportFor(tmp_path, {"domain/A.py": source})
        assert keysOf(report) == {("planexs.domain.A", "persistence")}

    def test_a_module_in_a_nested_package_is_judged_as_its_layer(self, tmp_path: Path) -> None:
        report = reportFor(
            tmp_path, {"domain/parameters/Range.py": "from planexs.persistence import X\n"}
        )
        assert keysOf(report) == {("planexs.domain.parameters.Range", "persistence")}
        assert report.violations[0].path == "src/planexs/domain/parameters/Range.py"

    def test_a_layer_that_does_not_exist_yet_is_still_judged(self, tmp_path: Path) -> None:
        # `adapters` is in the table but has no directory in this tree.
        report = reportFor(tmp_path, {"acquisition/A.py": "import planexs.adapters.X\n"})
        assert keysOf(report) == {("planexs.acquisition.A", "adapters")}

    def test_a_package_that_is_in_no_table_cannot_be_imported_by_a_layer(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(tmp_path, {"domain/A.py": "import planexs.mystery.X\n"})
        assert keysOf(report) == {("planexs.domain.A", "mystery")}
        assert "in no table of the rules file" in reasonsOf(report)[0]

    def test_one_violation_is_reported_for_several_imports_of_the_same_package(
        self, tmp_path: Path
    ) -> None:
        source = "\nimport planexs.persistence.A\nfrom planexs.persistence.B import C\n"
        report = reportFor(tmp_path, {"domain/A.py": source})
        (violation,) = report.violations
        assert violation.line == 2  # the first one

    def test_a_layer_may_use_what_is_not_planexs(self, tmp_path: Path) -> None:
        source = "import os\nimport numpy\nfrom collections.abc import Callable\n"
        assert reportFor(tmp_path, {"core/A.py": source}).violations == ()

    def test_every_violation_of_a_module_is_reported_with_its_own_target(
        self, tmp_path: Path
    ) -> None:
        source = "import planexs.persistence\nimport planexs.acquisition\n"
        report = reportFor(tmp_path, {"domain/A.py": source})
        assert keysOf(report) == {
            ("planexs.domain.A", "persistence"),
            ("planexs.domain.A", "acquisition"),
        }


# --------------------------------------------------------------------------------------------
# Legacy
# --------------------------------------------------------------------------------------------


class TestLegacy:
    @pytest.mark.parametrize(
        "source",
        [
            "from planexs.persistence.X import Y\n",
            "from planexs.domain.X import Y\n",
            "from planexs.acquisition.X import Y\n",
            "from planexs.models.X import Y\n",
            "import planexs.backends.ax\n",
            "from planexs import persistence\n",
            "import planexs\n",
            "from planexs import BayesianOptimizer\n",
            "import planexs.mystery\n",
        ],
    )
    def test_a_legacy_package_may_import_anything(self, tmp_path: Path, source: str) -> None:
        assert reportFor(tmp_path, {"backends/A.py": source}).violations == ()

    @pytest.mark.parametrize("legacy", ["models", "backends"])
    @pytest.mark.parametrize("importer", ["core", "domain", "persistence", "acquisition"])
    def test_no_layer_may_import_a_legacy_package(
        self, tmp_path: Path, importer: str, legacy: str
    ) -> None:
        report = reportFor(
            tmp_path, {f"{importer}/A.py": f"from planexs.{legacy}.X import Y\n"}
        )
        assert keysOf(report) == {(f"planexs.{importer}.A", legacy)}
        (reason,) = reasonsOf(report)
        assert f"{legacy!r} is a legacy package" in reason
        assert "only legacy packages, the top-level package and the tests" in reason

    def test_the_reason_is_the_legacy_one_and_not_a_missing_table_entry(
        self, tmp_path: Path
    ) -> None:
        # A legacy package is in no layer's list, but saying so would hide the real reason.
        report = reportFor(tmp_path, {"acquisition/A.py": "from planexs.models.X import Y\n"})
        assert len(reasonsOf(report)) == 1
        assert "legacy" in reasonsOf(report)[0]

    def test_a_legacy_package_may_import_another(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, {"models/A.py": "from planexs.backends.ax import X\n"})
        assert report.violations == ()

    def test_a_legacy_package_imported_from_a_layer_by_relative_import_is_caught(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(tmp_path, {"domain/A.py": "from ..models import X\n"})
        assert keysOf(report) == {("planexs.domain.A", "models")}


# --------------------------------------------------------------------------------------------
# The top-level package
# --------------------------------------------------------------------------------------------


class TestTheTopLevelPackage:
    @pytest.mark.parametrize(
        "source",
        [
            "import planexs\n",
            "import planexs as px\n",
            "from planexs import BayesianOptimizer\n",
            "from planexs import *\n",
            "from planexs import logging_config\n",
            "from planexs.logging_config import configureLogging\n",
            "import planexs.logging_config\n",
            "from .. import BayesianOptimizer\n",
            "from .. import logging_config\n",
        ],
    )
    def test_a_layer_may_not_import_the_top_level_package_or_its_modules(
        self, tmp_path: Path, source: str
    ) -> None:
        files = {"core/A.py": source, "logging_config.py": "X = 1\n"}
        report = reportFor(tmp_path, files)
        assert keysOf(report) == {("planexs.core.A", "planexs")}
        assert "loads every layer" in reasonsOf(report)[0]

    def test_naming_a_layer_after_planexs_is_not_importing_the_top_level_package(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(tmp_path, {"domain/A.py": "from planexs import core\n"})
        assert report.violations == ()

    def test_the_top_level_modules_may_import_anything(self, tmp_path: Path) -> None:
        source = (
            "from planexs.persistence.X import Y\n"
            "from planexs.models.X import Y\n"
            "from planexs.mystery import Z\n"
            "import planexs.logging_config\n"
        )
        report = reportFor(tmp_path, {"__init__.py": source, "logging_config.py": source})
        assert report.violations == ()

    def test_the_top_level_modules_are_still_subject_to_the_library_rules(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(tmp_path, {"__init__.py": "import ax\n"})
        assert keysOf(report) == {("planexs", "ax")}
        assert "the top-level package" in reasonsOf(report)[0]

    def test_a_module_directly_under_the_package_is_not_a_package_to_classify(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(tmp_path, {"logging_config.py": "import os\n", "__init__.py": ""})
        assert report.violations == ()
        assert report.unclassified == ()


# --------------------------------------------------------------------------------------------
# Libraries
# --------------------------------------------------------------------------------------------


class TestLibraries:
    @pytest.mark.parametrize("layer", ["core", "domain"])
    @pytest.mark.parametrize(
        "statement",
        [
            "import torch",
            "import torch.nn as nn",
            "from torch import nn",
            "from torch.nn.functional import relu",
            "import ax",
            "from ax.api.client import Client",
        ],
    )
    def test_a_pure_layer_may_not_import_an_engine(
        self, tmp_path: Path, layer: str, statement: str
    ) -> None:
        report = reportFor(tmp_path, {f"{layer}/A.py": statement + "\n"})
        library = statement.split()[1].split(".")[0]
        assert keysOf(report) == {(f"planexs.{layer}.A", library)}
        assert f"{library!r} may not be imported in {layer!r}" in reasonsOf(report)[0]

    @pytest.mark.parametrize(
        "source",
        [
            "def build():\n    import torch\n",
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from torch import Tensor\n",
            "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
        ],
        ids=["inside-a-function", "type-checking-only", "in-a-try"],
    )
    def test_a_deferred_or_type_only_import_of_an_engine_counts(
        self, tmp_path: Path, source: str
    ) -> None:
        # `core` must run without the engine installed, and a type-only import still makes
        # the type checker need it: the import is a dependency wherever it is written.
        report = reportFor(tmp_path, {"core/A.py": source})
        assert keysOf(report) == {("planexs.core.A", "torch")}

    def test_a_layer_that_is_not_pure_may_import_an_engine(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, {"persistence/A.py": "import torch\n"})
        assert report.violations == ()

    @pytest.mark.parametrize("unit", ["adapters", "backends"])
    def test_ax_is_allowed_in_adapters_and_in_the_legacy_backends(
        self, tmp_path: Path, unit: str
    ) -> None:
        source = "from ax.api.client import Client\nimport ax.core\n"
        assert reportFor(tmp_path, {f"{unit}/A.py": source}).violations == ()

    @pytest.mark.parametrize("unit", ["acquisition", "persistence", "models"])
    def test_ax_is_not_allowed_anywhere_else(self, tmp_path: Path, unit: str) -> None:
        report = reportFor(tmp_path, {f"{unit}/A.py": "from ax.api.client import Client\n"})
        assert keysOf(report) == {(f"planexs.{unit}.A", "ax")}
        (reason,) = reasonsOf(report)
        assert "'ax' may only be imported in adapters, backends" in reason
        assert "only adapters import Ax" in reason

    def test_both_rules_are_reported_when_an_import_breaks_both(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, {"core/A.py": "import ax\n"})
        (violation,) = report.violations
        assert len(violation.reasons) == 2
        assert any("may not be imported in 'core'" in r for r in violation.reasons)
        assert any(
            "may only be imported in adapters, backends" in r for r in violation.reasons
        )

    def test_the_local_package_named_ax_is_not_the_library(self, tmp_path: Path) -> None:
        # `planexs.backends.ax` is planexs code, imported from `api` here.
        report = reportFor(
            tmp_path,
            {
                "api/A.py": "from planexs.backends.ax.AxBackend import AxBackend\n",
                "backends/ax/B.py": "from .C import X\nfrom . import C\n",
            },
        )
        assert report.violations == ()

    def test_a_package_that_shares_a_name_with_a_library_is_not_that_library(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(tmp_path, {"core/A.py": "from .torch import X\nfrom . import ax\n"})
        assert report.violations == ()

    def test_the_standard_library_and_other_packages_are_not_constrained(
        self, tmp_path: Path
    ) -> None:
        source = "import os\nimport json\nimport numpy\nimport pydantic\nimport yaml\n"
        assert reportFor(tmp_path, {"core/A.py": source}).violations == ()


# --------------------------------------------------------------------------------------------
# Packages that are in no table
# --------------------------------------------------------------------------------------------


class TestUnclassifiedPackages:
    def test_a_package_that_is_in_no_table_is_reported(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, {"mystery/A.py": "X = 1\n", "core/B.py": "X = 1\n"})
        assert report.unclassified == ("mystery",)
        assert not report.ok

    def test_every_such_package_is_reported_in_sorted_order(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, {"zeta/A.py": "", "alpha/A.py": "", "core/A.py": ""})
        assert report.unclassified == ("alpha", "zeta")

    def test_a_nested_package_belongs_to_the_package_it_is_in(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, {"core/sub/deeper/A.py": "X = 1\n"})
        assert report.unclassified == ()

    def test_a_directory_with_no_python_file_is_not_a_package(self, tmp_path: Path) -> None:
        root = makeRoot(tmp_path, {"core/A.py": "X = 1\n"})
        (root / "src" / "planexs" / "docs").mkdir()
        (root / "src" / "planexs" / "docs" / "notes.txt").write_text("x", encoding="utf-8")
        (root / "src" / "planexs" / "core" / "__pycache__").mkdir()
        (root / "src" / "planexs" / "core" / "__pycache__" / "A.cpython-312.pyc").write_bytes(
            b""
        )
        report = checker.checkTree(
            root / "src" / "planexs", checker.parseRules(SMALL_RULES), root
        )
        assert report.unclassified == ()

    def test_a_layer_and_a_legacy_package_are_classified(self, tmp_path: Path) -> None:
        report = reportFor(
            tmp_path, {"core/A.py": "", "models/A.py": "", "backends/ax/B.py": ""}
        )
        assert report.unclassified == ()

    def test_the_counts_of_modules_and_packages(self, tmp_path: Path) -> None:
        report = reportFor(
            tmp_path,
            {"__init__.py": "", "core/__init__.py": "", "core/A.py": "", "domain/B.py": ""},
        )
        assert report.module_count == 4
        assert report.package_count == 3  # core, domain, and the top-level package


# --------------------------------------------------------------------------------------------
# The ratchet: known violations
# --------------------------------------------------------------------------------------------

DOMAIN_IMPORTS_PERSISTENCE = {"domain/A.py": "from planexs.persistence.X import Y\n"}


class TestKnownViolations:
    def test_a_violation_that_is_not_listed_is_new(self, tmp_path: Path) -> None:
        report = reportFor(tmp_path, DOMAIN_IMPORTS_PERSISTENCE)
        assert [v.key for v in report.new] == [("planexs.domain.A", "persistence")]
        assert not report.ok

    def test_a_listed_violation_is_accepted(self, tmp_path: Path) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "persistence")
        report = reportFor(tmp_path, DOMAIN_IMPORTS_PERSISTENCE, rules)
        assert len(report.violations) == 1
        assert report.new == ()
        assert report.stale == ()
        assert report.ok

    def test_an_entry_that_matches_nothing_is_stale_and_fails_the_check(
        self, tmp_path: Path
    ) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "persistence", fixed_by="P2-02")
        report = reportFor(tmp_path, {"domain/A.py": "from planexs.core.X import Y\n"}, rules)
        assert report.violations == ()
        assert [entry.key for entry in report.stale] == [("planexs.domain.A", "persistence")]
        assert not report.ok

    def test_fixing_the_import_makes_the_entry_stale(self, tmp_path: Path) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "persistence")
        before = reportFor(tmp_path, DOMAIN_IMPORTS_PERSISTENCE, rules)
        after = reportFor(tmp_path, {"domain/A.py": "from planexs.core.X import Y\n"}, rules)
        assert before.ok
        assert [entry.importer for entry in after.stale] == ["planexs.domain.A"]

    def test_an_entry_for_a_module_that_does_not_exist_is_stale(self, tmp_path: Path) -> None:
        rules = SMALL_RULES + known("planexs.domain.Deleted", "persistence")
        report = reportFor(tmp_path, {"core/A.py": ""}, rules)
        assert [entry.importer for entry in report.stale] == ["planexs.domain.Deleted"]

    def test_the_entry_matches_the_importing_module_and_what_it_imports(
        self, tmp_path: Path
    ) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "persistence")
        files = {
            "domain/A.py": "import planexs.persistence\nimport planexs.acquisition\n",
            "domain/B.py": "import planexs.persistence\n",
        }
        report = reportFor(tmp_path, files, rules)
        # Only the (A, persistence) pair is covered: another target of A, and another module
        # that imports the same package, are new.
        assert {v.key for v in report.new} == {
            ("planexs.domain.A", "acquisition"),
            ("planexs.domain.B", "persistence"),
        }

    def test_a_second_import_of_the_same_package_by_the_same_module_is_covered(
        self, tmp_path: Path
    ) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "persistence")
        source = "import planexs.persistence.A\nimport planexs.persistence.B\n"
        report = reportFor(tmp_path, {"domain/A.py": source}, rules)
        assert report.ok

    def test_the_line_of_the_import_does_not_matter(self, tmp_path: Path) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "persistence")
        source = "# a\n# b\n# c\n\nX = 1\n\nfrom planexs.persistence import Y\n"
        report = reportFor(tmp_path, {"domain/A.py": source}, rules)
        assert report.violations[0].line == 7
        assert report.ok

    def test_new_and_stale_are_reported_together(self, tmp_path: Path) -> None:
        rules = SMALL_RULES + known("planexs.domain.Gone", "persistence")
        report = reportFor(tmp_path, DOMAIN_IMPORTS_PERSISTENCE, rules)
        assert len(report.new) == 1
        assert len(report.stale) == 1

    def test_a_library_violation_can_be_known_too(self, tmp_path: Path) -> None:
        rules = SMALL_RULES + known("planexs.persistence.A", "ax")
        report = reportFor(tmp_path, {"persistence/A.py": "import ax\n"}, rules)
        assert report.ok
        assert [v.imported for v in report.violations] == ["ax"]

    def test_a_violation_of_the_top_level_package_can_be_known_too(
        self, tmp_path: Path
    ) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "planexs")
        report = reportFor(tmp_path, {"domain/A.py": "import planexs\n"}, rules)
        assert report.ok

    def test_an_unclassified_package_fails_even_when_nothing_else_does(
        self, tmp_path: Path
    ) -> None:
        report = reportFor(tmp_path, {"mystery/A.py": "X = 1\n"})
        assert report.violations == ()
        assert not report.ok

    def test_nothing_to_report_is_ok(self, tmp_path: Path) -> None:
        assert reportFor(tmp_path, {"core/A.py": "X = 1\n"}).ok

    def test_the_violations_are_sorted_by_file_and_line(self, tmp_path: Path) -> None:
        files = {
            "domain/B.py": "import planexs.persistence\n",
            "domain/A.py": "import os\nimport planexs.persistence\n",
        }
        report = reportFor(tmp_path, files)
        assert [(v.path, v.line) for v in report.violations] == [
            ("src/planexs/domain/A.py", 2),
            ("src/planexs/domain/B.py", 1),
        ]


# --------------------------------------------------------------------------------------------
# The report as text
# --------------------------------------------------------------------------------------------


class TestTheReport:
    def test_a_new_violation_is_printed_with_its_place_and_its_reason(
        self, tmp_path: Path
    ) -> None:
        text = "\n".join(checker.formatReport(reportFor(tmp_path, DOMAIN_IMPORTS_PERSISTENCE)))
        assert "1 new violation(s)" in text
        assert "src/planexs/domain/A.py:1: planexs.domain.A -> persistence" in text
        assert "layer 'domain' may not depend on layer 'persistence'" in text
        assert "specification 12.3" in text  # do not list it: raise it

    def test_a_stale_entry_says_which_task_was_to_fix_it_and_what_to_do(
        self, tmp_path: Path
    ) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "persistence", fixed_by="P2-02")
        text = "\n".join(checker.formatReport(reportFor(tmp_path, {"core/A.py": ""}, rules)))
        assert "1 stale known violation(s)" in text
        assert "planexs.domain.A -> persistence (to be fixed by P2-02)" in text
        assert "Remove each from [[known_violations]]" in text

    def test_an_unclassified_package_is_named_with_the_way_to_fix_it(
        self, tmp_path: Path
    ) -> None:
        text = "\n".join(checker.formatReport(reportFor(tmp_path, {"mystery/A.py": ""})))
        assert "planexs.mystery" in text
        assert "Add each to [layers]" in text

    def test_an_ok_report_says_how_much_was_checked_and_how_many_are_known(
        self, tmp_path: Path
    ) -> None:
        rules = SMALL_RULES + known("planexs.domain.A", "persistence")
        (line,) = checker.formatReport(reportFor(tmp_path, DOMAIN_IMPORTS_PERSISTENCE, rules))
        assert line.startswith("check_layering: OK (1 modules in 2 packages checked")
        assert "1 known violation(s) remain" in line

    def test_a_reason_is_printed_once_per_rule_even_for_several_imports(
        self, tmp_path: Path
    ) -> None:
        source = "import planexs.persistence.A\nimport planexs.persistence.B\n"
        report = reportFor(tmp_path, {"domain/A.py": source})
        assert len(report.violations[0].reasons) == 1


# --------------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------------


def writeRules(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rules.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestCommandLine:
    def test_a_clean_tree_exits_zero_and_says_ok(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeRoot(tmp_path / "repo", {"core/A.py": "X = 1\n"})
        config = writeRules(tmp_path, SMALL_RULES)
        code, out, err = run("--root", str(root), "--config", str(config), capsys=capsys)
        assert (code, err) == (0, "")
        assert out.startswith("check_layering: OK")

    def test_a_new_violation_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeRoot(tmp_path / "repo", DOMAIN_IMPORTS_PERSISTENCE)
        config = writeRules(tmp_path, SMALL_RULES)
        code, out, _ = run("--root", str(root), "--config", str(config), capsys=capsys)
        assert code == 1
        assert "src/planexs/domain/A.py:1: planexs.domain.A -> persistence" in out

    def test_a_stale_entry_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeRoot(tmp_path / "repo", {"core/A.py": ""})
        config = writeRules(tmp_path, SMALL_RULES + known("planexs.domain.A", "persistence"))
        code, out, _ = run("--root", str(root), "--config", str(config), capsys=capsys)
        assert code == 1
        assert "stale" in out

    def test_an_unclassified_package_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeRoot(tmp_path / "repo", {"mystery/A.py": ""})
        config = writeRules(tmp_path, SMALL_RULES)
        code, out, _ = run("--root", str(root), "--config", str(config), capsys=capsys)
        assert code == 1
        assert "planexs.mystery" in out

    def test_an_invalid_rules_file_exits_two_with_a_message_on_standard_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeRoot(tmp_path / "repo", {"core/A.py": ""})
        config = writeRules(tmp_path, '[layers]\na = ["b"]\nb = ["a"]\n')
        code, out, err = run("--root", str(root), "--config", str(config), capsys=capsys)
        assert (code, out) == (2, "")
        assert err.startswith("check_layering: error:")
        assert "cycle" in err

    def test_a_missing_rules_file_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeRoot(tmp_path / "repo", {"core/A.py": ""})
        code, _, err = run(
            "--root", str(root), "--config", str(tmp_path / "none.toml"), capsys=capsys
        )
        assert code == 2
        assert "cannot read the rules file" in err

    def test_a_missing_source_directory_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = writeRules(tmp_path, SMALL_RULES)
        code, _, err = run(
            "--root", str(tmp_path / "empty"), "--config", str(config), capsys=capsys
        )
        assert code == 2
        assert "is not a directory" in err

    def test_a_file_that_does_not_parse_exits_two_and_names_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeRoot(tmp_path / "repo", {"core/Broken.py": "def (:\n"})
        config = writeRules(tmp_path, SMALL_RULES)
        code, _, err = run("--root", str(root), "--config", str(config), capsys=capsys)
        assert code == 2
        assert "cannot parse src/planexs/core/Broken.py" in err

    def test_a_file_that_is_not_utf8_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeRoot(tmp_path / "repo", {"core/A.py": ""})
        (root / "src" / "planexs" / "core" / "Latin.py").write_bytes(b"# \xe9\n")
        config = writeRules(tmp_path, SMALL_RULES)
        code, _, err = run("--root", str(root), "--config", str(config), capsys=capsys)
        assert code == 2
        assert "cannot parse" in err

    def test_the_default_rules_file_is_the_one_next_to_the_script(self) -> None:
        assert checker.DEFAULT_CONFIG == RULES_FILE
        assert checker.DEFAULT_ROOT == REPO_ROOT

    def test_help_documents_the_options(self) -> None:
        helped = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert helped.returncode == 0
        for option in ("--root", "--config"):
            assert option in helped.stdout

    def test_the_script_runs_as_a_program_on_the_repository_and_passes(self) -> None:
        # The first Done-when item of the task, as CI runs it: no arguments, exit status 0.
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.startswith("check_layering: OK")
        assert result.stderr == ""

    def test_the_script_runs_as_a_program_with_the_documented_exit_statuses(
        self, tmp_path: Path
    ) -> None:
        config = writeRules(tmp_path, SMALL_RULES)
        failing_root = makeRoot(tmp_path / "failing", DOMAIN_IMPORTS_PERSISTENCE)
        invalid = tmp_path / "invalid.toml"
        invalid.write_text("[layers\n", encoding="utf-8")
        command = [sys.executable, str(SCRIPT), "--root"]
        failing = subprocess.run(
            [*command, str(failing_root), "--config", str(config)],
            capture_output=True,
            text=True,
            check=False,
        )
        broken = subprocess.run(
            [*command, str(failing_root), "--config", str(invalid)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert failing.returncode == 1
        assert broken.returncode == 2

    def test_a_reader_that_closes_the_pipe_early_does_not_get_a_traceback(
        self, tmp_path: Path
    ) -> None:
        files = {
            f"domain/M{number}.py": "import planexs.persistence\n" for number in range(40)
        }
        root = makeRoot(tmp_path / "repo", files)
        config = writeRules(tmp_path, SMALL_RULES)
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--root", str(root), "--config", str(config)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdout.close()  # the reader goes away before the script writes anything

        errors = process.stderr.read()
        process.wait()

        assert "Traceback" not in errors
        assert "BrokenPipeError" not in errors


# --------------------------------------------------------------------------------------------
# CI wiring
# --------------------------------------------------------------------------------------------


class TestContinuousIntegration:
    @staticmethod
    def stepsOf(job: str) -> list[str]:
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        return [str(step.get("run", "")) for step in workflow["jobs"][job]["steps"]]

    def test_the_lint_job_runs_the_layering_check(self) -> None:
        assert any("scripts/check_layering.py" in run for run in self.stepsOf("lint"))

    def test_it_runs_with_the_lint_tools_only(self) -> None:
        # The lint job installs the dev group and not the project: the script must be run
        # by the interpreter of that environment, with nothing that needs planexs or torch.
        (step,) = [run for run in self.stepsOf("lint") if "check_layering.py" in run]
        assert step.startswith("uv run --no-sync python scripts/check_layering.py")

    def test_it_runs_after_the_installation_of_the_tools(self) -> None:
        steps = self.stepsOf("lint")
        install = next(i for i, run in enumerate(steps) if "uv sync" in run)
        layering = next(i for i, run in enumerate(steps) if "check_layering.py" in run)
        assert install < layering

    def test_the_script_needs_nothing_outside_the_standard_library(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
        }
        assert imported <= set(sys.stdlib_module_names), imported - set(
            sys.stdlib_module_names
        )
        assert "planexs" not in imported
