#!/usr/bin/env python3
"""CI helper: checks the dependency direction between the layers of planexs (spec 5.1, 10.5).

Specification 5.1 gives every layer a list of the layers it may depend on, and says that
"a lower layer never imports a higher one". Specification 10.5 adds that `core` and `domain`
import no machine-learning library and that only `adapters` import Ax. This script is the
automated part of those rules. It reads the Python files under `src/planexs` with the `ast`
module (standard library only, so the dev-only environment of the CI `lint` job is enough)
and compares every `import` statement with the rules in `scripts/layers.toml`.

What it checks
    * **Layer direction.** A layer imports only the layers that its entry in `[layers]`
      lists, which is the table of specification 5.1. Importing a package of the same layer
      is always fine. A package that is in no table (neither a layer nor legacy) cannot be
      judged, so its existence is reported.
    * **Legacy.** The `[legacy]` packages (`api`, `backends`, `models`, `analysis`) are the
      code of the 0.1.0 architecture, kept until Phase 11 (roadmap section 2.3). They may
      import anything. Nothing outside the legacy group, the top-level package and the tests
      may import them, so that the new architecture never grows a dependency on code that
      will be deleted.
    * **The top-level package.** `planexs/__init__.py` re-exports the public API, and other
      modules directly under `src/planexs` (such as `logging_config.py`) are not layers. They
      may import anything, and no layer may import them: `import planexs` or
      `from planexs import BayesianOptimizer` inside a layer loads the whole library. Naming
      a layer, as in `from planexs import core`, is not that.
    * **Libraries.** The `[[library_rules]]` say where a third-party library may be
      imported: `ax`, `torch`, `botorch` and `gpytorch` never in `core` and `domain` (so
      those layers run without any engine installed), and `ax` only in `adapters` (and in
      the legacy `backends` until it is deleted).

What counts as an import
    Every `import` and `from ... import` statement of the file, at any depth: inside a
    function, under `if TYPE_CHECKING:` or in a `try` block. A dependency that only exists
    for the type checker is still a dependency of the layer. Relative imports are resolved.
    In `from planexs import name`, `name` is read as a layer or package if it is one, and as
    a symbol re-exported by the top-level package if not.

Known violations (the ratchet)
    The code that predates the new architecture breaks some of these rules, and fixing each
    one is a task of the roadmap. They are listed in `[[known_violations]]` with the task
    that fixes them, so that the check passes today and fails on anything new. A violation
    is identified by the importing module and what it imports (a package, or a library):
    `planexs.domain.X -> persistence`. It carries no line number on purpose, so that editing
    code above an import does not turn an accepted entry into a new violation plus a stale
    one. A second import from the same module to the same package is covered by the same
    entry.

    The check fails on a violation that no entry lists (a new violation) and on an entry
    that no longer matches any violation (a stale entry), so entries are only ever removed.
    Do not add an entry to silence a new violation: a request that needs one conflicts with
    the layering and must be raised, not worked around (specification 12.3).

What this script cannot see
    It reads import statements and nothing else. It cannot see an import made by name at
    run time (`importlib.import_module`, `__import__`), by plugin discovery (entry points
    are the one sanctioned way to load an adapter), or by `exec`. It does not see a
    dependency that goes through a string, such as a registry name in a configuration. It
    does not judge whether an import is necessary, nor whether a layer's own code has the
    right shape. Review is the control for those.

Usage
    python scripts/check_layering.py [--root DIR] [--config FILE]

`--root` is the repository root, where `src/planexs` is read and the paths of the report
are relative to (default: the root of this checkout). `--config` is the rules file
(default: `layers.toml` next to this script).

Exit status: 0 if there is nothing to report, 1 if there is a new violation, a stale known
violation or an unclassified package, 2 on a usage or input error (a rules file that is
missing or invalid, a missing source directory, a source file that does not parse).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

#: The name of the package that is checked.
PACKAGE_NAME = "planexs"

#: The checked directory, relative to the root.
SOURCE_DIRECTORY = Path("src") / PACKAGE_NAME

#: The repository root, when this file sits in `scripts/` at the top of it.
DEFAULT_ROOT = Path(__file__).resolve().parents[1]

#: The rules file that is read when `--config` is not given.
DEFAULT_CONFIG = Path(__file__).resolve().with_name("layers.toml")

#: How the report names a module that sits directly under `src/planexs`.
TOP_LEVEL_UNIT = "the top-level package"

#: The keys that the rules file may have, at its top level and in each of its tables.
_TOP_KEYS = frozenset({"layers", "legacy", "library_rules", "known_violations"})
_LEGACY_KEYS = frozenset({"packages"})
_LIBRARY_RULE_KEYS = frozenset({"libraries", "forbidden_in", "allowed_only_in", "reason"})
_KNOWN_KEYS = frozenset({"importer", "imported", "fixed_by", "reason"})


class LayeringCheckError(Exception):
    """The check cannot run: the rules or the source tree are not usable (exit status 2)."""


@dataclass(frozen=True)
class LibraryRule:
    """Where some third-party libraries may and may not be imported.

    Attributes:
        libraries: The top-level names of the libraries, for example `torch`.
        forbidden_in: Layers or legacy packages that may not import any of them.
        allowed_only_in: If not `None`, the only layers or legacy packages that may import
            them (the top-level package is never one of them).
        reason: Why, in a phrase, for the report.
    """

    libraries: frozenset[str]
    forbidden_in: frozenset[str]
    allowed_only_in: frozenset[str] | None
    reason: str


@dataclass(frozen=True)
class KnownViolation:
    """A violation that is accepted until the roadmap task that fixes it.

    Attributes:
        importer: The dotted name of the importing module.
        imported: What it imports: a package of planexs, a library, or `planexs` itself for
            the top-level package.
        fixed_by: The roadmap task that removes it, for example `P2-02`.
        reason: What the violation is, in a phrase.
    """

    importer: str
    imported: str
    fixed_by: str
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        """What identifies the violation: the importing module and what it imports."""
        return (self.importer, self.imported)


@dataclass(frozen=True)
class Rules:
    """The contents of the rules file.

    Attributes:
        layers: Each layer, with the layers it may depend on.
        legacy: The legacy packages.
        library_rules: The rules about third-party libraries.
        known_violations: The accepted violations.
    """

    layers: Mapping[str, frozenset[str]]
    legacy: frozenset[str]
    library_rules: tuple[LibraryRule, ...]
    known_violations: tuple[KnownViolation, ...]


@dataclass(frozen=True)
class Target:
    """What an import statement reaches, once classified.

    Attributes:
        kind: `unit` for a package of planexs (a layer, a legacy package or an unknown one),
            `top` for the top-level package or a module directly under it, `library` for
            anything outside planexs.
        name: The package for a `unit`, `planexs` for `top`, the top-level library name
            (`torch` for `torch.nn.functional`) for a `library`.
    """

    kind: Literal["unit", "top", "library"]
    name: str


@dataclass(frozen=True)
class Violation:
    """A module that imports what the rules do not allow.

    Attributes:
        importer: The dotted name of the importing module.
        path: The file, relative to the repository root, with `/` separators.
        line: The line of the first import of this kind.
        imported: What it imports, as in `KnownViolation.imported`.
        reasons: Which rules the import breaks.
    """

    importer: str
    path: str
    line: int
    imported: str
    reasons: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        """What identifies the violation: the importing module and what it imports."""
        return (self.importer, self.imported)

    def describe(self) -> str:
        """The violation as lines of a report."""
        head = f"  {self.path}:{self.line}: {self.importer} -> {self.imported}"
        return "\n".join([head, *(f"      {reason}" for reason in self.reasons)])


@dataclass(frozen=True)
class Report:
    """Everything the check found.

    Attributes:
        violations: Every violation that exists, whether a known one or not.
        new: The violations that no known entry lists.
        stale: The known entries that match no violation.
        unclassified: The packages under `src/planexs` that are in no table of the rules.
        module_count: How many modules were read.
        package_count: How many packages (directories) they sit in.
    """

    violations: tuple[Violation, ...]
    new: tuple[Violation, ...]
    stale: tuple[KnownViolation, ...]
    unclassified: tuple[str, ...]
    module_count: int
    package_count: int

    @property
    def ok(self) -> bool:
        """Whether there is nothing to report."""
        return not (self.new or self.stale or self.unclassified)


# --------------------------------------------------------------------------------------------
# The rules file
# --------------------------------------------------------------------------------------------


def _strings(value: Any, where: str, *, allow_empty: bool = True) -> list[str]:
    """Checks that a rules-file value is a list of non-blank strings.

    Args:
        value: The value read from the file.
        where: Where it is, for the error message.
        allow_empty: Whether an empty list is accepted.

    Returns:
        The strings.

    Raises:
        LayeringCheckError: if it is not.
    """
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise LayeringCheckError(f"{where} must be a list of non-empty strings.")
    if not value and not allow_empty:
        raise LayeringCheckError(f"{where} must not be empty.")
    if len(set(value)) != len(value):
        raise LayeringCheckError(f"{where} lists a name twice.")
    return list(value)


def _checkKeys(table: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    """Rejects a key that a table of the rules file does not know, so a typo is an error."""
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise LayeringCheckError(
            f"{where} has unknown key(s) {unknown}; the keys are {sorted(allowed)}."
        )


def _text(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LayeringCheckError(f"{where} needs a non-empty string `{key}`.")
    return value


def _findCycle(layers: Mapping[str, frozenset[str]]) -> list[str] | None:
    """A dependency cycle among the layers, or `None`.

    Args:
        layers: Each layer with the layers it may depend on.

    Returns:
        The names along a cycle, the first repeated at the end, if there is one.
    """
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(name: str) -> list[str] | None:
        state[name] = 1
        stack.append(name)
        for dependency in sorted(layers[name]):
            if state.get(dependency, 0) == 1:
                return [*stack[stack.index(dependency) :], dependency]
            if state.get(dependency, 0) == 0:
                found = visit(dependency)
                if found is not None:
                    return found
        stack.pop()
        state[name] = 2
        return None

    for name in sorted(layers):
        if state.get(name, 0) == 0:
            found = visit(name)
            if found is not None:
                return found
    return None


def parseRules(text: str, source: str = "layers.toml") -> Rules:
    """Reads and checks the rules.

    Args:
        text: The TOML text of the rules file.
        source: Its name, for the error messages.

    Returns:
        The rules.

    Raises:
        LayeringCheckError: if the text is not valid TOML or the rules are not coherent: an
            unknown key, a layer that depends on one that does not exist, on itself or in a
            cycle, a package that is both a layer and legacy, a name that a rule uses but no
            table defines, an incomplete or repeated known violation.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise LayeringCheckError(f"{source} is not valid TOML: {error}") from error
    _checkKeys(data, _TOP_KEYS, source)

    raw_layers = data.get("layers")
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise LayeringCheckError(f"{source} needs a non-empty [layers] table.")
    layers = {
        name: frozenset(_strings(allowed, f"{source}: layers.{name}"))
        for name, allowed in raw_layers.items()
    }
    for name, allowed in layers.items():
        if name in allowed:
            raise LayeringCheckError(f"{source}: layer {name!r} lists itself as a dependency.")
        missing = sorted(allowed - set(layers))
        if missing:
            raise LayeringCheckError(
                f"{source}: layer {name!r} depends on {missing}, which are not layers."
            )
    cycle = _findCycle(layers)
    if cycle is not None:
        raise LayeringCheckError(
            f"{source}: the layers depend on each other in a cycle: {' -> '.join(cycle)}."
        )

    raw_legacy = data.get("legacy", {})
    if not isinstance(raw_legacy, dict):
        raise LayeringCheckError(f"{source}: [legacy] must be a table.")
    _checkKeys(raw_legacy, _LEGACY_KEYS, f"{source}: [legacy]")
    legacy = frozenset(_strings(raw_legacy.get("packages", []), f"{source}: legacy.packages"))
    both = sorted(legacy & set(layers))
    if both:
        raise LayeringCheckError(f"{source}: {both} are both layers and legacy packages.")
    units = set(layers) | legacy

    library_rules: list[LibraryRule] = []
    for position, table in enumerate(data.get("library_rules", [])):
        where = f"{source}: library_rules[{position}]"
        if not isinstance(table, dict):
            raise LayeringCheckError(f"{where} must be a table.")
        _checkKeys(table, _LIBRARY_RULE_KEYS, where)
        forbidden = frozenset(_strings(table.get("forbidden_in", []), f"{where}.forbidden_in"))
        only = table.get("allowed_only_in")
        allowed_only = (
            frozenset(_strings(only, f"{where}.allowed_only_in")) if only is not None else None
        )
        if not forbidden and allowed_only is None:
            raise LayeringCheckError(f"{where} needs `forbidden_in` or `allowed_only_in`.")
        unknown = sorted((forbidden | (allowed_only or frozenset())) - units)
        if unknown:
            raise LayeringCheckError(
                f"{where} names {unknown}, which are neither layers nor legacy packages."
            )
        library_rules.append(
            LibraryRule(
                libraries=frozenset(
                    _strings(table.get("libraries"), f"{where}.libraries", allow_empty=False)
                ),
                forbidden_in=forbidden,
                allowed_only_in=allowed_only,
                reason=_text(table, "reason", where),
            )
        )

    known: list[KnownViolation] = []
    for position, table in enumerate(data.get("known_violations", [])):
        where = f"{source}: known_violations[{position}]"
        if not isinstance(table, dict):
            raise LayeringCheckError(f"{where} must be a table.")
        _checkKeys(table, _KNOWN_KEYS, where)
        entry = KnownViolation(
            importer=_text(table, "importer", where),
            imported=_text(table, "imported", where),
            fixed_by=_text(table, "fixed_by", where),
            reason=_text(table, "reason", where),
        )
        if any(entry.key == other.key for other in known):
            raise LayeringCheckError(
                f"{where} repeats the known violation {entry.importer} -> {entry.imported}."
            )
        known.append(entry)

    return Rules(
        layers=layers,
        legacy=legacy,
        library_rules=tuple(library_rules),
        known_violations=tuple(known),
    )


def loadRules(path: Path) -> Rules:
    """Reads the rules file `path`.

    Raises:
        LayeringCheckError: if it cannot be read, or `parseRules` rejects it.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LayeringCheckError(
            f"cannot read the rules file {path}: {error.strerror}."
        ) from error
    return parseRules(text, source=path.name)


# --------------------------------------------------------------------------------------------
# Reading imports
# --------------------------------------------------------------------------------------------


def resolveRelative(node: ast.ImportFrom, package: str) -> str | None:
    """The absolute name of the module that a `from ... import` statement reads from.

    Args:
        node: The statement.
        package: The package the importing file belongs to: relative imports start from it,
            and one dot is the package itself.

    Returns:
        The module name, or `None` if the dots go above the top-level package.
    """
    if node.level == 0:
        return node.module
    parts = package.split(".")
    kept = len(parts) - (node.level - 1)
    if kept <= 0:
        return None
    base = parts[:kept]
    return ".".join([*base, node.module] if node.module else base)


def importedNames(
    tree: ast.Module, package: str, known: frozenset[str]
) -> list[tuple[str, int]]:
    """The modules that the import statements of a file name, with their lines.

    `import a.b.c` names `a.b.c`. `from a.b import c` names `a.b`. The exception is a
    statement that reads from the top-level package itself: `from planexs import core` names
    `planexs.core` if `core` is a package or module of planexs, and `planexs` if it is not (a
    symbol that the top-level package re-exports).

    Args:
        tree: The parsed file.
        package: The package the file belongs to, for relative imports.
        known: The names of the packages and modules that exist directly under `planexs`.

    Returns:
        `(absolute dotted name, line)` for each name, in source order.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolveRelative(node, package)
            if base is None:
                continue
            if base != PACKAGE_NAME:
                found.append((base, node.lineno))
                continue
            for alias in node.names:
                name = alias.name if alias.name in known else None
                found.append((f"{PACKAGE_NAME}.{name}" if name else PACKAGE_NAME, node.lineno))
    return sorted(found, key=lambda item: item[1])


def classify(name: str, top_modules: frozenset[str]) -> Target:
    """Says what a dotted module name reaches.

    Args:
        name: An absolute dotted module name.
        top_modules: The modules that sit directly under `src/planexs`, not in a package.

    Returns:
        The target: a package of planexs, the top-level package (or a module directly under
        it), or a library.
    """
    parts = name.split(".")
    if parts[0] != PACKAGE_NAME:
        return Target("library", parts[0])
    if len(parts) == 1 or parts[1] in top_modules:
        return Target("top", PACKAGE_NAME)
    return Target("unit", parts[1])


# --------------------------------------------------------------------------------------------
# Judging imports
# --------------------------------------------------------------------------------------------


def judgeLibrary(importer_unit: str | None, library: str, rules: Rules) -> list[str]:
    """The rules about libraries that an import of `library` breaks.

    Args:
        importer_unit: The package of the importing module, or `None` for a module directly
            under `src/planexs`.
        library: The top-level name of the library.
        rules: The rules.

    Returns:
        One reason for each rule that is broken.
    """
    reasons: list[str] = []
    where = importer_unit if importer_unit is not None else TOP_LEVEL_UNIT
    for rule in rules.library_rules:
        if library not in rule.libraries:
            continue
        if importer_unit in rule.forbidden_in:
            reasons.append(f"{library!r} may not be imported in {where!r}: {rule.reason}")
        if rule.allowed_only_in is not None and importer_unit not in rule.allowed_only_in:
            allowed = ", ".join(sorted(rule.allowed_only_in))
            reasons.append(
                f"{library!r} may only be imported in {allowed}, not in {where!r}: "
                f"{rule.reason}"
            )
    return reasons


def judgePlanexs(importer_unit: str | None, target: Target, rules: Rules) -> list[str]:
    """The rules about layers that an import of planexs code breaks.

    Args:
        importer_unit: The package of the importing module, or `None` for a module directly
            under `src/planexs`.
        target: What it imports, a `unit` or `top`.
        rules: The rules.

    Returns:
        One reason for each rule that is broken.
    """
    if importer_unit not in rules.layers:
        # Not judged by the table of layers: the top-level modules (`None`) and the legacy
        # packages may import anything, and a package in no table is reported once, as
        # unclassified, rather than once per import.
        return []
    if target.kind == "top":
        return [
            f"layer {importer_unit!r} imports the top-level package, which loads every layer: "
            "import the layer that is needed"
        ]
    if target.name == importer_unit:
        return []
    if target.name in rules.legacy:
        return [
            f"{target.name!r} is a legacy package: only legacy packages, the top-level "
            "package and the tests may import it"
        ]
    if target.name not in rules.layers:
        return [f"{target.name!r} is in no table of the rules file: add it to [layers]"]
    if target.name not in rules.layers[importer_unit]:
        allowed = ", ".join(sorted(rules.layers[importer_unit])) or "no layer"
        return [
            f"layer {importer_unit!r} may not depend on layer {target.name!r} "
            f"(specification 5.1: it may depend on {allowed})"
        ]
    return []


# --------------------------------------------------------------------------------------------
# Scanning the tree
# --------------------------------------------------------------------------------------------


def _moduleName(relative: Path) -> tuple[str, str]:
    """The dotted module name of a file, and the package it belongs to.

    Args:
        relative: The file, relative to `src/planexs`.

    Returns:
        `(module, package)`: for `core/Registry.py` that is `planexs.core.Registry` and
        `planexs.core`; for `core/__init__.py`, `planexs.core` and `planexs.core`.
    """
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
        module = ".".join([PACKAGE_NAME, *parts])
        return module, module
    module = ".".join([PACKAGE_NAME, *parts])
    return module, module.rsplit(".", 1)[0]


def checkTree(source_dir: Path, rules: Rules, display_root: Path) -> Report:
    """Checks every module under `source_dir` against the rules.

    Args:
        source_dir: The directory of the top-level package, `src/planexs`.
        rules: The rules.
        display_root: The directory the paths of the report are relative to.

    Returns:
        The report.

    Raises:
        LayeringCheckError: if a source file does not parse.
    """
    files = sorted(source_dir.rglob("*.py"))
    on_disk = {
        path.relative_to(source_dir).parts[0] for path in files if path.parent != source_dir
    }
    top_modules = frozenset(
        path.stem for path in files if path.parent == source_dir and path.name != "__init__.py"
    )
    known_names = frozenset(rules.layers) | rules.legacy | frozenset(on_disk) | top_modules

    found: dict[tuple[str, str], Violation] = {}
    for path in files:
        relative = path.relative_to(source_dir)
        importer_unit = relative.parts[0] if len(relative.parts) > 1 else None
        module, package = _moduleName(relative)
        display = (
            path.relative_to(display_root).as_posix()
            if path.is_relative_to(display_root)
            else path.as_posix()
        )
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=display)
        except (SyntaxError, UnicodeDecodeError) as error:
            raise LayeringCheckError(f"cannot parse {display}: {error}") from error

        for name, line in importedNames(tree, package, known_names):
            target = classify(name, top_modules)
            if target.kind == "library":
                reasons = judgeLibrary(importer_unit, target.name, rules)
            else:
                reasons = judgePlanexs(importer_unit, target, rules)
            if not reasons:
                continue
            # One violation for each importing module and target, at the first import: the
            # reasons depend only on those two, so a later import of the same target says
            # nothing new.
            found.setdefault(
                (module, target.name),
                Violation(
                    importer=module,
                    path=display,
                    line=line,
                    imported=target.name,
                    reasons=tuple(reasons),
                ),
            )

    violations = tuple(sorted(found.values(), key=lambda v: (v.path, v.line, v.imported)))
    accepted = {entry.key for entry in rules.known_violations}
    current = {violation.key for violation in violations}
    return Report(
        violations=violations,
        new=tuple(v for v in violations if v.key not in accepted),
        stale=tuple(entry for entry in rules.known_violations if entry.key not in current),
        unclassified=tuple(sorted(on_disk - set(rules.layers) - rules.legacy)),
        module_count=len(files),
        package_count=len(on_disk) + 1,
    )


# --------------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------------


def formatReport(report: Report) -> list[str]:
    """The lines to print for a report.

    Args:
        report: The result of `checkTree`.

    Returns:
        The lines, ending with the verdict.
    """
    lines: list[str] = []
    if report.unclassified:
        lines.append(
            f"check_layering: {len(report.unclassified)} package(s) under src/{PACKAGE_NAME} "
            "are in no table of the rules file, so their imports cannot be judged:"
        )
        lines.extend(f"  {PACKAGE_NAME}.{name}" for name in report.unclassified)
        lines.append(
            "  Add each to [layers] (with the layers it may depend on) or to [legacy]."
        )
    if report.new:
        lines.append(f"check_layering: {len(report.new)} new violation(s):")
        lines.extend(violation.describe() for violation in report.new)
        lines.append(
            "  Fix the import. If the request needs it, it conflicts with the layering: raise "
            "it (specification 12.3) instead of listing it in [[known_violations]]."
        )
    if report.stale:
        lines.append(f"check_layering: {len(report.stale)} stale known violation(s):")
        lines.extend(
            f"  {entry.importer} -> {entry.imported} (to be fixed by {entry.fixed_by}) "
            "matches no violation any more"
            for entry in report.stale
        )
        lines.append("  Remove each from [[known_violations]] in the rules file: it is fixed.")
    if report.ok:
        remaining = len(report.violations)
        lines.append(
            f"check_layering: OK ({report.module_count} modules in {report.package_count} "
            f"packages checked; {remaining} known violation(s) remain, each listed in the "
            "rules file)."
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Runs the checker.

    Args:
        argv: The command line without the program name; `sys.argv[1:]` when omitted.

    Returns:
        The exit status described in the module docstring.
    """
    parser = argparse.ArgumentParser(
        prog="check_layering",
        description="Checks the dependency direction between the layers of planexs "
        "(specification 5.1, 10.5). See the module docstring for the rules.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="repository root; src/planexs under it is read and paths are reported relative "
        "to it (default: the root of this checkout)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        metavar="FILE",
        help="the rules file (default: layers.toml next to this script)",
    )
    arguments = parser.parse_args(argv)

    try:
        source_dir = arguments.root / SOURCE_DIRECTORY
        if not source_dir.is_dir():
            raise LayeringCheckError(f"{source_dir} is not a directory.")
        report = checkTree(source_dir, loadRules(arguments.config), arguments.root)
    except LayeringCheckError as error:
        print(f"check_layering: error: {error}", file=sys.stderr)
        return 2
    for line in formatReport(report):
        print(line)
    return 0 if report.ok else 1


def _runAsProgram() -> int:
    """Runs `main` for the command line, without a traceback if the output reader goes away.

    A reader that stops early (`... | head -1`) closes the pipe; the next write then raises
    `BrokenPipeError`. That is not a failure of the check, so it ends quietly with status 1.
    Standard output is pointed at the null device first, so that flushing it again when the
    interpreter exits does not raise a second error (the recipe of the `signal` documentation).
    """
    try:
        status = main()
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    return status


if __name__ == "__main__":
    raise SystemExit(_runAsProgram())
