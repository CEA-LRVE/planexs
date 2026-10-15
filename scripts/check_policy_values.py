#!/usr/bin/env python3
"""CI helper: finds literal policy values in library code (specification 3.4 and 10.7).

A *policy value* is a number that changes results, performance or behaviour and that a caller
could reasonably want to change: a tolerance, a budget, a sample count, a restart count, an
iteration cap, a seed-derivation constant. Specification 3.4 says every such value lives in a
typed configuration schema, never as a literal in library code. This script is the automated
part of that rule. It reads the Python files under `src/planexs` with the `ast` module
(standard library only) and reports every numeric literal other than 0, 1, -1 and 2.

What counts
    * Any spelling of a value: `0.0`, `1.0`, `2.0`, `1e0` and `0x2` are the values 0, 1, 2, 1
      and 2, so they are allowed. `1e-9`, `0.5` and `0x10` are not.
    * A minus sign in front of a number makes one signed value: `-1` and `-1.0` are allowed,
      `-2`, `-3` and `-0.5` are not (they are typically tensor axis indices, see below).
    * A binary minus is not a sign: in `a - 3` the value is 3.
    * Booleans are not numbers here. Strings, bytes and `None` are not numbers either.

Exemptions (decision D-5, proposed default)
    * A file whose single public symbol is a configuration schema is not scanned: its only
      top-level public `class`/`def`/`async def` (name not starting with `_`) is a class whose
      name ends with `Config`. Defaults legitimately live there (specification 10.7). This is
      the same definition of "single public symbol" as `check_naming_convention.py`.
    * A literal is exempt when its own line carries a comment `# structural: <reason>` with a
      non-empty reason. The marker is for mathematical definitions only: an axis index such as
      `-2`, the midpoint of the unit cube. Using it on a tunable value is a defect that
      review must catch. A marker without a reason exempts nothing, and the report says so.
    * Tests and scripts are never scanned: only `src/planexs`.

Position
    Each violation says where it sits: `module level` (module or class body), `signature
    default` (the default value of a parameter, which also names the parameter) or `function
    body` (including lambdas and nested functions). It also names the enclosing scope, for
    example `SearchSpaceEncoder.roundingNeighbors`.

Baseline (the ratchet)
    `--baseline FILE` compares the violations with a list of accepted ones. The check fails on
    a violation that the file does not list (a new violation) and on a listed entry that no
    longer matches any violation (a stale entry), so entries are only ever removed. The file
    has one entry per line, `<path> :: <scope> :: <position> :: <value>`; blank lines and lines
    starting with `#` are ignored. An entry carries no line number on purpose: editing code
    above a literal must not turn an accepted entry into a new violation plus a stale one.
    Entries are compared as a multiset, so two identical literals in one function need two
    lines.

    `--write-baseline` (with `--baseline FILE`) writes the current violations to that file.
    If the file does not exist it is created. If it exists, the write may only remove
    entries: a write that would add one is not done, unless `--allow-additions` is given too.
    Use that flag for a deliberate re-baseline, for example after a file move or a rename,
    and look at the diff of the baseline file.

What this script cannot see
    It finds numeric literals and nothing else. It cannot see strings (a default component
    name, a device name), booleans (a flag that selects a behaviour), or a rule hidden behind
    an allowed number: `len(values) > 2` standing for "more than two values means ordered",
    `MIN_TRIALS_FOR_MODEL = 2`, `DEFAULT_ALPHA_WHEN_FORCED = 0.0`. Those are policy values
    too. Review is the control for them, and for every use of the `# structural:` marker.

Usage
    python scripts/check_policy_values.py [--root DIR]
    python scripts/check_policy_values.py --baseline FILE
    python scripts/check_policy_values.py --baseline FILE --write-baseline [--allow-additions]

Exit status: 0 if there is nothing to report, 1 if there are violations (without a baseline),
or new violations or stale entries (with one), or if a baseline write would add entries without
`--allow-additions`, 2 on a usage or input error (a file that does not parse, a baseline file
that does not exist, a missing source directory).
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import re
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

#: The values that are never reported, whatever their spelling.
ALLOWED_VALUES: tuple[int, ...] = (0, 1, -1, 2)

#: A class whose name ends with this suffix, alone in its file, marks a configuration schema.
CONFIG_CLASS_SUFFIX = "Config"

#: The scanned directory, relative to the root.
SOURCE_DIRECTORY = Path("src") / "planexs"

#: The repository root, when this file sits in `scripts/` at the top of it.
DEFAULT_ROOT = Path(__file__).resolve().parents[1]

#: `# structural: <reason>` inside a comment token. The marker must follow a `#`, so a comment
#: that merely mentions the word (`# not structural: ...`) is not a marker.
_MARKER = re.compile(r"(?:^|#)\s*structural:(?P<reason>.*)$")

_BASELINE_HEADER = (
    "# Policy-value baseline (specification 10.7): the literal policy values that the\n"
    "# library still contains, accepted for now.\n"
    "#\n"
    "# Entries are only ever removed. Delete the line of an entry when the literal it lists\n"
    "# is gone: a stale entry fails the check, and so does a violation that is not listed.\n"
    "# Do not add an entry to silence the check: move the value into a configuration schema.\n"
    "# This file must be empty before the first release of the new architecture.\n"
    "#\n"
    "# One entry per line: <path> :: <scope> :: <position> :: <value>. Regenerate with\n"
    "# `python scripts/check_policy_values.py --baseline <this file> --write-baseline`.\n"
)

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


class PolicyValueCheckError(Exception):
    """An input the checker cannot work with (a file that does not parse, a missing path)."""


@dataclass(frozen=True)
class Violation:
    """One numeric literal that is neither allowed nor exempt.

    Attributes:
        path: The file, relative to the root, with forward slashes.
        line: The 1-based line of the literal.
        column: The 1-based column of the literal (of its sign, for a signed number).
        spelling: The literal as written in the source, without inner whitespace.
        value: The normalized value (`repr` of the number), the same for every spelling of it.
        position: `module level`, `signature default` or `function body`.
        scope: The enclosing function or class, dotted, or `<module>`. For a signature default
            it is the function that owns the signature.
        parameter: For a signature default, the name of the parameter. `None` otherwise.
        marker_without_reason: Whether the line carries `# structural:` with no reason.
    """

    path: str
    line: int
    column: int
    spelling: str
    value: str
    position: str
    scope: str
    parameter: str | None = None
    marker_without_reason: bool = False

    @property
    def key(self) -> str:
        """The baseline entry of this violation (no line number, see the module docstring)."""
        return f"{self.path} :: {self.scope} :: {self.position} :: {self.value}"

    def describe(self) -> str:
        """Renders the violation for the report: file, line, value and position.

        Returns:
            A one-line description, `path:line:column: value [position; scope]`.
        """
        parameter = f", parameter {self.parameter}" if self.parameter is not None else ""
        hint = (
            " (the `# structural:` marker on this line has no reason, so it exempts nothing)"
            if self.marker_without_reason
            else ""
        )
        return (
            f"{self.path}:{self.line}:{self.column}: {self.spelling} "
            f"[{self.position}; {self.scope}{parameter}]{hint}"
        )


@dataclass(frozen=True)
class BaselineComparison:
    """The result of comparing the violations with a baseline.

    Attributes:
        new: Violations that the baseline does not list (or lists fewer times).
        stale: Baseline entries that no violation matches (one item per missing match).
    """

    new: list[Violation]
    stale: list[str]


def isConfigSchemaModule(tree: ast.Module) -> bool:
    """Tells whether a module is a configuration-schema file, which is not scanned.

    Args:
        tree: The parsed module.

    Returns:
        `True` if the module has exactly one top-level public `class`/`def`/`async def` (name
        not starting with `_`) and it is a class whose name ends with `Config`.
    """
    public = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, *_FUNCTION_NODES)) and not node.name.startswith("_")
    ]
    return (
        len(public) == 1
        and isinstance(public[0], ast.ClassDef)
        and public[0].name.endswith(CONFIG_CLASS_SUFFIX)
    )


def findMarkerLines(source: str) -> tuple[set[int], set[int]]:
    """Finds the lines that carry a `# structural:` marker.

    Only real comments count: the text `# structural: x` inside a string or a docstring is not
    a marker, which is why this reads tokens and not lines.

    Args:
        source: The source code of a module that parses.

    Returns:
        `(lines_with_a_reason, lines_without_a_reason)`, as 1-based line numbers.
    """
    with_reason: set[int] = set()
    without_reason: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = _MARKER.search(token.string)
        if match is None:
            continue
        target = with_reason if match.group("reason").strip() else without_reason
        target.add(token.start[0])
    return with_reason, without_reason


def _isNumber(value: object) -> TypeGuard[int | float | complex]:
    """Whether a constant is a number here: an int, float or complex, never a bool."""
    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)


class _Collector(ast.NodeVisitor):
    """Walks a module and records the numeric literals that are neither allowed nor exempt."""

    def __init__(
        self, path: str, source: str, marker_lines: set[int], reasonless_lines: set[int]
    ) -> None:
        self.violations: list[Violation] = []
        self._path = path
        self._source = source
        self._marker_lines = marker_lines
        self._reasonless_lines = reasonless_lines
        self._scope: list[str] = []
        self._function_depth = 0
        # Set to the parameter name while a default value is being visited.
        self._default_of: str | None = None

    def _position(self) -> tuple[str, str | None]:
        if self._default_of is not None:
            return "signature default", self._default_of
        if self._function_depth > 0:
            return "function body", None
        return "module level", None

    def _check(self, node: ast.expr, value: complex) -> None:
        if value in ALLOWED_VALUES or node.lineno in self._marker_lines:
            return
        segment = ast.get_source_segment(self._source, node)
        spelling = "".join(segment.split()) if segment is not None else repr(value)
        position, parameter = self._position()
        self.violations.append(
            Violation(
                path=self._path,
                line=node.lineno,
                column=node.col_offset + 1,
                spelling=spelling,
                value=repr(value),
                position=position,
                scope=".".join(self._scope) or "<module>",
                parameter=parameter,
                marker_without_reason=node.lineno in self._reasonless_lines,
            )
        )

    def visit_Constant(self, node: ast.Constant) -> None:
        if _isNumber(node.value):
            self._check(node, node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        operand = node.operand
        is_sign = isinstance(node.op, (ast.USub, ast.UAdd))
        if is_sign and isinstance(operand, ast.Constant) and _isNumber(operand.value):
            self._check(
                node, -operand.value if isinstance(node.op, ast.USub) else operand.value
            )
            return
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for outer in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(outer)
        self._scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visitFunction(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visitFunction(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._scope.append("<lambda>")
        self._visitSignatureDefaults(node.args)
        self._visitBody([node.body])
        self._scope.pop()

    def _visitFunction(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Decorators and annotations are evaluated where the function is defined.
        for decorator in node.decorator_list:
            self.visit(decorator)
        arguments = node.args
        every_argument = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *([arguments.vararg] if arguments.vararg else []),
            *([arguments.kwarg] if arguments.kwarg else []),
        ]
        for argument in every_argument:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)

        self._scope.append(node.name)
        self._visitSignatureDefaults(arguments)
        self._visitBody(node.body)
        self._scope.pop()

    def _visitSignatureDefaults(self, arguments: ast.arguments) -> None:
        positional = [*arguments.posonlyargs, *arguments.args]
        with_default = positional[len(positional) - len(arguments.defaults) :]
        pairs = list(zip(with_default, arguments.defaults, strict=True))
        pairs += [
            (argument, default)
            for argument, default in zip(
                arguments.kwonlyargs, arguments.kw_defaults, strict=True
            )
            if default is not None
        ]
        saved = self._default_of
        for argument, default in pairs:
            self._default_of = argument.arg
            self.visit(default)
        self._default_of = saved

    def _visitBody(self, body: list[ast.stmt] | list[ast.expr]) -> None:
        saved = self._default_of
        self._default_of = None
        self._function_depth += 1
        for node in body:
            self.visit(node)
        self._function_depth -= 1
        self._default_of = saved


def findViolations(source: str, path: str) -> list[Violation]:
    """Finds the policy-value violations of one module.

    Args:
        source: The source code.
        path: The name to report the module under (relative to the root, forward slashes).

    Returns:
        The violations in file order. Empty for a configuration-schema file.

    Raises:
        SyntaxError: if the source does not parse.
    """
    tree = ast.parse(source, filename=path)
    if isConfigSchemaModule(tree):
        return []
    with_reason, without_reason = findMarkerLines(source)
    collector = _Collector(path, source, with_reason, without_reason)
    collector.visit(tree)
    return sorted(
        collector.violations, key=lambda violation: (violation.line, violation.column)
    )


def scanSourceTree(root: Path) -> list[Violation]:
    """Scans every Python file under `<root>/src/planexs`.

    Args:
        root: The repository root.

    Returns:
        The violations, ordered by file then position, with paths relative to `root`.

    Raises:
        PolicyValueCheckError: if the source directory is missing or a file does not parse.
    """
    source_directory = root / SOURCE_DIRECTORY
    if not source_directory.is_dir():
        raise PolicyValueCheckError(f"the source directory {source_directory} does not exist")
    violations: list[Violation] = []
    for file in sorted(source_directory.rglob("*.py")):
        relative = file.relative_to(root).as_posix()
        try:
            violations.extend(findViolations(file.read_text(encoding="utf-8"), relative))
        except SyntaxError as error:
            raise PolicyValueCheckError(
                f"{relative}: cannot parse the file: {error}"
            ) from error
    return violations


def readBaseline(path: Path) -> Counter[str]:
    """Reads a baseline file.

    Args:
        path: The baseline file.

    Returns:
        The entries with their multiplicity. Blank lines and `#` comments are ignored.

    Raises:
        PolicyValueCheckError: if the file does not exist.
    """
    if not path.is_file():
        raise PolicyValueCheckError(
            f"the baseline file {path} does not exist; create it with --write-baseline"
        )
    entries: Counter[str] = Counter()
    for raw in path.read_text(encoding="utf-8").split("\n"):
        line = raw.strip()
        if line and not line.startswith("#"):
            entries[line] += 1
    return entries


def formatBaseline(violations: list[Violation]) -> str:
    """Renders violations as the text of a baseline file.

    Args:
        violations: The violations to accept.

    Returns:
        The header comment, then one sorted entry per violation.
    """
    entries = sorted(violation.key for violation in violations)
    return _BASELINE_HEADER + "".join(f"{entry}\n" for entry in entries)


def compareToBaseline(
    violations: list[Violation], baseline: Counter[str]
) -> BaselineComparison:
    """Compares violations with a baseline, as a multiset.

    Args:
        violations: The violations found now.
        baseline: The accepted entries, with multiplicity.

    Returns:
        The violations beyond what the baseline allows (the later occurrences of an entry
        when there are more of them than listed) and the baseline entries that no violation
        matches.
    """
    grouped: dict[str, list[Violation]] = {}
    for violation in violations:
        grouped.setdefault(violation.key, []).append(violation)
    new: list[Violation] = []
    for key, found in grouped.items():
        new.extend(found[baseline.get(key, 0) :])
    stale: list[str] = []
    for key, accepted in sorted(baseline.items()):
        stale.extend([key] * (accepted - len(grouped.get(key, []))))
    return BaselineComparison(new=new, stale=stale)


def _printViolations(violations: list[Violation]) -> None:
    for violation in violations:
        print(f"  {violation.describe()}")


def _fileCount(violations: list[Violation]) -> int:
    return len({violation.path for violation in violations})


def _writeBaseline(path: Path, violations: list[Violation], allow_additions: bool) -> int:
    previous = readBaseline(path) if path.is_file() else None
    written = Counter(violation.key for violation in violations)
    added = sum((written - previous).values()) if previous is not None else 0
    removed = sum((previous - written).values()) if previous is not None else 0

    if previous is not None and added and not allow_additions:
        print(
            f"check_policy_values: baseline not written: {added} "
            f"{'entry' if added == 1 else 'entries'} would be added to {path}, and a baseline "
            "is meant to only shrink (specification 10.7). Would be added:",
            file=sys.stderr,
        )
        for violation in compareToBaseline(violations, previous).new:
            print(f"  {violation.describe()}", file=sys.stderr)
        print(
            "  To accept them on purpose (a file move, a rename), run again with "
            "--allow-additions and look at the diff.",
            file=sys.stderr,
        )
        return 1

    path.write_text(formatBaseline(violations), encoding="utf-8")
    print(
        f"check_policy_values: wrote {len(violations)} entries to {path}"
        + (f" ({added} added, {removed} removed)" if previous is not None else "")
    )
    if added:
        print(
            f"check_policy_values: note: {added} "
            f"{'entry was' if added == 1 else 'entries were'} added to an existing baseline "
            "(--allow-additions). A baseline is meant to only shrink (specification 10.7): "
            "worth a look at the diff.",
            file=sys.stderr,
        )
    return 0


def _checkAgainstBaseline(path: Path, violations: list[Violation]) -> int:
    comparison = compareToBaseline(violations, readBaseline(path))
    if not comparison.new and not comparison.stale:
        print(
            f"check_policy_values: OK ({len(violations)} violation(s) in "
            f"{_fileCount(violations)} file(s), all listed in {path})."
        )
        return 0
    if comparison.new:
        print(f"check_policy_values: {len(comparison.new)} violation(s) not in the baseline:")
        _printViolations(comparison.new)
        print(
            "  Fix: move the value into a configuration schema (specification 3.4). If it "
            "is a mathematical definition, mark its line `# structural: <reason>`."
        )
    if comparison.stale:
        print(
            f"check_policy_values: {len(comparison.stale)} stale baseline entr"
            f"{'y' if len(comparison.stale) == 1 else 'ies'} (no matching violation any more):"
        )
        for entry in comparison.stale:
            print(f"  {entry}")
        print(f"  Fix: delete these lines from {path}; entries are only ever removed.")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Runs the checker.

    Args:
        argv: The command line without the program name; `sys.argv[1:]` when omitted.

    Returns:
        The exit status described in the module docstring.
    """
    parser = argparse.ArgumentParser(
        prog="check_policy_values",
        description="Reports literal policy values (numeric literals other than 0, 1, -1 "
        "and 2) in src/planexs. See the module docstring for the rules.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="repository root; src/planexs under it is scanned and paths are reported "
        "relative to it (default: the root of this checkout)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        metavar="FILE",
        help="accepted violations: fail on a violation not listed in FILE and on a listed "
        "entry that no longer matches",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="write the current violations to the --baseline file instead of checking them",
    )
    parser.add_argument(
        "--allow-additions",
        action="store_true",
        help="with --write-baseline: also accept entries that the existing baseline does not "
        "list (not done otherwise: a baseline is meant to only shrink)",
    )
    arguments = parser.parse_args(argv)
    if arguments.write_baseline and arguments.baseline is None:
        parser.error("--write-baseline needs --baseline FILE")
    if arguments.allow_additions and not arguments.write_baseline:
        parser.error("--allow-additions only makes sense together with --write-baseline")

    try:
        violations = scanSourceTree(arguments.root)
        if arguments.baseline is None:
            return _reportWithoutBaseline(violations)
        if arguments.write_baseline:
            return _writeBaseline(arguments.baseline, violations, arguments.allow_additions)
        return _checkAgainstBaseline(arguments.baseline, violations)
    except PolicyValueCheckError as error:
        print(f"check_policy_values: error: {error}", file=sys.stderr)
        return 2


def _reportWithoutBaseline(violations: list[Violation]) -> int:
    if not violations:
        print("check_policy_values: OK (no literal policy value found).")
        return 0
    files = _fileCount(violations)
    print(f"check_policy_values: {len(violations)} violation(s) in {files} file(s):")
    _printViolations(violations)
    return 1


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
