"""The registry contract tests and the registry completeness test (specification 10.6, 10.11).

Two guarantees, each checked on test doubles and on the real tree, so that a later task that
adds a kind or a component is held to them without anyone remembering to:

**The contract.** Every entry of every registry has a valid name, a configuration schema that
derives from `ComponentConfig` with a description on every field, needs and supports that are
in the vocabulary, a callable builder and a one-line description (specification 6, 10.6).
`Registry.register` and `ComponentEntry` already refuse some of these, so this test is a second
line of defence for what they cannot see (the schema, the descriptions) and for an entry that
reached a registry some other way. It runs over every kind of the process-wide table, and
today that table holds no real kind: the test doubles below stand in until Phase 3, and the
real kinds are covered as they appear, with no change to this file.

**The completeness rule.** A decorated component is registered only when its module has been
imported, so the `__init__` of its package must import it (specification 5.12, "Population").
The check is static, like the one in `test_core_layer_rule.py`, and for the same reason:
the rule is "one import line per file", which is a statement about the source.

Tests of the two checkers themselves come first (they fail on a double that breaks each
rule, and on a decorated module that its `__init__` does not import), then the checks run on
the real tree.
"""

from __future__ import annotations

import ast
import copy
import importlib
import pkgutil
import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, create_model

import planexs
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.KindTable import KindTable
from planexs.core.Registry import Registry

SOURCE_ROOT = Path(planexs.__file__).resolve().parent


def importEveryLayer() -> None:
    """Imports each top-level package of planexs, so that every kind it defines exists.

    Only the packages are imported, not their modules one by one: that a package's `__init__`
    imports all of its components is the completeness rule, tested below. Walking every module
    here instead would hide a failure of that rule.
    """
    for module in pkgutil.iter_modules(planexs.__path__):
        if module.ispkg:
            importlib.import_module(f"planexs.{module.name}")


# Kinds are defined at import time, and the parametrization below needs them when pytest
# collects.
importEveryLayer()
REAL_TABLE = KindTable.default()
REAL_KINDS = REAL_TABLE.kinds()


# --------------------------------------------------------------------------------------------
# The contract checker
# --------------------------------------------------------------------------------------------

#: The rules of the contract, by the identifier each violation carries.
RULES = (
    "name",
    "config_schema",
    "field_descriptions",
    "needs",
    "supports",
    "builder",
    "description",
)


@dataclass(frozen=True)
class Violation:
    """One broken rule of the contract, for one entry of one kind.

    Attributes:
        rule: The identifier of the rule, one of `RULES`.
        kind: The kind whose registry holds the entry.
        name: The name the entry is registered under.
        message: What is wrong and how to fix it.
    """

    rule: str
    kind: str
    name: str
    message: str

    def __str__(self) -> str:
        """The violation as one readable line."""
        return f"[{self.rule}] {self.kind} {self.name!r}: {self.message}"


def entryViolations(
    kind: str, key: str, entry: ComponentEntry, vocabulary: CapabilityVocabulary
) -> list[Violation]:
    """The rules of the contract that one registered entry breaks.

    Args:
        kind: The kind of the registry that holds the entry.
        key: The name the entry is registered under.
        entry: The entry.
        vocabulary: The vocabulary its tags must be in.

    Returns:
        The violations, in the order of `RULES`; empty if the entry honours the contract.
    """
    found: list[Violation] = []

    def report(rule: str, message: str) -> None:
        found.append(Violation(rule, kind, key, message))

    if not Registry.isValidName(entry.name):
        report(
            "name",
            f"{entry.name!r} is not a valid component name: write lowercase snake_case, "
            "with an optional version suffix such as _v1",
        )
    if key != entry.name:
        report("name", f"registered under {key!r} but the entry's own name is {entry.name!r}")

    schema = entry.config_schema
    if not (isinstance(schema, type) and issubclass(schema, ComponentConfig)):
        report(
            "config_schema",
            f"the configuration schema {schema!r} is not a subclass of ComponentConfig, so "
            "its configuration has no 'type' field",
        )
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        for field_name, field in schema.model_fields.items():
            text = field.description
            if isinstance(text, str) and text.strip():
                continue
            hint = (
                ": a 'type' field narrowed with a bare default, as in "
                "`type: Literal['x_v1'] = 'x_v1'`, loses the description of "
                "ComponentConfig.type; write `Field(default='x_v1', description=...)`"
                if field_name == "type"
                else ": write it as `Field(..., description=...)`"
            )
            report(
                "field_descriptions",
                f"field {field_name!r} of {schema.__name__} has no description{hint}",
            )

    for rule in ("needs", "supports"):
        for tag in sorted(getattr(entry, rule), key=str):
            if tag not in vocabulary:
                report(
                    rule,
                    f"the tag {tag!r} is not in the capability vocabulary: check the "
                    "spelling, or register the tag with a description first",
                )

    if not callable(entry.builder):
        report("builder", f"the builder {entry.builder!r} is not callable")

    description = entry.description
    if not (
        isinstance(description, str)
        and description.strip()
        and description.splitlines() == [description]
    ):
        report("description", f"{description!r} is not a non-blank, one-line description")
    return found


def kindViolations(table: KindTable, kind: str) -> list[Violation]:
    """The violations of every entry of one kind of `table`.

    Args:
        table: The kind table.
        kind: A kind defined in it.

    Returns:
        The violations, entry by entry in sorted name order.
    """
    registry = table.registry(kind)
    found: list[Violation] = []
    for key in registry.names():
        found.extend(entryViolations(kind, key, registry.lookup(key), table.vocabulary))
    return found


def tableViolations(table: KindTable) -> list[Violation]:
    """The violations of every entry of every kind of `table`, kind by kind."""
    return [violation for kind in table.kinds() for violation in kindViolations(table, kind)]


# --------------------------------------------------------------------------------------------
# Test doubles: entries that honour the contract, and ways to break it
# --------------------------------------------------------------------------------------------


def describedConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real components will write theirs: every field described,
    the `type` field included.
    """
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config",
        __base__=ComponentConfig,
        type=(Literal[name], Field(default=name, description=f"The registered name, {name}.")),
        n_points=(int, Field(default=3, description="How many points, a count.")),
    )
    return schema


def noop() -> None:
    """A builder for test components."""


def compliantEntry(
    name: str = "a_v1",
    *,
    needs: tuple[str, ...] = (),
    supports: tuple[str, ...] = (),
    schema: type[BaseModel] | None = None,
) -> ComponentEntry:
    """An entry that honours every rule of the contract."""
    return ComponentEntry(
        name=name,
        builder=noop,
        config_schema=schema if schema is not None else describedConfig("a_v1"),
        description="A component for tests.",
        needs=frozenset(needs),
        supports=frozenset(supports),
    )


def corrupt(entry: ComponentEntry, **changes: Any) -> ComponentEntry:
    """A copy of `entry` with attributes replaced, past the checks of `ComponentEntry`.

    The entry is a frozen dataclass whose `__post_init__` refuses most of what these doubles
    need to model (a builder that is not callable, a blank description). A copy with the
    attributes set directly stands for an entry that was built some other way.
    """
    broken = copy.copy(entry)
    for attribute, value in changes.items():
        object.__setattr__(broken, attribute, value)
    return broken


def forceRegister(
    registry: Registry[Any], entry: ComponentEntry, key: str | None = None
) -> None:
    """Stores `entry` in `registry` with none of the checks of `Registry.register`.

    It stands for a registration that bypassed them (a registry subclass, a direct write): the
    case the contract test exists to catch. Private state is used on purpose, because the
    public method refuses exactly these entries.
    """
    registry._entries[entry.name if key is None else key] = entry


class PlainModel(BaseModel):
    """A schema that is a Pydantic model but not a `ComponentConfig`, with described fields."""

    n_points: int = Field(default=3, description="How many points, a count.")


class BlankDescriptionConfig(ComponentConfig):
    """A schema whose field has a description that is only whitespace."""

    type: Literal["blank_v1"] = Field(default="blank_v1", description="The registered name.")
    n_points: int = Field(default=3, description="   ")


class UndescribedFieldConfig(ComponentConfig):
    """A schema with one field that has no description at all."""

    type: Literal["undescribed_v1"] = Field(
        default="undescribed_v1", description="The registered name."
    )
    n_points: int = 3


class AnnotatedTypeConfig(ComponentConfig):
    """The `type` field narrowed and described in an annotation."""

    type: Annotated[Literal["annotated_v1"], Field(description="The registered name.")] = (
        "annotated_v1"
    )
    n_points: int = Field(default=3, description="How many points, a count.")


class BareTypeConfig(ComponentConfig):
    """The `type` field narrowed with a bare default, as `ComponentConfig` documents it."""

    type: Literal["bare_v1"] = "bare_v1"
    n_points: int = Field(default=3, description="How many points, a count.")


@dataclass(frozen=True)
class BreakingCase:
    """A double that breaks exactly one rule of the contract.

    Attributes:
        rule: The rule it breaks.
        label: What is wrong with it, for the test id.
        install: Puts the broken entry in a registry.
    """

    rule: str
    label: str
    install: Callable[[Registry[Any]], None]


def breaking(
    rule: str, label: str
) -> Callable[[Callable[[Registry[Any]], None]], BreakingCase]:
    """Turns a function that installs a broken entry into a `BreakingCase`."""

    def build(install: Callable[[Registry[Any]], None]) -> BreakingCase:
        return BreakingCase(rule, label, install)

    return build


@breaking("name", "uppercase name")
def _uppercaseName(registry: Registry[Any]) -> None:
    forceRegister(registry, compliantEntry("BadName"))


@breaking("name", "name that starts with a digit")
def _digitName(registry: Registry[Any]) -> None:
    forceRegister(registry, compliantEntry("1st_v1"))


@breaking("name", "key that is not the entry's own name")
def _otherKey(registry: Registry[Any]) -> None:
    forceRegister(registry, compliantEntry("a_v1"), key="b_v1")


@breaking("config_schema", "a Pydantic model that is not a ComponentConfig")
def _plainModel(registry: Registry[Any]) -> None:
    # Passes `Registry.register`, which only asks for a Pydantic model.
    registry.register(compliantEntry(schema=PlainModel))


@breaking("config_schema", "a schema that is not a class")
def _notAClass(registry: Registry[Any]) -> None:
    forceRegister(registry, corrupt(compliantEntry(), config_schema=object()))


@breaking("field_descriptions", "a field with no description")
def _undescribedField(registry: Registry[Any]) -> None:
    registry.register(compliantEntry(schema=UndescribedFieldConfig))


@breaking("field_descriptions", "a field whose description is blank")
def _blankDescription(registry: Registry[Any]) -> None:
    registry.register(compliantEntry(schema=BlankDescriptionConfig))


@breaking("field_descriptions", "a type field narrowed without a description")
def _bareType(registry: Registry[Any]) -> None:
    registry.register(compliantEntry(schema=BareTypeConfig))


@breaking("needs", "a need that is not in the vocabulary")
def _unknownNeed(registry: Registry[Any]) -> None:
    forceRegister(registry, compliantEntry(needs=("posterior:missing",)))


@breaking("needs", "a need that is not even a well-formed tag")
def _malformedNeed(registry: Registry[Any]) -> None:
    forceRegister(registry, compliantEntry(needs=("NoGroup",)))


@breaking("supports", "a support that is not in the vocabulary")
def _unknownSupport(registry: Registry[Any]) -> None:
    forceRegister(registry, compliantEntry(supports=("posterior:missing",)))


@breaking("builder", "a builder that is None")
def _noBuilder(registry: Registry[Any]) -> None:
    forceRegister(registry, corrupt(compliantEntry(), builder=None))


@breaking("builder", "a builder that is a string")
def _stringBuilder(registry: Registry[Any]) -> None:
    forceRegister(registry, corrupt(compliantEntry(), builder="build"))


@breaking("description", "an empty description")
def _emptyDescription(registry: Registry[Any]) -> None:
    forceRegister(registry, corrupt(compliantEntry(), description=""))


@breaking("description", "a blank description")
def _blankText(registry: Registry[Any]) -> None:
    forceRegister(registry, corrupt(compliantEntry(), description="   "))


@breaking("description", "a description of two lines")
def _twoLines(registry: Registry[Any]) -> None:
    forceRegister(registry, corrupt(compliantEntry(), description="One line.\nAnd another."))


@breaking("description", "a description that is not a string")
def _notText(registry: Registry[Any]) -> None:
    forceRegister(registry, corrupt(compliantEntry(), description=None))


BREAKING_CASES = [
    _uppercaseName,
    _digitName,
    _otherKey,
    _plainModel,
    _notAClass,
    _undescribedField,
    _blankDescription,
    _bareType,
    _unknownNeed,
    _malformedNeed,
    _unknownSupport,
    _noBuilder,
    _stringBuilder,
    _emptyDescription,
    _blankText,
    _twoLines,
    _notText,
]

#: Kinds of the doubles table, each with two entries that honour the contract, so that tags,
#: schemas and descriptions are all exercised by something that must pass.
DOUBLE_KINDS = ("design", "surrogate", "acquisition")


def newVocabulary() -> CapabilityVocabulary:
    """A vocabulary of its own with the tags the doubles use."""
    vocabulary = CapabilityVocabulary()
    vocabulary.register("posterior:gaussian", "The posterior is Gaussian.")
    vocabulary.register("space:categorical", "The space has categorical parameters.")
    return vocabulary


def newDoubleTable() -> KindTable:
    """A kind table of its own that holds only entries that honour the contract."""
    table = KindTable(newVocabulary())
    for kind in DOUBLE_KINDS:
        registry = table.define(kind)
        registry.register(
            ComponentEntry(
                name="first_v1",
                builder=noop,
                config_schema=describedConfig("first_v1"),
                description=f"The first {kind}.",
                supports=frozenset({"posterior:gaussian"}),
            )
        )
        registry.register(
            ComponentEntry(
                name="second_v2",
                builder=noop,
                config_schema=describedConfig("second_v2"),
                description=f"The second {kind}.",
                needs=frozenset({"posterior:gaussian"}),
                supports=frozenset({"space:categorical"}),
            )
        )
    return table


def tableWith(case: BreakingCase) -> KindTable:
    """A table of its own with one kind that holds the double of `case` and nothing else."""
    table = KindTable(newVocabulary())
    case.install(table.define("toy"))
    return table


# --------------------------------------------------------------------------------------------
# Tests of the contract
# --------------------------------------------------------------------------------------------


#: One case per kind of each table that holds entries. The process-wide table has no real kind
#: yet, so the doubles keep this list from being empty (an empty parametrization is skipped,
#: and a skipped test checks nothing). Each real kind that a later task defines joins it as
#: `("default", kind)` with no edit here.
CONTRACT_CASES = [("default", kind) for kind in REAL_KINDS] + [
    ("doubles", kind) for kind in DOUBLE_KINDS
]


class TestTheContractOverEveryKind:
    """The contract test proper."""

    @pytest.mark.parametrize(
        "source, kind",
        CONTRACT_CASES,
        ids=[f"{source}-{kind}" for source, kind in CONTRACT_CASES],
    )
    def test_every_entry_of_the_kind_honours_the_contract(
        self, source: str, kind: str
    ) -> None:
        table = REAL_TABLE if source == "default" else newDoubleTable()
        assert [str(violation) for violation in kindViolations(table, kind)] == []

    def test_the_default_table_has_no_kind_that_was_not_parametrized(self) -> None:
        # A kind defined between collection and run (a leak from another test, say) would
        # otherwise escape the parametrization above.
        assert REAL_TABLE.kinds() == REAL_KINDS

    def test_the_default_table_honours_the_contract_as_a_whole(self) -> None:
        assert [str(violation) for violation in tableViolations(REAL_TABLE)] == []

    def test_the_doubles_are_not_vacuous(self) -> None:
        table = newDoubleTable()
        tags: set[str] = set()
        for kind in DOUBLE_KINDS:
            assert table.registry(kind).names() == ["first_v1", "second_v2"]
            for name in table.registry(kind).names():
                entry = table.registry(kind).lookup(name)
                tags |= entry.needs | entry.supports
        # Tags are exercised too, not only schemas and descriptions.
        assert tags == {"posterior:gaussian", "space:categorical"}


class TestADoubleThatBreaksARuleIsReported:
    @pytest.mark.parametrize(
        "case", BREAKING_CASES, ids=[f"{case.rule}-{case.label}" for case in BREAKING_CASES]
    )
    def test_exactly_the_broken_rule_is_reported(self, case: BreakingCase) -> None:
        violations = tableViolations(tableWith(case))
        assert violations, f"{case.label}: no violation was reported"
        assert {violation.rule for violation in violations} == {case.rule}, [
            str(violation) for violation in violations
        ]

    def test_every_rule_of_the_contract_has_a_breaking_case(self) -> None:
        # Guards the card's "one case per rule": a rule added to `RULES` without a case fails.
        assert {case.rule for case in BREAKING_CASES} == set(RULES)

    def test_a_violation_names_the_kind_the_entry_and_a_way_to_fix_it(self) -> None:
        (violation,) = tableViolations(tableWith(_unknownNeed))
        text = str(violation)
        assert "toy" in text
        assert "'a_v1'" in text
        assert "'posterior:missing'" in text
        assert "register the tag" in text

    def test_the_hint_for_a_bare_type_field_says_how_to_write_it(self) -> None:
        (violation,) = tableViolations(tableWith(_bareType))
        assert "'type'" in violation.message
        assert "Field(default=" in violation.message

    def test_the_violations_of_every_kind_and_entry_are_reported_together(self) -> None:
        table = KindTable(newVocabulary())
        _uppercaseName.install(table.define("first_kind"))
        second = table.define("second_kind")
        forceRegister(second, corrupt(compliantEntry("b_v1"), builder=None))
        forceRegister(second, corrupt(compliantEntry("c_v1"), description=""))
        found = {(v.kind, v.name, v.rule) for v in tableViolations(table)}
        assert found == {
            ("first_kind", "BadName", "name"),
            ("second_kind", "b_v1", "builder"),
            ("second_kind", "c_v1", "description"),
        }

    def test_several_broken_rules_of_one_entry_are_all_reported(self) -> None:
        table = KindTable(newVocabulary())
        broken = corrupt(
            compliantEntry("BadName", needs=("nope:nope",), schema=UndescribedFieldConfig),
            builder=None,
            description="",
        )
        forceRegister(table.define("toy"), broken)
        assert [v.rule for v in tableViolations(table)] == [
            "name",
            "field_descriptions",
            "needs",
            "builder",
            "description",
        ]


class TestADoubleThatHonoursTheContractIsNotReported:
    def test_the_base_double_of_the_breaking_cases_passes(self) -> None:
        # Each breaking case differs from this entry by one change, so it is that change that
        # is reported and nothing else.
        table = KindTable(newVocabulary())
        table.define("toy").register(compliantEntry())
        assert tableViolations(table) == []

    def test_an_empty_kind_is_not_a_violation(self) -> None:
        table = KindTable(newVocabulary())
        table.define("toy")
        assert tableViolations(table) == []

    def test_an_empty_table_is_not_a_violation(self) -> None:
        assert tableViolations(KindTable(newVocabulary())) == []

    def test_tags_are_checked_against_the_vocabulary_of_the_table(self) -> None:
        # A tag that another vocabulary knows is still unknown to this table.
        other = CapabilityVocabulary()
        other.register("elsewhere:tag", "A tag of another vocabulary.")
        table = KindTable(newVocabulary())
        forceRegister(table.define("toy"), compliantEntry(needs=("elsewhere:tag",)))
        assert {violation.rule for violation in tableViolations(table)} == {"needs"}

    def test_a_description_given_in_an_annotation_is_accepted(self) -> None:
        # `Annotated[..., Field(description=...)]` is another way to describe the `type` field.
        table = KindTable(newVocabulary())
        table.define("toy").register(compliantEntry(schema=AnnotatedTypeConfig))
        assert tableViolations(table) == []


# --------------------------------------------------------------------------------------------
# The completeness checker
# --------------------------------------------------------------------------------------------

#: A decorator that registers a component: `register` followed by a capital, as in
#: `registerSurrogate` (specification 5.12, 10.2: functions are camelCase).
REGISTRATION_DECORATOR = re.compile(r"register[A-Z]\w*")


def decoratorName(decorator: ast.expr) -> str | None:
    """The name a decorator is written with: `registerX` in `@registerX`, `@registerX(...)`
    and `@module.registerX(...)`.
    """
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def registrationDecorators(source: str) -> list[str]:
    """The registration decorators that classes and functions in `source` are decorated with.

    Args:
        source: The Python source of one module.

    Returns:
        The decorator names, in the order found. A module that only *defines* a decorator, as
        `registerSurrogate.py` does, is not decorated by it and has none.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                name = decoratorName(decorator)
                if name is not None and REGISTRATION_DECORATOR.fullmatch(name):
                    found.append(name)
    return found


def isTypeCheckingGuard(test: ast.expr) -> bool:
    """Whether an `if` condition is `TYPE_CHECKING` or `typing.TYPE_CHECKING`."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def executedStatements(statements: list[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements that run when a module is imported, inside `if`, `try` and `with` too.

    Not included: the body of a function or class (an import there runs later, if ever) and the
    body of `if TYPE_CHECKING:` (it never runs; its `else` branch does).

    Args:
        statements: The statements of a block.

    Yields:
        Each statement of the block that runs on import, and those of the blocks nested in it.
    """
    for statement in statements:
        yield statement
        if isinstance(statement, ast.If):
            if not isTypeCheckingGuard(statement.test):
                yield from executedStatements(statement.body)
            yield from executedStatements(statement.orelse)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            for block in (
                statement.body,
                *(handler.body for handler in statement.handlers),
                statement.orelse,
                statement.finalbody,
            ):
                yield from executedStatements(block)
        elif isinstance(statement, ast.With):
            yield from executedStatements(statement.body)


def absoluteModule(node: ast.ImportFrom, package: str) -> str | None:
    """The absolute name of the module a `from ... import` statement reads from.

    Args:
        node: The statement.
        package: The package the importing `__init__` belongs to, which relative imports start
            from (one dot is the package itself).

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


def importedModules(source: str, package: str) -> set[str]:
    """The modules that running the `__init__` with this source imports.

    `import a.b.c` imports `a`, `a.b` and `a.b.c`. `from a.b import c` imports `a.b` and, if
    `c` is a module and not a name, `a.b.c`: both are counted, since the module name is what
    is looked for.

    Args:
        source: The Python source of the `__init__`.
        package: The dotted name of the package it initializes.

    Returns:
        The dotted names of the modules imported.
    """
    found: set[str] = set()

    def add(name: str) -> None:
        parts = name.split(".")
        found.update(".".join(parts[: count + 1]) for count in range(len(parts)))

    for statement in executedStatements(ast.parse(source).body):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                add(alias.name)
        elif isinstance(statement, ast.ImportFrom):
            base = absoluteModule(statement, package)
            if base is None:
                continue
            add(base)
            for alias in statement.names:
                add(f"{base}.{alias.name}")
    return found


@dataclass(frozen=True)
class CompletenessReport:
    """What `checkCompleteness` found in a source tree.

    Attributes:
        packages: The dotted names of the packages that were checked, sorted.
        decorated: The dotted names of the modules that hold a registration decorator, sorted.
        missing: The dotted names of what a package's `__init__` should import and does not,
            sorted: decorated modules, and subpackages that hold some.
    """

    packages: tuple[str, ...]
    decorated: tuple[str, ...]
    missing: tuple[str, ...]


def checkCompleteness(root: Path, root_package: str) -> CompletenessReport:
    """Checks that every package's `__init__` imports what registers components.

    Two things are required (specification 5.12, "Population"; 10.6):

    - a module that holds a registration decorator is imported by the `__init__` of the package
      that holds it, and a directory with no `__init__` cannot import anything;
    - a subpackage that holds such a module, directly or deeper, is imported by the `__init__`
      of its parent. The children of the top-level package are exempt: the top-level import is
      lazy (roadmap task P2-11), and a layer is imported by whoever uses it.

    Args:
        root: The directory of the top-level package, for example `src/planexs`.
        root_package: Its dotted name, for example `planexs`.

    Returns:
        The report.
    """

    def packageName(directory: Path) -> str:
        return ".".join([root_package, *directory.relative_to(root).parts])

    importsCache: dict[Path, set[str]] = {}

    def importsOf(directory: Path) -> set[str]:
        if directory not in importsCache:
            init = directory / "__init__.py"
            importsCache[directory] = (
                importedModules(init.read_text(encoding="utf-8"), packageName(directory))
                if init.is_file()
                else set()
            )
        return importsCache[directory]

    decorated: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if path.name != "__init__.py" and registrationDecorators(
            path.read_text(encoding="utf-8")
        ):
            decorated.append(path)

    missing: set[str] = set()
    holders: set[Path] = set()
    for path in decorated:
        module = f"{packageName(path.parent)}.{path.stem}"
        if module not in importsOf(path.parent):
            missing.add(module)
        for directory in path.parents:
            if directory == root:
                break
            holders.add(directory)
    for directory in holders:
        if directory.parent != root and packageName(directory) not in importsOf(
            directory.parent
        ):
            missing.add(packageName(directory))

    packages = sorted(
        {packageName(init.parent) for init in root.rglob("__init__.py")} | {root_package}
    )
    return CompletenessReport(
        packages=tuple(packages),
        decorated=tuple(sorted(f"{packageName(p.parent)}.{p.stem}" for p in decorated)),
        missing=tuple(sorted(missing)),
    )


def makeTree(root: Path, files: dict[str, str]) -> Path:
    """Writes `files` (path relative to `root` to source) and returns `root`."""
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


#: A module with a component registered by decorator.
DECORATED = (
    "from demo.core.registerThing import registerThing\n\n\n"
    "@registerThing('a_v1')\nclass A:\n    pass\n"
)

#: A module with nothing to register, like an abstract contract or a registry.
PLAIN = "class Contract:\n    pass\n"


def reportOf(tmp_path: Path, files: dict[str, str]) -> CompletenessReport:
    """The completeness report of a tree of package `demo` made of `files`."""
    return checkCompleteness(makeTree(tmp_path / "demo", files), "demo")


# --------------------------------------------------------------------------------------------
# Tests of the completeness checker
# --------------------------------------------------------------------------------------------


class TestWhichModulesAreDecorated:
    @pytest.mark.parametrize(
        "source",
        [
            "@registerThing('a_v1')\nclass A: ...\n",
            "@registerThing\nclass A: ...\n",
            "@registry.registerThing('a_v1')\nclass A: ...\n",
            "@registerThing('a_v1')\ndef build(): ...\n",
            "@registerThing('a_v1')\nasync def build(): ...\n",
            "class Outer:\n    @registerThing('a_v1')\n    class A: ...\n",
            "def late():\n    @registerThing('a_v1')\n    class A: ...\n",
            "@other\n@registerAcquisition('a_v1')\nclass A: ...\n",
            "@registerX1('a_v1')\nclass A: ...\n",
        ],
        ids=[
            "call",
            "bare",
            "attribute",
            "function",
            "async-function",
            "nested-class",
            "inside-a-function",
            "second-decorator",
            "digit-in-name",
        ],
    )
    def test_a_registration_decorator_is_found(self, source: str) -> None:
        assert registrationDecorators(source)

    @pytest.mark.parametrize(
        "source",
        [
            PLAIN,
            # The file that defines the decorator is not decorated by it.
            "def registerThing(name):\n    return lambda cls: cls\n",
            "@dataclass\nclass A: ...\n",
            "@register\nclass A: ...\n",
            "@registered('a')\nclass A: ...\n",
            "@register_thing('a')\nclass A: ...\n",
            "@thingRegister('a')\nclass A: ...\n",
            "@field_validator('x')\nclass A: ...\n",
            "registry.register(entry)\n",
            "x = registerThing('a')\n",
            "@(lambda f: f)\nclass A: ...\n",
        ],
        ids=[
            "plain",
            "the-definition",
            "other-decorator",
            "bare-register",
            "registered",
            "snake-case",
            "suffix",
            "validator",
            "method-call",
            "not-a-decorator",
            "lambda",
        ],
    )
    def test_anything_else_is_not(self, source: str) -> None:
        assert registrationDecorators(source) == []

    def test_the_names_come_in_source_order(self) -> None:
        source = "@registerB('b')\nclass B: ...\n\n@registerA('a')\nclass A: ...\n"
        assert registrationDecorators(source) == ["registerB", "registerA"]


class TestWhichModulesAnInitImports:
    PACKAGE = "demo.kinds"

    @pytest.mark.parametrize(
        "line, wanted",
        [
            ("from demo.kinds.A import A", "demo.kinds.A"),
            ("from demo.kinds import A", "demo.kinds.A"),
            ("import demo.kinds.A", "demo.kinds.A"),
            ("import demo.kinds.A as a", "demo.kinds.A"),
            ("from . import A", "demo.kinds.A"),
            ("from .A import A", "demo.kinds.A"),
            ("from .. import kinds", "demo.kinds"),
            ("from demo.kinds import (\n    A,\n    B,\n)", "demo.kinds.B"),
        ],
    )
    def test_each_way_to_import_a_module_counts(self, line: str, wanted: str) -> None:
        assert wanted in importedModules(line + "\n", self.PACKAGE)

    def test_importing_a_deep_module_imports_its_parents(self) -> None:
        found = importedModules("import demo.kinds.sub.Leaf\n", self.PACKAGE)
        assert {"demo", "demo.kinds", "demo.kinds.sub", "demo.kinds.sub.Leaf"} <= found

    def test_a_module_of_another_package_with_the_same_name_does_not_count(self) -> None:
        assert "demo.kinds.A" not in importedModules(
            "from demo.other.A import A\n", self.PACKAGE
        )
        assert "demo.kinds.A" not in importedModules("from .. import A\n", self.PACKAGE)

    @pytest.mark.parametrize(
        "package, line",
        [
            ("demo", "from ... import A"),
            ("demo.kinds", "from ... import A"),
            ("demo.a.b", "from ..... import A"),
            ("demo.a.b.c", "from ...... import A"),
        ],
    )
    def test_dots_above_the_top_level_package_name_nothing(
        self, package: str, line: str
    ) -> None:
        # More dots than levels must not wrap around to a package that is further down.
        assert importedModules(line + "\n", package) == set()

    def test_an_import_in_an_if_a_try_or_a_with_runs_on_import(self) -> None:
        source = (
            "import sys\n"
            "if sys.platform:\n    from .A import A\n"
            "try:\n    from .B import B\nexcept ImportError:\n    from .C import C\n"
            "else:\n    from .D import D\nfinally:\n    from .E import E\n"
            "with open('f'):\n    from .F import F\n"
        )
        found = importedModules(source, self.PACKAGE)
        assert {f"demo.kinds.{name}" for name in "ABCDEF"} <= found

    @pytest.mark.parametrize(
        "source",
        [
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .A import A\n",
            "import typing\nif typing.TYPE_CHECKING:\n    from .A import A\n",
        ],
        ids=["name", "attribute"],
    )
    def test_an_import_under_type_checking_never_runs(self, source: str) -> None:
        assert "demo.kinds.A" not in importedModules(source, self.PACKAGE)

    def test_the_else_branch_of_type_checking_does_run(self) -> None:
        source = "if TYPE_CHECKING:\n    pass\nelse:\n    from .A import A\n"
        assert "demo.kinds.A" in importedModules(source, self.PACKAGE)

    def test_an_import_inside_a_function_or_class_does_not_run_on_import(self) -> None:
        source = "def later():\n    from .A import A\n\nclass C:\n    from .B import B\n"
        found = importedModules(source, self.PACKAGE)
        assert "demo.kinds.A" not in found
        assert "demo.kinds.B" not in found

    def test_a_docstring_only_init_imports_nothing(self) -> None:
        assert importedModules('"""Just a docstring."""\n', self.PACKAGE) == set()


class TestACompleteTreeIsAccepted:
    @pytest.mark.parametrize(
        "line",
        [
            "from demo.core.A import A",
            "from .A import A",
            "from . import A",
            "import demo.core.A",
            "from demo.core import A",
        ],
    )
    def test_each_way_of_importing_the_decorated_module(
        self, tmp_path: Path, line: str
    ) -> None:
        report = reportOf(
            tmp_path,
            {"__init__.py": "", "core/__init__.py": line + "\n", "core/A.py": DECORATED},
        )
        assert report.decorated == ("demo.core.A",)
        assert report.missing == ()

    def test_a_tree_with_nothing_decorated_needs_no_import(self, tmp_path: Path) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": '"""Docs."""\n',
                "core/Contract.py": PLAIN,
            },
        )
        assert report.decorated == ()
        assert report.missing == ()

    def test_the_file_that_defines_the_decorator_needs_no_import(self, tmp_path: Path) -> None:
        definition = "def registerThing(name):\n    return lambda cls: cls\n"
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "from .A import A\n",
                "core/registerThing.py": definition,
                "core/A.py": DECORATED,
            },
        )
        assert report.missing == ()

    def test_every_decorated_module_of_a_package_is_imported_on_its_own_line(
        self, tmp_path: Path
    ) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "from .A import A\nfrom .B import B\n",
                "core/A.py": DECORATED,
                "core/B.py": DECORATED,
            },
        )
        assert report.decorated == ("demo.core.A", "demo.core.B")
        assert report.missing == ()

    def test_a_nested_package_that_its_parent_imports_is_accepted(
        self, tmp_path: Path
    ) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "from . import sub\n",
                "core/sub/__init__.py": "from .Leaf import Leaf\n",
                "core/sub/Leaf.py": DECORATED,
            },
        )
        assert report.decorated == ("demo.core.sub.Leaf",)
        assert report.missing == ()

    def test_the_layers_need_not_be_imported_by_the_top_level_package(
        self, tmp_path: Path
    ) -> None:
        # The top-level `__init__` is lazy (roadmap task P2-11): it imports no layer.
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "from .A import A\n",
                "core/A.py": DECORATED,
            },
        )
        assert "demo.core" not in report.missing

    def test_the_report_lists_the_packages_that_were_checked(self, tmp_path: Path) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "",
                "core/sub/__init__.py": "",
                "x/__init__.py": "",
            },
        )
        assert report.packages == ("demo", "demo.core", "demo.core.sub", "demo.x")


class TestADecoratedModuleThatItsInitDoesNotImportIsReported:
    def test_an_empty_init(self, tmp_path: Path) -> None:
        report = reportOf(
            tmp_path, {"__init__.py": "", "core/__init__.py": "", "core/A.py": DECORATED}
        )
        assert report.missing == ("demo.core.A",)

    def test_only_the_module_that_is_not_imported_is_reported(self, tmp_path: Path) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "from .A import A\n",
                "core/A.py": DECORATED,
                "core/B.py": DECORATED,
            },
        )
        assert report.missing == ("demo.core.B",)

    def test_a_plain_module_that_is_not_imported_is_not_reported(self, tmp_path: Path) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "from .A import A\n",
                "core/A.py": DECORATED,
                "core/Contract.py": PLAIN,
            },
        )
        assert report.missing == ()

    def test_a_module_in_the_top_level_package_itself(self, tmp_path: Path) -> None:
        report = reportOf(tmp_path, {"__init__.py": "", "Loose.py": DECORATED})
        assert report.missing == ("demo.Loose",)

    @pytest.mark.parametrize(
        "init",
        [
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .A import A\n",
            "def load():\n    from .A import A\n",
            "from demo.other.A import A\n",
            "# from .A import A\n",
            '"""from .A import A"""\n',
            "A = 'demo.core.A'\n",
        ],
        ids=[
            "type-checking-only",
            "deferred-in-a-function",
            "same-name-elsewhere",
            "commented-out",
            "in-the-docstring",
            "just-a-string",
        ],
    )
    def test_an_import_that_does_not_run_does_not_count(
        self, tmp_path: Path, init: str
    ) -> None:
        report = reportOf(
            tmp_path, {"__init__.py": "", "core/__init__.py": init, "core/A.py": DECORATED}
        )
        assert report.missing == ("demo.core.A",)

    def test_a_package_with_no_init_cannot_import_anything(self, tmp_path: Path) -> None:
        report = reportOf(tmp_path, {"__init__.py": "", "core/A.py": DECORATED})
        assert report.missing == ("demo.core.A",)

    def test_a_nested_package_that_its_parent_does_not_import(self, tmp_path: Path) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "",
                "core/sub/__init__.py": "from .Leaf import Leaf\n",
                "core/sub/Leaf.py": DECORATED,
            },
        )
        assert report.missing == ("demo.core.sub",)

    def test_both_the_module_and_its_nested_package_when_neither_is_imported(
        self, tmp_path: Path
    ) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "",
                "core/sub/__init__.py": "",
                "core/sub/Leaf.py": DECORATED,
            },
        )
        assert report.missing == ("demo.core.sub", "demo.core.sub.Leaf")

    def test_a_deeper_package_is_reported_at_every_level_that_misses_its_import(
        self, tmp_path: Path
    ) -> None:
        report = reportOf(
            tmp_path,
            {
                "__init__.py": "",
                "core/__init__.py": "",
                "core/a/__init__.py": "",
                "core/a/b/__init__.py": "from .Leaf import Leaf\n",
                "core/a/b/Leaf.py": DECORATED,
            },
        )
        assert report.missing == ("demo.core.a", "demo.core.a.b")


# --------------------------------------------------------------------------------------------
# The completeness test, on the real tree
# --------------------------------------------------------------------------------------------


class TestTheRealTree:
    def test_the_scanner_sees_the_packages_of_planexs(self) -> None:
        # Guards against the scan silently matching nothing.
        report = checkCompleteness(SOURCE_ROOT, "planexs")
        assert {
            "planexs",
            "planexs.core",
            "planexs.domain",
            "planexs.domain.parameters",
            "planexs.domain.constraints",
            "planexs.backends.ax",
        } <= set(report.packages)

    def test_every_decorated_module_is_imported_by_the_init_of_its_package(self) -> None:
        report = checkCompleteness(SOURCE_ROOT, "planexs")
        assert report.missing == (), (
            "These modules register components (or hold modules that do) but the __init__.py "
            f"of their package does not import them: {list(report.missing)}. Add one import "
            "line per module (specification 5.12, 10.6): until then the components are "
            "silently unregistered."
        )

    def test_the_check_fails_on_a_copy_of_the_real_tree_with_one_unimported_module(
        self, tmp_path: Path
    ) -> None:
        # The proof that the test above can fail: the same check, on the real source with one
        # decorated module added, reports that module; adding the import line makes it pass
        # again. Compared with the copy's own baseline, so that a defect of the real tree is
        # reported once, by the test above, and not a second time here.
        copy_of_tree = tmp_path / "planexs"
        shutil.copytree(
            SOURCE_ROOT, copy_of_tree, ignore=shutil.ignore_patterns("__pycache__")
        )
        baseline = set(checkCompleteness(copy_of_tree, "planexs").missing)

        (copy_of_tree / "core" / "Extra.py").write_text(
            DECORATED.replace("demo.core", "planexs.core"), encoding="utf-8"
        )
        added = set(checkCompleteness(copy_of_tree, "planexs").missing) - baseline
        assert added == {"planexs.core.Extra"}

        init = copy_of_tree / "core" / "__init__.py"
        init.write_text(
            init.read_text(encoding="utf-8") + "\nfrom . import Extra\n", encoding="utf-8"
        )
        assert set(checkCompleteness(copy_of_tree, "planexs").missing) == baseline
