# Contributing

## Setup

```bash
uv sync --extra dev --extra docs
uv run pytest
uv run mypy
uv run ruff check .
uv run python scripts/check_naming_convention.py
```

[Running the checks](#running-the-checks) lists every command that CI and the roadmap expect,
with what each one enforces. The [project specification](./PROJECT_SPECIFICATION.md) is the
ground truth for this project: when this guide and the specification disagree, the
specification wins, and the disagreement is a bug to fix in one of the two files. Section
numbers in the specification are stable references, and each section below cites the ones it
summarizes.

## Running the checks

**Specification:** 10.1, 10.13.

Run these from the repository root. Each one is a command that CI runs, except the
documentation build, which you run yourself today. The three CI jobs (`lint`, `types` and
`tests`, in `.github/workflows/ci.yaml`) run on Python 3.11 and 3.13.

| Check | Command | What it enforces |
|---|---|---|
| Lint | `uv run ruff check .` | The ruff rules of `pyproject.toml`, including that every public class and function has a docstring |
| Format | `uv run ruff format --check .` | The formatting of ruff, pinned to one exact version so that it never changes under you. Run `uv run ruff format .` to fix it |
| File names | `uv run python scripts/check_naming_convention.py` | A file is named after its one public symbol, including its case (specification 10.2) |
| Policy values | `uv run python scripts/check_policy_values.py --baseline scripts/policy_values_baseline.txt` | No new numeric literal that is a policy value, and no stale baseline entry (specification 3.4, 10.7) |
| Layering | `uv run python scripts/check_layering.py` | Imports follow the layer table of specification 5.1, and the library rules of 10.5 (the rules are in `scripts/layers.toml`) |
| Types | `uv run mypy` | Strict typing of `src/planexs` |
| Tests | `uv run pytest` | Every test, with coverage: the floor is `fail_under` in `pyproject.toml` and may only go up |
| Documentation | `uv run mkdocs build --strict` | Every page builds and every link resolves |

Some guards are tests, not scripts, and `uv run pytest` runs them all. To run one on its own:

| Guard | Command | What it enforces |
|---|---|---|
| Typography | `uv run pytest tests/unit/test_typography.py` | No em dash and no unicode arrow outside inline code (specification 10.3) |
| Registry contract and completeness | `uv run pytest tests/unit/core/test_registry_contract.py` | Every registered component meets the contract, and every concrete module is imported by its package (specification 10.6) |
| ADR records | `uv run pytest tests/unit/test_adr_records.py` | Every ADR is in the navigation, has the template sections and no placeholder, and supersession is recorded in both directions (specification 11) |
| Documentation links | `uv run pytest tests/unit/test_documentation_links.py` | Every relative link and anchor of the README and of `docs/` resolves |
| Contributor guide | `uv run pytest tests/unit/test_contributing_guide.py` | This guide has the sections below, and each cites sections that exist in the specification |

Two of the scripts are ratchets, and read the same way. `check_policy_values.py` and
`check_layering.py` each fail on something new and on an entry that no longer matches anything:
the list of accepted exceptions (`scripts/policy_values_baseline.txt`, and `known_violations`
in `scripts/layers.toml`) only ever shrinks. When your change removes a literal or an import
that was listed, delete its line. Do not add a line to silence a new violation: a request that
needs one conflicts with a rule, and it must be raised and not worked around (see
[the request check](#before-you-code-the-request-check)).

None of these checks sees everything. The policy-value check finds numeric literals and
nothing else, and the layering check reads import statements and nothing else. Review is the
control for the rest, and each script's docstring says what it cannot see.

## Before you code: the request check

**Specification:** 12.1, 12.2, 12.3, 3.1, 3.2.

Before you implement a request, answer these six questions in writing, and put the answers at
the top of the pull request description:

1. **Layer.** Which layer owns this? Name it. If the request is phrased against a different
   layer (for example "add a prediction method to the optimizer" when prediction is a
   surrogate operation), say so and propose the right place before you write anything.
2. **Axis.** What varies here? Name at least two other plausible values of the thing being
   requested. Deliver the request as the first instance of a mechanism for that axis, say which
   other instances the mechanism now allows, and implement only what was asked.
3. **Seam.** Is there a registry kind for that axis? If not, does the rule of two (a new kind
   needs at least two concrete implementations, specification 3.10) justify creating one?
4. **Values.** Which policy values does this introduce, where do they live in the
   configuration, and what are their defaults?
5. **Capabilities.** Which capability tags does it need, and which does it support?
6. **Use cases.** Which reference use cases (specification 7) does it touch or enable, and
   which tests change?

Stop and raise the point, instead of implementing it silently, when a request belongs to a
different layer from the one it targets, would hard-code a choice that has or should have a
registry kind, would add a literal policy value, would narrow a mechanism to the requested
instance, conflicts with a principle of specification 3 or with a reference use case, or needs
a decision that is still open in the [roadmap](./ROADMAP.md). When something is ambiguous,
ask. Do not guess.

## Registry rules

**Specification:** 10.6, 3.3, 3.7, 5.12.

Every decision on which a reasonable user could want a different answer is a registered
component, selected by name in configuration. A class hard-wired inside library code to make
such a choice is a defect, and so is a swap point that exists only in a docstring.

- **Go through the registry.** Every choice that has a registry kind goes through it. No
  algorithm imports a concrete component class to make a choice.
- **Register by decorator, at import time**, in the style
  `@registerSurrogate("single_task_gp_v1")`. A duplicate name is an error, and an unknown name
  is an error that lists the registered ones.
- **A registered name is permanent.** It is lowercase snake_case with an optional `_vN`
  suffix. A change that alters results gets a new name (a `_v2`), never an edit of the old
  one, so that a saved study keeps meaning what it meant.
- **Import every concrete module.** The `__init__` of each subpackage imports each of its
  concrete modules, one line per file. A decorated component is registered only when its
  module has been imported, so one that is not imported is silently unregistered. The
  completeness test checks it.
- **One file each.** Every kind has an abstract contract, a registry and a registration
  decorator, each in its own file, and every concrete implementation has its own file too.
- **Declare needs and supports.** Every entry declares the capability tags it needs and the
  tags it supports, drawn from the documented vocabulary: a tag that is not in it is refused
  when the component is registered, which catches a typo. The declaration must be truthful, and
  a test checks it: a component that declares it supports something works with it, and one that
  does not declare it fails the compatibility check with a clear message.
- **A new kind needs two implementations and an ADR** (rule of two, specification 3.10).
- **One contract test runs over every registry** (names, schemas, descriptions, tags,
  builders). It covers a new kind or component automatically, so a failure there is a defect
  of the component.

## Policy values

**Specification:** 3.4, 10.7.

A *policy value* is any number, string, flag or heuristic that changes numerical results,
performance or behaviour and could reasonably differ between users or problems: tolerances,
budgets, sample counts, restart counts, iteration caps, thresholds, jitter and floors,
seed-derivation constants, size limits, "use X when there are more than N" rules and default
component names.

- **It is a field of a typed configuration schema**, with a documented meaning, unit and
  default, and the default is written in exactly one place: that field. A constant buried in a
  function body, a module-level constant or a literal default in a function signature is a
  defect, however good its name. The caller must be able to reach every policy value through
  the study configuration, and the study stores its fully resolved configuration so that a
  later change of a default never alters how an old study behaves.
- **Exempt numbers.** 0, 1, -1 and 2 in any spelling (`0.0`, `2.0`) are structural, because
  they appear in formulas. Any other number that is part of a mathematical definition and not a
  tunable value (a tensor axis index, the midpoint of the unit cube) carries a marker on its own
  line, with a reason that says what it is:

  ```python
  n_candidates = x.shape[-2]  # structural: tensor axis
  middle = 0.5  # structural: unit-cube midpoint
  ```

  A marker without a reason exempts nothing. The marker is for mathematical definitions only:
  putting it on a tunable value is a defect that review must catch.
- **Configuration schemas are not scanned.** A file whose single public symbol is a class named
  `...Config` is where defaults legitimately live.
- **The check is a ratchet.** `scripts/check_policy_values.py` compares the violations with
  `scripts/policy_values_baseline.txt`. A new violation fails, and so does a baseline entry that
  no longer matches a literal. Entries are only ever removed: delete the line of any literal
  your change removes. `--write-baseline` creates the baseline or removes entries from it, and
  adding entries needs `--allow-additions`, which is for a deliberate re-baseline such as a file
  move. The baseline must be empty before the first release of the new architecture.
- **What the check cannot see.** Strings, booleans and a rule hidden behind an allowed number
  (`len(values) > 2` standing for "more than two values means ordered") are policy values too.
  Review is the control for those, and for every use of the marker.

## Configuration schemas and options objects

**Specification:** 10.8, 5.12.

- **A configuration schema is a Pydantic v2 model** with unknown fields rejected, a description
  on every field, and validators whose messages name the field. The description of a field says
  what the value means, its unit, and why its default is what it is. A component's schema
  derives from `ComponentConfig`, whose `type` field holds the registered name, and a component
  nested in another is declared with `ComponentOf`.
- **Defaults live only in schemas.** There is no dictionary-of-strings configuration scattered
  through the code.
- **An operation with tunable options takes one options object**, itself a configuration
  schema, instead of a list of keyword arguments with literal defaults. A literal default in a
  signature is a policy value, so it would be a defect.
- **One meaning in every form.** A configuration built in Python, loaded from JSON and loaded
  from YAML has the same meaning, and every validation error carries the path of the offending
  field.

## Adding a component: the checklist

**Specification:** 10.14, 6, 3.10.

1. Define the configuration schema, with every policy value described.
2. Implement the builder, with no literal policy value.
3. Register it under a new name, one class per file.
4. Declare its needs and supports.
5. Import it in its subpackage's `__init__`.
6. Add tests: validation of the configuration, behaviour, and capability truthfulness. The
   registry contract test covers it automatically.
7. Add or update the documentation entry.
8. If it creates a new kind, write an ADR (`scripts/new_adr.py`) and satisfy the rule of two.

A builder is free of side effects at import and registration time, and deterministic given its
configuration, its inputs and its random generator. It never touches a global random state.
Run the checks of [Running the checks](#running-the-checks) before you open the pull request.

## Naming convention (custom, overrides PEP8 default for callables and for file names)

**Specification:** 10.2.

This project deliberately deviates from PEP8's naming rules for callables and for file names.
The deviation is intentional and enforced (`tool.ruff.lint.ignore` in `pyproject.toml` disables
`N802`/`N803`/`N806` for function, argument and variable names and `N999` for module names, and
`scripts/check_naming_convention.py` runs in CI), so please do not "fix" it back to
`snake_case`: that would be reverting a deliberate decision, not a cleanup.

- **Classes** -> `CamelCase` (e.g. `SearchSpace`, `Real`, `ExplorationExploitationAcquisition`).
- **Variables** (including function/method parameters) -> `snake_case` (e.g. `n_points`,
  `random_seed`).
- **Functions and methods** -> same rule as classes but starting lowercase, i.e. `camelCase`
  (e.g. `suggestDefault`, `computeSensitivity`, `buildAcquisitionFunction`), *not* PEP8's usual
  `snake_case` for callables. Python's required dunder methods (`__init__`, `__repr__`, ...)
  are exempt: they keep their mandatory spelling.
- **File names match the file's single public symbol exactly**, including its case: a file
  defining `class BayesianOptimizer` is `BayesianOptimizer.py`, not `bayesian_optimizer.py`; a
  file defining `def buildSurrogateModel(...)` is `buildSurrogateModel.py`. This is the "one
  class per file" rule taken one step further: if a file's name and its one symbol's name can
  drift apart, they will, over refactors. A file with more than one closely related symbol and
  no single obvious name (rare, given "one responsibility per file") keeps a descriptive
  `snake_case` name instead; `logging_config.py` is that case, not an exception to avoid.
  Python's own required module names (`__init__.py`, `conftest.py`, ...) are exempt, the same
  way dunder methods are. `scripts/check_naming_convention.py` enforces this in CI, since
  `ruff`'s naming rules do not cover file names against class/function names.
- **Properties** (`@property`) are named by **cost, not by whether they compute anything**:
  - If accessing the property does real work (recomputes something from the full trial
    history, calls into a model, is not O(1)-ish) -> `camelCase`, exactly like a method, even
    though it is called without parentheses. Example: `BayesianOptimizer.paretoFront`.
  - If it just returns an already-stored or negligible-cost value -> `snake_case`, like a
    variable. Example: `BayesianOptimizer.n_trials`, `PredictionResult.variance`.
  - The point is that the caller can tell, from the name alone, whether touching this
    attribute is free or not.

## Typography

**Specification:** 10.3.

No em dashes (`—`) in code, comments, docstrings, commit messages, or project documentation.
Use a period, a colon, parentheses, or two sentences instead. Same for arrows (`→`): write
`->` instead. A test enforces it: `tests/unit/test_typography.py` fails on a banned character
in any tracked `.py`, `.md`, `.yaml` or `.toml` file, and ignores text inside an inline code
span, so that the rule itself can be written down (as it is here). A code span must sit on one
line to be recognized.

## Modularity

**Specification:** 10.5, 5.1.

One responsibility per file; one class per file. No god-files: a base class, its registry, and
every concrete strategy each get their own file (see `src/planexs/` for what this looks like in
practice: `parameters/`, `constraints/`, `acquisition/` each split this way).

Subpackages depend on each other only in the direction of the layer table of specification
5.1: a lower layer never imports a higher one. `core` and `domain` import no machine-learning
library (`torch`, `botorch`, `gpytorch`, `ax`), and only `adapters` import Ax. `uv run python
scripts/check_layering.py` enforces it from the rules in `scripts/layers.toml` (see
[Running the checks](#running-the-checks)). Do not add a `known_violations` entry to make a
new import pass: raise the request instead.

## Docstrings

**Specification:** 10.4.

Google-style, mandatory on every public class/function: purpose, `Args`, `Returns`, `Raises`.

`ruff` enforces that a docstring exists (rule family `D1` with the Google convention, set in
`pyproject.toml`; `tests/` is exempt). What the docstring says is left to review. Document a
constructor's arguments in the class docstring or in `__init__`, but say each thing once.

## Type checking

**Specification:** 10.1, 10.4.

`uv run mypy` runs in strict mode. numpy's own type stubs use Python 3.12 syntax, which mypy
rejects while `python_version` is 3.11 (the project's minimum), and that one error stops all
checking; `pyproject.toml` therefore tells mypy to skip numpy's stubs. If `uv run mypy` ever
reports a syntax error inside `site-packages`, check that override first.

## Version control

**Specification:** 10.13, 10.12, 11.

Conventional Commits, semantic versioning. Changelog entries are versioned files, not one
growing document: each release gets its own `docs/changelogs/changelog-vX.Y.Z.md`, written
once and never edited after release, and `docs/changelogs/CHANGELOG.md` is a short running
index linking to each of them. `scripts/cut_release.py` scaffolds this at release time.
Architectural decisions get a new ADR when they change, rather than an old ADR being edited in
place; `scripts/new_adr.py` scaffolds a new one from the ADR-0001 template.

## Don't recode what's already coded

**Specification:** 3.5, 10.9.

Before adding a new abstraction, check whether BoTorch, GPyTorch or Ax
(<https://ax.readthedocs.io/en/stable/api.html>) already has the shape needed. If it does,
register a builder for it instead of re-implementing it, and write new mathematics only as a
native engine subclass, so that it works with the engine's own optimizers. Verify third-party
names and signatures against the installed version (by introspection, or by reading the
installed source) and never write engine API from memory. If something a task names does not
exist in the installed version, stop and report it instead of inventing a substitute.
