# Contributing

## Setup

```bash
uv sync --extra dev --extra docs
uv run pytest
uv run mypy
uv run ruff check .
uv run python scripts/check_naming_convention.py
```

## Naming convention (custom, overrides PEP8 default for callables and for file names)

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

No em dashes (`—`) in code, comments, docstrings, commit messages, or project documentation.
Use a period, a colon, parentheses, or two sentences instead. This is a house style rule, not a
technical one, so there is no linter for it; review for it like any other style note. Same for
arrows (`→`): use `->` instead.

## Modularity

One responsibility per file; one class per file. No god-files: a base class, its registry, and
every concrete strategy each get their own file (see `src/planexs/` for what this looks like in
practice: `parameters/`, `constraints/`, `acquisition/` each split this way).

## Docstrings

Google-style, mandatory on every public class/function: purpose, `Args`, `Returns`, `Raises`.

`ruff` enforces that a docstring exists (rule family `D1` with the Google convention, set in
`pyproject.toml`; `tests/` is exempt). What the docstring says is left to review. Document a
constructor's arguments in the class docstring or in `__init__`, but say each thing once.

## Type checking

`uv run mypy` runs in strict mode. numpy's own type stubs use Python 3.12 syntax, which mypy
rejects while `python_version` is 3.11 (the project's minimum), and that one error stops all
checking; `pyproject.toml` therefore tells mypy to skip numpy's stubs. If `uv run mypy` ever
reports a syntax error inside `site-packages`, check that override first.

## Version control

Conventional Commits, semantic versioning. Changelog entries are versioned files, not one
growing document: each release gets its own `docs/changelogs/changelog-vX.Y.Z.md`, written
once and never edited after release, and `docs/changelogs/CHANGELOG.md` is a short running
index linking to each of them. `scripts/cut_release.py` scaffolds this at release time.
Architectural decisions get a new ADR when they change, rather than an old ADR being edited in
place; `scripts/new_adr.py` scaffolds a new one from the ADR-0001 template.

## Don't recode what's already coded

Before adding a new abstraction, check whether Ax (<https://ax.readthedocs.io/en/stable/api.html>)
or BoTorch already has the shape needed. `PROJECT_SPECIFICATION.md` sections 4.5 and 4.6 are
the running record of that check for this project; keep it up to date rather than doing it once
and forgetting.
