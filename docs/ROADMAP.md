# Roadmap

**Status:** living document, v0.1. Update each task's status when it completes, and refresh the baseline (section 2.1) at each release.
**Baseline:** commit `f5829e4` (version 0.1.0, five commits, no tag), measured on 2026-10-09. The static measurements come from that commit. The dynamic ones (tests, coverage, mypy, ruff, documentation build) were taken by task P0-02 on 2026-10-11 at commit `d1c0b46` (section 2.1).
**Authority:** the [project specification](PROJECT_SPECIFICATION.md) is the ground truth. When this file and the specification disagree, the specification wins, and the disagreement is a bug to fix in one of the two files. Open decisions are asked, never guessed (specification 11 and 12): every task that depends on one names it (`D-n`), and section 17 lists them all with a proposed default.

---

## 1. At a glance

The code at version 0.1.0, named boptim until the rename of D-1, is a Bayesian-optimization library built on Ax, with a custom BoTorch acquisition layer for what Ax cannot do. It works, and its parameter and constraint model, its search-space encoder and its constraint machinery are worth keeping. Its architecture cannot reach the scope of the specification: the objective is the centre of the domain, Ax is the foundation, the surrogate and the acquisition are hard-wired, two engines and two surrogates live side by side, and policy values are hidden constants.

**Decision recorded here: refactor the core, keep the periphery, do not restart.** The new architecture is built beside the old code, and the old path is deleted only once the twelve reference use cases pass on the new one (section 2.3).

| Phase | Theme | Outcome | Tasks |
|---|---|---|---|
| 0 | Guardrails | CI, checkers, fixed documentation entry points | 8 (7 S, 1 M) |
| 1 | Core mechanics | Registries, capabilities, configuration, plugins, layering check | 17 (14 S, 3 M) |
| 2 | Domain generalization | Pure, registry-driven domain; legacy types isolated | 12 (10 S, 2 M) |
| 3 | Encoding and surrogates | Encoder under `encoding/`, surrogate kind, GP and classifier builders | 17 (14 S, 3 M) |
| 4 | Goals and designs | Goal types (optimize, level set, explore) and two designs | 7 (6 S, 1 M) |
| 5 | Acquisitions and optimizers | Registered acquisitions and optimizers; outcome constraints reach the acquisition | 14 (8 S, 6 M) |
| 6 | Policies and the Study | Policies, recipes and the `Study` | 15 (12 S, 3 M) |
| 7 | Analyses and stopping | Best point, Pareto front, level-set estimate, importance, diagnostics; stopping criteria | 11 (7 S, 4 M) |
| 8 | Persistence | Snapshot, repositories, save and load | 8 (6 S, 2 M) |
| 9 | Acceptance | The twelve reference use cases pass | 12 (0 S, 12 M) |
| 10 | Input extensibility and the Ax adapter | Registry-driven inputs, Ax as an optional adapter | 10 (6 S, 4 M) |
| 11 | Cleanup and release | Legacy deleted, documentation, release 0.2.0 | 15 (15 S, 0 M) |

The plan has 146 tasks (105 of size S and 41 of size M). There is no larger size, on purpose (section 3.1): a task is small enough to be finished completely, tests and documentation included, in one sitting.

**Order.** Phase 0 and Phase 1 come first, in order. Phases 2 and 3 can then start in parallel (P3-01 to P3-05 only need Phase 1); the surrogate tasks of Phase 3 need the domain of Phase 2. Phase 4 needs Phases 2 and 3 (designs use the feasible sampler), Phase 5 needs Phases 3 and 4, and Phase 6 needs Phase 5. Phases 7 and 8 can proceed in parallel after Phase 6. Phase 9 needs everything before it, Phase 10 follows Phase 9, and Phase 11 closes the plan.

**Needed from the owner first.** Phase 0 can start today. Before the tasks that depend on them, the owner should answer D-6 (CI matrix, for P0-03), D-5 (policy-value exemptions, for P0-05), D-4 (configuration formats, for P1-11), D-10 (entry-point group, for P1-02 and P1-12) and D-12 (runtime defaults, for P1-14). Each has a proposed default in section 17, so "go with the defaults" is a valid answer.

---

## 2. Starting point

### 2.1 Baseline

Static measurements on a fresh clone of `f5829e4` (2026-10-09), completed by the dynamic measurements of task P0-02, taken on 2026-10-11 at commit `d1c0b46` (upstream `0491e04` plus the P0-01 fix; tree `8338ba65fe60`). The measurements ran on a fresh clone of a local commit of that same change (`9fdaa09`, before the owner's commit existed): its tree is identical to that of `d1c0b46`, so every number below holds for `d1c0b46`. Environment and commands: Appendix A. Paths and module names in this section use the current name `planexs`; commits `f5829e4`, `0491e04` and `d1c0b46` were made under the former name `boptim` (D-1).

| Check | Result |
|---|---|
| Commit and version | `f5829e4` ("phase 2"), 5 commits, version 0.1.0, no tag. `pyproject.toml` declares the license as TBD. |
| Source | 5,823 lines in 58 files under `src/planexs`. |
| Tests | 29 test files with 359 test functions (4,535 lines), collected by pytest as 425 tests (parametrization). Measured at `d1c0b46`: 425 passed, 0 failed, 0 skipped, 65 warnings, in 167 s on one CPU. The 65 warnings are 15 distinct messages: 14 are raised inside ax (8), botorch (3), torch, linear_operator and gpytorch, and 1 at `tests/unit/acquisition/test_encoded_constraints.py:143` (a tensor with `requires_grad` converted to a scalar). Coverage (`--cov=planexs`, set in `pyproject.toml`): 94.55% (1,667 of 1,763 statements), with 42 of 58 files at 100%. Two files are at 0%: `backends/ax/fromAxSearchSpace.py` (39 statements) and `logging_config.py` (13). |
| ruff | Measured at `d1c0b46` with ruff 0.16.9 (the version locked in `uv.lock`) and with ruff 0.17.0 (the version of the static baseline): identical results. `ruff check .`: 1 finding, E501 in `examples/nonlinear_constraint.py:91` (109 columns, limit 95), exit 1. `ruff format --check .`: 1 file would be reformatted and 109 are already formatted, exit 1. The file is `docs/PROJECT_SPECIFICATION.md` (line 447): ruff also formats the Python blocks of Markdown files, and the illustrative snippet added in `0491e04` is not in its format. At `f5829e4` all 109 files were formatted. P0-08 fixes both failures. |
| Naming check | `scripts/check_naming_convention.py` reports 1 violation: `src/planexs/domain/parameters/parameter.py` is tracked in lowercase while every import says `Parameter`. A fresh clone cannot import the package on a case-sensitive filesystem (P0-01). |
| mypy | mypy 2.3.1 with `strict = true` from `pyproject.toml`: 0 errors (`Success: no issues found in 58 source files`), exit 0. |
| CI | None: there is no `.github/` directory, although specification 10.13 requires CI. |
| Documentation build | `mkdocs build --strict` exits 1 with 3 warnings: links in `docs/index.md` to `../PROJECT_SPECIFICATION.md` and `../README.md`, and a link in `docs/changelogs/CHANGELOG.md` to `../../scripts/cut_release.py`, do not resolve. `PROJECT_SPECIFICATION.md`, `ROADMAP.md` and `changelogs/changelog-v0.1.0.md` are not in `nav` (information only). P0-07 covers it. |
| Policy values | 39 numeric literals outside {0, 1, -1, 2} in 11 of the 58 files. About 30 are policy values (restarts, sample counts, tolerances, caps, seed constants), 6 are tensor axis indices and 3 are unit-cube midpoints. The most: the encoder (7), the alpha strategy (6) and the feasible sampler (6). |
| Hard-wired engine choices | `SingleTaskGP`, `Standardize`, `ExactMarginalLogLikelihood`, `fit_gpytorch_mll`, `SobolQMCNormalSampler`, `optimize_acqf`, `optimize_acqf_mixed` and `ScalarizedPosteriorTransform` are named inside algorithms in 5 files. |
| Closed dispatch | 9 files branch on concrete parameter or constraint classes, or on their type tags (the encoder, the encoded constraints, both `...FromDict` functions, the parameter encoding, the facade and the Ax mapping files). |
| Typography | No em dash or unicode arrow in `src`, `tests`, `examples`, `scripts` or the README. In `docs/`: 5 lines with an em dash and 2 with an arrow (the old specification, the contributing guide quoting the rule, one changelog). |
| Documentation | 8 ADRs (ADR-0003 and ADR-0004 are stubs marked 'to write in full'), 4 examples, 3 scripts, 2,394 lines of Markdown. |
| Stale references | 43 `section N` mentions in code, tests, examples and scripts, all citing the numbering of the old specification. |

### 2.2 What is kept, reworked, replaced

| Current module | Fate | Where it goes, and why |
|---|---|---|
| `domain/parameters/*`, `domain/constraints/*`, `domain/SearchSpace.py` | Keep, rework | Same classes. Engine vocabulary removed (P2-03), types dispatched through registries (P2-04, P2-05), serialization key `type`, a pure evaluator for nonlinear constraints (P2-09). |
| `domain/constraints/validateExpression.py` | Keep | Its limits become configuration (P3-06). |
| `domain/Trial.py`, `Metric.py`, `Objective.py`, `OutcomeConstraint.py`, `StudySnapshot.py` | Replace | Renamed `Legacy*` (P2-02). Replaced by outcomes (P2-06), the trial lifecycle (P2-07), observations (P2-08), goals (P4-03) and the new snapshot (P8-02). Deleted in P11-03. |
| `models/SearchSpaceEncoder.py`, `ParameterEncoding.py`, `compileExpression.py` | Keep, move | `encoding/` (P3-01). Policy values to configuration (P3-03). Registry dispatch (P10-01 to P10-04). |
| `acquisition/EncodedConstraints.py`, `sampleFeasibleEncoded.py`, `toBotorchNonlinearConstraints.py` | Keep, move | `encoding/` (P3-04). Policy values to configuration (P3-05). Constraint-encoding registry (P10-05). |
| `models/buildSurrogateModel.py`, `encodeTrials.py`, `predictWithModel.py` | Replace | `surrogates/` (P3-08, P3-09, P3-14). Deleted in P11-02 after the parity test. |
| `acquisition/ExplorationExploitationAcquisition.py`, `MultiObjectiveExplorationExploitationAcquisition.py`, `modelParetoFront.py` | Keep, port | Registered acquisitions (P5-07, P5-08). The legacy copies are deleted in P11-02. |
| `acquisition/AlphaAcquisitionStrategy.py`, `AcquisitionStrategy.py` | Replace | Split into optimizers (P5-12, P5-13) and the composed policy (P6-03, P6-04). Deleted in P11-02. |
| `analysis/PredictionResult.py` | Replace | `Prediction` (P3-09). Deleted in P11-03. |
| `backends/OptimizationBackend.py`, `PredictionUnavailableError.py`, `backends/ax/*` | Replace | Ax logic moves to `adapters/ax` as an optional policy (P10-07 to P10-09). The abstract backend disappears in P11-01. |
| `persistence/JsonStudyRepository.py`, `ReproducibilityMetadata.py` | Keep, rework | The repository becomes a registered kind (P8-03). The metadata gains the seed entropy (P8-02). |
| `api/BayesianOptimizer.py` | Replace | `Study` (Phase 6). Deleted in P11-01. |
| `logging_config.py` | Keep | Unchanged. |
| `scripts/` (naming check, release, ADR) | Keep | Checkers are added (P0-05, P1-16 and the typography test of P0-04). |
| `tests/` | Keep, adapt | Tests of kept modules move with them. Tests of replaced modules stay until deletion and double as parity references. |
| `examples/` | Port | P11-05 and P11-06. The use-case examples are new (Phase 9). |

### 2.3 Migration strategy

1. New packages are built beside the old code. The old `BayesianOptimizer` path keeps working, and its tests keep passing, until the twelve reference use cases pass on `Study`.
2. Old code is never extended. It is only renamed (P2-02), moved (Phase 3) or deleted (Phase 11).
3. Before a legacy implementation is deleted, a parity test compares it with its replacement on a fixed problem, within tolerances stated in the test (P3-14, P5-07, P5-08, P5-12). The comparison is run, not assumed.
4. There is no compatibility shim for the 0.1.0 API (D-2): the project has five commits and no release tag, so a shim would cost more than it protects.
5. A task that finds it must touch legacy code for any other reason stops and asks.

---

## 3. How to work a task

### 3.1 Sizes

- **S**: one concept; at most about 150 changed lines of new or rewritten source and at most 3 source files created or substantially changed.
- **M**: one concept with several parts; at most about 400 changed lines and at most 6 source files.
- Mechanical moves and renames (`git mv`, import updates) do not count toward these limits, but a task touches at most 20 files that way.
- Tests, docstrings and documentation belong to the task and are never "later".
- There is no larger size. A task that turns out to be larger than M is stopped and split (step 1 below), never shrunk silently.

### 3.2 Protocol

1. Read the card, every specification section it cites and the files it names. If the task looks larger than its size, or its Do text is ambiguous, stop and say so before writing code.
2. Run the request check (specification 12.2) and write its six answers at the top of the pull request description.
3. Do everything under Do. All of it. Nothing is left as `pass`, `NotImplementedError`, `TODO`, `FIXME`, commented-out code or a skipped test unless the card says so.
4. Write the tests of the Done-when list first or alongside the code. A test that cannot fail does not count.
5. Run the verification commands of section 3.3.
6. Check each Done-when box against evidence (a test name, a command output). A box without evidence stays unchecked.
7. Report in the format of section 3.4. If any item is not done, the task is not done: say so plainly. Do not start another task in the same change and do not fix unrelated things; list them as findings.
8. Verify third-party names and signatures in the installed version before using them (specification 10.9). If something a card names does not exist, stop and report it.
9. Ask, rather than guess, when a decision of section 17 is needed or the card is ambiguous.

### 3.3 Verification commands

Run the ones that exist at the time of the task (the checkers arrive in Phases 0 and 1).

```
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/check_naming_convention.py
uv run python scripts/check_policy_values.py --baseline scripts/policy_values_baseline.txt
uv run python scripts/check_layering.py
uv run mkdocs build --strict
```

### 3.4 Report format

```
Task: P3-09 (title)
Request check: layer / axis / seam / policy values / capabilities / use cases
Done-when:
| item | done / not done / deviated | evidence (test name, command output) |
Files created or changed: ...
Deviations from the card, with the reason: ...
Findings (noticed, not changed): ...
```

### 3.5 What counts as shortcutting

These are failures even when the tests are green:

- delivering part of Do and calling the task done;
- a placeholder, a stub or a mock standing where the real thing is required;
- a test that cannot fail, that mocks the thing under test, or whose tolerance was loosened to pass without saying so;
- declaring parity without running the parity test;
- a literal policy value hidden in code, or a `# structural:` marker on a tunable value;
- editing the expected values of a legacy test so that the new code passes, unless the card says to;
- merging two tasks, or silently dropping a checklist item;
- inventing a third-party API instead of checking the installed one.

### 3.6 Task card format

Each card has an identifier (`Pn-mm`), a title, a size, the tasks it needs, the specification sections it implements, the decision it waits for (if any), a status, a Do paragraph, a Done-when checklist and, when useful, a note on what is out of scope. The status is one of `todo`, `doing` or `done` and is updated in the pull request that completes the task. Needs are direct dependencies; transitive ones are implied.

---

## 4. Sequencing rules

1. **Guardrails first.** CI and the checkers exist before any refactoring starts.
2. **Decisions before code.** The ADR tasks (P1-01 to P1-04, and the first task of Phases 2, 4 and 8) precede the code they justify.
3. **Dependencies are hard.** Tasks with no dependency between them can run in parallel.
4. **Strangler, not rewrite** (section 2.3).
5. **Ask, do not guess.** A task that depends on an open decision waits for the answer, or for the owner to accept the proposed default.
6. **Registry pattern, always.** No task adds a hard-coded choice or a literal policy value. If a task seems to need one, stop and raise it (specification 12.3).
7. **Every task ships whole**: code, tests, docstrings, and the removal of the baseline lines it makes obsolete.
8. **A phase is finished when its exit criteria hold**, not when its last task is merged.

---

## 5. Phase 0: Guardrails

Make the repository safe to refactor. Fix the defect that breaks a fresh clone on Linux, add CI, and add the check that enforces the policy-value rule, so that regressions are caught by machines and not by review. Nothing in this phase changes the library's behaviour.

**Exit criteria.** CI is green and required; the policy-value check runs with a baseline; the documentation entry points are fixed.

### P0-01. Fix the file-name case defect

**Size:** S | **Needs:** none | **Spec:** 10.2 | **Status:** done

**Do.** `git ls-files` tracks `src/planexs/domain/parameters/parameter.py` in lowercase, while every import and the naming rule expect `Parameter.py`. On a case-sensitive filesystem a fresh clone cannot import the package. Rename the tracked file with `git mv` through a temporary name (so that the case change is recorded on any filesystem) and fix any remaining reference.

**Done when.**
- [ ] `git ls-files` lists `Parameter.py` and no lowercase variant;
- [ ] `python scripts/check_naming_convention.py` exits 0;
- [ ] in a fresh clone on Linux with the dependencies installed, `python -c 'import planexs'` succeeds (paste the output in the report).

### P0-02. Measure the dynamic baseline

**Size:** S | **Needs:** P0-01 | **Spec:** 10.13 | **Status:** done

**Do.** Install the project (`uv sync --extra dev --extra docs`), then run `pytest` with coverage, `mypy`, `ruff check .` and `ruff format --check .`. Record in section 2.1 of this file, replacing the 'not measured' lines: the number of tests and failures, the coverage percentage, the mypy error count, the commit hash and the date. Change no code in this task.

**Done when.**
- [ ] section 2.1 holds the measured numbers with the commit hash and the date;
- [ ] every failing test or mypy error is listed as a finding for the owner (and, if it blocks a later task, as a new task card), not fixed here.

### P0-03. Add the CI workflow

**Size:** S | **Needs:** P0-02, P0-08 | **Spec:** 10.13 | **Decision:** D-6 | **Status:** todo

**Do.** Add `.github/workflows/ci.yaml`, triggered on push and on pull request, using `uv`. Three jobs: `lint` (`ruff check .`, `ruff format --check .`, `scripts/check_naming_convention.py`), `types` (`mypy`) and `tests` (`pytest` with coverage, failing under a floor one point below the coverage measured in P0-02), on the Python versions of D-6. Add a status badge to `README.md`.

**Done when.**
- [ ] the workflow is green on the main branch;
- [ ] a throwaway branch with a deliberate type error fails the `types` job (give the run link or the log);
- [ ] the coverage floor is written in one place, with a comment saying that it may only go up.

### P0-04. Add the typography guard

**Size:** S | **Needs:** P0-03 | **Spec:** 10.3 | **Status:** done

**Do.** Add `tests/unit/test_typography.py`: scan the git-tracked `.py`, `.md`, `.yaml` and `.toml` files for the em dash and the unicode right arrow, ignoring occurrences inside inline code spans (so that the rule itself can be documented). Fix what it finds: 3 em dashes in `docs/changelogs/changelog-v0.1.0.md` (punctuation-only edits to a released changelog are allowed for this one-time sweep); the other hits are in the old specification, which the new one replaces.

**Done when.**
- [ ] the test passes on the tree and fails when a banned character is added to a scratch file (state the evidence);
- [ ] no tracked file contains a banned character outside inline code spans.

### P0-05. Write the policy-value checker

**Size:** M | **Needs:** P0-03 | **Spec:** 3.4, 10.7 | **Decision:** D-5 | **Status:** done

**Do.** Create `scripts/check_policy_values.py` (standard library only, AST based). It reports every numeric literal other than 0, 1, -1 and 2 (any spelling: `0.0`, `1.0`, `2.0`) in `src/planexs`, except in files whose single public symbol is a configuration schema (class name ending in `Config`, where defaults legitimately live). A literal is exempt when its line carries `# structural: <reason>` with a non-empty reason. Each violation prints file, line, value and position (module level, signature default, function body). Add `--baseline FILE` (fail on violations absent from the file and on entries that no longer exist) and `--write-baseline`.

**Done when.**
- [ ] unit tests cover allowed literals, negative numbers, the marker with and without a reason, the configuration-file exclusion, signature defaults, and the baseline add and stale cases;
- [ ] on the current tree it reports 39 violations in 11 files (section 2.1), or the report explains each difference;
- [ ] the script's docstring states what it cannot see (strings, booleans, rules hidden behind 0, 1 or 2) and that review covers them.

**Note.** `--write-baseline` creates the baseline or removes entries from it; adding entries to an existing baseline is not done unless `--allow-additions` is given, for a deliberate re-baseline such as a file move.

### P0-06. Wire the policy-value baseline into CI

**Size:** S | **Needs:** P0-05 | **Spec:** 10.7 | **Status:** done

**Do.** Generate `scripts/policy_values_baseline.txt` from the current tree and run `check_policy_values.py --baseline` in the `lint` job. Edit no source file: the 39 entries disappear as the files that hold them are rewritten or deleted, and each later task that removes one deletes its line.

**Done when.**
- [ ] CI runs the check;
- [ ] adding a numeric literal to a file under `src/planexs` makes the job fail (state the evidence);
- [ ] the baseline file starts with a comment saying that entries are only ever removed.

### P0-07. Fix the documentation entry points

**Size:** S | **Needs:** P0-01 | **Spec:** 11 | **Status:** done

**Do.** Copy the new `PROJECT_SPECIFICATION.md` and `ROADMAP.md` into `docs/`. In `README.md`, replace the status section (it describes the old phases and test counts) with a short status line that points to the roadmap, and repair the links to the specification (`./PROJECT_SPECIFICATION.md` does not resolve from the repository root). Add both documents to the `mkdocs.yml` navigation.

**Done when.**
- [ ] every link in `README.md` and in the two documents resolves;
- [ ] `mkdocs build --strict` passes with both documents in the navigation;
- [ ] the README no longer states phase numbers or test counts.

### P0-08. Make ruff pass on the current tree

**Size:** S | **Needs:** P0-02 | **Spec:** 10.1, 10.13 | **Status:** done

**Do.** P0-02 measured that `ruff check .` and `ruff format --check .` both exit 1 on the current tree (section 2.1), so the `lint` job of P0-03 would fail on its first run. Fix both with the smallest change and no change of behaviour:
1. `examples/nonlinear_constraint.py:91` is 109 columns wide for a limit of 95 (E501). Split that f-string into two adjacent f-strings, breaking at the space before `(limit ...)`: the printed text is identical. The file is rewritten in P11-06, which keeps its own scope.
2. `ruff format` also formats the Python blocks inside Markdown files, and wants to reflow the illustrative `MyBoundaryScoreBuilder` snippet of section 9.2 of `docs/PROJECT_SPECIFICATION.md` (a line number would go stale: it was line 447 when P0-02 measured it). Add a `[tool.ruff.format]` table with `exclude = ["*.md"]` to `pyproject.toml`, with a comment saying that Markdown is not formatted by ruff and why. This option adds to ruff's global exclusions and does not touch linting. Do not edit the specification: it is the ground truth, and P0-07 puts its copy in `docs/`.

**Done when.**
- [ ] `ruff check .` and `ruff format --check .` both exit 0 with the ruff version locked in `uv.lock` (paste both outputs);
- [ ] a deliberately mis-formatted scratch `.py` file is still reported by `ruff format --check`, and a scratch line longer than 95 columns is still reported by `ruff check` (evidence that neither the exclusion nor the fix weakens the checks);
- [ ] `mypy` still reports 0 errors and `pytest` still reports 425 passed;
- [ ] the change touches exactly two files, `pyproject.toml` and `examples/nonlinear_constraint.py`, and the specification is unchanged.

**Note.** Added by P0-02, which checked this remedy on a throwaway copy: both ruff commands exit 0, mypy is unchanged, 5 lines changed. It is a proposal until the owner accepts the pull request that adds this card.

---

## 6. Phase 1: Core mechanics

Record the architecture decisions, then build the `core` layer: the error hierarchy, registries and the kind table, capability tags and the compatibility check, component configuration with nesting and resolution, configuration files, plugin discovery, random-number derivation and runtime settings. Add the layering checker. The `core` layer imports nothing from the rest of planexs and no machine-learning library.

**Exit criteria.** A toy component kind can be defined, registered, configured from YAML, compatibility-checked and discovered from a plugin, with tests, using nothing but `core`; the layering check runs in CI.

### P1-01. ADR-0009: study, policy and component architecture

**Size:** S | **Needs:** P0-07 | **Spec:** 5, 11 | **Status:** done

**Do.** Scaffold the ADR with `scripts/new_adr.py` (it takes the next number). Record the decision of the specification: planexs is organized around a study, a goal and a policy made of registered components; Ax stops being the foundation and becomes an optional adapter; the custom acquisition layer becomes ordinary components. Its context section also records the rename of the project from boptim to planexs (D-1, decided on 2026-10-11). Options considered: keep Ax as the foundation and add features on top; wrap BoTorch directly without registries; the design of the specification. Set the status line of ADR-0001, 0005, 0006, 0007 and 0008 to 'Superseded by ADR-0009' (status lines only; their text is untouched). Add the ADR to `mkdocs.yml`.

**Done when.**
- [ ] every section of the ADR template is filled, with no placeholder text;
- [ ] the five status lines are updated and the navigation lists ADR-0009;
- [ ] `mkdocs build --strict` passes.

### P1-02. ADR-0010: registries, capability tags and stable names

**Size:** S | **Needs:** P1-01 | **Spec:** 3.3, 3.7, 5.12 | **Decision:** D-10 | **Status:** done

**Do.** Record: one registry per component kind; decorator registration; names that are permanent identifiers, with a `_vN` suffix for behaviour changes; needs and supports tags with rejection of unknown tags; a compatibility check when a study is built; plugin discovery by entry point (group name chosen in D-10); the population rule. Options considered: Hydra-style import paths as the only mechanism (names would not be stable and capabilities need metadata; may be added later as a level-3 convenience), inheritance with factory methods, plain dictionaries.

**Done when.**
- [ ] ADR complete and in the navigation;
- [ ] the entry-point group name of D-10 appears in it.

### P1-03. ADR-0011: policy values, resolved configuration, seeds

**Size:** S | **Needs:** P1-01 | **Spec:** 3.4, 3.7, 5.11, 10.7 | **Decision:** D-5 | **Status:** done

**Do.** Record: the policy-value rule and its exemptions; the resolved configuration stored in snapshots; the derivation of all random streams from one study seed; the ratchet checker and its baseline. Options considered: named module constants (rejected: unreachable by the caller), a global settings object (rejected: hidden state), configuration schemas with a single documented default (chosen).

**Done when.**
- [ ] ADR complete and in the navigation.

### P1-04. Complete ADR-0003

**Size:** S | **Needs:** P1-01 | **Spec:** 3.9, 11 | **Status:** done

**Do.** ADR-0003 is a stub marked 'to write in full'. Write it from the stub: context (a domain independent of every engine), decision (Pydantic v2 models), options (dataclasses, attrs, plain dictionaries), consequences, including the purity rule of specification 3.9. Mark it Accepted and remove 'to write' from its title in `mkdocs.yml`.

**Done when.**
- [ ] no placeholder text remains in the ADR or in its navigation label.

### P1-05. Error hierarchy

**Size:** S | **Needs:** P1-02 | **Spec:** 3.6, 10.10 | **Status:** done

**Do.** Create the `core` package with `PlanexsError` and its subclasses `ConfigError`, `RegistryError`, `CompatibilityError` and `ModelNotReadyError`, one class per file. Every error carries a message and optional structured hints (component name, field path, suggestion) that `str()` renders on separate lines.

**Done when.**
- [ ] tests cover inheritance from `PlanexsError` and rendering with and without each hint;
- [ ] `core/__init__.py` has a docstring stating the layer's rule: it imports nothing from planexs and no machine-learning library.

### P1-06. Registry and the kind table

**Size:** S | **Needs:** P1-05 | **Spec:** 5.12, 10.6 | **Status:** done

**Do.** Add `core/Registry.py`: a typed registry for one component kind. It registers a name (a duplicate or an invalid name raises `RegistryError`), looks a name up (an unknown name raises `RegistryError` listing the available names), lists names sorted and tests membership. A valid name is lowercase snake_case with an optional `_vN` suffix; the pattern is defined once. Add `core/defineKind.py` to create a registry by kind name and record it in a table (a duplicate kind is an error), and a function to enumerate all kinds, so that tests and the compatibility check can iterate over them.

**Done when.**
- [ ] tests cover register, lookup, listing, duplicates, unknown names (the message lists the names) and invalid names (uppercase, spaces, leading digit, empty);
- [ ] two registries share no state;
- [ ] the kind table can be enumerated and reset in tests.

### P1-07. Component entry and capability vocabulary

**Size:** S | **Needs:** P1-06 | **Spec:** 5.12 | **Status:** done

**Do.** Add `core/ComponentEntry.py` (the record a registry stores: name, builder, configuration schema, description, needs, supports) and `core/CapabilityVocabulary.py` (the documented set of known tags: register a tag with a one-line description, check a tag, list tags by group). Tags are lowercase `group:name` strings. `Registry` validates the tags of an entry against the vocabulary when it registers it; an unknown tag raises `RegistryError` that lists the known tags of the same group. The vocabulary starts empty: each task that needs tags registers them.

**Done when.**
- [ ] tests cover an unknown tag, a duplicate tag registration, a malformed tag, entry validation and sorted listing;
- [ ] an entry without a description is rejected.

### P1-08. Compatibility check

**Size:** M | **Needs:** P1-07 | **Spec:** 3.6, 5.12 | **Status:** done

**Do.** Add the compatibility check to `core`. Participants (the problem, the goal and the components of a policy) are represented by their needs and supports. Every need must be supported by some participant. On failure raise one `CompatibilityError` that lists all unmet needs (not just the first), who needs each, and, for each need, the registered components (looked up in the kind table) that support it. A participant may compute its needs and supports from its own validated configuration.

**Done when.**
- [ ] tests with fake registries: all met; one unmet; several unmet reported together; alternatives listed; a need met by the goal participant; tags refined from configuration;
- [ ] a message snapshot test pins the wording of one failure.

### P1-09. Component configuration and resolution

**Size:** S | **Needs:** P1-07 | **Spec:** 5.12, 10.8 | **Status:** done

**Do.** Add `core/ComponentConfig.py` (Pydantic base class: a `type` field holding the registered name, unknown fields rejected) and a resolver that turns a mapping `{type: name, ...}` and a registry into the entry and the validated configuration. Errors name the registry, the offending field and the available names.

**Done when.**
- [ ] tests: valid input; missing `type`; unknown `type` (lists the names); unknown field; wrong field type.

### P1-10. Nested components and resolved configuration

**Size:** M | **Needs:** P1-09 | **Spec:** 3.4, 5.12 | **Status:** done

**Do.** Let a configuration field hold another component of a declared kind (the field names the registry; a single component or a list of components of that kind), resolved when the parent is validated, to any depth. Add a function returning the fully resolved configuration (a plain JSON-compatible dictionary with every default and every nested `type` written out) and a function rebuilding the validated objects from it. Validation errors carry the full path (for example `policy.surrogate.kernel.nu`).

**Done when.**
- [ ] tests: two levels of nesting; a list of components; the error path for a bad nested field; resolve then validate then resolve gives the same dictionary; the resolved dictionary contains a default that the caller omitted;
- [ ] the resolved dictionary survives `json.dumps` and `json.loads` unchanged.

### P1-11. Configuration files

**Size:** S | **Needs:** P1-10 | **Spec:** 5.12, 10.8 | **Decision:** D-4 | **Status:** done

**Do.** Add loading and dumping of configurations as JSON and YAML (PyYAML, declared as a dependency, D-4) through plain dictionaries, then validation by the resolver. Syntax errors and duplicate keys raise `ConfigError` with the file name and, when available, the line.

**Done when.**
- [ ] tests: JSON and YAML give identical objects; dump then load round trip; syntax error message; duplicate key detection;
- [ ] `pyproject.toml` declares the YAML dependency.

### P1-12. Plugin discovery

**Size:** S | **Needs:** P1-06 | **Spec:** 5.12 | **Decision:** D-10 | **Status:** done

**Do.** Add `loadPlugins`: import every entry point of the group chosen in D-10, plus module names passed explicitly. A failing plugin raises `RegistryError` naming the plugin and the original error. Calling it twice is harmless.

**Done when.**
- [ ] tests with a temporary distribution exposing an entry point whose module registers a component of a test kind;
- [ ] the failure path and idempotence are tested.

### P1-13. Random-number tree

**Size:** S | **Needs:** P1-05 | **Spec:** 3.7, 5.11 | **Status:** done

**Do.** Add `core/RngTree.py`: from one study seed, derive child seeds keyed by (ask index, component path) with `numpy.random.SeedSequence`, hashing the path with a stable function (never Python's randomized `hash`). Expose integer seeds and numpy generators; torch generators are built in `encoding` (P3-02) from these integers. When no seed is given, entropy is drawn from the operating system through `SeedSequence()` and returned so that it can be recorded. No literal derivation constants.

**Done when.**
- [ ] tests: the same inputs give the same stream, also across two processes; a different path or ask index gives a different stream; adding an unrelated path does not change another path's stream; the recorded entropy rebuilds the same tree;
- [ ] the file passes the policy-value check with no marker.

### P1-14. Runtime settings

**Size:** S | **Needs:** P1-09 | **Spec:** 8, 10.8 | **Decision:** D-12 | **Status:** done

**Do.** Add `core/RuntimeConfig.py`: dtype name (`float32` or `float64`), device name and a deterministic-algorithms flag, with the defaults of D-12 and documented fields. It holds names only; converting them to torch objects belongs to `encoding` (P3-02).

**Done when.**
- [ ] tests: defaults; an invalid dtype name is rejected with the field path;
- [ ] importing `core` still imports no machine-learning library (test).

### P1-15. Registry contract tests

**Size:** S | **Needs:** P1-10 | **Spec:** 10.6, 10.11 | **Status:** todo

**Do.** Add `tests/unit/core/test_registry_contract.py`, parametrized over every kind in the kind table. For every registered entry: a valid name; a configuration schema that is a `ComponentConfig` subclass whose every field has a non-empty description; needs and supports that exist in the vocabulary; a callable builder; a non-empty description. Add a completeness test: every module under a kind's package that contains a registration decorator is imported by the package `__init__`. Today the table holds test doubles; later tasks add real kinds and are covered automatically.

**Done when.**
- [ ] the contract test passes with test doubles and fails when a double breaks each rule (one case per rule);
- [ ] the completeness test fails on a decorated module that its `__init__` does not import.

### P1-16. Layering checker

**Size:** M | **Needs:** P1-05, P0-03 | **Spec:** 5.1, 10.5 | **Status:** todo

**Do.** Create `scripts/check_layering.py` (standard library, AST) with its rules in `scripts/layers.toml`: the dependency table of specification 5.1; no `ax`, `torch`, `botorch` or `gpytorch` import in `core` and `domain`; no `ax` import outside `adapters` and the legacy `backends`. Legacy packages (`api`, `backends`, `models`, `analysis`) form a `legacy` group that may import anything and that nothing outside the legacy group, the top-level package and the tests may import. The file lists `known_violations` (for example `domain` importing `persistence`, fixed in P2-02, and the current `acquisition` package importing `models`); entries are removed as they are fixed, and a stale entry fails the check.

**Done when.**
- [ ] tests with fixture trees for each rule, for known-violation handling and for the stale-entry failure;
- [ ] the check passes on the repository and runs in the `lint` job.

### P1-17. Contributor documentation

**Size:** S | **Needs:** P1-03, P1-15, P0-06, P1-16 | **Spec:** 10, 12 | **Status:** todo

**Do.** Update `docs/contributing.md`: add the registry rules (10.6), the policy-value rule and the `# structural:` marker (10.7), the options-object rule (10.8), the new-component checklist (10.14) and the request check (12.2), and say how to run each checker. Keep the existing naming and typography sections.

**Done when.**
- [ ] each section exists and cites the specification section it summarizes;
- [ ] `mkdocs build --strict` passes.

---

## 7. Phase 2: Domain generalization

Make the domain a pure, registry-driven description of problems: parameter and constraint types, outcomes and outcome kinds, trials with a lifecycle, observations, problems and their facts. The old classes that the new design replaces are renamed `Legacy*` so that the old path keeps working until Phase 11.

**Exit criteria.** `planexs.core` and `planexs.domain` import no machine-learning library; the legacy path still works under its `Legacy*` names; parameter and constraint types are registered.

### P2-01. ADR-0012: outcomes, trials and observations

**Size:** S | **Needs:** P1-01 | **Spec:** 5.4 | **Decision:** D-7 | **Status:** todo

**Do.** Record: outcomes with registered kinds (continuous and binary first, D-7); trials with a lifecycle and stable identifiers; observations typed per outcome kind with optional noise; direction and weights live in goals, not in outcomes. Options considered: keep `Metric` (a name plus a direction) as the outcome; free-form dictionaries for observations. Include the table of legal trial-state transitions.

**Done when.**
- [ ] ADR complete and in the navigation;
- [ ] the transition table lists every pair of states and says whether it is legal.

### P2-02. Prefix the legacy types and move the legacy snapshot

**Size:** S | **Needs:** P1-16 | **Spec:** 5.1 | **Status:** todo

**Do.** The new design reuses the names `Trial`, `Objective`, `Metric`, `OutcomeConstraint` and `StudySnapshot`. Rename the current classes (and their files) to `LegacyTrial`, `LegacyObjective`, `LegacyMetric`, `LegacyOutcomeConstraint` and `LegacyStudySnapshot`, update every import, test and example, and move `LegacyStudySnapshot` to `persistence/` (the domain imports a higher layer today). Remove the matching known violation from `scripts/layers.toml`. This is mechanical: no behaviour change.

**Done when.**
- [ ] the legacy test suite passes unchanged apart from the names;
- [ ] `git grep` finds no use of the old names outside the legacy files;
- [ ] the layering check no longer lists the domain-to-persistence violation.

**Not in this task.** Do not redesign any of these classes here.

### P2-03. Remove engine vocabulary from the domain

**Size:** S | **Needs:** P2-02 | **Spec:** 3.9, 5.3 | **Status:** todo

**Do.** Move `Range.toAxKwargs`, `Choice.toAxKwargs`, `Derived.toAxKwargs`, `LinearConstraint.toAxParameterConstraintString` and `LegacyOutcomeConstraint.toAxOutcomeConstraintString` into functions under `backends/ax` (one function per file, next to `toAxSearchSpace.py` and `toAxOptimizationConfig.py`), and update callers and tests. `requires_custom_acquisition_layer` on constraints stays for now because the legacy path uses it; new code must not use it, and P11-03 deletes it.

**Done when.**
- [ ] no file under `domain/` mentions Ax (`git grep -iw ax src/planexs/domain` is empty);
- [ ] the legacy tests pass.

### P2-04. Parameter-type registry

**Size:** M | **Needs:** P1-10, P2-02 | **Spec:** 5.3, 6 | **Status:** todo

**Do.** Create the `parameter` kind: register `range`, `choice` and `derived` under their serialization tags and replace the tag chain in `parameterFromDict` and the serializers of `SearchSpace` by registry lookup. The serialization key is `type` (it was `kind`). The sugar constructors (`Real`, `Integer`, `Categorical`, `Boolean`, `Fixed`) keep working and are not registered. An unknown-type error lists the registered types.

**Done when.**
- [ ] every type round-trips through JSON exactly;
- [ ] a toy type registered inside the test round-trips inside a `SearchSpace`;
- [ ] the unknown-type error is tested and the registry contract test covers the kind;
- [ ] legacy and new tests are updated for the `type` key.

### P2-05. Constraint-type registry

**Size:** S | **Needs:** P2-04 | **Spec:** 5.3, 6 | **Status:** todo

**Do.** Do the same for the `constraint` kind: register `linear` and `nonlinear`, replace the chain in `constraintFromDict`, and use the serialization key `type`.

**Done when.**
- [ ] round trips for both types and for a toy type; the unknown-type error; the contract test covers the kind.

### P2-06. Outcomes and outcome kinds

**Size:** M | **Needs:** P2-01, P1-10, P2-02 | **Spec:** 5.4, 6 | **Decision:** D-7 | **Status:** todo

**Do.** Add `domain/Outcome.py` (name, type, optional description) and the `outcome_kind` kind with `continuous` and `binary` (one file each). A kind validates observed values (continuous: a finite float; binary: a bool or 0 or 1, stored as 0 or 1), says whether a noise standard deviation is meaningful (continuous yes, binary no) and names its capability tag (`outcome_kind:continuous`, `outcome_kind:binary`, registered in the vocabulary here). Outcome serialization goes through the registry.

**Done when.**
- [ ] tests: good and bad values for each kind (NaN, infinity, strings, 2 for binary); the noise rule; round trip; unknown-type error;
- [ ] the contract test covers the kind.

### P2-07. Trial lifecycle

**Size:** S | **Needs:** P2-01, P2-02 | **Spec:** 5.4 | **Status:** todo

**Do.** Add `domain/TrialState.py` (pending, completed, failed, abandoned) and the new `domain/Trial.py`: an immutable record with an integer id, a parameterization, a state, an observation slot (typed in P2-08), JSON-compatible metadata and a failure reason. Transitions are methods that return a new trial and reject illegal moves (anything not in the table of ADR-0012) with a new `TrialStateError`.

**Done when.**
- [ ] tests: every legal transition; every illegal transition raises; serialization round trip; a failed trial requires a reason; a transition never changes the id.

### P2-08. Observations

**Size:** S | **Needs:** P2-06, P2-07 | **Spec:** 5.4 | **Status:** todo

**Do.** Add `domain/Observation.py`: one value per outcome name and an optional noise standard deviation per outcome, validated against a list of outcomes (every outcome present, no extras, each value valid for its kind, noise only where the kind allows it, noise finite and not negative). A completed trial requires an observation; the other states forbid one.

**Done when.**
- [ ] tests for each rule and for the trial and observation coupling; round trip.

### P2-09. Pure expression evaluator

**Size:** S | **Needs:** P2-05 | **Spec:** 3.9, 5.3 | **Status:** todo

**Do.** Add a pure-Python evaluator of validated expressions (the `math` module, the same function allowlist as `validateExpression`) and give `NonlinearConstraint` a way to say whether a parameterization of natural values satisfies it, so that constraints can be checked without torch. A result that is undefined (for example `sqrt` of a negative) counts as infeasible, as in the compiled form.

**Done when.**
- [ ] a test asserts that the evaluator, `validateExpression` and `compileExpression` accept the same function names, and that evaluator and compiled form agree on sample points;
- [ ] no machine-learning import in the new file.

### P2-10. Problem and problem facts

**Size:** S | **Needs:** P2-06, P2-09, P2-04, P2-05 | **Spec:** 5.3, 5.12 | **Status:** todo

**Do.** Add `domain/Problem.py`: a search space plus at least one outcome (unique names), with a function that validates a parameterization (legal values, inactive conditional parameters absent, every constraint satisfied on natural values) and a pure-data `ProblemFacts` summary (number of outcomes, outcome types, parameter and constraint types present, whether conditional parameters, equality constraints or categorical parameters exist). Mapping facts to capability tags is not done here: it needs the vocabulary and belongs to the policies layer (P6-05).

**Done when.**
- [ ] tests: each fact; each validation error; a feasible and an infeasible parameterization; serialization round trip.

### P2-11. Lazy top-level exports

**Size:** S | **Needs:** P2-02 | **Spec:** 8, 3.9 | **Status:** todo

**Do.** `planexs/__init__.py` imports every layer eagerly, so `import planexs.domain` imports torch. Make the re-exports lazy (a module-level `__getattr__`, with imports under `TYPE_CHECKING` for type checkers) and keep `__all__` and `__version__`.

**Done when.**
- [ ] importing `planexs` leaves `torch` out of `sys.modules` (test in a subprocess);
- [ ] every name in `__all__` still imports (test), and mypy strict still sees the names.

### P2-12. Domain and core purity tests

**Size:** S | **Needs:** P2-11, P2-03 | **Spec:** 3.9, 10.11 | **Status:** todo

**Do.** Add tests that import `planexs.core` and `planexs.domain` in a fresh subprocess and assert that `torch`, `botorch`, `gpytorch` and `ax` are not in `sys.modules`, and that `planexs.core` imports no other planexs layer.

**Done when.**
- [ ] both tests pass, and both fail when a forbidden import is added to a scratch module (state the evidence).

---

## 8. Phase 3: Encoding and surrogates

Move the encoder under `encoding/` with its policy values in configuration. Then replace the hard-wired surrogate by the `surrogate` kind: nested kernel, likelihood, outcome-transform and fitter kinds, a continuous GP builder, a binary GP classifier, a `Surrogate` wrapper and a cache. The legacy surrogate stays in place until parity tests pass.

**Exit criteria.** A `Surrogate` can be built from configuration for a continuous and for a binary outcome and used with no study; a parity test matches the legacy surrogate; the encoder and sampler files hold no policy-value literal.

### P3-01. Move the encoder and the expression compiler

**Size:** S | **Needs:** P1-16 | **Spec:** 5.1 | **Status:** todo

**Do.** `git mv` `models/SearchSpaceEncoder.py`, `models/ParameterEncoding.py` and `models/compileExpression.py` into `encoding/`, update imports, move their tests to `tests/unit/encoding/` and update `scripts/layers.toml`. No behaviour change.

**Done when.**
- [ ] the full test suite passes and the layering check passes;
- [ ] `models/` no longer holds these files.

### P3-02. Torch runtime helpers

**Size:** S | **Needs:** P1-13, P1-14, P3-01 | **Spec:** 3.4, 8 | **Status:** todo

**Do.** In `encoding/`, add the helpers that turn `RuntimeConfig` and `RngTree` seeds into torch objects (dtype, device, a seeded `torch.Generator`) and apply the deterministic-algorithms flag, restoring the previous value on exit. The encoder fixes `torch.double` at module level today: make the dtype an input taken from the runtime settings.

**Done when.**
- [ ] tests: float32 and float64 encoders; the generator is reproducible from a seed; the deterministic flag is restored;
- [ ] no module-level dtype constant remains in `encoding/`.

### P3-03. Encoder policy values to configuration

**Size:** S | **Needs:** P3-01, P3-02, P1-09, P0-06 | **Spec:** 3.4 | **Status:** todo

**Do.** Introduce `EncoderConfig` and move into it: the limit on enumerated rounding neighbours (8), the grid tolerance (1e-9, used twice), the attempts factor for the random subset of categorical assignments (20) and the ordering rule applied when a choice leaves `is_ordered` unset (today: numeric with more than two values; make it a named rule selected in the configuration). Mark the three unit-cube midpoint literals `# structural: unit-cube midpoint`. Delete the matching baseline lines.

**Done when.**
- [ ] no numeric literal other than a structural one remains in the encoder files;
- [ ] a test per configured value shows that changing it changes behaviour, and the defaults reproduce the old behaviour (existing tests pass unchanged);
- [ ] the baseline lines are removed.

### P3-04. Move constraint encoding and feasible sampling

**Size:** S | **Needs:** P3-01 | **Spec:** 5.1 | **Status:** todo

**Do.** `git mv` `acquisition/EncodedConstraints.py`, `acquisition/sampleFeasibleEncoded.py` and `acquisition/toBotorchNonlinearConstraints.py` into `encoding/`, update imports, tests and `scripts/layers.toml`. No behaviour change.

**Done when.**
- [ ] the full test suite passes and the layering check passes.

### P3-05. Sampling and constraint policy values to configuration

**Size:** S | **Needs:** P3-04, P3-02, P1-09 | **Spec:** 3.4 | **Status:** todo

**Do.** Introduce `FeasibleSamplingConfig` (rounds, oversampling factor, minimum draw, polytope burn-in and thinning, and the feasibility tolerance that `EncodedConstraints.isFeasible` takes by default) and pass it explicitly. Move the two long comments that justify burn-in and thinning into the field descriptions. Delete the matching baseline lines.

**Done when.**
- [ ] no module constant or signature literal remains in these files;
- [ ] tests: each field changes behaviour, and the defaults reproduce the old behaviour;
- [ ] every field description states its meaning and unit.

### P3-06. Expression limits to configuration

**Size:** S | **Needs:** P2-05, P1-09 | **Spec:** 3.4 | **Status:** todo

**Do.** Introduce `ExpressionLimitsConfig` (maximum length 1000 and nesting depth 200 today) and let `NonlinearConstraint` carry an optional `limits` field that is serialized only when it differs from the default. `validateExpression` takes the limits as an argument. Delete the matching baseline lines.

**Done when.**
- [ ] tests: an expression rejected under the default limits is accepted under larger ones and survives a round trip; the default round trip is unchanged;
- [ ] the baseline lines are removed.

### P3-07. Surrogate contract and registry

**Size:** S | **Needs:** P1-10, P1-08, P3-02, P2-10 | **Spec:** 5.6, 6 | **Status:** todo

**Do.** Create the `surrogates` package and the `surrogate` kind: the abstract builder (registered name, configuration schema, a build method that receives a `SurrogateBuildContext` and returns a `BuiltSurrogate`), the context (problem, encoder, train inputs, outcome tensors, optional noise tensors, random generator, runtime settings), `FitStatus` (observation count, low-data flag, warnings), the registry and `registerSurrogate`. Register the vocabulary tags `posterior:gaussian`, `posterior:latent_gaussian`, `posterior:marginal_normal`, `outputs:single`, `outputs:multi`, `noise:known` and `noise:inferred`.

**Done when.**
- [ ] tests with a fake builder: registration, tag validation, immutability of the context;
- [ ] the contract test covers the kind.

### P3-08. Trials to tensors

**Size:** S | **Needs:** P2-08, P3-07, P3-03 | **Spec:** 5.4, 5.6 | **Status:** todo

**Do.** Add to `surrogates/` the function that turns completed trials into train inputs, outcome tensors (one column per outcome) and noise tensors, using the encoder. Pending, failed and abandoned trials are excluded. Partial noise information follows `NoisePolicyConfig` (`ignore_with_warning` or `error`; the first is today's behaviour).

**Done when.**
- [ ] tests: order preserved; excluded states; both noise policies; a binary outcome;
- [ ] an equivalence test against the legacy `encodeTrials` on the same data.

### P3-09. Surrogate wrapper and predictions

**Size:** M | **Needs:** P3-07 | **Spec:** 5.6, 3.11 | **Status:** todo

**Do.** Add `surrogates/Surrogate.py`, binding a native model to the encoder and the outcomes: predict at parameterizations (returns a `Prediction` with a mean and a standard deviation per outcome, the space they live in, latent or response, and whether observation noise is included), draw posterior samples, return the native model, report the fit status and the outcomes. Options (noise inclusion, sample count) travel in an options object, not in defaulted arguments. `Prediction` replaces `PredictionResult`; `variance` stays a cheap derived property. The wrapper never imports a concrete model class and works with a fake model in tests.

**Done when.**
- [ ] tests with a fake native model: shapes, space flag, noise flag, sampling shapes and determinism;
- [ ] the property naming rule holds (cheap properties in snake_case, costly ones in camelCase).

### P3-10. Kernel kind

**Size:** S | **Needs:** P3-07 | **Spec:** 5.6, 6 | **Status:** todo

**Do.** Add the `kernel` kind with `botorch_default_v1` (leave the choice to the engine), `matern_v1` (smoothness `nu`, ARD flag) and `rbf_v1` (ARD flag). Builders receive the number of encoded columns and return a native GPyTorch kernel (or nothing, for the engine default).

**Done when.**
- [ ] three entries with described configurations; each builds on a 3-column dummy encoder;
- [ ] an unsupported `nu` is rejected with the field path;
- [ ] the contract test covers the kind.

### P3-11. Likelihood kind

**Size:** S | **Needs:** P3-07, P2-06 | **Spec:** 5.6, 6 | **Status:** todo

**Do.** Add the `likelihood` kind with `gaussian_v1` (inferred noise with an optional lower bound, and a floor `min_observation_variance` for known noise, which is `_MIN_OBSERVATION_VARIANCE = 1e-10` in the legacy code) and `bernoulli_v1`. Entries declare which outcome kinds they support.

**Done when.**
- [ ] tests: inferred noise with and without a floor; the known-noise floor is applied; the binary likelihood builds; the declared outcome kinds are enforced;
- [ ] the contract test covers the kind.

### P3-12. Outcome-transform kind

**Size:** S | **Needs:** P3-07 | **Spec:** 5.6, 6 | **Status:** todo

**Do.** Add the `outcome_transform` kind with `standardize_v1` (the floor on the standard deviation is configuration), `log_v1` and `none_v1`. A surrogate configuration may hold a list of transforms applied in order.

**Done when.**
- [ ] tests: each transform maps and un-maps predictions consistently on a toy model; chaining order;
- [ ] the contract test covers the kind.

### P3-13. MAP fitter

**Size:** S | **Needs:** P3-07 | **Spec:** 5.6, 6 | **Status:** todo

**Do.** Add the `fitter` kind with `map_v1`, which fits an exact GP's hyperparameters by maximizing the marginal likelihood with the engine's priors (a MAP fit, as today) through the engine's fitting routine. Expose as configuration fields the options of that routine that exist in the installed version (verify, do not assume).

**Done when.**
- [ ] a test fits a toy GP and checks that the marginal likelihood improved over its initial value;
- [ ] each exposed option has a test showing that it is passed through; the contract test covers the kind.

### P3-14. single_task_gp_v1 surrogate

**Size:** M | **Needs:** P3-08, P3-09, P3-10, P3-11, P3-12, P3-13 | **Spec:** 5.6, 3.4 | **Status:** todo

**Do.** Register `single_task_gp_v1`. Its configuration nests `kernel` (default `botorch_default_v1`), `likelihood` (`gaussian_v1`), `outcome_transform` (`standardize_v1`) and `fitter` (`map_v1`), and adds `min_observations` (`MIN_TRIALS_FOR_MODEL = 2` today), `low_data_threshold` (`minimum_points_for_free_fit = 5` today) and the noise policy. It builds a BoTorch `SingleTaskGP` (several outcomes as independent outputs, as today), fits it, returns the model with a `FitStatus`, raises `ModelNotReadyError` below `min_observations` and sets the low-data flag below the threshold. Supports `posterior:gaussian`, `posterior:marginal_normal`, `outputs:single`, `outputs:multi`, `noise:known`, `noise:inferred` and `outcome_kind:continuous`.

**Done when.**
- [ ] a parity test fits the legacy `buildSurrogateModel` and this builder on the same data and compares posterior means and variances at fixed points within tolerances stated in the test;
- [ ] tests for `ModelNotReadyError` and for the low-data flag;
- [ ] the new files contain no numeric literal outside configuration schemas.

### P3-15. Variational fitter

**Size:** S | **Needs:** P3-13 | **Spec:** 5.6, 6 | **Status:** todo

**Do.** Add `variational_elbo_v1` to the `fitter` kind: it maximizes the variational ELBO of a GPyTorch approximate GP with an optimizer loop whose iteration count, learning rate and optimizer name are configuration fields, using the seeded generator, and leaves the model in evaluation mode.

**Done when.**
- [ ] tests: the ELBO increases on a toy problem; each field changes behaviour; the same seed reproduces the result;
- [ ] the contract test covers the entry.

### P3-16. variational_gp_classifier_v1 surrogate

**Size:** M | **Needs:** P3-09, P3-10, P3-11, P3-15, P3-08 | **Spec:** 5.6, FR20, FR24 | **Status:** todo

**Do.** Register a surrogate for one binary outcome: an approximate GP over the encoded inputs with the `bernoulli_v1` likelihood, inducing points equal to the training inputs unless `n_inducing` is set, and nested `kernel` and `fitter` (default `variational_elbo_v1`). Verify what the installed BoTorch offers (`SingleTaskVariationalGP` or an equivalent); if nothing fits, build on GPyTorch's `ApproximateGP`. The `Surrogate` wrapper must return predictions in the latent space (mean and standard deviation of the latent function) and in the response space (probability, through the inverse link), and convert a response-space level into a latent threshold. Register the tag `link:probit` and support `posterior:latent_gaussian`, `posterior:marginal_normal`, `outputs:single` and `outcome_kind:binary`.

**Done when.**
- [ ] on a synthetic two-dimensional dataset labelled by a known boundary, response-space predictions on held-out points fall on the correct side of 0.5 with an accuracy stated in the test;
- [ ] predictions say which space they describe, and the level conversion is tested against the inverse link;
- [ ] training is reproducible from a seed.

### P3-17. Surrogate cache

**Size:** S | **Needs:** P3-09 | **Spec:** 5.6 | **Status:** todo

**Do.** Add `surrogates/SurrogateCache.py`: get-or-build keyed by the history version and the resolved surrogate configuration. Any new trial or any configuration change invalidates; only the latest entry is kept.

**Done when.**
- [ ] tests: hit and miss counts; invalidation on a new trial; a different configuration rebuilds; the builder is not called twice for the same key.

---

## 9. Phase 4: Goals and designs

Introduce the `goal` kind and its first three goal types, which replace the objective as the centre of the model, and the `design` kind with its first two designs. Goals come first because acquisitions and policies read them.

**Exit criteria.** Goals of type optimize, level set and explore validate against a problem and declare their tags; two designs produce constraint-satisfying points reproducibly.

### P4-01. ADR-0013: goals, policies and recipes

**Size:** S | **Needs:** P1-01 | **Spec:** 5.7, 5.9 | **Status:** todo

**Do.** Record: goals as typed configuration with tags, recipes and default analyses; composed and opaque policies behind one contract; recipes tried in order and the first compatible one used; the resolved policy stored; no later switching of policy. This supersedes the 'switch with a warning' behaviour of ADR-0006 (already marked superseded by ADR-0009).

**Done when.**
- [ ] ADR complete and in the navigation.

### P4-02. Goal contract and registry

**Size:** S | **Needs:** P4-01, P1-10, P2-10 | **Spec:** 5.7, 6 | **Status:** todo

**Do.** Create the `goals` package and the `goal` kind: the abstract goal (configuration schema; validation against a problem; its capability tags; recipes as an ordered list of policy configurations; default analyses), the registry and `registerGoal`. Register the vocabulary tags `goal:optimize`, `goal:level_set`, `goal:explore`, `objectives:single` and `objectives:multi`.

**Done when.**
- [ ] tests with a fake goal: validation against a problem (missing outcome, unsuitable outcome kind); tags;
- [ ] the contract test covers the kind.

### P4-03. optimize_v1 goal

**Size:** M | **Needs:** P4-02, P2-10 | **Spec:** 5.7, FR2, FR15 | **Status:** todo

**Do.** Add the `optimize_v1` goal: outcomes with a direction each, optional scalarizing weights (one per outcome), optional outcome constraints (outcome, comparator, bound; bounds relative to a baseline stay out of scope, as today), validated against the problem. It supports `goal:optimize` and `objectives:single` (one outcome, or several with weights) or `objectives:multi` (several without weights), and needs `outcome_constraints:supported` when it has outcome constraints. It replaces the roles of `LegacyObjective` and `LegacyOutcomeConstraint`. Recipes are added in P6-07.

**Done when.**
- [ ] tests: single, several and weighted outcomes; a constraint on an unknown outcome is rejected; a weight-count mismatch is rejected; tags in each case; round trip;
- [ ] a conversion test builds the same goal from a `LegacyObjective` fixture, to keep parity data for later tasks.

### P4-04. level_set_v1 goal

**Size:** S | **Needs:** P4-02 | **Spec:** 5.7, FR20 | **Status:** todo

**Do.** Add the `level_set_v1` goal: one outcome and a `level`. For a continuous outcome the level is required and is in the outcome's units; for a binary outcome it is a probability and has a documented default. It supports `goal:level_set`. Recipes are added in P6-08.

**Done when.**
- [ ] tests: the level is required for a continuous outcome and defaulted for a binary one; an unknown outcome is rejected; round trip.

### P4-05. explore_v1 goal

**Size:** S | **Needs:** P4-02 | **Spec:** 5.7, FR21 | **Status:** todo

**Do.** Add the `explore_v1` goal: learn the named outcomes (all of them by default) as well as possible, with no direction. It supports `goal:explore`.

**Done when.**
- [ ] tests: default outcomes; an unknown outcome is rejected; round trip.

### P4-06. Design contract and sobol_v1

**Size:** S | **Needs:** P3-05, P3-02, P1-10, P2-10 | **Spec:** 5.8, 6 | **Status:** todo

**Do.** Create the `designs` package and the `design` kind (abstract design, registry, `registerDesign`) and register `sobol_v1`: scrambled Sobol points in the encoded space, constraint-satisfying through the feasible sampler, with `n_points` and `scramble` as configuration, and an option to count existing observations so that a warm start shortens the initial phase.

**Done when.**
- [ ] tests: every point is feasible and decodes to a legal parameterization (on a space with linear and nonlinear constraints); the same seed gives the same points; counting existing observations reduces the request;
- [ ] the contract test covers the kind.

### P4-07. random_v1 design

**Size:** S | **Needs:** P4-06 | **Spec:** 5.8, 3.10 | **Status:** todo

**Do.** Register `random_v1`: uniform draws in the encoded space with the same constraint handling and options as `sobol_v1`. It is the second design (rule of two).

**Done when.**
- [ ] the same tests as `sobol_v1`, plus a test that the two designs differ on the same seed (they are not aliases).

---

## 10. Phase 5: Acquisitions and optimizers

Introduce the `acquisition` and `acquisition_optimizer` kinds. Port what the legacy layer does (the exploration/exploitation blend, the constrained and mixed-space optimization) as registered components, add the standard engine acquisitions and the level-set and exploration acquisitions, and make outcome constraints reach the acquisition. The legacy classes stay until parity tests pass.

**Exit criteria.** Every acquisition of the legacy layer exists as a registered builder; level-set and exploration acquisitions exist; three optimizers exist; each is selectable by name and declares truthful tags.

### P5-01. Acquisition contract and registry

**Size:** S | **Needs:** P3-09, P4-02, P1-08 | **Spec:** 5.8, 6 | **Status:** todo

**Do.** Create the `acquisition` package and the `acquisition` kind: the abstract builder (registered name, configuration schema, a build method receiving an `AcquisitionBuildContext` and returning a native acquisition function), the context (surrogate, goal, pending points, a reference set of encoded points, encoder, random generator), the registry and `registerAcquisition`. A builder declares through a class-level flag whether it needs a reference set, its size comes from its own configuration, and the policy draws the set with the feasible sampler. Register the tags `acquisition:differentiable`, `pending:supported` and `outcome_constraints:supported`.

**Done when.**
- [ ] tests with a fake builder: registration, tag validation, immutability of the context;
- [ ] the contract test covers the kind.

### P5-02. log_ei_v1 acquisition

**Size:** S | **Needs:** P5-01, P4-03 | **Spec:** 5.8, FR4 | **Status:** todo

**Do.** Register the analytic log expected improvement for a one-objective `optimize` goal. Configuration: how the incumbent is chosen (`best_observed` or `best_posterior_mean`). When the problem has several outcomes, it selects the objective with a posterior transform. It needs `posterior:gaussian`, `objectives:single` and `goal:optimize`, and supports `acquisition:differentiable`.

**Done when.**
- [ ] tests: the builder returns a native acquisition function; both incumbent rules are honoured; the direction (minimize or maximize) is respected;
- [ ] a goal with outcome constraints fails the compatibility check against this entry, with a message that names an entry that works.

### P5-03. qlog_nei_v1 acquisition

**Size:** S | **Needs:** P5-01, P4-03 | **Spec:** 5.8, FR10, FR15 | **Status:** todo

**Do.** Register the Monte Carlo log noisy expected improvement: it handles noisy data, pending points and batches, and passes the goal's outcome constraints to the acquisition as feasibility constraints (a limit of the legacy layer, which warns that it does not enforce them). Configuration: sample count and baseline pruning. It needs `posterior:gaussian`, `objectives:single` and `goal:optimize`, and supports `pending:supported`, `outcome_constraints:supported` and `acquisition:differentiable`.

**Done when.**
- [ ] tests: pending points change the value; direction respected; a candidate predicted to violate an outcome constraint scores lower than a feasible one on a toy surrogate;
- [ ] the sample count is configuration.

### P5-04. ucb_v1 acquisition

**Size:** S | **Needs:** P5-01, P4-03 | **Spec:** 5.8 | **Status:** todo

**Do.** Register the analytic upper confidence bound with `beta` as configuration (the default lives in the schema, with its rationale). The direction is respected. It needs `posterior:marginal_normal`, `objectives:single` and `goal:optimize`, and supports `acquisition:differentiable`.

**Done when.**
- [ ] tests: a higher `beta` raises the value of an uncertain point relative to a certain one; minimize and maximize.

### P5-05. qlog_nehvi_v1 acquisition

**Size:** M | **Needs:** P5-01, P4-03 | **Spec:** 5.8, FR2 | **Status:** todo

**Do.** Register the multi-objective log noisy expected hypervolume improvement: the reference point comes from configuration (explicit values, or a named inference rule), the partitioning from the model's own data; pending points and outcome constraints are supported. It needs `posterior:gaussian`, `objectives:multi` and `goal:optimize`, and supports `pending:supported`, `outcome_constraints:supported` and `acquisition:differentiable`.

**Done when.**
- [ ] tests: on a toy two-outcome surrogate a point that extends the Pareto front scores higher than a dominated one; the reference-point rules; directions;
- [ ] outcome constraints are passed through.

### P5-06. neg_integrated_variance_v1 acquisition

**Size:** S | **Needs:** P5-01, P4-05 | **Spec:** 5.8, FR21 | **Status:** todo

**Do.** Register the engine's negative integrated posterior variance (pure exploration, no direction), with the number of Monte Carlo points as configuration, drawn from the encoder. It needs `posterior:gaussian` and `goal:explore`, and supports `pending:supported`.

**Done when.**
- [ ] tests: on a one-dimensional toy surrogate the value is higher in an unexplored region than next to an observation; the point count is configuration.

### P5-07. exploration_exploitation_v1 acquisition

**Size:** M | **Needs:** P5-01, P4-03 | **Spec:** 5.8, FR5, 3.4 | **Status:** todo

**Do.** Port `ExplorationExploitationAcquisition` as a registered builder (FR5). The `alpha` weight, the size of the normalization reference set, the Monte Carlo sample count and the minimum span are configuration fields. Several outcomes with goal weights use the scalarizing posterior transform, as today. It needs `posterior:gaussian`, `objectives:single` and `goal:optimize`, and supports `pending:supported` and `acquisition:differentiable`.

**Done when.**
- [ ] a parity test compares the values of the legacy class and the ported one at fixed points on a fixture surrogate, for several `alpha` values, within tolerances stated in the test;
- [ ] no numeric literal other than structural ones in the new files (axis indices carry markers);
- [ ] an `alpha` outside [0, 1] is rejected with the field path.

### P5-08. exploration_exploitation_multi_v1 acquisition

**Size:** M | **Needs:** P5-07 | **Spec:** 5.8, FR5 | **Status:** todo

**Do.** Port `MultiObjectiveExplorationExploitationAcquisition` and `modelParetoFront` the same way, as `exploration_exploitation_multi_v1`. It needs `posterior:gaussian`, `objectives:multi` and `goal:optimize`.

**Done when.**
- [ ] a parity test as in P5-07 on a two-outcome fixture;
- [ ] no stray literals; the contract test covers the entry.

### P5-09. straddle_v1 acquisition

**Size:** M | **Needs:** P5-01, P4-04, P3-16 | **Spec:** 5.8, FR20 | **Status:** todo

**Do.** Write the straddle acquisition (a confidence-weighted distance to the level: high where the model is uncertain and close to the level) as a native BoTorch analytic acquisition class, registered as `straddle_v1` with `beta` as configuration. The level comes from the goal; for a surrogate in latent space the level is converted by the surrogate (P3-16). It needs `posterior:marginal_normal` and `goal:level_set`, and supports `acquisition:differentiable`.

**Done when.**
- [ ] a numerical test of the closed form against a direct evaluation of its definition on a grid;
- [ ] tests: the score is highest at the level where the model is uncertain, lower far from the level, and lower where the model is certain; `beta` changes the trade-off; the score is symmetric around the level (it has no direction);
- [ ] it works with both the exact GP and the classifier surrogate in a compatibility check.

### P5-10. Engine active-learning acquisitions

**Size:** S | **Needs:** P5-01, P4-05 | **Spec:** 5.8, 3.5, 10.9 | **Status:** todo

**Do.** List the active-learning acquisitions that the installed BoTorch provides (for example Bayesian active learning by disagreement and the statistical-distance variant) by inspecting the installed package, and register one builder per class that exists, each with its configuration. If a class named here does not exist in the installed version, say so in the report instead of inventing it.

**Done when.**
- [ ] the report lists the classes found and the entries registered;
- [ ] each entry has a test that builds the acquisition for a toy surrogate;
- [ ] declared needs are tested (an entry that needs a Gaussian posterior fails the compatibility check against the classifier surrogate).

### P5-11. Optimizer contract and registry

**Size:** S | **Needs:** P5-01 | **Spec:** 5.8, 6 | **Status:** todo

**Do.** Create the `optimizers` package and the `acquisition_optimizer` kind: the abstract optimizer (registered name, configuration schema, an optimize method receiving the acquisition, encoder, constraints, number of points, pending points and generator, and returning encoded points that satisfy the constraints), the registry and `registerOptimizer`. Register the tags `space:categorical`, `space:conditional`, `constraints:linear`, `constraints:equality` and `constraints:nonlinear`.

**Done when.**
- [ ] tests with a fake optimizer; tag validation;
- [ ] the contract test covers the kind.

### P5-12. botorch_relaxed_v1 optimizer

**Size:** M | **Needs:** P5-11, P3-05 | **Spec:** 5.8, FR9, FR10, FR17 | **Status:** todo

**Do.** Port the continuous part of `AlphaAcquisitionStrategy._optimizeOne` and `_resolveCandidate`: maximize the acquisition with the engine's optimizer on the relaxed encoding, with linear and nonlinear constraints and feasible starting points; snap; try neighbouring roundings when snapping breaks a constraint; then fall back to the best of a pool of random feasible points. Batches are built sequentially, each chosen point becoming pending for the next. Restarts, raw samples, the iteration cap, the fallback pool size and the sampling settings (`FeasibleSamplingConfig`) are configuration. It supports `constraints:linear`, `constraints:equality`, `constraints:nonlinear` and `space:conditional`, and needs `acquisition:differentiable`.

**Done when.**
- [ ] on a fixture problem with a nonlinear constraint, every returned point is feasible and its acquisition value is within a stated tolerance of the legacy result for the same acquisition;
- [ ] a batch of 3 contains 3 distinct points;
- [ ] the fallback path is exercised by a test that forces a rounding failure, and it logs a warning.

### P5-13. botorch_mixed_v1 optimizer

**Size:** M | **Needs:** P5-12 | **Spec:** 5.8, FR1 | **Status:** todo

**Do.** Add categorical handling by wrapping the previous optimizer: enumerate every assignment of the unordered choices, or a random subset of configurable size, through the engine's mixed optimizer. It supports additionally `space:categorical`. Conditional parameters stay as in the relaxed optimizer (optimized as if active and dropped on decoding).

**Done when.**
- [ ] tests on a space with a categorical and an integer parameter: the best categorical assignment is found on a toy acquisition; the subset size is configuration and a seed makes the subset reproducible;
- [ ] decoded points never contain inactive conditional parameters.

### P5-14. random_candidates_v1 optimizer

**Size:** S | **Needs:** P5-11, P3-05 | **Spec:** 5.8, 3.10 | **Status:** todo

**Do.** Register a gradient-free optimizer that scores a configurable number of feasible random candidates and returns the best (sequentially for batches). It supports all the problem-feature tags registered in P5-11 and needs no differentiability, so it works with acquisitions that are not differentiable. It is the second optimizer family (rule of two).

**Done when.**
- [ ] tests: the result is feasible; it beats the median of its own candidates; it is reproducible from a seed;
- [ ] it works with a non-differentiable fake acquisition that `botorch_relaxed_v1` refuses at the compatibility check.

---

## 11. Phase 6: Policies and the Study

Introduce the `policy` kind, the composed policy, the goals' default recipes and the need-resolution step, then build the `Study` facade: construction, ask, tell, fail, abandon, attach, seeds, surrogate access and configuration round trips. The legacy facade stays until Phase 11.

**Exit criteria.** A `Study` runs ask and tell with the default recipe of each goal type, handles pending, failed and abandoned trials, is deterministic for a seed, exposes its surrogate and its resolved configuration, and works with an opaque policy.

### P6-01. Policy contract and study-state view

**Size:** S | **Needs:** P1-10, P2-07, P2-10, P4-02 | **Spec:** 5.9, 6 | **Status:** todo

**Do.** Create the `policies` package and the `policy` kind: the abstract policy (registered name, configuration schema, a suggest operation, a notification for each trial state change, export and import of opaque state), the read-only `StudyStateView` it receives (problem, goal, trials in every state, history version, ask index, random sources) and a `Suggestion` (a parameterization plus metadata such as the phase that produced it). Registry and `registerPolicy`.

**Done when.**
- [ ] tests with a fake policy: registration; the view is read-only (mutation attempts fail); state export and import round trip;
- [ ] the contract test covers the kind.

### P6-02. Composed policy configuration

**Size:** S | **Needs:** P6-01, P4-06, P5-11, P5-01, P3-07 | **Spec:** 5.9, 5.12 | **Status:** todo

**Do.** Add the configuration of `composed_v1`: nested `design`, `surrogate`, `acquisition` and `acquisition_optimizer`; the switching rule (`model_phase_start`: the number of completed trials from which the model phase begins, by default the design's own point count); and the duplicate rule. No behaviour yet.

**Done when.**
- [ ] tests: nested resolution to concrete entries; an unknown nested type gives an error path; the resolved dictionary is written out in full.

### P6-03. Composed policy: design phase

**Size:** S | **Needs:** P6-02 | **Spec:** 5.9 | **Status:** todo

**Do.** Implement `composed_v1.suggest` for the design phase: draw the requested number of points from the design with a generator derived for the ask, skip duplicates of existing and pending parameterizations according to the duplicate rule, and tag each suggestion with its phase.

**Done when.**
- [ ] tests with the real `sobol_v1`: points are legal and feasible; no duplicates of pending points; deterministic given the study random source; the phase tag is `design`.

### P6-04. Composed policy: model phase

**Size:** M | **Needs:** P6-03, P3-17, P5-12 | **Spec:** 5.9, FR10 | **Status:** todo

**Do.** Implement the model phase: when enough completed trials exist (`model_phase_start`), obtain the surrogate through the cache, build the acquisition (pending points included, and a reference set when the acquisition asks for one), optimize it, decode and validate the points against the problem, and tag them with phase `model`. If the surrogate raises `ModelNotReadyError`, fall back to the design phase and log it at WARNING. If the acquisition does not support pending points while pending trials exist or several points are requested, log a WARNING.

**Done when.**
- [ ] tests with a fake surrogate, acquisition and optimizer registered in the test: orchestration order, pending points passed through, the switch at `model_phase_start`, the fallback path, the pending warning;
- [ ] an integration test with the real default components on a two-dimensional problem returns legal, feasible points.

### P6-05. Need resolution

**Size:** S | **Needs:** P6-02, P1-08, P2-10 | **Spec:** 5.12, FR25 | **Status:** todo

**Do.** Add to `policies/` the mapping from `ProblemFacts` and the goal to capability needs (categorical parameters need `space:categorical`, equality constraints need `constraints:equality`, several outcomes need `outputs:multi`, a binary outcome needs `outcome_kind:binary`, and so on), and the function that gathers the participants of a policy and runs the compatibility check, returning the resolved policy description.

**Done when.**
- [ ] tests per fact;
- [ ] an incompatible combination (a relaxed-only optimizer on a categorical space) fails with a message that names the optimizer that would work; a message snapshot test pins the wording.

### P6-06. random_policy_v1

**Size:** S | **Needs:** P6-01, P4-06 | **Spec:** 5.9, 3.10, FR27 | **Status:** todo

**Do.** Register an opaque policy that draws from a design (selected by name in its configuration) and ignores observations. It is the second policy implementation (rule of two) and the baseline for use-case tests.

**Done when.**
- [ ] tests: suggestions are legal and feasible; it honours pending points; its state export is empty and round-trips;
- [ ] the contract test covers the entry.

### P6-07. Recipes for optimize goals

**Size:** S | **Needs:** P6-05, P3-14, P4-03, P4-06, P5-03, P5-05, P5-13 | **Spec:** 5.7, 5.9, FR4 | **Status:** todo

**Do.** Give the `optimize_v1` goal its recipes, in order of preference: for one objective, `composed_v1` with `sobol_v1`, `single_task_gp_v1`, `qlog_nei_v1` and `botorch_mixed_v1`; for several unweighted objectives the same with `qlog_nehvi_v1`. The numbers of a recipe (for example the initial points per dimension) live in a `...RecipeConfig` schema, not in the recipe code.

**Done when.**
- [ ] tests: each case picks the expected recipe; the chosen recipe is the first compatible one; incompatibility with the problem moves on to the next recipe;
- [ ] the policy-value check passes on the new files.

### P6-08. Recipes for level-set and explore goals

**Size:** S | **Needs:** P6-05, P3-14, P3-16, P4-04, P4-05, P4-06, P5-06, P5-09, P5-13 | **Spec:** 5.7, 5.9, FR20, FR21 | **Status:** todo

**Do.** Recipes: for a continuous level-set goal, `composed_v1` with `sobol_v1`, `single_task_gp_v1`, `straddle_v1` and `botorch_mixed_v1`; for a binary level-set goal the same with `variational_gp_classifier_v1`; for the explore goal, `neg_integrated_variance_v1`. The numbers live in a recipe configuration schema.

**Done when.**
- [ ] tests: each goal and outcome kind picks the expected recipe; a level-set goal on a binary outcome never selects the exact GP (compatibility).

### P6-09. Study construction and history

**Size:** M | **Needs:** P6-07, P6-08, P2-10, P4-02, P1-13 | **Spec:** 5.2, 5.9 | **Status:** todo

**Do.** Create the `study` package and `Study`, built from a problem, a goal and, optionally, a policy configuration, a seed and runtime settings. Construction validates the goal against the problem, picks the recipe when no policy is given, runs the compatibility check, resolves the full configuration and creates the policy. The study owns the history (an append-only list of trials with ids from a counter, and a history version that changes at each state change). Nothing else yet.

**Done when.**
- [ ] tests: default recipe selection; an explicit policy; an incompatible policy raises `CompatibilityError`; the resolved configuration is complete and JSON-compatible; ids are unique and increasing; the history version changes only on state changes.

### P6-10. Study ask

**Size:** M | **Needs:** P6-09, P6-04, P6-03 | **Spec:** 5.2, FR10, FR23 | **Status:** todo

**Do.** Add `ask(n)`: give the policy a `StudyStateView`, validate each returned parameterization against the problem (an illegal one raises an error that names the policy), register the points as pending trials with new ids, notify the policy, increment the ask counter and return the trials. Pending trials appear in the next view.

**Done when.**
- [ ] tests with the real default policy and with a fake one: batches; two asks without a tell; an illegal suggestion is rejected with the policy's name;
- [ ] an ask on an empty history uses the design phase.

### P6-11. Study tell, fail and abandon

**Size:** S | **Needs:** P6-09 | **Spec:** 5.4, FR23 | **Status:** todo

**Do.** Add the three operations by trial id: `tell` validates the observation against the outcomes and completes the trial; `fail` records a reason; `abandon` withdraws a pending trial. Each notifies the policy and changes the history version. Unknown ids and illegal transitions raise `TrialStateError`.

**Done when.**
- [ ] tests for each operation, for each illegal move, and for the notification order.

### P6-12. Study attach

**Size:** S | **Needs:** P6-11 | **Spec:** 5.4, FR3 | **Status:** todo

**Do.** Add `attach` for data gathered outside the study: a parameterization with an observation (or a failure) enters as a new completed (or failed) trial after validation against the problem. A helper finds the pending trial with exactly given parameters, for callers who lost the id.

**Done when.**
- [ ] tests: valid and invalid data; replicates allowed; the helper; the policy is notified.

### P6-13. Study seeds and determinism

**Size:** S | **Needs:** P6-10, P1-13, P3-02 | **Spec:** 3.7, 5.11, FR12 | **Status:** todo

**Do.** The study builds an `RngTree` from its seed (recorded; drawn from the operating system when absent) and hands each component a generator derived for (ask index, component path). Components never receive a global random state.

**Done when.**
- [ ] a test runs the same asks on two studies with the same seed and gets identical suggestions; a different seed differs;
- [ ] the seed is part of the resolved configuration.

### P6-14. Study surrogate access and surrogate-only entry

**Size:** S | **Needs:** P6-09, P3-17 | **Spec:** 5.6, 5.13, FR16, FR22 | **Status:** todo

**Do.** Add `Study.surrogate()`: the fitted `Surrogate` for the current completed trials, through the cache, built from the policy's surrogate configuration (or from an explicit configuration argument), raising `ModelNotReadyError` when there is too little data. Add a standalone constructor that builds a `Surrogate` from a problem, a surrogate configuration and a list of observations, with no study. Document that `surrogate().model` is the native model the policy uses.

**Done when.**
- [ ] a test shows that the study's surrogate is the very object the composed policy used for its last ask (same cache entry);
- [ ] the standalone path predicts on a toy dataset without creating a study;
- [ ] the `ModelNotReadyError` case is tested.

### P6-15. Study describe, toConfig and fromConfig

**Size:** S | **Needs:** P6-09, P1-11 | **Spec:** 5.12, FR18, FR26 | **Status:** todo

**Do.** Add `describe()` (the resolved goal, policy and components, and the capability report, as a readable structure), `toConfig()` (the resolved configuration) and `Study.fromConfig` (problem, goal, policy, seed and runtime from a configuration; empty history).

**Done when.**
- [ ] tests: `toConfig`, then `fromConfig`, then `toConfig` is a fixed point; a configuration loaded from a YAML file builds a study; `describe` names the recipe that was chosen.

---

## 12. Phase 7: Analyses and stopping

Introduce the `analysis` and `stopping` kinds with their first components, and give the study the operations to run them. Parameter importance no longer depends on Ax.

**Exit criteria.** Each goal type has default analyses; a study can answer 'what did we learn' and 'should we stop' for optimize, level-set and explore goals.

### P7-01. Analysis contract and registry

**Size:** S | **Needs:** P6-01, P3-09, P4-02 | **Spec:** 5.10, 6 | **Status:** todo

**Do.** Create the `analyses` package and the `analysis` kind: the abstract analysis (registered name, configuration schema, needs and supports, a compute operation receiving a read-only study state and a way to obtain a surrogate on demand, returning a typed, serializable `AnalysisResult`), the registry and `registerAnalysis`.

**Done when.**
- [ ] tests with a fake analysis; result serialization round trip;
- [ ] the contract test covers the kind.

### P7-02. Best-point analyses

**Size:** S | **Needs:** P7-01, P4-03 | **Spec:** 5.10 | **Status:** todo

**Do.** Register `best_observed_v1` (the best completed trial by observed value, for a one-objective optimize goal, with ties and direction handled) and `best_posterior_mean_v1` (the observed point with the best posterior mean, or optionally the best of a candidate sample, selected in the configuration).

**Done when.**
- [ ] tests: minimize and maximize; failed trials ignored; both give the expected point on a toy history.

### P7-03. pareto_front_v1 analysis

**Size:** M | **Needs:** P7-01, P4-03 | **Spec:** 5.10, FR2 | **Status:** todo

**Do.** Register the Pareto-front analysis for a multi-objective optimize goal: the non-dominated completed trials by observed values, or by posterior means (configuration), with directions from the goal.

**Done when.**
- [ ] tests: a known non-dominated set; mixed directions; ties; observed versus model-based mode; an empty history returns an empty result.

### P7-04. level_set_estimate_v1 analysis

**Size:** M | **Needs:** P7-01, P4-04, P3-16, P3-14 | **Spec:** 5.10, FR20 | **Status:** todo

**Do.** Register the level-set estimate: on a candidate set (a design sample of configurable size, or a caller-supplied grid), classify each point as above, below or uncertain using a posterior band of configurable width (in latent space for non-Gaussian surrogates, using the surrogate's level conversion), and return the labels, the band and the points closest to the level (the boundary estimate).

**Done when.**
- [ ] on a synthetic one-dimensional function with a known crossing, the boundary estimate is within a tolerance stated in the test; labels are consistent with the band;
- [ ] it works with the classifier surrogate on a half-plane problem;
- [ ] the candidate-set size and the band width are configuration.

### P7-05. parameter_importance_v1 analysis

**Size:** M | **Needs:** P7-01 | **Spec:** 5.10, FR6, 3.5 | **Status:** todo

**Do.** Register variance-based global sensitivity (first-order and total-order indices per parameter and outcome), computed on the surrogate's posterior mean by Saltelli-type sampling with the sample count as configuration. Do not depend on Ax.

**Done when.**
- [ ] a test uses a stand-in surrogate whose prediction is the Ishigami function and compares the indices with the known values within a tolerance stated in the test;
- [ ] a categorical parameter is handled, or rejected with a clear message.

### P7-06. prediction_grid_v1 analysis

**Size:** S | **Needs:** P7-01 | **Spec:** 5.10, FR7 | **Status:** todo

**Do.** Register the analysis that evaluates the surrogate on a grid or a slice (chosen parameters varied, the others fixed at configured values or at their defaults) and returns arrays of mean and standard deviation with the space flag.

**Done when.**
- [ ] tests: slice shapes; fixed values honoured; defaults used when none are given; the space flag is propagated.

### P7-07. loo_cross_validation_v1 analysis

**Size:** M | **Needs:** P7-01 | **Spec:** 5.10, FR8 | **Status:** todo

**Do.** Register leave-one-out diagnostics for exact surrogates: the predictive mean and standard deviation of each held-out point, standardized residuals, and empirical coverage at configurable levels. Use the engine's cross-validation helpers if the installed version has them (verify), or the closed form for exact GPs.

**Done when.**
- [ ] a test on a noiseless toy function checks the closed form against explicit refits on a small dataset, within a tolerance stated in the test;
- [ ] coverage levels are configuration; the analysis declares that it needs `posterior:gaussian`.

### P7-08. Stopping contract and max_trials_v1

**Size:** S | **Needs:** P6-01 | **Spec:** 5.10, FR28 | **Status:** todo

**Do.** Create the `stopping` package and the `stopping` kind (abstract criterion, registry, `registerStopping`, a `StopDecision` with a reason) and register `max_trials_v1`, which stops when the completed trials reach a configured count.

**Done when.**
- [ ] tests: decision values and reasons;
- [ ] the contract test covers the kind.

### P7-09. no_improvement_v1 stopping

**Size:** S | **Needs:** P7-08, P4-03 | **Spec:** 5.10, FR28 | **Status:** todo

**Do.** Register a criterion that stops when the best observed value (direction from the goal) has not improved by more than a configured relative amount for a configured number of completed trials.

**Done when.**
- [ ] tests: an improving history continues; a flat history stops after the window; minimize and maximize.

### P7-10. level_set_resolved_v1 stopping

**Size:** S | **Needs:** P7-08, P7-04 | **Spec:** 5.10, FR28 | **Status:** todo

**Do.** Register a criterion for level-set goals that stops when the fraction of uncertain candidates in the level-set estimate falls below a configured threshold.

**Done when.**
- [ ] tests with a stand-in estimate for both outcomes of the decision.

### P7-11. Study analyze, shouldStop and default analyses

**Size:** S | **Needs:** P6-09, P7-02, P7-08 | **Spec:** 5.10, FR6, FR28 | **Status:** todo

**Do.** Add `Study.analyze` (run an analysis, given by name or configuration, on the study state, creating the surrogate on demand) and `Study.shouldStop` (evaluate the configured criteria). Goals list default analyses, which `analyze` can run without arguments. An analysis whose needs the study's components do not meet fails the compatibility check before it runs.

**Done when.**
- [ ] tests: analyze by name and by configuration; default analyses per goal; an unmet need raises `CompatibilityError` before the analysis runs;
- [ ] `shouldStop` returns a decision with a reason.

---

## 13. Phase 8: Persistence

Replace the snapshot that wrapped Ax's own state by the document of specification 5.11: resolved configuration, full history, ask counter, reproducibility metadata, component manifest and opaque component state. Repositories become a registered kind.

**Exit criteria.** A study saves to and reloads from one JSON document and continues with identical suggestions; missing plugins and version mismatches are reported clearly; older formats migrate.

### P8-01. ADR-0014: snapshot format v2

**Size:** S | **Needs:** P1-01 | **Spec:** 5.11 | **Status:** todo

**Do.** Record the snapshot document: format version; problem; goal; resolved policy, analysis and stopping configuration; full trial history; ask counter; reproducibility metadata; component manifest; opaque component states. Options considered: keep wrapping Ax's state (rejected by ADR-0009), a SQL store (later, behind the repository kind). ADR-0004 (JSON persistence format) is a stub that was never written: set its status line to 'Superseded by ADR-0014' and remove 'to write' from its navigation label.

**Done when.**
- [ ] ADR complete and in the navigation; ADR-0004's status line and navigation label updated.

### P8-02. Snapshot document model

**Size:** M | **Needs:** P8-01, P2-10, P2-07 | **Spec:** 5.11 | **Status:** todo

**Do.** Add the new `Snapshot` model in `persistence/` as specified in section 5.11, with JSON round trip, a mandatory `format_version` and unknown fields forbidden. `ReproducibilityMetadata` stays and gains the study seed entropy. The model holds configuration as plain JSON-compatible dictionaries (persistence does not depend on `config`).

**Done when.**
- [ ] tests: round trip of a full snapshot built by hand; unknown fields rejected; the version is mandatory;
- [ ] no import from a layer above `domain` (layering check).

### P8-03. Repository kind

**Size:** S | **Needs:** P8-02, P1-10 | **Spec:** 5.11, 6 | **Status:** todo

**Do.** Add the `repository` kind with `json_v1` (a single pretty-printed UTF-8 file) and `memory_v1` (the second implementation, also the test double), with registry plumbing.

**Done when.**
- [ ] tests: save and load round trip for both; a corrupted file raises `ConfigError` with the path;
- [ ] the contract test covers the kind.

### P8-04. Manifest and environment check

**Size:** S | **Needs:** P8-02, P1-06 | **Spec:** 5.11, 3.7 | **Status:** todo

**Do.** Record in the snapshot the names (and, for plugins, the distribution versions) of every component in use. On load, compare with the registries and with the installed library versions: a missing component raises `RegistryError` naming the component and, when known, its plugin; a version difference is logged at WARNING and returned as data.

**Done when.**
- [ ] tests: the missing-component error text; the version-mismatch warning; the plugin hint when the entry point is known.

### P8-05. Migration framework

**Size:** S | **Needs:** P8-02 | **Spec:** 5.11 | **Status:** todo

**Do.** Add a registry of migrations from format version N to N+1, applied in sequence on load, with a test fixture of an artificial version-0 document migrated to the current version. Loading a document newer than the code raises a clear error.

**Done when.**
- [ ] tests: a chain of two migrations; an unknown future version; the order of application.

### P8-06. Policy state in the snapshot

**Size:** S | **Needs:** P6-01, P8-02 | **Spec:** 5.9, 5.11 | **Status:** todo

**Do.** Make export and import of opaque policy state part of the snapshot: the study stores each component's exported state under its component path and gives it back on load.

**Done when.**
- [ ] tests with a toy stateful policy (a counter): the state survives save and load; components without state store nothing.

### P8-07. Study save and load

**Size:** M | **Needs:** P8-03, P8-04, P8-06, P6-15, P6-13 | **Spec:** 5.11, FR11, FR12 | **Status:** todo

**Do.** Add `Study.save` and `Study.load` through the repository kind: save writes the snapshot of the current state; load rebuilds the study from the document alone (problem, goal, resolved policy, history, ask counter, seed, component states), after the manifest check.

**Done when.**
- [ ] tests: save then load gives an equal resolved configuration, an equal history and an equal ask counter; loading with a missing plugin fails as specified;
- [ ] a study loaded from a snapshot still accepts `tell` for its pending trials.

### P8-08. Replay determinism test

**Size:** S | **Needs:** P8-07 | **Spec:** 5.11, 10.11 | **Status:** todo

**Do.** Add the scenario test: run N asks and tells with deterministic fake evaluations, save at step k, load, continue, and compare all later suggestions with an uninterrupted run that has the same seed.

**Done when.**
- [ ] the test passes for the design phase and for the model phase (default recipe on a small problem), with tolerances stated in the test;
- [ ] it fails when the ask counter is not restored (state the evidence).

---

## 14. Phase 9: Acceptance

One task per reference use case of specification section 7. These tests are the proof that the architecture does what the specification says; nothing in this phase adds library features, and a failure here is reported as a defect of an earlier phase, not patched around.

**Exit criteria.** All twelve use cases pass in CI. This is the point at which the legacy path may be deleted (Phase 11).

### P9-01. UC-1: single-objective optimization

**Size:** M | **Needs:** P6-07, P6-10, P6-11, P7-02, P7-11 | **Spec:** 7, FR1, FR4, FR9 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc01_single_objective.py` and `examples/uc01_single_objective.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: a noisy two-dimensional test function with one linear input constraint, default recipe. Acceptance: after the budget stated in the test, the best observed value beats the median of random-search runs of the same budget over the stated seeds; `Study.analyze` returns the best trial; the study saves and reloads.

**Done when.**
- [ ] test and example exist and pass;
- [ ] the acceptance numbers are in the test with a comment on how they were chosen;
- [ ] the example prints a short report (allowed in examples).

### P9-02. UC-2: multi-objective optimization

**Size:** M | **Needs:** P6-07, P7-03, P5-05, P9-01 | **Spec:** 7, FR2 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc02_multi_objective.py` and `examples/uc02_multi_objective.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: two outcomes with a known Pareto front, with and without weights. Acceptance: the hypervolume reached beats random search at equal budget; the Pareto analysis returns non-dominated points.

**Done when.**
- [ ] test and example exist and pass, for both the weighted and the unweighted case.

### P9-03. UC-3: outcome constraints

**Size:** M | **Needs:** P9-01, P5-03 | **Spec:** 7, FR15 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc03_outcome_constraints.py` and `examples/uc03_outcome_constraints.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: optimization with one outcome constraint. Acceptance: after the initial phase the suggestions violate the constraint less often than random search at equal budget; the best feasible point is reported; the constraint travels from the goal into the acquisition.

**Done when.**
- [ ] test and example exist and pass;
- [ ] a variant with an acquisition that does not support outcome constraints fails at construction with the specified message.

### P9-04. UC-4: level-set estimation

**Size:** M | **Needs:** P6-08, P7-04, P5-09, P7-10, P9-01 | **Spec:** 7, FR20 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc04_level_set.py` and `examples/uc04_level_set.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: one-dimensional and two-dimensional continuous functions with known level sets. Acceptance: the distance between the estimated boundary points and the true level set beats a random design at equal budget; `level_set_resolved_v1` stops the run.

**Done when.**
- [ ] test and example exist and pass for both dimensionalities.

### P9-05. UC-5: boundary between two phases from binary labels

**Size:** M | **Needs:** P9-04, P3-16 | **Spec:** 7, FR20, FR24 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc05_phase_boundary.py` and `examples/uc05_phase_boundary.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: a synthetic phase diagram (a circle or a curved boundary in two parameters, one of them on a log scale), noise-free labels. Acceptance: the mean distance of the estimated boundary points to the true boundary is below a threshold stated in the test and below that of a random design at equal budget; predictions state their space.

**Done when.**
- [ ] test and example exist and pass;
- [ ] the example explains, in comments, how the level is read in latent and in response space.

### P9-06. UC-6: pure exploration

**Size:** M | **Needs:** P6-08, P5-06 | **Spec:** 7, FR21 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc06_exploration.py` and `examples/uc06_exploration.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: an explore goal with no direction. Acceptance: after the stated budget, the surrogate's predictive error on a held-out grid is lower than with a random design at equal budget.

**Done when.**
- [ ] test and example exist and pass.

### P9-07. UC-7: surrogate only

**Size:** M | **Needs:** P6-14, P7-05, P7-06, P7-07 | **Spec:** 7, FR6, FR7, FR8, FR22 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc07_surrogate_only.py` and `examples/uc07_surrogate_only.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: build a surrogate from a table of existing observations without a study; predict, sample, run `parameter_importance_v1` (compared with the known importance of the test function), `loo_cross_validation_v1` and `prediction_grid_v1`. No ask or tell is called.

**Done when.**
- [ ] test and example exist and pass;
- [ ] a check shows that no `Study` object is created in the test.

### P9-08. UC-8: batches, asynchronous asks, failures

**Size:** M | **Needs:** P6-10, P6-11, P6-12 | **Spec:** 7, FR10, FR23 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc08_lifecycle.py` and `examples/uc08_lifecycle.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: ask batches of 4 twice before any tell; tell in a different order; fail and abandon trials. Acceptance: the policy never re-suggests a pending, failed or abandoned parameterization; ids and states stay consistent; the history round-trips.

**Done when.**
- [ ] test and example exist and pass.

### P9-09. UC-9: mixed spaces and constraints

**Size:** M | **Needs:** P5-13, P5-12, P4-06, P6-10 | **Spec:** 7, FR1, FR9, FR13, FR14, FR17 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc09_mixed_space.py` and `examples/uc09_mixed_space.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: a log-scaled real, an integer with a step, an unordered categorical, an ordered choice, a conditional parameter, a derived parameter, a linear equality constraint (a mixture summing to one) and a nonlinear constraint. Acceptance: all suggestions are legal and feasible over many asks; the encoder round trip holds.

**Done when.**
- [ ] test and example exist and pass;
- [ ] the example uses the default recipe, with no component named explicitly.

### P9-10. UC-10: warm start, save, reload, continue

**Size:** M | **Needs:** P6-12, P8-07, P8-08 | **Spec:** 7, FR3, FR11, FR12 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc10_warm_start.py` and `examples/uc10_warm_start.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. Scenario: attach a table of prior observations, check that the design phase is shortened according to its option, continue; save and reload mid-run and compare with an uninterrupted run (reuses the replay test).

**Done when.**
- [ ] test and example exist and pass.

### P9-11. UC-11: components defined outside the package

**Size:** M | **Needs:** P1-12, P5-01, P3-10, P7-01, P6-10, P8-07 | **Spec:** 7, 1.3, FR18, FR19 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc11_external_plugin.py` and `examples/uc11_external_plugin.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. A separate test package (`tests/use_cases/external_plugin/`, installed in the test as a distribution with an entry point) defines a kernel, an acquisition and an analysis; a study configuration uses all three by name; the study saves, reloads with the plugin present, and fails with the specified error when the plugin is absent. The plugin imports only public contracts.

**Done when.**
- [ ] test and example exist and pass;
- [ ] a check shows that no file under `src/planexs` was edited for this use case.

### P9-12. UC-12: opaque policies

**Size:** M | **Needs:** P6-06, P6-10, P8-06 | **Spec:** 7, FR27 | **Status:** todo

**Do.** Add `tests/use_cases/test_uc12_opaque_policy.py` and `examples/uc12_opaque_policy.py`. The test and the example use configuration and registered components only; the acceptance numbers are stated in the test, with a comment on how they were chosen; the test is deterministic through seeds. The same study driven by `random_policy_v1` and by a user-written opaque policy defined in the test; the study treats them like composed policies (pending trials, history, save, load with policy state). The Ax adapter joins this use case in P10-10.

**Done when.**
- [ ] test and example exist and pass.

---

## 15. Phase 10: Input extensibility and the Ax adapter

Finish the registry-driven design on the input side (parameter and constraint encoding rules), prove it with an example plugin that adds a new parameter type, and port the Ax code to an optional adapter behind the policy seam.

**Exit criteria.** Parameter and constraint types are fully registry-driven (a third-party type works end to end); Ax is an optional extra behind the policy seam; the core imports and runs without Ax.

### P10-01. Parameter-encoding contract and registry

**Size:** S | **Needs:** P3-03, P2-04 | **Spec:** 5.5, 6 | **Status:** todo

**Do.** Create the `parameter_encoding` kind: the abstract encoding rules for one parameter type (column layout, encoding a value, decoding with snapping, the neutral value of an inactive conditional parameter, sampling from a uniform latent, rounding neighbours), the registry and `registerParameterEncoding`. Nothing is ported yet.

**Done when.**
- [ ] tests with a fake rule;
- [ ] the contract test covers the kind.

### P10-02. Range encoding on the registry

**Size:** M | **Needs:** P10-01 | **Spec:** 5.5, FR1 | **Status:** todo

**Do.** Port the range handling of `SearchSpaceEncoder` and its module-level helpers (`_encodeRange`, `_rawToUnit`, `_unitToRaw`, `_snapRaw`, `_floorAndCeil`) into a registered `range` encoding rule. The encoder calls the registry for ranges.

**Done when.**
- [ ] every existing encoder test passes unchanged;
- [ ] a toy parameter type registered in a test with a range-like rule works in the encoder.

### P10-03. Choice encoding on the registry

**Size:** M | **Needs:** P10-02 | **Spec:** 5.5, FR1, FR14 | **Status:** todo

**Do.** Do the same for `choice`: ordered rank columns, 0/1 columns and one-hot blocks, neutral values for inactive conditional parameters, and the hooks that categorical enumeration uses.

**Done when.**
- [ ] existing tests pass unchanged;
- [ ] categorical enumeration in the optimizers still works (the UC-9 test passes).

### P10-04. Derived and fixed handling; delete the closed chains

**Size:** S | **Needs:** P10-03 | **Spec:** 5.5, 3.3 | **Status:** todo

**Do.** Port derived parameters and single-valued choices, then delete every remaining `isinstance` and type-tag chain over parameter classes in `encoding/`.

**Done when.**
- [ ] no concrete parameter class is named in the encoder files (`git grep` evidence in the report);
- [ ] all tests pass.

### P10-05. Constraint-encoding registry

**Size:** M | **Needs:** P10-04, P2-05, P3-05, P5-12 | **Spec:** 5.5, FR9, FR17 | **Status:** todo

**Do.** Create the `constraint_encoding` kind: for each constraint type, its encoded form (coefficient rows for linear, a differentiable callable for nonlinear), its violation measure on encoded points and the tags it needs from optimizers (`constraints:linear`, `constraints:equality`, `constraints:nonlinear`). Port `EncodedConstraints` to iterate over registered rules.

**Done when.**
- [ ] no concrete constraint class is named in `EncodedConstraints`; existing tests pass;
- [ ] a toy constraint type registered in a test (for example an ordering constraint between two ranges) is enforced by the sampler and by `botorch_relaxed_v1`.

### P10-06. Example plugin: a new parameter type

**Size:** S | **Needs:** P10-05, P1-12 | **Spec:** 1.3, 5.12 | **Status:** todo

**Do.** Under `examples/plugins/`, add a small distribution that registers a parameter type the existing ones cannot express (for example a cyclic angle encoded on a circle), with its encoding rule, and an example study that uses it.

**Done when.**
- [ ] the example runs end to end with the default recipe on a problem that includes the new type;
- [ ] the plugin imports only public contracts and has its own tests.

### P10-07. Ax adapter: policy

**Size:** M | **Needs:** P6-01, P9-12 | **Spec:** 1.4, 5.9, FR27 | **Decision:** D-3 | **Status:** todo

**Do.** Create `adapters/ax/` and register `ax_default_v1` as an opaque policy. Construction builds an Ax `Client` from the problem and the goal (the mapping helpers, now in `backends/ax`, move here, one function per file). Suggestions come from `get_next_trials`; each trial state change is mirrored into the client; state export and import use the client's JSON. Ax is imported lazily inside the adapter.

**Done when.**
- [ ] tests (skipped when Ax is absent) run a short study on a two-dimensional problem through the adapter and check legal suggestions;
- [ ] export and import reproduce the client's trial count;
- [ ] the layering check shows Ax imported only under `adapters/ax`.

### P10-08. Ax adapter: capabilities and analyses

**Size:** S | **Needs:** P10-07, P7-01 | **Spec:** 5.12, FR25 | **Decision:** D-3 | **Status:** todo

**Do.** Declare the adapter's capabilities truthfully (linear inequality constraints and outcome constraints yes; nonlinear and equality constraints no; the rest according to Ax's own support) and verify them by tests. Optionally register Ax's own analyses (for example its sensitivity analysis) as `analysis` entries provided by the adapter.

**Done when.**
- [ ] capability-truthfulness tests for every declared tag (skipped without Ax);
- [ ] the compatibility check rejects the adapter for a problem with a nonlinear constraint, with a message that names a compatible policy.

### P10-09. Optional extra and import isolation

**Size:** S | **Needs:** P10-07 | **Spec:** 1.4, 10.9 | **Decision:** D-3 | **Status:** todo

**Do.** Move `ax-platform` from `dependencies` to an `ax` extra in `pyproject.toml`, refresh the lock file with uv, and add a test that imports `planexs` and builds a default study in a subprocess where `ax` is blocked by an import hook. Add a CI job that installs the extra and runs the adapter tests.

**Done when.**
- [ ] the blocked-import test passes;
- [ ] `uv sync` without the extra works, and with it the adapter tests run in CI.

### P10-10. UC-12 with the Ax adapter

**Size:** S | **Needs:** P10-08, P9-12, P10-09 | **Spec:** 7, FR27 | **Decision:** D-3 | **Status:** todo

**Do.** Extend the UC-12 test with the Ax adapter (skipped without Ax): the same study protocol, with save and reload carrying the client state.

**Done when.**
- [ ] the extended test passes in the CI job that installs the extra.

---

## 16. Phase 11: Cleanup and release

Delete the legacy path, rebuild the public surface, port the examples, finish the documentation, bring the policy-value baseline to zero and release.

**Exit criteria.** No legacy code remains; the policy-value baseline is gone; the documentation is complete and its snippets are tested; 0.2.0 is tagged.

### P11-01. Remove the legacy facade and backends

**Size:** S | **Needs:** P9-01, P9-02, P9-03, P9-04, P9-05, P9-06, P9-07, P9-08, P9-09, P9-10, P9-11, P9-12, P10-07 | **Spec:** 11 | **Decision:** D-2 | **Status:** todo

**Do.** Delete `api/BayesianOptimizer.py`, `backends/OptimizationBackend.py`, `backends/PredictionUnavailableError.py`, `backends/ax/AxBackend.py` and the legacy tests that exercise them, and delete the baseline lines of the deleted files. Precondition: all Phase 9 tasks and P10-07 are done; list the evidence in the report.

**Done when.**
- [ ] the full suite passes without them;
- [ ] `git grep -n BayesianOptimizer` finds only changelog and migration text;
- [ ] the baseline lines of the deleted files are removed.

### P11-02. Remove the legacy surrogate and acquisition code

**Size:** S | **Needs:** P11-01 | **Spec:** 11 | **Decision:** D-2 | **Status:** todo

**Do.** Delete `models/buildSurrogateModel.py`, `models/encodeTrials.py`, `models/predictWithModel.py`, `acquisition/AlphaAcquisitionStrategy.py`, `acquisition/AcquisitionStrategy.py`, the legacy acquisition classes that were ported, and their tests. The parity tests that compared them with the new code are deleted with them; the new tests stay.

**Done when.**
- [ ] the suite passes; `models/` and the old `acquisition` files are gone;
- [ ] the baseline lines of the deleted files are removed.

### P11-03. Remove the legacy domain types

**Size:** S | **Needs:** P11-02 | **Spec:** 11 | **Decision:** D-2 | **Status:** todo

**Do.** Delete `LegacyTrial`, `LegacyObjective`, `LegacyMetric`, `LegacyOutcomeConstraint`, `LegacyStudySnapshot`, `analysis/PredictionResult.py`, the `requires_custom_acquisition_layer` property on constraints, and what remains of the `backends` and `api` packages.

**Done when.**
- [ ] the suite passes; `git grep -n Legacy` is empty;
- [ ] the `legacy` group is removed from `scripts/layers.toml`.

### P11-04. Public exports

**Size:** S | **Needs:** P11-03 | **Spec:** 8, 10.4 | **Status:** todo

**Do.** Rewrite the lazy exports of `planexs/__init__.py` for the public surface: `Study`, parameters and constraints, outcomes, trials, the goal configurations, the registration decorators of every kind, `loadPlugins`, the errors and `__version__`. Add a test that lists the exported names and a test that every export has a docstring.

**Done when.**
- [ ] the names test and the docstring test pass;
- [ ] mypy strict passes.

### P11-05. Port the lab and ML examples

**Size:** S | **Needs:** P11-04 | **Spec:** 7, 9.1 | **Status:** todo

**Do.** Rewrite `examples/lab_experiment.py` and `examples/ml_hyperparameter_search.py` on `Study`, keeping their scenarios (a campaign of four or five evaluations; batches with the exploration/exploitation weight).

**Done when.**
- [ ] both run to completion in a CI job that runs the examples;
- [ ] the exploration/exploitation weight appears as acquisition configuration.

### P11-06. Port the constraint and escape-hatch examples

**Size:** S | **Needs:** P11-04 | **Spec:** 5.13, 9.1 | **Status:** todo

**Do.** Rewrite `examples/nonlinear_constraint.py` (and fix its line over the length limit) and `examples/escape_hatch.py`, which now shows the native model obtained from `Study.surrogate().model` and the registered-component route.

**Done when.**
- [ ] both run in CI;
- [ ] the escape-hatch example shows a custom component registered in the example itself.

### P11-07. How-to: first study and configuration

**Size:** S | **Needs:** P11-04 | **Spec:** 10.12, FR4, FR18 | **Status:** todo

**Do.** Write `docs/how-to/first-study.md`: define a problem and a goal, run ask and tell, read the default analyses, save and reload; then the same study from a YAML file; then how to read `describe()` to see which recipe was chosen. All snippets are exercised by a test that extracts and runs them.

**Done when.**
- [ ] the page exists, is in the navigation, and its snippets run in a test.

### P11-08. How-to: surrogate-only use

**Size:** S | **Needs:** P11-04 | **Spec:** 10.12, FR22 | **Status:** todo

**Do.** Write `docs/how-to/surrogate-only.md`: build a surrogate from a table, predict, sample, run importance and diagnostics.

**Done when.**
- [ ] the page is in the navigation and its snippets run in a test.

### P11-09. How-to: write a plugin

**Size:** S | **Needs:** P11-04, P10-06 | **Spec:** 10.14, FR19 | **Status:** todo

**Do.** Write `docs/how-to/write-a-plugin.md`: the new-component checklist (10.14) applied to a kernel, an acquisition and an analysis; how a plugin package declares its entry point; how needs and supports are chosen; how to test a plugin against the contract tests.

**Done when.**
- [ ] the page is in the navigation; the example plugin of P10-06 is referenced and its code is the source of the tested snippets.

### P11-10. How-to: level-set estimation

**Size:** S | **Needs:** P11-04, P9-05 | **Spec:** 10.12, FR20 | **Status:** todo

**Do.** Write `docs/how-to/level-set-estimation.md`: boundaries for a continuous outcome and for binary phase labels, how the level is read in latent and in response space, reading the estimate, choosing a stopping criterion.

**Done when.**
- [ ] the page is in the navigation and its snippets run in a test.

### P11-11. API reference

**Size:** S | **Needs:** P11-04 | **Spec:** 10.12 | **Status:** todo

**Do.** Wire `mkdocstrings` so that every public export has a reference page generated from its docstring.

**Done when.**
- [ ] a reference page exists for every public export;
- [ ] `mkdocs build --strict` passes.

### P11-12. README, contributing and stale references

**Size:** S | **Needs:** P11-05, P11-06, P11-07, P11-08, P11-09, P11-10 | **Spec:** 0, 10.12 | **Status:** todo

**Do.** Rewrite `README.md` (what planexs is, a ten-line example, links) and refresh `docs/contributing.md`. Every `section N` mention left in `src`, `tests`, `examples` and `scripts` (43 at baseline, all pointing to the old specification's numbering) must point to a section that exists in the current specification, or be removed with the code it annotated.

**Done when.**
- [ ] a test scans for `section N` references and fails on one that does not exist in the specification;
- [ ] the README example runs in a test.

### P11-13. Policy-value baseline to zero

**Size:** S | **Needs:** P11-03 | **Spec:** 10.7 | **Status:** todo

**Do.** Verify that `scripts/policy_values_baseline.txt` is empty (every remaining entry is either a bug to fix in this task or a literal that deserves a `# structural:` marker with a real reason). Remove `--baseline` from CI so that the check runs with no baseline file, and delete the file.

**Done when.**
- [ ] the check passes with no baseline, and CI runs it without one.

### P11-14. License and metadata

**Size:** S | **Needs:** P11-12 | **Spec:** 10.13 | **Decision:** D-11 | **Status:** todo

**Do.** Apply the license chosen in D-11: the `LICENSE` file, the `license` field of `pyproject.toml` (it says TBD today), classifiers, keywords, authors, and a project description that matches the scope of the specification. The package name is settled (D-1: planexs). The `Homepage`, `Documentation` and `Changelog` URLs of `pyproject.toml` still use the former repository name `boptim`: update them if the repository has been renamed by then.

**Done when.**
- [ ] the license file and the metadata agree; `uv build` succeeds and the built metadata passes `twine check`.

### P11-15. Release 0.2.0

**Size:** S | **Needs:** P11-12, P11-13, P11-14 | **Spec:** 10.13 | **Status:** todo

**Do.** Run `scripts/cut_release.py 0.2.0 '<summary>'` (it scaffolds `docs/changelogs/changelog-v0.2.0.md`, updates the changelog index and bumps the version in `pyproject.toml`) and write the changelog: the new architecture, the template's Added, Changed and Fixed sections, a Removed section that gives the replacement for each removed API, and migration notes. Tag after the merge (the script does not touch git). Refresh section 2.1 of this roadmap with the new measurements.

**Done when.**
- [ ] the changelog is complete (Added, Changed, Fixed, Removed with replacements, Migration);
- [ ] the version is bumped in one place and CI is green on the release commit;
- [ ] section 2.1 of this roadmap is refreshed.

---

## 17. Decisions needed

"Proposed" is what the work assumes if the owner simply says "go with the defaults". A task that names a decision waits for the answer, or for the owner to accept the proposed default.

| ID | Question | Options | Proposed | Blocks |
|---|---|---|---|---|
| D-1 | **Package name.** `boptim` was narrower than the scope: sequential design with surrogates, of which optimization is one goal. | keep; rename before 0.2.0 | **Decided by the owner on 2026-10-11: `planexs`**, for PLANification EXpérimentale Séquentielle. Applied by the rename change: package, imports, identifiers, configuration, living documents. ADR-0001 to ADR-0008 and the 0.1.0 changelog keep the former name, as historical records. The GitHub URLs in `pyproject.toml` keep it until the repository is renamed. ADR-0009 (P1-01) records the decision. | none (decided) |
| D-2 | **Compatibility with the 0.1.0 API.** Remove the old facade and backends, and do not load 0.1.0 snapshots. | no shim; a shim for one release | No shim: five commits and no tag, so a shim costs more than it protects. | P11-01, P11-02, P11-03 |
| D-3 | **Ax.** Keep an optional Ax adapter, or drop Ax altogether. | optional extra; drop | Optional extra, ported in Phase 10 once the use cases pass. | P10-07 to P10-10 |
| D-4 | **Configuration formats.** Which file formats, and whether to depend on Hydra. | JSON only; JSON and YAML; Hydra | JSON natively and YAML through PyYAML as a core dependency; no Hydra dependency (configurations are plain data, so Hydra and OmegaConf users can feed them). | P1-11 |
| D-5 | **Policy-value exemptions.** Which literals the checker lets through. | stricter; as proposed | **Decided by the owner on 2026-10-13: as proposed.** 0, 1, -1 and 2 in any numeric spelling; `# structural: <reason>` for other structural constants; configuration-schema files exempt. What the checker cannot see (a number that defines a tunable value behind 0, 1 or 2, strings, booleans) is governed by the rule of specification 3.4 and by review, not by automated detection. The owner also confirmed (option A) that a literal default in a signature is a policy value, as specification 3.4 and 10.8 say, and may ask to revisit that if the options-object style proves too heavy for the Python API. Applied by P0-05; ADR-0011 (P1-03) records it. | none (decided) |
| D-6 | **CI matrix.** Python and dependency versions. | one version; several; newest dependencies too | **Decided by the owner on 2026-10-13: Python 3.11 and 3.13, ruff pinned exactly.** The locked dependency versions, ruff also pinned to `ruff==0.16.9` in `pyproject.toml`; no job on the newest dependencies for now. Applied in `.github/workflows/ci.yaml` (P0-03) and `pyproject.toml`. The classifiers still declare 3.11 and 3.12: P11-14 settles them. | none (decided) |
| D-7 | **Outcome kinds in the first release.** | continuous and binary; more | Continuous and binary. Ordinal, count and categorical outcomes arrive as plugins or later tasks. | P2-01, P2-06 |
| D-8 | **First catalogue of built-in components.** | as in the plan; larger | Those named in Phases 3 to 7. Anything else is a plugin or a new task. | none |
| D-9 | **Default recipes.** Which components each goal uses by default. | as in P6-07 and P6-08; others | As proposed in those tasks. The owner reviews them after Phase 9, with the numbers recorded by the use-case tests. | none (review after Phase 9) |
| D-10 | **Plugin entry-point group name.** | any | `planexs.plugins`. | P1-02, P1-12 |
| D-11 | **License.** | for example MIT, Apache-2.0 or CeCILL | The owner chooses. | P11-14 |
| D-12 | **Runtime defaults.** dtype, device, deterministic flag. | as proposed; others | float64, CPU, deterministic flag off. | P1-14 |
| D-13 | **Horizon use cases** (specification 7.3). Which, if any, to schedule after 0.2.0. | none; some | None before 0.2.0 is released. | none |

---

## 18. Not planned

- Executing evaluations: schedulers, job queues, instrument drivers (specification 1.2).
- A user interface or a service.
- Reading, pairing or splitting datasets.
- Surrogates other than Gaussian processes in the core (the contract allows them as plugins).
- Structured inputs (graphs, molecules, sequences) in the core.
- A compatibility layer for the 0.1.0 API (D-2).
- The horizon use cases of specification 7.3 (D-13).
- A SQL repository (the repository kind allows one; nothing is scheduled).
- A dependency on Hydra (D-4).

---

## 19. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| "Everything is possible" grows without bound | Endless scope, interfaces with nothing behind them | The rule of two (specification 3.10), the use-case matrix (specification 7), the list of section 18, phase exit criteria |
| LLM-executed tasks cut corners to save effort | Green tests over missing work | The size cap, checklists with evidence, the report format, the shortcutting list (section 3.5), a reviewer who reruns the commands |
| Porting changes numerical behaviour | Silent regressions in results | The legacy code stays until parity tests pass (section 2.3), with tolerances stated in the tests |
| Component compatibility becomes intricate | Confusing errors | A small tag vocabulary, truthfulness tests per entry, one message snapshot per kind of failure |
| Engine API drift (BoTorch, GPyTorch, Ax) | Breakage after upgrades | Thin builders, versions locked in `uv.lock`, the contract tests, the verify-the-API rule (specification 10.9) |
| Stochastic tests flake | Red CI with no code change | Fixed seeds, stated tolerances, no exact float equality on stochastic output |
| Snapshot format churn | Old studies become unreadable | `format_version` and migrations from the first version (P8-02, P8-05) |
| Hidden policy values return | The original defect | The ratchet checker gated in CI (P0-06) and review of non-numeric values |
| Configuration becomes verbose | Poor usability | Recipes (specification 5.9), three levels of use (specification 3.8), `describe()` (P6-15) |

---

## 20. Definition of done

Every task is done when:

1. The code follows specification 10: English, type hints, Google docstrings with `Args`, `Returns` and `Raises`, one class per file, camelCase callables, file names equal to their symbol, no em dash and no unicode arrow.
2. A new component follows the registry pattern: its own configuration schema, declared needs and supports, an import in its subpackage `__init__`; no core file is edited to add it.
3. No literal policy value is added, and the baseline lines made obsolete are removed.
4. Tests cover the behaviour listed under Done-when, plus the validation of every new configuration.
5. An ADR records any architectural decision, and a changed decision is a new ADR.
6. Documentation (docstrings, and the how-to when the card says so) is updated and `mkdocs build --strict` passes.
7. CI is green: ruff, mypy, pytest, the naming, policy-value and layering checks, and the documentation build.
8. The report of section 3.4 is written, every Done-when box has evidence, and the card's status is `done`.

---

## Appendix A. How the baseline was measured

- **Date and commit.** 2026-10-09, `f5829e4` (version 0.1.0).
- **Environment.** A Linux container whose network access is limited to package indexes. The project's heavy dependencies were not installed, so nothing that imports torch, botorch or ax was run.
- **Measured.** File and line counts with `find`, `wc` and `grep`; ruff 0.17.0 (`ruff check .` and `ruff format --check .`) with the repository's configuration; `scripts/check_naming_convention.py`; a one-off AST scan of `src/planexs` for numeric literals outside {0, 1, -1, 2} (the script is not part of the repository; P0-05 makes the check permanent); `git ls-files` and `grep` for the case defect, the closed dispatch points, the hard-wired engine names, the typography and the `section N` references.
- **Not measured at that date.** Test results, coverage and mypy: measured by P0-02, see the next bullet.
- **Dynamic measurements (P0-02).** 2026-10-11, fresh clone of tree `8338ba65fe60` (commit `d1c0b46`; run on the local commit `9fdaa09`, same tree), Linux x86_64, Python 3.12.3, uv 0.11.7. Install: `uv sync --locked --extra dev --extra docs` (the command of the task card with `--locked`, which fails instead of rewriting `uv.lock` when it is stale; `uv.lock` was left unchanged). Locked versions: torch 2.14.0+cu130, botorch 0.18.1, gpytorch 1.15.2, ax-platform 1.3.1, pydantic 2.13.5, pytest 9.1.1, mypy 2.3.1, ruff 0.16.9. Commands: `pytest` (coverage options come from `pyproject.toml`, plus a JSON coverage report for the exact percentage), `mypy`, `ruff check .`, `ruff format --check .` and `mkdocs build --strict`; ruff 0.17.0 was also run through `uvx` for comparison. The static counts re-run on the same tree matched the baseline exactly (29 test files, 359 test functions, 4,535 test lines, 58 source files, 5,823 source lines). The measurements changed no file of the repository.
- **Caveats.** The project does not pin ruff in `pyproject.toml` (`uv.lock` does), so its counts may differ with another version; P0-02 found ruff 0.16.9 and 0.17.0 identical on the current tree. The literal count excludes strings, booleans and heuristics hidden behind 0, 1 or 2.
