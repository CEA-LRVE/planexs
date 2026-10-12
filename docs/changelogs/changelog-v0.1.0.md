# v0.1.0 (2026-09-25)

Phase 1 of the roadmap: **Foundation (domain model + Ax-backed default flow)**.

## Added

- Full domain model (`boptim.domain`): `Range`/`Real`/`Integer`, `Choice`/`Categorical`/
  `Boolean`/`Fixed`/`Derived` (including `Choice.dependent_parameters`, FR14's definition
  side), `Constraint`/`LinearConstraint` (FR9), `SearchSpace`, `Metric`, `Objective` (`N >= 1`
  metrics, optional weights, FR2), `OutcomeConstraint` (FR15), `Trial`, `StudySnapshot`.
- `AxBackend` (`boptim.backends.ax`): maps `SearchSpace`/`Objective` to
  `ax.api.client.Client.configure_experiment`/`configure_optimization`/
  `configure_generation_strategy`; manual trial injection (FR3, `attachTrial`); `ask()` via
  Ax's own no-manual-tuning default generation strategy (FR4, `suggestDefault`); prediction
  with uncertainty in Ax's own `(mean, sem)` shape (FR7, FR8, `predict`); sensitivity analysis
  via Ax's `compute_analyses` (FR6, `computeSensitivity`); Pareto-frontier/best-trial lookup
  (`getParetoFrontier`, `getBestTrial`); and `exportState`/`importState` for persistence.
- `BayesianOptimizer` (`boptim.api`): the public facade. `tell()` (FR3), `ask()` (FR4; `FR5`'s
  `alpha` raises `NotImplementedError` until Phase 2), `predict()` (FR7, FR8),
  `parameterImportance()` (FR6), `n_trials`/`paretoFront`, `save()`/`load()` (FR11, FR12), and
  the `axClient`/`fitModel()` escape hatches (FR16, ADR-0005).
- `JsonStudyRepository` (`boptim.persistence`): saves/loads a whole study: domain objects,
  full trial history, and `ReproducibilityMetadata` (random seed, library versions, creation
  time), as a single JSON file, wrapping Ax's own `Client.save_to_json_file`/
  `load_from_json_file` for the Ax-backed portion of the state (FR11, FR12).
- `examples/lab_experiment.py` and `examples/escape_hatch.py`.
- ADR-0001, ADR-0002, ADR-0005, ADR-0006 written in full; ADR-0003, ADR-0004 stubbed
  ("to write") with their seeding context, per `PROJECT_SPECIFICATION.md` section 4.8.
- `scripts/check_naming_convention.py`, `scripts/new_adr.py`, `scripts/cut_release.py`.
- Unit tests for the domain layer, `toAxOptimizationConfig`/`LinearConstraint` string
  rendering, and `JsonStudyRepository`'s pure-domain round trip; an integration test outline
  for the full `tell`/`ask`/`save`/`load` loop against a dummy objective.

## Design note: explicit, type-tagged serialization of `SearchSpace`

`SearchSpace.parameters` / `.constraints` are lists of *abstract* base types. Left to
Pydantic they would have been serialized by declared type (subclass fields dropped) and
could not have been rebuilt on load. `Parameter.toDict` / `Constraint.toDict` therefore emit
`"kind"`-tagged dicts, rebuilt by `parameterFromDict` / `constraintFromDict`. Consequences:
`Real`/`Integer`/`Categorical`/`Boolean`/`Fixed` are constructors, and reload as `Range` /
`Choice`; `Choice.dependent_parameters` is stored as `[value, [names]]` pairs (JSON object
keys are always strings); Phase 2 must extend `constraintFromDict` with `NonlinearConstraint`.

## Known limitations (tracked for Phase 2/3, not regressions)

- `BayesianOptimizer.ask(alpha=...)` raises `NotImplementedError` for any `alpha != None`: the
  custom BoTorch exploration/exploitation acquisition layer (FR5) is Phase 2.
- `NonlinearConstraint` (FR17) does not exist yet; deferred to Phase 2 alongside the
  acquisition layer that is the only thing able to enforce it.
- `AxBackend.fitModel()` and `AxBackend.computeSensitivity()` walk parts of Ax's object graph
  that are not covered by Ax's own `ax.api` stability guarantees; both are defensive and
  clearly documented as such, but may need adjustment once run against the pinned Ax version.
- Not yet run against real `pydantic` / `ax-platform` / `botorch` (see the README's
  verification status): only syntax checks and a run against throwaway stand-ins.
