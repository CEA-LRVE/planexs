# ADR-0008: An equality `LinearConstraint` also forces the custom acquisition layer

**Status:** Superseded by ADR-0009
**Date:** 2026-10-01
**Deciders:** [project owner]
**Extends:** ADR-0006 (the scope of "a constraint Ax cannot enforce")

## Context

ADR-0006 made `ask()` switch to `boptim`'s own acquisition layer, with a warning, when the
search space holds a `NonlinearConstraint`, because Ax's default strategy cannot enforce it.
It assumed that every `LinearConstraint` was enforceable by Ax.

That assumption is false for the `"="` comparator, which `LinearConstraint` has always
offered (its docstring's own example is a mixture summing to one). Tested against Ax 1.3.1:

| How the equality is given to Ax | Result |
|---|---|
| `"a + b + c = 1.0"` as a `parameter_constraints` string | `UserInputError: Expected an inequality`, at study construction |
| Two inequalities, `<= 1.0` and `>= 1.0` | The study builds, but candidate generation fails with `SearchSpaceExhausted`: rejection sampling can never land on a zero-volume slice |

So no encoding makes Ax enforce an equality, and a study with one could not be created at all.

## Decision

Treat an equality `LinearConstraint` exactly like a `NonlinearConstraint`:

- `Constraint` gains a property, `requires_custom_acquisition_layer` (default `False`).
  `NonlinearConstraint` returns `True`; `LinearConstraint` returns `True` only for `"="`.
- `AxBackend.createExperiment` hands Ax only the constraints Ax can enforce, so the study
  builds. Inequalities are still enforced by Ax.
- `BayesianOptimizer.ask()` with `alpha=None` switches to the custom layer, with the same
  `DEFAULT_ALPHA_WHEN_FORCED` and the same warning, whenever any constraint in the space has
  that property. The warning names which kinds were found. An explicit `alpha` never warns.
- The custom layer already enforces equalities exactly: BoTorch's `equality_constraints`, with
  starting points drawn from the constraint's polytope (`sample_q_batches_from_polytope`)
  rather than by rejection.

## Options considered

| Option | Trade-off |
|---|---|
| Two inequalities in Ax | Does not work (table above) |
| Eliminate one variable through an Ax `DerivedParameter` (`c = 1 - a - b`) so Ax sees a full-dimensional polytope | Keeps Ax's strategy, but rewrites the parameterization Ax sees, interacts with persistence and with `attach_trial`, and needs a choice of which variable to eliminate when bounds are tight |
| Route it to the custom layer (chosen) | Reuses ADR-0006's mechanism and already-tested equality handling; costs the Ax default strategy for such studies |

## Consequences

- A mixture study builds and runs end to end. Every suggestion satisfies the equality (checked
  to `1e-5` in the tests), including after a save and reload.
- The Ax client behind `axClient` does not know about the equality: points drawn directly from
  it through the escape hatch (ADR-0005) are unconstrained, as with `NonlinearConstraint`.
- Studies with an equality inherit the caveats of the custom layer in this phase
  (`OutcomeConstraint`s are not enforced, and the forced default `alpha` of ADR-0006).

## Action items

1. [x] `Constraint.requires_custom_acquisition_layer`, `AxBackend` filtering, generalized switch.
2. [x] Tests: unit (flag, switch truth table) and integration against real Ax (study builds,
   suggestions sum to one, Ax is handed only the inequality, save/load).
