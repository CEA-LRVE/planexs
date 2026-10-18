# ADR-0007: The acquisition layer fits its own surrogate model from boptim's trial history

**Status:** Superseded by ADR-0009
**Date:** 2026-09-30
**Deciders:** [project owner]
**Refines:** ADR-0001 (the "fitted-model handoff" between Ax and the acquisition layer)

## Context

`PROJECT_SPECIFICATION.md` section 4.2 described the custom acquisition layer's data flow as:
`BayesianOptimizer` "asks the Ax backend to fit a surrogate model on the current data, then
hands that model to the `acquisition` layer". ADR-0005 accordingly exposes that model as
`BayesianOptimizer.fitModel()`.

Implementing Phase 2 against real Ax 1.3.1 and BoTorch 0.18.1 showed that this handoff does not
work as a foundation for the alpha layer:

1. **The model does not exist early.** Until Ax leaves its initial space-filling phase its
   adapter is a random-sampling one with no surrogate in it (`AxBackend.fitModel()` raises
   there, and so does `predict()`: this is also why
   `tests/integration/test_end_to_end.py::test_predict_after_a_few_trials` fails today). The
   4-5-evaluation case, the one the alpha dial matters most for, is exactly this regime.
2. **It is in the wrong space.** Ax's model works on Ax's own transformed inputs (unit
   cube, one-hot/ordinal encodings, log transforms, hierarchical-space flattening) chosen by
   Ax, and its `Adapter`/`Surrogate` object graph is explicitly outside `ax.api`'s stability
   guarantees (`AxBackend.fitModel()` walks it on a best-effort basis, ADR-0005).
   Optimizing an acquisition function over that space, and mapping the optimum back to
   parameter values, would depend on all of it staying put.
3. **Constraints must be expressed in the optimizer's space.** `NonlinearConstraint`
   (ADR-0006) and linear constraints are written over parameters' natural values; to hand them
   to `optimize_acqf` they have to be rewritten in whatever space the optimizer works in, and
   that space has to be one `boptim` controls.

## Decision

The alpha layer builds and owns its surrogate model:

- `models/SearchSpaceEncoder` maps a `SearchSpace` onto the unit cube `[0, 1]^d` (log scales,
  integer and step grids, ordered and categorical choices) and decodes candidates back into
  parameterizations Ax accepts, including dropping parameters that `dependent_parameters`
  switch off (Ax rejects a parameterization that includes them).
- `models/encodeTrials` and `models/buildSurrogateModel` fit a `SingleTaskGP` on
  `BayesianOptimizer`'s own trial history in that encoding.
- `BayesianOptimizer.ask(alpha=...)` uses that model and never consults the backend's
  generation strategy. Candidates reach Ax only when the caller `tell()`s them, like any
  other externally-evaluated point (FR3).

`BayesianOptimizer.fitModel()` keeps its ADR-0005 meaning (Ax's own model, best effort) and its
docstring now says so explicitly, including that it is not the model `ask(alpha=...)` uses.

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| Extract Ax's fitted model (the original design) | Walk `GenerationStrategy -> Adapter -> Surrogate -> Model` and optimize over Ax's transformed space | No second model to maintain, but it is unavailable during the space-filling phase, tied to Ax internals, and forces the constraint and decoding logic to replicate Ax's transforms |
| Fit boptim's own surrogate (chosen) | Encode trials ourselves and fit a `SingleTaskGP` | One more model to keep correct, and it can differ from the one Ax would fit; but it works from the second trial on, is independent of Ax internals, and keeps every transform under `boptim`'s control |

## Consequences

- The alpha layer works with any `OptimizationBackend` (the unit tests run it against a fake
  one); `backends/ax` remains the only place that imports `ax.*` (ADR-0001).
- Two surrogates can exist for one study (Ax's for `ask()` without `alpha` and
  `parameterImportance()`; boptim's for `ask(alpha=...)`). They are fit on the same data but
  are not guaranteed to agree.
- `predict()` uses Ax's model when Ax has one and boptim's own otherwise. Ax has none during
  its initial space-filling phase, where `predict()` used to fail with Ax's `UnsupportedError`
  (the failure of `test_full_loop`); `AxBackend` now reports that as
  `PredictionUnavailableError` and `BayesianOptimizer.predict` falls back to boptim's model,
  cached until the next `tell()`. Below two completed trials it raises
  `PredictionUnavailableError` itself.
- Below two completed trials there is no model to fit, so `ask(alpha=...)` returns
  space-filling (scrambled Sobol) points that still satisfy every constraint.
- Phase 3's small-sample robustness work (weakly informative priors, low-confidence
  signalling) applies to `buildSurrogateModel`, where it now has a single place to live.

## Action items

1. [x] `models/SearchSpaceEncoder.py`, `encodeTrials.py`, `buildSurrogateModel.py`.
2. [x] `BayesianOptimizer.ask(alpha=...)` built on them; `fitModel()` docstring corrected.
3. [x] `predict()` falls back to boptim's model while Ax has none.
4. [ ] Decide whether `predict()` should always use boptim's model (one source of truth for
   `ask(alpha=...)` and `predict()`), and whether `parameterImportance()` should follow, once
   the alpha layer has seen real use.
