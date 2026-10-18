# ADR-0001: Hybrid Ax + BoTorch architecture

**Status:** Superseded by ADR-0009
**Date:** 2026-09-23
**Deciders:** [project owner]

## Context

The project needs (a) a robust, low-maintenance implementation of the standard building
blocks of Bayesian optimization (search space definition, trial bookkeeping, a
no-manual-tuning default strategy, sensitivity analysis, persistence), and (b) a genuinely
custom, objective-independent exploration/exploitation control (FR5) that is not exposed by
any existing high-level BO library.

## Decision

Use Ax's `Client` API for everything in (a), and a custom BoTorch-based acquisition layer for
(b), connected through a fitted-model handoff rather than through Ax's generation-strategy
plugin mechanism.

## Options considered

| Dimension | Ax only | BoTorch only | Hybrid (chosen) |
|---|---|---|---|
| Effort for FR1-FR4, FR6, FR7, FR9, FR11, FR14, FR15 | Low (native) | High (all hand-rolled) | Low (native) |
| Effort for FR5 (alpha dial) | High (fighting the framework) | Low (native building blocks) | Low (native building blocks) |
| Long-term maintenance | Low | High | Medium |
| Risk from upstream API churn | Medium (already changed once) | Low (BoTorch is lower-level, more stable) | Medium, mitigated by the domain-layer boundary (design philosophy #2) |

## Trade-off analysis

Ax-only would mean either not delivering FR5 properly, or fighting Ax's generation-strategy
internals to inject a custom acquisition function, which is possible but brittle and poorly
documented for this specific use case. BoTorch-only would mean re-implementing trial
bookkeeping, persistence, and sensitivity analysis that Ax already provides solidly. The
hybrid keeps each library doing what it is strongest at.

## Consequences

- `backends/ax` is the only module allowed to import from `ax.*`.
- `acquisition` is the only module allowed to build custom `botorch.acquisition.*` subclasses.
- Any future backend swap (e.g. moving off Ax entirely) only requires rewriting
  `backends/ax` and the parts of `acquisition` that read a fitted model, not `domain` or `api`.

## Action items

1. [x] Implement `backends/ax/AxBackend.py` against the pinned Ax version. (Phase 1)
2. [ ] Implement `acquisition/ExplorationExploitationAcquisition.py` against the pinned
   BoTorch version. (Phase 2)
3. [ ] Pin compatible Ax/BoTorch/PyTorch/GPyTorch versions together in `pyproject.toml`
   (currently version *ranges*; exact, resolver-verified pins belong in the committed
   `uv.lock`, which requires running `uv sync` against a real package index).

## Phase 1 implementation note

`backends/ax/AxBackend.py`'s `fitModel()` walks Ax's internal `GenerationStrategy` ->
`Adapter`/`ModelBridge` -> `Surrogate` -> BoTorch `Model` object graph to satisfy the
`fitModel()` escape hatch (ADR-0005) ahead of Phase 2's acquisition layer existing. This is
explicitly *not* part of Ax's stable `ax.api` surface this ADR otherwise restricts `backends/ax`
to; it is isolated to that one method, documented there, and should be the first thing checked
if `fitModel()` breaks against a newer pinned Ax version.
