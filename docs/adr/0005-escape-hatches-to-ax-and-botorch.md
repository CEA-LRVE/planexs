# ADR-0005: Escape hatches to Ax and BoTorch

**Status:** Superseded by ADR-0009
**Date:** 2026-09-23
**Deciders:** [project owner]

## Context

`BayesianOptimizer` is a deliberately high-level facade (design philosophy #2): it does not,
and should not, expose every capability of Ax or BoTorch individually. Some callers will need
something the facade does not cover (`PROJECT_SPECIFICATION.md` section 4.7 lists several
concrete, real examples). A facade with no way out forces those callers to either fork
`boptim` or drop it entirely for a whole project just to reach one feature underneath it.

## Decision

`BayesianOptimizer` exposes two escape hatches directly: `axClient` (the live
`ax.api.client.Client` instance backing this optimizer, when the backend is `AxBackend`) and
`fitModel()` (the current fitted BoTorch model). Both are first-class, documented parts of the
public API, not private/internal attributes a caller has to know to reach into.

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| No escape hatch | Only what `BayesianOptimizer` explicitly wraps is reachable | Simplest surface, but strands any caller who needs one more thing Ax or BoTorch already has |
| Escape hatch via private attribute | e.g. `bo._backend._client`, undocumented | Technically possible in Python, but an undocumented private attribute is not a supported contract; it can change without notice |
| Escape hatch as a public, documented property (chosen) | `bo.axClient`, `bo.fitModel()` | A little more public surface to keep stable, in exchange for never trapping a caller behind the facade |

## Trade-off analysis

The cost is that `axClient`'s and `fitModel()`'s own stability is now, transitively, Ax's and
BoTorch's stability, not `boptim`'s. That is an explicit, accepted trade: it is scoped to
callers who opt into using it, and it does not weaken any guarantee the rest of the public API
makes. The alternative (no escape hatch) trades a cleaner-looking API for a real risk of
callers hitting a wall and abandoning the library for their whole project over one missing
capability, which is a worse outcome for a library meant to be "accessible to all" (section
1.1).

## Consequences

- `axClient` raises `TypeError` (not `None`) when `backend` is not an `AxBackend`, since
  silently returning `None` would push the "is this available" check onto every caller instead
  of failing where the mismatch actually is.
- `fitModel()` is available regardless of backend, since it is defined at the
  `OptimizationBackend` interface level (section 5.3), not Ax-specific.
- Anything reached through `axClient` is, by definition, outside what `boptim` validates or
  keeps in sync with its own domain objects; this is documented on the property itself
  (section 5.7), not just here.

## Action items

1. [x] Add `axClient` and `fitModel()` to `BayesianOptimizer` in the same change that
   implements `AxBackend` (Phase 1).
2. [ ] Document at least one worked example of each in `examples/`. `axClient` is covered by
   `examples/escape_hatch.py` (Phase 1); `fitModel()`'s more interesting worked example (a
   caller writing their own acquisition function against it) is more natural once Phase 2's
   `acquisition/` layer exists to contrast it with, so it is deferred there. `examples/
   escape_hatch.py` still calls `fitModel()` directly and prints what it returns, since the
   escape hatch itself is fully functional in Phase 1.

## Phase 1 implementation note

`AxBackend.fitModel()` reaches the fitted BoTorch model by walking Ax's internal
`GenerationStrategy` -> `Adapter`/`ModelBridge` -> `Surrogate` -> `Model` object graph, which
this ADR's own "Trade-off analysis" already anticipates ("transitively, Ax's... stability, not
`boptim`'s"). It is defensive (a clear `RuntimeError` naming the likely causes, rather than an
opaque `AttributeError`) precisely because that internal chain is not part of Ax's own
documented `ax.api` stability guarantees.
