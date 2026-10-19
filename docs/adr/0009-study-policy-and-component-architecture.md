# ADR-0009: Study, policy and component architecture

**Status:** Accepted
**Date:** 2026-10-17
**Deciders:** project owner
**Supersedes:** ADR-0001, ADR-0005, ADR-0006, ADR-0007, ADR-0008

## Context

Version 0.1.0 is a Bayesian-optimization library built on Ax. It was named `boptim` until the
rename recorded below. ADR-0001 chose a hybrid: Ax's `Client` for trial bookkeeping, the
default strategy, sensitivity analysis and persistence, plus a custom BoTorch acquisition
layer for the exploration/exploitation dial (FR5), joined by a fitted-model handoff. The
later ADRs recorded what happened at the seam between the two halves:

- ADR-0007: Ax's model does not exist during its initial space-filling phase, lives in a
  space Ax controls, and cannot take constraints written on natural values. The acquisition
  layer therefore fits its own surrogate, and one study can hold two surrogates that need not
  agree.
- ADR-0006 and ADR-0008: Ax's parameter constraints are linear inequalities only (tested
  against Ax 1.3.1), so it cannot enforce a nonlinear constraint or an equality. `ask()`
  switches to the custom layer, with a warning, whenever the search space holds a constraint
  Ax cannot enforce: the strategy that answers a call depends on the contents of the space.
- ADR-0005 and ADR-0007: the escape hatch `fitModel()` returns Ax's model, which, as
  ADR-0007 found, is not the model that `ask(alpha=...)` uses.

The measurements of the roadmap baseline (roadmap section 2.1, commit `f5829e4`) show the
same pattern in the code: 39 numeric literals outside {0, 1, -1, 2} in 11 of 58 source files,
about 30 of them policy values (restarts, sample counts, tolerances, caps, seed constants)
hidden in the code (most cannot be set at all, and the few that can are not saved with a
study); engine classes (`SingleTaskGP`, `Standardize`,
`fit_gpytorch_mll`, `optimize_acqf` and others) named inside algorithms in 5 files; and 9
files that branch on concrete parameter or constraint classes, so that a new kind of
parameter or constraint cannot be added without editing them.

The scope has also changed. The specification (v0.3, section 1.1) defines planexs as a
toolkit for sequential experimental design with probabilistic surrogate models, of which
optimization is one goal among several. Its acceptance matrix (section 7) includes level-set
estimation from binary labels (UC-5), exploration with no optimization target (UC-6), a
surrogate used with no loop (UC-7) and components defined outside the package (UC-11). In
0.1.0 the objective is the centre of the domain and the engine is fixed, so none of them can
be delivered without another special case.

Two records fix how this ADR relates to the others. Section 11 of the
[project specification](../PROJECT_SPECIFICATION.md) says that where an existing ADR
conflicts with it, the specification wins, and the [roadmap](../ROADMAP.md) carries a task to
supersede that ADR: this ADR is that task for ADR-0001, 0005, 0006, 0007 and 0008. And the
project was renamed (decision D-1 of the roadmap, taken by the owner on 2026-10-11): the
package is `planexs`, for *PLANification EXpérimentale Séquentielle* (the French for
sequential experimental design), because `boptim` named only one of the goals. The rename
covers the package, the imports, the identifiers, the configuration and the living documents.
ADR-0001 to ADR-0008 and the 0.1.0 changelog keep the former name as historical records,
which is why they say `boptim` and this ADR says `planexs`.

## Decision

planexs follows the architecture of the specification (sections 3, 5 and 6):

1. **A study, a goal and a policy.** A *study* is the object the caller drives. It owns the
   problem (search space and outcomes), a goal, a policy and the trial history, and offers
   ask, tell, fail, abandon, attach, analyses, stopping checks, surrogate access and save and
   load. A *goal* is typed configuration (optimize, level set, explore, ...), not an
   algorithm: it parameterizes acquisitions and analyses and proposes default recipes. A
   *policy* decides which parameterizations to evaluate next. It is either composed (a
   design, a surrogate, an acquisition and an acquisition optimizer) or opaque (any object
   that honours the policy contract).
2. **Every choice is a registered component.** Anything on which a reasonable user could want
   a different answer (parameter, constraint and outcome kinds, the surrogate with its
   kernel, likelihood, transforms and fitter, the acquisition, the acquisition optimizer, the
   design, the goal, the analysis, the stopping criterion, the policy, the repository) is a
   named component of a *kind*, selected by name in configuration. Components declare the
   capabilities they need and support, and the combination is checked when a study is built.
   New components are added from user code or from a plugin, with no change to planexs.
3. **Ax stops being the foundation.** BoTorch and GPyTorch are the engine: planexs registers
   builders for their classes and writes new mathematics only as native subclasses. Ax
   becomes one possible policy, an opaque adapter installed through an extra. Only the
   `adapters` layer imports `ax.*`, and the core imports and runs without Ax.
4. **The custom acquisition layer becomes ordinary components.** The blend of exploitation
   and exploration driven by one weight becomes one registered acquisition among others
   (FR5), optimizing an acquisition under constraints becomes registered acquisition
   optimizers, and the encoder and the surrogate become the `encoding` and `surrogates`
   layers. `AlphaAcquisitionStrategy` is split between optimizers and the composed policy.
5. **Layers own one job.** The package is organized in the layers of specification 5.1, with
   dependencies pointing down only. `core` and `domain` import no machine-learning library.
6. **Migration by strangler, not rewrite.** The core is refactored and the periphery is kept
   (the parameter and constraint model, the search-space encoder, the constraint machinery).
   The new packages are built beside the old code, which keeps working and is deleted only
   once the twelve reference use cases of specification 7 pass on the new one (roadmap
   sections 1 and 2.3).
7. **Supersession.** This ADR supersedes the five records below. Their status lines say so
   and their text is untouched (specification 11).

| Superseded | What it decided | Replaced by |
|---|---|---|
| [ADR-0001](./0001-hybrid-ax-botorch-architecture.md) | A hybrid: Ax for everything standard, a custom BoTorch acquisition layer, joined by a fitted-model handoff; only `backends/ax` imports `ax.*` | Ax as an optional adapter behind the policy seam, BoTorch as the engine (specification 1.4, 5.9). Still true: new mathematics is written as native BoTorch subclasses |
| [ADR-0005](./0005-escape-hatches-to-ax-and-botorch.md) | `axClient` and `fitModel()` as public escape hatches of `BayesianOptimizer` | Specification 5.13: the study's surrogate exposes the BoTorch model the policy actually uses, an adapter exposes its own client, and a custom component is the preferred escape hatch (P6-14, P10-07) |
| [ADR-0006](./0006-nonlinear-constraint-forces-custom-layer.md) | A `NonlinearConstraint` makes `ask()` switch to the custom layer, with a forced `alpha` and a warning | Needs and supports checked when the study is built, and recipes chosen from them; a study never switches policy afterwards (specification 3.6, 5.9, 5.12; ADR-0013, P4-01) |
| [ADR-0007](./0007-acquisition-layer-fits-its-own-surrogate.md) | The acquisition layer fits its own surrogate on its own encoding | Kept in substance and generalized: the `encoding` and `surrogates` layers. There is no second surrogate: the study's surrogate is the one the policy uses (specification 5.6, 5.13) |
| [ADR-0008](./0008-equality-linear-constraint-forces-custom-layer.md) | An equality `LinearConstraint` also forces the custom layer, through `requires_custom_acquisition_layer` | The capability tag `constraints:equality`, supported by the planexs optimizers and not by the Ax adapter (P5-11, P10-08); the property is deleted with the legacy path (P11-03) |

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| Keep Ax as the foundation and add features on top | Continue the 0.1.0 design: Ax owns trials, strategy, analyses and persistence, and each new need (a goal, an acquisition, a constraint kind) is one more special case beside it | Least effort now, and it keeps what Ax does well. But ADR-0006, 0007 and 0008 each ended by routing around Ax into code planexs controls, and ADR-0001 already found that injecting a custom acquisition function into Ax's generation strategy is possible but brittle and poorly documented. Every extension stays a change to the core, so FR19 and UC-11 cannot be met |
| Wrap BoTorch directly without registries | Drop Ax, build on BoTorch and GPyTorch, and write the choices of model, acquisition and optimizer in code | Removes the Ax dependency and the two-engine problem. It still has to write trial bookkeeping, persistence and parameter importance itself, like the chosen option, and it keeps every choice hard-wired: a different kernel or acquisition means editing the library, with no capability check and no named configuration, so FR18, FR19 and UC-11 are not met |
| The design of the specification (chosen) | A study, a goal and a policy made of registered components selected by name; BoTorch as the engine; Ax as an optional adapter | The only option that meets FR18, FR19, FR25 and the twelve use cases. It is the most work (the whole roadmap), it makes planexs the owner of what Ax provided, and it adds concepts (kinds, capability tags, registries) that must stay justified (specification 3.10) |

## Trade-off analysis

The three Ax-related ADRs of 0.1.0 (0006, 0007, 0008) each found a limit of Ax as the
foundation and each fixed it the same way: by routing around Ax into code planexs controls.
Keeping Ax as the foundation continues that pattern one special case at a time, which is a
mistake the specification records as having happened in this project: a second engine and a
second surrogate built beside the first to work around a limit, instead of moving the seam so
that both can use one (specification 12.4). Wrapping BoTorch without registries removes the
second engine but not the cause, since the choices stay hard-wired: it costs about as much as
the chosen option and delivers less of it.

The chosen option moves the seam. The surrogate, the acquisition, the optimizer and the
policy become components behind contracts, and Ax becomes one implementation of the policy
contract. What it gives up:

- **Ax's work becomes planexs' work.** ADR-0001 rated trial bookkeeping, persistence and
  sensitivity analysis as native with Ax and as hand-rolled without it. Under this decision
  planexs writes the trial lifecycle (P2-07), the snapshot (roadmap Phase 8) and parameter
  importance (P7-05) itself.
- **The default policy is planexs' own.** The default strategy used to be Ax's. The recipes
  of the goals (P6-07, P6-08) replace it, so their quality has to be shown by the use-case
  tests and not inherited; the owner reviews them after Phase 9 (D-9). Ax stays selectable as
  an adapter for anyone who prefers its strategy.
- **More concepts.** Kinds, capability tags and registries are more to learn and to keep
  correct. The specification limits them with the rule of two (3.10) and the reference use
  cases (section 7).
- **Porting risk.** Moving the alpha layer changes where numerical behaviour lives. Parity
  tests that run the old and the new code on a fixed problem are required before the old
  code is deleted (roadmap 2.3).

## Consequences

- Every choice is a registered component selected by configuration (specification 3.3), and a
  component name is a permanent identifier: a change that alters results gets a new `_vN`
  name (3.7). Adding a component follows the checklist of specification 10.14 and edits no
  core file.
- Layer dependencies point down only (specification 5.1) and are checked in CI (P1-16).
  `core` and `domain` import no machine-learning library. Only `adapters` imports `ax.*`, and
  `ax-platform` moves from a dependency to an optional extra (P10-09).
- A study stores its fully resolved configuration, and every policy value is a documented
  field of a configuration schema: no literal policy value is added (specification 3.4, 10.7;
  the checker of P0-05 enforces it against a baseline that only shrinks).
- The 0.1.0 API (`BayesianOptimizer`, `AxBackend`, `OptimizationBackend` and the legacy domain
  types) stays working and is never extended. It is renamed `Legacy*` where the new design
  reuses a name (P2-02) and deleted in roadmap Phase 11, after the reference use cases pass.
  The roadmap proposes no compatibility shim for it (D-2, awaiting the owner's acceptance).
- The facts established by ADR-0006, 0007 and 0008 remain inputs although their decisions are
  replaced: Ax 1.3.1 takes only linear inequalities as parameter constraints, its model is
  absent during its initial phase, and the measurements of the exploration weight in ADR-0006
  stay as evidence when choosing it.
- Whether the Ax adapter ships in a release is the owner's decision (D-3, which proposes an
  optional extra). This ADR decides only that Ax is not the foundation.
- The mechanisms this ADR relies on are recorded by the ADRs that follow: registries,
  capability tags and stable names (ADR-0010), policy values, resolved configuration and
  seeds (ADR-0011), outcomes, trials and observations (ADR-0012), goals, policies and recipes
  (ADR-0013), and the snapshot format (ADR-0014). ADR-0003 (Pydantic domain models) is not
  superseded: P1-04 completes it.
- ADR-0001, 0005, 0006, 0007 and 0008 keep the name `boptim` and their original text. Where
  they and the specification disagree, the specification wins (specification 11).

## Action items

1. [x] Set the status line of ADR-0001, 0005, 0006, 0007 and 0008 to "Superseded by
   ADR-0009", leaving their text untouched, and list ADR-0009 in the navigation (P1-01).
2. [ ] Record the mechanisms above: ADR-0010 (P1-02), ADR-0011 (P1-03), ADR-0012 (P2-01),
   ADR-0013 (P4-01) and ADR-0014 (P8-01), and complete ADR-0003 (P1-04).
3. [ ] Build the layers beside the legacy code: `core` (roadmap Phase 1), `domain` (Phase 2),
   `encoding` and `surrogates` (Phase 3), goals and designs (Phase 4), acquisitions and
   optimizers (Phase 5), policies and the study (Phase 6).
4. [ ] Port the alpha layer as registered components, each with a parity test against the
   legacy code (P5-07, P5-08, P5-12).
5. [ ] Show that the architecture works with the twelve reference use cases (Phase 9), then
   port Ax to the optional adapter (P10-07 to P10-10, once D-3 is answered) and delete the
   legacy path (P11-01 to P11-03).
