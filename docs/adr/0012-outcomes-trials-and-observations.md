# ADR-0012: Outcomes, trials and observations

**Status:** Accepted
**Date:** 2026-10-29
**Deciders:** project owner

## Context

[ADR-0009](./0009-study-policy-and-component-architecture.md) makes the study the object a
caller drives: it owns a problem, a goal, a policy and the history of trials. This ADR records
what the measured side of the problem and that history are made of: what an *outcome* is,
what a *trial* is and how it changes over time, and what an *observation* is (specification
5.4, 5.12 and 6; FR2, FR3, FR10, FR23 and FR24). The tasks that build them (P2-06, P2-07 and
P2-08) wait for this record.

Version 0.1.0 answers these questions with three small classes, and each answer breaks
something the specification requires:

- **One class for what is measured and what is wanted.** `Metric` is a name and a `minimize`
  flag, and `Objective` holds the metrics, optional weights and the `OutcomeConstraint`s. The
  direction is part of the metric, so a measured quantity cannot exist without one. A binary
  label (which phase a sample is in) has no direction, an exploration goal has no direction
  for anything, and a level-set goal has a level and not a direction. UC-4, UC-5 and UC-6 of
  the acceptance matrix (specification 7) cannot be said in this shape. Specification 5.7
  puts direction, weights and constraints in the goal for that reason.
- **A trial that exists only once it has a result.** `Trial` is `parameters`, `results`,
  `result_std` and an optional `trial_index`, and `BayesianOptimizer.tell()` creates it. A
  suggested point that has not been evaluated yet lives only inside the Ax client, and
  `AxBackend` matches a told point to it by the equality of the parameterization dictionary.
  The docstring of `AxBackend.attachTrial` states the consequence: if the caller changes a
  point between `ask()` and `tell()`, a new trial is attached and the first one stays
  running for good. There is no failed or abandoned state and no handle the caller can hold,
  so FR23 (pending, completed, failed and abandoned trials, asynchronous asking) cannot be
  met, and the lifecycle depends on a backend that ADR-0009 makes optional.
- **Observations that are bags of numbers.** `results` is a `dict[str, float]` and
  `result_std` an optional second dictionary. Nothing says that the value of a binary
  outcome is a label (FR24), and nothing refuses a value that is not finite. A partial
  `result_std` is handled by a rule hidden in `encodeTrials` (ignore every deviation and log a
  warning), where specification 5.4 says the surrogate's configuration decides, never a
  hidden rule.

Specification 5.4 and 6 set the target. An outcome has a name and a kind, and kinds are
registered, continuous and binary first (FR24). A trial has a stable integer identifier
assigned by the study, a parameterization, a state, an optional observation, optional
metadata and, when failed, a reason; three transitions are legal. An observation gives a
value for every outcome and, where known, a noise standard deviation. The rule of two
(specification 3.10) lets `outcome_kind` be a kind of its own, because two implementations
are needed from the start: UC-1 needs continuous outcomes and UC-5 needs binary ones.

Decision D-7 of the roadmap fixes the first catalogue of outcome kinds. Its proposed default,
continuous and binary, is the one this ADR uses, and it is the one the specification already
states in FR24 and 5.4. The owner has not yet accepted D-7 explicitly (the roadmap says that a
proposed default is used once accepted, and that its use is recorded in the pull request
description). If the owner answers differently, a new ADR supersedes the part that changes.

## Decision

1. **An outcome is a name and a kind.** `Outcome` (domain, P2-06) has a unique name within a
   problem, a registered kind and an optional description. It has no direction, no weight, no
   bound and no level: those say what the caller wants, and they live in the goal
   (decision 6).
2. **Outcome kinds are registered components.** The `outcome_kind` kind has two entries to
   start with, `continuous` and `binary` (D-7), registered under the name that is their
   serialization tag (the `type` key, specification 5.12). A kind provides what specification
   6 asks of it:
   - *Validation of observed values.* A continuous value is a finite number: NaN and
     infinity are refused. A binary value is a label, a `bool` or the integer 0 or 1, stored
     as 0 or 1. Anything else (a string, 2 for a binary outcome) is refused with an error that
     names the outcome.
   - *The rule on noise.* A noise standard deviation is meaningful for a continuous outcome
     and not for a binary one: a label has no standard deviation, and how a surrogate treats
     the uncertainty of labels is its own business (specification 5.6, P3-16).
   - *Its capability tag.* `outcome_kind:continuous` and `outcome_kind:binary`, registered in
     the vocabulary by P2-06. A surrogate declares the outcome kinds it supports, and the
     compatibility check meets them against the problem (specification 5.6, 5.12).
   Ordinal, count and categorical outcomes are plugins or later tasks (D-7): a plugin
   registers a kind and edits no core file.
3. **A trial is an immutable record with a lifecycle.** A `Trial` (domain, P2-07) has an
   integer identifier assigned by the study, unique and never reused, a parameterization, a
   state, an observation (decision 5), JSON-compatible free-form metadata and, when failed, a
   reason. A trial is a record and not a handle: the study validates its parameterization
   against the problem, and the trial itself knows nothing of any problem or backend. The
   states are *pending* (suggested, not yet reported), *completed* (has an observation),
   *failed* (the evaluation produced none, and carries a reason) and *abandoned* (withdrawn by
   the caller).
4. **Three transitions are legal, and they are the only ones.** A transition is a method that
   returns a new trial with the same identifier, and an illegal one raises `TrialStateError`
   (an error of the `PlanexsError` family, specification 10.10). The table lists every
   ordered pair of states, the sixteen of them:

   | From | To | Legal | Operation, or why not |
   |---|---|---|---|
   | pending | pending | no | No move: a trial stays pending until it is reported |
   | pending | completed | yes | `tell`: the evaluation finished and an observation is recorded |
   | pending | failed | yes | `fail`: the evaluation produced no observation, and a reason is recorded |
   | pending | abandoned | yes | `abandon`: the caller withdrew the suggestion |
   | completed | pending | no | Terminal: a reported trial is never reopened |
   | completed | completed | no | An observation is never replaced: a repeated evaluation is a new trial (a replicate) |
   | completed | failed | no | Terminal: an observation is never taken back |
   | completed | abandoned | no | Terminal: a completed evaluation cannot be withdrawn |
   | failed | pending | no | Terminal: a retry is a new trial with a new identifier |
   | failed | completed | no | Terminal: a result that arrives later is attached as a new trial |
   | failed | failed | no | A reason is never rewritten |
   | failed | abandoned | no | Terminal: the evaluation already ended |
   | abandoned | pending | no | Terminal: a withdrawn suggestion is not reopened |
   | abandoned | completed | no | Terminal: a result that arrives after withdrawal is attached as a new trial |
   | abandoned | failed | no | Terminal: the suggestion was withdrawn, not evaluated |
   | abandoned | abandoned | no | No move |

   *Creation is not a transition.* `ask` creates pending trials. `attach` (data gathered
   outside the study, FR3) creates a trial that is already completed, or already failed,
   because it never waited to be evaluated. A trial is never created abandoned, and
   `attach` never creates a pending one. Completed, failed and abandoned are terminal.
5. **An observation is one value per outcome, typed by its kind.** `Observation` (domain,
   P2-08) holds one value for every outcome of the problem and, where known, a noise
   standard deviation per outcome. It is validated against the list of outcomes: every
   outcome present, no extra name, each value valid for its kind, a deviation only where the
   kind allows one, and a deviation finite and not negative. A completed trial requires an
   observation, and a pending, failed or abandoned trial has none. Replicated evaluations
   of one parameterization are separate trials. Partial knowledge of noise (some
   observations with a deviation and some without) is not decided here: the surrogate's
   configuration decides (specification 5.4), and it is a configuration field and not a
   hidden rule (`NoisePolicyConfig`, P3-08).
6. **Direction and weights live in goals, not in outcomes.** The `optimize_v1` goal names the
   outcomes it optimizes, a direction for each, optional scalarizing weights and the
   constraints on outcomes (FR2, FR15, P4-03). The level-set goal has a level (P4-04) and the
   explore goal has none of these (P4-05). A goal is validated against the problem: the
   outcomes it names must exist and be of a suitable kind (specification 5.7). The same
   outcome can therefore be maximized in one study, constrained in another and only explored
   in a third.
7. **The 0.1.0 types are renamed, not extended.** `Trial`, `Metric`, `Objective`,
   `OutcomeConstraint` and `StudySnapshot` become `Legacy*` (P2-02), so that the new `Trial`
   can take its name. They keep working for the old path and are deleted in P11-03.
   `LegacyMetric` is replaced by an `Outcome` plus a direction in a goal, and the old
   `Trial.results` and `result_std` by an `Observation`.

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| Keep `Metric` as the outcome | A name and a `minimize` flag stay the measured quantity, with a kind added as a further field | The smallest change, and `LegacyObjective` keeps working as it is. But the direction stays welded to the measurement. A binary outcome, an exploration goal and a level-set goal each have to carry a direction they do not have, or ignore one, and a goal cannot be the home of direction and weights (specification 5.7). The same outcome cannot be maximized in one goal and constrained in another |
| Closed enumeration of outcome kinds | An `Outcome` with a `Literal["continuous", "binary"]` type and the validation written as one function that branches on it | The simplest code today. But an ordinal or a count outcome then means editing the core, against specification 3.3 and FR19, and it is the closed dispatch ADR-0009 counts as a defect of 0.1.0 (nine files that branch on concrete classes). Two implementations already exist, so the rule of two (3.10) allows a registry |
| Keep the 0.1.0 trial and the backend's bookkeeping | A trial exists once told, pending points are kept by the backend and matched by equality of the parameterization | No new type and nothing to port. But there is no failed or abandoned state and no identifier the caller can hold, the orphaned trial of `AxBackend.attachTrial` stays possible, and the lifecycle depends on Ax, which ADR-0009 makes an optional adapter |
| Free-form dictionaries for observations | `results` stays a `dict[str, float]` (or a `dict[str, Any]` for kinds that are not numbers) and whoever reads it checks it | Trivially JSON-compatible, and a plugin kind needs no schema. But every consumer validates again (finiteness, labels, noise), the rule on partial noise stays hidden, and an error has no field path. The reasons of ADR-0003 against plain dictionaries apply |
| A mutable trial with a status field | One mutable object whose `status` any code can assign | Fewer objects and no copy on a change. But a state can change without the study knowing, so the policy is not notified and the history version is not bumped. Transitions are unchecked, and specification 5.4 says the history is never silently rewritten |
| Outcomes with kinds, trials with a lifecycle, typed observations (chosen) | The three decisions above, with direction and weights in the goal | Meets FR2, FR3, FR10, FR23 and FR24 and the use cases that need them (UC-4, UC-5, UC-8). It adds types (`Outcome`, the kind, `TrialState`, `Observation`, a new `Trial`) and a copy on each transition, and it changes what a caller writes (a goal names its outcomes) |

## Trade-off analysis

The decision turns on one separation: what is measured is a fact, and what is wanted is a
choice. An outcome that carries its direction (the first option) cannot serve a goal that has
no direction, and the specification's own use cases include three of them. Putting the
direction in the goal costs one extra name to give: an optimize goal lists its outcomes and
their directions, and a mistake (an unknown outcome, a weight count that does not match) is
caught when the goal is validated against the problem and not when a point is told.

Making the trial an immutable record with a stable identifier fixes the failure that
`AxBackend` documents. `tell`, `fail` and `abandon` take the identifier of a trial the study
issued, so a caller who changes a point between `ask` and `tell` no longer creates an orphan.
What it gives up is that a trial cannot be edited in place: a retry or a late result is a new
trial. That is the intended behaviour, since it keeps the history append-only and keeps
identifiers meaningful, and the cost of one new small object per transition is negligible
beside the evaluation it records.

Registering outcome kinds costs a registry and two entries today, where a closed enumeration
costs a `Literal`. The registry is justified by the rule of two and by the reference use
cases: UC-1 and UC-5 need different kinds from day one, and a plugin that adds a kind must not
need a change to the core (FR19, UC-11).

What the decision explicitly leaves open: whether a binary outcome's predictions are latent or
response probabilities (the surrogate's contract, specification 5.6 and P3-16); how partial
noise is treated (surrogate configuration, P3-08); the schema of a trial's metadata, which
stays free-form JSON; and observations that are not one value per outcome. Comparisons and
rankings are a horizon use case (specification 7.3), and the specification says they change the
observation model and need their own ADR. This one does not forbid them: a new observation
type would be a new decision, not an edit of this one.

## Consequences

- P2-06 adds `domain/Outcome.py` and the `outcome_kind` kind with `continuous` and `binary`,
  one file each, and registers the tags `outcome_kind:continuous` and
  `outcome_kind:binary`. P2-07 adds `TrialState`, the new `Trial` and `TrialStateError`, and
  its tests cover every legal transition and every illegal one of the table above. P2-08 adds
  `Observation` and its coupling to a completed trial. The kind is covered by the registry
  contract test with no change to it (specification 10.6).
- The table above is the reference for P2-07. The trial-state tests are parametrized over its
  sixteen rows, and a change to the table (a new state, a new legal transition) is a new ADR
  that supersedes this one, because the table is part of what a saved study means (the
  snapshot stores every trial with its state, specification 5.11).
- Goals own direction, weights and outcome constraints (P4-03 to P4-05). Until P11-03 deletes
  them, `LegacyMetric`, `LegacyObjective` and `LegacyOutcomeConstraint` are the only place
  that holds them for the old path, and the new code never uses them. P4-03 keeps a
  conversion test that builds an `optimize_v1` goal from a `LegacyObjective` fixture, so
  that parity data exists for the later tasks.
- The study's operations take trial identifiers: `tell`, `fail` and `abandon` (P6-11), and
  `attach` for data gathered elsewhere (P6-12). The policy sees every trial in every state in
  a read-only view (P6-01), which is what makes batches and asynchronous asks work (FR10,
  FR23). The history only ever grows, and a state change bumps the history version that the
  surrogate cache keys on (P3-17, P6-09).
- Surrogates and likelihoods declare the outcome kinds they support (P3-11, P3-14, P3-16), and
  an exact Gaussian process on a binary outcome fails the compatibility check when the study
  is built, with a message that names a surrogate that works (specification 3.6, 5.12;
  P6-05).
- The snapshot stores outcomes and trials as dictionaries with a `type` key for kinds, so a
  later change of their serialized form is a change of the snapshot format and ships with a
  migration (specification 5.11, ADR-0014). There is no migration from the 0.1.0 snapshot
  (D-2).
- Until the owner accepts D-7, the catalogue of kinds is a proposal. A different answer
  changes P2-06 and the tags it registers, and is recorded by a new ADR.

## Action items

1. [x] Record the decision, with the table of legal trial-state transitions, and list the ADR
   in the navigation (P2-01).
2. [x] Rename the 0.1.0 types to `Legacy*` so that the new `Trial` can take its name (P2-02).
3. [ ] Add `Outcome` and the `outcome_kind` kind with `continuous` and `binary` (P2-06).
4. [ ] Add `TrialState`, the immutable `Trial` and `TrialStateError`, tested against every row
   of the table (P2-07).
5. [ ] Add `Observation`, validated against a list of outcomes (P2-08).
6. [ ] Give goals the direction, the weights and the constraints on outcomes (P4-03, P4-04,
   P4-05).
7. [ ] Make the study operations work on trial identifiers (P6-11, P6-12).
8. [ ] Ask the owner to accept D-7, or to answer it differently before P2-06 starts.
