# ADR-0003: Pydantic domain models, independent of every engine

**Status:** Accepted
**Date:** 2026-09-23
**Deciders:** project owner

## Context

The domain layer holds the objects a caller uses to describe a problem and its history:
parameters (`Range`, `Choice`, `Derived`), parameter constraints (`LinearConstraint`,
`NonlinearConstraint`), the search space, outcomes and goals, trials and observations, and the
snapshot that stores them. The decision was taken with the first domain code of 0.1.0 and
recorded only in outline; this text records it in full (roadmap task P1-04), against the
architecture of [ADR-0009](./0009-study-policy-and-component-architecture.md) and the current
specification.

These objects have three needs that the choice of representation has to meet:

- **Validation.** Their invariants must hold from construction on: a default lies inside the
  bounds and on the step grid (FR13), names are unique, one weight is given per metric, a
  constraint refers to a metric the objective has. An error names the object and the field.
- **Exact JSON round trip.** A study is saved as one JSON document and reloaded without loss
  (FR11, FR12, specification 5.3 and 5.11), through collections of an abstract type
  (`list[Parameter]`, `list[Constraint]`) whose concrete members differ in their fields.
- **Independence from every engine.** The domain contains data, validation and serialization
  only. It imports no machine-learning library and knows nothing about engines or adapters
  (specification 3.9, 1.4). ADR-0009 makes Ax an optional adapter, and the first specification
  already recorded that Ax's own public API had changed once, to `ax.api.client.Client` and
  `ax.api.configs`, while it was being written.

The last need rules out borrowing the engine's own objects, and the facts agree. None of the
parameter configs of Ax 1.3.1 (`RangeParameterConfig`, `ChoiceParameterConfig`,
`DerivedParameterConfig`) has a `default` field (checked against the installed version), and
Ax's parameter constraints are linear inequalities (ADR-0006, ADR-0008), so Ax's types cannot
hold a default (FR13), a nonlinear constraint (FR17) or an equality, all of which planexs
needs.

The 0.1.0 code already follows the decision in most respects. Every domain class is a Pydantic
model with `extra="forbid"`, `Metric`, `Objective`, `OutcomeConstraint` and the constraints are
frozen, and invariants are `model_validator(mode="after")` checks. `JsonStudyRepository` writes and reads the snapshot
with `model_dump_json` and `model_validate_json`. It also departs from the specification in
two places, listed in the Consequences.

## Decision

1. **Domain objects are Pydantic v2 models.** Unknown fields are rejected (`extra="forbid"`),
   invariants are validators that raise `ValueError` naming the object and the field, and value
   objects are frozen (`Metric`, `Objective`, the constraints, and the trial of the new
   design). `SearchSpace` is mutable by design, so that a study can be built up one parameter
   at a time. The public constructors keep the shape a caller expects (`Real("x", 0.0, 1.0)`)
   and call `super().__init__` with field names.
2. **The domain is pure** (specification 3.9). A domain module imports no machine-learning
   library, no engine, and no planexs layer above `core`. Anything that maps a domain object to
   the vocabulary of an engine, such as an Ax constraint string or a set of Ax config keyword
   arguments, lives with that engine's adapter and never in the domain.
3. **Serialization is explicit and type-tagged where a collection holds an abstract type.**
   Pydantic serializes a field by its declared type, so a `list[Parameter]` would drop the
   fields of each subclass and could not rebuild the abstract class. Each parameter and each
   constraint therefore has a `toDict()` that writes a stable type tag, and the matching
   dispatch rebuilds the right class, wired into `SearchSpace` by a `field_validator` in
   `before` mode and a `field_serializer`. Convenience constructors (`Real`, `Integer`,
   `Categorical`, `Boolean`, `Fixed`) build the base kinds and are not round-trip types: a
   `Real` reloads as an equivalent `Range`. The dispatch is by registry lookup (ADR-0010); in
   0.1.0 it is still a chain of tests that P2-04 and P2-05 replace.
4. **Pydantic is for objects with invariants and a JSON form.** Small internal records that
   need neither stay frozen dataclasses, as `ParameterEncoding`, `PredictionResult` and
   `ReproducibilityMetadata` are today. This ADR does not change that.

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| Standard-library dataclasses | `@dataclass(frozen=True)` with the checks written in `__post_init__` | No dependency, light and familiar. But field types are not checked at run time, every invariant and every error message is hand-written in each class, and so is the JSON round trip, with no field path in its errors. Pydantic is already required for configuration schemas (specification 10.8), so this adds a second idiom and saves no dependency |
| attrs | Third-party classes with validators, converters and `slots` | Mature, fast and richer than dataclasses. But building validated nested objects from JSON needs a second library or hand-written code, and Pydantic is required anyway for configuration, so it would add a dependency and a second way of writing the same thing |
| Plain dictionaries | Dictionaries with a documented key convention, perhaps typed with `TypedDict` | Trivially JSON-compatible and dependency-free. But nothing is validated or named at run time: a misspelled key passes silently, a `TypedDict` helps the type checker and not the program, and every consumer re-checks the same invariants, so they end up scattered |
| The engine's own config objects | Use Ax's `ax.api.configs` parameter configs as the domain | No duplication. But the domain would import Ax, against specification 3.9 and ADR-0009, would follow every change of Ax's API, and is too small: it has no default value, no nonlinear constraint and no equality |
| Pydantic v2 models (chosen) | Domain objects as Pydantic models, with explicit type tags for abstract collections | Declarative validation with field-path errors, frozen models, unknown-field rejection and JSON dump and load in one library that the configuration schemas already use. It brings Pydantic's own behaviour to learn, and polymorphic collections need explicit handling |

## Trade-off analysis

Pydantic is the only option that gives validation with field-path errors and a JSON round trip
for nested objects without hand-written code or a second library, and the project needs it
for configuration schemas (specification 10.8) in any case. Using it for the domain keeps one
way of writing a validated, serializable object. The rejected options each shift work to the
caller or to every class: dataclasses and dictionaries leave validation and serialization to
hand-written code, attrs needs a companion library for the part that matters here, and
borrowing Ax's objects breaks independence and cannot hold what planexs needs.

What it gives up:

- **Pydantic's behaviour has to be learned.** The code already records one case. Smart-mode
  union matching means the order of `bool | int | float | str` in `Parameter.default` matters:
  `bool` first so that `True` is not read as an int, `int` before `float` so that `3` stays
  `3`.
- **Abstract collections need explicit code.** A list of the abstract `Parameter` cannot be
  serialized or rebuilt by declared type, which is why `toDict()` and a dispatch exist.
  Pydantic also rebuilds a model whose `__init__` is overridden through that `__init__`,
  called with field names, which a convenience constructor with other argument names (`Real`
  takes `min_value` and `max_value`, not `bounds`) cannot satisfy; that is why the convenience
  constructors are not round-trip types.
- **The domain depends on Pydantic.** The purity rule bans machine-learning libraries and
  engines, not Pydantic. A later change of library would touch every domain class. The public
  constructors and the explicit `toDict()` form limit what callers see of it.

## Consequences

- A domain object is data, validation and serialization only (specification 3.9). The layering
  check (P1-16) fails on an `ax`, `torch`, `botorch` or `gpytorch` import in `domain`, and the
  purity tests (P2-12) import `planexs.domain` in a fresh process and fail if any of them is
  in `sys.modules`.
- Two parts of the 0.1.0 domain depart from this and are removed by the roadmap, not kept as
  designed. The domain still carries five methods that name Ax: `toAxKwargs()` on `Range`,
  `Choice` and `Derived`, `toAxParameterConstraintString()` on `LinearConstraint` and
  `toAxOutcomeConstraintString()` on `OutcomeConstraint`. P2-03 turns them into functions of
  the Ax backend. And `StudySnapshot` imports `ReproducibilityMetadata` from `persistence`,
  a higher layer; P2-02 moves the legacy snapshot there.
- Type tags and the dispatch of parameters and constraints go through registries (P2-04,
  P2-05, ADR-0010), under the key `type`. An unknown type is an error that lists the
  registered ones, and a new kind is added by registering it and not by editing a chain of
  tests in the same change.
- The new domain types follow the same pattern: outcomes with registered kinds (P2-06),
  trials with a lifecycle (P2-07), observations (P2-08) and the problem (P2-10), as ADR-0012
  records.
- Snapshots embed domain objects as dictionaries, so a change to the serialized form of one is
  a change of the snapshot format and ships with a migration (specification 5.11).

## Action items

1. [x] Write this ADR in full and mark it Accepted (P1-04).
2. [ ] Remove the engine vocabulary from the domain (P2-03) and move the legacy snapshot out
   of it (P2-02).
3. [ ] Dispatch parameter and constraint types through registries, under the key `type`
   (P2-04, P2-05).
4. [ ] Enforce the rule: the layering check (P1-16) and the purity tests (P2-12).
