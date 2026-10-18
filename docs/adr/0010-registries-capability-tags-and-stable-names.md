# ADR-0010: Registries, capability tags and stable names

**Status:** Proposed (the entry-point group name awaits the owner's acceptance of D-10)
**Date:** 2026-10-17
**Deciders:** project owner

## Context

[ADR-0009](./0009-study-policy-and-component-architecture.md) decides that every choice a
user could reasonably make differently is a registered component selected by name in
configuration, and that components declare what they need and support. This ADR records the
mechanism: how components are registered, named, described, checked against one another and
discovered (specification 3.3, 3.6, 3.7 and 5.12).

In 0.1.0 a choice is made by code that names the choice:

- `parameterFromDict` and `constraintFromDict` are chains of `if kind == "range"` style
  tests, and `SearchSpaceEncoder` branches with `isinstance` on `Range`, `Choice` and
  `Derived`. Adding a kind of parameter or constraint means editing each of them: the stub
  ADR-0003 notes that it has to be added to its dispatcher "in the same change".
- `buildSurrogateModel` calls itself "the single swap point" for a scalable or sparse
  Gaussian process, and names `SingleTaskGP`, `Standardize` and `fit_gpytorch_mll` in its body. The
  swap point exists only in a docstring, which specification 3.3 lists as a defect.
- A choice has no stable name. A saved study records its `backend_kind` and its domain
  objects, not which components produced its suggestions.

Four requirements follow from the specification, and nothing in 0.1.0 meets them:

- select every component by name in configuration (FR18), and add new ones from user code or
  from an external package with no change to planexs (FR19, exercised by UC-11);
- reject a combination of components that cannot work together when the study is built, with
  an error that says how to fix it (FR25, specification 3.6), instead of failing deep inside
  an engine call;
- keep a saved study meaning what it meant: the names in a snapshot must keep selecting the
  same behaviour (specification 3.7, FR12);
- load a snapshot without executing code from it (specification 8): a name may only select
  among components that are already registered.

## Decision

One mechanism serves every kind of component (specification 5.12 and 6):

1. **One registry per component kind.** A kind (surrogate, acquisition, design, goal, and the
   others listed in specification 6) has one registry. It is created by name through one
   function that also records it in a kind table, so that tests and the compatibility check
   can enumerate every kind. Looking up an unknown name fails with an error that lists the
   available names, and registering a duplicate name is an error. The set of kinds is itself
   open: a new kind needs two implementations (specification 3.10) and an ADR.
2. **Registration by decorator, at import time.** A component is registered with a decorator
   in the style `@registerSurrogate("single_task_gp_v1")`. The entry holds the builder, the
   configuration schema, a one-line description and the declared needs and supports. An
   entry without a description is rejected.
3. **Names are permanent identifiers.** A name is lowercase snake_case with an optional `_vN`
   suffix, a pattern defined once; any other name is rejected. A change that alters results
   gets a new name (`..._v2`) and the old name keeps its behaviour, so that a snapshot keeps
   its meaning (specification 3.7).
4. **Needs and supports are tags from a documented vocabulary.** A tag is a lowercase
   `group:name` string, for example `posterior:gaussian`, `space:categorical` or
   `goal:level_set`. A tag outside the vocabulary is rejected when a component is registered,
   which catches typos, and the error lists the known tags of the same group. The vocabulary
   starts empty and each task that adds a kind or a component registers the tags it uses. A
   component may refine its tags from its own validated configuration. The problem and the
   goal contribute tags as well.
5. **A compatibility check when a study is built.** Every need must be supported by some
   participant: the problem, the goal, or a component of the policy. All unmet needs are
   reported together in one error that says who needs each one and which registered
   components would support it. A study never changes its policy afterwards (specification
   5.9).
6. **Plugin discovery by entry point.** An external package registers components by
   decorating them. planexs imports every entry point of the group `planexs.plugins`, plus
   the modules a caller names explicitly. A failing plugin raises an error that names the
   plugin and the original error, and loading twice is harmless. Plugins are trusted code.
   The group name is the proposed default of D-10 (roadmap section 17): it is used here until
   the owner accepts it or names another, which is why this ADR stays Proposed until then.
7. **The population rule.** A decorated component is registered only when its module has been
   imported. Every subpackage's `__init__` imports each of its concrete modules, one line per
   file, and a test fails on a decorated module that its `__init__` does not import.

A plugin declares its entry point in its packaging metadata, and the module it names
registers its components when it is imported (the shape only; the decorators are defined by
the code):

```toml
[project.entry-points."planexs.plugins"]
my_kernels = "my_package.kernels"
```

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| Hydra-style import paths as the only mechanism | A configuration names a class by its import path (`_target_: package.module.Class`) and the class is instantiated, with no registration step | No registration, and any class works. But the name is a code path: moving or renaming a class breaks every saved study (specification 3.7), there is nowhere to attach needs, supports, a description or a schema so the compatibility check (FR25) cannot run before instantiation, no list of available names exists for an error message, and a snapshot would choose what gets imported, against the rule that loading it executes no code from the file (specification 8). It may be added later as a convenience for level 3 of specification 3.8, in Python code and never as what a snapshot stores (the roadmap plans no dependency on Hydra, D-4) |
| Inheritance with factory methods | An abstract base class per kind with a factory, the implementation found by walking `__subclasses__()` or by an `if`/`elif` factory | Natural for type checkers and no new concept. But an `if`/`elif` factory is the closed dispatch of 0.1.0, where a plugin cannot add a branch without editing it, and `__subclasses__()` gives no stable name, no place for a description or tags, no duplicate detection, and only finds classes already imported. Adding names, tags and a lookup to make it work gives the chosen option, written less explicitly |
| Plain dictionaries | One module-level `dict` per kind from a name to a class, edited by hand | The simplest thing that works, and explicit. But each dictionary is a central file that every new component must edit (specification 1.3: components are added without changing the core), each kind reimplements lookup, duplicate detection and the error message, and a bare class carries no description, schema or tags |
| One registry per kind with decorator registration and capability tags (chosen) | The mechanism of the Decision above | Meets the four requirements. It needs more machinery (a registry, a kind table, an entry record, a tag vocabulary, a compatibility check, plugin discovery) and registers only on import, so a forgotten import silently loses a component. Names cannot change, so `_vN` versions accumulate. Tags are strings that a component could declare falsely |

## Trade-off analysis

Each alternative fails at least one of the four requirements. Import paths cannot carry
metadata, cannot keep names stable, and let a file choose what is imported. An `if`/`elif`
factory is closed, so FR19 fails. A dictionary or `__subclasses__()` can be extended, but a
bare class gives the compatibility check nothing to read (FR25) and nothing enforces unique,
valid and permanent names. The registry is the smallest mechanism that gives a component a
permanent name, a place for its description, schema and tags, and a way to be added without
editing the core.

What it gives up:

- **Registration happens on import.** A component whose module is never imported is silently
  absent. The population rule and its completeness test are the price (P1-15), and a
  forgotten line in an `__init__` is caught by a test and not by the user.
- **Old names stay.** A fix that changes results ships as a new `_vN` name and the old one
  stays available, so versions accumulate and an old one can only be removed by a deliberate,
  documented break.
- **Tags are only as true as their declarations.** The vocabulary rejects a misspelled tag
  but not a false claim. Truthfulness tests (specification 10.6) check each declaration: a
  component that declares a support works with it, and one that does not fails the
  compatibility check with a clear message.
- **Ceremony for one-off code.** A plain import path would be quicker for a throwaway class.
  The three levels of use of specification 3.8 keep the simple case simple: most callers take
  a recipe and never type a name.

## Consequences

- Every kind has an abstract contract, a registry and a registration decorator, each in its
  own file (specification 10.5, 10.6). The mechanism itself lives in `core`, which imports no
  other planexs layer and no machine-learning library (specification 5.1).
- A component is selected from configuration by the `type` field of its configuration
  schema, which holds its registered name; an unknown name is an error that lists the
  registered ones (specification 5.12).
- A study stores the names of the components in use, and a snapshot loads by name only. A
  name missing from the registries is an error that names the component and, when known, the
  plugin that provides it, and loading a snapshot executes no code from the file
  (specification 5.11).
- Adding a component follows the checklist of specification 10.14. One contract test runs
  over every registry, and a truthfulness test covers each declaration (specification 10.6).
- A plugin imports only the public contracts, may extend the tag vocabulary, and is trusted
  code; this is documented, not enforced (specification 5.12).
- A study built from an incompatible set of components fails at construction. It is never
  repaired by switching components silently, which is the behaviour of ADR-0006 that
  ADR-0009 supersedes.
- The group name `planexs.plugins` is provisional until the owner answers D-10. Before
  acceptance this ADR is edited to match the answer; afterwards a change of group is a new
  ADR, since a name that plugins already declare is a permanent identifier like any other.

## Action items

1. [x] Record the decision (P1-02).
2. [ ] Owner: accept `planexs.plugins` as the entry-point group, or name another (D-10), then
   set this ADR to Accepted.
3. [ ] Build the mechanism in `core`: the registry and the kind table (P1-06), the component
   entry and the tag vocabulary (P1-07), the compatibility check (P1-08), component
   configuration and nested resolution (P1-09, P1-10), and plugin discovery (P1-12).
4. [ ] Add the contract and completeness tests over every registry (P1-15).
5. [ ] Use the mechanism for the first real kinds: parameter and constraint types (P2-04,
   P2-05), then the kinds of the later phases.
