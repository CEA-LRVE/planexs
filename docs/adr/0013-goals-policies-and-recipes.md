# ADR-0013: Goals, policies and recipes

**Status:** Accepted
**Date:** 2026-11-02
**Deciders:** project owner

## Context

[ADR-0009](./0009-study-policy-and-component-architecture.md) makes the study the object a
caller drives: it owns a problem, a goal, a policy and the trial history. It leaves two
questions to this record. What are a goal and a policy to each other, which decides what "the
default" means for each type of goal? And how does a study end up with the policy it uses?

Version 0.1.0 answered the second question one case at a time:

- `ask(alpha=None)` meant "Ax's own strategy", and a default existed for one goal only:
  optimizing a number.
- [ADR-0006](./0006-nonlinear-constraint-forces-custom-layer.md) and
  [ADR-0008](./0008-equality-linear-constraint-forces-custom-layer.md) decided that when the
  search space holds a constraint Ax cannot enforce (a `NonlinearConstraint`, or an equality
  `LinearConstraint`), `ask()` leaves Ax's strategy for the custom layer, forces `alpha` to a
  constant (`DEFAULT_ALPHA_WHEN_FORCED = 0.0`) and logs a warning. ADR-0006 named the cost
  itself: the strategy that answers a call depends on the contents of the search space and not
  only on the arguments of the call.

The specification asks for more than that. FR4 wants a default policy "for any goal type and
at any budget", FR18 wants every component selected by name, FR25 wants an incompatible choice
to fail when the study is built with an error that says how to fix it, and FR27 wants policies
that planexs does not implement (Ax) to honour the same lifecycle as its own. Sections 3.6 and
5.9 state the rule that settles the case of ADR-0006: "A study never silently changes its
policy afterwards." The reference use cases need a default for the boundary between two phases
from binary labels (UC-5) and for pure exploration (UC-6), and one study driven by an opaque
policy (UC-12). A single fixed default cannot serve them.

How this record relates to ADR-0006 needs saying, because the records guard
(`tests/unit/test_adr_records.py`) follows exactly one superseder per ADR. ADR-0009 supersedes
ADR-0006 as a whole, and its table says that what replaces the "switch with a warning"
behaviour is recorded here. ADR-0006 therefore keeps its one status line, "Superseded by
ADR-0009", and this ADR has no `Supersedes` header. The replacement is decided in item 7 of
the Decision. [ADR-0010](./0010-registries-capability-tags-and-stable-names.md) already
decides the compatibility check that makes it possible, and
[ADR-0011](./0011-policy-values-resolved-configuration-and-seeds.md) already decides that a
study stores its resolved configuration. This record decides how goals and policies use them.

## Decision

The specification describes goals, policies and recipes in sections 5.7, 5.9 and 5.10. This
ADR adopts that description and settles the points it leaves open:

1. **A goal is typed configuration that proposes defaults, and computes nothing.** A goal is a
   registered component of the kind `goal` (optimize, level set and explore first). It is
   validated against the problem when the study is built: the outcomes it names must exist and
   be of a suitable kind. It does three things. It *parameterizes* acquisitions and analyses
   (directions and weights, a level, constraints on outcomes), so that these are stated once,
   on the goal. It *declares capability tags*: those it supports (`goal:level_set`,
   `objectives:multi`) and those it needs (an optimize goal with outcome constraints needs
   `outcome_constraints:supported`). And it *proposes recipes and default analyses* (items 3
   and 6). Adding a goal type changes no core file.
2. **One policy contract, two forms.** A policy decides which parameterizations to evaluate
   next. It receives a read-only view of the study state (every trial in every state, the
   goal, the problem, the history version, the ask index and the random sources), is told of
   every change of a trial's state, and can export and import its own state so that
   persistence works. A *composed* policy is made of a design, a surrogate, an acquisition and
   an acquisition optimizer, with the rule that moves it from its design phase to its model
   phase held in its configuration. An *opaque* policy is any object that honours the contract:
   random search, an adapter such as Ax, or user code. The study treats both forms the same
   way, and validates every suggestion against the problem before it records it as a pending
   trial, so no policy can leave the history in an illegal state. Policies are a registered
   kind (`policy`): choosing one is configuration.
3. **A recipe is a complete policy configuration, and a goal proposes them in order of
   preference.** When the caller gives no policy, the study asks the goal for its recipes,
   runs the compatibility check of ADR-0010 on each in order, and uses the first one whose
   components are compatible with the problem and the goal. An incompatible recipe is skipped,
   never repaired. If no recipe is compatible, construction fails with a `CompatibilityError`
   that lists, for each unmet need, who needs it and which registered components would
   satisfy it. If the caller does give a policy, it is checked and never replaced: when it is
   incompatible, construction fails, and no recipe takes over. The numbers inside a recipe (the
   initial points per dimension, for example) are policy values: they live in a
   `...RecipeConfig` schema with documented defaults and are reachable through the
   configuration, not written in the recipe's code (ADR-0011).
4. **The resolved policy is stored, not the recipe.** The policy in use, whether it was given
   or chosen, is written into the study's configuration fully resolved (every default
   materialized, every nested `type` written out) and goes into the snapshot. The study can
   say which recipe produced it (`describe()`), but what is saved and what a loaded study
   rebuilds its policy from is the policy itself, never a reference to a recipe. A later
   release that changes the recipes of a goal therefore cannot change a saved study. Whether
   the snapshot also keeps the name of the recipe, as information only, is left to the
   snapshot format (ADR-0014).
5. **A study never switches policy afterwards.** The policy is fixed when the study is built.
   Neither the amount of data, nor the state of the trials, nor the contents of the search
   space changes which policy answers an ask. What a policy does inside itself is not a
   switch. A composed policy moves from its design phase to its model phase by the rule in its
   own configuration and tags every suggestion with the phase that produced it. A fallback it
   declares, such as returning to the design phase while its surrogate cannot be fitted yet,
   is part of its documented behaviour and is logged at `WARNING`. A caller who wants another
   policy on the same data builds a new study with that policy and attaches the observations
   (FR3).
6. **Goals list default analyses.** Each goal names the analyses that answer it (the best
   point, a Pareto front, a level-set estimate), which `Study.analyze` runs when it is given
   none. An analysis is held to the rule of a policy component: it declares what it needs, and
   one whose needs the study's components cannot meet fails the compatibility check before it
   runs.
7. **What replaces the behaviour of ADR-0006.**

| ADR-0006 (and ADR-0008) | planexs |
|---|---|
| A `NonlinearConstraint` or an equality makes `ask()` leave Ax's strategy for the custom layer | The constraint contributes a need (`constraints:nonlinear`, `constraints:equality`). The optimizer of the default recipe supports it, so there is nothing to switch to. A policy that does not declare the capability, such as the Ax adapter, is rejected when the study is built, with a message that names a policy that works |
| `alpha=None` selects the default, and `alpha` selects the custom layer | `Study.ask(n)` takes no `alpha`. The exploration weight is a field of the configuration of the `exploration_exploitation_v1` acquisition, a choice made in the policy and not an argument of `ask` |
| A forced `alpha` (`DEFAULT_ALPHA_WHEN_FORCED = 0.0`) | No forced value. The default recipe for optimizing uses acquisitions that need no exploration weight. Where an acquisition has such a parameter (the `alpha` of `exploration_exploitation_v1`, the `beta` of the straddle) it is a described schema field with its default written once (ADR-0011). The constant is deleted with the legacy facade |
| A warning when the switch happens | An error when no compatible policy exists, so that nothing runs, and the stored resolved policy when one does, so that nothing is hidden. `WARNING` stays for degraded situations inside a policy |

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| Choose the strategy at each ask from the current state | Generalize ADR-0006: the study inspects the search space and the history at every call and picks the policy that fits | Adaptive, and a caller never meets an error. But the policy that answered an ask is not in the configuration, a saved study cannot say which one it used, and a change of data silently changes the algorithm. This is what specification 3.6 and 5.9 forbid, and where they and an older ADR disagree the specification wins |
| A default hard-wired in code for each goal | A function in the study maps a goal type to a policy, with one class or an `if` per goal | Simple to write, but it is a closed dispatch (specification 3.3): a goal type added by a plugin brings no default, the default cannot be inspected or configured, and every new goal edits the study (FR19, UC-11) |
| Composed policies only, with no opaque form | Every policy is made of design, surrogate, acquisition and optimizer, and Ax is expressed as one of the four | One contract less to maintain. But ADR-0001 found that injecting a custom acquisition function into Ax's generation strategy means reaching into Ax's internals, which is possible but brittle, and ADR-0007 that Ax's model lives in a space Ax controls. Expressing Ax as one of the four parts would need exactly that, and random search or user code would have to pretend to have a surrogate. FR27 and UC-12 are not met |
| Store the name of the recipe and resolve it again on load | The snapshot records which recipe was chosen and rebuilds the policy from the recipes of the installed version | A smaller snapshot. But a change to a recipe in a later release changes how an old study behaves, which ADR-0011 and specification 3.7 rule out |
| Goals propose ordered recipes, the first compatible one is used, the resolved policy is stored, and one contract serves both forms (chosen) | The mechanism of the Decision above | Meets FR4, FR18, FR25 and FR27, and removes the dilemma of ADR-0006. It makes the default a list that someone has to order and maintain, and it gives up changing policy during a study |

## Trade-off analysis

ADR-0006 weighed three options and chose to switch and warn, because both others were worse:
raising an error defeated the promise of FR4 to a caller who just wants the default to work,
and ignoring the constraint returned wrong answers without saying so. The dilemma assumed one
default strategy that either can or cannot serve the problem. A list of recipes removes the
assumption. For a problem that some combination of registered components can serve, the
default path neither fails nor does hidden work: it takes the first combination that works,
once, and writes it down. The visibility that ADR-0006 lacked becomes a stored value and not a
log line. An error remains for the case where no recipe fits and for a policy the caller chose
that does not fit, which is where an error is the honest answer (specification 3.6).

What the decision gives up:

- **First compatible is not best compatible.** The order of the recipes is the judgement of
  whoever wrote the goal, and no measurement selects it. Their quality has to be shown by the
  use-case tests and not assumed, and the owner reviews the recipes after Phase 9 with those
  numbers (D-9). This is the same cost ADR-0009 accepted when the default stopped being Ax's.
- **No adaptivity in the choice of policy.** A study that starts with one policy never moves
  to another. Adaptive behaviour has to live inside a policy, as the move from design to model
  phase does, or in an opaque policy. Trying a different policy on the same data takes a new
  study and `attach`. This is deliberate: it is what makes the stored policy a true report of
  what ran.
- **Recipes repeat configuration.** Each recipe is a full policy configuration and not an
  override of a base one, so recipes of one goal share much of their text. The resolved
  configuration is already fully written out (ADR-0011), so the repetition costs little.
- **A lowest common contract.** An opaque policy takes part in the compatibility check as one
  participant with the tags it declares, and not through its parts, so the check can say less
  about it than about a composed policy.

## Consequences

- The goal contract carries the recipes and the default analyses (P4-02), and the three first
  goals declare their tags (P4-03 to P4-05). The policy contract, the read-only state view and
  the suggestion type live in `policies` (P6-01). The composed policy is built in three steps
  (P6-02 to P6-04), and `random_policy_v1` is the second policy implementation (P6-06), so the
  `policy` kind satisfies the rule of two (specification 3.10).
- Need resolution maps the facts of the problem and the goal to capability needs and runs the
  check (P6-05). The recipes of the first goals are written in P6-07 and P6-08. Study
  construction validates the goal, picks the recipe or checks the given policy, resolves the
  configuration and creates the policy (P6-09). `describe()`, `toConfig()` and `fromConfig`
  report and rebuild it (P6-15). The state of an opaque policy goes into the snapshot (P8-06).
- No function of the study branches on the type of a goal. The mapping from a goal to its
  default policy is data held by the goal, so a goal added by a plugin brings its own default
  (specification 3.3, 5.7).
- An incompatible choice, whether a recipe, a given policy or an analysis, fails with a
  `CompatibilityError`. It lists all the unmet needs together, who needs each and which
  registered components would meet it (ADR-0010). The wording of one failure is pinned by a
  message snapshot test (P6-05).
- This ADR changes no code. The legacy facade keeps switching with a warning until the legacy
  path is deleted (P11-01), because the 0.1.0 code is never extended (ADR-0009). The
  `alpha` argument of `ask()` and `DEFAULT_ALPHA_WHEN_FORCED` disappear with it.
- The facts that ADR-0006 established stay as inputs, as ADR-0009 already says: Ax takes
  linear inequalities only as parameter constraints, and the measurements of the exploration
  weight (the best value depends on the problem) remain evidence for choosing it.
- Because a study stores its resolved policy, changing the recipes of a goal never alters a
  saved study. It does change what a new study gets by default. Whether such a change needs a
  new goal name under specification 3.7 is not decided here: the recipes are proposals until
  the owner's review of D-9, after Phase 9.

## Action items

1. [x] Record the decision of this ADR and list ADR-0013 in the navigation (P4-01).
2. [ ] Goal contract and registry, with recipes and default analyses (P4-02), and the goals
   `optimize_v1`, `level_set_v1` and `explore_v1` (P4-03 to P4-05).
3. [ ] Policy contract and study-state view (P6-01), the composed policy (P6-02 to P6-04) and
   `random_policy_v1` (P6-06).
4. [ ] Need resolution (P6-05) and the recipes of the first goals (P6-07, P6-08).
5. [ ] Study construction picks the recipe or checks the given policy, and stores the resolved
   policy (P6-09, P6-15). The policy state goes into the snapshot (P8-06).
6. [ ] Default analyses, and the compatibility check of analyses before they run (P7-11).
7. [ ] Delete the legacy switch with the legacy facade (P11-01).
8. [ ] Review the contents of the recipes with the numbers of the use-case tests (D-9, after
   Phase 9).
