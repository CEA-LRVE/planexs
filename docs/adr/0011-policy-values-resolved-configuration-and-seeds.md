# ADR-0011: Policy values, resolved configuration and seeds

**Status:** Accepted
**Date:** 2026-10-18
**Deciders:** project owner

## Context

[ADR-0009](./0009-study-policy-and-component-architecture.md) decides that every choice is a
registered component and that a study is reproducible. This ADR records the three linked
mechanisms that make that true for numbers and for randomness: where the tunable values of
the library live, what a study stores about them, and how every random stream is derived
(specification 3.4, 3.7, 5.11 and 10.7; FR12 and FR26).

The 0.1.0 code shows each way this goes wrong:

- **Values that no study can reach.** The roadmap baseline (section 2.1) counts 39 numeric
  literals outside {0, 1, -1, 2} in 11 of 58 source files, about 30 of them policy values.
  Naming a value does not fix this: `MIN_TRIALS_FOR_MODEL = 2` and
  `DEFAULT_ALPHA_WHEN_FORCED = 0.0` in `BayesianOptimizer.py` are named, documented module
  constants, and no study configuration can change them. Specification 12.4 records this
  mistake.
- **Values that can be reached and are not saved.** The budgets of `AlphaAcquisitionStrategy`
  (restarts, raw samples, reference points, Monte Carlo samples, categorical combinations,
  iterations) are constructor arguments, so a caller can pass a customized strategy to
  `BayesianOptimizer`. But `StudySnapshot` has no field for it, and `BayesianOptimizer.load()`
  rebuilds `AlphaAcquisitionStrategy()` with its defaults. A reloaded study can behave
  differently from the one that was saved, and a later change of a default would change an
  old study.
- **Seeds derived by addition.** `BayesianOptimizer` derives the seed of an ask as
  `(random_seed + 1_000_003 * n_trials + asks_since_tell) % (2**31 - 1)`,
  `AlphaAcquisitionStrategy.suggest` gives its i-th batch point `seed + 1 + i`, and
  `sampleFeasibleEncoded` uses `seed + round_index`. Different purposes can therefore receive
  the same integer: the second ask between two tells has seed `s + 1`, which is also the seed
  the first ask hands to its first batch point. The multiplier `1_000_003` and the modulus
  are hidden policy values themselves, and `load()` resets the ask counter to 0, so the first
  ask after a reload can reuse the seed of an earlier ask.

The specification turns these into rules. Every policy value is a documented field of a typed
configuration schema and is reachable through the study configuration (3.4). Every study
stores its fully resolved configuration, so that a later change of a default never alters an
old study (3.4, 5.11). One study seed feeds every stochastic component through a documented,
stable derivation, with no global random state (3.7, 5.11). The rule on values is checked
automatically (10.7) by the script that task P0-05 wrote and P0-06 wired into CI against a
baseline of today's 39 values. The exemptions were decided by the owner on 2026-10-13
(decision D-5, "as proposed").

## Decision

1. **The policy-value rule.** A policy value is any number, string, flag or heuristic that
   changes numerical results, performance or behaviour and could reasonably differ between
   users or problems: tolerances, budgets, sample counts, restart counts, iteration caps,
   jitter and floors, seed-derivation constants, size limits, "use X when there are more than
   N" rules, default component names. Each one is a field of a typed configuration schema,
   with a documented meaning, unit and default, and the default is written once, in that
   field. It is never a literal in a function body, a module-level constant or a literal
   default in a function signature. An operation with tunable options takes one options
   object, itself a configuration schema (specification 10.8).
2. **The exemptions (D-5).** Four numbers are structural in any spelling: 0, 1, -1 and 2 in
   formulas. Any other structural constant (a tensor axis index, the midpoint of the unit
   cube) carries a comment `# structural: <reason>` with a non-empty reason, which is for
   mathematical definitions only: using it on a tunable value is a defect. A file whose
   single public symbol is a configuration schema (a class named `...Config`) is where
   defaults live and is not scanned. Tests and scripts are not scanned. The owner also
   confirmed that a literal default in a signature is a policy value, and may ask to revisit
   that if the options-object style proves too heavy for the Python API.
3. **What automation cannot see.** The checker finds numeric literals and nothing else.
   Strings, booleans and a rule hidden behind an allowed number (`len(values) > 2` standing
   for "more than two values means ordered") are policy values as well. Review is the control
   for those and for every use of the `# structural:` marker.
4. **The resolved configuration is stored.** A study stores its fully resolved
   configuration: every default materialized and every nested `type` written out, as plain
   JSON-compatible data, for the goal, the policy, the analyses and the stopping criteria.
   Rebuilding the validated objects from it and resolving again gives the same dictionary.
   The snapshot holds it (specification 5.11), so changing a default in a later release
   cannot change how a saved study behaves.
5. **One seed, derived per stream.** A study has one seed, recorded in its snapshot. Every
   random stream is derived from it, keyed by the ask index and the component path, by a
   stable function: the path is hashed with a stable hash, never Python's randomized `hash`,
   and the derivation uses `numpy.random.SeedSequence` (planned in P1-13). A component
   receives its own generator or integer seed and never touches a global random state.
   Adding a component, or a path nobody else uses, does not change the stream of any other.
   When the caller gives no seed, one is drawn from the operating system and recorded, so
   the whole tree can be rebuilt.
6. **The ratchet.** `scripts/check_policy_values.py` reports every numeric literal other
   than 0, 1, -1 and 2 under `src/planexs`, with its file, scope, position (module level,
   signature default or function body) and value. `scripts/policy_values_baseline.txt` lists
   the violations accepted for now, one per line and with no line number, so that editing the
   code above a literal does not turn an accepted entry into a new violation. CI runs the
   check against it in the `lint` job. It fails on a violation the baseline does not list,
   and on an entry whose literal is gone, so entries are only ever removed. Writing the
   baseline can only remove entries, unless `--allow-additions` is given for a deliberate
   re-baseline such as a file move. The baseline must be empty before the first release of
   the new architecture, after which the check runs with no baseline file.

## Options considered

| Option | Description | Trade-off |
|---|---|---|
| Named module constants | Each tunable value is a documented constant at the top of its module, such as `MIN_TRIALS_FOR_MODEL = 2` | Named, greppable and written once. But no study configuration can reach it: changing it means editing the library or patching the module. A saved study does not record it, so a new release that changes it silently changes old studies. A literal checker cannot see it behind an allowed number. Rejected: unreachable by the caller |
| A global settings object | A process-wide `settings` object that the caller mutates, such as `planexs.settings.n_restarts = 20` | Reachable by the caller, and one place to look. But it is hidden state: two studies in one process share it, a result depends on what ran before, nothing ties it to a saved study unless every read is also recorded, and it cannot differ per component or per study. Rejected: hidden state |
| Keyword arguments with literal defaults | Each function states its defaults in its signature, for example `n_restarts: int = 10` | The usual Python style, and light. But a default is repeated in every signature that takes the value, the defaults cannot be listed or stored as one resolved configuration, and a configuration file cannot set them. The owner confirmed (D-5) that it counts as a policy value, and may revisit that if options objects prove too heavy |
| Configuration schemas with a single documented default (chosen) | A typed schema per component holds each tunable value as a described field, whose default is written once; a study stores the resolved result and derives its random streams from one seed | Reachable from configuration in Python, JSON or YAML, written once, stored with the study, and checkable by script. It needs a schema per component and an options object for each tunable operation, which is more code than a keyword argument, and the script covers only numeric literals |

## Trade-off analysis

The three alternatives each lose at least one of three properties that the specification
asks for together: a value is reachable from the study's configuration, it is stored with the
study, and it is defined in one place. Module constants have the third only. A settings
object has the first and not the second, since it is process state. Signature defaults have
none of them fully, because they are repeated and neither listed nor stored. The schema
field has all three, and the resolved configuration is what turns "a default that can be
changed" into "a default that cannot change an old study".

What it gives up:

- **More ceremony.** A tunable value costs a described field in a schema, and an operation
  with options takes an options object. The owner accepted this in D-5 and may revisit it if
  the Python API proves too heavy for it.
- **Verbose configurations.** Writing out every default makes snapshots and resolved
  configurations long. Recipes (specification 5.9), the three levels of use (3.8) and
  `describe()` keep ordinary use short, since a caller only states what differs.
- **A check that sees half the rule.** The script finds numeric literals, so strings,
  booleans and hidden rules rest on review. The `# structural:` marker is an honour system:
  it exists for mathematical definitions, and review has to catch its misuse.
- **A baseline to burn down.** The 39 accepted values keep the legacy code in CI until the
  files that hold them are rewritten or deleted. Each task that removes one deletes its
  line, and the baseline reaches zero only at the end of the migration.
- **A derivation that cannot move.** The seed derivation is code and not a registered
  component, so changing it changes the random streams of every study. It can change only as
  a deliberate break, listed in the changelog like any other.

## Consequences

- A new tunable value is a described field of a `...Config` schema, and the script rejects a
  literal in any other file (specification 10.7). A new component's schema follows the
  checklist of specification 10.14.
- The baseline shrinks as legacy files are rewritten or deleted, and never grows in normal
  work. Roadmap task P11-13 checks that it is empty and removes `--baseline` from CI.
- A study loaded from a snapshot rebuilds its policy from the stored resolved
  configuration, not from the defaults of the installed version. A default can change in a
  later release without changing any saved study.
- The snapshot carries the study seed and the entropy drawn when none was given, with the
  library versions (specification 5.11). It also carries the ask counter, so that a reloaded
  study makes the same next suggestions as an uninterrupted one. The 0.1.0 `load()` resets
  that counter to 0 instead, and stays as it is until the legacy path is deleted.
- No component touches a global random state. A generator or an integer seed reaches it
  through its build context, and a test replays a study from a snapshot and compares its
  later suggestions with an uninterrupted run (specification 10.11).
- Default component names are policy values too: the recipe that a goal proposes is
  configuration, stored resolved, and not a choice hidden in code.
- Contributors learn the rule, the marker and the checkers from the contributing guide
  (P1-17).

## Action items

1. [x] Record the decision (P1-03). The checker (P0-05) and its baseline in CI (P0-06) already
   exist.
2. [ ] Build the random-number tree in `core` (P1-13) and the torch helpers that turn its
   integers into generators (P3-02); give each component of a study its derived generator
   (P6-13).
3. [ ] Build component configuration and resolution (P1-09, P1-10) and the runtime settings
   (P1-14), then store the resolved configuration and the seed entropy in the snapshot
   (P8-02) and expose it through `describe()`, `toConfig()` and `fromConfig()` (P6-15).
4. [ ] Move the baselined values into configuration as their files are rewritten (for
   example P3-03, P3-05 and P3-06), deleting the matching baseline lines, and bring the
   baseline to zero (P11-13).
5. [ ] Document the rule for contributors (P1-17).
