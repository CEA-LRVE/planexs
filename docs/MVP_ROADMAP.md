# MVP Roadmap

**Status:** proposal, v0.1. It becomes active when the owner accepts the pull request that adds it and answers MD-1 (section 7).
**Relation to the roadmap:** a shorter path through the same cards. It is not a second plan and it adds no code of its own. The [roadmap](ROADMAP.md) stays the complete path, and this file does not edit it. Both paths end in the same code, and they cross: the MVP path is a subset of the roadmap's cards, taken first.
**Authority:** the [project specification](PROJECT_SPECIFICATION.md) is the ground truth (specification 0, 11 and 12). This file amends neither it nor the roadmap. Where either disagrees with this file, it wins, and the disagreement is a bug to fix here.
**Baseline:** the roadmap at commit `84f9ea6` (2026-10-29): 146 cards, 23 done, 123 remaining.

---

## 1. The answer

**Yes, it is possible, and it needs no second code.** The shortest way to a usable `Study` that is also the final code takes 47 of the 123 cards that remain (36 S and 11 M). By the size caps of roadmap 3.1 (150 lines for an S card, 400 for an M card), that is a ceiling of about 9,800 lines of source, against 27,700 for everything that remains. All 47 are roadmap cards. 44 of them are done exactly as written, and 3 are delivered in two parts (section 4). The path ignores 3 `Needs` edges of the roadmap (section 3) and changes nothing else.

**It cannot be much shorter.** Every kind that the optimization loop passes through (domain, encoding, surrogate with its nested kinds, goal, design, acquisition, acquisition optimizer, policy, study) is a card of the roadmap, and merging cards is a shortcut that the roadmap forbids (roadmap 3.5). A first version in 10 cards would put code in the library that is not the final code. This file does not propose that.

| Milestone | What it gives | Cards | S / M | Waves |
|---|---|---|---|---|
| MVP-1 | a usable `Study` for single-objective optimization (section 3) | 47 | 36 / 11 | 15 |
| MVP-2 | MVP-1, plus saving and reloading a study | 8 more | 6 / 2 | 5 |
| the roadmap | everything that remains | 123 | 86 / 37 | |

**Where the saving comes from.** Of the 76 cards that MVP-1 leaves for later, 67 are left by order alone: nothing on the path needs them, so they simply come after. The other 9 are left because the path ignores three edges (section 3). Without those three edges the same targets need 56 cards.

**What MVP-1 gives.** It is the real architecture, so it is not poor in features: the parameter kinds, constraints and optimizers that the roadmap ports are all on the path.

| Requirement | In MVP-1 | Otherwise delivered by |
|---|---|---|
| FR1, FR13, FR14 search space with every parameter kind, steps, conditional parameters | yes | |
| FR2 one outcome to minimize or maximize, FR15 constraints on outcomes | yes | several unweighted outcomes: P5-05 and the second part of P6-07 |
| FR3 known data | yes (P6-12) | |
| FR4 default policy with no tuning | yes, for the optimize goal | other goal types: P6-08 |
| FR9 linear constraints, FR17 nonlinear constraints | yes | |
| FR10 batches, FR23 trial lifecycle | yes | |
| FR12 reproducibility | seed and resolved configuration (P6-13); reload in MVP-2 | |
| FR18, FR19, FR25, FR26 components by name, plugins, compatibility check, policy values | yes, for the kinds on the path | the stopping kind: P7-08. The repository kind: P8-03 (MVP-2) |
| FR11 save and reload | MVP-2 | |
| FR5 exploration weight | no | P5-07, P5-08 |
| FR6 importance | no | P7-05 |
| FR7, FR8, FR22 predictions and the surrogate alone | the `Surrogate` of P3-09 predicts; `Study.surrogate` is one small card | P6-14 |
| FR20, FR21, FR24 level sets, exploration, binary outcomes | no | P3-16, P4-04, P4-05, P5-09, P6-08 |
| FR27, FR28 other policies, stopping | no | P10-07, P7-08 to P7-10 |

---

## 2. How the two paths cross

These rules are what "does not call into question any point of the roadmap" means in practice.

| Rule | Statement | Checked by |
|---|---|---|
| R1 | **Same cards, same code.** Every card of the path is a roadmap card, done as written or as a slice (section 4). No file, name, registered component or configuration field exists on the path that the roadmap does not define, and nothing is written that a later card has to delete or rewrite. | review, and R8 |
| R2 | **The roadmap is not edited by this file.** Not its text, its order, its numbering, its sizes or its `Needs`. | the diff of the pull request that adds this file |
| R3 | **Three edges are ignored, for the path only.** A card on the path does not wait for the three `Needs` of section 3, because the part of the card that the path uses does not use what the edge points to. A card off the path still waits for them, and the card at the other end still delivers all of its work when its turn comes. | R8 |
| R4 | **A slice is additive.** A card is delivered in two parts only if the second part is a pure addition to the first: a new registry entry, a new method, a new test step. The first part is whole files and whole entries, with its own tests and docstrings, never a stub, a placeholder or a temporary implementation. A card whose second part would rewrite the first is done in full, whatever it costs. | review |
| R5 | **Order, and work at the same time.** Cards run in wave order, and the cards of one wave are independent. Instead of taking the next card in phase order, take the next card of the path. Cards off the path may be worked at the same time by anyone who is free, as the roadmap already allows. The path is only an order, so two sessions meet only if they start the same card. | `doing` in the roadmap |
| R6 | **Statuses.** A card done in full gets its roadmap status `done` as usual. A slice leaves its roadmap card `todo` (or `doing`) until its remainder is done, and its first part is tracked in section 4. A card that is off the path and needs a sliced card waits for the whole card. | section 4 |
| R7 | **Finishing a slice.** When the card comes up in the roadmap, its executor delivers the remainder only, checks that the Done-when items of the first part are still evidenced by existing tests, and reports each Done-when item as "from the slice (test name)" or "now". | the report (roadmap 3.4) |
| R8 | **The path is checked against the roadmap.** The guard test of this file recomputes the path from the `Needs` lines of the roadmap. It fails if the path is not closed under them (apart from the ignored edges), if a card is on it that no target needs, or if a title or a size has drifted. | `tests/unit/test_mvp_roadmap.py` |

---

## 3. The path

### 3.1 Targets and ignored edges

**MVP-1 targets:** P6-10, P6-11, P6-12, P6-13, P7-11, P9-01
**MVP-2 targets:** P8-07, P8-08

The path is computed, not chosen. It is every card that the targets need, following the `Needs` lines of the roadmap, except through the ignored edges, and stopping at cards that are already done. The three ignored edges:

**Ignored edges.**

| Card | Ignored need | Why the path can ignore it |
|---|---|---|
| P6-09 | P6-08 | P6-08 gives the level-set and explore goals their recipes. `Study` picks its recipe by asking the goal, so those recipes are an addition, and the optimize goal has its own (P6-07). |
| P6-07 | P5-05 | P5-05 is the multi-objective acquisition. Only the second recipe of P6-07 uses it, and that recipe is the remainder of the slice. |
| P7-11 | P7-08 | P7-08 is the stopping contract. Only `Study.shouldStop` uses it, and that method is the remainder of the slice. |

These edges are the whole difference between MVP-1 and the path that follows the letter of the roadmap (56 cards at the baseline). They save 9 cards: P3-15, P3-16, P4-04, P4-05, P5-05, P5-06, P5-09, P6-08 and P7-08.

### 3.2 MVP-1

**MVP-1 size:** 47 cards (36 S, 11 M)

A wave is a set of cards whose needs (outside the ignored edges and the cards already done) all lie in earlier waves. The cards of one wave are independent.

**MVP-1 path.**

| Wave | Card | Title | Size | Scope | Status |
|---|---|---|---|---|---|
| 1 | P2-01 | ADR-0012: outcomes, trials and observations | S | full | done |
| 1 | P2-02 | Prefix the legacy types and move the legacy snapshot | S | full | done |
| 1 | P3-01 | Move the encoder and the expression compiler | S | full | done |
| 1 | P4-01 | ADR-0013: goals, policies and recipes | S | full | done |
| 2 | P2-04 | Parameter-type registry | M | full | done |
| 2 | P2-06 | Outcomes and outcome kinds | M | full | done |
| 2 | P2-07 | Trial lifecycle | S | full | done |
| 2 | P3-02 | Torch runtime helpers | S | full | done |
| 2 | P3-04 | Move constraint encoding and feasible sampling | S | full | done |
| 3 | P2-05 | Constraint-type registry | S | full | done |
| 3 | P2-08 | Observations | S | full | done |
| 3 | P3-03 | Encoder policy values to configuration | S | full | done |
| 3 | P3-05 | Sampling and constraint policy values to configuration | S | full | done |
| 4 | P2-09 | Pure expression evaluator | S | full | todo |
| 5 | P2-10 | Problem and problem facts | S | full | todo |
| 6 | P3-07 | Surrogate contract and registry | S | full | todo |
| 6 | P4-02 | Goal contract and registry | S | full | todo |
| 6 | P4-06 | Design contract and sobol_v1 | S | full | todo |
| 7 | P3-08 | Trials to tensors | S | full | todo |
| 7 | P3-09 | Surrogate wrapper and predictions | M | full | todo |
| 7 | P3-10 | Kernel kind | S | full | todo |
| 7 | P3-11 | Likelihood kind | S | full | todo |
| 7 | P3-12 | Outcome-transform kind | S | full | todo |
| 7 | P3-13 | MAP fitter | S | full | todo |
| 7 | P4-03 | optimize_v1 goal | M | full | todo |
| 7 | P6-01 | Policy contract and study-state view | S | full | todo |
| 8 | P3-14 | single_task_gp_v1 surrogate | M | full | todo |
| 8 | P3-17 | Surrogate cache | S | full | todo |
| 8 | P5-01 | Acquisition contract and registry | S | full | todo |
| 8 | P7-01 | Analysis contract and registry | S | full | todo |
| 9 | P5-03 | qlog_nei_v1 acquisition | S | full | todo |
| 9 | P5-11 | Optimizer contract and registry | S | full | todo |
| 9 | P7-02 | Best-point analyses | S | full | todo |
| 10 | P5-12 | botorch_relaxed_v1 optimizer | M | full | todo |
| 10 | P6-02 | Composed policy configuration | S | full | todo |
| 11 | P5-13 | botorch_mixed_v1 optimizer | M | full | todo |
| 11 | P6-03 | Composed policy: design phase | S | full | todo |
| 11 | P6-05 | Need resolution | S | full | todo |
| 12 | P6-04 | Composed policy: model phase | M | full | todo |
| 12 | P6-07 | Recipes for optimize goals | S | slice | todo |
| 13 | P6-09 | Study construction and history | M | full | todo |
| 14 | P6-10 | Study ask | M | full | todo |
| 14 | P6-11 | Study tell, fail and abandon | S | full | todo |
| 14 | P7-11 | Study analyze, shouldStop and default analyses | S | slice | todo |
| 15 | P6-12 | Study attach | S | full | todo |
| 15 | P6-13 | Study seeds and determinism | S | full | todo |
| 15 | P9-01 | UC-1: single-objective optimization | M | slice | todo |

**ADR numbers.** `scripts/new_adr.py` takes the next free number, and roadmap cards cite ADR-0012, ADR-0013 and ADR-0014 by number. Run P2-01 before P4-01, and P8-01 (MVP-2) after both, as the phase order of the roadmap does.

**Checkpoints.**

- After wave 7 (26 cards): the domain, the encoder and the contracts of the surrogate, goal and design kinds are in place.
- After wave 12 (40 cards): the composed policy suggests points with the default recipe for one objective.
- After wave 14 (44 cards): a `Study` asks and tells, and reports the best point.
- After wave 15 (47 cards): attach, seeds and the acceptance test of UC-1. MVP-1 is reached.

**Acceptance.** MVP-1 is reached when the first part of P9-01 passes (a noisy two-dimensional problem with a linear constraint: the best observed value beats random search at equal budget, and `Study.analyze` returns the best trial) and the verification commands of roadmap 3.3 pass.

### 3.3 MVP-2

MVP-2 adds saving and reloading a study. It comes after MVP-1 (its waves start again at 1), and it lets the remainder of P9-01 be done. P8-05 (migrations) and the rest of Phase 8 stay for later, because P8-07 does not need them.

**MVP-2 size:** 8 more cards (6 S, 2 M)

**MVP-2 additions.**

| Wave | Card | Title | Size | Scope |
|---|---|---|---|---|
| 1 | P6-15 | Study describe, toConfig and fromConfig | S | full |
| 1 | P8-01 | ADR-0014: snapshot format v2 | S | full |
| 2 | P8-02 | Snapshot document model | M | full |
| 3 | P8-03 | Repository kind | S | full |
| 3 | P8-04 | Manifest and environment check | S | full |
| 3 | P8-06 | Policy state in the snapshot | S | full |
| 4 | P8-07 | Study save and load | M | full |
| 5 | P8-08 | Replay determinism test | S | full |

---

## 4. Slices

Three cards of the path are delivered in two parts. The first part is on the path. The remainder is done when the card comes up in the roadmap, and its card stays `todo` until then (R6). By R4 each remainder is an addition.

**Slices.**

| Card | First part, on the path | Remainder, done at the card's turn in the roadmap | Why the remainder is only an addition |
|---|---|---|---|
| P6-07 | The recipe for one objective with its `...RecipeConfig` schema, and the test that the one-objective case picks it. | The recipe for several unweighted objectives (needs P5-05), and the tests that the first compatible recipe is chosen and that an incompatible one moves on to the next. | A goal holds an ordered list of recipes: a second recipe is appended and the lookup does not change. |
| P7-11 | `Study.analyze` by name and by configuration (the surrogate is created on demand), the default analyses of a goal, and the compatibility check before an analysis runs. | `Study.shouldStop` and its test (needs P7-08). | A new method that reads the configured criteria. |
| P9-01 | The acceptance test and the example of UC-1: the budget, the comparison with random search, and `Study.analyze` returning the best trial. | The step of the same test that saves the study and reloads it (needs P8-07). | A new step at the end of an existing test. |

No other card is sliced. A card that registers several entries (P2-06, P3-10 to P3-12) is done in full: trimming an entry saves little, and tests of other cards on the path may rely on it (the tests of P6-05, for one, check the need of a binary outcome, whose tag P2-06 registers).

---

## 5. Back to the roadmap

After MVP-1 nothing changes in how the roadmap is followed. Take the cards in the roadmap's order, skip the ones that are done, and at a sliced card deliver its remainder (R7). At the baseline, the 76 cards that MVP-1 leaves are exactly the cards of the roadmap that are not on the path: analyses and stopping beyond the best point, the level-set and explore goals with the classifier, the exploration weight and the other acquisitions, persistence beyond MVP-2, the use cases beyond UC-1, the Ax adapter, the cleanup and the release. MVP-2 takes 8 of them.

Nothing on the path has to be redone when those cards come: that is what R1 and R4 guarantee, and what the guard test keeps true.

---

## 6. Working a card on the path

The protocol of roadmap 3.1 to 3.6 applies unchanged: the request check, all of Do, tests first, the verification commands, evidence for each Done-when box, and the report format. Two additions:

- The report names the card as `Pn-mm (MVP-1, full)`, `Pn-mm (MVP-1, first part)` or `Pn-mm (remainder)`.
- For a first part or a remainder, the Done-when table has a column `part` that says which items belong to which part (R7).

---

## 7. Decisions needed

"Proposed" is what the work assumes if the owner says "go with the defaults". These decisions are numbered `MD-n`, so that they never collide with the `D-n` of roadmap 17.

| ID | Question | Options | Proposed | Blocks |
|---|---|---|---|---|
| MD-1 | The three ignored edges and the three slices, which are the only departures from the letter of the roadmap. | accept; follow the letter (MVP-1 then has 56 cards) | Accept. | P6-07, P6-09, P7-11 and P9-01 |
| MD-2 | Is saving and reloading part of the first usable version? | MVP-1 first and MVP-2 right after; one milestone of 55 cards | MVP-1 first. A campaign can restart by attaching its data table again (P6-12). | MVP-2 |
| MD-3 | Categorical and boolean parameters in MVP-1 (P5-13, an M card). | keep; skip | Keep. The recipe of P6-07 names `botorch_mixed_v1`, so skipping would change that recipe later. | P5-13 |
| MD-4 | Add `Study.surrogate` (P6-14, one S card) to MVP-1, for predictions with their uncertainty. | add; leave for later | Leave for later. It is not strictly necessary, and it can come right after MVP-1. | P6-14 |

---

## 8. Findings

Noticed, not changed (R2).

| ID | Finding |
|---|---|
| F1 | P9-01 says that the study saves and reloads, but its `Needs` do not include P8-07, so as written it cannot be finished from its own needs. The path therefore delivers P9-01 in two parts (section 4). |
| F2 | The first `Study` needs 56 cards instead of 47 only because of three edges: P6-09 needs P6-08, P6-07 needs P5-05 and P7-11 needs P7-08. They are not changed here (R2). |

---

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| A remainder turns out not to be additive | rework, which R1 forbids | R4: slice only where the second part is an addition, and otherwise do the card in full and report it as a finding |
| The roadmap changes under the path (a new `Needs`, a renamed card) | the path is silently wrong | R8, the guard test |
| Two sessions start the same card | conflicting pull requests | `doing` in the roadmap, and wave order |
| A remainder is forgotten | a feature stays missing | its roadmap card stays `todo` (R6), so the normal flow reaches it |
| The first usable version lacks a recipe that someone asks for | a compatibility error | the error lists the registered components that would meet the need (specification 3.6), and the cards that add them are in the roadmap |
| The path hides how much is left | wrong expectations | the milestone table and section 5: MVP-1 is about a third of the remaining work, not the whole |

---

## 10. Definition of done

MVP-1 is done when:

1. Every card of the MVP-1 path is done, or its first part is done and tracked in section 4.
2. The acceptance test of UC-1 (first part of P9-01) passes.
3. The verification commands of roadmap 3.3 pass.
4. The guard test of this file passes.

A card is done by the rules of roadmap 20. MVP-2 is done when its 8 cards are done and the remainder of P9-01 passes.
