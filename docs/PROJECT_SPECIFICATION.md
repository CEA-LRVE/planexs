# planexs: Project Specification

**Status:** living document, v0.3.
**Purpose:** this document is the ground truth for the planexs project. Any conversation or contributor (human or LLM) working on the project MUST treat it as authoritative. When something here is ambiguous or missing, ask rather than assume (section 12).

---

## 0. Language and conventions

- Conversations about the project may happen in any language (typically French).
- All code, docstrings, comments, configuration keys, commit messages and technical documentation MUST be written in English (research and industry standard, and needed for publication and collaboration).
- Section numbers in this document are stable references: code comments and ADRs cite them (for example "section 10"). A section is never renumbered silently. If the structure must change, the roadmap carries a task to update every citation.

---

## 1. Vision and scope

### 1.1 What planexs is

planexs is a general-purpose, domain-agnostic toolkit for **sequential experimental design with probabilistic surrogate models**, Gaussian processes first. The caller describes what can be varied (the *search space*), what is measured (the *outcomes*) and what they want (a *goal*). planexs proposes which points to evaluate next, learns from the results, and reports what it has learned.

The name stands for *PLANification EXpérimentale Séquentielle*, the French for sequential experimental design.

Bayesian optimization is one goal among several. The same loop serves level-set estimation (finding the boundary where an outcome crosses a level, for example the boundary between two phases of a material), active learning (learning a surrogate as well as possible within a budget), feasibility search, calibration, and plain surrogate modelling with no loop at all. No outcome is privileged: nothing in the core assumes that the point of a study is to minimize a number.

planexs must be comfortable at every scale without the caller classifying the problem in advance: a campaign of four or five extremely costly physical experiments, a few hundred machine-learning trials, or thousands of cheap evaluations (section 8).

### 1.2 What planexs is not

- Not a re-implementation, and not a thin re-wrapping, of BoTorch or GPyTorch. Tensor-level work stays in those libraries (section 1.4).
- Not a neural-network training library and not a reinforcement-learning framework.
- Not an optimizer for structured inputs (graphs, molecules, sequences) in its core. A plugin may add a new parameter kind; the core makes no promise beyond tabular parameterizations (section 5.3).
- Not a service. No server, scheduler, job queue or user interface. Evaluating the suggested points is the caller's job, through the ask/tell cycle.
- Not a data tool. Reading files, pairing records and splitting data are the caller's responsibility.

### 1.3 "Everything is possible" is a property of the architecture

planexs cannot ship every acquisition function, kernel or goal. What it guarantees is this: anything that fits the loop *probabilistic model of the outcomes -> score of candidate points -> choice of points -> observation* can be added by writing components and registering them, without changing planexs' core. The guarantee is tested, not promised: the reference use cases (section 7) include one that runs on components defined entirely outside the package.

A request that cannot be satisfied this way reveals a design defect. The remedy is to move or add a seam (sections 3.3 and 3.10), never to special-case the request inside the core.

### 1.4 Relationship to the engines

- **BoTorch and GPyTorch are the engine.** They own model classes, kernels, likelihoods, acquisition functions, acquisition optimizers, sampling and fitting. planexs' contracts reuse their native interfaces: a surrogate is, or exposes, a BoTorch `Model`; an acquisition is a BoTorch `AcquisitionFunction`. Where the engine already has a class, planexs registers a builder for it instead of re-implementing it. New mathematics that planexs has to write (for example a level-set acquisition function that the engine lacks) is written as a native engine subclass, so that it works with the engine's own optimizers.
- **Ax is optional.** It is one possible policy adapter (section 5.9), installed through an extra. Only the `adapters` layer may import `ax.*`, and the core MUST import and run without Ax installed.
- **Other engines** (scikit-learn, JAX, ...) are outside the core. A plugin may adapt one by satisfying the surrogate contract.

---

## 2. Functional requirements

Requirements are stated at the generality the architecture must support. A requirement written for one case (for example "a weight on exploration") is the first instance of a mechanism, not the mechanism itself (section 3.2). Identifiers FR1 to FR17 are stable and are cited by code and tests; FR18 onward were added with the v0.3 rewrite.

| ID | Requirement |
|---|---|
| FR1 | Declare a search space from typed parameters: real, integer, ordinal, categorical, boolean, fixed and derived. |
| FR2 | Declare what is measured (outcomes, each of a kind) and what is wanted (a goal). Optimizing one or several outcomes is one goal type among others. |
| FR3 | Inject already-known data (points with observations) without asking for them first. |
| FR4 | Obtain suggestions from a default policy that needs no manual tuning, for any goal type and at any budget. |
| FR5 | Control the balance between exploration and exploitation. The blend driven by a single weight (the "alpha dial") is one built-in acquisition, selected like any other. |
| FR6 | Report how much each parameter matters for each outcome (an analysis of the surrogate). |
| FR7 | Predict the outcomes at any parameterization (a surrogate operation, available without any loop). |
| FR8 | Report calibrated uncertainty for predictions and suggestions, and state what it describes: the latent function or the observed response, with or without observation noise. |
| FR9 | Constrain the inputs with linear inequalities and equalities. |
| FR10 | Request batches of suggestions that complement each other, and keep asking while earlier suggestions are still being evaluated. |
| FR11 | Save a study to a single file and reload it to continue. |
| FR12 | Reproduce a study: its seed, library versions, full trial history and fully resolved configuration are stored. |
| FR13 | Give parameters defaults and discrete steps. |
| FR14 | Declare conditional parameters (parameters that only exist when another parameter takes a given value). |
| FR15 | Constrain observed outcomes (for example a throughput that must stay above a floor) as part of a goal. |
| FR16 | Reach the underlying engine objects and supply custom components without forking planexs. |
| FR17 | Constrain the inputs with nonlinear expressions that survive saving and reloading. |
| FR18 | Select every component by name in configuration: surrogate (with its kernel, likelihood, transforms and fitting procedure), acquisition, acquisition optimizer, design, goal, analysis, stopping criterion, policy and repository. |
| FR19 | Register new components of any kind from user code or from an external package, with no change to planexs. |
| FR20 | Estimate level sets and boundaries of an outcome, for continuous outcomes and for binary (label) outcomes. |
| FR21 | Learn a surrogate efficiently with no optimization target (active learning, exploration). |
| FR22 | Use the surrogate alone: fit, predict, draw samples and run analyses without an ask/tell loop. |
| FR23 | Track the trial lifecycle (pending, completed, failed, abandoned) and support asynchronous asking. |
| FR24 | Support several outcome kinds (continuous and binary first), extensible by plugins. |
| FR25 | Validate that the chosen components are compatible when a study is built, with actionable errors. |
| FR26 | Expose and record every policy value: no hidden numeric or behavioural constant. |
| FR27 | Allow policies that planexs does not implement (for example Ax) as optional adapters honouring the same lifecycle. |
| FR28 | Provide stopping criteria as components. |

---

## 3. Design principles

These principles are normative. Each one exists because ignoring it has already caused a defect.

### 3.1 Layers own one job; misplaced requests are flagged

Every piece of behaviour belongs to exactly one layer (section 5.1). A request is first classified by the layer it belongs to. If it is phrased against a different layer (for example "add a prediction method to the optimizer" when prediction is a surrogate operation), the contributor MUST say so and propose the right place before implementing anything. Silently grafting a feature onto the nearest facade is a defect.

### 3.2 Generalize the axis of variation; never implement only the letter

Every request is an instance of something. Identify what varies (the axis), build the mechanism for the axis (a registry kind, a configuration field, a capability), and deliver the request as the first instance of that mechanism. State which other instances the mechanism now allows; do not implement them unless asked. A feature delivered as a one-off special case is a defect.

### 3.3 Every choice is a registered component selected by configuration

Any decision on which a reasonable user could want a different answer MUST be a named, registered component picked by configuration: model class, kernel, likelihood, transform, fitting procedure, acquisition function, acquisition optimizer, design, goal, analysis, stopping criterion, policy, repository, parameter kind, constraint kind, outcome kind. A class or default hard-wired inside library code to make such a choice is a defect. So is a swap point that exists only in a docstring: an abstract class that no code path actually uses to choose between implementations.

### 3.4 No hard-coded policy values

A *policy value* is any number, string, flag or heuristic that changes numerical results, performance or behaviour and could reasonably differ between users or problems: tolerances, budgets, sample counts, restart counts, iteration caps, thresholds, jitter and floors, seed-derivation constants, size limits, "use X when there are more than N" rules, default component names.

- Every policy value MUST live in a typed configuration schema, with a documented meaning, unit and default. The default is written in exactly one place: the schema field.
- Every policy value MUST be reachable by the caller through the study configuration. A constant buried in a function body, a module-level constant or a literal default in a function signature is a defect, even if it has a good name.
- Every study MUST store its fully resolved configuration (all defaults materialized), so that a later change of a default never alters how an old study behaves.
- Exempt are only structural constants that are part of a mathematical definition: 0, 1, -1 and 2 in formulas, tensor axis indices, and the bounds and midpoint of the unit cube. A structural constant that is not 0, 1, -1 or 2 carries an explicit marker with its reason (section 10.7).
- The automated check (section 10.7) only finds numeric literals. Strings, booleans and rules hidden behind an allowed literal (for example `len(values) > 2` standing for "more than two values means ordered") are policy values too; review is the control for those.

### 3.5 Reuse the engine; do not wrap or re-implement it

If BoTorch, GPyTorch or Ax already does something, planexs registers a builder for it. planexs writes its own mathematics only when no engine class exists, and then as a native engine subclass. Do not mirror an engine class with a parallel abstraction, and do not invent engine API: names and signatures are verified against the installed versions (section 10.9).

### 3.6 Fail early, and name the problem

Compatibility between components is declared (needs and supports, section 5.12) and checked when a study is built, not discovered deep inside an engine call. Errors derive from one base class and name the component, the configuration field and a way to fix the problem. A study never silently changes its policy afterwards.

### 3.7 Reproducible by construction

One study seed feeds every stochastic component through a documented, stable derivation; no component uses a global random state. A registered component name is a permanent identifier: a change that alters results requires a new name (for example a `_v2` suffix), never an edit of the old one, so that saved studies keep meaning what they meant. Snapshots carry the resolved configuration, the library versions and a format version.

### 3.8 Simple things easy, hard things possible

There are three levels of use, and each can be entered without abandoning the study and persistence machinery of the level below: (1) declare a problem and a goal and take the default policy; (2) select and configure any component by name, in Python or in a configuration file; (3) write and register new components, or a whole policy.

### 3.9 The domain is pure

The domain layer contains data, validation and serialization only. It imports no machine-learning library, and it knows nothing about engines or adapters. Anything that maps domain objects to the vocabulary of an engine (for example Ax constraint strings) lives with that engine's adapter.

### 3.10 Rule of two for seams

A new registry kind, or an abstract interface, is created only when at least two concrete implementations exist or are required by a reference use case (section 7). One implementation behind an interface is speculation. A seam must also be real: the code that needs the choice MUST obtain it through the registry, never through a direct import of one implementation.

### 3.11 Honest about uncertainty and limits

Predictions state what they describe and whether observation noise is included. A surrogate fitted on very little data says so. The limits of a component are declared, not discovered by failure.

---

## 4. Glossary

These terms are canonical. Do not invent new names for the same concepts.

- **Problem**: a search space together with the outcomes that are measured.
- **Search space**: the parameters and the parameter constraints.
- **Parameter**: one dimension of the search space. Types: range (real or integer, bounds, optional step and log scaling), choice (ordinal or categorical, optionally with dependent parameters) and derived (computed from other parameters, never searched). Sugar constructors (real, integer, categorical, boolean, fixed) build these types and are not stored types.
- **Constraint** (parameter constraint): a condition on the input parameters. Types: linear and nonlinear. Constraints on observed outcomes belong to a goal, not to the search space.
- **Outcome**: a measured quantity, with a name and a type (its *outcome kind*).
- **Outcome kind**: how an outcome's values are validated and which surrogates suit it (continuous, binary, ...).
- **Trial**: one parameterization with an identity and a state. States: pending (suggested, not yet reported), completed (has an observation), failed (the evaluation did not produce one; carries a reason) and abandoned (withdrawn by the caller).
- **Observation**: the values of the outcomes for a completed trial, with optional per-outcome noise standard deviations.
- **Study**: the object a caller drives. It owns the problem, the goal, the policy and the trial history, and offers ask, tell, fail, abandon, attach, analyses, stopping checks, surrogate access and save/load.
- **Goal**: a typed statement of what the caller wants to know or decide (optimize, level set, explore, ...). It is configuration, not an algorithm: it parameterizes acquisitions and analyses and supplies default recipes.
- **Surrogate**: a probabilistic model of the outcomes as a function of the parameters, fitted on the observations (a Gaussian process by default), together with the encoding that binds it to a problem.
- **Acquisition**: a score over candidate points, built from a surrogate, a goal and the pending points. Higher means more worth evaluating.
- **Acquisition optimizer**: the procedure that searches the encoded space for the points that maximize an acquisition, subject to the constraints.
- **Design**: a procedure that proposes points without a model (space-filling or random), always satisfying the constraints.
- **Policy**: the component that decides which parameterizations to evaluate next. A *composed* policy is made of a design, a surrogate, an acquisition and an acquisition optimizer. An *opaque* policy is any object honouring the policy contract (random search, an Ax adapter, user code).
- **Recipe**: the default composed-policy configuration that a goal proposes when the caller gives none.
- **Analysis**: a computation over the study state producing a typed, serializable result (best point, Pareto front, level-set estimate, parameter importance, diagnostics). An analysis never mutates the study.
- **Stopping criterion**: a component that looks at the study state and answers continue or stop, with a reason.
- **Component**: any registered, named, configurable piece (a surrogate, an acquisition, a goal, ...). Components belong to a *kind*.
- **Kind**: a category of components that share a contract and a registry (surrogate, acquisition, goal, ...).
- **Registry**: the table of the components of one kind, keyed by name.
- **Builder**: what a registry entry uses to create the native object of a component from a validated configuration and a build context.
- **Build context**: the read-only information a builder receives (problem, encoder, data tensors, goal, pending points, random generator, runtime settings, and the components it depends on).
- **Component configuration**: the typed, validated settings of one component, discriminated by its `type` name. Nested components use the same form.
- **Resolved configuration**: a configuration with every default materialized; what snapshots store.
- **Capability tag**: a short string from a documented vocabulary naming a feature (for example `posterior:gaussian` or `space:categorical`). Participants declare tags they **need** and tags they **support**.
- **Plugin**: an external package that registers components, discovered through an entry point or loaded explicitly.
- **Encoding**: the mapping between parameterizations and tensors in the unit cube, on which surrogates, acquisitions and optimizers work.
- **Snapshot**: the single JSON document that stores a study.
- **Seam**: a place where one implementation can be swapped for another (in practice, a registry kind).
- **Reference use case**: an entry of the acceptance matrix of section 7.

---

## 5. Architecture

### 5.1 Layers and dependency direction

| Layer | Responsibility | May depend on |
|---|---|---|
| `core` | Registries and the kind table, component metadata, capability tags and the compatibility check, the component configuration base, errors, random-number derivation, runtime settings, plugin discovery. | nothing in planexs |
| `domain` | Parameters, constraints, outcomes, trials, observations, search space, problem. Pure data, validation and serialization. | `core` |
| `encoding` | Parameter and constraint encoding rules, the search-space encoder, encoded constraints, feasible sampling, the expression compiler. | `core`, `domain` |
| `surrogates` | Surrogate contract and builders, nested kinds (kernels, likelihoods, outcome transforms, fitters), the fitted-surrogate wrapper, prediction results, the cache. | `core`, `domain`, `encoding` |
| `goals` | Goal contract and built-in goals. | `core`, `domain` |
| `designs` | Design contract and built-in designs. | `core`, `domain`, `encoding` |
| `acquisition` | Acquisition contract, builders, native acquisition functions. | `core`, `domain`, `encoding`, `surrogates`, `goals` |
| `optimizers` | Acquisition-optimizer contract and built-ins. | `core`, `domain`, `encoding`, `acquisition` |
| `analyses` | Analysis contract and built-ins. | `core`, `domain`, `encoding`, `surrogates`, `goals` |
| `stopping` | Stopping-criterion contract and built-ins. | `core`, `domain`, `surrogates`, `goals`, `analyses` |
| `policies` | Policy contract, composed policy, recipes, opaque built-in policies. | every layer above |
| `persistence` | Snapshot document, repository contract and built-ins, migrations, reproducibility metadata. | `core`, `domain` |
| `config` | Study-level configuration and its loaders. | every registry layer |
| `study` | The `Study` facade. | `core`, `domain`, `goals`, `policies`, `analyses`, `stopping`, `persistence`, `config` |
| `adapters` | Optional adapters to third-party systems (Ax). | planexs' public contracts, plus the third-party package |

A lower layer never imports a higher one. `core` and `domain` import no machine-learning library (torch, BoTorch, GPyTorch, Ax); `adapters` are never imported by the core except through plugin discovery. The rules are checked in CI (section 10.5). Persistence holds configuration as plain JSON-compatible data, which is why it does not depend on `config`.

### 5.2 The central loop

A study holds a problem, a goal, a policy and a history of trials. The caller asks for suggestions; the study gives the policy a read-only view of its state (every trial in every state, the goal, the problem) and receives candidate parameterizations. The study validates them against the problem, records them as pending trials with fresh identifiers and returns them. The caller evaluates them wherever and whenever it likes, then reports back: a completed trial with an observation, a failed trial with a reason, or an abandoned trial. Data gathered elsewhere can be attached at any time. Analyses and stopping criteria can be run at any time and never change the history.

Asking never requires the previous suggestions to have been reported. Pending trials are part of the state the policy sees, so batches and asynchronous workers are first-class (FR10, FR23).

### 5.3 Problem description (domain)

**Search space.** An ordered set of uniquely named parameters and a set of parameter constraints. Parameter types are registered: range, choice and derived. A range has bounds, an integer or float type, an optional step and optional log scaling. A choice has values, an optional ordering flag (when unset, the encoding configuration decides how the choice is treated, section 3.4) and optional dependent parameters (conditional search spaces, FR14). A derived parameter is computed from others by an expression and is never searched. Any parameter may have a default value (FR13), validated against its own definition. Constraint types are registered too: linear (less-or-equal, greater-or-equal and equality, FR9) and nonlinear (an expression string in a restricted arithmetic grammar, never evaluated with Python's `eval`, FR17). Constraints are stated on natural values (what the caller sees), not on the encoding.

**Serialization.** Every type has a stable tag (the `type` key) and round-trips through JSON exactly. Sugar constructors (real, integer, categorical, boolean, fixed) produce the base types and do not survive a round trip as separate types.

**Purity.** The domain never contains engine vocabulary. In particular it contains no code that renders a parameter or a constraint in another system's syntax (section 3.9).

### 5.4 Outcomes, trials and observations

An **outcome** has a name and a kind. The kind decides how values are validated (a continuous value is a finite number; a binary value is a label), whether a noise standard deviation is meaningful, and which surrogates suit it. Kinds are registered; continuous and binary come first (FR24).

A **trial** has a stable integer identifier assigned by the study and never reused, a parameterization, a state, an optional observation, optional free-form JSON-compatible metadata and, when failed, a reason. The allowed transitions are pending to completed, pending to failed and pending to abandoned. Data attached from outside enters directly as completed (or failed). The history is never silently rewritten: trials are not deleted or overwritten.

An **observation** gives a value for every outcome of the problem and, where known, a noise standard deviation per outcome. Replicated evaluations of the same parameterization are allowed. Partial knowledge of noise (some observations with a deviation, some without) is handled by the surrogate according to its configuration, never by a hidden rule.

### 5.5 Encoding

Surrogates, acquisitions, optimizers and designs work on one encoding of the search space: a tensor in the unit cube whose columns are laid out by the parameter types. A range is one column, linear or logarithmic; an integer range or a stepped range is a continuous relaxation that is snapped back onto its grid; an ordered choice is one rank column; an unordered choice is a 0/1 column or a one-hot block; fixed and derived parameters occupy no column. Conditional parameters are encoded as ordinary columns, with a documented neutral value when inactive.

The encoder is deterministic. Encoding then decoding a legal parameterization returns it exactly; decoding any point of the unit cube returns a legal parameterization (rounded, with inactive conditional parameters removed and derived parameters computed). Constraints are rewritten onto the encoding, and a point that violates them is never returned to the caller.

Each parameter type and each constraint type owns its encoding rules, registered under its type name in the `parameter_encoding` and `constraint_encoding` kinds. The encoder iterates over registered rules; it does not branch on concrete classes. A type with no registered rule is an error raised when the encoder is built, naming the type and, when known, the plugin that should provide it.

### 5.6 Surrogates

A surrogate builder takes a build context (problem, encoder, observations as tensors with optional noise, random generator, runtime settings) and returns a fitted native model plus a **fit status**. The result is wrapped in a `Surrogate`, which binds the model to the encoding and offers the surrogate operations: predict at parameterizations, draw posterior samples, expose the native model, report the fit status and the outcomes it covers. A `Surrogate` works without a study (FR22).

- **Composition.** A builder may expose its internal choices as nested kinds: kernel, likelihood, outcome transform, fitting procedure. Nested kinds use the same registry and configuration mechanism as any other (section 5.12). Which nested kinds exist is not fixed by this document.
- **Outcome kinds.** A surrogate declares the outcome kinds it supports. A binary outcome needs a non-Gaussian likelihood and approximate inference; its predictions exist in two spaces, the latent function and the response probability, and a prediction always states which one it returns (section 3.11).
- **Several outcomes.** How several outcomes are modelled (independent models, joint model) is the builder's declared choice and appears in its capability tags.
- **Prediction semantics.** By default a prediction reports the uncertainty of the underlying function (epistemic, without observation noise). Including observation noise is an option of the prediction (FR8).
- **Small data.** The default surrogate MUST remain usable with a handful of observations (section 8) and MUST report a low-data fit status instead of hiding it. The minimum number of observations needed to fit, and the threshold below which a fit is flagged as low-data, are policy values.
- **Caching.** Refitting a surrogate whose data and configuration are unchanged is avoided. The cache key is the history version and the resolved configuration.

### 5.7 Goals

A goal is typed configuration, validated against the problem (the outcomes it names must exist and be of a suitable kind). It does three things and computes nothing. It **parameterizes** acquisitions and analyses (directions and weights, a level, constraints on outcomes). It **declares** capability tags (for example that it is a level-set goal). It **proposes** recipes (default policy configurations, in order of preference) and default analyses.

The first goal types are *optimize* (one or several outcomes, a direction for each, optional scalarizing weights, optional constraints on outcomes: FR2, FR15), *level set* (one outcome and a level: the boundary where the outcome crosses it or, for a binary outcome, where the probability crosses the level: FR20) and *explore* (learn the outcomes as well as possible: FR21). A goal type is a registered component. Adding one never changes the core.

### 5.8 Acquisition, acquisition optimizers and designs

An **acquisition builder** receives the surrogate, the goal, the pending points, a reference set of encoded points where it needs one, and a random generator, and returns a native acquisition function. The catalogue is open. The exploration/exploitation blend with a single weight (FR5) is one entry among others, alongside the expected-improvement and upper-confidence-bound families, multi-objective hypervolume improvement, variance-based exploration and level-set acquisitions such as the straddle.

An **acquisition optimizer** searches the encoded space for the point or batch that maximizes an acquisition under the constraints. It handles the mixed space (relaxation of grids and ordered choices followed by snapping, enumeration of categorical assignments) as its configuration says. It receives the pending points so that batches complement each other (FR10), and it returns only constraint-satisfying encoded points. Its fallbacks (what to do when rounding breaks a constraint, how many random candidates to try) are configuration, never hidden code paths.

A **design** proposes points without a model: space-filling sequences or random draws. It always honours the constraints, and it can count the existing observations so that a warm start (FR3) does not repeat an initial phase unnecessarily.

### 5.9 Policies

A policy answers one question: which parameterizations should be evaluated next, given the state of the study. It is told about every change of a trial's state and is asked for suggestions. It may keep internal state, which it MUST be able to export and import so that persistence works.

A **composed policy** is made of a design, a surrogate, an acquisition and an acquisition optimizer. It uses the design until a configured amount of data exists, then switches to the model-based phase; the switching rule is configuration. In the model-based phase it obtains a surrogate from the history (through the cache), builds the acquisition with the pending points, optimizes it, and decodes and validates the result.

An **opaque policy** is any object honouring the policy contract: random search, grid search, an Ax adapter (FR27), or user code. A study treats both forms identically.

**Defaults.** When the caller gives no policy, the study asks the goal for its recipes and takes the first one whose components are compatible with the problem (section 5.12). The *resolved* policy is stored in the study's configuration and can be inspected. It is never silently different from what is reported, and a study never switches policy afterwards. If no recipe is compatible, construction fails with an error that lists, for each unmet need, the registered components that would satisfy it.

### 5.10 Analyses and stopping

An **analysis** is a registered computation over a read-only study state that returns a typed, serializable result. It may need a surrogate (it asks the study for one), a goal type, or specific capabilities, and it declares so. The catalogue is open: recommendations (the best point, a Pareto front, a level-set estimate: the answer to the goal), parameter importance, prediction grids and slices, and model diagnostics such as leave-one-out checks and calibration (FR6, FR7, FR8). Each goal type lists default analyses.

A **stopping criterion** is a registered evaluation of the study state that answers continue or stop, with a reason (a trial budget, no recent improvement, a resolved level set). The study reports the answer; it never stops the caller's loop by itself (FR28).

### 5.11 Persistence and reproducibility

A study is saved as one JSON document, human-inspectable (FR11, FR12). It contains: a format version; the study name; the problem; the goal; the **resolved** policy, analysis and stopping configuration; the full trial history with every state and reason; the number of asks so far; reproducibility metadata (study seed, versions of planexs and of the tracked libraries, creation time); a manifest of the registered component names in use (with plugin versions where known); and, where a component provides one, its exported opaque state (for example an adapter's own client state).

Loading rebuilds an equivalent study from the document alone. In the same environment, a reloaded study makes the same next suggestions as the uninterrupted one would have, up to documented hardware nondeterminism. Loading fails with a clear error that names any component missing from the registries and, when known, the plugin that provides it. A library-version mismatch is reported, not ignored. A snapshot is data: loading it never executes code from the file. Format changes ship with migrations so that older documents stay loadable. A repository (the storage medium) is a registered component: JSON files first and an in-memory repository for tests; others, such as a SQL store, can be added without touching the study.

Randomness: one study seed, recorded in the snapshot, feeds a stable derivation keyed by the ask index and the component path. Components never touch a global random state, and adding a component does not change the random streams of the others. When the caller gives no seed, one is drawn from the operating system and recorded.

### 5.12 Registries, configuration, capabilities and plugins

**Registries.** There is one registry per component kind. A component is registered under a unique name with a decorator, in the style `@registerSurrogate("single_task_gp_v1")`. A duplicate name is an error; an unknown name is an error that lists the available names. Names are lowercase snake_case, with a `_vN` suffix when behaviour differs between versions (section 3.7). An entry holds the builder, the configuration schema, a one-line description and the declared needs and supports.

**Configuration.** Every component has a typed configuration (a Pydantic model) whose `type` field is the registered name; unknown fields are rejected. Components can nest other components in the same form (a surrogate holds a kernel, a composed policy holds a design, and so on). Configurations load from Python objects, JSON or YAML with identical meaning, and every validation error carries the path of the offending field. Defaults exist only in the schemas (section 3.4).

**Capabilities.** Tags come from a documented vocabulary; an unknown tag is an error (this catches typos). The problem and the goal contribute tags too: a problem with categorical parameters needs `space:categorical`, a goal of type level set supports `goal:level_set`, several outcomes need `outputs:multi`. When a study is built, the compatibility check verifies that every need is met by a support from some participant (the problem, the goal, or a component of the policy). Unmet needs are reported together, with who needs each one and which registered components would meet it. A component may refine its declared tags from its own configuration.

**Population.** A decorated component is registered only when its module has been imported. Every subpackage's `__init__` MUST import each of its concrete modules, and a test checks this.

**Plugins.** An external package registers components by decorating them and is discovered through an entry-point group, or loaded explicitly by module name. A plugin needs no change to planexs and may extend the capability vocabulary. Plugins are trusted code; this is documented.

### 5.13 Escape hatches

The native objects are always reachable (FR16). The study's current surrogate exposes the BoTorch model the policy is actually using, never a different one. The policy object is accessible, and an adapter exposes its own client. Anything obtained this way is outside what planexs validates or keeps in sync, and the documentation says so. Writing a custom component is the preferred escape hatch, because it keeps persistence, reproducibility and compatibility checking.

---

## 6. Extension contracts

Every component, whatever its kind, MUST provide:

- a stable registered name (section 5.12);
- a configuration schema whose every field is described, including the meaning, unit and default of each policy value (section 3.4);
- a one-line description;
- declared needs and supports, truthful with respect to its behaviour (section 5.12);
- a builder that is free of side effects at import and registration time, and deterministic given its configuration, its inputs and its random generator.

What a component of each kind adds:

| Kind | A component provides | It receives and returns |
|---|---|---|
| `parameter` | Schema, validation of values and defaults, a serialization tag. | Builds a domain parameter object. |
| `constraint` | Schema, serialization tag, a feasibility check on natural values. | Builds a domain constraint object. |
| `outcome_kind` | Value validation, rules on noise information, the tags of the surrogates that suit it. | Validates observed values. |
| `parameter_encoding` | Column layout, encoding, decoding with snapping, the neutral value when inactive, sampling and rounding rules for one parameter type. | Used by the encoder. |
| `constraint_encoding` | The encoded form of one constraint type (coefficient rows, or a differentiable callable) and its violation measure. | Used by the encoder and the optimizers. |
| `surrogate` | A fitted native model for the given data, a fit status, the outcome kinds it supports. | Receives a build context; returns a model and a status. |
| `kernel`, `likelihood`, `outcome_transform`, `fitter` | The corresponding native objects or procedures for a surrogate builder. | Receive the dimensions or the model they act on. |
| `design` | Encoded points without a model. | Receives problem, encoder, constraints, existing trials, a count and a generator. |
| `goal` | Configuration, tags, recipes, default analyses, validation against the problem. | Receives the problem. |
| `acquisition` | A native acquisition function. | Receives surrogate, goal, pending points, reference points and a generator. |
| `acquisition_optimizer` | Encoded candidates that maximize an acquisition under the constraints. | Receives acquisition, encoder, constraints, a count, pending points and a generator. |
| `policy` | Suggestions, notification of trial changes, export and import of opaque state. | Receives the read-only study state. |
| `analysis` | A typed, serializable result. | Receives the read-only study state and, on demand, a surrogate. |
| `stopping` | A decision and a reason. | Receives the read-only study state. |
| `repository` | Saving and loading snapshot documents. | Receives and returns a snapshot document. |

The set of kinds is itself open. A new kind needs two concrete implementations (section 3.10) and an ADR.

---

## 7. Reference use cases

The reference use cases are the acceptance matrix of the architecture. They exist so that "everything is possible" (section 1.3) is verified and not merely claimed, and so that no contributor narrows the design to the exact request in front of them.

### 7.1 The matrix

| ID | Use case | Outcomes | Goal | What it proves |
|---|---|---|---|---|
| UC-1 | Single-objective optimization, noisy or deterministic, with input constraints | continuous | optimize | The default policy and the basic loop (FR1, FR4, FR9). |
| UC-2 | Multi-objective optimization (Pareto front), with and without weights | several continuous | optimize | Multi-output surrogates, hypervolume acquisition, the Pareto analysis (FR2). |
| UC-3 | Optimization with outcome constraints | several continuous | optimize with constraints | Goal-level constraints reaching the acquisition (FR15). |
| UC-4 | Level-set estimation of a continuous outcome | continuous | level set | A goal that is not optimization: level-set acquisition and estimate (FR20). |
| UC-5 | Boundary between two phases from binary labels | binary | level set | A non-Gaussian surrogate, latent and response spaces: the headline "phase boundary" case (FR20, FR24). |
| UC-6 | Pure exploration (active learning) | continuous | explore | An acquisition with no direction (FR21). |
| UC-7 | Surrogate only: fit, predict, sample, importance, diagnostics, no loop | continuous | none | The separation of layers (FR6, FR7, FR8, FR22). |
| UC-8 | Batches, asynchronous asks, failed and abandoned trials | any | any | The trial lifecycle and pending points (FR10, FR23). |
| UC-9 | Mixed, conditional and derived parameters with linear (including equality) and nonlinear constraints | any | any | The search space and its encoding (FR1, FR9, FR13, FR14, FR17). |
| UC-10 | Warm start from existing data, then save, reload and continue | any | any | FR3, FR11, FR12 and deterministic replay. |
| UC-11 | A study that uses a kernel, an acquisition and an analysis defined outside the package | any | any | The extension guarantee (FR18, FR19, section 1.3). |
| UC-12 | An opaque policy (random search, a user-written policy, later an Ax adapter) driving the same study | any | any | The policy seam (FR27). |

### 7.2 Rules for the matrix

- Each use case MUST run from configuration plus registered components alone. UC-11 additionally runs from a separate package that imports only public contracts.
- Each use case MUST have at least one integration test and one runnable example.
- Each test states its acceptance numbers (budgets, tolerances, comparison with a random baseline) in the test itself, and is deterministic through seeds.
- A change that breaks a use case is a regression even if every unit test passes.

### 7.3 Horizon use cases

Multi-fidelity and cost-aware optimization; safe exploration; high-dimensional trust-region search; preference and ranking data (comparisons instead of point observations); contextual problems; Bayesian quadrature; low-change (proximal) suggestions for laboratory practicality; transfer across studies.

None is scheduled. The architecture MUST NOT make any of them impossible, and implementing one may require an ADR (for example, preference data changes the observation model). Horizon use cases are used in design reviews to challenge abstractions: if a proposed interface would forbid one of them, say so.

---

## 8. Quality attributes

- **Budget range.** Defaults work from four or five observations to thousands. Scaling is a matter of components: a surrogate declares how it scales through a capability tag, and scalable surrogates are plugins.
- **Numerical robustness.** Double precision by default for fitting. Jitter, noise floors and minimum variances are policy values. Non-finite values never propagate silently: they raise a named error.
- **Determinism.** As in section 5.11.
- **Errors.** As in section 3.6: one base class, messages that name the component, the field and a fix.
- **Efficiency.** A surrogate is not refitted when its data and configuration are unchanged. Constant data (for example a reference set) is not recomputed within one ask. This document sets no performance figures.
- **Observability.** Standard `logging`. Each ask logs at INFO which phase and which components were used. Degraded situations (a low-data fit, a fallback taken, a failed plugin) log a warning. Library code never prints.
- **API stability.** The public API is the set of names re-exported by the top-level package. Versions are semantic; before 1.0 a minor version may break the API, and every break is listed in the changelog.
- **Security.** Expressions are never evaluated with `eval`. A snapshot is data and loading it executes no code from the file. A plugin is code and is trusted.
- **Portability.** Pure Python. CPU by default; the device is a runtime setting. No module-level global state.
- **Documentation.** Every extension kind and every reference use case has a how-to.

---

## 9. Package map and illustrative configurations

### 9.1 Package map

The package layout mirrors the layers of section 5.1: `src/planexs/<layer>/`, plus the top-level `logging_config.py` and the public, lazily loaded re-exports in `__init__.py`. Other directories:

- `tests/` mirrors `src/` (`tests/unit/<layer>/`), with `tests/integration/` for cross-layer tests and `tests/use_cases/` for the reference use cases of section 7.
- `examples/` holds runnable walkthroughs (at least one per reference use case), and `examples/plugins/` shows components that live in external packages.
- `scripts/` holds repository tooling: the naming check, the policy-value check, the layering check, and the ADR and release scaffolding.
- `docs/` holds this specification, the roadmap, the ADRs, the changelogs, the how-to guides and the API reference.

### 9.2 Illustrative configurations

Non-normative. Component names and numbers below are the caller's choices in these examples; the catalogue of built-in components is documented separately and is not fixed by this specification.

**Optimization with the default policy.** The goal proposes recipes and the first compatible one is used. The resolved policy is stored in the snapshot.

```yaml
problem:
  parameters:
    - {type: range, name: temperature, bounds: [20.0, 80.0]}
    - {type: range, name: concentration, bounds: [0.01, 1.0], scaling: log}
    - {type: choice, name: catalyst, values: [A, B, C], parameter_type: str, is_ordered: false}
  outcomes:
    - {name: yield, type: continuous}
goal:
  type: optimize
  outcomes: {yield: maximize}
```

**Boundary between two phases from binary labels, with an explicit policy.** The surrogate is a classifier; the acquisition scores closeness to the boundary weighted by uncertainty.

```yaml
problem:
  parameters:
    - {type: range, name: temperature, bounds: [300.0, 900.0]}
    - {type: range, name: pressure, bounds: [1.0, 100.0], scaling: log}
  outcomes:
    - {name: phase, type: binary}
goal:
  type: level_set
  outcome: phase
  level: 0.5
policy:
  type: composed_v1
  design: {type: sobol_v1, n_points: 12}
  surrogate:
    type: variational_gp_classifier_v1
    kernel: {type: matern_v1, nu: 2.5, ard: true}
  acquisition: {type: straddle_v1, beta: 1.96}
  acquisition_optimizer: {type: botorch_mixed_v1}
analyses:
  - {type: level_set_estimate_v1}
stopping:
  - {type: max_trials_v1, n_trials: 60}
seed: 7
```

**A component defined in user code.** The shape only; the exact decorator arguments are defined by the code and its docstrings.

```python
@registerAcquisition("my_boundary_score_v1")
class MyBoundaryScoreBuilder(AbstractAcquisitionBuilder):
    ...  # configuration schema, needs and supports, a build method
```

After registration, `acquisition: {type: my_boundary_score_v1}` works in any configuration, and the study stores, reloads and compatibility-checks it like a built-in component.

---

## 10. Coding standards

### 10.1 Tooling and runtime

Python 3.11 or later. `uv` manages the project and its dependencies (ADR-0002). `ruff` lints and formats (its rules live in `pyproject.toml`). `mypy` runs in strict mode. `pytest` runs the tests. `mkdocs` with `mkdocstrings` builds the documentation. GitHub Actions runs everything on every push and pull request (section 10.13).

### 10.2 Naming

The project deliberately deviates from PEP 8 for callables and for file names. The deviation is enforced, so do not "fix" it back to snake_case.

- **Classes**: `CamelCase`.
- **Variables**, including function and method parameters: `snake_case`.
- **Functions and methods**: `camelCase` (for example `buildSurrogate`, `registerAcquisition`). Python's required dunder methods keep their mandatory spelling.
- **File names match the file's single public symbol exactly, including case**: a file defining `class Study` is `Study.py`; a file defining `def buildSurrogate` is `buildSurrogate.py`. A file with several closely related symbols and no single obvious name keeps a descriptive `snake_case` name (`logging_config.py` is that case). Python's required module names (`__init__.py`, `conftest.py`) are exempt.
- **Properties** are named by cost, not by whether they compute anything. If reading the property does real work (recomputes from the history, calls a model), it is `camelCase` like a method. If it returns an already-stored or negligible-cost value, it is `snake_case` like a variable. The caller can tell from the name whether touching it is free.
- **Registered component names** are lowercase snake_case with an optional `_vN` suffix. **Capability tags** are lowercase `group:name`. **Configuration keys** are snake_case.

`ruff` has the `N802`, `N803`, `N806` and `N999` rules disabled for this reason, and `scripts/check_naming_convention.py` checks the file-name rule in CI.

### 10.3 Typography

No em dashes in code, comments, docstrings, commit messages or project documentation. Use a period, a colon, parentheses or two sentences instead. No unicode arrows: write `->`. A test enforces this, ignoring inline code spans so that the rule itself can be documented.

### 10.4 Typing and docstrings

Type hints are mandatory everywhere, and `mypy --strict` passes. Docstrings are Google style and mandatory on every public class, function and method: purpose, `Args`, `Returns`, `Raises`. The description of a configuration field states what the value means, its unit, and why its default is what it is.

### 10.5 Modularity and layering

One responsibility per file; one class per file for components. A base class, its registry, its decorator and every concrete implementation each get their own file. No god-files. Subpackages depend on each other only in the direction of section 5.1; `scripts/check_layering.py` enforces it, and enforces that `core` and `domain` import no machine-learning library and that only `adapters` import Ax.

### 10.6 Registry rules

- Every choice that has a registry kind goes through the registry (section 3.3). No algorithm imports a concrete component class to make a choice.
- Registration is by decorator, at import time.
- Every subpackage's `__init__` imports each of its concrete modules, one line per file, or the component is silently unregistered. A completeness test checks this.
- A registered name is permanent. A change that alters results gets a new `_vN` name.
- Every kind has an abstract contract, a registry and a registration decorator, each in its own file.
- Every entry declares needs and supports, and a test checks that the declaration is truthful: a component that declares it supports something works with it, and one that does not declare it fails the compatibility check with a clear message.
- One contract test runs over every registry (names, schemas, descriptions, tags, builders).

### 10.7 Policy values

Section 3.4 states the rule. The mechanism:

- `scripts/check_policy_values.py` finds numeric literals other than 0, 1, -1 and 2 (any numeric spelling) in library code. Files whose single public symbol is a configuration schema (a class named `...Config`) are where defaults legitimately live and are not scanned. Tests and scripts are not scanned.
- A literal is exempt when its line carries `# structural: <reason>` with a non-empty reason (for example `# structural: tensor axis`, `# structural: unit-cube midpoint`). The marker is for mathematical definitions only; using it on a tunable value is a defect that review must catch.
- The check is a ratchet against a baseline file. A new violation fails the check, and so does a stale baseline entry: entries are only ever removed. Writing the baseline creates it or removes entries; adding entries to an existing one is not done unless `--allow-additions` is given, for a deliberate re-baseline such as a file move. The baseline MUST be empty before the first release of the new architecture.
- The check cannot see strings, booleans or rules hidden behind an allowed literal. Review is the control for those.

### 10.8 Configuration

Configuration schemas are Pydantic v2 models with unknown fields rejected, a description on every field, and validators whose messages name the field. Defaults live only in schemas. There is no dictionary-of-strings configuration scattered through the code. An operation with tunable options takes one options object (itself a configuration schema) instead of a list of keyword arguments with literal defaults. A configuration built in Python, loaded from JSON and loaded from YAML has identical meaning.

### 10.9 Dependencies and engine APIs

Core dependencies: torch, botorch, gpytorch, pydantic, numpy, and a YAML reader. Optional extras: `ax`, `dev`, `docs`. Optional packages are imported lazily, inside the adapter that needs them, and the core MUST import and run without them. Version ranges go in `pyproject.toml` and exact versions in `uv.lock`.

Third-party names and signatures are verified against the installed version before they are used (introspection, or reading the installed source). Never write engine API from memory. If something a task or a specification names does not exist in the installed version, stop and report it instead of inventing a substitute.

### 10.10 Errors and logging

Every error raised by the library derives from one base class and carries a message that names the component, the field and a fix. Loggers are `logging.getLogger(__name__)` and are children of the `planexs` logger, which carries a `NullHandler`; library code never configures handlers. Library code never calls `print` (examples and scripts may print their results).

### 10.11 Testing

`pytest`. The following are required:

- unit tests per module, mirroring the source tree;
- the registry contract test over every kind (section 10.6) and the completeness test;
- round-trip tests for everything serializable (domain objects, configurations, snapshots) and for the encoder (encode then decode returns the parameterization);
- capability-truthfulness tests (section 10.6);
- determinism tests: the same seed gives the same suggestions, and save, reload and continue gives the same suggestions as an uninterrupted run;
- one integration test per reference use case (section 7);
- a test that `core` and `domain` import no machine-learning library, and one that the package imports and runs a default study without Ax;
- the typography test (section 10.3);
- parity tests whenever a component replaces another: the new one is compared with the old on a fixed problem, within tolerances stated in the test, before the old one is deleted.

No test uses the network. Stochastic tests use fixed seeds and state their tolerances; a tolerance is never loosened to make a test pass without saying so. Coverage has a floor in CI that only ever goes up.

### 10.12 Documentation and records

`mkdocs` and `mkdocstrings` build the API reference from docstrings. Every extension kind and every reference use case has a how-to. Architectural decisions are ADRs under `docs/adr/`, written from the project's template with `scripts/new_adr.py`.

### 10.13 Version control and CI

Conventional Commits and semantic versioning. One task, one pull request. Each release gets its own `docs/changelogs/changelog-vX.Y.Z.md`, written once and not edited afterwards, and `docs/changelogs/CHANGELOG.md` is the running index (`scripts/cut_release.py` scaffolds both).

CI runs on every push and pull request: `ruff check`, `ruff format --check`, the naming check, the policy-value check, the layering check, `mypy`, `pytest` with coverage, and the documentation build.

### 10.14 New-component checklist

1. Define the configuration schema, with every policy value described.
2. Implement the builder, with no literal policy value.
3. Register it under a new name, one class per file.
4. Declare its needs and supports.
5. Import it in its subpackage's `__init__`.
6. Add tests: validation of the configuration, behaviour, capability truthfulness. The registry contract test covers it automatically.
7. Add or update the documentation entry.
8. If it creates a new kind, write an ADR and satisfy the rule of two (section 3.10).

---

## 11. Decisions and records

- Architectural decisions are ADRs: context, decision, options considered, consequences. When a decision changes, a new ADR supersedes the old one; the old one's status line says so and its text is not edited.
- Where an existing ADR conflicts with this specification, this specification wins, and the roadmap carries a task to supersede that ADR with a new one.
- Open decisions are tracked in the roadmap's decision table, each with a proposed default. A contributor who needs an open decision MUST ask. A proposed default is used only once the owner has accepted it, and its use is recorded in the pull request description.

---

## 12. How contributors (human or LLM) use this document

### 12.1 Rules

1. Treat the terms of section 4 as canonical.
2. Before implementing a request, run the request check (section 12.2) and write down its answers.
3. Flag, do not silently adapt (section 12.3).
4. New choices are components: registry, configuration, capabilities (sections 3.3, 5.12, 10.6).
5. New tunable values are configuration fields, never literals (sections 3.4, 10.7).
6. Do not narrow the design to the exact request. Deliver the request as the first instance of a mechanism and say what other instances the mechanism now allows (section 3.2).
7. Respect the task boundaries of the roadmap: one task at a time, all of it. If a task is too big, stop and split it; never shrink it silently. Placeholders (`pass`, `NotImplementedError`, `TODO`) are not deliverables.
8. Tests, documentation and ADRs ship with the code.
9. When something is ambiguous, or an open decision is needed, ask.

### 12.2 The request check

Answer these six questions, in writing, before coding:

1. **Layer.** Which layer owns this? Name it.
2. **Axis.** What varies here? Name at least two other plausible values of the thing being requested.
3. **Seam.** Is there a registry kind for that axis? If not, does the rule of two (section 3.10) justify creating one?
4. **Values.** Which policy values does this introduce, where do they live in the configuration, and what are their defaults?
5. **Capabilities.** Which tags does it need, and which does it support?
6. **Use cases.** Which reference use cases does it touch or enable, and which tests change?

### 12.3 What must be flagged

A contributor MUST stop and raise the point, instead of implementing silently, when a request:

- belongs to a different layer from the one it targets;
- would hard-code a choice that has, or should have, a registry kind;
- would add a literal policy value;
- would narrow a mechanism to the requested instance;
- conflicts with a principle of section 3 or with a reference use case;
- needs a decision that is still open in the roadmap.

### 12.4 Errors this section exists to prevent

These happened in this project's history. They are listed so that they are recognized when they come back in a new form.

- A model-level operation (prediction, parameter importance) was attached to the optimizer facade without saying that it belonged to the surrogate layer.
- A specific model class, outcome transform and fitting routine were hard-wired inside a function documented as the "swappable" place to change them.
- Numeric constants with good names were treated as acceptable, although the caller had no way to change them.
- An exploration weight was delivered as the only way of choosing points, instead of as one acquisition among others.
- A second engine and a second surrogate were built beside the first to work around a limit, instead of moving the seam so that both could use one.
