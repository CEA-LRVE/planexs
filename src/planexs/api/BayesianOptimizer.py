"""The public facade: `BayesianOptimizer`."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from planexs.acquisition.AcquisitionStrategy import AcquisitionStrategy
from planexs.acquisition.AlphaAcquisitionStrategy import AlphaAcquisitionStrategy
from planexs.analysis.PredictionResult import PredictionResult
from planexs.backends.ax.AxBackend import AxBackend
from planexs.backends.OptimizationBackend import OptimizationBackend
from planexs.backends.PredictionUnavailableError import PredictionUnavailableError
from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.LinearConstraint import LinearConstraint
from planexs.domain.constraints.NonlinearConstraint import NonlinearConstraint
from planexs.domain.Metric import Metric
from planexs.domain.Objective import Objective
from planexs.domain.OutcomeConstraint import OutcomeConstraint
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.SearchSpace import SearchSpace
from planexs.domain.StudySnapshot import StudySnapshot
from planexs.domain.Trial import Trial
from planexs.models.buildSurrogateModel import buildSurrogateModel
from planexs.models.encodeTrials import encodeTrials
from planexs.models.predictWithModel import predictWithModel
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.persistence.JsonStudyRepository import JsonStudyRepository
from planexs.persistence.ReproducibilityMetadata import ReproducibilityMetadata

if TYPE_CHECKING:
    # See OptimizationBackend.py's own comment: gated behind TYPE_CHECKING so
    # importing this facade does not force a BoTorch install by itself.
    from botorch.models.model import Model

logger = logging.getLogger(__name__)

#: The metric name used for the implicit single objective built by the
#: "common case" constructor path (`objective="minimize"`/`"maximize"`).
#: Matches section 5.8's usage example (`bo.tell(x, {"objective": 0.62})`)
#: exactly.
IMPLICIT_METRIC_NAME = "objective"

#: The `alpha` used when a `NonlinearConstraint` forces `ask()` onto the custom
#: acquisition layer although the caller left `alpha` at `None` (ADR-0006).
#: Pure exploitation: with no opinion expressed, go for the best predicted
#: point that satisfies the constraint.
DEFAULT_ALPHA_WHEN_FORCED = 0.0

#: Number of completed trials below which the custom acquisition layer cannot
#: fit a surrogate model and suggests space-filling points instead.
MIN_TRIALS_FOR_MODEL = 2

_MAX_SEED = 2**31 - 1


class BayesianOptimizer:
    """The main entry point. Two ways to call it:

    Common case: `BayesianOptimizer(parameters=[Real(...), Integer(...)])`.
        A single, unnamed metric is assumed; `objective="minimize"` or
        `"maximize"` picks its direction.
    Advanced case: pass an already-built `SearchSpace` (carrying its own
        constraints) and an already-built `Objective` (named metrics,
        optional weights, optional outcome_constraints, single or
        multi-objective). `constraints`/`outcome_constraints` must be left
        `None` in that case (a `ValueError` is raised otherwise: the
        `SearchSpace`/`Objective` already own their own constraints, so
        passing both is an ambiguous request, not a merge).

    No other combination of `parameters`/`objective` is accepted (e.g. a
    plain parameter list together with an already-built `Objective`): the
    two `@overload` signatures above the real `__init__` are the whole
    contract, and the implementation raises a `TypeError` naming exactly
    what was passed for anything outside it, rather than silently guessing
    what a mixed call was supposed to mean.

    `backend` defaults to `AxBackend(random_seed=random_seed)`; injectable
    for testing and for a future non-Ax backend. `acquisition_strategy`
    defaults to `AlphaAcquisitionStrategy()`, the layer behind `ask(alpha=...)`;
    injectable to change how candidates are chosen from the surrogate model
    (FR16).
    """

    @overload
    def __init__(
        self,
        parameters: Sequence[Parameter],
        objective: Literal["minimize", "maximize"] = "minimize",
        constraints: Sequence[Constraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        name: str = "study",
        random_seed: int | None = None,
        backend: OptimizationBackend | None = None,
        acquisition_strategy: AcquisitionStrategy | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        parameters: SearchSpace,
        objective: Objective,
        constraints: None = None,
        outcome_constraints: None = None,
        name: str = "study",
        random_seed: int | None = None,
        backend: OptimizationBackend | None = None,
        acquisition_strategy: AcquisitionStrategy | None = None,
    ) -> None: ...

    def __init__(
        self,
        parameters: Sequence[Parameter] | SearchSpace,
        objective: Objective | Literal["minimize", "maximize"] = "minimize",
        constraints: Sequence[Constraint] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
        name: str = "study",
        random_seed: int | None = None,
        backend: OptimizationBackend | None = None,
        acquisition_strategy: AcquisitionStrategy | None = None,
    ) -> None:
        """Creates the study and configures the backend.

        Args:
            parameters: A sequence of parameters (common case) or an already-built
                `SearchSpace` (advanced case).
            objective: `"minimize"` or `"maximize"` for the implicit single metric
                (common case), or an already-built `Objective` (advanced case).
            constraints: Parameter constraints (common case only).
            outcome_constraints: Constraints on the implicit metric (common case only).
            name: The study's name.
            random_seed: Seed recorded in the reproducibility metadata and given to the default
                backend. A concrete seed is generated and recorded when left `None`.
            backend: The optimization backend. Defaults to an `AxBackend`.
            acquisition_strategy: How candidates are chosen from the surrogate model when
                `ask()` is given an `alpha`. Defaults to an `AlphaAcquisitionStrategy`.

        Raises:
            ValueError: if constraints are passed together with a `SearchSpace`/`Objective`, or
                `objective` is a string other than `"minimize"`/`"maximize"`.
            TypeError: if `parameters` and `objective` are combined in an unsupported way.
        """
        self._search_space, self._objective = _resolveSearchSpaceAndObjective(
            parameters, objective, constraints, outcome_constraints
        )
        self._name = name
        self._trials: list[Trial] = []
        self._acquisition_strategy: AcquisitionStrategy = (
            acquisition_strategy
            if acquisition_strategy is not None
            else AlphaAcquisitionStrategy()
        )
        self._asks_since_tell = 0
        self._own_model_cache: tuple[int, Model] | None = None

        self._random_seed = (
            random_seed if random_seed is not None else secrets.randbelow(2**31 - 1)
        )
        if backend is not None:
            if random_seed is not None:
                logger.warning(
                    "Both `backend` and `random_seed` were passed to "
                    "BayesianOptimizer(): `random_seed` is recorded in "
                    "ReproducibilityMetadata, but it is not passed to the "
                    "already-constructed `backend`. If exact reproducibility "
                    "matters, construct `backend` with this same seed yourself."
                )
            self._backend = backend
        else:
            self._backend = AxBackend(random_seed=self._random_seed)

        self._backend.createExperiment(self._search_space, self._objective)

        # Captured once, at creation, not re-captured on every save(): FR12
        # asks for reproducibility metadata about the study, and `created_at`
        # in particular should mean "when this study began", not "when it was
        # last saved".
        self._reproducibility = ReproducibilityMetadata.captureCurrentEnvironment(
            self._random_seed
        )

    def tell(
        self,
        x: dict[str, float | int | str | bool],
        y: dict[str, float],
        y_std: dict[str, float] | None = None,
    ) -> Trial:
        """Manually inject an already-known point and its result(s). FR3.

        `x` maps parameter name to value. `y` maps metric name to value,
        matching however `Objective`'s metrics were named (or
        `IMPLICIT_METRIC_NAME` when the shorthand constructor was used).
        `y_std`, when known (e.g. from repeated measurements), is passed
        through as a fixed observation noise instead of being inferred by
        the surrogate model.

        If `x` exactly matches a parameterization most recently returned by
        `ask()` and not yet told, the backend completes that same pending
        trial rather than registering a new one; see
        `AxBackend.attachTrial`'s own docstring for the exact matching rule
        and its one known limitation.
        """
        pending_trial = Trial(parameters=dict(x), results=dict(y), result_std=y_std)
        trial_index = self._backend.attachTrial(pending_trial)
        recorded_trial = Trial(
            parameters=pending_trial.parameters,
            results=pending_trial.results,
            result_std=pending_trial.result_std,
            trial_index=trial_index,
        )
        self._trials.append(recorded_trial)
        self._asks_since_tell = 0
        return recorded_trial

    def ask(
        self,
        n_points: int = 1,
        alpha: float | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        """Ask for the next `n_points` parameterizations to evaluate.

        With `alpha=None` (the default), the backend's own no-manual-tuning
        strategy chooses (FR4). With `alpha` set, planexs' own acquisition layer
        chooses instead (FR5): it fits a surrogate model on the trial history
        and maximizes a blend of exploitation and exploration weighted by
        `alpha`. Batches (`n_points > 1`) are built one point at a time, each
        chosen point treated as already pending when picking the next (FR10).

        The effective strategy also depends on the search space: if it holds a
        `NonlinearConstraint`, which Ax cannot enforce, then `alpha=None` does
        NOT use Ax's strategy. `ask()` switches to the custom layer with
        `DEFAULT_ALPHA_WHEN_FORCED` (`0.0`, pure exploitation) and logs a
        warning saying so (ADR-0006). Passing `alpha` explicitly never warns.

        With fewer than `MIN_TRIALS_FOR_MODEL` completed trials there is no model
        to consult, so the custom layer suggests space-filling points (that
        still satisfy every constraint).

        The custom layer does not track points it has suggested but that were
        not yet told: asking twice with no `tell()` in between yields different
        points, but not points chosen to complement each other. Ask for a
        batch instead.

        Every suggested point satisfies the search space's parameter
        constraints. `OutcomeConstraint`s are not yet enforced by the custom
        layer: a warning is logged when there are some.

        Args:
            n_points: batch size requested at once. FR10.
            alpha: exploration/exploitation trade-off in `[0, 1]`. `0.0`
                favors the best predicted objective, `1.0` favors the
                least-known region, `0.5` balances both. FR5. If `None`,
                delegates to the backend's own no-manual-tuning default
                strategy (FR4), except as described above.

        Returns:
            `n_points` parameterizations, each a dict from parameter name to
            value, ready to pass to `tell()` once evaluated.

        Raises:
            ValueError: if `n_points < 1` or `alpha` is outside `[0, 1]`.
        """
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1; got {n_points!r}.")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha!r}.")

        forcing = self._constraintsForcingCustomLayer()
        if alpha is None and forcing:
            logger.warning(
                "The search space has %s, which Ax's default generation strategy cannot "
                "enforce: ask() is switching to planexs' own acquisition layer with "
                "alpha=%s (DEFAULT_ALPHA_WHEN_FORCED, pure exploitation). Pass alpha "
                "explicitly to choose the trade-off yourself and silence this warning.",
                " and ".join(forcing),
                DEFAULT_ALPHA_WHEN_FORCED,
            )
            alpha = DEFAULT_ALPHA_WHEN_FORCED

        if alpha is None:
            return self._backend.suggestDefault(n_points)
        return self._askWithAlpha(n_points, alpha)

    def predict(self, x: dict[str, float | int | str | bool]) -> PredictionResult:
        """The model's belief about `x`: predicted mean and standard error of
        the mean per metric, in the metrics' own units. FR7, FR8.

        Uses the backend's own model when it has one. Early in a study it does
        not (Ax's generation strategy only builds a model after its initial
        space-filling phase), so in that case planexs fits its own surrogate model
        to the trial history instead (ADR-0007); that is also the model
        `ask(alpha=...)` uses. Either way the numbers carry the same meaning.

        Args:
            x: A parameterization (parameter name to value). Parameters that
                `dependent_parameters` switch off may be left out.

        Returns:
            The prediction.

        Raises:
            PredictionUnavailableError: if no model can be fit yet, because
                fewer than `MIN_TRIALS_FOR_MODEL` trials have been told.
        """
        try:
            raw_prediction = self._backend.predict(dict(x))
        except PredictionUnavailableError:
            raw_prediction = self._predictWithOwnModel(x)
        mean = {name: mean_sem[0] for name, mean_sem in raw_prediction.items()}
        sem = {name: mean_sem[1] for name, mean_sem in raw_prediction.items()}
        return PredictionResult(mean=mean, sem=sem)

    def parameterImportance(self) -> dict[str, dict[str, float]]:
        """Refits/queries the surrogate model: has a cost, camelCase. FR6."""
        return self._backend.computeSensitivity()

    @property
    def n_trials(self) -> int:
        """`len()` over an already-held list: cheap, snake_case."""
        return len(self._trials)

    @property
    def paretoFront(self) -> list[Trial]:
        """Delegates to the backend's `getParetoFrontier()` (Ax's own
        `Client.get_pareto_frontier` under `AxBackend`, section 5.3), not a
        hand-rolled non-domination scan. Has a cost, camelCase. Collapses to
        a single-element list for a single-objective optimizer, via the
        backend's `getBestTrial()`.
        """
        if self._objective.is_multi_objective:
            return self._backend.getParetoFrontier()
        best_trial = self._backend.getBestTrial()
        return [best_trial] if best_trial is not None else []

    def save(self, path: str | Path) -> None:
        """FR11, FR12. See `JsonStudyRepository`, section 5.6."""
        snapshot = StudySnapshot(
            name=self._name,
            search_space=self._search_space,
            objective=self._objective,
            trials=list(self._trials),
            reproducibility=self._reproducibility,
            backend_state=self._backend.exportState(),
            backend_kind=_backendKind(self._backend),
        )
        JsonStudyRepository().save(snapshot, path)

    @classmethod
    def load(cls, path: str | Path) -> BayesianOptimizer:
        """FR11, FR12.

        Bypasses `__init__` (via `cls.__new__`) rather than reusing it: the
        normal constructor is for *creating* a new experiment inside the
        backend, while loading needs to *restore* a backend that already has
        its own experiment and trial history, plus planexs' own domain
        objects and reproducibility metadata exactly as they were saved, not
        rebuilt from scratch. This is documented here since it is the one
        place `BayesianOptimizer` is constructed without going through
        `__init__`.
        """
        snapshot = JsonStudyRepository().load(path)
        if snapshot.backend_kind != "ax":
            raise ValueError(
                f"Cannot load a study with backend_kind={snapshot.backend_kind!r}: "
                "Phase 1 only implements the 'ax' backend."
            )
        instance = cls.__new__(cls)
        instance._name = snapshot.name
        instance._search_space = snapshot.search_space
        instance._objective = snapshot.objective
        instance._trials = list(snapshot.trials)
        instance._reproducibility = snapshot.reproducibility
        instance._random_seed = snapshot.reproducibility.random_seed
        instance._acquisition_strategy = AlphaAcquisitionStrategy()
        instance._asks_since_tell = 0
        instance._own_model_cache = None
        instance._backend = AxBackend.importState(snapshot.backend_state)
        return instance

    def _predictWithOwnModel(
        self, x: dict[str, float | int | str | bool]
    ) -> dict[str, tuple[float, float]]:
        """`predict()` for when the backend has no model: planexs' own surrogate,
        refit only when the trial history has changed since the last call.
        """
        if self.n_trials < MIN_TRIALS_FOR_MODEL:
            raise PredictionUnavailableError(
                f"predict() needs at least {MIN_TRIALS_FOR_MODEL} completed trials to fit a "
                f"model; {self.n_trials} told so far."
            )
        encoder = SearchSpaceEncoder(self._search_space)
        if self._own_model_cache is None or self._own_model_cache[0] != self.n_trials:
            logger.info(
                "The backend has no predictive model yet: predicting with planexs' own "
                "surrogate model, fit on %d trial(s).",
                self.n_trials,
            )
            train_x, train_y, train_yvar = encodeTrials(encoder, self._objective, self._trials)
            self._own_model_cache = (
                self.n_trials,
                buildSurrogateModel(train_x, train_y, train_yvar),
            )
        return predictWithModel(
            self._own_model_cache[1],
            encoder.encodeParameters(x),
            [metric.name for metric in self._objective.metrics],
        )

    def _constraintsForcingCustomLayer(self) -> list[str]:
        """Names the kinds of constraint in the search space that only planexs'
        own acquisition layer can enforce (empty when Ax can enforce them all).
        """
        labels: list[str] = []
        for constraint in self._search_space.constraints:
            if not constraint.requires_custom_acquisition_layer:
                continue
            label = (
                "a NonlinearConstraint"
                if isinstance(constraint, NonlinearConstraint)
                else "an equality LinearConstraint"
                if isinstance(constraint, LinearConstraint)
                else f"a {type(constraint).__name__}"
            )
            if label not in labels:
                labels.append(label)
        return labels

    def _askWithAlpha(
        self, n_points: int, alpha: float
    ) -> list[dict[str, float | int | str | bool]]:
        """The custom acquisition layer's path through `ask()`. Independent of
        the backend: the surrogate is fit from planexs' own trial history
        (ADR-0007), not taken from Ax.
        """
        # Deterministic given the study's seed and history, and different for
        # successive asks between two tells.
        seed = (
            self._random_seed + 1_000_003 * self.n_trials + self._asks_since_tell
        ) % _MAX_SEED
        self._asks_since_tell += 1

        if self._objective.outcome_constraints:
            logger.warning(
                "The custom acquisition layer does not enforce OutcomeConstraints yet: "
                "the suggested points respect the parameter constraints but may not "
                "satisfy %d outcome constraint(s). Use alpha=None to have Ax enforce them.",
                len(self._objective.outcome_constraints),
            )
        if self.n_trials < MIN_TRIALS_FOR_MODEL:
            logger.info(
                "Only %d completed trial(s) (< %d): suggesting space-filling points instead "
                "of consulting a surrogate model.",
                self.n_trials,
                MIN_TRIALS_FOR_MODEL,
            )
            return self._acquisition_strategy.suggestSpaceFilling(
                self._search_space, n_points, seed=seed
            )

        encoder = SearchSpaceEncoder(self._search_space)
        train_x, train_y, train_yvar = encodeTrials(encoder, self._objective, self._trials)
        model = buildSurrogateModel(train_x, train_y, train_yvar)
        return self._acquisition_strategy.suggest(
            model, self._objective, self._search_space, alpha, n_points, seed=seed
        )

    @property
    def axClient(self) -> Any:
        """Escape hatch (FR16, ADR-0005): the live `ax.api.client.Client`
        instance backing this optimizer. Raises `TypeError` if `backend` is
        not an `AxBackend`. Typed `Any` here to avoid forcing an Ax import on
        every caller of this file; the real return type is
        `ax.api.client.Client`. Anything reached through this property is,
        by definition, outside what planexs validates or keeps in sync with
        its own domain objects: for example, a trial attached directly via
        `axClient.attach_trial(...)` will not appear as a planexs `Trial`
        until the caller also updates the planexs side, since planexs only
        learns about it through this same escape hatch, not automatically.
        """
        if not isinstance(self._backend, AxBackend):
            raise TypeError(
                "axClient is only available when the backend is an AxBackend; "
                f"this optimizer's backend is a {type(self._backend).__name__!r}."
            )
        return self._backend.client

    def fitModel(self) -> Model:
        """Escape hatch (FR16, ADR-0005): the BoTorch surrogate model Ax
        currently holds, so a caller can write and optimize their own
        acquisition function with plain BoTorch and feed the result back
        through `tell()`, without forking planexs to get an acquisition
        behavior the alpha dial does not cover.

        This is Ax's model, in Ax's own transformed input space, and it only
        exists once Ax has left its initial space-filling phase. It is not the
        model `ask(alpha=...)` uses: the alpha layer fits its own from planexs'
        trial history in the unit-cube encoding of `SearchSpaceEncoder`
        (ADR-0007).
        """
        return self._backend.fitModel()


def _resolveSearchSpaceAndObjective(
    parameters: Sequence[Parameter] | SearchSpace,
    objective: Objective | Literal["minimize", "maximize"],
    constraints: Sequence[Constraint] | None,
    outcome_constraints: Sequence[OutcomeConstraint] | None,
) -> tuple[SearchSpace, Objective]:
    """Turns the constructor's two call shapes into a `SearchSpace` and an `Objective`.

    Args:
        parameters: A parameter sequence or a `SearchSpace`.
        objective: `"minimize"`/`"maximize"` or an `Objective`.
        constraints: Parameter constraints, accepted only with a parameter sequence.
        outcome_constraints: Metric constraints, accepted only with a parameter sequence.

    Returns:
        The `(search_space, objective)` pair.

    Raises:
        ValueError: if constraints accompany a `SearchSpace`/`Objective`, or the objective
            string is not `"minimize"`/`"maximize"`.
        TypeError: for any other combination of `parameters` and `objective`.
    """
    if isinstance(parameters, SearchSpace) and isinstance(objective, Objective):
        if constraints is not None or outcome_constraints is not None:
            raise ValueError(
                "When `parameters` is a SearchSpace and `objective` is an "
                "Objective (the advanced case), `constraints` and "
                "`outcome_constraints` must be left None: the SearchSpace "
                "and Objective already own their own constraints, so "
                "passing both is an ambiguous request, not a merge."
            )
        return parameters, objective

    if not isinstance(parameters, SearchSpace) and not isinstance(objective, Objective):
        if objective not in ("minimize", "maximize"):
            raise ValueError(
                "objective must be 'minimize' or 'maximize' when "
                f"`parameters` is a plain parameter sequence; got {objective!r}."
            )
        search_space = SearchSpace(
            parameters=list(parameters),
            constraints=list(constraints) if constraints is not None else [],
        )
        implicit_objective = Objective(
            metrics=[Metric(name=IMPLICIT_METRIC_NAME, minimize=(objective == "minimize"))],
            outcome_constraints=(
                list(outcome_constraints) if outcome_constraints is not None else []
            ),
        )
        return search_space, implicit_objective

    is_search_space = isinstance(parameters, SearchSpace)
    is_objective = isinstance(objective, Objective)
    raise TypeError(
        "BayesianOptimizer() takes either (parameters=Sequence[Parameter], "
        "objective='minimize'|'maximize') for the common case, or "
        "(parameters=SearchSpace, objective=Objective) for the advanced case "
        "(section 5.7); got parameters as "
        f"{'a SearchSpace' if is_search_space else 'a parameter sequence'} "
        f"together with objective as "
        f"{'an Objective' if is_objective else type(objective).__name__!r}."
    )


def _backendKind(backend: OptimizationBackend) -> str:
    """Names the kind of a backend for persistence.

    Args:
        backend: The backend to describe.

    Returns:
        `"ax"` for an `AxBackend`.

    Raises:
        TypeError: for any other backend, which cannot be saved yet.
    """
    if isinstance(backend, AxBackend):
        return "ax"
    raise TypeError(
        f"save() does not know how to persist backend kind "
        f"{type(backend).__name__!r}: only AxBackend is supported in Phase 1."
    )
