"""The `OptimizationBackend` implementation backed by `ax.api.client.Client`."""

from __future__ import annotations

import json
import logging
import secrets
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from ax.api.client import Client
from ax.exceptions.core import UnsupportedError
from botorch.models.model import Model

from planexs.backends.ax.toAxOptimizationConfig import toAxOptimizationConfig
from planexs.backends.ax.toAxSearchSpace import toAxSearchSpace
from planexs.backends.OptimizationBackend import OptimizationBackend
from planexs.backends.PredictionUnavailableError import PredictionUnavailableError
from planexs.domain.constraints.LinearConstraint import LinearConstraint
from planexs.domain.LegacyObjective import LegacyObjective
from planexs.domain.LegacyTrial import LegacyTrial
from planexs.domain.SearchSpace import SearchSpace

logger = logging.getLogger(__name__)

_AxValue = float | int | str | bool
_AxRawDataValue = float | tuple[float, float]


class AxBackend(OptimizationBackend):
    """The `OptimizationBackend` implementation backed by
    `ax.api.client.Client`.

    Implements every `OptimizationBackend` method by delegating to a private
    `ax.api.client.Client` instance; exact accessor names confirmed against
    https://ax.readthedocs.io/en/stable/api.html for this draft, re-check
    against whatever Ax version `pyproject.toml` ends up pinning.

    `random_seed`, `method`, `initialization_budget`, `torch_device` are
    passed straight through to `Client.configure_generation_strategy`
    (section 4.4); not reinvented, just exposed at construction time.
    """

    def __init__(
        self,
        random_seed: int | None = None,
        method: Literal["quality", "fast", "random_search"] = "fast",
        initialization_budget: int | None = None,
        torch_device: str | None = None,
    ) -> None:
        """Creates the Ax client; `createExperiment` configures the experiment later.

        Args:
            random_seed: Seed of the study. A concrete seed is always resolved and
                recorded, even when left `None`, so that reproducibility metadata never
                has to say "unknown".
            method: Ax's generation-strategy preset: `"fast"`, `"quality"` or
                `"random_search"`.
            initialization_budget: Number of initial space-filling trials before Ax fits
                a model. `None` lets Ax choose.
            torch_device: The torch device Ax's models run on. `None` lets Ax choose.
        """
        # A concrete seed is always resolved and recorded, even when the
        # caller leaves `random_seed=None`, so ReproducibilityMetadata (FR12)
        # never has to record "unknown". See BayesianOptimizer.__init__.
        self._random_seed: int = (
            random_seed if random_seed is not None else secrets.randbelow(2**31 - 1)
        )
        self._method: Literal["quality", "fast", "random_search"] = method
        self._initialization_budget = initialization_budget
        self._torch_device = torch_device

        self._client: Client = Client(random_seed=self._random_seed)

        # Maps a canonical (sorted-items-tuple) parameterization to the Ax
        # trial_index that `suggestDefault` created it under, so that
        # `attachTrial` can `complete_trial` the trial Ax already has RUNNING
        # for a point returned by `ask()`, instead of creating a second,
        # orphaned one via `attach_trial`. Not part of section 5.3's ABC:
        # a purely internal bookkeeping detail of this backend. See
        # `attachTrial`'s own docstring for the full rationale.
        self._pending_trials: dict[tuple[tuple[str, _AxValue], ...], int] = {}

        self._search_space: SearchSpace | None = None
        self._objective: LegacyObjective | None = None
        self._metric_names: list[str] = []

    def createExperiment(self, search_space: SearchSpace, objective: LegacyObjective) -> None:
        """Configures Ax's experiment, optimization goal and generation strategy.

        Only the parameter constraints Ax can enforce are given to it: linear
        inequalities. An equality or a `NonlinearConstraint` is enforced by planexs' own
        acquisition layer instead (ADR-0006, ADR-0008).

        Args:
            search_space: The parameters and parameter constraints.
            objective: The metrics, weights and outcome constraints.
        """
        self._search_space = search_space
        self._objective = objective
        self._metric_names = [metric.name for metric in objective.metrics]

        ax_parameters = toAxSearchSpace(search_space)
        linear_constraint_strings = [
            constraint.toAxParameterConstraintString()
            for constraint in search_space.constraints
            # Ax takes inequalities only: an equality (and any NonlinearConstraint) is
            # enforced by planexs' own acquisition layer instead (ADR-0006, ADR-0008).
            if isinstance(constraint, LinearConstraint)
            and not constraint.requires_custom_acquisition_layer
        ]
        self._client.configure_experiment(
            parameters=ax_parameters,
            parameter_constraints=linear_constraint_strings or None,
        )

        objective_string, outcome_constraint_strings = toAxOptimizationConfig(objective)
        self._client.configure_optimization(
            objective=objective_string,
            outcome_constraints=outcome_constraint_strings or None,
        )

        self._client.configure_generation_strategy(
            method=self._method,
            initialization_budget=self._initialization_budget,
            initialization_random_seed=self._random_seed,
            torch_device=self._torch_device,
        )

    def attachTrial(self, trial: LegacyTrial) -> int:
        """Registers `trial`, returning its Ax trial index.

        If `trial.parameters` exactly matches a parameterization most
        recently returned by `suggestDefault` (and not yet told), the
        already-RUNNING Ax trial for it is completed in place. Otherwise (the
        FR3 case: prior data, or a point evaluated outside planexs entirely)
        a brand-new trial is attached and immediately completed.

        Matching is by exact equality of the parameterization dict. If the
        caller mutates a point between `ask()` and `tell()`, a new trial is
        attached instead of completing the pending one, leaving the original
        Ax trial stuck at RUNNING; this is a known, accepted limitation of
        keying by value rather than by an explicit trial handle (the
        `OptimizationBackend.attachTrial(trial: LegacyTrial)` signature has no
        trial-handle parameter to key on instead).
        """
        raw_data = _toAxRawData(trial.results, trial.result_std)
        key = _canonicalParameters(trial.parameters)
        pending_index = self._pending_trials.pop(key, None)
        trial_index = (
            pending_index
            if pending_index is not None
            else self._client.attach_trial(parameters=trial.parameters)
        )
        self._client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        return trial_index

    def suggestDefault(self, n_points: int) -> list[dict[str, _AxValue]]:
        """Asks Ax's own generation strategy for the next trials.

        Each suggestion is remembered as pending, so that a later `attachTrial` of the
        same point completes Ax's running trial rather than creating a second one.

        Args:
            n_points: Number of trials requested.

        Returns:
            The suggested parameterizations.
        """
        next_trials = self._client.get_next_trials(max_trials=n_points)
        suggestions: list[dict[str, _AxValue]] = []
        for trial_index, parameterization in next_trials.items():
            parameters: dict[str, _AxValue] = dict(parameterization)
            self._pending_trials[_canonicalParameters(parameters)] = trial_index
            suggestions.append(parameters)
        return suggestions

    def fitModel(self) -> Model:
        """Best-effort extraction of the BoTorch `Model` currently fitted
        inside Ax's `GenerationStrategy`.

        This walks Ax's internal object graph (in Ax 1.x: `GenerationStrategy.adapter`
        -> `.generator` -> `.surrogate` -> `.model`), which is *not* part of the stable
        `ax.api` surface this file otherwise restricts itself to. It is deliberately
        isolated to this one method so that, per ADR-0005's own accepted trade-off
        ("anything reached through this escape hatch is outside what planexs
        validates"), a future Ax version needing a different attribute chain only
        requires editing this one method.

        Returns:
            The fitted BoTorch model, in Ax's own transformed input space.

        Raises:
            RuntimeError: if no BoTorch model can be reached. While Ax is still in its
                initial space-filling phase its adapter has no surrogate; the message
                names the adapter found so that case can be told apart from an Ax
                version that moved the attributes.
        """
        adapter: Any = None
        try:
            generation_strategy = self._client._generation_strategy
            adapter = getattr(generation_strategy, "adapter", None)
            generator = getattr(adapter, "generator", None)
            surrogate = getattr(generator, "surrogate", None)
            model = getattr(surrogate, "model", None)
            if not isinstance(model, Model):
                raise TypeError(
                    f"found {type(model).__name__!r} where a botorch Model belongs"
                )
            return model
        except Exception as error:
            raise RuntimeError(
                "Could not extract a fitted BoTorch model from Ax's current "
                f"GenerationStrategy (its adapter is a {type(adapter).__name__!r}): {error}. "
                "This usually means Ax is still in its initial space-filling phase and has "
                "not fit a model yet (see `initialization_budget`). If a model should exist, "
                "the pinned Ax version may have restructured the internal "
                "GenerationStrategy/Adapter/Generator/Surrogate object graph, which "
                "`fitModel()` walks outside of Ax's own `ax.api` stability guarantees "
                "(ADR-0005)."
            ) from error

    def predict(self, x: dict[str, _AxValue]) -> dict[str, tuple[float, float]]:
        """Delegates to `Client.predict`. Until Ax's generation strategy has
        moved past its initial space-filling phase it holds no model and raises
        `UnsupportedError`; that is reported as `PredictionUnavailableError` so
        `BayesianOptimizer.predict` can use planexs' own surrogate instead.
        """
        try:
            return self._client.predict([x])[0]
        except UnsupportedError as error:
            raise PredictionUnavailableError(
                "Ax has no predictive model yet (its generation strategy is still in its "
                f"initial space-filling phase): {error}"
            ) from error

    def computeSensitivity(self) -> dict[str, dict[str, float]]:
        """Delegates to Ax's own `Client.compute_analyses` with a
        `SensitivityAnalysisPlot` per metric.

        Ax's `Analysis` module is explicitly documented as not part of the
        stable `ax.api` surface ("its methods are subject to change
        incompatibly between minor versions"). This method is therefore
        defensive: a metric whose sensitivity cannot be extracted (too few
        trials, an unrecognized `AnalysisCard` shape for the pinned Ax
        version, etc.) logs a warning and contributes an empty `{}` rather
        than raising, so one metric's failure does not take down FR6 for
        every other metric.
        """
        metric_names = self._metric_names or self._resolveMetricNames()
        try:
            from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
        except ImportError as error:
            logger.warning(
                "computeSensitivity() could not import Ax's "
                "SensitivityAnalysisPlot (its module path may have moved for "
                "the pinned Ax version): %s",
                error,
            )
            return {metric_name: {} for metric_name in metric_names}

        result: dict[str, dict[str, float]] = {}
        for metric_name in metric_names:
            try:
                cards = self._client.compute_analyses(
                    analyses=[SensitivityAnalysisPlot(metric_name=metric_name)],
                    display=False,
                )
                result[metric_name] = _extractParameterImportance(cards)
            except Exception as error:
                logger.warning(
                    "computeSensitivity() could not extract sensitivity for metric %r: %s",
                    metric_name,
                    error,
                )
                result[metric_name] = {}
        return result

    def getParetoFrontier(self) -> list[LegacyTrial]:
        """The Pareto frontier according to Ax, using model predictions.

        Returns:
            The frontier as trials, or an empty list (with a logged warning) if Ax cannot
            compute it, for instance while there are too few trials.
        """
        try:
            entries = self._client.get_pareto_frontier(use_model_predictions=True)
        except Exception as error:
            logger.warning("getParetoFrontier() failed: %s", error)
            return []
        return [
            _axBestEntryToTrial(parameters, means, trial_index)
            for parameters, means, trial_index, _arm_name in entries
        ]

    def getBestTrial(self) -> LegacyTrial | None:
        """The best parameterization according to Ax, using model predictions.

        Returns:
            The best trial, or `None` (with a logged warning) if Ax cannot name one yet.
        """
        try:
            parameters, means, trial_index, _arm_name = self._client.get_best_parameterization(
                use_model_predictions=True
            )
        except Exception as error:
            logger.warning("getBestTrial() failed (no completed trials yet?): %s", error)
            return None
        return _axBestEntryToTrial(parameters, means, trial_index)

    def exportState(self) -> dict[str, Any]:
        """Serializes the whole Ax client.

        Ax only offers a file-based save, so this saves to a temporary file and reads it back.

        Returns:
            The JSON-compatible state `Client.save_to_json_file` writes.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "ax_client_snapshot.json"
            self._client.save_to_json_file(str(tmp_path))
            return dict(json.loads(tmp_path.read_text(encoding="utf-8")))

    @classmethod
    def importState(cls, state: dict[str, Any]) -> AxBackend:
        """Rebuilds a backend from state produced by `exportState`.

        Args:
            state: The state `exportState` returned.

        Returns:
            A backend whose Ax client holds the saved experiment and trials. Its metric
            names are read back from the Ax experiment.
        """
        backend = cls()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "ax_client_snapshot.json"
            tmp_path.write_text(json.dumps(state), encoding="utf-8")
            backend._client = Client.load_from_json_file(str(tmp_path))
        backend._metric_names = backend._resolveMetricNames()
        return backend

    def _resolveMetricNames(self) -> list[str]:
        """Falls back to Ax's own `Experiment.optimization_config` when
        `_metric_names` was not populated by `createExperiment` (i.e. after
        `importState`, where planexs' own `LegacyObjective` is supplied separately
        by the caller and this backend only has the raw Ax state to go on).
        """
        try:
            optimization_config = self._client._experiment.optimization_config
            if optimization_config is None:
                raise ValueError("The Ax experiment has no optimization config yet.")
            # The objective's own metrics, not `optimization_config.metrics` (which
            # Ax 1.x does not have): this matches what `createExperiment` records
            # from planexs' `LegacyObjective`, for single, multi and weighted objectives.
            return list(optimization_config.objective.metric_names)
        except Exception as error:
            logger.warning(
                "Could not resolve metric names from the Ax experiment; "
                "computeSensitivity() will return an empty result: %s",
                error,
            )
            return []

    @property
    def client(self) -> Any:
        """The live `ax.api.client.Client`. Returned as `Any` here to avoid
        leaking an Ax import into this file's own public signature; exposed
        to end users, typed as Ax's own `Client`, via
        `BayesianOptimizer.axClient` (section 5.7, ADR-0005).
        """
        return self._client


def _canonicalParameters(
    parameters: Mapping[str, _AxValue],
) -> tuple[tuple[str, _AxValue], ...]:
    """Turns a parameterization into a hashable, order-independent key.

    Args:
        parameters: Parameter name to value.

    Returns:
        The `(name, value)` pairs sorted by name.
    """
    return tuple(sorted(parameters.items()))


def _toAxRawData(
    results: dict[str, float], result_std: dict[str, float] | None
) -> dict[str, _AxRawDataValue]:
    """Builds the `raw_data` Ax expects for a completed trial.

    Args:
        results: Metric name to observed value.
        result_std: Metric name to the known standard deviation of that observation, if any.

    Returns:
        Metric name to a mean, or to a `(mean, sem)` pair where a deviation is known.
    """
    raw_data: dict[str, _AxRawDataValue] = {}
    for name, mean in results.items():
        if result_std is not None and name in result_std:
            raw_data[name] = (mean, result_std[name])
        else:
            raw_data[name] = mean
    return raw_data


def _meanAndSem(value: float | tuple[float, float]) -> tuple[float, float | None]:
    """Splits an Ax metric value into its mean and its standard error.

    Args:
        value: A plain mean, or a `(mean, sem)` pair.

    Returns:
        `(mean, sem)`, with `sem` being `None` when only a mean was given.
    """
    if isinstance(value, tuple):
        return value[0], value[1]
    return value, None


def _axBestEntryToTrial(
    parameters: Mapping[str, _AxValue],
    means: Mapping[str, float | tuple[float, float]],
    trial_index: int,
) -> LegacyTrial:
    """Converts one entry of Ax's best-parameterization or Pareto output into a `LegacyTrial`.

    Args:
        parameters: The parameterization.
        means: Metric name to a mean, or a `(mean, sem)` pair.
        trial_index: The Ax trial index.

    Returns:
        The trial, with `result_std` set only where Ax gave a standard error.
    """
    results: dict[str, float] = {}
    result_std: dict[str, float] = {}
    for metric_name, value in means.items():
        mean, sem = _meanAndSem(value)
        results[metric_name] = mean
        if sem is not None:
            result_std[metric_name] = sem
    return LegacyTrial(
        parameters=dict(parameters),
        results=results,
        result_std=result_std or None,
        trial_index=trial_index,
    )


def _extractParameterImportance(cards: list[Any]) -> dict[str, float]:
    """Best-effort extraction of `{parameter_name: importance}` from the
    `AnalysisCardBase` list `Client.compute_analyses` returns for a
    `SensitivityAnalysisPlot`.

    `AnalysisCard`'s own shape is explicitly not part of Ax's stable API
    (see `computeSensitivity`'s docstring), so this tries a couple of
    plausible attribute/column names rather than committing to one; it
    returns `{}` if none match, which `computeSensitivity` logs as a
    per-metric warning rather than a hard failure.
    """
    importance: dict[str, float] = {}
    for card in cards:
        data_frame = getattr(card, "df", None)
        if data_frame is None:
            continue
        columns = set(data_frame.columns)
        parameter_column = next(
            (c for c in ("parameter_name", "parameter") if c in columns), None
        )
        value_column = next(
            (c for c in ("sensitivity", "importance", "value") if c in columns), None
        )
        if parameter_column is None or value_column is None:
            continue
        for _, row in data_frame.iterrows():
            importance[str(row[parameter_column])] = float(row[value_column])
    return importance
