"""Turns a trial history into the tensors a surrogate model trains on."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import Tensor

from planexs.domain.Objective import Objective
from planexs.domain.Trial import Trial
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder

logger = logging.getLogger(__name__)


def encodeTrials(
    encoder: SearchSpaceEncoder, objective: Objective, trials: Sequence[Trial]
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Encodes `trials` for `buildSurrogateModel`.

    Outcomes are in the trials' own (raw) units, one column per metric of
    `objective`, in `objective.metrics` order; the minimize/maximize direction
    is applied later, by the acquisition function.

    Known observation noise is only used when every trial gives a standard
    deviation for every metric: a GP cannot mix fixed and inferred noise, so a
    partial set is ignored (with a warning) rather than half-applied.

    Args:
        encoder: The search space encoder.
        objective: Says which metrics to read from each trial.
        trials: The trial history.

    Returns:
        `(train_x, train_y, train_yvar)` with shapes `(n, d)`, `(n, m)` and
        `(n, m)` (or `None` when noise is inferred).

    Raises:
        ValueError: if a trial lacks a value for one of the objective's metrics
            or a searched parameter.
    """
    metric_names = [metric.name for metric in objective.metrics]
    rows: list[Tensor] = []
    outcomes: list[list[float]] = []
    variances: list[list[float] | None] = []
    for position, trial in enumerate(trials):
        missing = [name for name in metric_names if name not in trial.results]
        if missing:
            raise ValueError(
                f"Trial #{position} has no result for metric(s) {missing!r}; the "
                f"objective needs {metric_names!r}."
            )
        rows.append(encoder.encodeParameters(trial.parameters))
        outcomes.append([trial.results[name] for name in metric_names])
        std = trial.result_std
        if std is not None and all(name in std for name in metric_names):
            variances.append([std[name] ** 2 for name in metric_names])
        else:
            variances.append(None)

    train_x = torch.stack(rows)
    train_y = torch.tensor(outcomes, dtype=torch.double)
    known = [entry for entry in variances if entry is not None]
    if not known:
        return train_x, train_y, None
    if len(known) < len(variances):
        logger.warning(
            "Only %d of %d trials give a result_std for every metric: ignoring result_std "
            "and inferring observation noise instead (a GP cannot mix fixed and inferred "
            "noise).",
            len(known),
            len(variances),
        )
        return train_x, train_y, None
    return train_x, train_y, torch.tensor(known, dtype=torch.double)
