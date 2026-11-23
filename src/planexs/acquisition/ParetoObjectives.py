"""The objectives of a Pareto problem, as they are placed on the outputs of a model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective


@dataclass(frozen=True)
class ParetoObjectives:
    """The outcomes that an optimize goal optimizes as separate objectives, with the output of
    the model that holds each one and the direction it is optimized in (specification 4,
    "Acquisition").

    An acquisition for a Pareto problem works in a space where every objective is maximized, so
    that a larger value is always a better one and a point dominates another when it is larger
    in every objective: a minimized outcome is negated on the way in. This class says which
    outputs of the model are objectives, in which order they come out, and which are negated,
    and it builds the objective that does it. The order is that of the outputs of the model and
    not that of the goal, so that the order in which a configuration happens to list the
    outcomes never matters.

    Attributes:
        names: The names of the objectives, in the order of the model's outputs.
        indices: For each objective, the output of the model that holds it.
        signs: For each objective, 1.0 if it is maximized and -1.0 if it is minimized.
        num_outputs: The number of outputs of the model, the objectives and the others (an
            output that the goal only constrains is not an objective).
    """

    names: tuple[str, ...]
    indices: tuple[int, ...]
    signs: tuple[float, ...]
    num_outputs: int

    def objective(
        self, *, dtype: torch.dtype, device: torch.device
    ) -> WeightedMCMultiOutputObjective:
        """The BoTorch objective that maps the outputs of the model to the objectives, all
        maximized.

        Args:
            dtype: The precision of the weights, that of the model.
            device: The device of the weights, that of the model.

        Returns:
            An objective that takes samples (or means) of shape `... x m`, one column for each
            output of the model, and gives `... x k`, one column for each objective, negated
            where the outcome is minimized.
        """
        return WeightedMCMultiOutputObjective(
            weights=torch.tensor(self.signs, dtype=dtype, device=device),
            outcomes=list(self.indices),
            num_outcomes=self.num_outputs,
        )
