"""The observed points that can be on the Pareto front, chosen reproducibly."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.multi_objective.utils import prune_inferior_points_multi_objective
from botorch.models.model import Model
from torch import Tensor


def prunedParetoBaseline(
    model: Model,
    baseline: Tensor,
    *,
    ref_point: Tensor,
    objective: MCMultiOutputObjective,
    constraints: list[Callable[[Tensor], Tensor]] | None,
    n_samples: int,
    seed: int,
) -> Tensor:
    """Drops the observed points that are very unlikely to be on the Pareto front, better than
    the reference point and feasible, so that they are left out of the baseline of a noisy
    hypervolume acquisition (specification 5.8).

    The pruning samples the joint posterior of the observed points and keeps the points that
    are on the front in some sample, as BoTorch's
    `prune_inferior_points_multi_objective` does. That function draws its samples from the
    global random number generator and takes no sampler, so two calls from one seed could keep
    different points and a run would not be reproducible. The call is made here inside a fork
    of the generator that is seeded with `seed`, so it is deterministic for the seed, and the
    global generator is the one it was before the call: a component does not read or move
    state that belongs to the program (specification 5.8).

    Args:
        model: The fitted native model.
        baseline: The observed points, shape `(r, d)`.
        ref_point: The reference point, shape `(k,)`, with all objectives maximized.
        objective: The objective from the outputs of the model to the objectives.
        constraints: The feasibility constraints of the goal, as BoTorch callables, or `None`.
        n_samples: The number of posterior samples that decide the pruning.
        seed: The seed of the samples, drawn from the generator of the build.

    Returns:
        The points that are kept, shape `(r', d)` with `r'` at most `r`, in their original
        order.
    """
    devices = [baseline.device] if baseline.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return prune_inferior_points_multi_objective(
            model=model,
            X=baseline,
            ref_point=ref_point,
            objective=objective,
            constraints=constraints,
            num_samples=n_samples,
        )
