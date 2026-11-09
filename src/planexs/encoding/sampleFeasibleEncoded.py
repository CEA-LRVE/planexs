"""Draws space-filling encoded points that satisfy the search space's constraints."""

from __future__ import annotations

import torch
from botorch.optim.initializers import sample_q_batches_from_polytope
from torch import Tensor

from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.FeasibleSamplingConfig import FeasibleSamplingConfig
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder


def sampleFeasibleEncoded(
    encoder: SearchSpaceEncoder,
    constraints: EncodedConstraints,
    n: int,
    seed: int | None = None,
    snap: bool = True,
    config: FeasibleSamplingConfig | None = None,
) -> Tensor:
    """Draws `n` encoded points that satisfy every constraint.

    Without constraints these are plain scrambled-Sobol points. With linear
    constraints the draws come from BoTorch's polytope sampler (so an equality
    such as a mixture summing to one is met exactly, which rejection sampling
    could never do). Nonlinear constraints are then met by rejection, oversampling
    until `n` points survive.

    The draws go in rounds. The first draws the larger of `oversampling_factor * n` and
    `min_draw` candidates, and round `k` (counted from 1) draws `k` times as many, until `n`
    feasible points exist or `max_rounds` rounds have been drawn. Every number that these
    rounds, the polytope chain and the feasibility check use is a field of
    `FeasibleSamplingConfig`: none is written in this file (specification 3.4).

    Used for the alpha layer's reference set, its cold-start (no model yet)
    suggestions, and as the source of feasible starting points for constrained
    optimization.

    Args:
        encoder: The search space encoder.
        constraints: The encoded constraints.
        n: Number of points wanted.
        seed: Seed of the draws.
        snap: Whether to round points onto valid values. Leave `False` for
            continuous relaxations.
        config: The sampling settings: the rounds, the polytope chain and the tolerance
            with which a candidate counts as feasible. `None` means the settings that the
            constraints were built with (`constraints.config`).

    Returns:
        A tensor of shape `(n, d)`. If fewer than `n` distinct feasible points
        were found, the ones found are repeated to fill it.

    Raises:
        RuntimeError: if no feasible point could be found at all (the feasible
            region may be empty, or too small to hit by sampling).
    """
    if not constraints.has_constraints:
        return encoder.sampleEncoded(n, seed=seed, snap=snap)

    settings = constraints.config if config is None else config
    kept: list[Tensor] = []
    n_kept = 0
    n_drawn = 0
    for round_index in range(settings.max_rounds):
        first_round = max(settings.oversampling_factor * n, settings.min_draw)
        n_draw = first_round * (1 + round_index)
        round_seed = None if seed is None else seed + round_index
        if constraints.inequality is not None or constraints.equality is not None:
            draws = sample_q_batches_from_polytope(
                n=n_draw,
                q=1,
                bounds=encoder.unit_bounds,
                n_burnin=settings.polytope_burn_in,
                n_thinning=settings.polytope_thinning,
                seed=round_seed,
                inequality_constraints=constraints.inequality,
                equality_constraints=constraints.equality,
            ).squeeze(1)
            if snap:
                draws = encoder.snapColumns(draws)
        else:
            draws = encoder.sampleEncoded(n_draw, seed=round_seed, snap=snap)
        n_drawn += n_draw
        feasible = draws[constraints.isFeasible(draws, settings.feasibility_tolerance)]
        if feasible.shape[0] > 0:
            kept.append(feasible)
            n_kept += feasible.shape[0]
        if n_kept >= n:
            break

    if n_kept == 0:
        raise RuntimeError(
            f"Found no point satisfying the search space's constraints among {n_drawn} "
            "space-filling candidates: the feasible region may be empty or very small."
        )
    pool = torch.cat(kept, dim=0)
    if pool.shape[0] >= n:
        return pool[:n]
    repeats = -(-n // pool.shape[0])
    return pool.repeat(repeats, 1)[:n]
