"""Draws space-filling encoded points that satisfy the search space's constraints."""

from __future__ import annotations

import torch
from botorch.optim.initializers import sample_q_batches_from_polytope
from torch import Tensor

from planexs.acquisition.EncodedConstraints import EncodedConstraints
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder

#: Defaults of the hit-and-run sampler that draws points inside a linear polytope (see
#: `sampleFeasibleEncoded`). They are the only two knobs of that Markov chain. BoTorch's own
#: defaults (burn-in 10000, thinning 32) are much more expensive, and these lower values were
#: chosen from two measurements rather than from theory:
#:
#: - `n_burnin`: steps discarded before the first kept point (the chain starts at an arbitrary
#:   interior point and must forget it). On a simplex, whose exact law is known, 200 and 10000
#:   gave indistinguishable means and variances in 3 and 10 dimensions, so the cheaper value
#:   is used.
#: - `n_thinning`: steps between two kept points (consecutive states are correlated). This is
#:   the expensive knob: on an equality-constrained mixed problem, 32 instead of 10 made
#:   suggesting 3 points 2.7 times slower. The price of 10 is more correlated samples in high
#:   dimension (lag-1 autocorrelation 0.76 in 10 dimensions, against 0.43 with 32). The
#:   samples only seed the optimizer's restarts and normalize the acquisition terms, so this
#:   was judged acceptable; raise it for a high-dimensional polytope.
DEFAULT_POLYTOPE_BURN_IN = 200
DEFAULT_POLYTOPE_THINNING = 10

#: How many rounds of oversampled draws to try before giving up.
_MAX_ROUNDS = 25
_OVERSAMPLE = 8


def sampleFeasibleEncoded(
    encoder: SearchSpaceEncoder,
    constraints: EncodedConstraints,
    n: int,
    seed: int | None = None,
    snap: bool = True,
    tolerance: float = 1e-6,
    n_burnin: int = DEFAULT_POLYTOPE_BURN_IN,
    n_thinning: int = DEFAULT_POLYTOPE_THINNING,
) -> Tensor:
    """Draws `n` encoded points that satisfy every constraint.

    Without constraints these are plain scrambled-Sobol points. With linear
    constraints the draws come from BoTorch's polytope sampler (so an equality
    such as a mixture summing to one is met exactly, which rejection sampling
    could never do). Nonlinear constraints are then met by rejection, oversampling
    until `n` points survive.

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
        tolerance: Constraint violation accepted (see `EncodedConstraints`).
        n_burnin: Burn-in steps of the polytope sampler, used only when the space has
            linear constraints. See `DEFAULT_POLYTOPE_BURN_IN`.
        n_thinning: Thinning of the polytope sampler, used only when the space has
            linear constraints. See `DEFAULT_POLYTOPE_THINNING`.

    Returns:
        A tensor of shape `(n, d)`. If fewer than `n` distinct feasible points
        were found, the ones found are repeated to fill it.

    Raises:
        RuntimeError: if no feasible point could be found at all (the feasible
            region may be empty, or too small to hit by sampling).
    """
    if not constraints.has_constraints:
        return encoder.sampleEncoded(n, seed=seed, snap=snap)

    kept: list[Tensor] = []
    n_kept = 0
    n_drawn = 0
    for round_index in range(_MAX_ROUNDS):
        n_draw = max(_OVERSAMPLE * n, 64) * (1 + round_index)
        round_seed = None if seed is None else seed + round_index
        if constraints.inequality is not None or constraints.equality is not None:
            draws = sample_q_batches_from_polytope(
                n=n_draw,
                q=1,
                bounds=encoder.unit_bounds,
                n_burnin=n_burnin,
                n_thinning=n_thinning,
                seed=round_seed,
                inequality_constraints=constraints.inequality,
                equality_constraints=constraints.equality,
            ).squeeze(1)
            if snap:
                draws = encoder.snapColumns(draws)
        else:
            draws = encoder.sampleEncoded(n_draw, seed=round_seed, snap=snap)
        n_drawn += n_draw
        feasible = draws[constraints.isFeasible(draws, tolerance)]
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
