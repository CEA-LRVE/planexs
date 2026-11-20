"""A design of uniform random points."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.DesignContext import DesignContext
from planexs.designs.RandomDesignConfig import RandomDesignConfig
from planexs.designs.registerDesign import registerDesign
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.sampleFeasibleEncoded import sampleFeasibleEncoded
from planexs.encoding.UniformDraw import UniformDraw


@registerDesign(
    "random_v1",
    config=RandomDesignConfig,
    description="Uniform random points in the encoded space, honouring the constraints.",
)
class RandomDesign(AbstractDesign[RandomDesignConfig]):
    """Independent uniform random points, in the unit cube of the encoding.

    Every point is drawn on its own, so unlike a Sobol sequence nothing keeps a small design
    from leaving a gap or putting two points close together. It is the baseline that a
    space-filling design is measured against. The points are drawn in the encoded space, so a
    log-scaled range, a grid, an ordered choice and a categorical are all covered as the
    encoder lays them out (each level of a categorical is equally likely), and each point is
    snapped onto valid values.

    The constraints are handled exactly as `sobol_v1` handles them, by the feasible sampler
    (`sampleFeasibleEncoded`), which is handed uniform draws as its raw points in place of a
    sequence. Without linear constraints the points are the first feasible draws: with no
    constraint at all, the first draws. With nonlinear constraints the sampler draws more
    candidates, in rounds, and keeps those that satisfy them, taken in order from one stream
    that continues from round to round. With linear constraints, which include an equality
    such as a mixture that sums to one, the points are drawn inside the polytope by a
    hit-and-run chain, which needs no raw draws: those points are the ones that `sobol_v1`
    draws from the same generator, because both designs hand the same chain the same seed.
    The values that govern the sampler (rounds, chain, tolerance) are those of the constraints
    that the context gives.

    Every random number comes from the generator of the context: the seed of the draws, and
    of the chain, is drawn from it, so the same generator seed gives the same points. The
    stream is made anew at each call and does not continue from one call to the next, so two
    calls with generators of different seeds give two different designs.

    The size of the design is `n_points`, and `pointsNeeded` is that size less the observations
    that exist already if the configuration counts them (`count_existing_observations`).
    """

    def pointsNeeded(self, trials: Sequence[Trial]) -> int:
        """The points still needed: `n_points`, less the observations that exist already when
        they are counted, and never less than zero.

        Args:
            trials: Every trial of the study, in every state. Only the completed ones have an
                observation, so only they are counted.

        Returns:
            A number of points, zero or more.
        """
        counted = 0
        if self.config.count_existing_observations:
            counted = sum(1 for trial in trials if trial.state is TrialState.completed)
        return max(0, self.config.n_points - counted)

    def drawPoints(self, context: DesignContext, n: int) -> Tensor:
        """Draws `n` uniform random points that satisfy the constraints, snapped onto valid
        values.

        Args:
            context: The problem, the encoder, the constraints, the history and the generator.
            n: The number of points, one or more.

        Returns:
            A tensor of shape `(n, d)` of encoded points.

        Raises:
            RuntimeError: if the constraints leave no feasible point to propose (the feasible
                region may be empty, or too small to hit by sampling).
        """
        # The generator may live on another device than the default one, and `randint` wants
        # them to agree. The seed is drawn, and not read from the generator, so that a
        # generator that has been used does not give the same points as a fresh one.
        generator = context.generator
        high = torch.iinfo(torch.int32).max
        seed = int(torch.randint(high, (1,), generator=generator, device=generator.device))
        return sampleFeasibleEncoded(
            context.encoder,
            context.constraints,
            n,
            seed=seed,
            snap=True,
            draw=_uniformStream(seed),
        )


def _uniformStream(seed: int) -> UniformDraw:
    """A draw that continues one stream of uniform numbers from call to call.

    The feasible sampler calls its draw again, for more candidates, when a round leaves too few
    that satisfy the constraints. A draw that made a new stream at each call, from the same
    seed, would offer the same candidates again, so this one keeps its generator and each call
    takes the numbers that follow.

    Args:
        seed: The seed of the stream.

    Returns:
        The draw, `draw(n, dimension)`: `n` rows of `dimension` numbers in `[0, 1)`.
    """
    stream = torch.Generator()
    stream.manual_seed(seed)

    def draw(n: int, dimension: int) -> Tensor:
        """Takes the next `n` points of the stream, in `dimension` dimensions."""
        return torch.rand(n, dimension, generator=stream, dtype=torch.float64)

    return draw
