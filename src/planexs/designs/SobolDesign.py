"""A design of scrambled Sobol points."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.DesignContext import DesignContext
from planexs.designs.registerDesign import registerDesign
from planexs.designs.SobolDesignConfig import SobolDesignConfig
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.sampleFeasibleEncoded import sampleFeasibleEncoded
from planexs.encoding.UniformDraw import UniformDraw


@registerDesign(
    "sobol_v1",
    config=SobolDesignConfig,
    description="Scrambled Sobol points in the encoded space, honouring the constraints.",
)
class SobolDesign(AbstractDesign[SobolDesignConfig]):
    """Space-filling points from a Sobol sequence, in the unit cube of the encoding.

    A Sobol sequence spreads its first points evenly over the cube, so a small design covers
    the space far better than the same number of random points. The points are drawn in the
    encoded space, so a log-scaled range, a grid, an ordered choice and a categorical are all
    covered as the encoder lays them out, and each point is snapped onto valid values.

    The constraints are honoured by the feasible sampler (`sampleFeasibleEncoded`), which this
    design hands the sequence to as its raw points. Without linear constraints the points are
    the first feasible points of the sequence: with no constraint at all, its first points.
    With nonlinear constraints the sampler draws more candidates, in rounds, and keeps those
    that satisfy them, taken in order from one sequence that continues from round to round.
    With linear constraints, which include an equality such as a mixture that sums to one, the
    points are drawn inside the polytope by a hit-and-run chain, which needs no sequence: the
    points are then feasible and spread by the chain, and `scramble` does not change them. The
    values that govern the sampler (rounds, chain, tolerance) are those of the constraints
    that the context gives.

    Every random number comes from the generator of the context: the seed of the sequence, and
    of the chain, is drawn from it, so the same generator seed gives the same points. The
    sequence is made anew at each call and does not continue from one call to the next, so two
    calls with generators of different seeds give two different designs, each spread evenly on
    its own.

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
        """Draws `n` points of a scrambled (or, if the configuration says so, plain) Sobol
        sequence that satisfy the constraints, snapped onto valid values.

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
            draw=_sobolSequence(self.config.scramble, seed),
        )


def _sobolSequence(scramble: bool, seed: int) -> UniformDraw:
    """A draw that continues one Sobol sequence from call to call.

    The feasible sampler calls its draw again, for more candidates, when a round leaves too few
    that satisfy the constraints. A draw that made a new sequence at each call, from the same
    seed, would offer the same first points again, so this one keeps its engine and each call
    takes the points that follow. The engine is made at the first call, when the number of
    searched parameters is known.

    Args:
        scramble: Whether the sequence is scrambled.
        seed: The seed of the scrambling.

    Returns:
        The draw, `draw(n, dimension)`.
    """
    engine: SobolEngine | None = None

    def draw(n: int, dimension: int) -> Tensor:
        """Takes the next `n` points of the sequence, in `dimension` dimensions."""
        nonlocal engine
        if engine is None:
            engine = SobolEngine(  # type: ignore[no-untyped-call]
                dimension=dimension, scramble=scramble, seed=seed
            )
        return engine.draw(n)

    return draw
