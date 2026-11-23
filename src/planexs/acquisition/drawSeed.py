"""A seed for a sampler, drawn from the generator of a build."""

from __future__ import annotations

import torch


def drawSeed(generator: torch.Generator) -> int:
    """A seed for a quasi-random sampler, drawn from the generator of a build.

    The seed is drawn, and not read from the generator, so that a generator that has been used
    does not give the same samples as a fresh one. The generator may live on another device
    than the default one, and `randint` wants them to agree.

    Args:
        generator: The generator of the build context. Drawing advances it.

    Returns:
        An integer seed.
    """
    high = torch.iinfo(torch.int32).max
    return int(torch.randint(high, (1,), generator=generator, device=generator.device))
