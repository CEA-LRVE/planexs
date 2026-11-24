"""The seed of one fit, drawn from the generator that the fit is given."""

from __future__ import annotations

import torch


def drawSeed(generator: torch.Generator) -> int:
    """Draws the seed of one fit from the generator that the fit is given.

    The seed is drawn, and not read from the generator, so that a generator that has been used
    does not give the same fit as a fresh one. The generator may live on another device than
    the default one, and `randint` wants them to agree.

    Args:
        generator: The seeded generator.

    Returns:
        An integer that `torch.manual_seed` accepts.
    """
    high = torch.iinfo(torch.int32).max
    return int(torch.randint(high, (1,), generator=generator, device=generator.device))
