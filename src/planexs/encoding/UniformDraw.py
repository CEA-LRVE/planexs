"""The type of a function that draws the raw points a sampler turns into encoded ones."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

#: A function `draw(n, dimension)` that returns a tensor of shape `(n, dimension)` of values
#: in `[0, 1)`, one row per point and one column per searched parameter. `SearchSpaceEncoder`
#: turns each column into the encoding of its parameter (a rank, a category, a position on a
#: log scale), so a draw never needs to know about any of them: it is the one thing that
#: tells a space-filling design (a scrambled Sobol sequence) from a random one (uniform
#: draws).
#:
#: A draw owns its randomness. It is built by whoever knows the seed (a design builds one from
#: the generator it was given) and it may keep state, so that each call continues where the
#: last stopped: the sampler calls it again, for more points, when a round of candidates leaves
#: too few that satisfy the constraints, and a sequence that restarted would offer the same
#: candidates again. No seed is passed to it for that reason.
UniformDraw = Callable[[int, int], Tensor]
