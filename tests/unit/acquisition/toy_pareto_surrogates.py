"""A toy surrogate of a Pareto problem, for the tests of the multi-objective acquisitions.

It is a real surrogate over a real BoTorch model (see `toy_surrogates`). Its parameter is one
real `x` in `[0, 1]`, and it can model three outcomes of it, known exactly at the observed
points:

- `f1 = x`: it grows over the whole space;
- `f2 = sin(3 x)`: it rises to a single maximum at `x = pi / 6` (about 0.52) and falls after
  it;
- `g = scale * x`: like `f1`, an outcome that a goal may constrain instead of optimize, and
  `scale` stands for the unit it is measured in.

With both objectives maximized, the observed points at `x < 0.52` are dominated, since moving
right improves `f1` and `f2` together, and the points at `x > 0.52` are the Pareto front:
moving right still improves `f1` but now costs `f2`. With `f1` minimized and `f2` maximized it
is the other way round: the front is at `x < 0.52` and the points at the right are dominated.
So a candidate far to the right extends the front in the first problem and is dominated in the
second, which is what the tests of the directions use.

The model is left at the hyperparameters it starts with: the tests are about what the
acquisition does with the model, not about how a model is fitted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from toy_surrogates import makeSurrogateOf

from planexs.surrogates.Surrogate import Surrogate

#: The first toy objective, `x`.
F1 = "f1"

#: The second toy objective, `sin(3 x)`.
F2 = "f2"

#: The toy outcome that a goal constrains, `scale * x`.
G = "g"


def makeParetoSurrogate(
    outcomes: Sequence[str] = (F1, F2, G),
    constraint_scale: float = 1.0,
    observed_x: Sequence[float] | None = None,
) -> Surrogate:
    """A surrogate of some of the toy outcomes, over eight observed points.

    Args:
        outcomes: The outcomes to model, by name, in the order of the outputs of the model.
            Any of `F1`, `F2` and `G`.
        constraint_scale: The factor that `g = scale * x` is measured in, so that a test can
            change the unit of an outcome and nothing else.
        observed_x: The places of the observed points, in `[0, 1]`. `None` means eight points
            spread over the space.

    Returns:
        The surrogate, whose native model is in evaluation mode.
    """
    available: dict[str, Callable[[Tensor], Tensor]] = {
        F1: lambda x: x,
        F2: lambda x: torch.sin(3.0 * x),
        G: lambda x: constraint_scale * x,
    }
    return makeSurrogateOf({name: available[name] for name in outcomes}, observed_x)
