"""The probit link: a label is true with the probability that the standard normal gives."""

from __future__ import annotations

import numpy
import torch
from torch import Tensor

from planexs.surrogates.ResponseLink import ResponseLink

#: The order of the Gauss-Legendre rule that integrates Owen's T function. The integrand is
#: smooth on the unit interval, so this order gives it to rounding error for every argument and
#: does not change a result beyond it: a test compares the moments with a brute-force
#: integration of the definition.
_OWEN_RULE_ORDER = 32  # structural: accuracy of a quadrature rule, not a policy value


def probitLink() -> ResponseLink:
    """The link of the Bernoulli likelihood of GPyTorch (a probit), as a `ResponseLink`.

    The response of a latent value `f` is `Phi(f)`, with `Phi` the standard normal
    distribution function, and the latent value of a probability `p` is `Phi^-1(p)`. When the
    latent value is normal, with mean `m` and variance `v`, both moments of `Phi(f)` have an
    exact form, so that the response of a prediction is not a numerical approximation of the
    engine's (a quadrature of a few dozen points is wrong by several points of probability
    where the posterior is very wide):

    - the mean is `Phi(m / sqrt(1 + v))`, which is what GPyTorch's `BernoulliLikelihood` gives
      as the predictive probability;
    - the second moment is `Phi(a) - 2 T(a, 1 / sqrt(1 + 2 v))` with `a = m / sqrt(1 + v)`
      and `T` Owen's T function, from which the variance follows. Owen's T has no function in
      torch, so it is integrated from its definition, `T(h, b) = 1 / (2 pi) * integral from 0
      to b of exp(-h^2 (1 + x^2) / 2) / (1 + x^2) dx`, with a Gauss-Legendre rule.

    Returns:
        The link. It holds no state, and every call makes an equal one.
    """
    return ResponseLink(
        name="probit",
        inverse=torch.special.ndtr,
        forward=torch.special.ndtri,
        marginal=_probitMarginal,
    )


def _probitMarginal(mean: Tensor, variance: Tensor) -> tuple[Tensor, Tensor]:
    """The mean and the standard deviation of `Phi(f)` for `f` normal.

    Args:
        mean: The mean of the latent value, of any shape.
        variance: Its variance, of the shape of `mean`, zero or more.

    Returns:
        The mean and the standard deviation of the probability, of the shape of the arguments:
        the mean in `[0, 1]`, the standard deviation in `[0, 0.5]`.
    """
    scale = torch.sqrt(1.0 + variance)
    probability = torch.special.ndtr(mean / scale)
    second_moment = probability - 2.0 * _owensT(
        mean / scale, 1.0 / torch.sqrt(1.0 + 2.0 * variance)
    )
    spread = (second_moment - probability * probability).clamp_min(0.0)
    return probability, torch.sqrt(spread)


def _owensT(h: Tensor, upper: Tensor) -> Tensor:
    """Owen's T function `T(h, upper)` for `upper` in `[0, 1]`.

    Args:
        h: The first argument, of any shape.
        upper: The second argument, of the shape of `h`.

    Returns:
        `T(h, upper)`, of the shape of the arguments.
    """
    nodes, weights = numpy.polynomial.legendre.leggauss(_OWEN_RULE_ORDER)
    points = torch.as_tensor(nodes, dtype=h.dtype, device=h.device)
    rule = torch.as_tensor(weights, dtype=h.dtype, device=h.device)
    # The nodes are on [-1, 1] and are mapped onto [0, upper]; half the length of that interval
    # multiplies the weights, and is written as a division.
    x = upper.unsqueeze(-1) * (points + 1.0) / 2.0
    curve = torch.exp(-(h.unsqueeze(-1) ** 2) * (1.0 + x**2) / 2.0) / (1.0 + x**2)
    return (upper / 2.0 / (2.0 * torch.pi)) * (curve * rule).sum(dim=-1)
