"""How a latent function is read as the response of an outcome: a link and its inverse."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ResponseLink:
    """The link that reads a latent function as the response of an outcome (specification
    5.6, 3.11).

    A surrogate for a non-Gaussian outcome has a latent function with a Gaussian posterior, and
    the outcome is read from it through a link: a label is true with the probability that the
    inverse link gives to the latent value. This record holds what the `Surrogate` wrapper
    needs to speak about both spaces without knowing which model it is wrapping, so that a
    prediction can say which of the two it describes:

    - `inverse` maps the latent function to the response, as the model reads it (for a
      probit link the standard normal distribution function, which gives a probability);
    - `forward` maps a response back to the latent function: the conversion of a level that
      is given in the response space (a probability) into a threshold on the latent function,
      where a level-set acquisition or analysis works;
    - `marginal` gives the mean and the standard deviation of the response when the latent
      function is not known but believed to be normal, which is what the posterior of the model
      says at a point.

    It is a frozen record of functions and not a class to derive from: there is one link so
    far (`probitLink`), and a second one is a second function that returns this record, not a
    new kind of component (specification 3.10). It is made by the builder that fits the model,
    which knows how the model reads its outcomes, and travels with the fitted model in
    `BuiltSurrogate`. Every function works on tensors of any shape, element by element, and
    keeps the precision and the device of its argument.

    Attributes:
        name: The name of the link, as in the capability tag `link:<name>` that a surrogate
            declares (`probit` for `link:probit`).
        inverse: Latent value to response, increasing.
        forward: Response to latent value, the inverse of `inverse` on the open interval of the
            responses: a response outside it has no latent value.
        marginal: `(mean, variance)` of a normal latent value to `(mean, standard deviation)`
            of its response.
    """

    name: str
    inverse: Callable[[Tensor], Tensor]
    forward: Callable[[Tensor], Tensor]
    marginal: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
