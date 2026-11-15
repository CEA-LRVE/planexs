"""The type of the space that a prediction lives in."""

from __future__ import annotations

from typing import Literal

#: The space a prediction describes (specification 5.6, 3.11).
#:
#: `"latent"` is the underlying function of the surrogate's own model of an outcome, with
#: no observation noise: the epistemic uncertainty about the function. `"response"` is what
#: is observed: a new measurement, so a Gaussian outcome carries its observation noise there,
#: and the outcome of a non-Gaussian likelihood (a label) is read through its link. A
#: prediction always states which of the two it returns, so that a mean and a standard
#: deviation are never mistaken for the other space's.
PredictionSpace = Literal["latent", "response"]
