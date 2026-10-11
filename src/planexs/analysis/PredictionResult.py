"""The result of predicting an arbitrary parameterization's outcome."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    """The model's belief about an arbitrary (not necessarily evaluated)
    parameterization's outcome (FR7), together with calibrated uncertainty
    (FR8).

    Attributes:
        mean: Maps metric name to the predicted mean.
        sem: Maps metric name to the predicted standard error of the mean:
            matches `Client.predict`'s own `(mean, sem)` return shape
            directly rather than converting to variance and introducing a
            field Ax itself does not use.
    """

    mean: dict[str, float]
    sem: dict[str, float]

    @property
    def variance(self) -> dict[str, float]:
        """`sem` squared over a handful of metrics: negligible cost,
        snake_case even though it "computes" something, per section 10.
        """
        return {name: value * value for name, value in self.sem.items()}
