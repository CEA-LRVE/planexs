"""What a surrogate believes about the outcomes at some parameterizations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from planexs.surrogates.PredictionSpace import PredictionSpace


class Prediction(BaseModel):
    """The belief of a surrogate about the outcomes at one or several parameterizations,
    with what it describes (specification 5.6, 3.11; FR7, FR8).

    For each outcome it holds the predicted mean and the standard deviation of that belief at
    every point asked about, in the units of the outcome, and it states which space they live
    in and whether the observation noise is included. A prediction is never a bare pair of
    numbers: the same mean and standard deviation mean different things for the latent
    function and for a noisy measurement, and the record says which one this is.

    `Prediction` replaces the legacy `PredictionResult`, which holds one point and a standard
    error of the mean, and is deleted with the rest of the legacy code (roadmap task P11-03).
    The standard deviation here is the same quantity as that standard error, for the same
    space, and `variance` is derived from it, as it was there.

    A prediction is an immutable record of plain data: no attribute can be assigned, the two
    mappings it holds are read-only copies and the numbers are tuples, so it can be compared,
    hashed and written as JSON (`model_dump_json`) and read back. It is built with keyword
    arguments, and checked when it is built: the outcomes are named once each, `mean` and
    `std` have exactly those outcomes and the same number of points, every number is finite,
    a standard deviation is zero or more, and a prediction in the latent space includes no
    observation noise, since a function has none.

    Attributes:
        outcomes: The names of the outcomes, in the order of the surrogate's outputs.
        mean: Maps outcome name to the predicted mean at each point, in the order the points
            were asked for.
        std: Maps outcome name to the predicted standard deviation at each point. It is the
            standard deviation of the belief about the function in the latent space (what Ax
            calls the standard error of the mean), and that of a new measurement in the
            response space when the noise is included.
        space: `"latent"` or `"response"`: the space that `mean` and `std` describe.
        includes_observation_noise: Whether `std` includes the observation noise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcomes: tuple[str, ...]
    mean: Mapping[str, tuple[float, ...]]
    std: Mapping[str, tuple[float, ...]]
    space: PredictionSpace
    includes_observation_noise: bool = Field(strict=True)

    @field_validator("mean", "std", mode="after")
    @classmethod
    def _freezeMappings(
        cls, value: Mapping[str, tuple[float, ...]]
    ) -> Mapping[str, tuple[float, ...]]:
        """Stores a validated mapping as a read-only copy of itself."""
        return MappingProxyType(dict(value))

    @field_serializer("mean", "std")
    def _dumpMappings(self, value: Mapping[str, tuple[float, ...]]) -> dict[str, list[float]]:
        """Writes a read-only mapping back as plain JSON-compatible data."""
        return {name: list(numbers) for name, numbers in value.items()}

    @model_validator(mode="after")
    def _validateConsistency(self) -> Prediction:
        """Checks that the pieces of the prediction describe the same outcomes and points.

        Returns:
            The prediction itself.

        Raises:
            ValueError: if there is no outcome or a name repeats, `mean` or `std` has other
                outcomes than `outcomes`, the tables do not all hold the same number of
                points, a number is not finite, a standard deviation is negative, or the
                prediction is in the latent space and includes observation noise. The
                message names the field.
        """
        names = list(self.outcomes)
        if not names:
            raise ValueError("outcomes is empty: a prediction is about at least one outcome")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"outcomes names {duplicates!r} more than once")
        for field_name, table in (("mean", self.mean), ("std", self.std)):
            if set(table) != set(names):
                raise ValueError(
                    f"{field_name} has the outcomes {sorted(table)!r}, but outcomes says "
                    f"{sorted(names)!r}: every outcome needs a value for each point"
                )
            for name, numbers in table.items():
                if not all(math.isfinite(number) for number in numbers):
                    raise ValueError(
                        f"{field_name}[{name!r}] holds a value that is not finite: a "
                        "prediction never passes on nan or an infinity"
                    )
        sizes = {len(numbers) for table in (self.mean, self.std) for numbers in table.values()}
        if len(sizes) > 1:
            raise ValueError(
                f"mean and std do not hold the same number of points ({sorted(sizes)!r}): "
                "every outcome needs one value per point, in both"
            )
        for name, numbers in self.std.items():
            if any(number < 0.0 for number in numbers):
                raise ValueError(
                    f"std[{name!r}] holds a negative value: a standard deviation is zero or "
                    "more"
                )
        if self.space == "latent" and self.includes_observation_noise:
            raise ValueError(
                "a prediction in the latent space includes no observation noise: the latent "
                "function has none. Say space 'response' for a noisy measurement"
            )
        return self

    @property
    def n_points(self) -> int:
        """The number of points the prediction is about (a length read: cheap, snake_case)."""
        return len(self.mean[self.outcomes[0]])

    @property
    def variance(self) -> dict[str, tuple[float, ...]]:
        """`std` squared, per outcome and point, in the units of the outcome squared.

        A multiplication over numbers the prediction already holds, which is negligible next to
        the model call that produced them, so it is a property in snake_case, as the
        legacy `PredictionResult.variance` was (specification 10.2). It is derived and never
        stored, so it cannot disagree with `std`.

        Returns:
            A new dictionary: changing it does not change the prediction.
        """
        return {
            name: tuple(number * number for number in numbers)
            for name, numbers in self.std.items()
        }

    def __hash__(self) -> int:
        """Hashes the contents: equal predictions hold equal items, and the read-only
        mappings a prediction holds cannot be hashed themselves.
        """
        return hash(
            (
                self.outcomes,
                tuple(sorted(self.mean.items())),
                tuple(sorted(self.std.items())),
                self.space,
                self.includes_observation_noise,
            )
        )
