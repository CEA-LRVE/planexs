"""The observed values of a completed trial, with the noise of each when it is known."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from planexs.domain.ObservationError import ObservationError
from planexs.domain.Outcome import Outcome


class Observation(BaseModel):
    """What the evaluation of one trial measured: a value for every outcome and, where
    it is known, the noise standard deviation of that value (specification 4, 5.4;
    ADR-0012, decision 5).

    An observation is an immutable record: no attribute can be assigned, and the two
    mappings it holds are read-only copies, so neither the caller of the constructor nor a
    reader of the observation can rewrite the history through it.

    Whether an observation is the right one for a problem depends on the outcomes of that
    problem, which the record itself does not know (a trial knows no problem either, and the
    study checks it, ADR-0012, decision 3). So there are two ways to make one:

    - `Observation.create(outcomes, values, noise_std)` is the way to build an observation
      from what a caller reports. It checks the values and the noise against the outcomes
      (below) and stores each value as its kind returns it, so a label given as `True` is
      stored as `1.0`.
    - `Observation(values=..., noise_std=...)` takes numbers as they are, and checks only what
      can be checked without the outcomes: every number is finite, a noise standard deviation
      is not negative, and no noise standard deviation is given for a name that has no
      value. It is what a stored document is read back with: the values in it are the stored
      form (a label is `1.0`), which a kind does not accept as a reported value, so a stored
      observation is not checked against the outcomes a second time.

    The rules against a list of outcomes are, in this order:

    1. every outcome has a value, and there is no value (and no noise standard deviation) for
       a name that is no outcome;
    2. each value is valid for the kind of its outcome, as the kind says (a finite number
       for a continuous outcome, a label for a binary one);
    3. a noise standard deviation is given only for an outcome whose kind allows one, and it
       is a finite number that is zero or more.

    The rules ask the outcome and its kind, and never name a kind: a kind that a plugin
    registers (an ordinal or a count outcome) is checked in the same way. Noise is known for
    some outcomes and not for others: an outcome without a noise standard deviation is
    simply absent from `noise_std`, and what a surrogate does with partial knowledge is its
    configuration's business (specification 5.4, ADR-0012). Replicated evaluations of one
    parameterization are separate trials, each with its own observation.

    Attributes:
        values: Maps outcome name to the observed value, a finite number. A label is `0.0` or
            `1.0`.
        noise_std: Maps outcome name to the known standard deviation of that value, in the
            units of the outcome. It holds only the outcomes whose noise is known, and may be
            empty.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Mapping[str, float]
    noise_std: Mapping[str, float] = Field(default_factory=dict, validate_default=True)

    @field_validator("values", mode="before")
    @classmethod
    def _checkValues(cls, mapping: Any) -> Any:
        """Refuses a value that is not a finite number, and reads the others as floats.

        Args:
            mapping: The raw `values`. Anything that is not a mapping is left for pydantic
                to report.

        Returns:
            A new dictionary of floats.

        Raises:
            ObservationError: for a value that is not a finite number.
        """
        if not isinstance(mapping, Mapping):
            return mapping
        return {name: _finite(item, name, "values") for name, item in mapping.items()}

    @field_validator("noise_std", mode="before")
    @classmethod
    def _checkNoise(cls, mapping: Any) -> Any:
        """Refuses a noise standard deviation that is not a finite number of zero or more.

        Args:
            mapping: The raw `noise_std`. Anything that is not a mapping is left for pydantic
                to report.

        Returns:
            A new dictionary of floats.

        Raises:
            ObservationError: for a standard deviation that is not a finite number, or is
                negative.
        """
        if not isinstance(mapping, Mapping):
            return mapping
        return {name: _noiseStd(item, name) for name, item in mapping.items()}

    @field_validator("values", "noise_std", mode="after")
    @classmethod
    def _freezeMappings(cls, mapping: Mapping[str, float]) -> Mapping[str, float]:
        """Stores a validated mapping as a read-only copy of itself."""
        return MappingProxyType(dict(mapping))

    @field_serializer("values", "noise_std")
    def _dumpMappings(self, mapping: Mapping[str, float]) -> dict[str, float]:
        """Writes a read-only mapping back as a plain dictionary."""
        return dict(mapping)

    @model_validator(mode="after")
    def _validateNoiseHasValue(self) -> Observation:
        """Checks that no noise standard deviation is given for a name without a value.

        Returns:
            The observation itself.

        Raises:
            ObservationError: if `noise_std` has a name that `values` does not.
        """
        orphans = [name for name in self.noise_std if name not in self.values]
        if orphans:
            raise ObservationError(
                f"The observation gives a noise standard deviation for {_quoted(orphans)}, "
                "which has no value: a standard deviation describes a value.",
                component="observation",
                field_path="noise_std",
                suggestion="give the value as well, or leave the standard deviation out",
            )
        return self

    @classmethod
    def create(
        cls,
        outcomes: Sequence[Outcome],
        values: Mapping[str, object],
        noise_std: Mapping[str, object] | None = None,
    ) -> Observation:
        """Builds the observation that a caller reports, checked against the outcomes.

        Args:
            outcomes: The outcomes of the problem. Every one needs a value.
            values: Maps outcome name to the observed value, as the caller gives it. Each
                value must be valid for the kind of its outcome.
            noise_std: Maps outcome name to the known standard deviation of its value, for
                the outcomes where it is known. `None` means it is known for none.

        Returns:
            The observation, with every value stored as its kind returns it.

        Raises:
            ObservationError: if an outcome has no value, a value or a noise standard
                deviation names something that is no outcome, a noise standard deviation is
                given for an outcome whose kind has none, or one is not a finite number of
                zero or more.
            OutcomeValueError: if a value is not legal for the kind of its outcome.
        """
        checked, noise = _checkAgainst(outcomes, values, noise_std or {})
        return cls(values=checked, noise_std=noise)

    def __hash__(self) -> int:
        """Hashes the contents: equal observations hold equal items, and the read-only
        mappings an observation holds cannot be hashed themselves.
        """
        return hash(
            (tuple(sorted(self.values.items())), tuple(sorted(self.noise_std.items())))
        )


def _checkAgainst(
    outcomes: Sequence[Outcome],
    values: Mapping[str, object],
    noise_std: Mapping[str, object],
) -> tuple[dict[str, float], dict[str, float]]:
    """Applies the rules of an observation to `values` and `noise_std`.

    Args:
        outcomes: The outcomes of the problem.
        values: Outcome name to the value, as given.
        noise_std: Outcome name to the noise standard deviation, as given.

    Returns:
        The values as their kinds return them, and the noise standard deviations as floats.

    Raises:
        ObservationError: for a missing or unknown name, a noise standard deviation for a
            kind that has none, or one that is not a finite number of zero or more.
        OutcomeValueError: if a value is not legal for the kind of its outcome.
    """
    by_name = {outcome.name: outcome for outcome in outcomes}
    names = list(by_name)

    missing = [name for name in names if name not in values]
    unknown = [name for name in values if name not in by_name]
    unknown_noise = [name for name in noise_std if name not in by_name]
    problems: list[str] = []
    if missing:
        problems.append(f"no value for {_quoted(missing)}")
    if unknown:
        problems.append(f"a value for {_quoted(unknown)}, which is no outcome")
    if unknown_noise:
        problems.append(
            f"a noise standard deviation for {_quoted(unknown_noise)}, which is no outcome"
        )
    if problems:
        raise ObservationError(
            f"The observation does not match the outcomes {names!r}: it has "
            f"{' and it has '.join(problems)}.",
            component="observation",
            field_path="values" if missing or unknown else "noise_std",
            suggestion="give exactly one value for each outcome, under the outcome's name",
        )

    kinds = {name: outcome.kind() for name, outcome in by_name.items()}
    checked = {name: kinds[name].validateValue(values[name], name) for name in names}

    noise: dict[str, float] = {}
    for name, deviation in noise_std.items():
        if not kinds[name].allows_noise:
            raise ObservationError(
                f"A noise standard deviation was given for the outcome {name!r}, whose kind "
                f"{by_name[name].type!r} has none: it is not meaningful for such an outcome.",
                component=f"outcome {name!r}",
                field_path="noise_std",
                suggestion="leave the noise standard deviation of this outcome out",
            )
        noise[name] = _noiseStd(deviation, name)
    return checked, noise


def _finite(value: object, name: str, field: str) -> float:
    """Checks that a value is a finite number, and returns it as a float.

    Args:
        value: The value to check. A `bool` is refused: it is a label, and an observation
            stores a label as `0.0` or `1.0` once its kind has accepted it.
        name: The name of the outcome, which the error names.
        field: `"values"` or `"noise_std"`, the field the value is in.

    Returns:
        The value as a `float`.

    Raises:
        ObservationError: if the value is not a number, is too large for a float, or is
            `nan` or infinite.
    """
    label = "value" if field == "values" else "noise standard deviation"
    component = f"outcome {name!r}"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ObservationError(
            f"The {label} {value!r} of the outcome {name!r} is not a number "
            f"(got {type(value).__name__}): an observation holds finite numbers.",
            component=component,
            field_path=field,
            suggestion="give a finite int or float",
        )
    try:
        number = float(value)
    except OverflowError:
        raise ObservationError(
            f"The {label} of the outcome {name!r} is too large to be a float.",
            component=component,
            field_path=field,
            suggestion="check the unit of the measurement",
        ) from None
    if not math.isfinite(number):
        raise ObservationError(
            f"The {label} {number!r} of the outcome {name!r} is not finite: nan and the "
            "infinities are refused.",
            component=component,
            field_path=field,
            suggestion=(
                "report the evaluation as failed instead of giving it a value"
                if field == "values"
                else "leave the noise standard deviation out when it is not known"
            ),
        )
    return number


def _noiseStd(value: object, name: str) -> float:
    """Checks that a noise standard deviation is a finite number of zero or more.

    Args:
        value: The standard deviation to check.
        name: The name of the outcome, which the error names.

    Returns:
        The standard deviation as a `float`.

    Raises:
        ObservationError: if it is not a finite number, or is negative.
    """
    number = _finite(value, name, "noise_std")
    if number < 0.0:
        raise ObservationError(
            f"The noise standard deviation {number!r} of the outcome {name!r} is negative: "
            "a standard deviation is zero or more.",
            component=f"outcome {name!r}",
            field_path="noise_std",
            suggestion="give the standard deviation (not the variance), as a number of zero "
            "or more",
        )
    return number


def _quoted(names: Sequence[object]) -> str:
    """Writes names as a list for a message: `'a', 'b'`."""
    return ", ".join(repr(name) for name in names)
