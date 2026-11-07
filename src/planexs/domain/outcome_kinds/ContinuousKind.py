"""The `continuous` outcome kind: a measured quantity, a finite number."""

from __future__ import annotations

import math
from numbers import Real
from typing import ClassVar

from planexs.domain.outcome_kinds.ContinuousKindConfig import ContinuousKindConfig
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.OutcomeValueError import OutcomeValueError
from planexs.domain.outcome_kinds.registerOutcomeKind import registerOutcomeKind


@registerOutcomeKind(
    "continuous",
    config=ContinuousKindConfig,
    description="A measured quantity: a finite number.",
)
class ContinuousKind(OutcomeKind):
    """A continuous outcome: a finite number, with a noise that may be known.

    A value is any real number the caller gives, an integer or a float (NumPy numbers
    included), as long as it is finite: `nan` and the infinities are refused, because a
    surrogate cannot be fitted on them and they must never propagate silently (specification
    8). A `bool` is refused too: it is a label, and passing one where a number is expected is
    almost always a mistake.

    A noise standard deviation is meaningful for such an outcome (specification 5.4).

    Attributes:
        capability_tag: `outcome_kind:continuous`.
        allows_noise: `True`.
    """

    capability_tag: ClassVar[str] = "outcome_kind:continuous"
    allows_noise: ClassVar[bool] = True

    def validateValue(self, value: object, outcome: str) -> float:
        """Checks that `value` is a finite number, and returns it as a `float`.

        Args:
            value: The observed value.
            outcome: The name of the outcome, which the error names.

        Returns:
            The value as a `float`.

        Raises:
            OutcomeValueError: if the value is not a number (a string, `None`, a `bool`, ...),
                is `nan` or infinite, or is too large to be a `float`.
        """
        component = f"outcome {outcome!r}"
        if isinstance(value, bool) or not isinstance(value, Real):
            raise OutcomeValueError(
                f"The value {value!r} of the continuous outcome {outcome!r} is not a number "
                f"(got {type(value).__name__}): a continuous outcome takes a finite number.",
                component=component,
                suggestion="give the measured value as an int or a float",
            )
        try:
            number = float(value)
        except OverflowError:
            raise OutcomeValueError(
                f"The value of the continuous outcome {outcome!r} is too large to be a float.",
                component=component,
                suggestion="check the unit of the measurement",
            ) from None
        if not math.isfinite(number):
            raise OutcomeValueError(
                f"The value {number!r} of the continuous outcome {outcome!r} is not finite: "
                "a continuous outcome takes a finite number, so nan and the infinities are "
                "refused.",
                component=component,
                suggestion="report the evaluation as failed instead of giving it a value",
            )
        return number
