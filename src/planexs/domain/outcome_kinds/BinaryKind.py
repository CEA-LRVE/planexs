"""The `binary` outcome kind: a label that is true or false."""

from __future__ import annotations

from numbers import Integral
from typing import ClassVar

from planexs.domain.outcome_kinds.BinaryKindConfig import BinaryKindConfig
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.OutcomeValueError import OutcomeValueError
from planexs.domain.outcome_kinds.registerOutcomeKind import registerOutcomeKind


@registerOutcomeKind(
    "binary",
    config=BinaryKindConfig,
    description="A label that is true or false: a bool, or the integer 0 or 1.",
)
class BinaryKind(OutcomeKind):
    """A binary outcome: a label, with no noise standard deviation.

    A value is a `bool` or the integer `0` or `1` (NumPy booleans and integers included),
    and is stored as `0.0` or `1.0`. Anything else is refused, `2` and a float such as `1.0`
    included: a label is not a measurement, and a value that is not one is more likely a
    mistake than a label.

    No noise standard deviation is meaningful for a label (specification 5.4). A surrogate for
    such an outcome predicts in two spaces, the latent function and the response probability,
    and says which one it returns (specification 5.6, 3.11).

    Attributes:
        capability_tag: `outcome_kind:binary`.
        allows_noise: `False`.
    """

    capability_tag: ClassVar[str] = "outcome_kind:binary"
    allows_noise: ClassVar[bool] = False

    def validateValue(self, value: object, outcome: str) -> float:
        """Checks that `value` is a label, and returns it as `0.0` or `1.0`.

        Args:
            value: The observed value.
            outcome: The name of the outcome, which the error names.

        Returns:
            `0.0` or `1.0`.

        Raises:
            OutcomeValueError: if the value is neither a `bool` nor the integer 0 or 1.
        """
        if _isBoolean(value):
            return 1.0 if value else 0.0
        if isinstance(value, Integral) and value in (0, 1):
            return float(value)
        raise OutcomeValueError(
            f"The value {value!r} of the binary outcome {outcome!r} is not a label "
            f"(got {type(value).__name__}): a binary outcome takes a bool or the integer 0 "
            "or 1.",
            component=f"outcome {outcome!r}",
            suggestion="give True or False, or 1 or 0 (cast a float with int(value))",
        )


def _isBoolean(value: object) -> bool:
    """Whether `value` is a Python `bool` or a NumPy boolean.

    A NumPy boolean (what a comparison of arrays gives) is not a `bool` subclass. It is
    recognized by the kind of its `dtype`, so that the domain does not import NumPy. Only a
    scalar is: an array of booleans, even of one element, is not a label.

    Args:
        value: The value.

    Returns:
        `True` for a `bool` or a NumPy boolean scalar.
    """
    if isinstance(value, bool):
        return True
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    return kind == "b" and getattr(value, "shape", None) == ()
