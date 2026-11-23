"""A hypervolume reference point that the configuration gives as numbers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ExplicitReferencePointConfig(BaseModel):
    """A reference point for the hypervolume, given as one number for each objective
    (specification 5.8).

    The hypervolume of a set of points is the volume of the region that they dominate and
    that lies on the better side of a reference point, so the reference point says how bad an
    outcome may be and still count as a result. It is the one thing about a Pareto problem that
    the data cannot tell: only the person who knows what a useless result is can say. This is
    the form that says it. `InferredReferencePointConfig` is the form that asks the data.

    The values are in the natural units of the outcomes, as they are measured, and not in the
    form the acquisition works in. For a maximized outcome the value is the lowest result that
    still counts, and for a minimized one the highest. It is built with keywords and is
    frozen. Unknown fields are rejected.

    Attributes:
        kind: Always `"explicit"`. It tells this form from the inferred one in a stored
            configuration.
        values: Maps the name of each objective of the goal to its reference value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["explicit"] = Field(
        default="explicit",
        description=(
            "Which form of reference point this is, always 'explicit': the values are the "
            "ones given in 'values'. It tells this form from 'inferred', which reads the "
            "reference point off the data, in a stored configuration. It is the only value "
            "the field accepts."
        ),
    )
    values: dict[str, FiniteFloat] = Field(
        min_length=2,
        description=(
            "Maps the name of each objective of the goal, as the problem names it, to its "
            "reference value, in the unit of the outcome as it is measured. For a maximized "
            "outcome it is the lowest value that still counts as a result, and for a "
            "minimized outcome the highest: a point adds hypervolume only where it is better "
            "than the reference value in every objective. There must be exactly one value "
            "for each objective of the goal, which is checked when the acquisition is built, "
            "and at least two, since a hypervolume needs two objectives. Every value must be "
            "finite. There is no default: only the person who knows what a useless result "
            "is can say."
        ),
    )
