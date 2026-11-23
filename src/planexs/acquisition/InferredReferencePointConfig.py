"""A hypervolume reference point that is read off the data by a named rule."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InferredReferencePointConfig(BaseModel):
    """A reference point for the hypervolume that a named rule reads off the model's own data
    (specification 5.8).

    The rule is applied each time the acquisition is built, to the Pareto front that the
    surrogate predicts at the points observed so far, so the reference point follows the data
    and nothing has to be known beforehand. `ExplicitReferencePointConfig` is the form for
    when something is known. The only rule so far is `nadir_margin`: the nadir of the
    predicted front, which is the worst value that the front has in each objective, moved
    further out by a fraction of the range of the front, so that the points at the ends of
    the front still add volume. A new rule is a new value of `rule`. It is built with keywords
    and is frozen. Unknown fields are rejected.

    Attributes:
        kind: Always `"inferred"`. It tells this form from the explicit one in a stored
            configuration.
        rule: The name of the rule.
        margin: How far beyond the nadir the reference point is put, as a fraction of the
            range of the front.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["inferred"] = Field(
        default="inferred",
        description=(
            "Which form of reference point this is, always 'inferred': the point is read off "
            "the data by the rule named in 'rule'. It tells this form from 'explicit', which "
            "gives the numbers, in a stored configuration. It is the only value the field "
            "accepts."
        ),
    )
    rule: Literal["nadir_margin"] = Field(
        default="nadir_margin",
        description=(
            "The name of the rule that infers the reference point from the Pareto front that "
            "the surrogate predicts at the observed points. 'nadir_margin' puts it at the "
            "nadir of that front (its worst value in each objective) moved out by 'margin' "
            "times the range of the front in that objective, the rule of BoTorch's "
            "'infer_reference_point', which Ax uses as well. With a single point on the front "
            "the margin is taken of the absolute value of the point. It is the only rule so "
            "far, and the default."
        ),
    )
    margin: float = Field(
        default=0.1,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "How far beyond the nadir of the predicted front the reference point is put, as a "
            "fraction of the range of the front in each objective, so a unitless number that "
            "does not depend on the unit of an outcome. 0 puts the reference point on the "
            "nadir, where the points that have the worst value of an objective add no volume "
            "at all; the larger it is, the more weight the extremes of the front get against "
            "its middle. It must be 0 or more and finite. The default is 0.1, the multiplier "
            "that Ishibuchi and colleagues (2011) found robust for scaling the nadir point "
            "and that BoTorch uses by default, a starting point and not a recommendation for "
            "every problem."
        ),
    )
