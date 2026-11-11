"""The tolerances with which a parameterization is checked against a problem."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ParameterizationCheckConfig(BaseModel):
    """The settings of the check that a parameterization is legal for its problem
    (specification 3.4, 10.8).

    A parameterization is a floating-point record, so some of what makes it legal cannot be
    judged exactly: a stepped range holds the values `lower + k * step`, which a computer
    stores only approximately, and a mixture that sums to one sums to one up to the rounding of
    its terms. These two settings say how much slack the check allows. They are the only policy
    values of the check: everything else it decides is exact.

    The two defaults are the ones the rest of the library already uses for the same
    questions, so that a parameterization that the library produced is not refused by the
    library: `Range` accepts a default that is off its grid by less than `1e-9` steps, and the
    acquisition layer accepts an encoded point that violates a constraint by less than `1e-6`
    of the constraint's scale (`EncodedConstraints.isFeasible`).

    A configuration is frozen: it is shared by every check of a study and must not change
    under them. Build a new one to change a setting.

    Attributes:
        grid_tolerance: How far a value of a stepped range may be from its grid, in steps.
        constraint_tolerance: How much a constraint may be violated, relative to its scale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid_tolerance: float = Field(
        default=1e-9,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "How far a value of a range with a step size may be from the nearest value of "
            "its grid (`lower + k * step_size` for a whole number `k`), measured in steps, so "
            "that 0.5 is half way between two grid values. A value is on the grid if it is "
            "within this distance of one. The default is 1e-9 steps, which is the tolerance "
            "that `Range` applies to its own default value: far above the rounding of a value "
            "that was computed on the grid (the encoder keeps twelve significant digits of "
            "it), far below any step a person would choose. Zero asks for exact values, which "
            "rounding makes impossible for a step such as 0.1."
        ),
    )
    constraint_tolerance: float = Field(
        default=1e-6,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "The largest violation of a parameter constraint that is still accepted, relative "
            "to the scale of the constraint (`Constraint.violation`): 1e-6 accepts a mixture "
            "that sums to 1.000001. Dimensionless. The default is the tolerance that the "
            "acquisition layer accepts for an encoded point "
            "(`EncodedConstraints.isFeasible`), so that a point its optimizer returned is not "
            "refused here, and it is far above the rounding of a sum of a few floats. Zero "
            "asks for exact satisfaction, which rounding makes impossible for an equality."
        ),
    )
