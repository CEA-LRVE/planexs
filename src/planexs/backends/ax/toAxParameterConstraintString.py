"""Renders a planexs `LinearConstraint` as one of Ax's `parameter_constraints` strings."""

from __future__ import annotations

from planexs.domain.constraints.LinearConstraint import LinearConstraint


def toAxParameterConstraintString(constraint: LinearConstraint) -> str:
    """Renders `constraint` as one entry of the `parameter_constraints` sequence that
    `Client.configure_experiment` expects, for example `"a + b + c = 1.0"`.

    A coefficient of exactly `1.0` is rendered as a bare parameter name (`"a"`, not
    `"1.0 * a"`), which suits mixture-style constraints; every other coefficient is
    rendered explicitly (`"2.0 * b"`). Any comparator is rendered, `"="` included. Whether
    Ax accepts the result is up to the caller: Ax takes inequalities only, which is why
    `AxBackend.createExperiment` leaves an equality out and lets planexs' own acquisition
    layer enforce it.

    Args:
        constraint: The linear constraint.

    Returns:
        The constraint as the string-expression mini-language of Ax.
    """
    terms: list[str] = []
    for name, coefficient in constraint.coefficients.items():
        if coefficient == 1.0:
            terms.append(name)
        else:
            terms.append(f"{coefficient} * {name}")
    left_hand_side = " + ".join(terms)
    return f"{left_hand_side} {constraint.comparator} {constraint.bound}"
