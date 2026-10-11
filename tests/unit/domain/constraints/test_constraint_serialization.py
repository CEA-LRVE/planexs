"""Unit tests for `Constraint.toDict` and `constraintFromDict`."""

from __future__ import annotations

import json

import pytest

from planexs import LinearConstraint
from planexs.domain.constraints.constraintFromDict import constraintFromDict


class TestLinearConstraintRoundTrip:
    def test_round_trips_through_json_text(self) -> None:
        original = LinearConstraint({"a": 1.0, "b": 2.5}, bound=3.0, comparator="<=")
        reloaded = constraintFromDict(json.loads(json.dumps(original.toDict())))

        assert isinstance(reloaded, LinearConstraint)
        assert reloaded == original
        assert (
            reloaded.toAxParameterConstraintString()
            == original.toAxParameterConstraintString()
        )

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            constraintFromDict({"kind": "quadratic", "expression": "x * y"})
