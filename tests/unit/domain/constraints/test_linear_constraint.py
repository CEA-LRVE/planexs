"""Unit tests for `LinearConstraint`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs import LinearConstraint


class TestLinearConstraint:
    def test_requires_at_least_one_coefficient(self) -> None:
        with pytest.raises(ValidationError, match="at least one coefficient"):
            LinearConstraint({}, bound=1.0, comparator="<=")
