"""Unit tests for `toBotorchNonlinearConstraints`."""

from __future__ import annotations

import pytest
import torch

from planexs import NonlinearConstraint
from planexs.acquisition.toBotorchNonlinearConstraints import toBotorchNonlinearConstraints


def _point(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.double)


class TestSignConvention:
    def test_upper_bound_is_positive_when_satisfied(self) -> None:
        [(function, _)] = toBotorchNonlinearConstraints(
            [NonlinearConstraint("x * y ** 2", "<=", 50.0)], ["x", "y"]
        )

        assert float(function(_point(2.0, 3.0))) == pytest.approx(32.0)  # 50 - 18
        assert float(function(_point(10.0, 3.0))) == pytest.approx(-40.0)  # 50 - 90
        assert float(function(_point(2.0, 5.0))) == pytest.approx(
            0.0
        )  # exactly on the boundary

    def test_lower_bound_is_positive_when_satisfied(self) -> None:
        [(function, _)] = toBotorchNonlinearConstraints(
            [NonlinearConstraint("x + y", ">=", 4.0)], ["x", "y"]
        )

        assert float(function(_point(3.0, 3.0))) == pytest.approx(2.0)
        assert float(function(_point(1.0, 1.0))) == pytest.approx(-2.0)


class TestBotorchContract:
    def test_every_constraint_is_intra_point(self) -> None:
        compiled = toBotorchNonlinearConstraints(
            [NonlinearConstraint("x * y", "<=", 1.0), NonlinearConstraint("x - y", ">=", 0.0)],
            ["x", "y"],
        )
        assert [flag for _, flag in compiled] == [True, True]

    def test_a_single_point_gives_a_scalar_tensor(self) -> None:
        [(function, _)] = toBotorchNonlinearConstraints(
            [NonlinearConstraint("x * y", "<=", 1.0)], ["x", "y"]
        )
        result = function(_point(0.5, 0.5))
        assert isinstance(result, torch.Tensor) and result.ndim == 0

    def test_a_batch_gives_one_value_per_row(self) -> None:
        [(function, _)] = toBotorchNonlinearConstraints(
            [NonlinearConstraint("x * y", "<=", 1.0)], ["x", "y"]
        )
        batch = torch.tensor([[0.5, 0.5], [2.0, 2.0], [1.0, 1.0]], dtype=torch.double)
        assert function(batch).tolist() == pytest.approx([0.75, -3.0, 0.0])

    def test_is_differentiable(self) -> None:
        [(function, _)] = toBotorchNonlinearConstraints(
            [NonlinearConstraint("x * y ** 2", "<=", 50.0)], ["x", "y"]
        )
        point = _point(2.0, 3.0).requires_grad_()

        function(point).backward()

        assert point.grad.tolist() == pytest.approx([-9.0, -12.0])  # -y^2, -2xy

    def test_preserves_order_of_constraints(self) -> None:
        compiled = toBotorchNonlinearConstraints(
            [NonlinearConstraint("x", "<=", 1.0), NonlinearConstraint("x", ">=", 1.0)], ["x"]
        )
        assert float(compiled[0][0](_point(3.0))) == -2.0
        assert float(compiled[1][0](_point(3.0))) == 2.0

    def test_parameter_order_decides_which_column_is_which(self) -> None:
        constraint = NonlinearConstraint("x - 10 * y", ">=", 0.0)
        [(xy, _)] = toBotorchNonlinearConstraints([constraint], ["x", "y"])
        [(yx, _)] = toBotorchNonlinearConstraints([constraint], ["y", "x"])

        assert float(xy(_point(20.0, 1.0))) == 10.0
        assert float(yx(_point(20.0, 1.0))) == -199.0

    def test_no_constraints_gives_an_empty_list(self) -> None:
        assert toBotorchNonlinearConstraints([], ["x"]) == []


class TestConstants:
    def test_constants_can_be_referenced(self) -> None:
        [(function, _)] = toBotorchNonlinearConstraints(
            [NonlinearConstraint("var * x ** z", "<=", 50.0)],
            ["x", "z"],
            constants={"var": 2.0},
        )
        assert float(function(_point(3.0, 2.0))) == pytest.approx(32.0)

    def test_constants_broadcast_over_batches(self) -> None:
        [(function, _)] = toBotorchNonlinearConstraints(
            [NonlinearConstraint("k * x", "<=", 10.0)], ["x"], constants={"k": 2.0}
        )
        assert function(torch.tensor([[1.0], [4.0]], dtype=torch.double)).tolist() == [
            8.0,
            2.0,
        ]


class TestErrors:
    def test_unknown_name_is_reported_with_the_constraint_and_the_reason(self) -> None:
        with pytest.raises(
            ValueError, match=r"x \* w.*'w'.*Range parameters and numeric Fixed"
        ):
            toBotorchNonlinearConstraints(
                [NonlinearConstraint("x * w", "<=", 1.0)], ["x", "y"]
            )
