"""Unit tests for `BuiltSurrogate` (roadmap task P3-07, specification 5.6)."""

from __future__ import annotations

import dataclasses
import importlib

import pytest
from surrogate_fakes import FakeModel

from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus


def status() -> FitStatus:
    """A fit status."""
    return FitStatus(n_observations=4, is_low_data=True, warnings=("few points",))


class TestABuiltSurrogate:
    def test_it_holds_the_model_itself_and_its_status(self) -> None:
        model = FakeModel()
        fit = status()

        built = BuiltSurrogate(model=model, status=fit)  # type: ignore[arg-type]

        # The native object, not a copy: it is the model the policy really uses (5.13).
        assert built.model is model
        assert built.status is fit

    def test_it_can_be_built_positionally_in_the_order_model_then_status(self) -> None:
        model = FakeModel()

        built = BuiltSurrogate(model, status())  # type: ignore[arg-type]

        assert built.model is model

    @pytest.mark.parametrize("field", ["model", "status"])
    def test_it_is_frozen(self, field: str) -> None:
        built = BuiltSurrogate(FakeModel(), status())  # type: ignore[arg-type]

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(built, field, None)

    def test_two_results_are_equal_when_they_hold_the_same_model_and_an_equal_status(
        self,
    ) -> None:
        model = FakeModel()

        assert BuiltSurrogate(model, status()) == BuiltSurrogate(model, status())  # type: ignore[arg-type]
        assert BuiltSurrogate(model, status()) != BuiltSurrogate(FakeModel(), status())  # type: ignore[arg-type]

    def test_importing_it_does_not_import_a_model_class(self) -> None:
        # The model is only a type: no concrete class of the engine is named (5.6), and the
        # result works for a model of a plugin that satisfies the contract.
        module = importlib.import_module("planexs.surrogates.BuiltSurrogate")

        assert "Model" not in vars(module)
