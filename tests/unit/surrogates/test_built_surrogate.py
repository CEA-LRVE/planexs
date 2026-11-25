"""Unit tests for `BuiltSurrogate` (roadmap task P3-07, specification 5.6)."""

from __future__ import annotations

import dataclasses
import importlib

import pytest
from surrogate_fakes import FakeModel

from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.probitLink import probitLink


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

    def test_it_has_no_link_unless_it_is_given_one(self) -> None:
        assert BuiltSurrogate(FakeModel(), status()).link is None  # type: ignore[arg-type]

    def test_it_holds_the_link_that_the_builder_gives(self) -> None:
        link = probitLink()

        built = BuiltSurrogate(FakeModel(), status(), link=link)  # type: ignore[arg-type]

        assert built.link is link

    def test_the_link_is_the_third_field_and_after_the_two_that_were_always_there(
        self,
    ) -> None:
        names = [field.name for field in dataclasses.fields(BuiltSurrogate)]

        assert names == ["model", "status", "link"]

    def test_a_built_surrogate_with_a_link_is_not_equal_to_one_without(self) -> None:
        model = FakeModel()

        assert BuiltSurrogate(model, status(), link=probitLink()) != BuiltSurrogate(  # type: ignore[arg-type]
            model,
            status(),  # type: ignore[arg-type]
        )

    @pytest.mark.parametrize("field", ["model", "status", "link"])
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
