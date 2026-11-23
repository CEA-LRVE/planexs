"""Unit tests for `AbstractFitter` (roadmap task P3-13, specification 5.6, 6).

The contract is tested with fakes, before and beside the real fitter: what `fit` checks, what
it guarantees about the model it leaves, and that a subclass only has to provide
`fitParameters`.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from botorch.models.model import Model
from fitter_fakes import RecordingFitter, fakeConfig, generatorOf, toyModel

from planexs.core.ConfigError import ConfigError
from planexs.surrogates.fitters.AbstractFitter import AbstractFitter
from planexs.surrogates.fitters.FitterError import FitterError
from planexs.surrogates.fitters.MapFitter import MapFitter

CONFIG = fakeConfig("fake_fitter_v1")()


class SetsAParameter(AbstractFitter[Any]):
    """A fitter that leaves a hyperparameter at a value that it was told to."""

    def __init__(self, config: Any, value: float) -> None:
        super().__init__(config)
        self._value = value

    def fitParameters(self, model: Model, generator: torch.Generator) -> None:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(self._value)
                break


class TestTheContract:
    def test_the_contract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractFitter(CONFIG)  # type: ignore[abstract]

    def test_a_fitter_that_does_not_define_fit_parameters_cannot_be_instantiated(self) -> None:
        class Incomplete(AbstractFitter[Any]):
            pass

        with pytest.raises(TypeError, match="fitParameters"):
            Incomplete(CONFIG)  # type: ignore[abstract]

    def test_the_least_a_fitter_defines_is_fit_parameters(self) -> None:
        assert isinstance(RecordingFitter(CONFIG), AbstractFitter)

    def test_the_entry_point_is_not_overridden_by_the_real_fitter(self) -> None:
        # `fit` is the one entry point: it checks, so that every fitter is held to the rules.
        assert MapFitter.fit is AbstractFitter.fit

    def test_from_config_makes_the_fitter_that_keeps_its_configuration(self) -> None:
        fitter = RecordingFitter.fromConfig(CONFIG)

        assert isinstance(fitter, RecordingFitter)
        assert fitter.config is CONFIG


class TestWhatFitGivesToTheSubclass:
    def test_it_hands_over_the_very_model_and_generator_it_was_given(self) -> None:
        fitter = RecordingFitter(CONFIG)
        model, generator = toyModel(), generatorOf(7)

        fitter.fit(model, generator=generator)

        assert len(fitter.received) == 1
        received_model, received_generator = fitter.received[0]
        assert received_model is model
        assert received_generator is generator

    def test_it_returns_nothing_because_the_model_is_changed_in_place(self) -> None:
        assert RecordingFitter(CONFIG).fit(toyModel(), generator=generatorOf(0)) is None  # type: ignore[func-returns-value]

    def test_it_leaves_the_model_in_evaluation_mode_whatever_the_fit_left(self) -> None:
        model = toyModel()
        fitter = RecordingFitter(
            CONFIG
        )  # leaves the model in training mode, as a lazy fit does

        fitter.fit(model, generator=generatorOf(0))

        assert fitter.received[0][0].training is False
        assert model.training is False

    def test_the_generator_is_a_keyword_argument(self) -> None:
        with pytest.raises(TypeError):
            RecordingFitter(CONFIG).fit(toyModel(), generatorOf(0))  # type: ignore[misc]


class TestWhatFitRefuses:
    def test_something_that_is_not_a_botorch_model(self) -> None:
        fitter = RecordingFitter(CONFIG)
        not_a_model: Any = torch.nn.Linear(2, 1)

        with pytest.raises(FitterError, match="not a BoTorch model") as caught:
            fitter.fit(not_a_model, generator=generatorOf(0))

        assert caught.value.component == "fitter 'fake_fitter_v1'"
        assert caught.value.field_path == "model"
        assert caught.value.suggestion
        assert "Linear" in caught.value.message
        assert fitter.received == []

    @pytest.mark.parametrize("bad", [None, 3, "seed", torch.manual_seed])
    def test_a_generator_that_is_not_a_torch_generator(self, bad: Any) -> None:
        fitter = RecordingFitter(CONFIG)

        with pytest.raises(ConfigError, match=r"seeded torch\.Generator") as caught:
            fitter.fit(toyModel(), generator=bad)

        assert caught.value.field_path == "generator"
        assert caught.value.component == "fitter 'fake_fitter_v1'"
        assert fitter.received == []

    def test_a_fit_that_leaves_a_parameter_that_is_not_finite(self) -> None:
        model = toyModel()
        # `SetsAParameter` changes the first parameter, and the error must name that one.
        first_parameter = next(name for name, _ in model.named_parameters())

        with pytest.raises(FitterError, match="not finite") as caught:
            SetsAParameter(CONFIG, float("nan")).fit(model, generator=generatorOf(0))

        assert caught.value.component == "fitter 'fake_fitter_v1'"
        assert caught.value.field_path == "model"
        assert caught.value.suggestion
        assert repr(first_parameter) in caught.value.message

    def test_a_fit_that_leaves_an_infinite_parameter(self) -> None:
        with pytest.raises(FitterError, match="not finite"):
            SetsAParameter(CONFIG, float("inf")).fit(toyModel(), generator=generatorOf(0))

    def test_a_fit_that_leaves_finite_parameters_is_accepted(self) -> None:
        SetsAParameter(CONFIG, 0.0).fit(toyModel(), generator=generatorOf(0))


class TestObservability:
    def test_a_fit_says_at_debug_level_which_model_and_fitter(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(
            logging.DEBUG, logger="planexs.surrogates.fitters.AbstractFitter"
        ):
            RecordingFitter(CONFIG).fit(toyModel(), generator=generatorOf(0))

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "SingleTaskGP" in message and "fake_fitter_v1" in message for message in messages
        )
