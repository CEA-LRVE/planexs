"""Unit tests for the `none_v1` outcome transform (roadmap task P3-12, specification 5.6).

`none_v1` makes no transform: it returns `None`. What makes that worth a test of its own is
what `None` means to BoTorch, which is not what an omitted argument means.
"""

from __future__ import annotations

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from outcome_transform_fakes import fitModel, toyData

import planexs.surrogates  # noqa: F401
from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.surrogates.outcome_transforms.NoneOutcomeTransformBuilder import (
    NoneOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.NoneOutcomeTransformConfig import (
    NoneOutcomeTransformConfig,
)
from planexs.surrogates.outcome_transforms.outcomeTransformRegistry import (
    outcomeTransformRegistry,
)

# The model of a raw outcome of order ten is not standardized, and BoTorch warns about it.
pytestmark = pytest.mark.filterwarnings("ignore::botorch.exceptions.warnings.InputDataWarning")


def build(n_outcomes: int = 1) -> None:
    return NoneOutcomeTransformBuilder(NoneOutcomeTransformConfig()).build(n_outcomes)  # type: ignore[func-returns-value]


class TestTheTransform:
    @pytest.mark.parametrize("n_outcomes", [1, 2, 40])
    def test_it_makes_no_transform_whatever_the_number_of_outcomes(
        self, n_outcomes: int
    ) -> None:
        assert build(n_outcomes) is None

    def test_it_has_no_setting_but_its_type(self) -> None:
        assert set(NoneOutcomeTransformConfig.model_fields) == {"type"}

    def test_a_setting_that_does_not_exist_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="scale"):
            resolveComponent(outcomeTransformRegistry(), {"type": "none_v1", "scale": 2.0})

    def test_the_bad_number_of_outcomes_of_every_transform_is_refused_here_too(self) -> None:
        with pytest.raises(ConfigError, match="whole number of one or more"):
            NoneOutcomeTransformBuilder(NoneOutcomeTransformConfig()).build(0)


class TestWhatNoneMeansToBoTorch:
    """The reason the docstrings insist that the result be passed to the model as it is.

    These pin BoTorch's behaviour: if one fails after an upgrade of the engine, the contract of
    the kind (`None` is "do not rescale", an omitted argument is "standardize") has to be read
    again.
    """

    def test_a_model_given_no_transform_argument_standardizes_its_outcomes(self) -> None:
        train_x, train_y, _, _ = toyData()

        model = SingleTaskGP(train_x, train_y)

        assert isinstance(model.outcome_transform, Standardize)

    def test_a_model_given_none_does_not_rescale_its_outcomes(self) -> None:
        train_x, train_y, _, _ = toyData()

        model = SingleTaskGP(train_x, train_y, outcome_transform=build())

        # BoTorch does not even set the attribute for `None`: it is absent, not `None`, so code
        # that inspects a model must read it with `getattr(model, "outcome_transform", None)`.
        assert not hasattr(model, "outcome_transform")
        assert torch.equal(model.train_targets, train_y.squeeze(-1))


class TestOnAToyModel:
    # Measured on this toy: the posterior mean is within 0.12 of the observations at the
    # training inputs, whose standard deviation is 2.6. The tolerance is 10% of that
    # deviation, as for the other transforms, plus the margin that the unscaled fit needs.
    TOLERANCE = 0.35

    def test_the_predictions_are_those_of_a_model_that_was_given_none(self) -> None:
        train_x, train_y, test_x, _ = toyData()
        plain = SingleTaskGP(train_x, train_y, outcome_transform=None)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(plain.likelihood, plain))
        plain.eval()

        ours = fitModel(build(), train_x, train_y)

        with torch.no_grad():
            expected, actual = plain.posterior(test_x), ours.posterior(test_x)
        assert torch.allclose(actual.mean, expected.mean, atol=1e-10)
        assert torch.allclose(actual.variance, expected.variance, atol=1e-10)

    def test_the_predictions_are_in_the_units_of_the_outcome_with_nothing_to_undo(
        self,
    ) -> None:
        train_x, train_y, _, _ = toyData()
        model = fitModel(build(), train_x, train_y)

        with torch.no_grad():
            posterior = model.posterior(train_x)

        assert float((posterior.mean - train_y).abs().max()) < self.TOLERANCE

    def test_it_is_not_the_model_that_the_engine_builds_when_it_is_given_nothing(self) -> None:
        # Leaving the argument out would have been a different model: the whole point.
        train_x, train_y, test_x, _ = toyData()
        default = SingleTaskGP(train_x, train_y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(default.likelihood, default))
        default.eval()
        ours = fitModel(build(), train_x, train_y)

        with torch.no_grad():
            default_mean, our_mean = (
                default.posterior(test_x).mean,
                ours.posterior(test_x).mean,
            )

        assert not torch.allclose(default_mean, our_mean, atol=1e-3)
