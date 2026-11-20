"""Unit tests for the `bernoulli_v1` likelihood (roadmap task P3-11, specification 5.6, 6).

The declared need (`posterior:latent_gaussian`) is checked against the engine: an exact GP
refuses the likelihood, and a variational GP trains with it.
"""

from __future__ import annotations

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from likelihood_fakes import builderOf, makeContext
from pydantic import ValidationError

from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.likelihoods.BernoulliLikelihoodBuilder import (
    BernoulliLikelihoodBuilder,
)
from planexs.surrogates.likelihoods.BernoulliLikelihoodConfig import BernoulliLikelihoodConfig
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry

BERNOULLI = {"type": "bernoulli_v1"}


def separableData() -> tuple[torch.Tensor, torch.Tensor]:
    """Eight points on a line, labelled 0 below the middle and 1 above it."""
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    return train_x, (train_x > 0.5).double()


class TestTheRegisteredEntry:
    def test_it_models_binary_outcomes_and_needs_a_latent_gaussian_posterior(self) -> None:
        entry = likelihoodRegistry().lookup("bernoulli_v1")

        assert entry.supports == frozenset({"outcome_kind:binary"})
        assert entry.needs == frozenset({"posterior:latent_gaussian"})

    def test_a_mapping_builds_it_through_the_registry(self) -> None:
        assert isinstance(builderOf(BERNOULLI), BernoulliLikelihoodBuilder)

    def test_it_has_no_setting_and_refuses_an_unknown_one(self) -> None:
        assert BernoulliLikelihoodConfig().type == "bernoulli_v1"
        with pytest.raises(ValidationError):
            BernoulliLikelihoodConfig(link="logit")  # type: ignore[call-arg]
        with pytest.raises(ConfigError, match="link"):
            validateConfig(
                BernoulliLikelihoodConfig, {"type": "bernoulli_v1", "link": "logit"}
            )


class TestTheNativeLikelihood:
    def test_it_makes_a_bernoulli_likelihood(self) -> None:
        likelihood = builderOf(BERNOULLI).build(makeContext("binary"))

        assert isinstance(likelihood, BernoulliLikelihood)

    def test_a_label_is_true_with_the_probability_of_the_standard_normal(self) -> None:
        # The link is the probit: the probability is the standard normal cdf of the latent
        # function, 0.5 at 0 and about 0.8413 at 1.
        likelihood = builderOf(BERNOULLI).build(makeContext("binary"))

        probabilities = likelihood.forward(torch.tensor([0.0, 1.0], dtype=torch.double)).probs

        assert probabilities.tolist() == pytest.approx([0.5, 0.8413447460685429])

    def test_the_batch_shape_does_not_matter_for_a_likelihood_with_no_parameter(self) -> None:
        likelihood = builderOf(BERNOULLI).build(
            makeContext(
                "binary", "binary", batch_shape=(2,), runtime=RuntimeConfig(dtype="float32")
            )
        )

        assert isinstance(likelihood, BernoulliLikelihood)
        assert list(likelihood.parameters()) == []


class TestTheOutcomeKindsItSupports:
    def test_it_refuses_a_measured_quantity_and_names_a_likelihood_that_takes_one(
        self,
    ) -> None:
        with pytest.raises(ConfigError, match="cannot model the outcome 'y0'") as error:
            builderOf(BERNOULLI).build(makeContext("continuous"))

        assert error.value.component == "likelihood 'bernoulli_v1'"
        assert error.value.field_path == "outcomes[0]"
        assert "binary" in error.value.message
        assert error.value.suggestion is not None
        assert "gaussian_v1" in error.value.suggestion

    def test_it_refuses_a_mix_of_kinds_at_the_first_outcome_it_cannot_take(self) -> None:
        with pytest.raises(ConfigError) as error:
            builderOf(BERNOULLI).build(makeContext("binary", "continuous"))

        assert error.value.field_path == "outcomes[1]"


class TestNoObservationNoise:
    def test_a_label_has_no_noise_to_know(self) -> None:
        with pytest.raises(ConfigError, match="has none") as error:
            builderOf(BERNOULLI).build(makeContext("binary", noise=[0.1, 0.2]))

        assert error.value.component == "likelihood 'bernoulli_v1'"
        assert error.value.field_path == "noise_variance"
        assert error.value.suggestion is not None

    def test_it_does_not_declare_that_it_handles_a_noise(self) -> None:
        entry = likelihoodRegistry().lookup("bernoulli_v1")

        assert "noise:known" not in entry.supports
        assert "noise:inferred" not in entry.supports


class TestItsDeclaredNeed:
    def test_an_exact_gp_cannot_take_it(self) -> None:
        # The need is truthful: the engine itself refuses the likelihood for an exact GP.
        train_x, labels = separableData()
        likelihood = builderOf(BERNOULLI).build(makeContext("binary"))

        with pytest.raises(RuntimeError, match="Gaussian likelihoods"):
            SingleTaskGP(train_x, labels, likelihood=likelihood)

    # Labels are not standardized, and BoTorch's warning that they are not is about a
    # continuous outcome: it does not apply to the 0 and 1 of a label.
    @pytest.mark.filterwarnings("ignore:Data \\(outcome observations\\) is not standardized")
    def test_a_variational_gp_trains_with_it(self) -> None:
        # The need can be met: with approximate inference the likelihood trains, and the
        # evidence lower bound rises.
        train_x, labels = separableData()
        targets = labels.squeeze(-1)
        likelihood = builderOf(BERNOULLI).build(makeContext("binary"))
        model = SingleTaskVariationalGP(
            train_x, labels, likelihood=likelihood, inducing_points=train_x.clone()
        )
        elbo = VariationalELBO(model.likelihood, model.model, num_data=train_x.shape[0])
        model.train()

        with torch.no_grad():
            before = float(elbo(model.model(train_x), targets))
        fit_gpytorch_mll(elbo)
        model.eval()
        with torch.no_grad():
            after = float(elbo(model.model(train_x), targets))
            latent = model.model(train_x).mean

        assert after > before
        # The fitted latent function puts the two classes on opposite sides of each other.
        assert bool((latent[targets > 0.5].min() > latent[targets < 0.5].max()).item())

    def test_the_compatibility_check_refuses_it_next_to_a_surrogate_with_a_gaussian_posterior(
        self,
    ) -> None:
        entry = likelihoodRegistry().lookup("bernoulli_v1")
        exact_gp = DeclaredParticipant(
            "surrogate 'an_exact_gp'", supports=frozenset({"posterior:gaussian"})
        )

        with pytest.raises(CompatibilityError, match="posterior:latent_gaussian") as error:
            checkCompatibility([DeclaredParticipant.fromEntry("likelihood", entry), exact_gp])

        assert "likelihood 'bernoulli_v1'" in error.value.message

    def test_the_compatibility_check_accepts_it_with_approximate_inference(self) -> None:
        entry = likelihoodRegistry().lookup("bernoulli_v1")
        variational = DeclaredParticipant(
            "surrogate 'a_variational_gp'", supports=frozenset({"posterior:latent_gaussian"})
        )

        checkCompatibility([DeclaredParticipant.fromEntry("likelihood", entry), variational])
