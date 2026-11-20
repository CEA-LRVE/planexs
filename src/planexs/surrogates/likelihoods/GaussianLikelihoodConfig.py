"""The configuration schema of the `gaussian_v1` likelihood."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class GaussianLikelihoodConfig(ComponentConfig):
    """The configuration of the `gaussian_v1` likelihood: a Gaussian observation noise.

    It is the stored form of the likelihood (what a configuration file says), and
    `GaussianLikelihoodBuilder.fromConfig` builds the builder from it (specification 5.6,
    5.12). Two settings, one for each way the noise can reach the model: a lower bound for a
    noise that is inferred, and a floor for a noise that is known.

    Attributes:
        type: The registered name, always `"gaussian_v1"`.
        noise_lower_bound: The smallest noise variance that fitting may infer, or `None` for
            the engine's own.
        min_observation_variance: The floor that a known observation variance is raised to.
    """

    type: Literal["gaussian_v1"] = Field(
        default="gaussian_v1",
        description=(
            "The registered name of this likelihood, always 'gaussian_v1'. It selects the "
            "'gaussian_v1' entry of the 'likelihood' registry, so that a stored likelihood "
            "says which one it is (specification 5.6). It is the only value the field "
            "accepts."
        ),
    )
    noise_lower_bound: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The smallest observation noise variance that fitting may infer, in the units of "
            "the outcome squared as the model sees it (after any outcome transform). It "
            "applies only when no observation noise is reported, so that the noise is a "
            "parameter of the model. It must be strictly positive. None leaves the choice to "
            "BoTorch: its default likelihood for an inferred noise, which keeps it above "
            "1e-4 and puts its own prior on it. A number replaces that floor and keeps "
            "BoTorch's prior. A lower floor lets the model explain the data as nearly "
            "noise-free, and a higher one stops it from ever trusting an observation more "
            "than that. The default is None, the choice of the legacy surrogate, so that a "
            "study that sets nothing fits as it did."
        ),
    )
    min_observation_variance: float = Field(
        default=1e-10,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "The floor that a known observation variance is raised to, in the units of the "
            "outcome squared as the model sees it (after any outcome transform). It applies "
            "only when observation noise is reported, so that the noise is fixed and not "
            "inferred. A deterministic observation, one reported with a standard deviation "
            "of 0, would give a variance of 0 and a singular covariance matrix that the fit "
            "cannot invert; the floor keeps it usable. 0 turns the floor off. GPyTorch also "
            "raises a fixed noise to a minimum of its own (`gpytorch.settings."
            "min_fixed_noise`: 1e-6 in double precision, 1e-4 in single precision), and the "
            "larger of the two wins: a floor below that minimum has no effect, and only a "
            "floor above it changes the fit. The default is 1e-10, the value of the legacy "
            "surrogate, which the engine's minimum has always dominated, so a study that "
            "sets nothing fits as it did."
        ),
    )
