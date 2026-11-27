"""The configuration schema of the `single_task_gp_v1` surrogate."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf
from planexs.surrogates.NoisePolicyConfig import NoisePolicyConfig


class SingleTaskGPConfig(ComponentConfig):
    """The configuration of the `single_task_gp_v1` surrogate: an exact Gaussian process whose
    kernel, likelihood, outcome transforms and fitting procedure are chosen by name.

    It is the stored form of the surrogate (what a configuration file says), and
    `SingleTaskGPBuilder.fromConfig` builds the builder from it (specification 5.6, 5.12). The
    four choices that a reasonable user could want to take differently (specification 3.3) are
    nested components of the kinds `kernel`, `likelihood`, `outcome_transform` and `fitter`,
    each written as a mapping whose `type` is a registered name. Their defaults are written as
    mappings too, so that this schema does not import the components that it names, and are
    resolved like any other value when the configuration is validated: with nothing set, the
    surrogate is the one that the legacy `buildSurrogateModel` built (BoTorch's own kernel, a
    Gaussian likelihood with its noise inferred or known, standardized outcomes, a MAP fit).

    The three other settings are the policy values that the legacy surrogate hid in its code
    (specification 3.4): how many observations a fit needs, below how many it is called
    low-data, and what to do with partial knowledge of noise.

    Attributes:
        type: The registered name, always `"single_task_gp_v1"`.
        kernel: The kernel of the process.
        likelihood: The likelihood of the observations.
        outcome_transform: The transforms applied to the outcomes before the fit, in order.
        fitter: The procedure that fits the hyperparameters.
        min_observations: The fewest observations from which a model is fitted.
        low_data_threshold: The number of observations below which a fit is flagged low-data.
        noise_policy: What to do when only some observations give a noise standard deviation.
    """

    type: Literal["single_task_gp_v1"] = Field(
        default="single_task_gp_v1",
        description=(
            "The registered name of this surrogate, always 'single_task_gp_v1'. It selects "
            "the 'single_task_gp_v1' entry of the 'surrogate' registry, so that a stored "
            "surrogate says which one it is (specification 5.6). It is the only value the "
            "field accepts."
        ),
    )
    kernel: Annotated[ComponentConfig, ComponentOf("kernel")] = Field(
        default_factory=lambda: {"type": "botorch_default_v1"},
        validate_default=True,
        description=(
            "The kernel (covariance function) of the Gaussian process: the component of the "
            "'kernel' registry to use, written as a mapping whose 'type' is its registered "
            "name and whose other fields are that kernel's own settings. The default is "
            "'botorch_default_v1', which names no kernel and leaves the choice to BoTorch, "
            "whose default carries priors that keep the fit usable on a handful of "
            "observations (specification 8). With an explicit kernel and several outcomes the "
            "surrogate builds one model for each outcome, each with a kernel of its own, "
            "because an explicit kernel has no batch shape: the outcomes are then still "
            "independent, and are fitted one after the other."
        ),
    )
    likelihood: Annotated[ComponentConfig, ComponentOf("likelihood")] = Field(
        default_factory=lambda: {"type": "gaussian_v1"},
        validate_default=True,
        description=(
            "The likelihood of the observations: the component of the 'likelihood' registry "
            "to use, written as a mapping with the registered name as 'type'. An exact "
            "Gaussian process takes a Gaussian likelihood only, so the default is "
            "'gaussian_v1'. It infers the noise when no observation reports one, and fixes it "
            "to the reported noise when every observation does (see 'noise_policy'). A "
            "likelihood that does not support the kind of an outcome refuses it with an error "
            "that names the outcome."
        ),
    )
    outcome_transform: list[Annotated[ComponentConfig, ComponentOf("outcome_transform")]] = (
        Field(
            default_factory=lambda: [{"type": "standardize_v1"}],
            validate_default=True,
            description=(
                "The transforms that rescale the outcomes before the fit, applied in the "
                "order given, each a component of the 'outcome_transform' registry written as "
                "a mapping with the registered name as 'type'. The model is fitted to the "
                "rescaled outcomes and its predictions are brought back to their units. The "
                "default is a list of 'standardize_v1', which centres each outcome and "
                "divides it by its standard deviation: the kernel carries no output scale, so "
                "a signal variance of one is what the covariance needs. An empty list (or "
                "'none_v1') fits the raw outcomes, which suits outcomes that are already on "
                "such a scale. The order matters when transforms do not commute: put 'log_v1' "
                "before 'standardize_v1'."
            ),
        )
    )
    fitter: Annotated[ComponentConfig, ComponentOf("fitter")] = Field(
        default_factory=lambda: {"type": "map_v1"},
        validate_default=True,
        description=(
            "The procedure that fits the hyperparameters of the model to the data: the "
            "component of the 'fitter' registry to use, written as a mapping with the "
            "registered name as 'type'. The default is 'map_v1', which maximizes the marginal "
            "likelihood together with the engine's priors (a MAP fit), which is what keeps "
            "the fit usable on few points. It draws its random numbers from the generator of "
            "the build, so the fit is reproducible."
        ),
    )
    min_observations: int = Field(
        default=2,
        ge=1,
        description=(
            "The fewest observations from which a model is fitted, in observations. A build "
            "that is given fewer raises 'ModelNotReadyError', which is a lack of data and not "
            "an invalid request: asking again once more trials have been completed can "
            "succeed. It must be at least 1, because a model cannot be fitted on no data. The "
            "default is 2, the value of the legacy surrogate: a single observation can be "
            "fitted, but it has no spread to standardize and says nothing about the noise, so "
            "a model fitted on it is a prior with one point pinned. Raise it to make a study "
            "use its design for longer."
        ),
    )
    low_data_threshold: int = Field(
        default=5,
        ge=0,
        description=(
            "The number of observations below which the fit is flagged low-data, in "
            "observations: a fit on fewer is reported by its fit status (and logged as a "
            "warning) as having uncertainty estimates that are low-confidence, which "
            "specification 3.11 requires instead of hiding it. The fit is still made. The "
            "default is 5, the value of the legacy surrogate: with fewer points than that, "
            "the priors on the lengthscales and the noise carry more of the fit than the data "
            "does. 0 never flags a fit. It is independent of 'min_observations', and a value "
            "at or below it flags nothing."
        ),
    )
    noise_policy: NoisePolicyConfig = Field(
        default_factory=NoisePolicyConfig,
        description=(
            "What to do when the observations give a noise standard deviation for only some "
            "of the values that can have one: the settings of 'NoisePolicyConfig', described "
            "there. A Gaussian process cannot mix fixed and inferred noise in one fit. It is "
            "read where the history becomes tensors (`trialsToTensors`), before the build, "
            "and is kept here so that the surrogate's configuration says everything that "
            "decides what it was fitted on and is stored with the study. The default is the "
            "policy's own: ignore the known deviations and infer the noise, with a warning."
        ),
    )
