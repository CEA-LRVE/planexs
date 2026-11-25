"""The configuration schema of the `variational_gp_classifier_v1` surrogate."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf


class VariationalGPClassifierConfig(ComponentConfig):
    """The configuration of the `variational_gp_classifier_v1` surrogate: an approximate
    Gaussian process for one binary outcome (specification 5.6, 3.4).

    It nests the two components that the surrogate is composed of and chooses by name, the
    covariance function (kind `kernel`) and the fitting procedure (kind `fitter`), and it holds
    the settings that are the surrogate's own: how many inducing points the approximation uses,
    and the two policy values about little data that every surrogate has (the fewest
    observations it can fit on and the count below which the fit is flagged). The likelihood is
    not a field: a classifier is a model of a label with a Bernoulli likelihood, read from the
    latent function by a probit, and another likelihood would be another surrogate.

    Attributes:
        type: The registered name, always `"variational_gp_classifier_v1"`.
        kernel: The covariance function of the latent function.
        fitter: The procedure that fits the model.
        n_inducing: The number of inducing points, or `None` for as many as observations.
        min_observations: The fewest observations that a model can be fitted on.
        low_data_threshold: The number of observations below which a fit is flagged low-data.
    """

    type: Literal["variational_gp_classifier_v1"] = Field(
        default="variational_gp_classifier_v1",
        description=(
            "The registered name of this surrogate, always 'variational_gp_classifier_v1'. It "
            "selects the 'variational_gp_classifier_v1' entry of the 'surrogate' registry, so "
            "that a stored surrogate says which one it is (specification 5.6). It is the "
            "only value the field accepts."
        ),
    )
    kernel: Annotated[ComponentConfig, ComponentOf("kernel")] = Field(
        default_factory=lambda: {"type": "botorch_default_v1"},
        validate_default=True,
        description=(
            "The covariance function of the latent function: a component of the 'kernel' "
            "registry, written as a mapping whose 'type' is the registered name of the kernel "
            "('botorch_default_v1', 'matern_v1' or 'rbf_v1') and whose other fields are that "
            "kernel's own settings. The default is 'botorch_default_v1', which leaves the "
            "choice to BoTorch: for a variational GP the library's own default, an RBF "
            "kernel with one lengthscale for each encoded column and a prior that keeps the "
            "fit usable on few labels, and one that follows the library from one version to "
            "the next. A kernel that is named here is used as it is, with an output scale "
            "of one: the latent function of a classifier is not standardized, and the scale "
            "of the labels is carried by the variational distribution and not by the "
            "kernel."
        ),
    )
    fitter: Annotated[ComponentConfig, ComponentOf("fitter")] = Field(
        default_factory=lambda: {"type": "variational_elbo_v1"},
        validate_default=True,
        description=(
            "The procedure that fits the parameters of the model: a component of the 'fitter' "
            "registry, written as a mapping whose 'type' is the registered name of the fitter "
            "and whose other fields are that fitter's own settings, such as the number of "
            "iterations and the learning rate. It must be a fitter of approximate models, "
            "which needs a latent Gaussian posterior (the compatibility check of a study "
            "refuses one that does not). The default is 'variational_elbo_v1', which "
            "maximizes the variational evidence lower bound, the objective that a model "
            "with a Bernoulli likelihood is fitted by."
        ),
    )
    n_inducing: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The number of inducing points of the variational approximation, in points, or "
            "null for as many as there are observations. The inducing points are the "
            "locations in the encoded space at which the latent function is summarized: "
            "the cost of a fit grows with the cube of their number and the quality of the "
            "approximation falls with fewer of them. With null they are the observed points "
            "themselves, which loses nothing and is cheap for the dozens or hundreds of "
            "observations of an experiment; a number smaller than the count of observations "
            "chooses up to that many of them greedily, to cover the latent function as well "
            "as it can, and fewer when the observations left out are already explained by the "
            "ones chosen. A number larger than the count of observations is reduced to it, "
            "and the fit status says so, as it does when fewer points were chosen than asked "
            "for. The points are always some of the observed inputs and the fit does not move "
            "them. It must be at least 1. The default is null, because the "
            "studies this library is for have few observations and an approximation that "
            "is exact in the number of points is the safest."
        ),
    )
    min_observations: int = Field(
        default=2,
        ge=1,
        description=(
            "The fewest observations that the model can be fitted on, in observations. With "
            "fewer the builder raises an error that says how many it has and how many it "
            "needs, and the study asks again after more trials have been completed. It must "
            "be at least 1. The default is 2, the smallest count for which there can be two "
            "labels to tell apart, and the same as the minimum of the other surrogates: the "
            "classifier is usable with a handful of labels, and says so through the "
            "threshold below instead of refusing to fit."
        ),
    )
    low_data_threshold: int = Field(
        default=10,
        ge=1,
        description=(
            "The number of observations below which the fit is flagged as low-data, in "
            "observations. A label carries at most one bit, so a boundary between two classes "
            "is learned from many fewer bits than a smooth outcome is learned from numbers: "
            "a surrogate fitted on fewer observations than this reports a fit status with "
            "its low-data flag set, which says that the probabilities it gives are not to be "
            "trusted yet (specification 3.11). It does not change the fit. It must be at "
            "least 1. The default is 10, a judgement and not an engine value: about the count "
            "at which a boundary in a few dimensions begins to be placed by the data and "
            "not by the prior. A study that reads its labels more slowly or in a larger "
            "space will want a larger value."
        ),
    )
