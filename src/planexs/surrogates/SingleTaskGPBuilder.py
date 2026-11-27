"""The `single_task_gp_v1` surrogate: an exact GP for the outcomes, as independent outputs."""

from __future__ import annotations

import logging
from typing import Any

import torch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import Kernel
from torch import Tensor

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.ConfigError import ConfigError
from planexs.core.ModelNotReadyError import ModelNotReadyError
from planexs.core.Registry import Registry
from planexs.encoding.torch_runtime import torchDevice, torchDtype
from planexs.surrogates.AbstractSurrogateBuilder import AbstractSurrogateBuilder
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry
from planexs.surrogates.likelihoods.LikelihoodBuildContext import LikelihoodBuildContext
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry
from planexs.surrogates.outcome_transforms.buildOutcomeTransform import buildOutcomeTransform
from planexs.surrogates.registerSurrogate import registerSurrogate
from planexs.surrogates.SingleTaskGPConfig import SingleTaskGPConfig
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext
from planexs.surrogates.surrogateRegistry import surrogateRegistry

logger = logging.getLogger(__name__)


@registerSurrogate(
    "single_task_gp_v1",
    config=SingleTaskGPConfig,
    description=(
        "An exact GP for the outcomes, as independent outputs, with its kernel, likelihood, "
        "outcome transforms and fitter chosen in configuration."
    ),
    supports=frozenset(
        {
            "posterior:gaussian",
            "posterior:marginal_normal",
            "outputs:single",
            "outputs:multi",
            "noise:known",
            "noise:inferred",
            "outcome_kind:continuous",
        }
    ),
)
class SingleTaskGPBuilder(AbstractSurrogateBuilder[SingleTaskGPConfig]):
    """Fits an exact Gaussian process to the observations, one independent output for each
    outcome (specification 5.6, 6).

    It is the surrogate that the legacy `buildSurrogateModel` was, with each of its hard-wired
    choices a component of the configuration (specification 3.3): the kernel, the likelihood,
    the outcome transforms and the fitting procedure are nested components, and the number of
    observations that a fit needs and the threshold of a low-data fit are fields. With the
    defaults it builds what the legacy function built and fits it as it did (BoTorch's own
    kernel and priors, standardized outcomes, a MAP fit), and a test compares the two.

    **The model.** The outcomes are independent: there is no correlation between them.

    - With the kernel left to the engine (`botorch_default_v1`) the outcomes are the outputs of
      one batched `SingleTaskGP`, which is what the engine does best and what the legacy
      surrogate built.
    - With an explicit kernel and one outcome, one `SingleTaskGP` with that kernel.
    - With an explicit kernel and several outcomes, a `ModelListGP` of one `SingleTaskGP` for
      each outcome, each with a kernel of its own, because an explicit kernel has no batch
      shape and cannot be the covariance of one batched model (see `MaternKernelBuilder`).

    **Noise.** When the context holds a noise standard deviation for every value, it is squared
    to a variance and brought to the space in which the model trains, by a transform of the
    configured chain that learns the same statistics from the same targets as the model's own
    (a standardized outcome has its noise divided by the square of the standard deviation, a
    logarithm gives it the delta method). The likelihood then fixes the noise to it. The model
    is given a likelihood and never a `train_Yvar`, because a model that has a custom
    likelihood ignores `train_Yvar`. When the context holds no noise, the likelihood infers it.
    Partial knowledge is settled before the context is built, by the noise policy that the
    configuration holds and `trialsToTensors` reads (specification 5.4).

    **Fit status.** The model is fitted by the configured fitter, which draws every random
    number from the generator of the context, and is returned in evaluation mode. The status
    holds the number of observations, and says the fit is low-data, in a warning of its own and
    in a logged one (specification 8), when there are fewer than `low_data_threshold`: the
    uncertainty of such a model is mostly its priors (specification 3.11).

    **What it refuses.** Fewer than `min_observations` observations is `ModelNotReadyError`,
    which a later ask can cure. An outcome whose kind the surrogate does not declare
    (`outcome_kind:continuous` only) is a `ConfigError` that names the outcome, as the
    likelihood's own check does, so a label is never fitted by an exact GP.
    """

    def build(self, context: SurrogateBuildContext) -> BuiltSurrogate:
        """Fits a model on the observations of the context.

        Args:
            context: The problem, the encoder, the data as tensors, the generator and the
                runtime settings. Read-only.

        Returns:
            The fitted model, in evaluation mode, and its fit status.

        Raises:
            ModelNotReadyError: if the context holds fewer than `min_observations`
                observations.
            ConfigError: if an outcome is of a kind that the surrogate does not support, or a
                nested component refuses what it is given.
            FitterError: if the fit fails.
        """
        self._requireSupportedOutcomes(context)
        self._requireEnoughObservations(context)
        model = self._buildModel(context)
        self._nested(fitterRegistry(), self.config.fitter).fit(
            model, generator=context.generator
        )
        return BuiltSurrogate(model=model, status=self._statusOf(context))

    def _requireSupportedOutcomes(self, context: SurrogateBuildContext) -> None:
        """Refuses an outcome whose kind the registry entry does not declare as supported.

        Args:
            context: The context whose problem is checked.

        Raises:
            ConfigError: if an outcome is of a kind that is not among the `outcome_kind:*`
                tags of the entry. The message lists the kinds the surrogate supports and the
                registered surrogates that support the kind of the outcome.
        """
        registry = surrogateRegistry()
        name = self.config.type
        supported = registry.lookup(name).supports
        for position, outcome in enumerate(context.problem.outcomes):
            tag = outcome.kind().capability_tag
            if tag in supported:
                continue
            kinds = sorted(
                tag_name.partition(":")[2]
                for tag_name in supported
                if tag_name.partition(":")[0] == tag.partition(":")[0]
            )
            alternatives = [
                other for other in registry.names() if tag in registry.lookup(other).supports
            ]
            raise ConfigError(
                f"The surrogate {name!r} cannot model the outcome {outcome.name!r}: it is of "
                f"the kind {outcome.type!r}, and the surrogate supports the kinds "
                f"{', '.join(kinds) or 'none'}.",
                component=f"surrogate {name!r}",
                field_path=f"outcomes[{position}]",
                suggestion=(
                    f"use one of the surrogates that support {outcome.type!r}: "
                    f"{', '.join(alternatives)}"
                    if alternatives
                    else f"register a surrogate that supports the kind {outcome.type!r}"
                ),
            )

    def _requireEnoughObservations(self, context: SurrogateBuildContext) -> None:
        """Refuses a context that holds fewer observations than `min_observations`.

        Args:
            context: The context whose observations are counted.

        Raises:
            ModelNotReadyError: if it holds fewer. It is a lack of data and not an invalid
                request: asking again once more trials are completed can succeed.
        """
        needed = self.config.min_observations
        if context.n_observations < needed:
            raise ModelNotReadyError(
                f"The surrogate {self.config.type!r} needs at least {needed} observation(s) "
                f"to fit a model, and the context holds {context.n_observations}.",
                component=f"surrogate {self.config.type!r}",
                field_path="min_observations",
                suggestion="complete more trials and ask again, or lower min_observations",
            )

    def _buildModel(self, context: SurrogateBuildContext) -> Model:
        """Builds the unfitted native model: batched, single or a list, as the kernel says.

        Args:
            context: The context to build the model for.

        Returns:
            One model for all the outcomes if the kernel is left to the engine or there is one
            outcome, and a `ModelListGP` of one model for each outcome otherwise.
        """
        n_outcomes = len(context.problem.outcomes)
        kernel_builder = self._nested(kernelRegistry(), self.config.kernel)
        n_columns = context.encoder.n_columns
        first = kernel_builder.build(n_columns)
        if first is None or n_outcomes == 1:
            return self._makeModel(context, list(range(n_outcomes)), first)
        return ModelListGP(
            *[
                self._makeModel(
                    context,
                    [column],
                    first if column == 0 else kernel_builder.build(n_columns),
                )
                for column in range(n_outcomes)
            ]
        )

    def _makeModel(
        self, context: SurrogateBuildContext, columns: list[int], kernel: Kernel | None
    ) -> SingleTaskGP:
        """Builds one `SingleTaskGP` for the outcomes at `columns`, as independent outputs.

        Args:
            context: The context that holds the data.
            columns: The columns of the outcomes of the model, in the order of its outputs.
            kernel: The kernel of the model, or `None` to leave the choice to the engine.

        Returns:
            The model, not fitted. Its likelihood holds the known noise, if any, and its
            outcome transform is a new one that nothing else holds.
        """
        n_outputs = len(columns)
        likelihood = self._nested(likelihoodRegistry(), self.config.likelihood).build(
            LikelihoodBuildContext(
                outcomes=tuple(context.problem.outcomes[column] for column in columns),
                runtime=context.runtime,
                batch_shape=torch.Size([n_outputs]) if n_outputs > 1 else torch.Size(),
                noise_variance=self._noiseInModelSpace(context, columns),
            )
        )
        options: dict[str, Any] = {}
        if kernel is not None:
            options["covar_module"] = kernel.to(
                dtype=torchDtype(context.runtime), device=torchDevice(context.runtime)
            )
        # `outcome_transform=None` means "do not rescale", and leaving the argument out would
        # make the engine standardize: it is always passed (see `buildOutcomeTransform`).
        return SingleTaskGP(
            train_X=context.train_inputs,
            train_Y=context.train_outcomes[:, columns],
            likelihood=likelihood,
            outcome_transform=buildOutcomeTransform(self.config.outcome_transform, n_outputs),
            **options,
        )

    def _noiseInModelSpace(
        self, context: SurrogateBuildContext, columns: list[int]
    ) -> Tensor | None:
        """The known noise variances, in the space in which the model trains.

        The standard deviations of the context are squared, and sent through a transform of
        the configured chain of their own, which learns the same statistics from the same
        targets as the model's transform will. The result has the layout of a likelihood
        context: `(n,)` for one output, `(outputs, n)` for several.

        Args:
            context: The context that holds the noise.
            columns: The columns of the outcomes of the model.

        Returns:
            The variances, or `None` if the context holds no noise: it is then inferred.
        """
        deviations = context.train_noise_std
        if deviations is None:
            return None
        variance = deviations[:, columns] ** 2
        transform = buildOutcomeTransform(self.config.outcome_transform, len(columns))
        if transform is not None:
            transform.train()
            with torch.no_grad():
                _, transformed = transform(context.train_outcomes[:, columns], variance)
            assert transformed is not None  # a transform given a variance returns one
            variance = transformed
        return variance.transpose(0, 1) if len(columns) > 1 else variance.reshape(-1)

    def _statusOf(self, context: SurrogateBuildContext) -> FitStatus:
        """How the model was fitted: on how many observations, and whether that is few.

        A low-data fit is reported in the status and logged as a warning (specification 8).

        Args:
            context: The context the model was fitted on.

        Returns:
            The fit status.
        """
        n_observations = context.n_observations
        threshold = self.config.low_data_threshold
        if n_observations >= threshold:
            return FitStatus(n_observations=n_observations, is_low_data=False)
        message = (
            f"Fitted on only {n_observations} observation(s), fewer than {threshold}: the "
            "uncertainty estimates of the model are low-confidence."
        )
        logger.warning("The surrogate %r: %s", self.config.type, message)
        return FitStatus(n_observations=n_observations, is_low_data=True, warnings=(message,))

    @staticmethod
    def _nested(registry: Registry[ComponentEntry], config: ComponentConfig) -> Any:
        """Makes the component that a nested configuration names, through its registry.

        Args:
            registry: The registry of the kind of the component.
            config: The validated configuration of the component.

        Returns:
            What the builder of the registry entry makes from the configuration.
        """
        return registry.lookup(config.type).builder(config)
