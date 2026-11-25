"""The `variational_gp_classifier_v1` surrogate: an approximate GP for one binary outcome."""

from __future__ import annotations

import warnings

from botorch.exceptions.warnings import InputDataWarning
from botorch.models.approximate_gp import SingleTaskVariationalGP
from gpytorch.likelihoods import Likelihood
from torch import Tensor

from planexs.core.ConfigError import ConfigError
from planexs.core.ModelNotReadyError import ModelNotReadyError
from planexs.surrogates.AbstractSurrogateBuilder import AbstractSurrogateBuilder
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry
from planexs.surrogates.likelihoods.LikelihoodBuildContext import LikelihoodBuildContext
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry
from planexs.surrogates.probitLink import probitLink
from planexs.surrogates.registerSurrogate import registerSurrogate
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext
from planexs.surrogates.SurrogateContextError import SurrogateContextError
from planexs.surrogates.VariationalGPClassifierConfig import VariationalGPClassifierConfig

#: The registered name of this surrogate, which its errors are about.
_NAME = "variational_gp_classifier_v1"

#: The likelihood of a classifier. It is the identity of this surrogate and not a choice of
#: the study: a Bernoulli likelihood read by a probit is what makes the model a classifier, and
#: what the link of its predictions is, so another likelihood is another surrogate (a new
#: name, specification 3.7). It is still obtained from the `likelihood` registry and not
#: imported (specification 3.10).
_LIKELIHOOD = "bernoulli_v1"


@registerSurrogate(
    _NAME,
    config=VariationalGPClassifierConfig,
    description=(
        "An approximate GP classifier for one binary outcome: a Bernoulli likelihood read "
        "from a latent function by a probit, fitted by variational inference."
    ),
    supports=frozenset(
        {
            "posterior:latent_gaussian",
            "posterior:marginal_normal",
            "outputs:single",
            "outcome_kind:binary",
            "link:probit",
        }
    ),
)
class VariationalGPClassifierBuilder(AbstractSurrogateBuilder[VariationalGPClassifierConfig]):
    """Fits an approximate Gaussian process to the labels of one binary outcome (specification
    5.6; FR20, FR24).

    The model is BoTorch's `SingleTaskVariationalGP`, which the installed BoTorch (0.18) offers
    for exactly this: a latent function with a Gaussian process prior, a variational
    distribution over its values at the inducing points, and the likelihood of the labels given
    by the `bernoulli_v1` component of the `likelihood` kind. Nothing had to be built on
    GPyTorch's `ApproximateGP` directly. The pieces that are chosen are the nested components
    of the configuration: the `kernel` is built by the kernel registry (the engine's default
    when it says so), and the `fitter` of the `fitter` kind fits the model (by default
    `variational_elbo_v1`, which draws from the generator of the context and leaves the model
    in evaluation mode).

    **The data.** The problem must have exactly one outcome, of a kind that the likelihood
    supports (binary): the likelihood builder refuses another kind and names it. A surrogate
    for several outcomes would choose how to model them (specification 5.6) and declares
    `outputs:single` because this one does not. The labels are `0.0` and `1.0`, as the
    observations store them, and a value that is neither is refused. They are not standardized:
    a label has no scale to remove.

    **The inducing points.** By default they are the training inputs, one for each observation
    (`n_inducing` unset), which loses nothing and is cheap for the number of observations of an
    experiment. A smaller `n_inducing` has BoTorch choose up to that many of them among the
    training inputs by greedy variance reduction: the selection stops early when the
    observations left out are explained by those chosen (BoTorch's pivoted Cholesky ends when
    the remaining variance is below 1e-6), and the fit status then says how many were chosen.
    They are a copy of the inputs and **are not learnt**, which departs from BoTorch's default
    on purpose: with one inducing point at each observation the variational bound cannot be
    made tighter by moving them, and learning them only moved them (on the test problem by up
    to 1.5 encoded units, out of a unit cube, for the same accuracy), so that "the inducing
    points are the training inputs" stopped being true after the fit. Left where they are,
    every inducing point of the fitted model is one of the observed inputs, the fit has fewer
    parameters, and the training data of the context is never touched.

    **What it returns.** The fitted model, in evaluation mode, and a `FitStatus`: the number of
    observations, the low-data flag (fewer observations than `low_data_threshold`) and the
    warnings that the data deserves, which are that every label is the same (no boundary can
    be read from it), that `n_inducing` was reduced to the number of observations, and that
    fewer inducing points were chosen than `n_inducing` asked for. It also
    returns the probit link (`probitLink`), so that the `Surrogate` wrapper can give
    predictions in both spaces: the latent function and the probability.

    **Reproducibility.** The only random draw of the build is the one the fitter makes, from
    the generator of the context (specification 3.7). The construction of the model draws
    nothing and touches no global random state, and the same context gives the same model.

    It supports `posterior:latent_gaussian` (the posterior of the latent function is Gaussian,
    and the labels are read from it through a link), `posterior:marginal_normal` (the latent
    function has a normal marginal at each point, which the level-set acquisitions need),
    `outputs:single`, `outcome_kind:binary` and `link:probit`.
    """

    def build(self, context: SurrogateBuildContext) -> BuiltSurrogate:
        """Fits the classifier on the labels of the context.

        Args:
            context: The problem (one binary outcome), the encoder, the encoded points and
                their labels, the generator and the runtime settings. Read-only.

        Returns:
            The fitted model in evaluation mode, its fit status and the probit link.

        Raises:
            ModelNotReadyError: if the context holds fewer observations than
                `config.min_observations`.
            ConfigError: if the problem has more or fewer than one outcome, or its outcome is
                not binary, or the nested kernel or fitter refuses its configuration.
            SurrogateContextError: if a label is neither 0 nor 1.
            FitterError: if the fitter refuses the model or the fit diverges.
        """
        self._requireOneOutcome(context)
        self._requireEnoughObservations(context)
        self._requireLabels(context.train_outcomes)
        likelihood = self._makeLikelihood(context)
        kernel = (
            kernelRegistry()
            .lookup(self.config.kernel.type)
            .builder(self.config.kernel)
            .build(context.encoder.n_columns)
        )
        inducing, warnings_ = self._inducingPoints(context)
        with warnings.catch_warnings():
            # BoTorch recommends standardizing the outcomes, which would turn labels into
            # numbers that the Bernoulli likelihood cannot read, so the advice is ignored here.
            warnings.filterwarnings(
                "ignore", message=".*not standardized.*", category=InputDataWarning
            )
            model = SingleTaskVariationalGP(
                context.train_inputs,
                context.train_outcomes,
                likelihood=likelihood,
                covar_module=kernel,
                inducing_points=inducing,
                learn_inducing_points=False,
            )
        fitterRegistry().lookup(self.config.fitter.type).builder(self.config.fitter).fit(
            model, generator=context.generator
        )
        return BuiltSurrogate(
            model=model,
            status=FitStatus(
                n_observations=context.n_observations,
                is_low_data=context.n_observations < self.config.low_data_threshold,
                warnings=(
                    *warnings_,
                    *self._selectionWarnings(inducing, model),
                    *self._labelWarnings(context.train_outcomes),
                ),
            ),
            link=probitLink(),
        )

    def _requireOneOutcome(self, context: SurrogateBuildContext) -> None:
        """Refuses a problem that does not have exactly one outcome.

        Args:
            context: The build context.

        Raises:
            ConfigError: if the problem has no outcome, or several. The message lists them.
        """
        outcomes = context.problem.outcomes
        if len(outcomes) == 1:
            return
        listed = [f"{outcome.name} ({outcome.type})" for outcome in outcomes]
        raise ConfigError(
            f"The surrogate {_NAME!r} models one binary outcome, and the problem has "
            f"{len(outcomes)}: {', '.join(listed)}.",
            component=f"surrogate {_NAME!r}",
            field_path="problem.outcomes",
            suggestion=(
                "use a problem with a single binary outcome, or a surrogate that models "
                "several outcomes (one that declares outputs:multi)"
            ),
        )

    def _requireEnoughObservations(self, context: SurrogateBuildContext) -> None:
        """Raises `ModelNotReadyError` when there are fewer observations than the minimum.

        Args:
            context: The build context.

        Raises:
            ModelNotReadyError: if `context.n_observations` is below `config.min_observations`.
        """
        needed = self.config.min_observations
        if context.n_observations >= needed:
            return
        raise ModelNotReadyError(
            f"The surrogate {_NAME!r} needs at least {needed} observation(s) to fit and has "
            f"{context.n_observations}.",
            component=f"surrogate {_NAME!r}",
            field_path="min_observations",
            suggestion=(
                "complete more trials before asking for a model, or lower min_observations"
            ),
        )

    def _requireLabels(self, labels: Tensor) -> None:
        """Refuses a label that is not 0 or 1.

        Args:
            labels: The outcome tensor of the context, shape `(n, 1)`.

        Raises:
            SurrogateContextError: if a value is neither 0 nor 1.
        """
        if bool(((labels == 0.0) | (labels == 1.0)).all()):
            return
        raise SurrogateContextError(
            f"The surrogate {_NAME!r} reads labels, and train_outcomes holds a value that is "
            "neither 0 nor 1: a Bernoulli likelihood has no probability for it.",
            component=f"surrogate {_NAME!r}",
            field_path="train_outcomes",
            suggestion="give the labels of a binary outcome as 0.0 and 1.0, as Observation "
            "stores them",
        )

    def _makeLikelihood(self, context: SurrogateBuildContext) -> Likelihood:
        """Makes the Bernoulli likelihood through the registry.

        Args:
            context: The build context.

        Returns:
            The native likelihood, on the precision and device of the runtime settings.

        Raises:
            ConfigError: if the outcome is of a kind that the likelihood does not support.
        """
        entry = likelihoodRegistry().lookup(_LIKELIHOOD)
        return entry.builder(entry.config_schema()).build(
            LikelihoodBuildContext(
                outcomes=tuple(context.problem.outcomes), runtime=context.runtime
            )
        )

    def _inducingPoints(
        self, context: SurrogateBuildContext
    ) -> tuple[Tensor | int, list[str]]:
        """The inducing points that the model is given, and what to say about the choice.

        Args:
            context: The build context.

        Returns:
            The training inputs, copied, when `n_inducing` is unset or at least the number of
            observations, and the number of points to choose otherwise, with the warnings of
            the choice.
        """
        wanted = self.config.n_inducing
        n = context.n_observations
        if wanted is not None and wanted < n:
            return wanted, []
        said = []
        if wanted is not None and wanted > n:
            said.append(
                f"n_inducing is {wanted} and there are {n} observations: the inducing points "
                "are the observed points, as many as there are observations."
            )
        return context.train_inputs.clone(), said

    @staticmethod
    def _selectionWarnings(
        requested: Tensor | int, model: SingleTaskVariationalGP
    ) -> list[str]:
        """What the choice of the inducing points by BoTorch deserves to be warned about.

        Args:
            requested: What `_inducingPoints` gave the model: the points themselves, or the
                number of points to choose.
            model: The model that was built with it.

        Returns:
            One message if fewer points were chosen than were asked for: the greedy selection
            ends when the observations left out are explained by those chosen. Nothing
            otherwise, and nothing when the points were given.
        """
        if isinstance(requested, Tensor):
            return []
        chosen = int(model.model.variational_strategy.inducing_points.shape[0])
        if chosen >= requested:
            return []
        return [
            f"n_inducing is {requested} and {chosen} inducing points were chosen: the "
            "selection stops when the observations left out are explained by those chosen, "
            "so more of them would add nothing."
        ]

    @staticmethod
    def _labelWarnings(labels: Tensor) -> tuple[str, ...]:
        """What the labels deserve to be warned about.

        Args:
            labels: The outcome tensor of the context, shape `(n, 1)`.

        Returns:
            One message if every label is the same: no boundary between the classes can be
            read from them. Nothing otherwise. The labels are not empty: a model needs at
            least one observation.
        """
        if bool((labels == labels[0]).all()):
            only = "true" if float(labels[0]) == 1.0 else "false"
            return (
                f"Every label is {only}: the data holds no boundary between the classes, so "
                "the probabilities follow the prior and the labels seen so far.",
            )
        return ()
