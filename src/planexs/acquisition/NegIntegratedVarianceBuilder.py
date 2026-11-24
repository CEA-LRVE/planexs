"""The `neg_integrated_variance_v1` acquisition: the negative integrated posterior variance."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.sampling.normal import SobolQMCNormalSampler

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.drawSeed import drawSeed
from planexs.acquisition.exploreOutcomeWeights import exploreOutcomeWeights
from planexs.acquisition.NegIntegratedVarianceConfig import NegIntegratedVarianceConfig
from planexs.acquisition.registerAcquisition import registerAcquisition
from planexs.acquisition.requireExploreGoal import requireExploreGoal

#: Who the errors of this module are about.
_COMPONENT = "acquisition 'neg_integrated_variance_v1'"


@registerAcquisition(
    "neg_integrated_variance_v1",
    config=NegIntegratedVarianceConfig,
    description=(
        "Negative integrated posterior variance for pure exploration: how little the model "
        "would still be unsure of over the whole space once the candidate is measured."
    ),
    needs=frozenset({"posterior:gaussian", "goal:explore"}),
    supports=frozenset({"pending:supported"}),
)
class NegIntegratedVarianceBuilder(AbstractAcquisitionBuilder[NegIntegratedVarianceConfig]):
    """Builds BoTorch's `qNegIntegratedPosteriorVariance`: minus the variance that the model
    would have, averaged over the search space, once the candidate had been measured
    (specification 5.8, FR21).

    It is active learning, with no direction. Nothing is maximized or searched for: the aim is
    a model that is accurate everywhere, and a candidate is worth what it would teach the
    model. A point next to an observation teaches little (the variance there is small already,
    and so it is around it), a point in a region that has no observation teaches a lot (it
    brings down the variance of every point around it), so the score is highest, that is
    closest to zero, in the unexplored regions, and it never needs the observed values: the
    variance of a Gaussian model depends on where it was measured and not on what was found.
    The score is negative or zero, because it is minus a variance.

    **What it reads.** It needs a fitted surrogate whose posterior is Gaussian
    (`posterior:gaussian`) and an explore goal (`goal:explore`), which names the outcomes to
    learn (all of them by default). The outcomes that the goal does not name are ignored,
    and the order of the outcomes of the surrogate never matters, since they are placed by
    name. With several outcomes the variances are added, each divided by the square of the
    spread of its outcome over the observed points (see `exploreOutcomeWeights`): a variance
    is in the unit of its outcome, and without that division the outcome that is measured in
    the largest unit would be the only one that counts. With one outcome the division changes
    the scale of the score and not the order of the candidates, so the score is the share of
    the variation of the outcome that the model is still unsure of, as a negative number.

    **The points.** The variance is averaged over `n_points` points of the search space
    (`NegIntegratedVarianceConfig`), drawn here from the encoder of the problem, which is
    what the builder is given: a scrambled Sobol sequence rounded onto the values that the
    parameters can take, seeded from the generator of the context. They cover the whole
    search space, not only the part that satisfies its constraints, so the score also values
    teaching the model about points that could never be suggested. They are the same for every
    candidate of one build, so the score is a deterministic function of the candidate.

    **Pending points and batches** (`pending:supported`, FR10). The pending points of the
    context are the points chosen but not measured yet, and the model is conditioned on them
    together with the candidate: a candidate next to a pending point teaches the model little
    more than the pending point does, so it scores lower than it would have, and a policy that
    builds a batch by choosing one point at a time and passing the chosen ones back as pending
    gets a spread-out batch. A batch of `q` points is scored as a whole, by the variance after
    all of them.

    **What it does not do.** It does not declare `acquisition:differentiable`: that tag is not
    part of its contract, although the engine's function does give a gradient. It also relies
    on a property of the engine's function: the variance of the model after conditioning must
    not depend on the values that the candidate would bring, which is true of a Gaussian
    process and not of a model whose noise depends on its outputs. It needs a model that can
    condition itself on points that have not been measured (a fantasy model), as the Gaussian
    processes do. It draws two seeds from the generator of the context (one for the points, one
    for the sampler of the engine's function, whose value does not depend on it but whose
    default would draw from the global random number generator).

    A builder is configured once and builds as often as it is asked; it needs no reference
    set.
    """

    def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
        """Builds the negative integrated posterior variance for the surrogate and the goal of
        the context.

        Args:
            context: The surrogate, the goal, the pending points and the generator. Read-only.
                Its goal must be an explore goal, and its surrogate must model every outcome
                that the goal explores.

        Returns:
            A BoTorch `qNegIntegratedPosteriorVariance` over the encoded space, scoring a batch
            of `q` candidate points (shape `batch x q x d`) by minus the variance, averaged
            over the sample points, that the model would have once they were measured. It
            takes the pending points of the context into account.

        Raises:
            ConfigError: if the goal is not an explore goal, or names an outcome that the
                problem does not have.
            AcquisitionContextError: if the surrogate does not model an outcome that the goal
                explores, or its model holds no observed point.
        """
        goal = requireExploreGoal(context, _COMPONENT)
        model = context.surrogate.model
        baseline = baselinePoints(model)
        weights = exploreOutcomeWeights(
            goal.effectiveOutcomes(context.problem),
            context.surrogate.outcome_names,
            model,
            baseline,
        )
        points = context.encoder.sampleEncoded(
            self.config.n_points, seed=drawSeed(context.generator)
        ).to(device=baseline.device, dtype=baseline.dtype)
        return qNegIntegratedPosteriorVariance(
            model=model,
            mc_points=points,
            # The value does not depend on the samples that the model is conditioned on, but
            # the engine's default sampler takes its seed from the global generator.
            sampler=SobolQMCNormalSampler(
                torch.Size([1]),  # structural: one sample, which the value does not use
                seed=drawSeed(context.generator),
            ),
            posterior_transform=ScalarizedPosteriorTransform(weights),
            X_pending=context.pending_points,
        )
