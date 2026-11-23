"""The `ucb_v1` acquisition: the analytic upper confidence bound."""

from __future__ import annotations

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.objectiveWeights import objectiveWeights
from planexs.acquisition.registerAcquisition import registerAcquisition
from planexs.acquisition.requireOptimizeGoalConfig import requireOptimizeGoalConfig
from planexs.acquisition.UCBConfig import UCBConfig

#: Who the errors of this module are about.
_COMPONENT = "acquisition 'ucb_v1'"


@registerAcquisition(
    "ucb_v1",
    config=UCBConfig,
    description=(
        "Analytic upper confidence bound for one objective: the predicted mean plus beta "
        "posterior standard deviations, for one point at a time."
    ),
    needs=frozenset({"posterior:marginal_normal", "objectives:single", "goal:optimize"}),
    supports=frozenset({"acquisition:differentiable"}),
)
class UCBBuilder(AbstractAcquisitionBuilder[UCBConfig]):
    """Builds BoTorch's `UpperConfidenceBound`: the mean of the objective that the surrogate
    predicts at a point, plus `beta` standard deviations of that prediction (specification
    5.8).

    It is the simplest way to trade exploitation for exploration. The mean says how good a
    point is expected to be, the standard deviation how little the surrogate knows about it,
    and `beta` how much the second counts against the first: at 0 the bound is the mean and
    the search exploits, and the larger it is the more an unexplored point is worth. The score
    has the unit of the objective, and it needs nothing but the mean and the standard
    deviation of the posterior at the point (`posterior:marginal_normal`), so it works with a
    surrogate whose posterior is not jointly Gaussian, such as a classifier read in its latent
    space.

    **What it reads.** An optimize goal with one scalar objective (`goal:optimize`,
    `objectives:single`): one outcome, or several that the goal's weights combine into one
    preference. The objective is the weighted sum of the outcomes that the goal optimizes, each
    in its own direction: a minimized outcome is negated, so that a higher score always means a
    better candidate (specification 4, "Acquisition"). The sum is made on the posterior, by a
    scalarizing transform that gives each output of the model its weight by name, so that the
    order of the outputs never matters and an output that the goal only constrains does not
    count. The mean of the sum is the sum of the means, and its variance is read from the
    joint posterior of the outputs, so a weighted goal is as certain as the model says it is.

    **The direction.** For a maximized outcome the bound is `mean + beta * std`. For a
    minimized one it is `-mean + beta * std`: the uncertainty is added in both cases, because
    it is the *optimism* about the outcome, and for a minimizer the optimistic end is the low
    one. So at equal means the more uncertain point scores higher whichever way the goal
    points.

    **What it does not do.** It scores one point at a time: the native function refuses a batch
    of more than one point (`q = 1`), so a batch is built by an optimizer that picks points
    one after the other. It does not support `pending:supported`: its value at a point depends
    on the posterior at that point and on nothing else, so the pending points of the context
    are not used, and a point next to a pending one scores as if it were not pending. It does
    not support `outcome_constraints:supported` either: an analytic bound has no feasibility
    weighting, so a goal with outcome constraints is refused by the compatibility check
    for this entry; `qlog_nei_v1` handles them. It draws no random number, so the generator of
    the context is not advanced.

    **The setting.** `beta` is a number of standard deviations (`UCBConfig`). BoTorch's own
    parameter of the same name is the *square* of it (its bound is `mean + sqrt(beta) * std`),
    so the builder squares the configured value, and the native object, reached through the
    escape hatch, holds that square. The function is differentiable in the candidate
    (`acquisition:differentiable`).

    A builder is configured once and builds as often as it is asked; it needs no reference set.
    """

    def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
        """Builds the upper confidence bound for the surrogate and the goal of the context.

        Args:
            context: The surrogate, the goal, the pending points and the generator. Read-only.
                Its goal must be an optimize goal with one scalar objective, and its surrogate
                must model every outcome that the goal optimizes. The pending points are not
                used (see the class docstring) and the generator is not drawn from.

        Returns:
            A BoTorch `UpperConfidenceBound` over the encoded space, scoring a batch of
            candidates of one point each (shape `batch x 1 x d`) by their upper confidence
            bound.

        Raises:
            ConfigError: if the goal is not an optimize goal, or has several outcomes and no
                weights, so that its objective is not a single scalar.
            AcquisitionContextError: if the surrogate does not model an outcome that the goal
                optimizes.
        """
        goal_config = requireOptimizeGoalConfig(context, _COMPONENT)
        weights = objectiveWeights(
            goal_config,
            context.surrogate.outcome_names,
            dtype=context.encoder.dtype,
            device=context.generator.device,
            component=_COMPONENT,
        )
        return UpperConfidenceBound(
            model=context.surrogate.model,
            # BoTorch weighs the standard deviation by the square root of its `beta`.
            beta=self.config.beta**2,
            # The directions are in the weights, so the engine always maximizes.
            posterior_transform=ScalarizedPosteriorTransform(weights),
            maximize=True,
        )
