"""The `straddle_v1` acquisition: the straddle of the level of a level-set goal."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.registerAcquisition import registerAcquisition
from planexs.acquisition.requireLevelSetGoal import requireLevelSetGoal
from planexs.acquisition.StraddleConfig import StraddleConfig
from planexs.acquisition.StraddleFunction import StraddleFunction

#: Who the errors of this module are about.
_COMPONENT = "acquisition 'straddle_v1'"


@registerAcquisition(
    "straddle_v1",
    config=StraddleConfig,
    description=(
        "Straddle of a level: beta posterior standard deviations minus the distance of the "
        "predicted mean from the level, high where the model is uncertain and close to the "
        "level, for one point at a time."
    ),
    needs=frozenset({"goal:level_set", "posterior:marginal_normal"}),
    supports=frozenset({"acquisition:differentiable"}),
)
class StraddleBuilder(AbstractAcquisitionBuilder[StraddleConfig]):
    """Builds `StraddleFunction`: `beta` posterior standard deviations minus the distance of
    the predicted mean from the level, for a level-set goal (specification 5.8, FR20).

    A level-set goal wants the boundary where an outcome crosses a level, and the boundary is
    best learnt where it is least known: where the surrogate predicts a value close to the
    level and is unsure of the prediction. The straddle scores exactly that. The mean says how
    close the point is predicted to lie to the boundary, the standard deviation how little the
    surrogate knows about it, and `beta` how much a standard deviation of doubt counts against
    a unit of distance. It needs nothing but the mean and the standard deviation of the
    posterior at the point (`posterior:marginal_normal`), so it works with an exact Gaussian
    process and with a classifier read in its latent space.

    **What it reads.** A level-set goal (`goal:level_set`): the outcome and the level. The
    outcome is placed on an output of the surrogate by name, so the order of the outputs never
    matters and an output that the goal does not name takes no part. The level is the goal's
    own (`LevelSetGoal.effectiveLevel`): in the units of the outcome for a continuous outcome,
    a probability for a binary one, where it defaults to 0.5.

    **The latent space.** The surrogate decides the space the score lives in, and converts the
    level to it (`Surrogate.latentLevel`). A surrogate whose model is the outcome itself, the
    exact Gaussian process, keeps the level as it is, and the score is in the units of the
    outcome. A surrogate that reads the outcome from a latent function through a link, the
    classifier, is scored on the latent function: the level is moved through the link (a
    probability of 0.5 becomes the latent threshold 0 of a probit link), and the mean and the
    standard deviation are those of the latent function, so the score is in the units of the
    latent function and a level that is not strictly between 0 and 1 is refused, as the
    surrogate does.

    **The direction.** There is none: the mean enters through its distance to the level, so a
    point predicted above the level scores the same as one predicted below it.

    **What it does not do.** It scores one point at a time: the native function refuses a batch
    of more than one point (`q = 1`), so a batch is built by an optimizer that picks points one
    after the other. It does not support `pending:supported`: its value at a point depends on
    the posterior at that point and on nothing else, so the pending points of the context are
    not used. A level-set goal has no outcome constraints, so `outcome_constraints:supported`
    does not arise. It draws no random number, so the generator of the context is not
    advanced. The distance to the level has a kink where the predicted mean equals the level
    (see `StraddleFunction`); the function is differentiable elsewhere
    (`acquisition:differentiable`). Like the other analytic acquisitions, it reads the
    posterior through the engine's scalarizing transform, which reads GPyTorch posteriors: a
    future surrogate that declares `posterior:marginal_normal` with a posterior of another kind
    would be refused by the engine at build time.

    **The setting.** `beta` is a number of posterior standard deviations (`StraddleConfig`), as
    for `ucb_v1`, and the function uses it as it is.

    A builder is configured once and builds as often as it is asked; it needs no reference set.
    """

    def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
        """Builds the straddle for the surrogate and the goal of the context.

        Args:
            context: The surrogate, the goal, the pending points and the generator. Read-only.
                Its goal must be a level-set goal whose outcome the surrogate models and whose
                level the surrogate can use. The pending points are not used and the
                generator is not drawn from.

        Returns:
            A `StraddleFunction` over the encoded space, scoring a batch of candidates of one
            point each (shape `batch x 1 x d`) by their straddle of the level.

        Raises:
            ConfigError: if the goal is not a level-set goal, has no level for a continuous
                outcome or a level that does not fit its outcome, or names an outcome that the
                problem does not have, or if the level cannot be converted by the surrogate (a
                level outside 0 to 1 for a surrogate with a link).
            AcquisitionContextError: if the surrogate does not model the outcome of the goal.
        """
        goal = requireLevelSetGoal(context, _COMPONENT)
        surrogate = context.surrogate
        level = surrogate.latentLevel(goal.effectiveLevel(context.problem))
        outcome = goal.config.outcome
        names = surrogate.outcome_names
        if outcome not in names:
            raise AcquisitionContextError(
                f"The goal looks for the level of the outcome {outcome!r}, which the "
                f"surrogate does not model. The surrogate covers: "
                f"{', '.join(map(repr, names))}.",
                component="acquisition build context",
                field_path="goal.outcome",
                suggestion="fit the surrogate on the outcome that the goal names",
            )
        weights = torch.tensor(
            [1.0 if name == outcome else 0.0 for name in names],
            dtype=context.encoder.dtype,
            device=context.generator.device,
        )
        return StraddleFunction(
            model=surrogate.model,
            beta=self.config.beta,
            level=level,
            posterior_transform=ScalarizedPosteriorTransform(weights),
        )
