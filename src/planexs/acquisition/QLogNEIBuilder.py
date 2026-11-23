"""The `qlog_nei_v1` acquisition: the Monte Carlo log noisy expected improvement."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.acquisition.utils import prune_inferior_points
from botorch.sampling.normal import SobolQMCNormalSampler

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.drawSeed import drawSeed
from planexs.acquisition.objectiveWeights import objectiveWeights
from planexs.acquisition.outcomeConstraintCallables import outcomeConstraintCallables
from planexs.acquisition.QLogNEIConfig import QLogNEIConfig
from planexs.acquisition.registerAcquisition import registerAcquisition
from planexs.acquisition.requireOptimizeGoalConfig import requireOptimizeGoalConfig

#: Who the errors of this module are about.
_COMPONENT = "acquisition 'qlog_nei_v1'"


@registerAcquisition(
    "qlog_nei_v1",
    config=QLogNEIConfig,
    description=(
        "Monte Carlo log noisy expected improvement for one objective: handles noisy "
        "observations, pending points, batches and outcome constraints."
    ),
    needs=frozenset({"posterior:gaussian", "objectives:single", "goal:optimize"}),
    supports=frozenset(
        {"pending:supported", "outcome_constraints:supported", "acquisition:differentiable"}
    ),
)
class QLogNEIBuilder(AbstractAcquisitionBuilder[QLogNEIConfig]):
    """Builds BoTorch's `qLogNoisyExpectedImprovement`, the log of the expected improvement
    over the best observed value, estimated by Monte Carlo (specification 5.8, FR10, FR15).

    The improvement of a candidate is measured against the best value among the observed
    points, as the model believes it to be. It does not take the observed values as they are
    (a noisy observation could then be an incumbent that is not real): the observed points are
    sampled jointly with the candidate, so the incumbent is as uncertain as the model says.
    That is what makes the acquisition right for noisy data, and what makes it cost more than
    an analytic expected improvement. It is a log, so that the value stays finite and keeps a
    gradient where the improvement is almost zero, which is where an optimizer starts.

    **What it reads.** It needs a fitted surrogate whose posterior is Gaussian
    (`posterior:gaussian`, which the Monte Carlo samples are drawn from) and an optimize goal
    with one scalar objective (`goal:optimize`, `objectives:single`): one outcome, or several
    that the goal's weights combine into one preference. The objective is the weighted sum of
    the outcomes that the goal optimizes, each in its own direction: a minimized outcome is
    negated, so that a higher score always means a better candidate (specification 4,
    "Acquisition"). The weights are the goal's own, keyed by outcome name, and are applied to
    the outputs of the model by name, so that the order of the model's outputs never matters.

    **Pending points and batches** (`pending:supported`, FR10). The pending points of the
    context are the points chosen but not evaluated yet, and they are part of the incumbent:
    a candidate is scored by its *incremental* improvement over the best of the observed and
    the pending points, so a candidate at a pending point, or next to it, scores low, and a
    policy that builds a batch by choosing one point at a time and passing the chosen ones
    back as pending gets a spread-out batch. The native function keeps the pending points
    when its `set_X_pending` is called, so an optimizer that builds a batch sequentially works
    with it.

    **Outcome constraints** (`outcome_constraints:supported`, FR15). The legacy layer only
    warned that it did not enforce the constraints of the goal. Here each one is passed to the
    acquisition as a feasibility constraint on the posterior samples (see
    `outcomeConstraintCallables`): a sample that violates a constraint contributes less in
    proportion to how clearly it violates it, so a candidate that is predicted to violate a
    constraint scores lower than one that is predicted to meet it, and an observed point that
    violates one does not count as the incumbent. The constraint may be on any outcome that the
    surrogate models, not only the ones the goal optimizes. The feasibility is a smooth
    weighting of the improvement, not a hard filter: the acquisition ranks candidates, and
    only the policy's feasible sampler guarantees constraints on the parameters.

    **The settings** are the number of Monte Carlo samples and whether the baseline is pruned
    (`QLogNEIConfig`). The samples are quasi-random, seeded from the generator of the context,
    and the same for every candidate of one build, and so are the ones that decide the pruning,
    so the function is deterministic for the configuration, the context and the generator, and
    differentiable in the candidate (`acquisition:differentiable`).

    A builder is configured once and builds as often as it is asked; it needs no reference set.
    """

    def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
        """Builds the log noisy expected improvement for the surrogate and the goal of the
        context.

        Args:
            context: The surrogate, the goal, the pending points and the generator. Read-only.
                Its goal must be an optimize goal with one scalar objective, and its surrogate
                must model every outcome that the goal optimizes or constrains.

        Returns:
            A BoTorch `qLogNoisyExpectedImprovement` over the encoded space, scoring a batch of
            `q` candidate points (shape `batch x q x d`) by their log improvement. It takes the
            pending points of the context into account.

        Raises:
            ConfigError: if the goal is not an optimize goal, or has several outcomes and no
                weights, so that its objective is not a single scalar.
            AcquisitionContextError: if the surrogate does not model an outcome that the goal
                optimizes or constrains, or its model holds no observed point.
        """
        goal_config = requireOptimizeGoalConfig(context, _COMPONENT)
        names = context.surrogate.outcome_names
        model = context.surrogate.model
        baseline = baselinePoints(model)
        objective = LinearMCObjective(
            objectiveWeights(
                goal_config,
                names,
                dtype=baseline.dtype,
                device=baseline.device,
                component=_COMPONENT,
            )
        )
        # An empty list is not "no constraint" for the engine, which would build an indicator
        # over nothing.
        constraints = (
            outcomeConstraintCallables(model, baseline, goal_config.outcome_constraints, names)
            or None
        )
        sample_shape = torch.Size([self.config.n_samples])
        sampler = SobolQMCNormalSampler(sample_shape, seed=drawSeed(context.generator))
        if self.config.prune_baseline:
            # The engine can prune the baseline itself (`prune_baseline=True`), but it draws
            # the samples that decide the pruning from an unseeded sampler: two builds from
            # one seed could keep different incumbents, and a run would not be reproducible.
            # The pruning is done here with a sampler seeded from the generator, and the
            # engine is told not to prune again.
            baseline = prune_inferior_points(
                model=model,
                X=baseline,
                objective=objective,
                constraints=constraints,
                sampler=SobolQMCNormalSampler(sample_shape, seed=drawSeed(context.generator)),
            )
        return qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=baseline,
            sampler=sampler,
            objective=objective,
            X_pending=context.pending_points,
            constraints=constraints,
            prune_baseline=False,
        )
