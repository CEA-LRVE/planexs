"""The `qlog_nehvi_v1` acquisition: the log noisy expected hypervolume improvement."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.drawSeed import drawSeed
from planexs.acquisition.outcomeConstraintCallables import outcomeConstraintCallables
from planexs.acquisition.prunedParetoBaseline import prunedParetoBaseline
from planexs.acquisition.QLogNEHVIConfig import QLogNEHVIConfig
from planexs.acquisition.registerAcquisition import registerAcquisition
from planexs.acquisition.requireOptimizeGoalConfig import requireOptimizeGoalConfig
from planexs.acquisition.resolveReferencePoint import resolveReferencePoint
from planexs.acquisition.selectParetoObjectives import selectParetoObjectives

#: Who the errors of this module are about.
_COMPONENT = "acquisition 'qlog_nehvi_v1'"


@registerAcquisition(
    "qlog_nehvi_v1",
    config=QLogNEHVIConfig,
    description=(
        "Monte Carlo log noisy expected hypervolume improvement for several objectives: "
        "handles noisy observations, pending points, batches and outcome constraints."
    ),
    needs=frozenset({"posterior:gaussian", "objectives:multi", "goal:optimize"}),
    supports=frozenset(
        {"pending:supported", "outcome_constraints:supported", "acquisition:differentiable"}
    ),
)
class QLogNEHVIBuilder(AbstractAcquisitionBuilder[QLogNEHVIConfig]):
    """Builds BoTorch's `qLogNoisyExpectedHypervolumeImprovement`, the log of the expected
    gain in hypervolume that a candidate brings to the Pareto front, estimated by Monte Carlo
    (specification 5.8, FR2, FR10, FR15).

    The hypervolume of a set of points is the volume of the region that they dominate and that
    lies on the better side of a reference point. A candidate is worth what it adds to it: a
    point that extends the Pareto front, or pushes it outward, adds volume, and a point that is
    dominated by the front adds none. So it ranks candidates without trading the objectives
    against each other, and a candidate is good when it is a plausible new member of the
    front. It is the multi-objective counterpart of `qlog_nei_v1`. It does not take the
    observed values as they are, which are noisy: the observed points are sampled jointly with
    the candidate, so the front it improves on is as uncertain as the model says. It is a log,
    so that the value stays finite and keeps a gradient where the gain is almost zero, which is
    where an optimizer starts.

    **What it reads.** It needs a fitted surrogate whose posterior is Gaussian
    (`posterior:gaussian`, which the Monte Carlo samples are drawn from) and an optimize goal
    with several objectives (`goal:optimize`, `objectives:multi`): two or more outcomes and no
    weights. Each objective keeps its direction: a minimized outcome is negated, so that the
    front is always built towards larger values (specification 4, "Acquisition"). The
    objectives are placed on the outputs of the model by name, so that the order of the
    outputs, or of the goal, never matters. An output that the goal only constrains is not an
    objective. A goal with weights, or with one outcome, is refused: it is a single objective,
    and `qlog_nei_v1` scores it.

    **The reference point** comes from the configuration (`QLogNEHVIConfig`), because it says
    what a useless result is and the data cannot tell. It is either given as one value for
    each objective, in the natural units of the outcomes (`explicit`), or read off by a named
    rule (`inferred`): the rule `nadir_margin` puts it just beyond the worst values of the
    Pareto front that the model predicts at the observed points, so that it follows the data
    and needs no knowledge of the scale of the outcomes.

    **The partitioning** of the dominated and the non-dominated region, which the hypervolume
    improvement is computed from, is made by BoTorch for each posterior sample, from the
    observed points that the model itself holds (the baseline). With
    `prune_baseline` the points that are very unlikely to be on the front are left out first,
    which keeps the cost down once there are many observations; the pruning is seeded from the
    generator of the context, since BoTorch's own is not reproducible, and it leaves the
    global random number generator as it found it.

    **Pending points and batches** (`pending:supported`, FR10). The pending points of the
    context are the points chosen but not evaluated yet. A candidate is scored by its
    *incremental* improvement over the front of the observed and the pending points together,
    so a candidate at a pending point, or next to it, scores low, and a policy that builds a
    batch by choosing one point at a time and passing the chosen ones back as pending gets a
    spread-out batch. The native function keeps the pending points when its `set_X_pending` is
    called, so an optimizer that builds a batch sequentially works with it. It folds them into
    the partitioning that it caches, which makes each score cheaper and has a consequence:
    pending points can be added to a built function and not taken back (clearing them leaves
    the cached partitioning as it was), so a different pending set is a new build, which is
    what a policy does anyway, since the pending points are in its context.

    **Outcome constraints** (`outcome_constraints:supported`, FR15). Each constraint of the
    goal is passed to the acquisition as a feasibility constraint on the posterior samples
    (see `outcomeConstraintCallables`, which `qlog_nei_v1` uses too): a sample that violates a
    constraint contributes less in proportion to how clearly it violates it, so a candidate
    that is predicted to violate a constraint scores lower than one that is predicted to meet
    it, and an observed point that violates one is not part of the front. The constraint may
    be on any outcome that the surrogate models, not only the ones the goal optimizes. The
    feasibility is a smooth weighting of the improvement, not a hard filter: the acquisition
    ranks candidates, and only the policy's feasible sampler guarantees constraints on the
    parameters. When the reference point is inferred, the front it is read from leaves out the
    observed points that are predicted to violate a constraint.

    **The settings** are the reference point, the number of Monte Carlo samples and whether the
    baseline is pruned. The samples are quasi-random, seeded from the generator of the context,
    and the same for every candidate of one build, and so are the ones that decide the pruning,
    so the function is deterministic for the configuration, the context and the generator, and
    differentiable in the candidate (`acquisition:differentiable`).

    A builder is configured once and builds as often as it is asked; it needs no reference
    set (the reference points of the context are a set of points for an integration, which
    this acquisition does not use, and are not the reference point of the hypervolume).
    """

    def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
        """Builds the log noisy expected hypervolume improvement for the surrogate and the
        goal of the context.

        Args:
            context: The surrogate, the goal, the pending points and the generator. Read-only.
                Its goal must be an optimize goal with several objectives and no weights, and
                its surrogate must model every outcome that the goal optimizes or constrains.

        Returns:
            A BoTorch `qLogNoisyExpectedHypervolumeImprovement` over the encoded space,
            scoring a batch of `q` candidate points (shape `batch x q x d`) by the log of
            their hypervolume improvement. It takes the pending points of the context into
            account.

        Raises:
            ConfigError: if the goal is not an optimize goal, has weights, or has fewer than
                two outcomes; or if an explicit reference point does not give one value for
                each objective of the goal.
            AcquisitionContextError: if the surrogate does not model an outcome that the goal
                optimizes or constrains, or its model holds no observed point.
        """
        goal_config = requireOptimizeGoalConfig(context, _COMPONENT)
        names = context.surrogate.outcome_names
        objectives = selectParetoObjectives(goal_config, names, component=_COMPONENT)
        model = context.surrogate.model
        baseline = baselinePoints(model)
        objective = objectives.objective(dtype=baseline.dtype, device=baseline.device)
        # An empty list is not "no constraint" for the engine, which would build an indicator
        # over nothing.
        constraints = (
            outcomeConstraintCallables(model, baseline, goal_config.outcome_constraints, names)
            or None
        )
        ref_point = resolveReferencePoint(
            self.config.reference_point,
            objectives,
            model=model,
            baseline=baseline,
            constraints=constraints,
            component=_COMPONENT,
        )
        sampler = SobolQMCNormalSampler(
            torch.Size([self.config.n_samples]), seed=drawSeed(context.generator)
        )
        if self.config.prune_baseline:
            baseline = prunedParetoBaseline(
                model,
                baseline,
                ref_point=ref_point,
                objective=objective,
                constraints=constraints,
                n_samples=self.config.n_samples,
                seed=drawSeed(context.generator),
            )
        return qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=context.pending_points,
            # The baseline is pruned above, with a seed.
            prune_baseline=False,
        )
