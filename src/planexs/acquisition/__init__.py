"""The `acquisition` layer: the scores that rank candidate points (specification 5.1, 5.8).

An acquisition is a score over candidate points, built from a surrogate, a goal and the pending
points: higher means more worth evaluating. Which one is used is a choice of configuration, so
this layer is organized around the `acquisition` kind of components: the contract that every
acquisition honours (`AbstractAcquisitionBuilder`), what it is given
(`AcquisitionBuildContext`) and the error for a context that does not fit
(`AcquisitionContextError`), the registry (`acquisitionRegistry`) and the registration
decorator (`registerAcquisition`). The registered acquisitions come with the tasks of roadmap
Phase 5: so far the Monte Carlo log noisy expected improvement, `qlog_nei_v1`
(`QLogNEIBuilder`, with its settings in `QLogNEIConfig`), which handles noisy data, pending
points, batches and the outcome constraints of the goal, and the analytic upper confidence
bound, `ucb_v1` (`UCBBuilder`, with its settings in `UCBConfig`), which scores one point at a
time, and the multi-objective Monte Carlo log noisy expected hypervolume improvement,
`qlog_nehvi_v1` (`QLogNEHVIBuilder`, with its settings in `QLogNEHVIConfig`, whose reference
point of the hypervolume is either explicit, `ExplicitReferencePointConfig`, or read off the
data by a named rule, `InferredReferencePointConfig`). The acquisitions that score the value of
one objective share `requireOptimizeGoalConfig` (the optimize goal of a build context) and
`objectiveWeights` (the goal's scalar objective as a weight for each output of the model); the
one for a Pareto problem reads the objectives apart instead (`selectParetoObjectives`,
`ParetoObjectives`) and resolves the reference point (`resolveReferencePoint`,
`predictedParetoFront`). The Monte Carlo ones also share `baselinePoints` (the observed points
of a model), `outcomeConstraintCallables` (the goal's constraints as BoTorch feasibility
constraints) and `drawSeed` (a seed for a sampler, drawn from the generator of the build).

The rule of this layer: it may import `planexs.core`, `planexs.domain`, `planexs.encoding`,
`planexs.surrogates` and `planexs.goals`, and no higher layer. Like `encoding`, it may import
torch and BoTorch; only `adapters` may import Ax.

Importing the package defines the `acquisition` kind and registers the capability tags of the
layer (`acquisition:differentiable`, `pending:supported`): that is what `acquisitionRegistry`
is imported for below. The third tag of the layer, `outcome_constraints:supported`, is
registered with the goal tags, by `planexs.goals.goalRegistry`, which `acquisitionRegistry`
imports for that effect. A module that registers an acquisition must be imported here, one
line per module, or the acquisition is silently unregistered (specification 5.12, 10.6).

The modules of the legacy alpha layer that this package still holds (`AcquisitionStrategy`,
`AlphaAcquisitionStrategy`, `ExplorationExploitationAcquisition`,
`MultiObjectiveExplorationExploitationAcquisition` and `modelParetoFront`: the blend with its
single weight, FR5, and the optimization of it) are not imported here and are not registered.
They are ported as registered builders (roadmap tasks P5-07 and P5-08), and deleted with the
rest of the legacy layer (P11-02).
"""

from planexs.acquisition import (
    QLogNEHVIBuilder,
    QLogNEIBuilder,
    UCBBuilder,
    acquisitionRegistry,
)

__all__ = ["QLogNEHVIBuilder", "QLogNEIBuilder", "UCBBuilder", "acquisitionRegistry"]
