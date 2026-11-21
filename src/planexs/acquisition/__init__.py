"""The `acquisition` layer: the scores that rank candidate points (specification 5.1, 5.8).

An acquisition is a score over candidate points, built from a surrogate, a goal and the pending
points: higher means more worth evaluating. Which one is used is a choice of configuration, so
this layer is organized around the `acquisition` kind of components: the contract that every
acquisition honours (`AbstractAcquisitionBuilder`), what it is given
(`AcquisitionBuildContext`) and the error for a context that does not fit
(`AcquisitionContextError`), the registry (`acquisitionRegistry`) and the registration
decorator (`registerAcquisition`). The registered acquisitions come with the tasks of roadmap
Phase 5.

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

from planexs.acquisition import acquisitionRegistry

__all__ = ["acquisitionRegistry"]
