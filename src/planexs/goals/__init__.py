"""The `goals` layer: what a caller wants to know or decide (specification 5.1, 5.7).

A goal is typed configuration that proposes defaults and computes nothing (ADR-0013): it
parameterizes acquisitions and analyses, declares capability tags, and proposes recipes
(default policy configurations, in order of preference) and default analyses. Which goals
exist is a choice of configuration, so this layer is organized around the `goal` kind of
components: the contract that every goal honours (`AbstractGoal`), the registry
(`goalRegistry`) and the registration decorator (`registerGoal`). The goals themselves
(optimize, level set, explore) come with the next tasks of roadmap Phase 4, and the recipes of
the first goals with Phase 6.

The rule of this layer: it may import `planexs.core` and `planexs.domain`, and no higher layer.
A goal is configuration, so the layer has no use for torch, BoTorch or Ax.

Importing the package defines the `goal` kind and registers the capability tags of the layer
(`goal:optimize`, `objectives:multi`, ...): that is what `goalRegistry` is imported for
below. A module that registers a goal must be imported here, one line per module, or the goal
is silently unregistered (specification 5.12, 10.6).
"""

from planexs.goals import goalRegistry

__all__ = ["goalRegistry"]
