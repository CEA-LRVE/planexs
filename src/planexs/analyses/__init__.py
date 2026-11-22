"""Analyses: what is computed over the state of a study (specification 5.10, 6).

An analysis is a registered computation over a read-only state of the study that returns a
typed, serializable result: the answer to the goal (the best point, a Pareto front, a
level-set estimate), parameter importance, prediction grids and slices, and model diagnostics.
It may ask the study for a surrogate on demand, and it declares what it needs. The catalogue is
open. This layer is organized around the `analysis` kind of components: the contract that every
analysis honours (`AbstractAnalysis`), the read-only state that it receives (`AnalysisState`),
the way it obtains a surrogate (`SurrogateProvider`), what it returns (`AnalysisResult`), the
registry (`analysisRegistry`) and the registration decorator (`registerAnalysis`). The analyses
themselves arrive one task at a time in roadmap Phase 7.

The rule of this layer: it may import `core`, `domain`, `encoding`, `surrogates` and `goals`
(specification 5.1), and no higher one. In particular it does not import the policies, which
sit above it: it has a state of its own, with what an analysis reads.

Importing the package defines the `analysis` kind: that is what `analysisRegistry` is imported
for below. A module that registers an analysis must be imported here, one line per module, or
the analysis is silently unregistered (specification 5.12, 10.6).
"""

from planexs.analyses import analysisRegistry

__all__ = ["analysisRegistry"]
