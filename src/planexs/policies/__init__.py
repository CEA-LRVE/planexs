"""Policies: what decides which parameterizations to evaluate next (specification 5.9, 6).

A policy answers one question: given the state of the study, which parameterizations should be
evaluated next (ADR-0013). It is told about every change of a trial's state, and it can export
and import its own state so that persistence works. Which policy a study uses is a choice of
configuration, so this layer is organized around the `policy` kind of components: the contract
that every policy honours (`AbstractPolicy`), the read-only view of the study that a policy
receives (`StudyStateView`), what it returns (`Suggestion`), the registry (`policyRegistry`)
and the registration decorator (`registerPolicy`). The policies themselves arrive one task at
a time in roadmap Phase 6: the composed policy, made of a design, a surrogate, an acquisition
and an optimizer, and the opaque ones, which are any object that honours the contract, of which
random search (`random_policy_v1`, `RandomPolicy`) is the first.

The rule of this layer: it may import every layer below it (specification 5.1), and no higher
one. A policy does not import the study, which drives it: it sees the study through the view.

Importing the package defines the `policy` kind: that is what `policyRegistry` is imported for
below. A module that registers a policy must be imported here, one line per module, or the
policy is silently unregistered (specification 5.12, 10.6).
"""

from planexs.policies import RandomPolicy, policyRegistry

__all__ = ["RandomPolicy", "policyRegistry"]
