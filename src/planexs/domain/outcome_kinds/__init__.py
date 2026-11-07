"""Outcome kinds: how the values of an outcome are validated (specification 5.4, 6).

An outcome names a kind (`Outcome`, in the parent package), and the kind decides what an
observed value is, whether a noise standard deviation means anything for it, and which
capability tag stands for it. The two built-in kinds, `continuous` and `binary` (decision
D-7), are the registered entries of the `outcome_kind` kind of components
(`registerOutcomeKind`, specification 5.12): a kind is registered when its module is
imported, so every module that holds one is imported here.
"""

from planexs.domain.outcome_kinds import (
    BinaryKind,
    ContinuousKind,
)

__all__ = ["BinaryKind", "ContinuousKind"]
