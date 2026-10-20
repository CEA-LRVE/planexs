"""Raised when a registry operation is refused."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class RegistryError(PlanexsError):
    """A registry operation is refused.

    Raised for a component name that is already taken or is not a valid name, for a lookup of
    a name that is not registered (the suggestion lists the available names), for a capability
    tag that is not in the vocabulary, and for a plugin that cannot be loaded (specification
    5.12).
    """
