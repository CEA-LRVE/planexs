"""Raised when the chosen components cannot work together."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class CompatibilityError(PlanexsError):
    """The components chosen for a study cannot work together.

    Raised when a study is built (not deep inside an engine call) if a capability need is not
    met by any participant. The message lists every unmet need, who has it, and which
    registered components would meet it (specification 3.6, 5.12).
    """
