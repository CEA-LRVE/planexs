"""Raised when a configuration is invalid or cannot be read."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class ConfigError(PlanexsError):
    """A configuration is invalid or cannot be read.

    Raised for a field that fails validation (unknown field, wrong type, value out of range),
    for a component `type` that names nothing, and for a configuration file that cannot be
    parsed. The `field_path` hint carries the path of the offending field (specification 5.12).
    """
