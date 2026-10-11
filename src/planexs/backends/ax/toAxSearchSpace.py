"""Maps a planexs `SearchSpace`'s parameters to Ax's own parameter configs."""

from __future__ import annotations

from typing import Any

from ax.api.configs import ChoiceParameterConfig, DerivedParameterConfig, RangeParameterConfig

from planexs.domain.parameters.Choice import Choice
from planexs.domain.parameters.Derived import Derived
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.Range import Range
from planexs.domain.SearchSpace import SearchSpace


def toAxSearchSpace(search_space: SearchSpace) -> list[Any]:
    """Maps planexs `Parameter`s to Ax's `RangeParameterConfig` /
    `ChoiceParameterConfig` / `DerivedParameterConfig` list. Return type kept
    as `Any` to avoid leaking an Ax import into this file's own public
    signature.

    `Real`/`Integer` (subclasses of `Range`) and `Categorical`/`Boolean`/
    `Fixed` (subclasses of `Choice`) are handled by their base class's own
    `isinstance` check: they are sugar over `Range`/`Choice`, not distinct Ax
    concepts.

    `search_space.constraints` (`LinearConstraint`, FR9) is deliberately not
    handled here: `Client.configure_experiment` takes parameters and
    parameter constraints as two separate arguments, and each `Constraint`
    subclass already knows how to render itself (see
    `LinearConstraint.toAxParameterConstraintString`), so
    `AxBackend.createExperiment` builds that second list directly from
    `search_space.constraints` rather than this function returning a tuple
    that would not match section 5.3's documented `-> list[Any]` signature.
    """
    ax_parameters: list[Any] = []
    for parameter in search_space.parameters:
        ax_parameters.append(_toAxParameterConfig(parameter))
    return ax_parameters


def _toAxParameterConfig(parameter: Parameter) -> Any:
    """Converts one planexs `Parameter` into the matching Ax parameter config.

    Args:
        parameter: A `Range`, `Choice` or `Derived` (or a subclass such as `Real`).

    Returns:
        The Ax `RangeParameterConfig`, `ChoiceParameterConfig` or `DerivedParameterConfig`.

    Raises:
        TypeError: if the parameter is of no known kind.
    """
    if isinstance(parameter, Range):
        return RangeParameterConfig(**parameter.toAxKwargs())
    if isinstance(parameter, Choice):
        return ChoiceParameterConfig(**parameter.toAxKwargs())
    if isinstance(parameter, Derived):
        return DerivedParameterConfig(**parameter.toAxKwargs())
    raise TypeError(
        f"No Ax parameter config mapping is defined for parameter kind "
        f"{type(parameter).__name__!r} (parameter {parameter.name!r})."
    )
