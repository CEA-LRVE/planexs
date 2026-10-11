"""Reconstructs a planexs `SearchSpace` from Ax's own core `Parameter` objects."""

from __future__ import annotations

from typing import Any, Literal, cast

from ax.core.parameter import ChoiceParameter, DerivedParameter, FixedParameter, RangeParameter
from ax.core.parameter import ParameterType as AxParameterType

from planexs.domain.parameters.Choice import Choice
from planexs.domain.parameters.Derived import Derived
from planexs.domain.parameters.Fixed import Fixed
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.Range import Range
from planexs.domain.SearchSpace import SearchSpace

_AX_TO_PLANEXS_TYPE: dict[AxParameterType, Literal["float", "int", "str", "bool"]] = {
    AxParameterType.FLOAT: "float",
    AxParameterType.INT: "int",
    AxParameterType.STRING: "str",
    AxParameterType.BOOL: "bool",
}


def fromAxSearchSpace(ax_parameters: list[Any]) -> SearchSpace:
    """The inverse of `toAxSearchSpace`, used when reconstructing a
    `SearchSpace` from a study loaded through `Client.load_from_json_file`
    (section 5.6).

    Accepts Ax's own *core* parameter objects (`ax.core.parameter.
    RangeParameter`/`ChoiceParameter`/`FixedParameter`/`DerivedParameter`,
    e.g. `list(client._experiment.search_space.parameters.values())`), not
    the `ax.api.configs` config objects `toAxSearchSpace` produces: those are
    write-only inputs to `configure_experiment`, while the experiment Ax
    actually holds (and serializes) is made of the core objects instead.

    Scope note (flagged for review per section 12): a `Parameter`'s `default`
    (FR13) and a `Choice`'s `dependent_parameters` (FR14) have no equivalent
    field on Ax's core parameter objects either (mirroring `ax.api.configs`'
    own lack of a `default` field), so neither round-trips through this
    function. `JsonStudyRepository` therefore treats planexs' own
    `StudySnapshot.search_space` (kept alongside, not re-derived from Ax's
    state) as the source of truth for those two pieces of metadata, and only
    uses this function where a planexs `SearchSpace` must be derived from a
    `Client` that planexs did not itself create (e.g. one built directly
    through the `axClient` escape hatch, ADR-0005).
    """
    parameters: list[Parameter] = []
    for ax_parameter in ax_parameters:
        parameters.append(_fromAxParameter(ax_parameter))
    return SearchSpace(parameters=parameters)


def _fromAxParameter(ax_parameter: Any) -> Parameter:
    """Converts one Ax core parameter into the matching planexs `Parameter`.

    Args:
        ax_parameter: An Ax `RangeParameter`, `FixedParameter`, `ChoiceParameter` or
            `DerivedParameter`.

    Returns:
        The planexs parameter.

    Raises:
        ValueError: if a fixed or choice parameter has no value, or a range has a string or
            bool type.
        TypeError: for any other kind of Ax parameter.
    """
    if isinstance(ax_parameter, RangeParameter):
        return Range(
            name=ax_parameter.name,
            bounds=(float(ax_parameter.lower), float(ax_parameter.upper)),
            parameter_type=_axRangeType(ax_parameter.parameter_type),
            step_size=getattr(ax_parameter, "step_size", None),
            scaling="log" if getattr(ax_parameter, "log_scale", False) else None,
        )
    if isinstance(ax_parameter, FixedParameter):
        if ax_parameter.value is None:
            raise ValueError(f"Ax FixedParameter {ax_parameter.name!r} has no value.")
        return Fixed(name=ax_parameter.name, value=ax_parameter.value)
    if isinstance(ax_parameter, ChoiceParameter):
        if any(value is None for value in ax_parameter.values):
            raise ValueError(f"Ax ChoiceParameter {ax_parameter.name!r} has a None value.")
        return Choice(
            name=ax_parameter.name,
            # Ax types its values as one mixed list; a Choice holds a homogeneous one.
            values=cast(
                "list[int] | list[float] | list[str] | list[bool]", ax_parameter.values
            ),
            parameter_type=_axChoiceType(ax_parameter.parameter_type),
            is_ordered=ax_parameter.is_ordered,
        )
    if isinstance(ax_parameter, DerivedParameter):
        return Derived(
            name=ax_parameter.name,
            expression=ax_parameter.expression_str,
            parameter_type=_axChoiceType(ax_parameter.parameter_type),
        )
    raise TypeError(
        f"No planexs Parameter mapping is defined for Ax parameter kind "
        f"{type(ax_parameter).__name__!r}."
    )


def _axChoiceType(ax_type: AxParameterType) -> Literal["float", "int", "str", "bool"]:
    """Maps an Ax parameter type to the planexs type name used by `Choice` and `Derived`.

    Args:
        ax_type: The Ax parameter type.

    Returns:
        `"float"`, `"int"`, `"str"` or `"bool"`.
    """
    return _AX_TO_PLANEXS_TYPE[ax_type]


def _axRangeType(ax_type: AxParameterType) -> Literal["float", "int"]:
    """Maps an Ax parameter type to the planexs type name used by `Range`.

    Args:
        ax_type: The Ax parameter type.

    Returns:
        `"float"` or `"int"`.

    Raises:
        ValueError: if the type is a string or a bool, which a `Range` cannot hold.
    """
    planexs_type = _AX_TO_PLANEXS_TYPE[ax_type]
    if planexs_type == "float":
        return "float"
    if planexs_type == "int":
        return "int"
    raise ValueError(
        f"Ax RangeParameter reported parameter_type={ax_type!r}, which has "
        "no float/int planexs Range equivalent."
    )
