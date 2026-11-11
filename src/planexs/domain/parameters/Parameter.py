"""Base class for every kind of search space parameter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from planexs.core.ComponentConfig import ComponentConfig

if TYPE_CHECKING:
    from planexs.domain.ParameterizationCheckConfig import ParameterizationCheckConfig


class Parameter(BaseModel, ABC):
    """Base class for every kind of search space parameter.

    A `Parameter` is a pure data object (Pydantic v2 model): it describes one
    dimension of a `SearchSpace`, independently of any optimization backend.
    Concrete subclasses (`Range`, `Choice`, `Derived`, and their convenience
    sugar `Real`/`Integer`/`Categorical`/`Boolean`/`Fixed`) are mapped to an
    engine's own parameter configuration by that engine's adapter, never
    reimplemented as a parallel abstraction (design philosophy #5; the domain
    knows no engine, specification 3.9).

    This class cannot be instantiated directly: it exists to give every
    parameter kind a common `name`/`default` shape and a common validation
    hook (`_validate_default`), not to be a parameter kind on its own.

    It is also the contract of the `parameter` kind of components (specification 5.3, 6). A
    parameter type is a concrete subclass that is registered under its type name with
    `registerParameter`, together with the configuration schema that is its stored form:

    - `toDict` writes that form, with the type name under the key `type`;
    - `fromConfig` is the builder: it makes the parameter from the validated configuration;
    - `parameterFromDict` reads the form back through the registry, so a type that someone
      registers (a plugin, for example) is saved and reloaded like the built-in ones;
    - `validateValue` says whether a value is legal for the parameter, so that a problem can
      check a parameterization without knowing any type by name (specification 6).

    Two more members are optional, because most types are neither: `is_computed` says that
    the parameter's value is computed from the others and not given, and `activations` says
    which parameters a value of this one switches on (conditional parameters, FR14).

    Attributes:
        name: The parameter's name. Must be unique within a `SearchSpace`.
        default: A per-parameter default value (FR13). Ergonomics only: this
            is a planexs-only concept with no equivalent field on an engine's
            own parameter configuration, so it is not passed to an engine and
            is instead preserved by `persistence/JsonStudyRepository.py` as one
            of the "extras" that an engine's own saved state does not carry.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str
    # Union order matters for Pydantic's smart-mode matching: `bool` first so
    # `True` is not read as an int, `int` before `float` so `3` stays `3`.
    default: bool | int | float | str | None = None

    def __init__(self, **data: Any) -> None:
        """Validates and stores the fields of a concrete parameter.

        Declared explicitly so that type checkers accept the keyword arguments each
        concrete subclass forwards from its own constructor (`bounds`, `values`, ...):
        without it they only know the base class's own fields, `name` and `default`.
        Pydantic still validates everything at runtime.

        Args:
            **data: The fields of the concrete subclass, as keyword arguments.
        """
        super().__init__(**data)

    @classmethod
    def fromConfig(cls, config: ComponentConfig) -> Self:
        """The builder of a registered parameter type: makes the parameter that a validated
        configuration describes.

        This default passes every field of the configuration, except `type`, to the
        constructor as a keyword argument, which is right when the fields of the
        configuration schema are those of the constructor. A type whose stored form differs
        (a `Choice` keeps its conditional parameters as pairs) overrides it. The constructor
        runs the validation that a schema cannot express alone, such as bounds in order.

        Args:
            config: The configuration, an instance of the schema that the type is registered
                with.

        Returns:
            The parameter, an instance of the class this is called on.
        """
        return cls(**config.model_dump(exclude={"type"}))

    @abstractmethod
    def toDict(self) -> dict[str, Any]:
        """Explicit, type-tagged, JSON-compatible form of this parameter.

        The `"type"` entry (`"range"`, `"choice"` or `"derived"`) is the name that the
        parameter type is registered under: it lets `parameterFromDict` rebuild the right
        class on reload, by looking it up in the registry of the `parameter` kind. The rest
        of the form is what the configuration schema of that type accepts, and nothing else.

        Serialization is explicit rather than left to Pydantic because `SearchSpace` holds
        a *list of the abstract base type*: Pydantic would serialize each item by the
        declared type and drop the subclass fields, and would then try to instantiate the
        abstract `Parameter` on reload.

        The convenience subclasses (`Real`, `Integer`, `Categorical`,
        `Boolean`, `Fixed`) are constructors, not round-trip types: they
        serialize as their base kind and reload as `Range` or `Choice`.
        """

    @abstractmethod
    def validateValue(
        self, value: object, options: ParameterizationCheckConfig | None = None
    ) -> None:
        """Checks that `value` is legal for this parameter, as the type says.

        This is the contract that lets `Problem.validateParameterization` check a
        parameterization without naming a type (specification 3.3, 6): a type that a plugin
        registers is checked in the same way as `Range` and `Choice`. It judges one value on
        its own. Whether the parameter is present, or has to be absent, depends on the other
        parameters and is the problem's to judge.

        Args:
            value: The value as the caller gave it.
            options: The tolerances of the check (the grid tolerance of a stepped range, for
                example). `None` means the defaults of `ParameterizationCheckConfig`.

        Raises:
            ParameterValueError: if the value is not legal. The message names the parameter
                and the value, and says what the parameter accepts.
        """

    @property
    def is_computed(self) -> bool:
        """Whether this parameter's value is computed from the others, and not given.

        A computed parameter (`Derived`) is part of a legal parameterization when the study
        produces one, but a caller who reports a point does not have to supply it. `False`
        unless a type says otherwise. Cheap (no work), snake_case.
        """
        return False

    def activations(self) -> Mapping[Any, Sequence[str]]:
        """Which parameters a value of this one switches on.

        Maps a value to the names of the parameters that only exist when the parameter takes
        that value (conditional search spaces, FR14). A parameter that is named in no
        `activations` at all is always there. Empty unless a type says otherwise: most types
        switch nothing on.

        Returns:
            Value to the names of the parameters it switches on. The mapping is the type's own:
            read it, do not change it.
        """
        return {}

    @abstractmethod
    def _validate_default(self) -> None:
        """Raise `ValueError` if `default` is set but not a legal value for
        this parameter (e.g. out of bounds for a `Range`, not a member of
        `values` for a `Choice`). Implemented by every concrete subclass;
        this is a protected implementation hook, not part of the public API
        documented in section 5.2.
        """

    @model_validator(mode="after")
    def _run_default_validation(self) -> Parameter:
        """Runs the subclass's `_validate_default` once the fields are set.

        Returns:
            The parameter itself.
        """
        self._validate_default()
        return self
