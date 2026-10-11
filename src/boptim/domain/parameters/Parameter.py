"""Base class for every kind of search space parameter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class Parameter(BaseModel, ABC):
    """Base class for every kind of search space parameter.

    A `Parameter` is a pure data object (Pydantic v2 model): it describes one
    dimension of a `SearchSpace`, independently of any optimization backend.
    Concrete subclasses (`Range`, `Choice`, `Derived`, and their convenience
    sugar `Real`/`Integer`/`Categorical`/`Boolean`/`Fixed`) are mapped to Ax's
    own `ax.api.configs` parameter configs by `backends/ax/toAxSearchSpace.py`,
    never reimplemented as a parallel abstraction (design philosophy #5).

    This class cannot be instantiated directly: it exists to give every
    parameter kind a common `name`/`default` shape and a common validation
    hook (`_validate_default`), not to be a parameter kind on its own.

    Attributes:
        name: The parameter's name. Must be unique within a `SearchSpace`.
        default: A per-parameter default value (FR13). Ergonomics only: this
            is a boptim-only concept with no equivalent field on Ax's own
            parameter configs, so it is not passed to Ax and is instead
            preserved by `persistence/JsonStudyRepository.py` as one of the
            "extras" Ax's own snapshot does not carry.
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

    @abstractmethod
    def toDict(self) -> dict[str, Any]:
        """Explicit, type-tagged, JSON-compatible form of this parameter.

        The `"kind"` entry (`"range"`, `"choice"` or `"derived"`) lets
        `parameterFromDict` rebuild the right class on reload. Serialization
        is explicit rather than left to Pydantic because `SearchSpace` holds
        a *list of the abstract base type*: Pydantic would serialize each
        item by the declared type and drop the subclass fields, and would
        then try to instantiate the abstract `Parameter` on reload.

        The convenience subclasses (`Real`, `Integer`, `Categorical`,
        `Boolean`, `Fixed`) are constructors, not round-trip types: they
        serialize as their base kind and reload as `Range` or `Choice`.
        """

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
