"""One measured quantity of a problem: a name, the kind of its values, and a description."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.selectComponent import selectComponent
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.outcomeKindRegistry import outcomeKindRegistry


class Outcome(BaseModel):
    """One measured quantity of a problem, with the kind of its values (specification 5.4).

    An outcome has a name and a kind, and nothing about what the caller wants: the direction
    to optimize and the weights live in the goal, so the same outcome serves any goal
    (ADR-0012). The kind is named by `type`, the registered name of an entry of the
    `outcome_kind` kind of components (`continuous` and `binary` come first, FR24), and the
    kind decides how an observed value is validated, whether a noise standard deviation means
    anything for it, and which capability tag stands for it.

    The type is checked against the registry when the outcome is built, so an outcome never
    names a kind that does not exist: the error lists the registered kinds. It is built with
    keyword arguments, `Outcome(name="yield", type="continuous")`, which is also the form of
    its stored version (`toDict`, `outcomeFromDict`). The names of the outcomes of one problem
    are unique: the problem checks it, not the outcome.

    Attributes:
        name: The outcome's name. It is the key of the outcome in an observation.
        type: The registered name of the outcome kind.
        description: An optional free-text description, for the people who read the study.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    type: str
    description: str | None = None

    @model_validator(mode="after")
    def _validateTypeIsRegistered(self) -> Outcome:
        """Checks that `type` names a registered outcome kind.

        Returns:
            The outcome itself.

        Raises:
            ConfigError: if the type names no registered kind. The message lists the
                registered ones. (Pydantic lets it through as it is.)
        """
        selectComponent(outcomeKindRegistry(), {"type": self.type})
        return self

    def kind(self) -> OutcomeKind:
        """The outcome kind that this outcome names, built through the registry.

        It looks the kind up and builds it at each call, which is why it is a method and not a
        property (specification 10.2).

        Returns:
            The kind: what validates the values of the outcome and says whether a noise
            standard deviation is meaningful for it.

        Raises:
            ConfigError: if the type names no registered kind, which can only happen in a
                registry that was swapped after the outcome was built.
            RegistryError: if the builder registered for the kind does not return an
                `OutcomeKind`.
        """
        resolved = resolveComponent(outcomeKindRegistry(), {"type": self.type})
        kind = resolved.entry.builder(resolved.config)
        if not isinstance(kind, OutcomeKind):
            raise RegistryError(
                f"The builder of the outcome kind {self.type!r} returned "
                f"{type(kind).__name__}, which is not an OutcomeKind.",
                component=f"outcome_kind {self.type!r}",
                suggestion="make the builder return the OutcomeKind the configuration says",
            )
        return kind

    def validateValue(self, value: object) -> float:
        """Checks one observed value of this outcome, as its kind says.

        Args:
            value: The value as the caller gave it.

        Returns:
            The value as it is stored: a `float` (a label is `0.0` or `1.0`).

        Raises:
            OutcomeValueError: if the value is not legal for the kind. The message names this
                outcome.
        """
        return self.kind().validateValue(value, self.name)

    def toDict(self) -> dict[str, Any]:
        """Serializes the outcome.

        Returns:
            A JSON-compatible form, rebuilt by `outcomeFromDict`. The `type` entry is the
            registered name of the kind.
        """
        return {"name": self.name, "type": self.type, "description": self.description}
