"""The contract of the `outcome_kind` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

from planexs.core.ComponentConfig import ComponentConfig


class OutcomeKind(ABC):
    """How the values of an outcome of one kind are validated (specification 5.4, 6).

    An outcome has a name and a kind, and the kind decides what an observed value is, whether
    a noise standard deviation means anything for it, and which tag of the capability
    vocabulary stands for it. A kind is a concrete subclass of this class, registered under its
    name with `registerOutcomeKind`, together with the configuration schema that holds its
    settings (the two built-in kinds have none). It provides:

    - `validateValue`, which checks one observed value and returns the value as it is stored;
    - `allows_noise`, whether a noise standard deviation is meaningful for the outcome;
    - `capability_tag`, the tag that stands for the kind in the capability vocabulary. A
      problem that has an outcome of the kind needs it, and a surrogate that can model such an
      outcome supports it, so that the compatibility check can meet the two (specification
      5.12). `registerOutcomeKind` registers the tag in the vocabulary.

    The kind decides, and an outcome only names it: `Outcome` holds the name of a kind and
    builds the kind through the registry of the `outcome_kind` kind when it needs it.

    Attributes:
        capability_tag: The capability tag of the kind, `outcome_kind:<name>`.
        allows_noise: Whether a noise standard deviation is meaningful for an outcome of the
            kind (an observation may give one only where the kind allows it).
    """

    capability_tag: ClassVar[str]
    allows_noise: ClassVar[bool]

    @classmethod
    def fromConfig(cls, config: ComponentConfig) -> Self:
        """The builder of a registered outcome kind: makes the kind that a validated
        configuration describes.

        This default passes every field of the configuration, except `type`, to the
        constructor as a keyword argument. A kind with no settings has none to pass.

        Args:
            config: The configuration, an instance of the schema that the kind is registered
                with.

        Returns:
            The kind, an instance of the class this is called on.
        """
        return cls(**config.model_dump(exclude={"type"}))

    @abstractmethod
    def validateValue(self, value: object, outcome: str) -> float:
        """Checks one observed value of an outcome of this kind.

        Args:
            value: The value as the caller gave it.
            outcome: The name of the outcome, which the error names.

        Returns:
            The value as it is stored: a `float`, so that every outcome reads the same way
            whatever its kind (a label is stored as `0.0` or `1.0`).

        Raises:
            OutcomeValueError: if the value is not legal for the kind. The message names the
                outcome and the value.
        """
