"""A parameterization that a policy proposes, with what it knows about it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from planexs.domain.Trial import Trial


@dataclass(frozen=True, eq=True, kw_only=True, slots=True)
class Suggestion:
    """One parameterization that a policy proposes to evaluate, and metadata about it
    (specification 5.9, 6).

    A policy returns suggestions and the study turns each into a pending trial: it checks the
    parameterization against the problem (a policy cannot leave the history in an illegal
    state), gives it a fresh identifier and keeps the metadata with the trial. The metadata is
    what the policy knows about the suggestion and the caller may want to read back, such as
    the phase that produced it: a composed policy writes `{"phase": "design"}` or
    `{"phase": "model"}` (`phase` below reads it).

    A suggestion is built with keywords, `Suggestion(parameters={"x": 0.5})`, and is validated
    as the trial it becomes: `parameters` maps each parameter name to a value of one of the
    four types that a trial holds (a bool, an int, a float or a str), `metadata` holds
    JSON-compatible data, and a non-finite number in either is refused, because JSON has none
    and a snapshot would change it. What a suggestion is checked against is therefore what a
    trial checks, and the study never receives one that it cannot record. Whether the
    parameterization fits a problem is the study's check, and the policy's guarantee.

    It is immutable: no attribute can be assigned, and the mappings it holds are deep read-only
    copies of the ones it was given (a mapping becomes a read-only view, a JSON array becomes a
    tuple), so neither the policy that built it nor the study that reads it can change it
    through the other.

    Attributes:
        parameters: Maps parameter name to the suggested value, read-only.
        metadata: What the policy knows about the suggestion, JSON-compatible and read-only.
            Empty by default.
    """

    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates `parameters` and `metadata` as a trial does, and stores read-only copies.

        Raises:
            ValueError: if a parameter value is not a bool, an int, a float or a str, a
                number is not finite, `metadata` is not JSON-compatible, or a key is not a
                string. It is the error of the trial that the suggestion would become, a
                pydantic `ValidationError` that names the field and the key at fault.
        """
        # The trial of this check is a probe: it is never kept, and its identifier is not a
        # property of a suggestion. It is built so that the two checks are one and the same.
        probe = Trial(id=0, parameters=self.parameters, metadata=self.metadata)
        # Frozen dataclass: `object.__setattr__` is the sanctioned way to normalize.
        object.__setattr__(self, "parameters", probe.parameters)
        object.__setattr__(self, "metadata", probe.metadata)

    @property
    def phase(self) -> str | None:
        """The phase that produced the suggestion, as the policy tagged it in the metadata
        under `phase`, or `None` if it did not (a lookup: cheap, snake_case).
        """
        value = self.metadata.get("phase")
        return value if isinstance(value, str) else None
