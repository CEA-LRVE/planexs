"""What a surrogate builder reports about how well it could fit."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FitStatus(BaseModel):
    """How a surrogate was fitted, as plain data (specification 5.6, 3.11).

    A surrogate fitted on very little data says so: the default surrogate "MUST report a
    low-data fit status instead of hiding it". A builder returns one of these next to the
    fitted model (`BuiltSurrogate`), and the `Surrogate` wrapper reports it to the caller.

    What counts as little data is a policy value of the builder (specification 5.6): the
    threshold lives in the configuration of the builder that uses it, never here. This record
    only states the outcome of that decision.

    It is frozen and holds nothing that is not plain data, so it can be logged, compared and
    written as JSON (`model_dump_json`).

    Attributes:
        n_observations: How many observations the model was fitted on, zero or more.
        is_low_data: Whether the builder judged that count too small for the uncertainty
            of the model to be trusted.
        warnings: Messages about the fit that the caller should read, such as why a fit is
            low-confidence or which fallback the builder took. Empty when there is nothing to
            say. No message is blank.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_observations: int = Field(
        strict=True,
        ge=0,
        description="How many observations the model was fitted on, in observations.",
    )
    is_low_data: bool = Field(
        strict=True,
        description="Whether the builder judged the fit to be low-data, hence low-confidence.",
    )
    warnings: tuple[str, ...] = Field(
        default=(),
        description="Messages about the fit that the caller should read. Empty by default.",
    )

    @field_validator("warnings")
    @classmethod
    def _checkWarnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Refuses a blank warning: a message that says nothing is not a warning.

        Args:
            value: The warnings.

        Returns:
            The warnings, unchanged.

        Raises:
            ValueError: if one of them is empty or only whitespace. The message names the
                field and the position.
        """
        for position, message in enumerate(value):
            if not message.strip():
                raise ValueError(
                    f"warnings[{position}] is blank: write what the caller should know, or "
                    "leave the warning out"
                )
        return value
