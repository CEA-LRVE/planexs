"""One evaluated point: its parameters and its observed metric value(s)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Trial(BaseModel):
    """One evaluated point: its parameters and its observed metric value(s).

    `result_std`, when known (e.g. from repeated measurements), is passed
    through as a fixed observation noise instead of being inferred by the
    surrogate model.

    Attributes:
        parameters: Maps parameter name to the value this trial used.
        results: Maps metric name to the observed value.
        result_std: Optionally maps metric name to the known standard
            deviation of that observation.
        trial_index: The backend-assigned index of this trial, once it has
            been attached to a backend. `None` for a `Trial` that has not
            been registered with any backend yet.
    """

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, bool | int | float | str]
    results: dict[str, float]
    result_std: dict[str, float] | None = None
    trial_index: int | None = None

    def __init__(
        self,
        parameters: dict[str, bool | int | float | str],
        results: dict[str, float],
        result_std: dict[str, float] | None = None,
        trial_index: int | None = None,
    ) -> None:
        """Creates a trial.

        Args:
            parameters: Parameter name to the value this trial used.
            results: Metric name to the observed value.
            result_std: Optional metric name to the known standard deviation of that
                observation.
            trial_index: The backend-assigned index, once attached to a backend.
        """
        super().__init__(
            parameters=parameters,
            results=results,
            result_std=result_std,
            trial_index=trial_index,
        )
