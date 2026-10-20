"""The base class of every error that planexs raises."""

from __future__ import annotations


class PlanexsError(Exception):
    """Base class of every error that planexs raises on purpose.

    A caller who wants to handle "anything planexs rejected" catches this one class
    (specification 3.6, 10.10). A message says what went wrong; the optional hints say where
    (which component, which configuration field) and how to fix it. `str()` renders the
    message first and then each hint that was given on a line of its own, always in the order
    component, field, suggestion::

        Unknown surrogate type 'gp'.
        Component: surrogate
        Field: policy.surrogate.type
        Suggestion: use one of: single_task_gp_v1, variational_gp_classifier_v1

    A hint that is `None` is left out. A message is not edited: if it spans several lines, it
    is rendered as given.

    Args:
        message: What went wrong, in a sentence a user can act on.
        component: The component the error is about: its kind, its registered name, or both.
        field_path: The dotted path of the offending configuration field, for example
            `policy.surrogate.kernel.nu`.
        suggestion: A way to fix the problem.

    Attributes:
        message: The message, without the hints.
        component: The component hint, or `None`.
        field_path: The field-path hint, or `None`.
        suggestion: The suggestion hint, or `None`.
    """

    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        field_path: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Creates an error; the hints are keyword-only so that a call site names each one."""
        super().__init__(message)
        self.message = message
        self.component = component
        self.field_path = field_path
        self.suggestion = suggestion

    def __str__(self) -> str:
        """Renders the message, then each hint that was given, one per line."""
        lines = [self.message]
        if self.component is not None:
            lines.append(f"Component: {self.component}")
        if self.field_path is not None:
            lines.append(f"Field: {self.field_path}")
        if self.suggestion is not None:
            lines.append(f"Suggestion: {self.suggestion}")
        return "\n".join(lines)
