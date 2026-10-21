"""What the compatibility check needs to know about anything that takes part in it."""

from __future__ import annotations

from typing import Protocol


class Participant(Protocol):
    """Anything that takes part in the compatibility check (specification 5.12).

    The participants of a study are its problem, its goal and the components of its policy.
    The check sees each of them as three things: a label that says who it is in an error
    message, the capability tags it **needs** some participant to support, and the tags it
    **supports**.

    This is a protocol, not a base class, so that a layer does not have to import anything
    to take part: an object with these three read-only attributes is a participant. The tags
    may be plain stored values (see `DeclaredParticipant`) or computed on demand, for example
    from the participant's own validated configuration, which is how a component refines the
    tags it declared. The check reads `needs` and `supports` once per participant, when it
    runs, so it always sees the current configuration.

    Attributes:
        label: Who this is, as it should appear in an error message, for example
            `surrogate 'single_task_gp_v1'` or `problem`.
        needs: The capability tags this participant needs some participant to support.
        supports: The capability tags this participant supports.
    """

    @property
    def label(self) -> str:
        """Who this participant is, for error messages."""
        ...

    @property
    def needs(self) -> frozenset[str]:
        """The capability tags this participant needs some participant to support."""
        ...

    @property
    def supports(self) -> frozenset[str]:
        """The capability tags this participant supports."""
        ...
