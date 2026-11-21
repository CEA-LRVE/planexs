"""The registry of the `acquisition` kind: the scores that rank candidate points."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

# Imported for its effect, which is to register `outcome_constraints:supported`, the third tag
# of this layer, with the goal tags (see `_CAPABILITY_TAGS` below). An acquisition declares it,
# and the registry checks the tags of an entry when the entry is registered, so it must exist
# whenever the `acquisition` kind does, whichever module of the layer is imported first.
from planexs.goals import goalRegistry  # noqa: F401

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers acquisitions in
#: it), so it is a constant and not a setting.
_KIND = "acquisition"

#: The capability tags that the acquisition layer introduces, with the one line that documents
#: each (specification 5.12). An acquisition declares them as `supports`, and the other
#: participants need them from it: an optimizer that follows gradients needs
#: `acquisition:differentiable`, and a policy that builds batches wants `pending:supported`.
#: A plugin may register more of its own.
#:
#: The third tag of the layer, `outcome_constraints:supported`, is not registered here: it is
#: registered with the goal tags, in `planexs.goals.goalRegistry`, because a goal is what
#: needs it (an optimize goal with outcome constraints) and the entry of that goal has to be
#: able to declare it before any acquisition exists (roadmap task P4-03). An acquisition
#: declares it in its `supports` like the two below, and a tag cannot be registered twice. The
#: import above is what makes it exist whenever this kind does.
_CAPABILITY_TAGS = {
    "acquisition:differentiable": "The acquisition function is differentiable in the "
    "candidate point, so a gradient-based optimizer can maximize it.",
    "pending:supported": "The acquisition takes into account the points that were chosen but "
    "not evaluated yet.",
}


def _defineAcquisitionKind() -> None:
    """Defines the `acquisition` kind and registers the capability tags that go with it.

    The tags are registered here, next to the kind, and not in a module of their own: this
    module is imported by `registerAcquisition`, which every acquisition (a plugin's too)
    imports before it declares a tag, so the tags always exist by then.
    """
    registry = defineKind(_KIND)
    for tag, description in _CAPABILITY_TAGS.items():
        registry.vocabulary.register(tag, description)


# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
_defineAcquisitionKind()


def acquisitionRegistry() -> Registry[ComponentEntry]:
    """The registry of the `acquisition` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is a permanent identifier (specification 3.7). The entries are added by
    the tasks that write acquisitions (`log_ei_v1`, `qlog_nei_v1`, `ucb_v1`, the
    exploration/exploitation blend and the others of roadmap Phase 5), and a plugin may add
    its own: adding an acquisition changes no core file (specification 5.8).

    Returns:
        The registry, whose entries describe the acquisitions: the configuration schema of
        each, the builder that makes the `AbstractAcquisitionBuilder` from a validated
        configuration, and the capability tags it needs and supports.

    Raises:
        RegistryError: if the process-wide table has no `acquisition` kind, which only
            happens in a table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
