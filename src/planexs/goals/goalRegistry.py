"""The registry of the `goal` kind: what a caller can want to know or decide."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers goals in it), so
#: it is a constant and not a setting.
_KIND = "goal"

#: The capability tags that the goal layer introduces, with the one line that documents each
#: (specification 5.12). A goal supports them, and acquisitions, analyses and stopping
#: criteria need them from it: `log_ei_v1` needs a goal that is `goal:optimize` and
#: `objectives:single`, the straddle a goal that is `goal:level_set`. A plugin may register
#: more of its own.
_CAPABILITY_TAGS = {
    "goal:optimize": "The goal is to optimize one or several outcomes.",
    "goal:level_set": "The goal is to find where an outcome crosses a level.",
    "goal:explore": "The goal is to learn the outcomes as well as possible, with no "
    "direction.",
    "objectives:single": "The goal has a single objective: one outcome, or several scalarized "
    "by weights.",
    "objectives:multi": "The goal has several objectives with no preference between them.",
}


def _defineGoalKind() -> None:
    """Defines the `goal` kind and registers the capability tags that go with it.

    The tags are registered here, next to the kind, and not in a module of their own: this
    module is imported by `registerGoal`, which every goal (a plugin's too) imports before it
    declares a tag, so the tags always exist by then.
    """
    registry = defineKind(_KIND)
    for tag, description in _CAPABILITY_TAGS.items():
        registry.vocabulary.register(tag, description)


# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
_defineGoalKind()


def goalRegistry() -> Registry[ComponentEntry]:
    """The registry of the `goal` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is permanent (specification 3.7). The entries are added by the tasks
    that write goals (`optimize_v1`, `level_set_v1`, `explore_v1`), and a plugin may add its
    own: adding a goal type changes no core file (specification 5.7).

    Returns:
        The registry, whose entries describe the goal types: the configuration schema of each,
        the builder that makes the `AbstractGoal` from a validated configuration, and the
        capability tags the type needs and supports.

    Raises:
        RegistryError: if the process-wide table has no `goal` kind, which only happens in a
            table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
