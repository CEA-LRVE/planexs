"""The registry of the `analysis` kind: what is computed over the state of a study."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers analyses in it),
#: so it is a constant and not a setting.
_KIND = "analysis"


def _defineAnalysisKind() -> None:
    """Defines the `analysis` kind.

    The kind introduces no capability tag of its own: what an analysis needs and supports is
    stated with the tags of the layers whose objects it uses (an analysis of a Gaussian
    posterior needs the tag that the surrogates give to one) or of the goal it answers, and
    those are registered where they are defined.
    """
    defineKind(_KIND)


# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
_defineAnalysisKind()


def analysisRegistry() -> Registry[ComponentEntry]:
    """The registry of the `analysis` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is permanent (specification 3.7). The entries are added by the tasks
    that write analyses (the best point, the Pareto front, the level-set estimate, parameter
    importance, prediction grids and leave-one-out diagnostics), and a plugin may add its own:
    the catalogue is open (specification 5.10).

    Returns:
        The registry, whose entries describe the analysis types: the configuration schema of
        each, the builder that makes the `AbstractAnalysis` from a validated configuration and
        what it is built for, and the capability tags the type needs and supports.

    Raises:
        RegistryError: if the process-wide table has no `analysis` kind, which only happens in
            a table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
