"""The registry of the `surrogate` kind: the models that can be fitted on observations."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers surrogates in
#: it), so it is a constant and not a setting.
_KIND = "surrogate"

#: The capability tags that the surrogate layer introduces, with the one line that documents
#: each (specification 5.12). They are what a builder passes to `registerSurrogate` as `needs`
#: and `supports`, and what acquisitions, analyses and the problem need from a surrogate. A
#: plugin may register more of its own.
_CAPABILITY_TAGS = {
    "posterior:gaussian": "The posterior over the outcomes at any set of points is jointly "
    "Gaussian.",
    "posterior:latent_gaussian": "The posterior over a latent function is jointly Gaussian, "
    "and the outcomes are read from it through a link.",
    "posterior:marginal_normal": "The posterior at each point has a normal marginal, with a "
    "mean and a standard deviation.",
    "outputs:single": "The surrogate can model a problem with a single outcome.",
    "outputs:multi": "The surrogate can model a problem with several outcomes.",
    "noise:known": "The surrogate can use the observation noise that the caller reports.",
    "noise:inferred": "The surrogate can infer the observation noise from the data.",
}


def _defineSurrogateKind() -> None:
    """Defines the `surrogate` kind and registers the capability tags that go with it.

    The tags are registered here, next to the kind, and not in a module of their own: this
    module is imported by `registerSurrogate`, which every surrogate (a plugin's too) imports
    before it declares a tag, so the tags always exist by then.
    """
    registry = defineKind(_KIND)
    for tag, description in _CAPABILITY_TAGS.items():
        registry.vocabulary.register(tag, description)


# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
_defineSurrogateKind()


def surrogateRegistry() -> Registry[ComponentEntry]:
    """The registry of the `surrogate` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is a permanent identifier (specification 3.7). The entries are added by
    the tasks that write surrogates (`single_task_gp_v1`, `variational_gp_classifier_v1`), and
    a plugin may add its own.

    Returns:
        The registry, whose entries describe the surrogates: the configuration schema of each,
        the builder that makes the `AbstractSurrogateBuilder` from a validated configuration,
        and the capability tags it needs and supports.

    Raises:
        RegistryError: if the process-wide table has no `surrogate` kind, which only happens
            in a table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
