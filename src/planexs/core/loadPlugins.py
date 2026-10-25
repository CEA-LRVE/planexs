"""Imports the plugins that register components (specification 5.12, ADR-0010)."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable
from functools import partial
from importlib.metadata import entry_points

from planexs.core.RegistryError import RegistryError

logger = logging.getLogger(__name__)

#: The entry-point group that a plugin declares to be discovered (decision D-10, ADR-0010). It
#: is a name that other packages write into their own metadata, so like a registered name it
#: is a permanent identifier: it is shared with plugin authors and not tuned by a user, which
#: is why it is a constant and not a setting.
PLUGIN_ENTRY_POINT_GROUP = "planexs.plugins"


def loadPlugins(modules: str | Iterable[str] = ()) -> tuple[str, ...]:
    """Imports every plugin: the entry points of the plugin group, then the modules named here.

    A plugin is a module that registers components when it is imported, with the decorators of
    their kinds (specification 5.12). A package makes its plugin discoverable by declaring an
    entry point whose value is the module, in its packaging metadata::

        [project.entry-points."planexs.plugins"]
        my_kernels = "my_package.kernels"

    Every entry point of that group in the environment is imported, in the order of the entry
    point names, and then each module in `modules`, in the order given: a module that is not
    packaged as a plugin, or one that must come after the installed ones. An entry point may
    also be written `module:attribute`, in which case the attribute must exist, but nothing is
    called: importing is what registers.

    Calling it again is harmless. Python imports a module once, so a second call runs no
    plugin code a second time, does not register anything twice, and loads only what has
    appeared since the first (a package installed, a module added to `modules`).

    Plugins are trusted code (specification 5.12, 8): importing one runs it with the
    permissions of the caller, and every installed package that declares an entry point in the
    group is imported by this call. Install only plugins that you trust.

    A plugin that fails stops the loading. The plugins before it stay loaded, and a plugin that
    registered some components before it failed leaves them registered, so a second call after
    the failure raises again, but possibly with the error of a name registered twice rather
    than the first error. Fix the plugin and start again.

    Args:
        modules: Names of modules to import as plugins, in addition to the entry points: a
            sequence of names, or a single name.

    Returns:
        The names of the plugins imported, in the order they were imported: the entry point
        names, then the module names. The same on every call, for the same plugins.

    Raises:
        RegistryError: if a plugin cannot be imported, whatever the reason (a missing
            dependency, an error in its code, a component that it registers under a name
            already taken), or a module name is not a string. The message names the plugin,
            the distribution that provides it when known, and the original error, which is
            also the `__cause__`. An exception that is not an `Exception` (such as
            `KeyboardInterrupt`) is not caught.
    """
    loaded: list[str] = []
    for entry_point in sorted(
        entry_points(group=PLUGIN_ENTRY_POINT_GROUP),
        key=lambda found: (found.name, found.value),
    ):
        distribution = entry_point.dist
        origin = f" of {distribution.name} {distribution.version}" if distribution else ""
        _importPlugin(
            short=f"plugin {entry_point.name!r}",
            long=f"plugin {entry_point.name!r} (entry point {entry_point.value!r}{origin})",
            importer=entry_point.load,
            suggestion="fix the plugin, or uninstall the package that provides it",
        )
        loaded.append(entry_point.name)

    for module in [modules] if isinstance(modules, str) else modules:
        if not isinstance(module, str):
            raise RegistryError(
                f"A plugin module must be given by its name, a string; got "
                f"{type(module).__name__}.",
                component="plugin",
                suggestion="pass the dotted name of the module, such as 'my_package.kernels'",
            )
        _importPlugin(
            short=f"plugin module {module!r}",
            long=f"plugin module {module!r}",
            importer=partial(importlib.import_module, module),
            suggestion="check the module name, and that the module can be imported on its own",
        )
        loaded.append(module)
    return tuple(loaded)


def _importPlugin(
    short: str, long: str, importer: Callable[[], object], suggestion: str
) -> None:
    """Imports one plugin, turning any failure into a `RegistryError` that names it.

    Args:
        short: The plugin as the error's component hint names it.
        long: The plugin as the message describes it, with where it comes from.
        importer: Imports the plugin: the `load` of an entry point or an import of a module.
        suggestion: What to do about a plugin that fails.

    Raises:
        RegistryError: if the import raises an `Exception`. The original is the cause.
    """
    logger.debug("Importing the %s.", long)
    try:
        importer()
    except Exception as error:
        detail = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        message = f"Cannot load the {long}: {detail}"
        logger.warning(message)
        raise RegistryError(message, component=short, suggestion=suggestion) from error
