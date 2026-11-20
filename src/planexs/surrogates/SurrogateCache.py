"""The cache that keeps a study from refitting a surrogate whose data has not changed."""

from __future__ import annotations

import json
import logging
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.toResolvedConfig import toResolvedConfig

if TYPE_CHECKING:
    from planexs.surrogates.Surrogate import Surrogate

logger = logging.getLogger(__name__)

#: Who the errors of this module are about.
_COMPONENT = "surrogate cache"


class SurrogateCache:
    """Gets a fitted surrogate, or builds it once and keeps it (specification 5.6, 8).

    Fitting a Gaussian process is the expensive step of an ask, and a study asks for the
    surrogate more than once for the same data: the composed policy fits it to choose the next
    points, and `Study.surrogate()` hands the caller the very same model. "Refitting a
    surrogate whose data and configuration are unchanged is avoided" (specification 5.6), and
    this is where it is. It is a get-or-build: `getOrBuild` returns the surrogate it holds if
    it was built for the same key, and otherwise calls the builder it is given, keeps the
    result and returns it.

    **The key** is the history version and the resolved configuration of the surrogate:

    - The *history version* is a whole number that the caller keeps. The cache trusts it: it
      must change whenever the observations that the surrogate is fitted on may have changed
      (a study changes it at each state change of a trial) and must not repeat for another
      history.
    - The *resolved configuration* is `toResolvedConfig` of the configuration: every default
      written out, every nested component in full, so that a setting left out and the same
      setting given as its default are one configuration, and a change in a nested kernel is a
      change of the configuration. It is compared as a snapshot taken at each call (canonical
      JSON text), and not as the configuration object: a configuration is mutable, so one that
      was changed in place after the surrogate was built is a different configuration and
      invalidates, where comparing the object with itself would find no change. A `nan` in a
      configuration equals itself in the snapshot, as a setting should.

    Anything else that a fit depends on is not in the key, and is the same for the whole life
    of a cache: the problem, its encoding, the runtime settings. **A cache belongs to one
    study**: share it with nothing else. The random numbers of the fit are not in the key
    either: a fit is reused for its key whatever ask it was made for, so a surrogate whose
    fitting procedure draws random numbers is fitted once for each key and not once for each
    ask.

    **Invalidation.** Any new trial (a new history version) or any change of the configuration
    is a different key, so the surrogate is built again. Only the latest entry is kept: the
    history only grows, so a surrogate of an older version is never asked for again, and the
    model that it holds is released as soon as a newer one replaces it. Going back to a key
    that was replaced is a miss, not a hit.

    **A builder that fails** raises as it raised, and nothing is stored: the cache never holds
    a failure, so the next call tries again (a `ModelNotReadyError` for lack of data, for
    example, is how a policy learns that it must propose design points). The entry that was
    there stays, as it is still right for its own key. The call counts as a miss.

    The cache is not safe to use from several threads at once.

    Attributes:
        hits: How many calls were answered with the entry that the cache held.
        misses: How many calls had to call the builder, whether it succeeded or not.
    """

    def __init__(self) -> None:
        """Creates an empty cache."""
        self._entry: tuple[int, str, Surrogate] | None = None
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """The number of calls answered from the cache (a stored value: cheap, snake_case)."""
        return self._hits

    @property
    def misses(self) -> int:
        """The number of calls that called the builder (a stored value: cheap, snake_case)."""
        return self._misses

    def getOrBuild(
        self,
        history_version: int,
        config: ComponentConfig,
        build: Callable[[], Surrogate],
    ) -> Surrogate:
        """Returns the surrogate for this key, building it if the cache does not hold it.

        Args:
            history_version: The version of the history that the surrogate is fitted on: a
                whole number of zero or more (a Python or NumPy integer) that changes
                whenever that history may have changed.
            config: The validated configuration of the surrogate, whose resolved form is the
                other half of the key.
            build: Fits and returns the surrogate. It is called only when the cache does not
                hold this key, and takes nothing: the caller closes over what it needs.

        Returns:
            The surrogate that the cache holds for this key: the very same object on every
            call with the same key, and the one that `build` returned the first time.

        Raises:
            ConfigError: if `history_version` is not a whole number of zero or more, or
                `config` is not a validated component configuration.
            Exception: whatever `build` raises, unchanged. Nothing is stored then, and the
                entry that the cache held is kept.
        """
        version = _version(history_version)
        snapshot = _snapshot(config)
        entry = self._entry
        if entry is not None and entry[0] == version and entry[1] == snapshot:
            self._hits += 1
            logger.debug("Reusing the surrogate fitted for history version %d.", version)
            return entry[2]
        self._misses += 1
        logger.debug(
            "No surrogate for history version %d and this configuration: building one.",
            version,
        )
        surrogate = build()
        self._entry = (version, snapshot, surrogate)
        return surrogate


def _version(history_version: Any) -> int:
    """Reads a history version: a whole number of zero or more.

    Args:
        history_version: The value to read. Any integer type is accepted, NumPy's included; a
            `bool` is not, because `True` is not a version.

    Returns:
        The version as a Python `int`.

    Raises:
        ConfigError: if it is not a whole number of zero or more.
    """
    try:
        number = operator.index(history_version)
    except TypeError:
        number = None
    if number is None or isinstance(history_version, bool) or number < 0:
        raise ConfigError(
            f"The history version must be a whole number of zero or more; got "
            f"{history_version!r}.",
            component=_COMPONENT,
            field_path="history_version",
            suggestion="pass the version of the study's history, which changes whenever a "
            "trial changes state",
        )
    return number


def _snapshot(config: object) -> str:
    """The resolved configuration, written as canonical text.

    Args:
        config: The configuration of the surrogate.

    Returns:
        Its resolved form (`toResolvedConfig`) as JSON with sorted keys and no spaces: the same
        text for the same configuration, whatever the order its fields were given in, and
        independent of the object, which may change afterwards.

    Raises:
        ConfigError: if `config` is not a component configuration.
    """
    if not isinstance(config, ComponentConfig):
        raise ConfigError(
            f"The configuration of a surrogate must be a validated component configuration "
            f"(a ComponentConfig); got {type(config).__name__}.",
            component=_COMPONENT,
            field_path="config",
            suggestion="validate it first, with resolveComponent(registry, mapping).config",
        )
    return json.dumps(toResolvedConfig(config), sort_keys=True, separators=(",", ":"))
