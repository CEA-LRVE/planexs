"""Central logging configuration for planexs.

Kept snake_case despite the file otherwise looking like it could be named
after a single symbol: its job is package-wide logging setup (a
`NullHandler` on import, plus an opt-in convenience configurator), not one
class or function meant to be imported and called like a type constructor.
Section 5.1 and section 10 both name this file as the deliberate exception
to the "file name matches its one public symbol" rule.
"""

from __future__ import annotations

import logging

#: Every planexs module logs through `logging.getLogger(__name__)`, all of
#: which are children of this one logger name. A caller can therefore
#: control (or silence) every log message planexs ever emits by configuring
#: just `"planexs"`.
PACKAGE_LOGGER_NAME = "planexs"

# A library must never configure handlers on its own initiative (the
# consuming application decides where logs go); attaching a `NullHandler` is
# the standard-library-recommended way to avoid Python's "No handlers could
# be found" warning for a library that a caller has not configured logging
# for at all.
logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())


def configureLogging(level: int = logging.INFO, *, propagate: bool = True) -> None:
    """Convenience setup for a caller who just wants planexs' logging to
    show up somewhere, without hand-rolling `logging.basicConfig`
    themselves.

    planexs' own library code never calls this (see this module's own
    docstring); it exists for `examples/` scripts and for applications that
    want a one-line starting point, and is safe to call more than once (it
    will not duplicate the stream handler it adds).

    Args:
        level: The logging level to set on planexs' package logger.
        propagate: Whether planexs' log records should also propagate to the
            root logger. Left `True` by default so planexs' logs still show
            up if the caller's own application-wide logging is configured at
            the root logger instead of per-package.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.setLevel(level)
    package_logger.propagate = propagate
    already_configured = any(
        isinstance(handler, logging.StreamHandler) for handler in package_logger.handlers
    )
    if not already_configured:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        package_logger.addHandler(handler)
