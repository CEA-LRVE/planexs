"""Writes a validated configuration as its fully resolved, JSON-compatible dictionary."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def toResolvedConfig(config: BaseModel) -> dict[str, Any]:
    """The fully resolved configuration of `config`, which is what a snapshot stores.

    The result is a plain dictionary of dictionaries, lists, strings, numbers, booleans and
    `None`, which `json.dumps` writes and `json.loads` reads back unchanged. Every field is
    written out, the ones the caller left to a default included, so that a later change of a
    default in the code never alters how a stored configuration behaves. Every nested component
    is written out in full with its own `type`, however deep, because a component field is
    dumped as the schema of the component it holds and not as the base class it is declared
    with (see `ComponentOf`). Tuples are written as lists, and `validateConfig` reads them back
    as tuples.

    A non-finite float is kept as a float, which Python's `json` writes as `Infinity` or
    `NaN`: an extension to JSON that Python reads back, and that other readers may refuse.

    `validateConfig(type(config), toResolvedConfig(config))` is equal to `config`, and
    resolving that again gives the same dictionary.

    Args:
        config: A validated configuration: the schema of a component, or any model that holds
            components.

    Returns:
        A new dictionary, with the fields in the order the model declares them.
    """
    return config.model_dump(mode="json")
