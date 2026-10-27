"""The random-number tree of a study: one seed, and one stream for each ask and component."""

from __future__ import annotations

import hashlib
import operator
from typing import Any

import numpy as np

from planexs.core.ConfigError import ConfigError


class RngTree:
    """Derives every random stream of a study from one seed (specification 3.7, 5.11).

    A study has a single seed. Each stochastic component asks the tree for its own stream,
    named by the index of the ask and by the path of the component (for example
    `policy.surrogate`), and gets an integer seed or a numpy generator. Nothing is drawn from a
    global random state, so no component can disturb the stream of another, and the same seed
    gives the same streams in every process.

    The derivation is a pure function of the seed, the ask index and the path, with no counter
    and no memory: asking for one stream never changes another, whatever else was asked before
    or since, and adding a component to a study leaves the streams of the others as they were.
    It is:

    - the key of a stream is `(ask_index, H(path))`, where `H` is the SHA-256 digest of the
      UTF-8 path read as one big-endian integer. It is never Python's `hash`, which differs
      from one process to the next;
    - the stream's `SeedSequence` is `SeedSequence(entropy=seed, spawn_key=key)`. `spawn` is
      not used: it numbers the children in the order they are asked for, so a stream would
      depend on the others;
    - the integer seed is the first 64-bit word of that sequence's `generate_state`, and the
      generator is `Generator(PCG64(sequence))`, with the bit generator named so that a change
      of numpy's default cannot move it.

    The derivation is part of what a saved study means: it cannot change without breaking the
    reproducibility of every study, so a change is a deliberate break listed in the changelog
    (ADR-0011).

    A tree holds only its entropy and never changes, so it can be shared freely.

    Args:
        seed: The study seed: a non-negative integer of any size. When it is left out, entropy
            is drawn from the operating system (`SeedSequence()`), and the tree can be rebuilt
            from `entropy`, which is what to record.

    Attributes:
        entropy: The seed of the tree, as an `int`: the seed that was given, or the entropy
            drawn from the operating system when none was. `RngTree(tree.entropy)` is a tree
            with the same streams.

    Raises:
        ConfigError: if `seed` is not a non-negative integer.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Creates the tree, drawing entropy from the operating system if there is no seed."""
        if seed is None:
            sequence = np.random.SeedSequence()
        else:
            sequence = np.random.SeedSequence(_nonNegativeInt(seed, "seed"))
        self._entropy: int = int(sequence.entropy)

    @property
    def entropy(self) -> int:
        """The seed that rebuilds this tree: the one given, or the entropy drawn when made."""
        return self._entropy

    def seed(self, ask_index: int, path: str) -> int:
        """The integer seed of one stream, for code that takes a seed and not a generator.

        Args:
            ask_index: The index of the ask, counted from 0.
            path: The path of the component, for example `policy.surrogate`. Compared exactly,
                as written: any string is a path.

        Returns:
            An integer in `[0, 2**64)`, which `torch.Generator.manual_seed` and
            `SobolEngine` accept. The same for the same arguments, in any process.

        Raises:
            ConfigError: if `ask_index` is not a non-negative integer, or `path` is not a
                string.
        """
        return int(self._sequence(ask_index, path).generate_state(1, np.uint64)[0])

    def generator(self, ask_index: int, path: str) -> np.random.Generator:
        """A numpy generator at the start of one stream.

        Every call returns a new generator, so two calls with the same arguments give two
        generators that produce the same numbers, each from its own beginning.

        Args:
            ask_index: The index of the ask, counted from 0.
            path: The path of the component, for example `policy.surrogate`.

        Returns:
            A `numpy.random.Generator` over a `PCG64` bit generator.

        Raises:
            ConfigError: if `ask_index` is not a non-negative integer, or `path` is not a
                string.
        """
        return np.random.Generator(np.random.PCG64(self._sequence(ask_index, path)))

    def _sequence(self, ask_index: int, path: str) -> Any:
        """The `SeedSequence` of the stream named by an ask index and a path.

        Args:
            ask_index: The index of the ask.
            path: The path of the component.

        Returns:
            The sequence, keyed by the ask index and the hash of the path.

        Raises:
            ConfigError: if an argument is of the wrong kind.
        """
        index = _nonNegativeInt(ask_index, "ask_index")
        if not isinstance(path, str):
            raise ConfigError(
                f"The path of a stream must be a string; got {type(path).__name__}.",
                field_path="path",
                suggestion="give the path of the component, for example 'policy.surrogate'",
            )
        digest = hashlib.sha256(path.encode("utf-8")).digest()
        key = (index, int.from_bytes(digest, "big"))
        return np.random.SeedSequence(entropy=self._entropy, spawn_key=key)


def _nonNegativeInt(value: Any, name: str) -> int:
    """Checks that a value is a whole number that is zero or more, and makes it an `int`.

    Args:
        value: The value to check. Any integer type is accepted, numpy's included; a `bool` is
            not, because `True` is not a seed or an index.
        name: The name of the argument, for the error.

    Returns:
        The value as a Python `int`.

    Raises:
        ConfigError: if it is not.
    """
    try:
        number = operator.index(value)
    except TypeError:
        number = None
    if number is None or isinstance(value, bool) or number < 0:
        raise ConfigError(
            f"The {name} of a random-number tree must be a non-negative integer; "
            f"got {value!r}.",
            field_path=name,
            suggestion="give a whole number that is zero or more",
        )
    return number
