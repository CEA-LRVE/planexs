"""Turns the runtime settings and the study's seeds into torch objects.

`RuntimeConfig` holds names only, because `core` imports no machine-learning library
(specification 5.1). This module is where those names become a `torch.dtype`, a
`torch.device`, a seeded `torch.Generator` and a setting of torch's deterministic mode, and
where a name that torch does not know is reported (roadmap task P3-02). It is kept in one file
with a descriptive name, like `logging_config.py`, because it holds several closely related
functions and no single one of them names it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig


def torchDtype(runtime: RuntimeConfig) -> torch.dtype:
    """The torch floating-point type that the runtime settings name.

    The name is read the way `RuntimeConfig.dtype` documents it: torch's own name for the type,
    without the `torch.` prefix.

    Args:
        runtime: The runtime settings.

    Returns:
        The matching `torch.dtype`.

    Raises:
        ConfigError: if torch has no type of that name. `RuntimeConfig` only lets through the
            names it knows, so this guards a configuration built around its validation.
    """
    dtype = getattr(torch, runtime.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ConfigError(
            f"torch has no floating-point type named {runtime.dtype!r}.",
            component="runtime",
            field_path="dtype",
            suggestion="use 'float32' or 'float64'",
        )
    return dtype


def torchDevice(runtime: RuntimeConfig) -> torch.device:
    """The torch device that the runtime settings name, checked to be usable here.

    `RuntimeConfig` only checks how a device name is written. A name torch cannot read (a
    misspelled device type) and a device that is well formed but absent from this machine
    (`cuda` without a GPU) are both found here, by building the device and allocating an empty
    tensor on it. For a CUDA device that allocation initializes CUDA.

    Args:
        runtime: The runtime settings.

    Returns:
        The device.

    Raises:
        ConfigError: if the device cannot be built or used. The message gives torch's own
            explanation.
    """
    try:
        device = torch.device(runtime.device)
        torch.empty(0, device=device)
    except (RuntimeError, AssertionError) as error:
        detail = (str(error).splitlines() or [type(error).__name__])[0]
        raise ConfigError(
            f"The runtime device {runtime.device!r} cannot be used: {detail}",
            component="runtime",
            field_path="device",
            suggestion=(
                "use 'cpu', or a device that exists on this machine, such as 'cuda' where a "
                "GPU and a CUDA build of torch are present"
            ),
        ) from error
    return device


def torchGenerator(seed: int, runtime: RuntimeConfig) -> torch.Generator:
    """A torch generator on the runtime device, seeded with `seed`.

    The same seed always gives the same stream. The seed is meant to come from
    `RngTree.seed`, which derives one for each ask and component, so that no component draws
    from torch's global random state (specification 3.7).

    Args:
        seed: The seed, an integer as `RngTree.seed` returns it. Torch decides which integers
            it accepts.
        runtime: The runtime settings, which say which device the generator lives on.

    Returns:
        A new generator at the start of its stream.

    Raises:
        ConfigError: if `seed` is not an integer (a `bool` is not one) or torch refuses it, or
            the device cannot be used (see `torchDevice`).
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(
            f"A generator seed must be an integer; got {seed!r}.",
            component="runtime",
            field_path="seed",
            suggestion="use a seed from RngTree.seed",
        )
    generator = torch.Generator(device=torchDevice(runtime))
    try:
        return generator.manual_seed(seed)
    except (ValueError, OverflowError, RuntimeError) as error:
        raise ConfigError(
            f"torch does not accept {seed} as a generator seed: {error}",
            component="runtime",
            field_path="seed",
            suggestion="use a seed from RngTree.seed",
        ) from error


@contextmanager
def deterministicAlgorithms(runtime: RuntimeConfig) -> Iterator[None]:
    """Sets torch's deterministic mode to the runtime setting, and restores it on exit.

    Inside the block torch's mode equals `runtime.deterministic_algorithms`, whatever it was
    before: the study's configuration decides, not whatever the process did earlier. On exit,
    also when the block raises, both the mode and its warn-only flag go back to what they
    were. The blocks can be nested.

    Args:
        runtime: The runtime settings.

    Yields:
        Nothing: the setting holds while the block runs.
    """
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(runtime.deterministic_algorithms)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)
