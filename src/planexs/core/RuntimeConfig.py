"""The runtime settings of a study: numeric precision, device and determinism."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The syntax of a device name, as torch writes one: a lowercase device type (`cpu`, `cuda`,
#: `mps`, ...) and, optionally, a colon and a device index. Only the syntax is part of this
#: module: which device types exist is a fact about the installed engine, and `core` imports
#: none. Matched with `fullmatch`, so that a trailing newline does not pass.
_DEVICE_NAME = re.compile(r"[a-z][a-z0-9_]*(?::[0-9]+)?")


class RuntimeConfig(BaseModel):
    """The settings that describe where and how a study computes (specification 8, 10.8).

    Three settings that no single component owns but every component of a study shares: the
    floating-point precision of its tensors, the device they live on, and whether the engine
    must use deterministic algorithms only. They are a configuration of the study, not of a
    registered component: this class has no `type` field, is not registered in any kind and is
    not a `ComponentConfig`. A study-level model holds it as one of its fields, and
    `validateConfig` reads it like any other model, with the same field paths in its errors
    (for example `runtime.dtype`).

    It holds **names only**, never torch objects: this layer imports no machine-learning
    library (specification 5.1), and a name is what a snapshot can store and a configuration
    file can say. Turning the names into a `torch.dtype`, a `torch.device` and a call that sets
    the deterministic mode belongs to the `encoding` layer (roadmap task P3-02), which is also
    where a name that torch does not know is reported.

    The defaults are those of decision D-12: double precision, the CPU, and the deterministic
    mode off. They are written in the fields below and nowhere else (specification 3.4). A
    study stores its runtime settings in its resolved configuration, so a later change of a
    default never alters how an old study computes.

    A configuration is frozen: it is shared by every component of a study and must not change
    under them. Build a new one to change a setting.

    Attributes:
        dtype: The floating-point precision, `"float32"` or `"float64"`.
        device: The device name, for example `"cpu"` or `"cuda:0"`.
        deterministic_algorithms: Whether the engine may use deterministic algorithms only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dtype: Literal["float32", "float64"] = Field(
        default="float64",
        description=(
            "The floating-point precision of the tensors that surrogates, acquisitions and "
            "optimizers work on, named as torch names its dtypes without the prefix: "
            "'float32' (32 bits per number) or 'float64' (64 bits). The default is 'float64': "
            "fitting a Gaussian process inverts covariance matrices that are often close to "
            "singular, and double precision keeps them usable (specification 8). 'float32' "
            "halves the memory and is faster on some devices, at the price of less reliable "
            "fits."
        ),
    )
    device: str = Field(
        default="cpu",
        description=(
            "The name of the device the tensors live on, written as torch writes a device: a "
            "lowercase device type, optionally followed by a colon and a device index, for "
            "example 'cpu', 'cuda' or 'cuda:1'. Only the syntax of the name is checked here, "
            "not whether the device exists: that is known when torch builds it. The default "
            "is 'cpu', which every machine has, so that a study runs anywhere with no "
            "configuration (specification 8)."
        ),
    )
    deterministic_algorithms: bool = Field(
        default=False,
        description=(
            "Whether the engine must use deterministic algorithms only, as "
            "`torch.use_deterministic_algorithms` does: results then repeat exactly on one "
            "machine, at a cost in speed, and an operation that has no deterministic "
            "implementation raises an error instead of running. The default is False: the "
            "study seed already makes a study reproducible up to the hardware "
            "nondeterminism that the documentation lists (specification 5.11), which is "
            "enough for most work. Turn it on to remove that remaining nondeterminism."
        ),
    )

    @field_validator("device")
    @classmethod
    def _checkDeviceName(cls, value: str) -> str:
        """Checks that the device is written the way torch writes a device name.

        Args:
            value: The device name.

        Returns:
            The name, unchanged.

        Raises:
            ValueError: if it is not a lowercase device type with an optional `:index`. The
                message names the field and gives examples of valid names.
        """
        if _DEVICE_NAME.fullmatch(value) is None:
            raise ValueError(
                f"device {value!r} is not a device name: write a lowercase device type, "
                "optionally followed by a colon and a device index, for example 'cpu', "
                "'cuda' or 'cuda:1'"
            )
        return value
