"""Turns the list of outcome transforms of a surrogate's configuration into one native one."""

from __future__ import annotations

from collections.abc import Sequence

from botorch.models.transforms.outcome import ChainedOutcomeTransform, OutcomeTransform

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.outcomeTransformRegistry import (
    outcomeTransformRegistry,
)


def buildOutcomeTransform(
    configs: Sequence[ComponentConfig], n_outcomes: int
) -> OutcomeTransform | None:
    """Builds the one native outcome transform that a list of configured transforms stands for
    (specification 5.6).

    A surrogate's configuration holds a list of outcome transforms, applied in the order given:
    the first one rescales the observed outcomes, the second rescales the result, and so on. A
    model takes a single transform, so this is where the list becomes one. Predictions are
    brought back through the same transforms in the reverse order.

    - Each configuration is looked up in the registry of the kind through its `type`, and the
      builder of its entry makes the transform (`AbstractOutcomeTransformBuilder.build`).
    - A transform that is no transform (`none_v1`) is left out of the chain.
    - If one transform is left, it is returned as it is, and not wrapped: a list of one
      `standardize_v1` gives a plain `Standardize`, which is what BoTorch builds by default.
    - If several are left, they are chained in the order given (`ChainedOutcomeTransform`).
      Each is a new object, so the statistics that a stateful transform learns belong to the
      model that applies the chain.
    - If none is left (an empty list, or only `none_v1`), the result is `None`.

    **`None` is "do not rescale", not "use the engine default".** A surrogate must pass the
    result to the model as `outcome_transform=...` in every case. Leaving the argument out when
    the result is `None` would make BoTorch standardize the outcomes, which is the opposite of
    what an empty list says (see `AbstractOutcomeTransformBuilder`).

    The order matters when the transforms do not commute: the logarithm of a standardized
    outcome is undefined where it is negative, so `log_v1` goes before `standardize_v1`.

    Args:
        configs: The validated configurations of the transforms, in the order they are applied:
            for example the value of a field marked `ComponentOf("outcome_transform")` on the
            element of a list. May be empty.
        n_outcomes: The number of outcomes of the surrogate, one or more.

    Returns:
        A new native transform, or `None` if there is nothing to apply.

    Raises:
        ConfigError: if `configs` is a single configuration and not a sequence of them, or if
            `n_outcomes` is not a whole number of one or more.
        RegistryError: if a configuration names a transform that is not registered (the
            message lists the registered ones). A configuration that was validated through a
            `ComponentOf` field cannot: it is refused there, with its path.
        OutcomeTransformError: if the builder of a transform returns something that is not an
            outcome transform.
    """
    if isinstance(configs, ComponentConfig):
        raise ConfigError(
            "A single outcome transform configuration was given where a list of them is "
            "expected.",
            component="outcome_transform",
            field_path="outcome_transform",
            suggestion="wrap it in a list: [config]",
        )
    AbstractOutcomeTransformBuilder.requireOutcomeCount(
        n_outcomes, component="outcome_transform"
    )
    registry = outcomeTransformRegistry()
    transforms: dict[str, OutcomeTransform] = {}
    for position, config in enumerate(configs):
        builder = registry.lookup(config.type).builder(config)
        transform = builder.build(n_outcomes)
        if transform is not None:
            # The position keeps the names unique when a transform is listed twice, and does
            # not shift when a transform that is no transform is left out.
            transforms[f"{position}_{config.type}"] = transform
    if not transforms:
        return None
    if len(transforms) == 1:
        return next(iter(transforms.values()))
    return ChainedOutcomeTransform(**transforms)
