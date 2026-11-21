"""Outcome transforms: how the outcomes are rescaled before a Gaussian-process surrogate fits
them (specification 5.6, 6).

A Gaussian process is fitted to rescaled outcomes and its predictions are brought back to the
units of the outcomes. A surrogate's configuration holds the transforms by name, as a list
applied in order, so the transforms are the registered entries of the `outcome_transform` kind
of components (specification 5.12): the contract that every transform honours
(`AbstractOutcomeTransformBuilder`), the registry (`outcomeTransformRegistry`), the
registration decorator (`registerOutcomeTransform`) and the transforms themselves. A transform
receives the number of outcomes and returns a native BoTorch `OutcomeTransform`, or `None` for
no transform at all:

- `standardize_v1` (`StandardizeOutcomeTransformBuilder`): zero mean and unit standard
  deviation, with a configurable floor on the standard deviation;
- `log_v1` (`LogOutcomeTransformBuilder`): the natural logarithm of each outcome, which must be
  strictly positive (`StrictlyPositiveLog` refuses anything else);
- `none_v1` (`NoneOutcomeTransformBuilder`): no transform.

`buildOutcomeTransform` turns the list of configurations of a surrogate into the one native
transform that its model is given: the transforms are chained in the order given, and `None`
means "do not rescale", which a surrogate must pass to the model as `outcome_transform=None`
and not by leaving the argument out (BoTorch standardizes the outcomes of a model that is given
no transform).

A module that registers a transform must be imported here, one line per module, or the
transform is silently unregistered (specification 5.12, 10.6). Importing the package also
defines the `outcome_transform` kind, which `outcomeTransformRegistry` is imported for.
"""

from planexs.surrogates.outcome_transforms import (
    LogOutcomeTransformBuilder,
    NoneOutcomeTransformBuilder,
    StandardizeOutcomeTransformBuilder,
    outcomeTransformRegistry,
)

__all__ = [
    "LogOutcomeTransformBuilder",
    "NoneOutcomeTransformBuilder",
    "StandardizeOutcomeTransformBuilder",
    "outcomeTransformRegistry",
]
