"""Helpers shared by the tests of the `likelihood` kind (roadmap task P3-11).

Kept next to `surrogate_fakes.py`, in the directory whose `conftest.py` puts it on the import
path, so that the tests of `likelihoods/` can import it by name.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.surrogates.likelihoods.AbstractLikelihoodBuilder import AbstractLikelihoodBuilder
from planexs.surrogates.likelihoods.LikelihoodBuildContext import LikelihoodBuildContext
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry


def makeContext(
    *kinds: str,
    noise: list[float] | torch.Tensor | None = None,
    batch_shape: tuple[int, ...] = (),
    runtime: RuntimeConfig | None = None,
) -> LikelihoodBuildContext:
    """A build context with one outcome per kind (a single continuous one by default).

    Args:
        *kinds: The registered outcome kinds, one outcome each, named `y0`, `y1`, ...
        noise: The known noise variances, as a list (read in the precision of the runtime
            settings) or as a tensor. `None` means the noise is inferred.
        batch_shape: The batch shape of the likelihood.
        runtime: The runtime settings. `None` means `RuntimeConfig()`.
    """
    settings = RuntimeConfig() if runtime is None else runtime
    dtype = torch.float32 if settings.dtype == "float32" else torch.float64
    variances = torch.tensor(noise, dtype=dtype) if isinstance(noise, list) else noise
    return LikelihoodBuildContext(
        outcomes=tuple(
            Outcome(name=f"y{position}", type=kind)
            for position, kind in enumerate(kinds or ("continuous",))
        ),
        runtime=settings,
        batch_shape=torch.Size(batch_shape),
        noise_variance=variances,
    )


def builderOf(configuration: Mapping[str, Any]) -> AbstractLikelihoodBuilder[Any]:
    """The builder that a configuration mapping describes, resolved through the registry."""
    resolved = resolveComponent(likelihoodRegistry(), configuration)
    builder: AbstractLikelihoodBuilder[Any] = resolved.entry.builder(resolved.config)
    return builder
