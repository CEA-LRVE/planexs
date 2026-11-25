"""A synthetic classification problem and the builders that fit it, shared by the tests of the
`variational_gp_classifier_v1` surrogate (roadmap task P3-16, specification 5.6).

The problem has two real parameters on the unit square and one binary outcome, `phase`. A point
is labelled true when it lies inside a disc of radius `RADIUS` around the centre of the square:
a known boundary, curved, so that no linear model can follow it and a surrogate that has
learnt nothing is caught. The points are drawn from a seeded generator, so that every test sees
the same data, and nothing here touches the global random state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from planexs import Real, SearchSpace
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchDtype, torchGenerator
from planexs.surrogates.Surrogate import Surrogate
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext
from planexs.surrogates.surrogateRegistry import surrogateRegistry

#: The registered name of the surrogate under test.
NAME = "variational_gp_classifier_v1"

#: The name of the outcome that the classifier models.
OUTCOME = "phase"

#: The centre and the radius of the disc whose inside is the true class.
CENTRE = 0.5
RADIUS = 0.4

#: The number of labelled points that a classifier is fitted on, and of held-out points that
#: its predictions are scored on.
N_TRAIN = 60
N_TEST = 400

#: The seed of the points that are drawn, and the seed of the fit.
DATA_SEED = 11
FIT_SEED = 3

Point = dict[str, float]


def makeProblem() -> Problem:
    """A problem of two real parameters in `[0, 1]` and one binary outcome, `phase`."""
    return Problem(
        search_space=SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)]),
        outcomes=[Outcome(name=OUTCOME, type="binary")],
    )


def isInside(point: Mapping[str, float]) -> bool:
    """Whether the known boundary labels `point` true: it lies inside the disc."""
    return (point["a"] - CENTRE) ** 2 + (point["b"] - CENTRE) ** 2 < RADIUS**2


def drawPoints(n: int, seed: int) -> list[Point]:
    """`n` points of the unit square, uniform, the same for a given seed."""
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.rand(n, 2, generator=generator, dtype=torch.float64)
    return [{"a": float(row[0]), "b": float(row[1])} for row in drawn]


def labelsOf(
    points: Sequence[Mapping[str, float]], dtype: torch.dtype = torch.float64
) -> Tensor:
    """The labels of `points` as the observations store them: 1.0 or 0.0, a column."""
    return torch.tensor([[1.0 if isInside(point) else 0.0] for point in points], dtype=dtype)


def makeContext(
    *,
    n: int = N_TRAIN,
    seed: int = FIT_SEED,
    runtime: RuntimeConfig | None = None,
    problem: Problem | None = None,
    labels: Tensor | None = None,
    points: Sequence[Point] | None = None,
) -> SurrogateBuildContext:
    """A build context of `n` points of the synthetic problem, labelled by the disc.

    Args:
        n: The number of observations. The points are the first `n` of the seeded data.
        seed: The seed of the generator of the context, which the fit draws from.
        runtime: The runtime settings. `None` means `RuntimeConfig()`.
        problem: The problem. `None` means `makeProblem()`.
        labels: The labels, shape `(n, m)`, replacing those of the disc.
        points: The points, replacing the seeded ones.
    """
    settings = RuntimeConfig() if runtime is None else runtime
    bound = makeProblem() if problem is None else problem
    encoder = SearchSpaceEncoder(bound.search_space, settings)
    chosen = list(drawPoints(N_TRAIN, DATA_SEED)[:n] if points is None else points)
    dtype = torchDtype(settings)
    inputs = torch.stack([encoder.encodeParameters(point) for point in chosen]).to(dtype)
    outcomes = labelsOf(chosen, dtype) if labels is None else labels
    return SurrogateBuildContext(
        problem=bound,
        encoder=encoder,
        train_inputs=inputs,
        train_outcomes=outcomes,
        generator=torchGenerator(seed, settings),
        runtime=settings,
    )


def builderFor(configuration: Mapping[str, Any] | None = None) -> Any:
    """The registered builder of the classifier, made from a configuration that holds the
    fields to change (the type is added), by the path a study takes: the registry.
    """
    resolved = resolveComponent(surrogateRegistry(), {"type": NAME, **(configuration or {})})
    return resolved.entry.builder(resolved.config)


def buildClassifier(
    configuration: Mapping[str, Any] | None = None,
    context: SurrogateBuildContext | None = None,
) -> Surrogate:
    """A fitted classifier wrapped as a `Surrogate`, ready to predict.

    Args:
        configuration: The fields of the configuration to change from their defaults.
        context: The build context. `None` means `makeContext()`.
    """
    built_for = makeContext() if context is None else context
    built = builderFor(configuration).build(built_for)
    return Surrogate(
        built,
        problem=built_for.problem,
        encoder=built_for.encoder,
        runtime=built_for.runtime,
        outcome_names=[OUTCOME],
    )
