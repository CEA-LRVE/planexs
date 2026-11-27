"""A toy classifier surrogate over the one-parameter space of `toy_surrogates`, for the tests
of the acquisitions that read the latent space of a surrogate with a link.

The surrogate is a real `variational_gp_classifier_v1`, made by the path a study takes (the
registry) and fitted for real on a seeded set of labelled points: the label of a point is true
when `x` is above `BOUNDARY`. Its latent function therefore crosses 0, the latent threshold of
the probability 0.5, near `BOUNDARY`, and is far from it on either side of it. Kept in a module
of its own, and not in `conftest.py`, so that the test files can import what is not a fixture.
"""

from __future__ import annotations

import torch

from planexs import Real, SearchSpace
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchDtype, torchGenerator
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig
from planexs.surrogates.Surrogate import Surrogate
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext
from planexs.surrogates.surrogateRegistry import surrogateRegistry

#: The binary outcome that the classifier models.
LABEL = "phase"

#: The value of `x` above which the label is true.
BOUNDARY = 0.5

#: How many labelled points the classifier is fitted on.
N_LABELLED = 40


def makeClassifierSurrogate(seed: int = 3) -> Surrogate:
    """A classifier of the label of `x`, fitted on `N_LABELLED` seeded points.

    Args:
        seed: The seed of the fit.

    Returns:
        The surrogate, whose native model is in evaluation mode and whose link is probit.
    """
    runtime = RuntimeConfig()
    problem = Problem(
        search_space=SearchSpace([Real("x", 0.0, 1.0)]),
        outcomes=[Outcome(name=LABEL, type="binary")],
    )
    encoder = SearchSpaceEncoder(problem.search_space, runtime)
    dtype = torchDtype(runtime)
    draws = torch.rand(
        N_LABELLED, 1, generator=torch.Generator().manual_seed(11), dtype=torch.float64
    ).to(dtype)
    labels = (draws > BOUNDARY).to(dtype)
    context = SurrogateBuildContext(
        problem=problem,
        encoder=encoder,
        train_inputs=draws,
        train_outcomes=labels,
        generator=torchGenerator(seed, runtime),
        runtime=runtime,
    )
    resolved = resolveComponent(surrogateRegistry(), {"type": "variational_gp_classifier_v1"})
    built = resolved.entry.builder(resolved.config).build(context)
    return Surrogate(
        built, problem=problem, encoder=encoder, runtime=runtime, outcome_names=[LABEL]
    )


def makeClassifierContext(
    surrogate: Surrogate, level: float | None = None, seed: int = 7
) -> AcquisitionBuildContext:
    """A build context over a classifier surrogate for a level-set goal on its label.

    Args:
        surrogate: The surrogate, from `makeClassifierSurrogate`.
        level: The probability of the level, or `None` for the default of a binary outcome.
        seed: The seed of the generator of the build.
    """
    return AcquisitionBuildContext(
        surrogate=surrogate,
        goal=LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=LABEL, level=level)),
        generator=torchGenerator(seed, RuntimeConfig()),
    )
