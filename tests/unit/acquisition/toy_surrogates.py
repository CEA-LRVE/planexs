"""A toy two-outcome surrogate for the tests of the acquisitions that read a goal.

The surrogate is a real one over a real BoTorch model, so that what an acquisition builds from
it is a native function of the engine. Its parameter is one real `x` in `[0, 1]`, and it models
two outcomes of that one parameter, both known exactly at eight observed points:

- `y = sin(3 x)`: it rises to a single maximum at `x = pi / 6` (about 0.52) and falls on both
  sides of it, so a maximizer and a minimizer want different candidates, and the candidates
  `x = 0.3` and `x = 0.75` have almost the same predicted `y` on either side of the maximum;
- `c = scale * x`: it grows with `x`, so a constraint on it tells the left of the space from
  the right. `scale` stands for the unit the outcome is measured in.

The model is left at the hyperparameters it starts with: the tests are about what the
acquisition does with the model, not about how a model is fitted. Kept in a module of its own,
and not in `conftest.py`, so that the test files can import what is not a fixture.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from torch import Tensor

from planexs import Real, SearchSpace
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchDtype, torchGenerator
from planexs.goals.OptimizeGoal import OptimizeGoal
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.Surrogate import Surrogate

#: The outcome the toy goals optimize.
OBJECTIVE = "y"

#: The outcome the toy goals constrain.
CONSTRAINED = "c"

#: How many points the toy model was trained on.
N_OBSERVATIONS = 8


def makeToySurrogate(
    constraint_scale: float = 1.0,
    objective_only: bool = False,
    observed_x: Sequence[float] | None = None,
) -> Surrogate:
    """A surrogate of the two toy outcomes, over eight observed points.

    Args:
        constraint_scale: The factor that the constrained outcome `c = scale * x` is measured
            in, so that a test can change the unit of an outcome and nothing else.
        objective_only: Whether the surrogate models `y` alone, so that it does not cover the
            constrained outcome `c` that the problem has.
        observed_x: The places of the observed points, in `[0, 1]`. `None` means eight points
            spread over the space. A test that needs a place the model knows well and a place
            it does not gives points on one side only.

    Returns:
        The surrogate, whose native model is in evaluation mode.
    """
    functions: dict[str, Callable[[Tensor], Tensor]] = {
        OBJECTIVE: lambda x: torch.sin(3.0 * x)
    }
    if not objective_only:
        functions[CONSTRAINED] = lambda x: constraint_scale * x
    return makeSurrogateOf(functions, observed_x)


def makeSurrogateOf(
    functions: Mapping[str, Callable[[Tensor], Tensor]],
    observed_x: Sequence[float] | None = None,
) -> Surrogate:
    """A surrogate of outcomes that are known functions of one parameter `x` in `[0, 1]`, at
    observed points that are exact.

    Args:
        functions: The outcomes, by name and in the order of the outputs of the model: each
            maps the observed `x` (shape `(r, 1)`) to its values (shape `(r, 1)`).
        observed_x: The places of the observed points, in `[0, 1]`. `None` means eight points
            spread over the space.

    Returns:
        The surrogate, whose native model is in evaluation mode.
    """
    runtime = RuntimeConfig()
    problem = Problem(
        search_space=SearchSpace([Real("x", 0.0, 1.0)]),
        outcomes=[Outcome(name=name, type="continuous") for name in functions],
    )
    dtype = torchDtype(runtime)
    if observed_x is None:
        train_x = torch.linspace(0.05, 0.95, N_OBSERVATIONS, dtype=dtype).unsqueeze(-1)
    else:
        train_x = torch.tensor(list(observed_x), dtype=dtype).unsqueeze(-1)
    train_y = torch.cat([function(train_x) for function in functions.values()], dim=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InputDataWarning)
        model = SingleTaskGP(
            train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1])
        )
    model.eval()
    return Surrogate(
        BuiltSurrogate(
            model=model, status=FitStatus(n_observations=train_x.shape[0], is_low_data=True)
        ),
        problem=problem,
        encoder=SearchSpaceEncoder(problem.search_space, runtime),
        runtime=runtime,
        outcome_names=tuple(functions),
    )


def makeToyContext(
    surrogate: Surrogate,
    goal: dict[str, Any],
    *,
    pending_points: Tensor | None = None,
    seed: int = 7,
) -> AcquisitionBuildContext:
    """A build context over the toy surrogate for an optimize goal.

    Args:
        surrogate: The surrogate, usually from `makeToySurrogate`.
        goal: The fields of the optimize goal, for example `{"outcomes": {"y": "maximize"}}`.
        pending_points: The pending points, shape `(n, 1)`, or `None` for none.
        seed: The seed of the generator of the build.

    Returns:
        The context.
    """
    return AcquisitionBuildContext(
        surrogate=surrogate,
        goal=OptimizeGoal(OptimizeGoalConfig(**goal)),
        generator=torchGenerator(seed, RuntimeConfig()),
        pending_points=pending_points,
    )


def candidates(*values: float) -> Tensor:
    """Single-point candidates of the toy space, shape `(len(values), 1, 1)`: a batch of
    `len(values)` candidates, each of `q = 1` point.

    Args:
        values: The value of `x` of each candidate.
    """
    dtype = torchDtype(RuntimeConfig())
    return torch.tensor(values, dtype=dtype).reshape(-1, 1, 1)
