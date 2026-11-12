"""Fakes shared by the tests of `planexs.surrogates`: problems, contexts, models, builders.

A fake builder stands for a real surrogate (roadmap tasks P3-14, P3-16) that does not exist
yet, so that the contract of the kind (registration, tags, the build context) is tested on its
own. Kept in a module of its own and not in `conftest.py`, so that the test files can import
what is not a fixture.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field, create_model

from planexs import Categorical, Real, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchDtype, torchGenerator
from planexs.surrogates.AbstractSurrogateBuilder import AbstractSurrogateBuilder
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.registerSurrogate import registerSurrogate
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext

#: The encoded columns of `makeSpace`: one for `x`, one for `rate`, three for `solvent`.
N_COLUMNS = 5

#: The outcomes of `makeProblem`, in column order.
OUTCOME_NAMES = ("yield", "cost")

#: The observations of `makeContext`: parameterizations and one value per outcome.
POINTS: list[dict[str, float | str]] = [
    {"x": 0.0, "rate": 1e-3, "solvent": "a"},
    {"x": 2.5, "rate": 1e-2, "solvent": "b"},
    {"x": 7.5, "rate": 1e-1, "solvent": "c"},
    {"x": 10.0, "rate": 1.0, "solvent": "a"},
]
OUTCOMES = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]


def makeSpace(extra_name: str | None = None) -> SearchSpace:
    """A search space with a real, a log-scaled real and a categorical (`N_COLUMNS` columns).

    Args:
        extra_name: If given, one more real parameter of that name, so that two spaces differ.
    """
    parameters: list[Any] = [
        Real("x", 0.0, 10.0),
        Real("rate", 1e-3, 1.0, scaling="log"),
        Categorical("solvent", ["a", "b", "c"]),
    ]
    if extra_name is not None:
        parameters.append(Real(extra_name, 0.0, 1.0))
    return SearchSpace(parameters)


def makeProblem(extra_name: str | None = None) -> Problem:
    """A problem over `makeSpace` with the two continuous outcomes of `OUTCOME_NAMES`."""
    return Problem(
        search_space=makeSpace(extra_name),
        outcomes=[Outcome(name=name, type="continuous") for name in OUTCOME_NAMES],
    )


def makeContext(
    runtime: RuntimeConfig | None = None, **overrides: Any
) -> SurrogateBuildContext:
    """A valid build context of four observations, with any field replaced.

    The inputs are encoded by a real encoder, so they are what a study would hand a builder.

    Args:
        runtime: The runtime settings. `None` means `RuntimeConfig()`.
        **overrides: Fields of `SurrogateBuildContext` to replace, for example
            `train_noise_std=...` or `train_inputs=...`.

    Returns:
        The context.
    """
    settings = RuntimeConfig() if runtime is None else runtime
    problem = makeProblem()
    encoder = SearchSpaceEncoder(problem.search_space, settings)
    dtype = torchDtype(settings)
    fields: dict[str, Any] = {
        "problem": problem,
        "encoder": encoder,
        "train_inputs": torch.stack([encoder.encodeParameters(point) for point in POINTS]),
        "train_outcomes": torch.tensor(OUTCOMES, dtype=dtype),
        "generator": torchGenerator(7, settings),
        "runtime": settings,
    }
    fields.update(overrides)
    return SurrogateBuildContext(**fields)


class FakeModel:
    """A native model that is not one: all the contract tests need is an object."""


def fakeConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real surrogates will write theirs: every field described, the
    `type` field pinned to `name`.

    Args:
        name: The registered name the schema is for.
    """
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config",
        __base__=ComponentConfig,
        type=(Literal[name], Field(default=name, description=f"The registered name, {name}.")),  # type: ignore[valid-type]
        min_observations=(
            int,
            Field(default=3, ge=0, description="How many observations are needed, a count."),
        ),
    )
    return schema


def defineFake(
    name: str = "fake_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset({"posterior:gaussian", "outputs:single"}),
    description: str = "A fake surrogate for tests.",
) -> type[AbstractSurrogateBuilder[Any]]:
    """Registers a fake surrogate in the kind that is current, and returns its class.

    The fake builds a `FakeModel` and reports a low-data fit if it was given fewer observations
    than `min_observations` of its configuration. Call it inside a test that uses the
    `isolatedSurrogateKind` fixture: a registered name is permanent, and must not reach the
    real registry.

    Args:
        name: The registered name.
        needs: The capability tags it needs.
        supports: The capability tags it supports.
        description: Its one-line description.

    Returns:
        The registered class.
    """

    @registerSurrogate(
        name,
        config=fakeConfig(name),
        description=description,
        needs=needs,
        supports=supports,
    )
    class FakeBuilder(AbstractSurrogateBuilder[Any]):
        """A surrogate that fits nothing."""

        def build(self, context: SurrogateBuildContext) -> BuiltSurrogate:
            """Returns a fake model and the fit status that the configuration implies."""
            return BuiltSurrogate(
                model=FakeModel(),  # type: ignore[arg-type]
                status=FitStatus(
                    n_observations=context.n_observations,
                    is_low_data=context.n_observations < self.config.min_observations,
                ),
            )

    return FakeBuilder
