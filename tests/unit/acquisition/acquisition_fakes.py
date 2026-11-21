"""Fakes shared by the tests of the acquisition contract: problems, surrogates, contexts,
builders.

A fake builder stands for a real acquisition (roadmap tasks P5-02 and the ones after it) that
does not exist yet, so that the contract of the kind (registration, tags, the build context) is
tested on its own. The surrogate is a real one over a small BoTorch model, unfitted, so that
what a fake builder returns is a native acquisition function of the engine, as the contract
says. Kept in a module of its own and not in `conftest.py`, so that the test files can import
what is not a fixture.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions.warnings import InputDataWarning
from botorch.models import SingleTaskGP
from pydantic import Field, create_model
from torch import Tensor

from planexs import Real, SearchSpace
from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.registerAcquisition import registerAcquisition
from planexs.core.ComponentConfig import ComponentConfig
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

#: The encoded columns of `makeProblem`: one for `x`, one for `z`.
N_COLUMNS = 2

#: The single outcome of `makeProblem`.
OUTCOME = "y"


def makeProblem() -> Problem:
    """A problem over two real parameters with one continuous outcome."""
    return Problem(
        search_space=SearchSpace([Real("x", 0.0, 1.0), Real("z", 0.0, 1.0)]),
        outcomes=[Outcome(name=OUTCOME, type="continuous")],
    )


def makeSurrogate(runtime: RuntimeConfig | None = None) -> Surrogate:
    """A surrogate over `makeProblem`: a real BoTorch model on four points, left unfitted.

    Args:
        runtime: The runtime settings. `None` means `RuntimeConfig()`.

    Returns:
        The surrogate, whose native model is in evaluation mode.
    """
    settings = RuntimeConfig() if runtime is None else runtime
    problem = makeProblem()
    encoder = SearchSpaceEncoder(problem.search_space, settings)
    dtype = torchDtype(settings)
    train_x = torch.tensor([[0.1, 0.2], [0.4, 0.9], [0.7, 0.3], [0.9, 0.8]], dtype=dtype)
    train_y = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=dtype)
    with warnings.catch_warnings():
        # A single-precision model is what a test of the precision wants, and BoTorch warns
        # that double precision is recommended: that is not what is under test.
        warnings.simplefilter("ignore", InputDataWarning)
        model = SingleTaskGP(train_x, train_y)
    model.eval()
    return Surrogate(
        BuiltSurrogate(model=model, status=FitStatus(n_observations=4, is_low_data=True)),
        problem=problem,
        encoder=encoder,
        runtime=settings,
        outcome_names=(OUTCOME,),
    )


def makeGoal() -> OptimizeGoal:
    """An optimize goal for the outcome of `makeProblem`."""
    return OptimizeGoal(OptimizeGoalConfig(outcomes={OUTCOME: "maximize"}))


def points(*rows: list[float], runtime: RuntimeConfig | None = None) -> Tensor:
    """A set of encoded points in the precision of the runtime settings.

    Args:
        *rows: One list of `N_COLUMNS` numbers per point.
        runtime: The runtime settings. `None` means `RuntimeConfig()`.
    """
    dtype = torchDtype(RuntimeConfig() if runtime is None else runtime)
    return torch.tensor(list(rows), dtype=dtype).reshape(len(rows), N_COLUMNS)


def makeContext(
    runtime: RuntimeConfig | None = None, **overrides: Any
) -> AcquisitionBuildContext:
    """A valid build context over `makeSurrogate`, with any field replaced.

    Args:
        runtime: The runtime settings. `None` means `RuntimeConfig()`.
        **overrides: Fields of `AcquisitionBuildContext` to replace, for example
            `pending_points=...` or `reference_points=...`.

    Returns:
        The context, with no pending points and no reference set unless overridden.
    """
    settings = RuntimeConfig() if runtime is None else runtime
    fields: dict[str, Any] = {
        "surrogate": makeSurrogate(settings),
        "goal": makeGoal(),
        "generator": torchGenerator(7, settings),
    }
    fields.update(overrides)
    return AcquisitionBuildContext(**fields)


class ToyAcquisition(AcquisitionFunction):
    """A native acquisition function that scores a point by the sum of its coordinates.

    It keeps what the builder was given, so that a test can see that it was passed through.
    """

    def __init__(self, model: Any, pending: Tensor | None, reference: Tensor | None) -> None:
        super().__init__(model=model)
        self.set_X_pending(pending)
        self.reference = reference

    def forward(self, X: Tensor) -> Tensor:
        """The sum of the coordinates of each candidate."""
        return X.sum(dim=(-2, -1))


def fakeConfig(name: str, *, with_size: bool = False) -> type[ComponentConfig]:
    """A configuration schema as real acquisitions will write theirs: every field described,
    the `type` field pinned to `name`.

    Args:
        name: The registered name the schema is for.
        with_size: Whether the schema has the `n_reference_points` field that a builder with
            a reference set reads its size from.
    """
    fields: dict[str, Any] = {
        "type": (
            Literal[name],
            Field(default=name, description=f"The registered name, {name}."),
        ),
        "beta": (
            float,
            Field(default=2.0, gt=0.0, description="A weight above zero, without a unit."),
        ),
    }
    if with_size:
        fields["n_reference_points"] = (
            int,
            Field(default=16, ge=1, description="The size of the reference set, in points."),
        )
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config", __base__=ComponentConfig, **fields
    )
    return schema


class _FakeBase(AbstractAcquisitionBuilder[Any]):
    """The build of every fake: a toy acquisition over the model of the context, which is
    handed the pending points and, if the class asks for one, the reference set.
    """

    def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
        """Returns a toy acquisition over the model of the context."""
        reference = (
            self.requireReferencePoints(context) if self.needs_reference_points else None
        )
        return ToyAcquisition(context.surrogate.model, context.pending_points, reference)


class _FakeWithReferenceSet(_FakeBase):
    """A fake that needs a reference set, of the size that its configuration gives."""

    needs_reference_points: ClassVar[bool] = True

    def referencePointCount(self) -> int:
        """The `n_reference_points` of the configuration."""
        return int(self.config.n_reference_points)


def defineFake(
    name: str = "fake_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset({"pending:supported"}),
    description: str = "A fake acquisition for tests.",
    with_reference_set: bool = False,
) -> type[AbstractAcquisitionBuilder[Any]]:
    """Registers a fake acquisition in the kind that is current, and returns its class.

    The fake builds a `ToyAcquisition` on the model of the context and hands it the pending
    points and the reference set. Call it inside a test that uses the
    `isolatedAcquisitionKind` fixture: a registered name is permanent, and must not reach the
    real registry.

    Args:
        name: The registered name.
        needs: The capability tags it needs.
        supports: The capability tags it supports.
        description: Its one-line description.
        with_reference_set: Whether the fake needs a reference set, of the size that the
            `n_reference_points` field of its configuration gives.

    Returns:
        The registered class.
    """
    base = _FakeWithReferenceSet if with_reference_set else _FakeBase

    @registerAcquisition(
        name,
        config=fakeConfig(name, with_size=with_reference_set),
        description=description,
        needs=needs,
        supports=supports,
    )
    class FakeBuilder(base):  # type: ignore[valid-type, misc]
        """An acquisition that scores nothing worth scoring."""

    return FakeBuilder  # type: ignore[no-any-return]
