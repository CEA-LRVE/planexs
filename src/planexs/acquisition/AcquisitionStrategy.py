"""ABC for how the alpha layer turns a fitted model into candidate points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from planexs.domain.Objective import Objective
from planexs.domain.SearchSpace import SearchSpace

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import AcquisitionFunction
    from botorch.models.model import Model
    from torch import Tensor


class AcquisitionStrategy(ABC):
    """Chooses the next points to evaluate from a fitted surrogate model.

    `AlphaAcquisitionStrategy` is the only implementation in Phase 2. The ABC
    exists so that a caller (or a later phase) can substitute a different way
    of choosing candidates without touching `BayesianOptimizer`.
    """

    @abstractmethod
    def buildAcquisitionFunction(
        self,
        model: Model,
        objective: Objective,
        alpha: float,
        reference_points: Tensor,
    ) -> AcquisitionFunction:
        """Builds the acquisition function for `model` and `objective`: the
        single-objective or multi-objective variant as `objective` requires.

        Args:
            model: A fitted surrogate model over the encoded search space.
            objective: The optimization goal.
            alpha: Exploration weight in `[0, 1]`.
            reference_points: A `(N, d)` encoded, space-filling set used to
                normalize the exploration and exploitation terms.

        Returns:
            The acquisition function, to be maximized.
        """

    @abstractmethod
    def suggest(
        self,
        model: Model,
        objective: Objective,
        search_space: SearchSpace,
        alpha: float,
        n_points: int,
        seed: int | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        """Chooses `n_points` parameterizations to evaluate next.

        Args:
            model: A surrogate fitted on `SearchSpaceEncoder(search_space)`
                encoded data.
            objective: The optimization goal.
            search_space: The space to search, including its constraints.
            alpha: Exploration weight in `[0, 1]`.
            n_points: Batch size.
            seed: Seed making the suggestion reproducible.

        Returns:
            `n_points` parameterizations, each satisfying the search space's
            constraints.
        """

    @abstractmethod
    def suggestSpaceFilling(
        self,
        search_space: SearchSpace,
        n_points: int,
        seed: int | None = None,
    ) -> list[dict[str, float | int | str | bool]]:
        """Chooses `n_points` space-filling, constraint-satisfying
        parameterizations without any model: the cold start, used while there
        is too little data to fit a surrogate.

        Args:
            search_space: The space to search, including its constraints.
            n_points: Batch size.
            seed: Seed making the suggestion reproducible.

        Returns:
            `n_points` parameterizations.
        """
