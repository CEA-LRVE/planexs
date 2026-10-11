"""The alpha layer: picks candidates by optimizing the exploration/exploitation blend."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.model import Model
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.sampling import manual_seed
from torch import Tensor

from planexs.acquisition.AcquisitionStrategy import AcquisitionStrategy
from planexs.acquisition.EncodedConstraints import EncodedConstraints
from planexs.acquisition.ExplorationExploitationAcquisition import (
    ExplorationExploitationAcquisition,
)
from planexs.acquisition.modelParetoFront import modelParetoFront
from planexs.acquisition.MultiObjectiveExplorationExploitationAcquisition import (
    MultiObjectiveExplorationExploitationAcquisition,
)
from planexs.acquisition.sampleFeasibleEncoded import (
    DEFAULT_POLYTOPE_BURN_IN,
    DEFAULT_POLYTOPE_THINNING,
    sampleFeasibleEncoded,
)
from planexs.domain.Objective import Objective
from planexs.domain.SearchSpace import SearchSpace
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder

logger = logging.getLogger(__name__)

_Parameterization = dict[str, float | int | str | bool]


class AlphaAcquisitionStrategy(AcquisitionStrategy):
    """Chooses candidates by maximizing `ExplorationExploitationAcquisition`
    (or its multi-objective variant) over the encoded search space. FR5, FR10,
    FR17.

    Which acquisition function is used follows the `Objective` (design
    philosophy #3, one concept for N = 1 and N > 1):

    - One metric: `ExplorationExploitationAcquisition`, in that metric's
      direction.
    - Several metrics with explicit `weights`: the weights express a scalar
      preference, exactly as `toAxOptimizationConfig` turns them into a weighted
      sum for Ax. The same single-objective class is used on the weighted sum
      of the (direction-signed) metrics.
    - Several metrics, no weights: `MultiObjectiveExplorationExploitationAcquisition`
      (hypervolume improvement).

    A batch is built sequentially: each chosen point is passed to the
    acquisition function as pending (fantasized) before the next is optimized,
    so points are neither copies of one another nor independent of one another.

    Search-space handling:

    - Unordered choices (categorical, boolean) are enumerated with BoTorch's
      `optimize_acqf_mixed`: every combination if there are at most
      `max_categorical_combinations`, otherwise a random subset of that size.
    - Integer ranges, grid ranges and ordered choices are optimized as
      continuous relaxations and rounded afterwards. If rounding breaks a
      constraint, neighbouring roundings are tried, then the best feasible
      random point.
    - `LinearConstraint`s are passed to BoTorch as linear constraints.
      `NonlinearConstraint`s are passed as nonlinear inequality constraints
      with feasible starting points (ADR-0006); every returned candidate is
      checked against all constraints, falling back to the best feasible
      random point if the optimizer's result is not feasible.
    - Conditional parameters (`dependent_parameters`) are optimized as if all
      were active; the inactive ones are dropped from each returned candidate.

    Args:
        num_restarts: Number of optimizer restarts per candidate.
        raw_samples: Number of raw samples used to pick restart points, and the
            size of the feasible fallback pool.
        n_reference_points: Size of the space-filling set the two acquisition
            terms are normalized over.
        n_mc_samples: Number of Monte Carlo samples of the acquisition
            function's joint-posterior term (used once a batch has pending
            points).
        max_categorical_combinations: Cap on enumerated categorical assignments.
        max_iterations: Iteration cap of each optimizer run.
        polytope_burn_in: Burn-in steps of the hit-and-run sampler that draws starting
            points inside linear constraints (only used when there are some). BoTorch's
            own default is 10000. See `DEFAULT_POLYTOPE_BURN_IN`.
        polytope_thinning: Steps between two kept points of that sampler. BoTorch's own
            default is 32. See `DEFAULT_POLYTOPE_THINNING`.
    """

    def __init__(
        self,
        num_restarts: int = 10,
        raw_samples: int = 512,
        n_reference_points: int = 512,
        n_mc_samples: int = 256,
        max_categorical_combinations: int = 32,
        max_iterations: int = 200,
        polytope_burn_in: int = DEFAULT_POLYTOPE_BURN_IN,
        polytope_thinning: int = DEFAULT_POLYTOPE_THINNING,
    ) -> None:
        """Stores the budgets and sampler settings of the optimizer.

        Every argument is described in the class docstring.
        """
        self._polytope_burn_in = polytope_burn_in
        self._polytope_thinning = polytope_thinning
        self._num_restarts = num_restarts
        self._raw_samples = raw_samples
        self._n_reference_points = n_reference_points
        self._n_mc_samples = n_mc_samples
        self._max_categorical_combinations = max_categorical_combinations
        self._max_iterations = max_iterations

    def buildAcquisitionFunction(
        self,
        model: Model,
        objective: Objective,
        alpha: float,
        reference_points: Tensor,
    ) -> AcquisitionFunction:
        """Builds the acquisition function that fits `objective`.

        One metric gives `ExplorationExploitationAcquisition` in that metric's direction.
        Several metrics with explicit weights give the same class on the weighted sum of the
        direction-signed metrics (a scalar preference, as Ax is given). Several metrics without
        weights give `MultiObjectiveExplorationExploitationAcquisition`.

        Args:
            model: A fitted surrogate model over the encoded search space.
            objective: The optimization goal.
            alpha: Exploration weight in `[0, 1]`.
            reference_points: A `(N, d)` encoded, space-filling set used to normalize the
                exploration and exploitation terms.

        Returns:
            The acquisition function, to be maximized.

        Raises:
            ValueError: if the model's output count differs from the objective's metric count.
        """
        metrics = objective.metrics
        if model.num_outputs != len(metrics):
            raise ValueError(
                f"The model has {model.num_outputs} output(s) but the objective has "
                f"{len(metrics)} metric(s); they must match."
            )
        signs = torch.tensor(
            [-1.0 if metric.minimize else 1.0 for metric in metrics], dtype=torch.double
        )
        sampler = SobolQMCNormalSampler(torch.Size([self._n_mc_samples]))

        if not objective.is_multi_objective:
            return ExplorationExploitationAcquisition(
                model=model,
                alpha=alpha,
                minimize=metrics[0].minimize,
                reference_points=reference_points,
                sampler=sampler,
            )
        if objective.weights is not None:
            weights = torch.tensor(objective.weights, dtype=torch.double) * signs
            return ExplorationExploitationAcquisition(
                model=model,
                alpha=alpha,
                minimize=False,
                reference_points=reference_points,
                sampler=sampler,
                posterior_transform=ScalarizedPosteriorTransform(weights=weights),
            )
        front = modelParetoFront(model, signs)
        return MultiObjectiveExplorationExploitationAcquisition(
            model=model,
            alpha=alpha,
            objective_weights=signs,
            ref_point=infer_reference_point(front),
            reference_points=reference_points,
            sampler=sampler,
        )

    def suggest(
        self,
        model: Model,
        objective: Objective,
        search_space: SearchSpace,
        alpha: float,
        n_points: int,
        seed: int | None = None,
    ) -> list[_Parameterization]:
        """Chooses `n_points` parameterizations by optimizing the acquisition function.

        Points are chosen one at a time: each is handed back to the acquisition function as
        pending before the next is optimized. Every returned point satisfies the search space's
        constraints and is a legal parameterization (rounded onto grids, with conditional
        parameters pruned).

        Args:
            model: A surrogate fitted on `SearchSpaceEncoder(search_space)`-encoded data.
            objective: The optimization goal.
            search_space: The space to search, including its constraints.
            alpha: Exploration weight in `[0, 1]`.
            n_points: Batch size.
            seed: Seed making the suggestion reproducible. `None` leaves torch's random state
                alone.

        Returns:
            `n_points` parameterizations.

        Raises:
            ValueError: if `alpha` is outside `[0, 1]`, `n_points < 1`, or the model does not
                match the objective.
            RuntimeError: if the constraints leave no feasible point to start from.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha!r}.")
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1; got {n_points!r}.")

        encoder = SearchSpaceEncoder(search_space)
        constraints = EncodedConstraints(search_space, encoder)
        # `manual_seed(None)` leaves torch's random state alone, so no special case for it.
        with manual_seed(seed):
            reference_points = sampleFeasibleEncoded(
                encoder,
                constraints,
                self._n_reference_points,
                seed=seed,
                snap=True,
                n_burnin=self._polytope_burn_in,
                n_thinning=self._polytope_thinning,
            )
            acquisition = self.buildAcquisitionFunction(
                model, objective, alpha, reference_points
            )
            chosen: list[Tensor] = []
            for position in range(n_points):
                step_seed = None if seed is None else seed + 1 + position
                relaxed = self._optimizeOne(acquisition, encoder, constraints, step_seed)
                point = self._resolveCandidate(
                    acquisition, encoder, constraints, relaxed, step_seed
                )
                chosen.append(point)
                acquisition.set_X_pending(torch.stack(chosen))
        return [encoder.decode(point) for point in chosen]

    def suggestSpaceFilling(
        self,
        search_space: SearchSpace,
        n_points: int,
        seed: int | None = None,
    ) -> list[_Parameterization]:
        """Chooses `n_points` constraint-satisfying scrambled-Sobol points, without a model.

        Args:
            search_space: The space to search, including its constraints.
            n_points: Batch size.
            seed: Seed making the suggestion reproducible.

        Returns:
            `n_points` parameterizations.

        Raises:
            ValueError: if `n_points < 1`.
            RuntimeError: if no point satisfying the constraints can be found.
        """
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1; got {n_points!r}.")
        encoder = SearchSpaceEncoder(search_space)
        constraints = EncodedConstraints(search_space, encoder)
        points = sampleFeasibleEncoded(
            encoder,
            constraints,
            n_points,
            seed=seed,
            snap=True,
            n_burnin=self._polytope_burn_in,
            n_thinning=self._polytope_thinning,
        )
        return [encoder.decode(point) for point in points]

    def _optimizeOne(
        self,
        acquisition: AcquisitionFunction,
        encoder: SearchSpaceEncoder,
        constraints: EncodedConstraints,
        seed: int | None,
    ) -> Tensor:
        """Maximizes `acquisition` for one relaxed candidate, returned as `(d,)`."""
        options: dict[str, bool | float | int | str] = {"maxiter": self._max_iterations}
        if seed is not None:
            options["seed"] = seed
        if constraints.inequality is not None or constraints.equality is not None:
            # BoTorch's defaults (10000 burn-in steps, thinning 32) for sampling
            # starting points inside a linear polytope are far more than these
            # small problems need and dominate the run time.
            options["n_burnin"] = self._polytope_burn_in
            options["n_thinning"] = self._polytope_thinning
        arguments: dict[str, Any] = {
            "acq_function": acquisition,
            "bounds": encoder.unit_bounds,
            "q": 1,
            "num_restarts": self._num_restarts,
            "raw_samples": self._raw_samples,
            "options": options,
            "inequality_constraints": constraints.inequality,
            "equality_constraints": constraints.equality,
            "nonlinear_inequality_constraints": constraints.nonlinear,
        }
        if constraints.nonlinear is not None:
            # BoTorch cannot draw feasible starting points for a nonlinear
            # constraint by itself: it needs a generator that only proposes them.
            def feasibleStarts(n: int, q: int, generator_seed: int | None) -> Tensor:
                draws = sampleFeasibleEncoded(
                    encoder,
                    constraints,
                    n * q,
                    seed=generator_seed,
                    snap=False,
                    n_burnin=self._polytope_burn_in,
                    n_thinning=self._polytope_thinning,
                )
                return draws.reshape(n, q, -1)

            arguments["ic_generator"] = partial(
                gen_batch_initial_conditions, generator=feasibleStarts
            )

        fixed_features_list = encoder.categoricalFixedFeatures(
            self._max_categorical_combinations, seed
        )
        if fixed_features_list:
            candidate, _ = optimize_acqf_mixed(
                fixed_features_list=fixed_features_list, **arguments
            )
        else:
            candidate, _ = optimize_acqf(**arguments)
        return candidate.detach().reshape(-1)

    def _resolveCandidate(
        self,
        acquisition: AcquisitionFunction,
        encoder: SearchSpaceEncoder,
        constraints: EncodedConstraints,
        relaxed: Tensor,
        seed: int | None,
    ) -> Tensor:
        """Turns an optimizer result into a valid, constraint-satisfying point."""
        snapped = encoder.snapColumns(relaxed)
        if not constraints.has_constraints or bool(constraints.isFeasible(snapped)):
            return snapped

        neighbors = encoder.roundingNeighbors(relaxed)
        feasible = constraints.isFeasible(neighbors)
        if bool(feasible.any()):
            candidates = neighbors[feasible]
            distance = (candidates - relaxed).pow(2).sum(dim=-1)
            return candidates[int(distance.argmin())]

        logger.warning(
            "The optimizer's candidate does not satisfy the search space's constraints after "
            "rounding; falling back to the best of %d random feasible points.",
            self._raw_samples,
        )
        pool = sampleFeasibleEncoded(
            encoder,
            constraints,
            self._raw_samples,
            seed=seed,
            snap=True,
            n_burnin=self._polytope_burn_in,
            n_thinning=self._polytope_thinning,
        )
        with torch.no_grad():
            values = acquisition(pool.unsqueeze(1))
        return pool[int(values.argmax())]
