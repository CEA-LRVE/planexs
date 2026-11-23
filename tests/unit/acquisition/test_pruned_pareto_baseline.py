"""Unit tests for `prunedParetoBaseline`, the observed points that can be on the Pareto front,
chosen reproducibly (roadmap task P5-05, specification 5.8)."""

from __future__ import annotations

import torch
from toy_pareto_surrogates import F1, F2, G, makeParetoSurrogate

from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.outcomeConstraintCallables import outcomeConstraintCallables
from planexs.acquisition.ParetoObjectives import ParetoObjectives
from planexs.acquisition.prunedParetoBaseline import prunedParetoBaseline
from planexs.goals.OutcomeConstraint import OutcomeConstraint

NAMES = (F1, F2, G)


def pruned(
    *, seed: int = 11, constraints: list[OutcomeConstraint] | None = None, n_samples: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pruned baseline of the toy surrogate, and its full baseline."""
    model = makeParetoSurrogate(NAMES).model
    baseline = baselinePoints(model)
    objective = ParetoObjectives(
        names=(F1, F2), indices=(0, 1), signs=(1.0, 1.0), num_outputs=3
    ).objective(dtype=baseline.dtype, device=baseline.device)
    callables = outcomeConstraintCallables(model, baseline, constraints or [], NAMES) or None
    kept = prunedParetoBaseline(
        model,
        baseline,
        ref_point=torch.tensor([0.0, 0.0], dtype=baseline.dtype),
        objective=objective,
        constraints=callables,
        n_samples=n_samples,
        seed=seed,
    )
    return kept, baseline


class TestPrunedParetoBaseline:
    def test_it_keeps_the_points_that_can_be_on_the_front_and_drops_the_dominated_ones(
        self,
    ) -> None:
        kept, baseline = pruned()

        # The observed points at x < 0.4 are beaten by the ones to their right in both
        # objectives, whatever the samples say; the ones at x > 0.52 are the front.
        assert kept.shape[0] < baseline.shape[0]
        assert (kept > 0.4).all()
        assert all(any(torch.equal(point, row) for row in kept) for point in baseline[-4:])

    def test_the_points_kept_are_observed_points_in_their_original_order(self) -> None:
        kept, baseline = pruned()

        rows = [int((baseline == point).all(dim=-1).nonzero()[0]) for point in kept]
        assert rows == sorted(rows)

    def test_a_point_predicted_to_violate_a_constraint_is_dropped(self) -> None:
        constraint = OutcomeConstraint(outcome=G, comparator="<=", bound=0.5)

        free, _ = pruned()
        constrained, _ = pruned(constraints=[constraint])

        assert (free > 0.4).all()
        assert (constrained <= 0.5).all()
        assert constrained.shape[0] < free.shape[0]

    def test_the_same_seed_keeps_the_same_points_whatever_the_global_generator_was(
        self,
    ) -> None:
        results = []
        for global_seed in range(12):
            torch.manual_seed(global_seed)
            results.append(pruned(seed=11, n_samples=4)[0])

        assert all(torch.equal(results[0], other) for other in results[1:])

    def test_the_global_generator_is_left_as_it_was(self) -> None:
        torch.manual_seed(5)
        before = torch.get_rng_state()

        pruned()

        assert torch.equal(before, torch.get_rng_state())

    def test_the_global_generator_is_not_the_source_of_its_samples(self) -> None:
        # What a call draws from the global generator afterwards is the same whatever the seed.
        torch.manual_seed(5)
        pruned(seed=1)
        after_one = torch.rand(3)
        torch.manual_seed(5)
        pruned(seed=2)
        after_two = torch.rand(3)

        assert torch.equal(after_one, after_two)
