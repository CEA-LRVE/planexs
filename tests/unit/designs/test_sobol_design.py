"""Unit tests for `sobol_v1`, its design and its configuration (roadmap task P4-06,
specification 5.8).

The acceptance tests of the task: every point is feasible and decodes to a legal
parameterization, on a space with linear and nonlinear constraints; the same seed gives the
same points; counting existing observations reduces the request. The rest tests what makes
the points Sobol points, and each setting of the configuration.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from design_fakes import (
    assertLegal,
    makeContext,
    makeMixtureSpace,
    makeNonlinearSpace,
    makeSpace,
    makeTrials,
)
from pydantic import ValidationError
from torch.quasirandom import SobolEngine

import planexs.designs  # noqa: F401
from planexs import Real, SearchSpace
from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.designRegistry import designRegistry
from planexs.designs.SobolDesign import SobolDesign
from planexs.designs.SobolDesignConfig import SobolDesignConfig
from planexs.encoding.FeasibleSamplingConfig import FeasibleSamplingConfig


def sobol(n_points: int = 8, **fields: Any) -> SobolDesign:
    """A `sobol_v1` design of the given size, with any other setting replaced."""
    return SobolDesign(SobolDesignConfig(n_points=n_points, **fields))


def unitSquare() -> SearchSpace:
    """Two reals in `[0, 1]`: the encoding is the identity, so the points are the sequence."""
    return SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)])


class TestTheConfiguration:
    def test_the_defaults(self) -> None:
        config = SobolDesignConfig(n_points=8)

        assert config.type == "sobol_v1"
        assert config.scramble is True
        assert config.count_existing_observations is True

    def test_the_size_has_no_default(self) -> None:
        with pytest.raises(ValidationError) as caught:
            SobolDesignConfig()  # type: ignore[call-arg]

        assert [error["loc"] for error in caught.value.errors()] == [("n_points",)]

    @pytest.mark.parametrize("n_points", [0, -1, -50])
    def test_a_size_below_one_is_refused(self, n_points: int) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SobolDesignConfig(n_points=n_points)

    @pytest.mark.parametrize("n_points", [2.5, "many", None])
    def test_a_size_that_is_not_a_whole_number_is_refused(self, n_points: Any) -> None:
        with pytest.raises(ValidationError):
            SobolDesignConfig(n_points=n_points)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="n_point"):
            SobolDesignConfig(n_points=8, n_point=3)  # type: ignore[call-arg]

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ValidationError):
            SobolDesignConfig(type="random_v1", n_points=8)  # type: ignore[arg-type]

    def test_every_field_says_what_it_means(self) -> None:
        for name, field in SobolDesignConfig.model_fields.items():
            assert field.description, name
            assert len(field.description) > 40, name

    def test_the_description_of_the_flag_states_the_limits_of_scrambling(self) -> None:
        description = SobolDesignConfig.model_fields["scramble"].description or ""

        assert "hit-and-run" in description
        assert "corner" in description

    def test_the_resolved_configuration_has_every_default_written_out(self) -> None:
        config = resolveComponent(designRegistry(), {"type": "sobol_v1", "n_points": 8}).config

        assert toResolvedConfig(config) == {
            "type": "sobol_v1",
            "n_points": 8,
            "scramble": True,
            "count_existing_observations": True,
        }

    def test_it_survives_json_and_validates_back_to_an_equal_configuration(self) -> None:
        config = SobolDesignConfig(
            n_points=12, scramble=False, count_existing_observations=False
        )

        text = json.dumps(toResolvedConfig(config))
        back = validateConfig(SobolDesignConfig, json.loads(text))

        assert back == config
        assert toResolvedConfig(back) == toResolvedConfig(config)

    def test_a_configuration_without_a_size_names_the_field(self) -> None:
        with pytest.raises(ConfigError, match="n_points") as caught:
            resolveComponent(designRegistry(), {"type": "sobol_v1"})

        assert caught.value.field_path == "n_points"
        assert "design 'sobol_v1'" in caught.value.message


class TestTheRegisteredDesign:
    def test_the_registry_makes_a_sobol_design_from_a_configuration(self) -> None:
        resolved = resolveComponent(designRegistry(), {"type": "sobol_v1", "n_points": 5})

        design = resolved.entry.builder(resolved.config)

        assert type(design) is SobolDesign
        assert isinstance(design, AbstractDesign)
        assert design.config is resolved.config

    def test_the_entry_declares_no_tag(self) -> None:
        # It proposes points that satisfy whatever constraints it is given.
        entry = designRegistry().lookup("sobol_v1")

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()
        assert entry.config_schema is SobolDesignConfig


class TestPointsNeeded:
    """The size of the design, less the observations that exist if they are counted."""

    @pytest.mark.parametrize(
        ("completed", "expected"),
        [(0, 8), (1, 7), (3, 5), (7, 1), (8, 0), (9, 0), (100, 0)],
    )
    def test_counted_observations_shorten_the_phase_down_to_zero(
        self, completed: int, expected: int
    ) -> None:
        trials = makeTrials(completed=completed)

        assert sobol(8).pointsNeeded(trials) == expected

    @pytest.mark.parametrize("state", ["pending", "failed", "abandoned"])
    def test_only_a_completed_trial_is_an_observation(self, state: str) -> None:
        assert sobol(8).pointsNeeded(makeTrials(**{state: 5})) == 8

    def test_a_mixed_history_counts_only_its_completed_trials(self) -> None:
        trials = makeTrials(completed=2, pending=4, failed=1, abandoned=3)

        assert sobol(8).pointsNeeded(trials) == 6

    @pytest.mark.parametrize("completed", [0, 3, 8, 20])
    def test_with_counting_off_the_design_always_needs_its_whole_size(
        self, completed: int
    ) -> None:
        design = sobol(8, count_existing_observations=False)

        assert design.pointsNeeded(makeTrials(completed=completed)) == 8

    def test_it_reads_only_the_trials_it_is_given(self) -> None:
        assert sobol(8).pointsNeeded([]) == 8
        assert sobol(1).pointsNeeded(makeTrials(completed=1)) == 0


class TestCountingExistingObservationsReducesTheRequest:
    """Acceptance test of the task: a warm start shortens the initial phase."""

    def test_the_whole_design_is_proposed_when_there_is_no_data(self) -> None:
        assert sobol(8).propose(makeContext(makeNonlinearSpace(1.0))).shape[0] == 8

    def test_existing_observations_reduce_the_number_of_points_proposed(self) -> None:
        context = makeContext(makeNonlinearSpace(1.0), trials=makeTrials(completed=3))

        points = sobol(8).propose(context)

        assert points.shape[0] == 5

    def test_without_counting_the_same_history_leaves_the_request_whole(self) -> None:
        context = makeContext(makeNonlinearSpace(1.0), trials=makeTrials(completed=3))

        points = sobol(8, count_existing_observations=False).propose(context)

        assert points.shape[0] == 8

    def test_enough_data_leaves_nothing_to_propose(self) -> None:
        context = makeContext(makeNonlinearSpace(1.0), trials=makeTrials(completed=8))

        points = sobol(8).propose(context)

        assert tuple(points.shape) == (0, context.encoder.n_columns)

    def test_a_trial_that_is_not_an_observation_leaves_the_request_whole(self) -> None:
        context = makeContext(
            makeNonlinearSpace(1.0), trials=makeTrials(pending=4, failed=2, abandoned=1)
        )

        assert sobol(8).propose(context).shape[0] == 8

    def test_an_explicit_count_is_honoured_even_with_data(self) -> None:
        # A user who asks for a batch of four gets four, whatever is left of the phase.
        context = makeContext(makeNonlinearSpace(1.0), trials=makeTrials(completed=8))

        assert sobol(8).propose(context, count=4).shape[0] == 4

    def test_a_count_of_zero_gives_nothing(self) -> None:
        assert sobol(8).propose(makeContext(), count=0).shape[0] == 0


class TestEveryPointIsFeasibleAndLegal:
    """Acceptance test of the task, on a space with linear and nonlinear constraints."""

    def test_on_a_space_with_a_linear_and_a_nonlinear_constraint(self) -> None:
        context = makeContext(makeSpace())

        points = sobol(24).propose(context)

        assert points.shape[0] == 24
        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_for_every_seed(self, seed: int) -> None:
        context = makeContext(makeSpace(), seed=seed)

        points = sobol(16).propose(context)

        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)

    def test_the_decoded_values_are_what_the_constraints_say(self) -> None:
        context = makeContext(makeSpace())

        for row in sobol(24).propose(context):
            values = context.encoder.decode(row)
            assert values["x"] + values["y"] <= 12.0 + 1e-6
            assert values["x"] * values["y"] >= 4.0 - 1e-6

    def test_the_points_are_snapped_onto_valid_values(self) -> None:
        context = makeContext(makeSpace())

        points = sobol(16).propose(context)

        assert torch.allclose(context.encoder.snapColumns(points), points)

    def test_integers_categories_and_reals_all_vary(self) -> None:
        context = makeContext(makeSpace())

        decoded = [context.encoder.decode(row) for row in sobol(32).propose(context)]

        assert all(
            isinstance(values["n"], int) and 1 <= values["n"] <= 9 for values in decoded
        )
        assert {values["kind"] for values in decoded} == {"a", "b", "c"}
        assert len({round(values["x"], 6) for values in decoded}) > 16

    def test_a_mixture_that_sums_to_one_is_met(self) -> None:
        context = makeContext(makeMixtureSpace())

        points = sobol(8).propose(context)

        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)
        assert torch.allclose(points.sum(dim=1), torch.ones(8, dtype=points.dtype))

    def test_a_small_feasible_region_is_found_by_drawing_more_candidates(self) -> None:
        # About one candidate in eleven is feasible, and the first round draws as few as one
        # per point: several rounds are needed, and the sampler asks the draw for more.
        sampling = FeasibleSamplingConfig(oversampling_factor=1, min_draw=1)
        context = makeContext(makeNonlinearSpace(60.0), sampling=sampling)

        points = sobol(6).propose(context)

        assert points.shape[0] == 6
        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)

    @pytest.mark.parametrize("scramble", [True, False])
    def test_the_candidates_of_later_rounds_continue_the_sequence(
        self, scramble: bool
    ) -> None:
        # If each round restarted the sequence it would offer the same candidates again, and
        # the points returned would repeat: no two may be equal.
        sampling = FeasibleSamplingConfig(oversampling_factor=1, min_draw=1)
        context = makeContext(makeNonlinearSpace(60.0), sampling=sampling)

        points = sobol(6, scramble=scramble).propose(context)

        assert torch.unique(points, dim=0).shape[0] == 6

    def test_a_single_precision_runtime_gives_single_precision_points(self) -> None:
        context = makeContext(makeSpace(), runtime=RuntimeConfig(dtype="float32"))

        points = sobol(8).propose(context)

        assert points.dtype == torch.float32
        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)

    def test_an_empty_feasible_region_is_an_error_and_not_a_point_that_breaks_a_constraint(
        self,
    ) -> None:
        context = makeContext(makeNonlinearSpace(1000.0))

        with pytest.raises(RuntimeError, match="no point satisfying"):
            sobol(4).propose(context)

    def test_the_settings_of_the_sampler_are_those_of_the_constraints(self) -> None:
        # One round of one candidate cannot find a region that holds one point in eleven, and
        # the default settings can: the design uses what the constraints were built with.
        space = makeNonlinearSpace(60.0)
        stingy = FeasibleSamplingConfig(max_rounds=1, oversampling_factor=1, min_draw=1)

        with pytest.raises(RuntimeError, match="no point satisfying"):
            sobol(1).propose(makeContext(space, sampling=stingy))
        assert sobol(1).propose(makeContext(space)).shape[0] == 1


class TestTheSameSeedGivesTheSamePoints:
    """Acceptance test of the task."""

    def test_on_a_space_with_linear_and_nonlinear_constraints(self) -> None:
        first = sobol(12).propose(makeContext(makeSpace(), seed=11))
        second = sobol(12).propose(makeContext(makeSpace(), seed=11))

        assert torch.equal(first, second)

    def test_without_constraints(self) -> None:
        first = sobol(8).propose(makeContext(unitSquare(), seed=3))
        second = sobol(8).propose(makeContext(unitSquare(), seed=3))

        assert torch.equal(first, second)

    def test_with_a_nonlinear_constraint_that_takes_several_rounds(self) -> None:
        sampling = FeasibleSamplingConfig(oversampling_factor=1, min_draw=1)

        first = sobol(6).propose(
            makeContext(makeNonlinearSpace(60.0), seed=4, sampling=sampling)
        )
        second = sobol(6).propose(
            makeContext(makeNonlinearSpace(60.0), seed=4, sampling=sampling)
        )

        assert torch.equal(first, second)

    def test_with_an_equality_constraint(self) -> None:
        first = sobol(4).propose(makeContext(makeMixtureSpace(), seed=9))
        second = sobol(4).propose(makeContext(makeMixtureSpace(), seed=9))

        assert torch.equal(first, second)

    def test_two_designs_made_from_the_same_configuration_agree(self) -> None:
        context_a = makeContext(makeSpace(), seed=5)
        context_b = makeContext(makeSpace(), seed=5)

        assert torch.equal(sobol(8).propose(context_a), sobol(8).propose(context_b))

    @pytest.mark.parametrize(
        "space",
        [unitSquare, makeSpace, makeMixtureSpace],
        ids=["no-constraint", "linear-and-nonlinear", "equality"],
    )
    def test_another_seed_gives_other_points(self, space: Any) -> None:
        first = sobol(8).propose(makeContext(space(), seed=1))
        second = sobol(8).propose(makeContext(space(), seed=2))

        assert not torch.equal(first, second)

    def test_a_generator_that_has_been_drawn_from_gives_other_points_than_a_fresh_one(
        self,
    ) -> None:
        # The design draws from the generator, so the same object used twice is not the same
        # stream twice, and a fresh generator of the same seed starts it again.
        context = makeContext(unitSquare(), seed=6)
        design = sobol(8)

        first = design.propose(context)
        second = design.propose(context)
        again = design.propose(makeContext(unitSquare(), seed=6))

        assert not torch.equal(first, second)
        assert torch.equal(first, again)


class TestTheyAreSobolPoints:
    def test_unscrambled_the_points_are_the_classical_sequence(self) -> None:
        points = sobol(8, scramble=False).propose(makeContext(unitSquare()))

        expected = SobolEngine(dimension=2, scramble=False).draw(8).to(torch.float64)
        assert torch.equal(points, expected)

    def test_the_unscrambled_sequence_starts_at_the_corner_of_the_cube(self) -> None:
        points = sobol(4, scramble=False).propose(makeContext(unitSquare()))

        assert points[0].tolist() == [0.0, 0.0]

    def test_unscrambled_the_seed_changes_nothing(self) -> None:
        first = sobol(8, scramble=False).propose(makeContext(unitSquare(), seed=1))
        second = sobol(8, scramble=False).propose(makeContext(unitSquare(), seed=99))

        assert torch.equal(first, second)

    def test_scrambled_the_points_are_not_the_classical_sequence(self) -> None:
        context = makeContext(unitSquare(), seed=1)

        scrambled = sobol(8, scramble=True).propose(context)

        plain = SobolEngine(dimension=2, scramble=False).draw(8).to(torch.float64)
        assert not torch.equal(scrambled, plain)
        assert scrambled[0].tolist() != [0.0, 0.0]

    def test_scrambling_is_the_default(self) -> None:
        by_default = SobolDesign(SobolDesignConfig(n_points=8))
        explicit = sobol(8, scramble=True)

        first = by_default.propose(makeContext(unitSquare(), seed=2))
        second = explicit.propose(makeContext(unitSquare(), seed=2))

        assert torch.equal(first, second)

    @pytest.mark.parametrize("scramble", [True, False])
    @pytest.mark.parametrize("seed", range(6))
    def test_eight_points_are_balanced_over_the_four_quadrants(
        self, scramble: bool, seed: int
    ) -> None:
        # The property of a Sobol sequence that random points lack: its first 2**m points put
        # exactly the same number in each of the equal boxes. Random points are balanced
        # about one time in twenty, so six seeds that all are cannot be luck.
        points = sobol(8, scramble=scramble).propose(makeContext(unitSquare(), seed=seed))

        left = points[:, 0] < 0.5
        below = points[:, 1] < 0.5
        counts = [
            int(((left == in_left) & (below == in_below)).sum())
            for in_left in (True, False)
            for in_below in (True, False)
        ]

        assert counts == [2, 2, 2, 2]

    def test_with_linear_constraints_scrambling_changes_nothing(self) -> None:
        # Documented limit: a hit-and-run chain draws the points inside a polytope, and the
        # sequence is not used.
        space = makeMixtureSpace()

        scrambled = sobol(4, scramble=True).propose(makeContext(space, seed=5))
        plain = sobol(4, scramble=False).propose(makeContext(space, seed=5))

        assert torch.equal(scrambled, plain)

    def test_with_a_nonlinear_constraint_scrambling_still_matters(self) -> None:
        space = makeNonlinearSpace(4.0)

        scrambled = sobol(8, scramble=True).propose(makeContext(space, seed=5))
        plain = sobol(8, scramble=False).propose(makeContext(space, seed=5))

        assert not torch.equal(scrambled, plain)


class TestProposingDoesNotChangeWhatItIsGiven:
    def test_the_history_is_left_as_it_was(self) -> None:
        trials = makeTrials(completed=2, pending=1)
        context = makeContext(makeSpace(), trials=trials)

        sobol(8).propose(context)

        assert list(context.trials) == trials

    def test_a_design_keeps_no_state_from_one_call_to_the_next(self) -> None:
        # The same design object, two fresh contexts of the same seed: the same points.
        design = sobol(8)

        first = design.propose(makeContext(makeSpace(), seed=1))
        design.propose(makeContext(makeSpace(), seed=2))
        again = design.propose(makeContext(makeSpace(), seed=1))

        assert torch.equal(first, again)
