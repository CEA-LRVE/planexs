"""Unit tests for `random_v1`, its design and its configuration (roadmap task P4-07,
specification 5.8, 3.10).

The acceptance tests of the task are those of `sobol_v1` (`test_sobol_design.py`), run on this
design: every point is feasible and decodes to a legal parameterization, on a space with linear
and nonlinear constraints; the same seed gives the same points; counting existing observations
reduces the request. On top of them, the test that the two designs are not aliases: on the same
seed they do not give the same points. The rest tests what makes the points uniform random
points, and each setting of the configuration (all of them, less `scramble`, which only a Sobol
sequence has).
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
from planexs import Categorical, Real, SearchSpace
from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.designRegistry import designRegistry
from planexs.designs.RandomDesign import RandomDesign
from planexs.designs.RandomDesignConfig import RandomDesignConfig
from planexs.designs.SobolDesign import SobolDesign
from planexs.designs.SobolDesignConfig import SobolDesignConfig
from planexs.encoding.FeasibleSamplingConfig import FeasibleSamplingConfig


def randomDesign(n_points: int = 8, **fields: Any) -> RandomDesign:
    """A `random_v1` design of the given size, with any other setting replaced."""
    return RandomDesign(RandomDesignConfig(n_points=n_points, **fields))


def sobol(n_points: int = 8) -> SobolDesign:
    """A `sobol_v1` design of the given size."""
    return SobolDesign(SobolDesignConfig(n_points=n_points))


def unitSquare() -> SearchSpace:
    """Two reals in `[0, 1]`: the encoding is the identity, so the points are the raw draws."""
    return SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)])


class TestTheConfiguration:
    def test_the_defaults(self) -> None:
        config = RandomDesignConfig(n_points=8)

        assert config.type == "random_v1"
        assert config.count_existing_observations is True

    def test_the_size_has_no_default(self) -> None:
        with pytest.raises(ValidationError) as caught:
            RandomDesignConfig()  # type: ignore[call-arg]

        assert [error["loc"] for error in caught.value.errors()] == [("n_points",)]

    @pytest.mark.parametrize("n_points", [0, -1, -50])
    def test_a_size_below_one_is_refused(self, n_points: int) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            RandomDesignConfig(n_points=n_points)

    @pytest.mark.parametrize("n_points", [2.5, "many", None])
    def test_a_size_that_is_not_a_whole_number_is_refused(self, n_points: Any) -> None:
        with pytest.raises(ValidationError):
            RandomDesignConfig(n_points=n_points)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="n_point"):
            RandomDesignConfig(n_points=8, n_point=3)  # type: ignore[call-arg]

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ValidationError):
            RandomDesignConfig(type="sobol_v1", n_points=8)  # type: ignore[arg-type]

    def test_scrambling_is_not_an_option_of_random_points(self) -> None:
        with pytest.raises(ValidationError, match="scramble"):
            RandomDesignConfig(n_points=8, scramble=True)  # type: ignore[call-arg]

    def test_the_options_are_those_of_sobol_v1_less_scrambling(self) -> None:
        random_fields = set(RandomDesignConfig.model_fields)
        sobol_fields = set(SobolDesignConfig.model_fields)

        assert random_fields == sobol_fields - {"scramble"}

    def test_the_options_that_both_have_mean_and_default_the_same(self) -> None:
        for name in ("n_points", "count_existing_observations"):
            random_field = RandomDesignConfig.model_fields[name]
            sobol_field = SobolDesignConfig.model_fields[name]

            assert random_field.is_required() == sobol_field.is_required(), name
            assert random_field.default == sobol_field.default, name
            assert random_field.metadata == sobol_field.metadata, name

    def test_every_field_says_what_it_means(self) -> None:
        for name, field in RandomDesignConfig.model_fields.items():
            assert field.description, name
            assert len(field.description) > 40, name

    def test_the_resolved_configuration_has_every_default_written_out(self) -> None:
        config = resolveComponent(
            designRegistry(), {"type": "random_v1", "n_points": 8}
        ).config

        assert toResolvedConfig(config) == {
            "type": "random_v1",
            "n_points": 8,
            "count_existing_observations": True,
        }

    def test_it_survives_json_and_validates_back_to_an_equal_configuration(self) -> None:
        config = RandomDesignConfig(n_points=12, count_existing_observations=False)

        text = json.dumps(toResolvedConfig(config))
        back = validateConfig(RandomDesignConfig, json.loads(text))

        assert back == config
        assert toResolvedConfig(back) == toResolvedConfig(config)

    def test_a_configuration_without_a_size_names_the_field(self) -> None:
        with pytest.raises(ConfigError, match="n_points") as caught:
            resolveComponent(designRegistry(), {"type": "random_v1"})

        assert caught.value.field_path == "n_points"
        assert "design 'random_v1'" in caught.value.message


class TestTheRegisteredDesign:
    def test_the_registry_makes_a_random_design_from_a_configuration(self) -> None:
        resolved = resolveComponent(designRegistry(), {"type": "random_v1", "n_points": 5})

        design = resolved.entry.builder(resolved.config)

        assert type(design) is RandomDesign
        assert isinstance(design, AbstractDesign)
        assert design.config is resolved.config

    def test_the_entry_declares_no_tag(self) -> None:
        # It proposes points that satisfy whatever constraints it is given.
        entry = designRegistry().lookup("random_v1")

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()
        assert entry.config_schema is RandomDesignConfig

    def test_it_is_a_design_of_its_own_next_to_sobol_v1(self) -> None:
        random_entry = designRegistry().lookup("random_v1")
        sobol_entry = designRegistry().lookup("sobol_v1")

        assert random_entry.name != sobol_entry.name
        assert random_entry.config_schema is not sobol_entry.config_schema
        assert random_entry.builder is not sobol_entry.builder
        assert random_entry.description != sobol_entry.description

    def test_the_two_built_in_designs_are_the_whole_catalogue(self) -> None:
        assert {"random_v1", "sobol_v1"} <= set(designRegistry().names())


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

        assert randomDesign(8).pointsNeeded(trials) == expected

    @pytest.mark.parametrize("state", ["pending", "failed", "abandoned"])
    def test_only_a_completed_trial_is_an_observation(self, state: str) -> None:
        assert randomDesign(8).pointsNeeded(makeTrials(**{state: 5})) == 8

    def test_a_mixed_history_counts_only_its_completed_trials(self) -> None:
        trials = makeTrials(completed=2, pending=4, failed=1, abandoned=3)

        assert randomDesign(8).pointsNeeded(trials) == 6

    @pytest.mark.parametrize("completed", [0, 3, 8, 20])
    def test_with_counting_off_the_design_always_needs_its_whole_size(
        self, completed: int
    ) -> None:
        design = randomDesign(8, count_existing_observations=False)

        assert design.pointsNeeded(makeTrials(completed=completed)) == 8

    def test_it_reads_only_the_trials_it_is_given(self) -> None:
        assert randomDesign(8).pointsNeeded([]) == 8
        assert randomDesign(1).pointsNeeded(makeTrials(completed=1)) == 0

    @pytest.mark.parametrize("completed", [0, 1, 4, 8, 12])
    @pytest.mark.parametrize("counting", [True, False])
    def test_it_asks_for_the_same_number_of_points_as_sobol_v1(
        self, completed: int, counting: bool
    ) -> None:
        trials = makeTrials(completed=completed, pending=2, failed=1)
        sobol_design = SobolDesign(
            SobolDesignConfig(n_points=8, count_existing_observations=counting)
        )

        random_design = randomDesign(8, count_existing_observations=counting)

        assert random_design.pointsNeeded(trials) == sobol_design.pointsNeeded(trials)


class TestCountingExistingObservationsReducesTheRequest:
    """Acceptance test of the task: a warm start shortens the initial phase."""

    def test_the_whole_design_is_proposed_when_there_is_no_data(self) -> None:
        assert randomDesign(8).propose(makeContext(makeNonlinearSpace(1.0))).shape[0] == 8

    def test_existing_observations_reduce_the_number_of_points_proposed(self) -> None:
        context = makeContext(makeNonlinearSpace(1.0), trials=makeTrials(completed=3))

        points = randomDesign(8).propose(context)

        assert points.shape[0] == 5

    def test_without_counting_the_same_history_leaves_the_request_whole(self) -> None:
        context = makeContext(makeNonlinearSpace(1.0), trials=makeTrials(completed=3))

        points = randomDesign(8, count_existing_observations=False).propose(context)

        assert points.shape[0] == 8

    def test_enough_data_leaves_nothing_to_propose(self) -> None:
        context = makeContext(makeNonlinearSpace(1.0), trials=makeTrials(completed=8))

        points = randomDesign(8).propose(context)

        assert tuple(points.shape) == (0, context.encoder.n_columns)

    def test_a_trial_that_is_not_an_observation_leaves_the_request_whole(self) -> None:
        context = makeContext(
            makeNonlinearSpace(1.0), trials=makeTrials(pending=4, failed=2, abandoned=1)
        )

        assert randomDesign(8).propose(context).shape[0] == 8

    def test_an_explicit_count_is_honoured_even_with_data(self) -> None:
        # A user who asks for a batch of four gets four, whatever is left of the phase.
        context = makeContext(makeNonlinearSpace(1.0), trials=makeTrials(completed=8))

        assert randomDesign(8).propose(context, count=4).shape[0] == 4

    def test_a_count_of_zero_gives_nothing(self) -> None:
        assert randomDesign(8).propose(makeContext(), count=0).shape[0] == 0


class TestEveryPointIsFeasibleAndLegal:
    """Acceptance test of the task, on a space with linear and nonlinear constraints."""

    def test_on_a_space_with_a_linear_and_a_nonlinear_constraint(self) -> None:
        context = makeContext(makeSpace())

        points = randomDesign(24).propose(context)

        assert points.shape[0] == 24
        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_for_every_seed(self, seed: int) -> None:
        context = makeContext(makeSpace(), seed=seed)

        points = randomDesign(16).propose(context)

        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)

    def test_the_decoded_values_are_what_the_constraints_say(self) -> None:
        context = makeContext(makeSpace())

        for row in randomDesign(24).propose(context):
            values = context.encoder.decode(row)
            assert values["x"] + values["y"] <= 12.0 + 1e-6
            assert values["x"] * values["y"] >= 4.0 - 1e-6

    def test_the_points_are_snapped_onto_valid_values(self) -> None:
        context = makeContext(makeSpace())

        points = randomDesign(16).propose(context)

        assert torch.allclose(context.encoder.snapColumns(points), points)

    def test_integers_categories_and_reals_all_vary(self) -> None:
        context = makeContext(makeSpace())

        decoded = [context.encoder.decode(row) for row in randomDesign(32).propose(context)]

        assert all(
            isinstance(values["n"], int) and 1 <= values["n"] <= 9 for values in decoded
        )
        assert {values["kind"] for values in decoded} == {"a", "b", "c"}
        assert len({round(values["x"], 6) for values in decoded}) > 16

    def test_a_mixture_that_sums_to_one_is_met(self) -> None:
        context = makeContext(makeMixtureSpace())

        points = randomDesign(8).propose(context)

        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)
        assert torch.allclose(points.sum(dim=1), torch.ones(8, dtype=points.dtype))

    def test_a_small_feasible_region_is_found_by_drawing_more_candidates(self) -> None:
        # About one candidate in eleven is feasible, and the first round draws as few as one
        # per point: several rounds are needed, and the sampler asks the draw for more.
        sampling = FeasibleSamplingConfig(oversampling_factor=1, min_draw=1)
        context = makeContext(makeNonlinearSpace(60.0), sampling=sampling)

        points = randomDesign(6).propose(context)

        assert points.shape[0] == 6
        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)

    def test_the_candidates_of_later_rounds_continue_the_stream(self) -> None:
        # If each round restarted the stream it would offer the same candidates again, and
        # the points returned would repeat: no two may be equal.
        sampling = FeasibleSamplingConfig(oversampling_factor=1, min_draw=1)
        context = makeContext(makeNonlinearSpace(60.0), sampling=sampling)

        points = randomDesign(6).propose(context)

        assert torch.unique(points, dim=0).shape[0] == 6

    def test_a_single_precision_runtime_gives_single_precision_points(self) -> None:
        context = makeContext(makeSpace(), runtime=RuntimeConfig(dtype="float32"))

        points = randomDesign(8).propose(context)

        assert points.dtype == torch.float32
        assert bool(context.constraints.isFeasible(points).all())
        assertLegal(context, points)

    def test_an_empty_feasible_region_is_an_error_and_not_a_point_that_breaks_a_constraint(
        self,
    ) -> None:
        context = makeContext(makeNonlinearSpace(1000.0))

        with pytest.raises(RuntimeError, match="no point satisfying"):
            randomDesign(4).propose(context)

    def test_the_settings_of_the_sampler_are_those_of_the_constraints(self) -> None:
        # One round of one candidate cannot find a region that holds one point in eleven, and
        # the default settings can: the design uses what the constraints were built with.
        space = makeNonlinearSpace(60.0)
        stingy = FeasibleSamplingConfig(max_rounds=1, oversampling_factor=1, min_draw=1)

        with pytest.raises(RuntimeError, match="no point satisfying"):
            randomDesign(1).propose(makeContext(space, sampling=stingy))
        assert randomDesign(1).propose(makeContext(space)).shape[0] == 1


class TestTheSameSeedGivesTheSamePoints:
    """Acceptance test of the task."""

    def test_on_a_space_with_linear_and_nonlinear_constraints(self) -> None:
        first = randomDesign(12).propose(makeContext(makeSpace(), seed=11))
        second = randomDesign(12).propose(makeContext(makeSpace(), seed=11))

        assert torch.equal(first, second)

    def test_without_constraints(self) -> None:
        first = randomDesign(8).propose(makeContext(unitSquare(), seed=3))
        second = randomDesign(8).propose(makeContext(unitSquare(), seed=3))

        assert torch.equal(first, second)

    def test_with_a_nonlinear_constraint_that_takes_several_rounds(self) -> None:
        sampling = FeasibleSamplingConfig(oversampling_factor=1, min_draw=1)

        first = randomDesign(6).propose(
            makeContext(makeNonlinearSpace(60.0), seed=4, sampling=sampling)
        )
        second = randomDesign(6).propose(
            makeContext(makeNonlinearSpace(60.0), seed=4, sampling=sampling)
        )

        assert torch.equal(first, second)

    def test_with_an_equality_constraint(self) -> None:
        first = randomDesign(4).propose(makeContext(makeMixtureSpace(), seed=9))
        second = randomDesign(4).propose(makeContext(makeMixtureSpace(), seed=9))

        assert torch.equal(first, second)

    def test_two_designs_made_from_the_same_configuration_agree(self) -> None:
        context_a = makeContext(makeSpace(), seed=5)
        context_b = makeContext(makeSpace(), seed=5)

        assert torch.equal(
            randomDesign(8).propose(context_a), randomDesign(8).propose(context_b)
        )

    @pytest.mark.parametrize(
        "space",
        [unitSquare, makeSpace, makeMixtureSpace],
        ids=["no-constraint", "linear-and-nonlinear", "equality"],
    )
    def test_another_seed_gives_other_points(self, space: Any) -> None:
        first = randomDesign(8).propose(makeContext(space(), seed=1))
        second = randomDesign(8).propose(makeContext(space(), seed=2))

        assert not torch.equal(first, second)

    def test_a_generator_that_has_been_drawn_from_gives_other_points_than_a_fresh_one(
        self,
    ) -> None:
        # The design draws from the generator, so the same object used twice is not the same
        # stream twice, and a fresh generator of the same seed starts it again.
        context = makeContext(unitSquare(), seed=6)
        design = randomDesign(8)

        first = design.propose(context)
        second = design.propose(context)
        again = design.propose(makeContext(unitSquare(), seed=6))

        assert not torch.equal(first, second)
        assert torch.equal(first, again)

    def test_the_size_of_the_design_does_not_change_the_points_it_starts_with(self) -> None:
        # The stream is one sequence of independent numbers, so asking for more points of the
        # same seed extends the design and does not redraw it.
        small = randomDesign(4).propose(makeContext(unitSquare(), seed=8))
        large = randomDesign(8).propose(makeContext(unitSquare(), seed=8))

        assert torch.equal(large[:4], small)


class TestTheTwoDesignsAreNotAliases:
    """Acceptance test of the task: on the same seed `random_v1` and `sobol_v1` differ."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 42, 1234])
    def test_on_the_same_seed_they_give_other_points(self, seed: int) -> None:
        random_points = randomDesign(8).propose(makeContext(unitSquare(), seed=seed))
        sobol_points = sobol(8).propose(makeContext(unitSquare(), seed=seed))

        assert random_points.shape == sobol_points.shape
        assert not torch.equal(random_points, sobol_points)

    @pytest.mark.parametrize("n_points", [1, 2, 4, 16, 64])
    def test_whatever_the_size(self, n_points: int) -> None:
        random_points = randomDesign(n_points).propose(makeContext(unitSquare(), seed=5))
        sobol_points = sobol(n_points).propose(makeContext(unitSquare(), seed=5))

        assert not torch.equal(random_points, sobol_points)

    def test_with_a_nonlinear_constraint(self) -> None:
        space = makeNonlinearSpace(4.0)

        random_points = randomDesign(8).propose(makeContext(space, seed=5))
        sobol_points = sobol(8).propose(makeContext(space, seed=5))

        assert not torch.equal(random_points, sobol_points)

    def test_in_a_space_with_categories_and_integers(self) -> None:
        space = SearchSpace(
            [Real("x", 0.0, 1.0), Categorical("kind", ["a", "b", "c"]), Real("z", 0.0, 1.0)]
        )

        random_points = randomDesign(12).propose(makeContext(space, seed=5))
        sobol_points = sobol(12).propose(makeContext(space, seed=5))

        assert not torch.equal(random_points, sobol_points)

    def test_they_are_not_the_unscrambled_sequence_either(self) -> None:
        points = randomDesign(8).propose(makeContext(unitSquare(), seed=1))

        classical = SobolEngine(dimension=2, scramble=False).draw(8).to(torch.float64)
        assert not torch.equal(points, classical)

    def test_with_linear_constraints_the_two_designs_share_the_chain(self) -> None:
        # Documented limit, the counterpart of the one of `scramble` in `sobol_v1`: a
        # hit-and-run chain draws the points inside a polytope and needs no raw draws, so
        # what tells the two designs apart is not used, and both seed the chain from the
        # generator in the same way.
        space = makeMixtureSpace()

        random_points = randomDesign(4).propose(makeContext(space, seed=5))
        sobol_points = sobol(4).propose(makeContext(space, seed=5))

        assert torch.equal(random_points, sobol_points)


class TestTheyAreUniformRandomPoints:
    def test_the_raw_points_are_in_the_unit_interval(self) -> None:
        points = randomDesign(500).propose(makeContext(unitSquare(), seed=2))

        assert bool((points >= 0.0).all())
        assert bool((points < 1.0).all())

    def test_every_column_is_spread_evenly_over_its_range(self) -> None:
        # 4000 points in ten bins: 400 expected in each, a standard deviation of about 19, so
        # a count within 100 of it is more than five of them away from failing.
        points = randomDesign(4000).propose(makeContext(unitSquare(), seed=3))

        for column in range(points.shape[1]):
            counts = torch.histc(points[:, column], bins=10, min=0.0, max=1.0)
            assert bool(((counts - 400.0).abs() < 100.0).all()), counts.tolist()

    def test_the_two_columns_are_not_tied_to_each_other(self) -> None:
        points = randomDesign(4000).propose(makeContext(unitSquare(), seed=4))

        correlation = torch.corrcoef(points.T)[0, 1]

        assert abs(float(correlation)) < 0.1

    def test_the_levels_of_a_category_are_equally_likely(self) -> None:
        space = SearchSpace([Real("x", 0.0, 1.0), Categorical("kind", ["a", "b", "c", "d"])])
        context = makeContext(space, seed=5)

        decoded = [context.encoder.decode(row) for row in randomDesign(2000).propose(context)]

        counts = {
            kind: sum(1 for values in decoded if values["kind"] == kind) for kind in "abcd"
        }
        assert all(abs(count - 500) < 100 for count in counts.values()), counts

    def test_a_small_design_is_not_balanced_the_way_a_sobol_sequence_is(self) -> None:
        # The property of a Sobol sequence that random points lack: its first 2**m points put
        # exactly the same number in each of the equal boxes. Eight random points do so in
        # about one case in twenty-six, so across six seeds some are not balanced.
        def balanced(seed: int) -> bool:
            points = randomDesign(8).propose(makeContext(unitSquare(), seed=seed))
            left = points[:, 0] < 0.5
            below = points[:, 1] < 0.5
            counts = [
                int(((left == in_left) & (below == in_below)).sum())
                for in_left in (True, False)
                for in_below in (True, False)
            ]
            return counts == [2, 2, 2, 2]

        assert not all(balanced(seed) for seed in range(6))


class TestProposingDoesNotChangeWhatItIsGiven:
    def test_the_history_is_left_as_it_was(self) -> None:
        trials = makeTrials(completed=2, pending=1)
        context = makeContext(makeSpace(), trials=trials)

        randomDesign(8).propose(context)

        assert list(context.trials) == trials

    def test_a_design_keeps_no_state_from_one_call_to_the_next(self) -> None:
        # The same design object, two fresh contexts of the same seed: the same points.
        design = randomDesign(8)

        first = design.propose(makeContext(makeSpace(), seed=1))
        design.propose(makeContext(makeSpace(), seed=2))
        again = design.propose(makeContext(makeSpace(), seed=1))

        assert torch.equal(first, again)
