"""Unit tests for `random_policy_v1` (roadmap task P6-06, specification 5.9, 3.10, FR27).

The policy is tested with the real designs (`random_v1` and `sobol_v1`) on spaces with linear
and nonlinear constraints, an equality, and finite spaces where two draws can land on the same
point; and with a design registered by the test, which stands for a plugin's and records what
the policy asks of it. Every test that registers a design uses `isolatedPolicyKind`, because a
registered name is permanent and must never reach the real registry.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import pytest
import torch
from policy_fakes import makeGoal, makeProblem, makeTrial
from pydantic import Field, ValidationError
from torch import Tensor

import planexs.policies  # noqa: F401
from planexs import (
    Categorical,
    Integer,
    LinearConstraint,
    NonlinearConstraint,
    Real,
    SearchSpace,
)
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RngTree import RngTree
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.DesignContext import DesignContext
from planexs.designs.designRegistry import designRegistry
from planexs.designs.registerDesign import registerDesign
from planexs.domain.Observation import Observation
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.EncoderConfig import EncoderConfig
from planexs.encoding.FeasibleSamplingConfig import FeasibleSamplingConfig
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.policies.AbstractPolicy import AbstractPolicy
from planexs.policies.policyRegistry import policyRegistry
from planexs.policies.RandomPolicy import RandomPolicy
from planexs.policies.RandomPolicyConfig import RandomPolicyConfig
from planexs.policies.StudyStateView import StudyStateView
from planexs.policies.Suggestion import Suggestion

NAME = "random_policy_v1"

#: The registered designs of the real tree that the policy is tested with.
DESIGNS = ["random_v1", "sobol_v1"]


# --------------------------------------------------------------------------------------------
# Problems and builders
# --------------------------------------------------------------------------------------------


def problemOver(space: SearchSpace) -> Problem:
    """A problem over `space` with one continuous outcome."""
    return Problem(search_space=space, outcomes=[Outcome(name="yield", type="continuous")])


def constrainedProblem() -> Problem:
    """A mixed space with a linear and a nonlinear constraint: `x + y <= 12`, `x * y >= 4`.

    Two reals, an integer and an unordered categorical, so that snapping, one-hot columns and
    both kinds of constraint are exercised.
    """
    return problemOver(
        SearchSpace(
            [
                Real("x", 0.0, 10.0),
                Real("y", 0.0, 10.0),
                Integer("n", 1, 9),
                Categorical("kind", ["a", "b", "c"]),
            ],
            [
                LinearConstraint({"x": 1.0, "y": 1.0}, 12.0, "<="),
                NonlinearConstraint("x * y", ">=", 4.0),
            ],
        )
    )


def mixtureProblem() -> Problem:
    """Three reals in `[0, 1]` that sum to one: an equality."""
    return problemOver(
        SearchSpace(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
            [LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")],
        )
    )


def finiteProblem(levels: int = 3) -> Problem:
    """One categorical parameter with `levels` levels: the whole space is `levels` points."""
    names = [chr(ord("a") + index) for index in range(levels)]
    return problemOver(SearchSpace([Categorical("kind", names)]))


def build(
    problem: Problem | None = None, runtime: RuntimeConfig | None = None, **fields: Any
) -> AbstractPolicy[Any]:
    """Builds the registered policy the way a study does: resolve, then call the builder."""
    resolved = resolveComponent(policyRegistry(), {"type": NAME, **fields})
    policy: AbstractPolicy[Any] = resolved.entry.builder(
        resolved.config,
        problem=problemOver(makeProblem().search_space) if problem is None else problem,
        goal=makeGoal(),
        runtime=RuntimeConfig() if runtime is None else runtime,
    )
    return policy


def viewOf(
    problem: Problem,
    trials: list[Trial] | None = None,
    *,
    ask_index: int = 0,
    seed: int = 7,
    history_version: int = 0,
) -> StudyStateView:
    """A study-state view over `problem`."""
    return StudyStateView(
        problem=problem,
        goal=makeGoal(),
        trials=[] if trials is None else trials,
        history_version=history_version,
        ask_index=ask_index,
        random_sources=RngTree(seed),
    )


def reseed(seed: int) -> None:
    """Seeds every global random state that a component could wrongly draw from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pendingWith(*parameterizations: dict[str, Any]) -> list[Trial]:
    """Pending trials with the given parameterizations."""
    return [Trial(id=index, parameters=p) for index, p in enumerate(parameterizations)]


def levelsOf(suggestions: list[Suggestion]) -> list[str]:
    """The level of the categorical parameter `kind` of each suggestion."""
    return [str(suggestion.parameters["kind"]) for suggestion in suggestions]


# --------------------------------------------------------------------------------------------
# A design registered by the test: it records what the policy asks of it
# --------------------------------------------------------------------------------------------

#: What the recording design was asked, one entry per call: the number of points, the initial
#: seed of the generator, and the context.
CALLS: list[tuple[int, int, DesignContext]] = []


class PlugDesignConfig(ComponentConfig):
    """The configuration of the design that the tests register."""

    type: Literal["plug_design_v1"] = Field(default="plug_design_v1", description="The name.")
    n_points: int = Field(default=2, ge=1, description="The size of the design, in points.")
    first_at_centre: bool = Field(
        default=False, description="Whether the first point of the first call is the centre."
    )


class PlugDesign(AbstractDesign[PlugDesignConfig]):
    """A design that stands for a plugin's: random points, and a record of every call."""

    def pointsNeeded(self, trials: Any) -> int:
        """The size of the design."""
        return self.config.n_points

    def drawPoints(self, context: DesignContext, n: int) -> Tensor:
        """`n` uniform points of the unit cube, the first one optionally at the centre."""
        calls_before = len(CALLS)
        CALLS.append((n, context.generator.initial_seed(), context))
        points = torch.rand(
            n,
            context.encoder.n_columns,
            generator=context.generator,
            dtype=context.encoder.dtype,
            device=context.generator.device,
        )
        if self.config.first_at_centre and calls_before == 0:
            points[0] = 0.5
        return points


@pytest.fixture
def plugDesign(isolatedPolicyKind: KindTable) -> Iterator[type[PlugDesign]]:
    """Registers `plug_design_v1` in the swapped table, and starts an empty record."""
    CALLS.clear()
    registerDesign(
        "plug_design_v1",
        config=PlugDesignConfig,
        description="A design that records what it is asked, for tests.",
    )(PlugDesign)
    # The swapped table holds the policy kind empty: the policy under test is registered in it
    # again, as the real tree has it, so that the registry can build it.
    registerPolicyUnderTest()
    yield PlugDesign
    CALLS.clear()


def registerPolicyUnderTest() -> None:
    """Registers `RandomPolicy` in the swapped table, as `planexs.policies` does at import."""
    from planexs.policies.registerPolicy import registerPolicy

    registerPolicy(
        NAME,
        config=RandomPolicyConfig,
        description="Random search: draws points from a design, ignoring the observations.",
    )(RandomPolicy)


# --------------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------------


class TestRegistration:
    def test_the_policy_is_registered_in_the_real_registry(self) -> None:
        entry = policyRegistry().lookup(NAME)

        assert entry.name == NAME
        assert NAME in policyRegistry().names()
        assert entry.builder == RandomPolicy.fromConfig
        assert entry.config_schema is RandomPolicyConfig
        assert entry.config_schema.model_fields["type"].default == NAME

    def test_it_is_a_policy_with_a_one_line_description(self) -> None:
        entry = policyRegistry().lookup(NAME)

        assert issubclass(RandomPolicy, AbstractPolicy)
        assert entry.description.splitlines() == [entry.description]
        assert "random" in entry.description.lower()

    def test_the_entry_passes_the_contract_of_specification_6(self) -> None:
        entry = policyRegistry().lookup(NAME)

        assert policyRegistry().isValidName(entry.name)
        assert all(
            field.description and field.description.strip()
            for field in entry.config_schema.model_fields.values()
        )
        assert callable(entry.builder)

    def test_the_entry_declares_no_tag(self) -> None:
        # The tags of the problem features are registered with the optimizers (P5-11): until
        # then there is nothing to declare, and a declaration of what the policy needs is
        # empty because it needs nothing from any other participant.
        entry = policyRegistry().lookup(NAME)

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_a_fresh_interpreter_that_imports_the_package_has_the_policy_registered(
        self,
    ) -> None:
        import subprocess
        import sys

        code = (
            "import planexs.policies\n"
            "from planexs.policies.policyRegistry import policyRegistry\n"
            "from planexs.core.resolveComponent import resolveComponent\n"
            "resolved = resolveComponent(policyRegistry(), {'type': 'random_policy_v1'})\n"
            "print(resolved.config.design.type)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == "random_v1"


# --------------------------------------------------------------------------------------------
# Configuration: the design is selected by name
# --------------------------------------------------------------------------------------------


class TestConfiguration:
    def test_the_defaults_are_written_out_in_full(self) -> None:
        resolved = resolveComponent(policyRegistry(), {"type": NAME})

        assert toResolvedConfig(resolved.config) == {
            "type": NAME,
            "design": {
                "type": "random_v1",
                "n_points": 1,
                "count_existing_observations": True,
            },
            "max_redraws": 10,
            "encoder": EncoderConfig().model_dump(),
            "sampling": FeasibleSamplingConfig().model_dump(),
        }

    def test_the_default_design_is_random_search(self) -> None:
        assert RandomPolicyConfig().design.type == "random_v1"

    def test_the_design_is_selected_by_name_and_has_the_schema_of_that_design(self) -> None:
        resolved = resolveComponent(
            policyRegistry(),
            {"type": NAME, "design": {"type": "sobol_v1", "n_points": 8, "scramble": False}},
        )

        design = resolved.config.design
        assert design.type == "sobol_v1"
        assert design.model_dump() == {
            "type": "sobol_v1",
            "n_points": 8,
            "scramble": False,
            "count_existing_observations": True,
        }
        assert type(design) is designRegistry().lookup("sobol_v1").config_schema

    def test_an_unknown_design_names_the_field_and_lists_the_registered_ones(self) -> None:
        with pytest.raises(ConfigError, match="design") as caught:
            resolveComponent(policyRegistry(), {"type": NAME, "design": {"type": "halton_v1"}})

        assert "random_v1" in caught.value.message
        assert "sobol_v1" in caught.value.message
        assert caught.value.field_path == "design.type"

    def test_a_setting_of_the_design_that_is_wrong_is_reported_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(
                policyRegistry(),
                {"type": NAME, "design": {"type": "random_v1", "n_points": 0}},
            )

        assert caught.value.field_path == "design.n_points"

    def test_an_unknown_field_is_refused_with_its_name(self) -> None:
        with pytest.raises(ConfigError, match="max_redraw") as caught:
            resolveComponent(policyRegistry(), {"type": NAME, "max_redraw": 3})

        assert "policy 'random_policy_v1'" in caught.value.message

    @pytest.mark.parametrize("value", [-1, 2.5, "ten", None])
    def test_a_number_of_redraws_that_is_not_a_whole_number_from_zero_is_refused(
        self, value: Any
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(policyRegistry(), {"type": NAME, "max_redraws": value})

        assert caught.value.field_path == "max_redraws"

    def test_zero_redraws_is_a_legal_setting(self) -> None:
        assert RandomPolicyConfig(max_redraws=0).max_redraws == 0

    def test_the_settings_of_the_encoder_and_the_sampler_are_those_of_a_study(self) -> None:
        resolved = resolveComponent(
            policyRegistry(),
            {
                "type": NAME,
                "encoder": {"unset_choice_ordering": "never"},
                "sampling": {"max_rounds": 3},
            },
        )

        assert resolved.config.encoder.unset_choice_ordering == "never"
        assert resolved.config.sampling.max_rounds == 3
        assert resolved.config.sampling.min_draw == FeasibleSamplingConfig().min_draw

    def test_a_setting_of_the_sampler_that_is_wrong_is_reported_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(policyRegistry(), {"type": NAME, "sampling": {"max_rounds": 0}})

        assert caught.value.field_path == "sampling.max_rounds"

    def test_the_resolved_configuration_resolves_to_the_same_configuration_again(self) -> None:
        first = resolveComponent(
            policyRegistry(),
            {"type": NAME, "design": {"type": "sobol_v1", "n_points": 4}, "max_redraws": 3},
        )

        stored = json.loads(json.dumps(toResolvedConfig(first.config)))
        again = resolveComponent(policyRegistry(), stored)

        assert again.config == first.config

    def test_the_policy_type_is_pinned(self) -> None:
        with pytest.raises(ValidationError):
            RandomPolicyConfig(type="random_v1")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Building: the design comes from the registry
# --------------------------------------------------------------------------------------------


class TestBuilding:
    def test_the_builder_keeps_the_configuration_and_labels_the_policy(self) -> None:
        resolved = resolveComponent(policyRegistry(), {"type": NAME})

        policy = resolved.entry.builder(
            resolved.config, problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
        )

        assert isinstance(policy, RandomPolicy)
        assert policy.config is resolved.config
        assert policy.label == "policy 'random_policy_v1'"

    def test_building_draws_nothing_and_asks_the_design_for_nothing(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        build(design={"type": "plug_design_v1"})

        assert CALLS == []

    def test_the_design_is_obtained_through_the_registry(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        # A design that is not in the code of the policy is used: the seam is real (3.10).
        problem = problemOver(makeProblem().search_space)
        policy = build(problem, design={"type": "plug_design_v1", "n_points": 5})

        suggestions = policy.suggest(viewOf(problem), 3)

        assert len(suggestions) == 3
        assert len(CALLS) == 1
        assert {suggestion.metadata["design"] for suggestion in suggestions} == {
            "plug_design_v1"
        }

    def test_the_goal_is_not_looked_at(self) -> None:
        resolved = resolveComponent(policyRegistry(), {"type": NAME})

        policy = resolved.entry.builder(
            resolved.config, problem=makeProblem(), goal=None, runtime=RuntimeConfig()
        )

        assert isinstance(policy, RandomPolicy)


class TestWhatTheDesignIsGiven:
    def test_it_is_asked_for_the_number_of_points_that_the_policy_was_asked_for(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        # The size of the design is not the policy's business: a batch of 7 is 7 points.
        problem = problemOver(makeProblem().search_space)
        policy = build(problem, design={"type": "plug_design_v1", "n_points": 2})

        suggestions = policy.suggest(viewOf(problem), 7)

        assert len(suggestions) == 7
        assert [n for n, _, _ in CALLS] == [7]

    def test_the_context_holds_the_problem_the_history_and_the_encoder_of_the_space(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        problem = constrainedProblem()
        trials = [makeTrial(0, TrialState.pending), makeTrial(1, TrialState.failed)]
        policy = build(problem, design={"type": "plug_design_v1"})
        view = viewOf(problem, trials)

        policy.suggest(view, 1)

        (_, _, context) = CALLS[0]
        assert context.problem is problem
        assert context.trials == tuple(trials)
        assert isinstance(context.encoder, SearchSpaceEncoder)
        assert isinstance(context.constraints, EncodedConstraints)

    def test_the_generator_is_the_stream_of_the_design_for_the_ask(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem, design={"type": "plug_design_v1"})
        view = viewOf(problem, ask_index=4, seed=11)

        policy.suggest(view, 1)

        assert CALLS[0][1] == view.seedFor("policy.design")

    def test_the_runtime_settings_reach_the_encoder_and_the_generator(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(
            problem, RuntimeConfig(dtype="float32"), design={"type": "plug_design_v1"}
        )

        policy.suggest(viewOf(problem), 1)

        assert CALLS[0][2].encoder.dtype is torch.float32

    def test_the_settings_of_the_encoder_and_the_sampler_reach_the_design(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        problem = constrainedProblem()
        policy = build(
            problem,
            design={"type": "plug_design_v1"},
            encoder={"max_ordinal_columns": 3},
            sampling={"max_rounds": 4},
        )

        policy.suggest(viewOf(problem), 1)

        context = CALLS[0][2]
        assert context.encoder.config.max_ordinal_columns == 3
        assert context.constraints.config.max_rounds == 4


# --------------------------------------------------------------------------------------------
# Suggestions: legal and feasible
# --------------------------------------------------------------------------------------------


@pytest.fixture(params=DESIGNS)
def design(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The configuration of each design of the real tree, with a size of its own."""
    return {"type": request.param, "n_points": 4}


@pytest.fixture(params=["constrained", "mixture", "plain"])
def problem(request: pytest.FixtureRequest) -> Problem:
    """A problem with linear and nonlinear constraints, with an equality, and with none."""
    return {
        "constrained": constrainedProblem,
        "mixture": mixtureProblem,
        "plain": lambda: problemOver(makeProblem().search_space),
    }[request.param]()


class TestSuggestionsAreLegalAndFeasible:
    @pytest.mark.parametrize("n", [1, 2, 9, 17])
    def test_it_returns_exactly_the_number_asked_for(
        self, problem: Problem, design: dict[str, Any], n: int
    ) -> None:
        suggestions = build(problem, design=design).suggest(viewOf(problem), n)

        assert len(suggestions) == n
        assert all(isinstance(suggestion, Suggestion) for suggestion in suggestions)

    def test_every_suggestion_is_a_legal_parameterization_of_the_problem(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        # Values in range, inactive parameters absent, every constraint satisfied on the
        # natural values: the check the study makes before it records a trial.
        suggestions = build(problem, design=design).suggest(viewOf(problem), 25)

        for suggestion in suggestions:
            problem.validateParameterization(suggestion.parameters)

    def test_every_suggestion_is_feasible_in_the_encoded_space(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        space = problem.search_space
        encoder = SearchSpaceEncoder(space, RuntimeConfig())
        constraints = EncodedConstraints(space, encoder)
        suggestions = build(problem, design=design).suggest(viewOf(problem), 25)

        encoded = torch.stack([encoder.encodeParameters(s.parameters) for s in suggestions])

        assert bool(constraints.isFeasible(encoded).all())

    def test_suggestions_with_other_asks_and_seeds_are_legal_too(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        policy = build(problem, design=design)

        for ask_index in range(4):
            for suggestion in policy.suggest(viewOf(problem, ask_index=ask_index, seed=3), 6):
                problem.validateParameterization(suggestion.parameters)

    def test_each_suggestion_says_that_it_comes_from_a_design_and_which(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        suggestions = build(problem, design=design).suggest(viewOf(problem), 3)

        assert [s.phase for s in suggestions] == ["design"] * 3
        assert [s.metadata["design"] for s in suggestions] == [design["type"]] * 3

    def test_a_suggestion_is_a_value_that_a_trial_can_be_made_from(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        (suggestion,) = build(problem, design=design).suggest(viewOf(problem), 1)

        trial = Trial(id=0, parameters=suggestion.parameters, metadata=suggestion.metadata)

        assert trial.state is TrialState.pending

    def test_a_design_of_the_policy_that_is_not_the_default_is_the_one_used(self) -> None:
        # The two designs draw different points where nothing constrains them (with linear
        # constraints they coincide: both feed the same chain with the same seed).
        problem = problemOver(makeProblem().search_space)
        view = viewOf(problem)

        uniform = build(problem, design={"type": "random_v1", "n_points": 4}).suggest(view, 4)
        sobol = build(problem, design={"type": "sobol_v1", "n_points": 4}).suggest(view, 4)

        assert [s.parameters for s in uniform] != [s.parameters for s in sobol]


# --------------------------------------------------------------------------------------------
# Reproducibility, and ignoring the observations
# --------------------------------------------------------------------------------------------


class TestDeterminism:
    def test_the_same_seed_ask_index_and_history_give_the_same_suggestions(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        first = build(problem, design=design).suggest(viewOf(problem, ask_index=2), 5)
        second = build(problem, design=design).suggest(viewOf(problem, ask_index=2), 5)

        assert first == second

    def test_asking_twice_for_the_same_view_gives_the_same_suggestions(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        # The policy has no memory: nothing from the first call reaches the second.
        policy = build(problem, design=design)
        view = viewOf(problem, ask_index=1)

        assert policy.suggest(view, 4) == policy.suggest(view, 4)

    def test_another_ask_index_or_another_seed_gives_other_suggestions(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        policy = build(problem, design=design)
        reference = policy.suggest(viewOf(problem, ask_index=0, seed=5), 4)

        later = policy.suggest(viewOf(problem, ask_index=1, seed=5), 4)
        other = policy.suggest(viewOf(problem, ask_index=0, seed=6), 4)

        assert later != reference
        assert other != reference

    def test_the_global_random_state_is_not_read(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        policy = build(problem, design=design)
        view = viewOf(problem)

        reseed(1)
        first = policy.suggest(view, 4)
        reseed(2)
        second = policy.suggest(view, 4)

        assert first == second

    def test_asking_leaves_the_global_random_state_where_it_was(
        self, problem: Problem, design: dict[str, Any]
    ) -> None:
        policy = build(problem, design=design)
        view = viewOf(problem)
        torch.manual_seed(1)
        before = torch.get_rng_state()

        policy.suggest(view, 4)

        assert torch.equal(torch.get_rng_state(), before)


class TestTheObservationsAreIgnored:
    def historyOf(self, kinds: list[TrialState]) -> list[Trial]:
        return [makeTrial(index, state) for index, state in enumerate(kinds)]

    def test_completed_trials_do_not_change_the_suggestions(
        self, design: dict[str, Any]
    ) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem, design=design)
        empty = policy.suggest(viewOf(problem), 4)

        with_data = policy.suggest(
            viewOf(problem, self.historyOf([TrialState.completed] * 6), history_version=6), 4
        )

        assert with_data == empty

    def test_trials_in_every_state_but_pending_do_not_change_the_suggestions(self) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem)
        history = self.historyOf(
            [
                TrialState.completed,
                TrialState.failed,
                TrialState.abandoned,
                TrialState.completed,
            ]
        )

        assert policy.suggest(viewOf(problem, history), 4) == policy.suggest(
            viewOf(problem), 4
        )

    def test_the_outcomes_that_were_observed_are_not_read(self) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem)

        def history(value: float) -> list[Trial]:
            observation = Observation(values={"yield": value})
            parameters = {"x0": 0.25, "x1": 0.75}
            return [
                Trial(
                    id=0,
                    parameters=parameters,
                    state=TrialState.completed,
                    observation=observation,
                )
            ]

        poor = policy.suggest(viewOf(problem, history(-1e9)), 3)
        rich = policy.suggest(viewOf(problem, history(1e9)), 3)

        assert poor == rich

    def test_a_completed_point_may_be_drawn_again(self) -> None:
        # Random search does not learn: only the pending points are avoided (see below).
        problem = problemOver(makeProblem().search_space)
        policy = build(problem)
        (first,) = policy.suggest(viewOf(problem), 1)

        for state in (TrialState.completed, TrialState.failed, TrialState.abandoned):
            trial = _trialIn(state, first.parameters)
            (again,) = policy.suggest(viewOf(problem, [trial]), 1)

            assert again == first


def _trialIn(state: TrialState, parameters: Any) -> Trial:
    """A trial in `state` with the given parameterization."""
    if state is TrialState.completed:
        reference = makeTrial(0, TrialState.completed)
        return Trial(
            id=0, parameters=parameters, state=state, observation=reference.observation
        )
    if state is TrialState.failed:
        return Trial(id=0, parameters=parameters, state=state, reason="it broke")
    return Trial(id=0, parameters=parameters, state=state)


# --------------------------------------------------------------------------------------------
# Pending points
# --------------------------------------------------------------------------------------------


class TestPendingPoints:
    def test_a_point_that_the_design_would_repeat_is_replaced_by_a_new_one(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        # The first point of the first draw is the centre, and the centre is being evaluated.
        problem = problemOver(makeProblem().search_space)
        design = {"type": "plug_design_v1", "first_at_centre": True}
        policy = build(problem, design=design)
        centre = {"x0": 0.5, "x1": 0.5}

        suggestions = policy.suggest(viewOf(problem, pendingWith(centre)), 3)

        assert len(suggestions) == 3
        assert all(dict(s.parameters) != centre for s in suggestions)

    def test_without_the_pending_trial_the_same_draw_would_have_returned_the_centre(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        # The witness of the test above: what is avoided is the pending trial.
        problem = problemOver(makeProblem().search_space)
        policy = build(problem, design={"type": "plug_design_v1", "first_at_centre": True})

        suggestions = policy.suggest(viewOf(problem), 3)

        assert dict(suggestions[0].parameters) == {"x0": 0.5, "x1": 0.5}

    def test_the_replacement_is_a_draw_for_the_points_still_missing_from_a_new_stream(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem, design={"type": "plug_design_v1", "first_at_centre": True})
        view = viewOf(problem, pendingWith({"x0": 0.5, "x1": 0.5}), ask_index=2, seed=9)

        policy.suggest(view, 3)

        assert [n for n, _, _ in CALLS] == [3, 1]
        assert [seed for _, seed, _ in CALLS] == [
            view.seedFor("policy.design"),
            view.seedFor("policy.design.redraw.1"),
        ]
        assert CALLS[0][1] != CALLS[1][1]

    def test_the_points_that_are_kept_stay_in_the_order_they_were_drawn(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem, design={"type": "plug_design_v1", "first_at_centre": True})
        free = policy.suggest(viewOf(problem), 3)
        CALLS.clear()  # the design puts the centre first in the first call it records

        avoided = policy.suggest(viewOf(problem, pendingWith({"x0": 0.5, "x1": 0.5})), 3)

        assert [s.parameters for s in avoided[:2]] == [s.parameters for s in free[1:]]

    def test_a_batch_has_no_point_twice(self) -> None:
        problem = finiteProblem(4)
        policy = build(problem, max_redraws=200)

        for seed in range(12):
            suggestions = policy.suggest(viewOf(problem, seed=seed), 4)

            assert sorted(levelsOf(suggestions)) == ["a", "b", "c", "d"]

    def test_a_finite_space_never_gets_a_point_that_is_being_evaluated(self) -> None:
        problem = finiteProblem(3)
        policy = build(problem, max_redraws=200)
        pending = pendingWith({"kind": "a"}, {"kind": "b"})

        for ask_index in range(25):
            (suggestion,) = policy.suggest(viewOf(problem, pending, ask_index=ask_index), 1)

            assert suggestion.parameters["kind"] == "c"

    def test_a_point_that_is_pending_is_avoided_by_every_design(
        self, design: dict[str, Any]
    ) -> None:
        problem = finiteProblem(3)
        policy = build(problem, design=design, max_redraws=200)
        pending = pendingWith({"kind": "b"})

        for ask_index in range(10):
            suggestions = policy.suggest(viewOf(problem, pending, ask_index=ask_index), 2)

            assert sorted(levelsOf(suggestions)) == ["a", "c"]

    def test_parameterizations_that_are_not_pending_are_not_avoided(self) -> None:
        # Random search ignores what it has learnt: only the points that are being evaluated.
        problem = finiteProblem(3)
        policy = build(problem, max_redraws=0)
        settled = [
            _trialIn(TrialState.completed, {"kind": "a"}),
            Trial(id=1, parameters={"kind": "b"}, state=TrialState.failed, reason="it broke"),
            Trial(id=2, parameters={"kind": "c"}, state=TrialState.abandoned),
        ]
        free = policy.suggest(viewOf(problem), 1)

        assert policy.suggest(viewOf(problem, settled), 1) == free

    def test_a_pending_trial_in_a_continuous_space_costs_nothing(self) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem, max_redraws=0)

        suggestions = policy.suggest(viewOf(problem, pendingWith({"x0": 0.1, "x1": 0.2})), 5)

        assert len(suggestions) == 5

    def test_pending_trials_of_a_mixed_space_are_compared_by_value(
        self,
    ) -> None:
        # What is compared is the parameterization that the caller evaluates, whatever its
        # types: an int, a float and a string all compare by value.
        problem = constrainedProblem()
        policy = build(problem, max_redraws=0)
        (first,) = policy.suggest(viewOf(problem), 1)
        pending = pendingWith(dict(first.parameters))

        with pytest.raises(RuntimeError, match="found only 0 of the 1"):
            policy.suggest(viewOf(problem, pending), 1)


class TestWhenTheSpaceRunsOut:
    def test_the_error_names_the_policy_what_was_found_and_what_to_do(self) -> None:
        problem = finiteProblem(3)
        policy = build(problem, max_redraws=5)
        pending = pendingWith({"kind": "a"}, {"kind": "b"}, {"kind": "c"})

        with pytest.raises(RuntimeError) as caught:
            policy.suggest(viewOf(problem, pending), 1)

        message = str(caught.value)
        assert "policy 'random_policy_v1'" in message
        assert "found only 0 of the 1" in message
        assert "5 redraws" in message
        assert "'random_v1'" in message
        assert "max_redraws" in message

    def test_the_design_is_drawn_from_once_and_then_once_for_each_redraw(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        problem = finiteProblem(3)
        policy = build(problem, design={"type": "plug_design_v1"}, max_redraws=3)
        pending = pendingWith({"kind": "a"}, {"kind": "b"}, {"kind": "c"})

        with pytest.raises(RuntimeError):
            policy.suggest(viewOf(problem, pending), 2)

        assert [n for n, _, _ in CALLS] == [2, 2, 2, 2]
        assert len({seed for _, seed, _ in CALLS}) == 4

    def test_with_no_redraw_the_first_draw_is_the_only_one(
        self, plugDesign: type[PlugDesign]
    ) -> None:
        problem = finiteProblem(3)
        policy = build(problem, design={"type": "plug_design_v1"}, max_redraws=0)
        pending = pendingWith({"kind": "a"}, {"kind": "b"}, {"kind": "c"})

        with pytest.raises(RuntimeError, match="after 0 redraws"):
            policy.suggest(viewOf(problem, pending), 1)

        assert len(CALLS) == 1

    def test_a_batch_larger_than_the_space_cannot_be_filled(self) -> None:
        problem = finiteProblem(3)
        policy = build(problem, max_redraws=50)

        with pytest.raises(RuntimeError, match="found only 3 of the 5"):
            policy.suggest(viewOf(problem), 5)

    def test_a_design_that_cannot_find_a_feasible_point_raises_its_own_error(self) -> None:
        # The constraints leave nothing: that is the design's error, and it is not retried.
        empty = problemOver(
            SearchSpace(
                [Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)],
                [NonlinearConstraint("x + y", ">=", 5.0)],
            )
        )
        policy = build(empty)

        with pytest.raises(RuntimeError):
            policy.suggest(viewOf(empty), 1)


# --------------------------------------------------------------------------------------------
# The question that is asked
# --------------------------------------------------------------------------------------------


class TestTheNumberOfSuggestions:
    @pytest.mark.parametrize("n", [0, -1, 2.0, True, "3", None])
    def test_a_count_that_is_not_a_whole_number_of_one_or_more_is_refused(
        self, n: Any
    ) -> None:
        problem = problemOver(makeProblem().search_space)

        with pytest.raises(ConfigError, match="policy 'random_policy_v1'") as caught:
            build(problem).suggest(viewOf(problem), n)

        assert caught.value.field_path == "n"
        assert caught.value.component == "policy 'random_policy_v1'"


# --------------------------------------------------------------------------------------------
# State and notifications
# --------------------------------------------------------------------------------------------


class TestState:
    def test_the_exported_state_is_empty(self) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem)

        assert policy.exportState() == {}
        policy.suggest(viewOf(problem), 3)
        assert policy.exportState() == {}

    def test_each_export_is_a_new_dictionary(self) -> None:
        policy = build()

        first = policy.exportState()
        first["oops"] = 1

        assert policy.exportState() == {}

    def test_the_empty_state_round_trips_through_import_and_through_json(self) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem)
        policy.suggest(viewOf(problem), 2)

        stored = json.loads(json.dumps(policy.exportState()))
        restored = build(problem)
        restored.importState(stored)

        assert stored == {}
        assert restored.exportState() == {}

    def test_a_policy_that_imported_the_state_suggests_as_the_one_that_exported_it(
        self,
    ) -> None:
        problem = problemOver(makeProblem().search_space)
        uninterrupted = build(problem)
        view = viewOf(problem, ask_index=3)
        uninterrupted.suggest(viewOf(problem, ask_index=0), 4)
        uninterrupted.suggest(viewOf(problem, ask_index=1), 2)
        reloaded = build(problem)

        reloaded.importState(uninterrupted.exportState())

        assert reloaded.suggest(view, 4) == uninterrupted.suggest(view, 4)

    @pytest.mark.parametrize("state", [{"asked": 3}, {"notes": []}, {"a": 1, "b": 2}])
    def test_a_state_that_is_not_empty_is_refused_with_the_name_of_the_policy(
        self, state: dict[str, Any]
    ) -> None:
        with pytest.raises(ConfigError, match="policy 'random_policy_v1'") as caught:
            build().importState(state)

        assert caught.value.field_path == "state"
        assert caught.value.component == "policy 'random_policy_v1'"
        assert "exported" in caught.value.suggestion


class TestNotifications:
    def test_every_change_of_a_trial_is_accepted_and_changes_nothing(self) -> None:
        problem = problemOver(makeProblem().search_space)
        policy = build(problem)
        before = policy.suggest(viewOf(problem), 3)

        policy.trialChanged(makeTrial(0, TrialState.pending), None)
        policy.trialChanged(makeTrial(0, TrialState.completed), TrialState.pending)
        policy.trialChanged(makeTrial(1, TrialState.failed), None)
        policy.trialChanged(makeTrial(2, TrialState.abandoned), TrialState.pending)

        assert policy.suggest(viewOf(problem), 3) == before
        assert policy.exportState() == {}


# --------------------------------------------------------------------------------------------
# A participant of the compatibility check
# --------------------------------------------------------------------------------------------


class TestTheCompatibilityCheck:
    def test_the_policy_has_a_label_and_needs_and_supports_nothing(self) -> None:
        policy = build()

        assert policy.label == "policy 'random_policy_v1'"
        assert policy.needs == frozenset()
        assert policy.supports == frozenset()

    def test_a_study_made_of_the_policy_alone_is_compatible(self) -> None:
        checkCompatibility([build()])
