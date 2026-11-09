"""Unit tests for `FeasibleSamplingConfig` and for what it changes in the sampler.

Specification 3.4 and 10.8, roadmap task P3-05. Four groups: the schema (defaults,
descriptions, validation, the stored form), one test per configured value showing that
changing it changes what `sampleFeasibleEncoded` or `EncodedConstraints` does while the
default reproduces what they did before, the rule about which configuration the sampler
uses, and the check that no module constant and no numeric signature default is left in
the two files.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from pydantic import BaseModel, Field, ValidationError

from planexs import LinearConstraint, NonlinearConstraint, Real, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.FeasibleSamplingConfig import FeasibleSamplingConfig
from planexs.encoding.sampleFeasibleEncoded import sampleFeasibleEncoded
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder

REPO_ROOT = Path(__file__).resolve().parents[3]
ENCODING_PACKAGE = REPO_ROOT / "src" / "planexs" / "encoding"
_SAMPLER_MODULE = "planexs.encoding.sampleFeasibleEncoded"

#: The values that the sampler and the constraints used before they became settings:
#: `_MAX_ROUNDS`, `_OVERSAMPLE`, the `64` of `max(_OVERSAMPLE * n, 64)`,
#: `DEFAULT_POLYTOPE_BURN_IN`, `DEFAULT_POLYTOPE_THINNING` and the `tolerance=1e-6` default of
#: both `sampleFeasibleEncoded` and `EncodedConstraints.isFeasible`. The defaults of the schema
#: must reproduce them, so a study that sets nothing samples as it always did.
OLD_BEHAVIOUR: dict[str, Any] = {
    "max_rounds": 25,
    "oversampling_factor": 8,
    "min_draw": 64,
    "polytope_burn_in": 200,
    "polytope_thinning": 10,
    "feasibility_tolerance": 1e-6,
}

#: How each default is spelled in the description of its field.
DEFAULT_SPELLINGS: dict[str, str] = {
    "max_rounds": "the default is 25",
    "oversampling_factor": "the default is 8",
    "min_draw": "the default is 64",
    "polytope_burn_in": "the default is 200",
    "polytope_thinning": "the default is 10",
    "feasibility_tolerance": "the default is 1e-6",
}

#: The unit that each description has to state (specification 10.4, roadmap P3-05).
UNIT_PHRASES: dict[str, str] = {
    "max_rounds": "in rounds",
    "oversampling_factor": "in candidates per requested point",
    "min_draw": "in candidates",
    "polytope_burn_in": "in steps of the chain",
    "polytope_thinning": "in steps of the chain",
    "feasibility_tolerance": "as a fraction of that constraint's own scale",
}

#: What the two long comments of `sampleFeasibleEncoded.py` said, which now live in the
#: descriptions of the two fields. Each is a measurement or a reason that must not be lost.
MOVED_FROM_THE_COMMENTS: dict[str, tuple[str, ...]] = {
    "polytope_burn_in": (
        "10000",
        "simplex",
        "indistinguishable means and variances",
        "3 and 10 dimensions",
        "chosen from a measurement and not from theory",
    ),
    "polytope_thinning": (
        "2.7 times slower",
        "0.76",
        "0.43",
        "seed the optimizer's restarts",
        "Raise it for a high-dimensional polytope",
    ),
}


def failureOf(data: Any) -> ConfigError:
    """The `ConfigError` that validating `data` as a `FeasibleSamplingConfig` raises."""
    with pytest.raises(ConfigError) as caught:
        validateConfig(FeasibleSamplingConfig, data)
    return caught.value


def configWith(**changes: Any) -> FeasibleSamplingConfig:
    """The default configuration with some fields changed."""
    return FeasibleSamplingConfig(**changes)


def candidatesDrawn(
    expression: str, comparator: Any, bound: float, n: int, config: FeasibleSamplingConfig
) -> int:
    """How many candidates the sampler draws before it gives up on a region nobody can reach.

    Args:
        expression: A nonlinear constraint over `x` and `y`, both in `[0, 1]`.
        comparator: Its comparator.
        bound: Its bound.
        n: The number of points asked for.
        config: The sampling settings.

    Returns:
        The number the error message states.
    """
    _, encoder, constraints = unitSquare(expression, comparator, bound)
    with pytest.raises(RuntimeError) as caught:
        sampleFeasibleEncoded(encoder, constraints, n, seed=0, config=config)
    match = re.search(r"among (\d+) space-filling", str(caught.value))
    assert match is not None, str(caught.value)
    return int(match.group(1))


def unitSquare(
    expression: str, comparator: Any, bound: float
) -> tuple[SearchSpace, SearchSpaceEncoder, EncodedConstraints]:
    """The unit square with one nonlinear constraint, and its encoder and constraints."""
    space = SearchSpace(
        [Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)],
        constraints=[NonlinearConstraint(expression, comparator, bound)],
    )
    encoder = SearchSpaceEncoder(space)
    return space, encoder, EncodedConstraints(space, encoder)


def mixture() -> tuple[SearchSpace, SearchSpaceEncoder, EncodedConstraints]:
    """Three fractions that sum to one: a space the polytope sampler draws from."""
    space = SearchSpace(
        [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
        constraints=[LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")],
    )
    encoder = SearchSpaceEncoder(space)
    return space, encoder, EncodedConstraints(space, encoder)


@pytest.fixture
def polytope_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Records the arguments that the hit-and-run sampler of BoTorch is called with."""
    module = importlib.import_module(_SAMPLER_MODULE)
    calls: list[dict[str, Any]] = []
    original = module.sample_q_batches_from_polytope

    def recording(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(
            {
                "n": kwargs["n"],
                "n_burnin": kwargs["n_burnin"],
                "n_thinning": kwargs["n_thinning"],
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "sample_q_batches_from_polytope", recording)
    return calls


class StudyLikeConfig(BaseModel):
    """A study-level model that holds the sampling settings as one of its fields."""

    name: str = Field(default="study", description="The study's name.")
    sampling: FeasibleSamplingConfig = Field(
        default_factory=FeasibleSamplingConfig, description="The sampling settings."
    )


class TestDefaults:
    def test_a_bare_configuration_reproduces_the_old_behaviour(self) -> None:
        assert FeasibleSamplingConfig().model_dump() == OLD_BEHAVIOUR

    def test_the_defaults_are_written_in_the_fields_of_the_schema(self) -> None:
        # Specification 3.4: a default is written in exactly one place, the schema field.
        written = {
            name: field.default for name, field in FeasibleSamplingConfig.model_fields.items()
        }
        assert written == OLD_BEHAVIOUR

    def test_an_empty_mapping_validates_to_the_defaults(self) -> None:
        assert validateConfig(FeasibleSamplingConfig, {}) == FeasibleSamplingConfig()

    def test_setting_one_field_leaves_the_others_at_their_defaults(self) -> None:
        assert configWith(max_rounds=3).model_dump() == {**OLD_BEHAVIOUR, "max_rounds": 3}

    def test_constraints_built_with_no_configuration_use_the_defaults(self) -> None:
        _, _, constraints = mixture()
        assert constraints.config == FeasibleSamplingConfig()

    def test_constraints_keep_the_configuration_they_were_given(self) -> None:
        space, encoder, _ = mixture()
        config = configWith(min_draw=7)
        assert EncodedConstraints(space, encoder, config).config is config


class TestTheSchema:
    def test_it_has_exactly_the_six_settings_of_the_card(self) -> None:
        assert list(FeasibleSamplingConfig.model_fields) == list(OLD_BEHAVIOUR)

    def test_every_field_has_a_description(self) -> None:
        for name, field in FeasibleSamplingConfig.model_fields.items():
            assert field.description, f"field {name!r} has no description"
            assert field.description.strip() == field.description, name

    def test_every_description_says_what_the_default_is(self) -> None:
        for name, field in FeasibleSamplingConfig.model_fields.items():
            assert DEFAULT_SPELLINGS[name] in (field.description or "").lower(), name

    def test_every_description_states_its_unit(self) -> None:
        # Roadmap P3-05: "every field description states its meaning and unit".
        for name, field in FeasibleSamplingConfig.model_fields.items():
            assert UNIT_PHRASES[name] in (field.description or "").lower(), name

    @pytest.mark.parametrize("name", ["polytope_burn_in", "polytope_thinning"])
    def test_the_two_comments_of_the_sampler_moved_into_the_descriptions(
        self, name: str
    ) -> None:
        description = FeasibleSamplingConfig.model_fields[name].description or ""
        for fact in MOVED_FROM_THE_COMMENTS[name]:
            assert fact in description, (name, fact)

    def test_the_sampler_file_no_longer_holds_those_comments(self) -> None:
        source = (ENCODING_PACKAGE / "sampleFeasibleEncoded.py").read_text(encoding="utf-8")
        for fact in ("lag-1 autocorrelation", "2.7 times", "simplex, whose exact law"):
            assert fact not in source, fact

    def test_it_is_a_model_of_its_own_and_not_a_component_configuration(self) -> None:
        assert issubclass(FeasibleSamplingConfig, BaseModel)
        assert not issubclass(FeasibleSamplingConfig, ComponentConfig)
        assert "type" not in FeasibleSamplingConfig.model_fields

    def test_a_configuration_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            FeasibleSamplingConfig().min_draw = 1  # type: ignore[misc]

    def test_a_misspelled_setting_is_an_error_with_its_path(self) -> None:
        error = failureOf({"polytope_burnin": 5})
        assert error.field_path == "polytope_burnin"
        assert "Unknown field" in error.message

    def test_the_suggestion_lists_the_fields_of_the_schema(self) -> None:
        suggestion = failureOf({"rounds": 5}).suggestion
        assert suggestion is not None
        for name in OLD_BEHAVIOUR:
            assert name in suggestion

    def test_it_is_a_field_of_a_study_level_model_with_the_full_path_in_errors(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(StudyLikeConfig, {"sampling": {"polytope_burn_in": -1}})
        assert caught.value.field_path == "sampling.polytope_burn_in"

    def test_a_study_level_model_without_the_field_gets_the_defaults(self) -> None:
        assert validateConfig(StudyLikeConfig, {}).sampling == FeasibleSamplingConfig()


class TestValidation:
    @pytest.mark.parametrize("name", ["max_rounds", "oversampling_factor", "min_draw"])
    @pytest.mark.parametrize("value", [1, 2, 1000])
    def test_a_count_of_one_or_more_is_accepted(self, name: str, value: int) -> None:
        assert getattr(configWith(**{name: value}), name) == value

    @pytest.mark.parametrize("name", ["max_rounds", "oversampling_factor", "min_draw"])
    @pytest.mark.parametrize("bad", [0, -1, 2.5, "many", None], ids=repr)
    def test_a_bad_count_is_rejected_with_the_field_path(self, name: str, bad: Any) -> None:
        assert failureOf({name: bad}).field_path == name

    @pytest.mark.parametrize("value", [0, 1, 200, 10000])
    def test_a_burn_in_of_zero_or_more_is_accepted(self, value: int) -> None:
        assert configWith(polytope_burn_in=value).polytope_burn_in == value

    @pytest.mark.parametrize("bad", [-1, 1.5, "long", None], ids=repr)
    def test_a_bad_burn_in_is_rejected_with_the_field_path(self, bad: Any) -> None:
        assert failureOf({"polytope_burn_in": bad}).field_path == "polytope_burn_in"

    @pytest.mark.parametrize("value", [1, 10, 32])
    def test_a_thinning_of_one_or_more_is_accepted(self, value: int) -> None:
        assert configWith(polytope_thinning=value).polytope_thinning == value

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "thin", None], ids=repr)
    def test_a_bad_thinning_is_rejected_with_the_field_path(self, bad: Any) -> None:
        assert failureOf({"polytope_thinning": bad}).field_path == "polytope_thinning"

    @pytest.mark.parametrize("value", [0.0, 1e-12, 1e-6, 0.1, 5.0])
    def test_a_tolerance_of_zero_or_more_is_accepted(self, value: float) -> None:
        assert configWith(feasibility_tolerance=value).feasibility_tolerance == value

    @pytest.mark.parametrize(
        "bad", [-1e-9, -1.0, float("nan"), float("inf"), "tiny", None], ids=repr
    )
    def test_a_bad_tolerance_is_rejected_with_the_field_path(self, bad: Any) -> None:
        assert failureOf({"feasibility_tolerance": bad}).field_path == "feasibility_tolerance"

    def test_several_problems_are_reported_together(self) -> None:
        error = failureOf({"max_rounds": 0, "polytope_thinning": 0, "bogus": 1})
        for name in ("max_rounds", "polytope_thinning", "bogus"):
            assert name in error.message
        # `field_path` is the path of the first problem, in the order the model declares.
        assert error.field_path == "max_rounds"


class TestTheStoredForm:
    """A study stores its resolved configuration (specification 3.4, 5.11)."""

    def test_the_resolved_configuration_writes_every_field_out(self) -> None:
        assert toResolvedConfig(FeasibleSamplingConfig()) == OLD_BEHAVIOUR

    def test_a_stored_configuration_keeps_its_values_whatever_the_defaults_become(
        self,
    ) -> None:
        stored = toResolvedConfig(configWith(polytope_burn_in=50))
        assert stored == {**OLD_BEHAVIOUR, "polytope_burn_in": 50}
        assert validateConfig(FeasibleSamplingConfig, stored).polytope_burn_in == 50

    def test_resolving_then_validating_then_resolving_gives_the_same_dictionary(self) -> None:
        config = FeasibleSamplingConfig(
            max_rounds=3,
            oversampling_factor=4,
            min_draw=16,
            polytope_burn_in=50,
            polytope_thinning=2,
            feasibility_tolerance=1e-9,
        )
        resolved = toResolvedConfig(config)
        assert toResolvedConfig(validateConfig(FeasibleSamplingConfig, resolved)) == resolved

    def test_the_resolved_dictionary_survives_json_unchanged(self) -> None:
        resolved = toResolvedConfig(configWith(feasibility_tolerance=1e-9))
        assert json.loads(json.dumps(resolved)) == resolved

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_a_configuration_written_to_a_file_format_and_read_back_is_equal(
        self, file_format: Any
    ) -> None:
        # Specification 10.8: Python, JSON and YAML have the same meaning.
        config = configWith(max_rounds=7, feasibility_tolerance=1e-9)
        text = dumpConfig(toResolvedConfig(config), file_format)
        assert validateConfig(FeasibleSamplingConfig, parseConfig(text, file_format)) == config


class TestMaxRounds:
    def test_the_default_gives_up_after_the_candidates_the_old_constants_allowed(self) -> None:
        # 25 rounds of 64, 128, ... candidates: 64 * (1 + 2 + ... + 25) = 64 * 325.
        config = FeasibleSamplingConfig()
        assert candidatesDrawn("x + y", "<=", -1.0, 5, config) == 64 * 325 == 20800

    @pytest.mark.parametrize(
        ("rounds", "expected"), [(1, 64), (2, 64 * 3), (3, 64 * 6), (10, 64 * 55)]
    )
    def test_fewer_rounds_give_up_sooner(self, rounds: int, expected: int) -> None:
        config = configWith(max_rounds=rounds)
        assert candidatesDrawn("x + y", "<=", -1.0, 5, config) == expected

    def test_a_region_the_first_round_misses_is_found_by_the_later_ones(self) -> None:
        # About 0.35% of the unit square: a first round of 64 candidates expects 0.2 of them.
        _, encoder, constraints = unitSquare("x * y", "<=", 0.0004)
        for seed in range(4):
            points = sampleFeasibleEncoded(encoder, constraints, 5, seed=seed)
            assert bool(constraints.isFeasible(points).all())

    def test_with_a_single_round_the_same_region_is_missed(self) -> None:
        _, encoder, constraints = unitSquare("x * y", "<=", 0.0004)
        for seed in range(4):
            with pytest.raises(RuntimeError, match="Found no point"):
                sampleFeasibleEncoded(
                    encoder, constraints, 5, seed=seed, config=configWith(max_rounds=1)
                )

    def test_fewer_points_than_asked_for_are_repeated_to_fill_the_request(self) -> None:
        # One round that finds some points but not all: the ones found fill the request.
        _, encoder, constraints = unitSquare("x * y", "<=", 0.05)
        config = configWith(max_rounds=1, oversampling_factor=1, min_draw=1)
        points = sampleFeasibleEncoded(encoder, constraints, 40, seed=0, config=config)
        assert points.shape == (40, 2)
        assert bool(constraints.isFeasible(points).all())
        # Fewer than 40 distinct points were found, so the request was filled by repeating.
        assert torch.unique(points, dim=0).shape[0] < 40


class TestOversamplingFactor:
    def test_the_first_round_draws_the_factor_times_the_points_asked_for(
        self, polytope_calls: list[dict[str, Any]]
    ) -> None:
        _, encoder, constraints = mixture()
        sampleFeasibleEncoded(encoder, constraints, 20, seed=0)
        sampleFeasibleEncoded(
            encoder, constraints, 20, seed=0, config=configWith(oversampling_factor=12)
        )
        # The default factor of 8 gives 160, above the minimum of 64; 12 gives 240.
        assert [call["n"] for call in polytope_calls] == [8 * 20, 12 * 20]

    def test_the_budget_before_giving_up_follows_the_factor(self) -> None:
        # n = 5 and a factor of 20: 100 candidates in the first round, above the minimum.
        config = configWith(oversampling_factor=20)
        assert candidatesDrawn("x + y", "<=", -1.0, 5, config) == 100 * 325

    def test_the_default_factor_matters_only_when_it_beats_the_minimum(self) -> None:
        config = FeasibleSamplingConfig()
        assert candidatesDrawn("x + y", "<=", -1.0, 1, config) == 64 * 325
        assert candidatesDrawn("x + y", "<=", -1.0, 20, config) == 8 * 20 * 325

    def test_a_larger_factor_finds_a_region_that_one_round_would_miss(self) -> None:
        _, encoder, constraints = unitSquare("x * y", "<=", 0.0004)
        config = configWith(max_rounds=1, oversampling_factor=400)
        for seed in range(4):
            points = sampleFeasibleEncoded(encoder, constraints, 5, seed=seed, config=config)
            assert bool(constraints.isFeasible(points).all())


class TestMinDraw:
    def test_the_first_round_draws_at_least_the_minimum(
        self, polytope_calls: list[dict[str, Any]]
    ) -> None:
        _, encoder, constraints = mixture()
        sampleFeasibleEncoded(encoder, constraints, 5, seed=0)
        sampleFeasibleEncoded(encoder, constraints, 5, seed=0, config=configWith(min_draw=200))
        # Five points at the default factor of 8 are 40, so the default minimum of 64 decides.
        assert [call["n"] for call in polytope_calls] == [64, 200]

    def test_a_minimum_below_the_factor_changes_nothing(
        self, polytope_calls: list[dict[str, Any]]
    ) -> None:
        _, encoder, constraints = mixture()
        sampleFeasibleEncoded(encoder, constraints, 20, seed=0, config=configWith(min_draw=1))
        assert [call["n"] for call in polytope_calls] == [160]

    def test_the_budget_before_giving_up_follows_the_minimum(self) -> None:
        config = configWith(min_draw=200)
        assert candidatesDrawn("x + y", "<=", -1.0, 5, config) == 200 * 325

    def test_a_larger_minimum_finds_a_region_that_one_round_would_miss(self) -> None:
        _, encoder, constraints = unitSquare("x * y", "<=", 0.0004)
        config = configWith(max_rounds=1, min_draw=2000)
        for seed in range(4):
            points = sampleFeasibleEncoded(encoder, constraints, 5, seed=seed, config=config)
            assert bool(constraints.isFeasible(points).all())


class TestPolytopeChain:
    def test_the_defaults_are_the_ones_the_chain_is_run_with(
        self, polytope_calls: list[dict[str, Any]]
    ) -> None:
        _, encoder, constraints = mixture()
        sampleFeasibleEncoded(encoder, constraints, 5, seed=0)
        assert polytope_calls
        assert {(c["n_burnin"], c["n_thinning"]) for c in polytope_calls} == {(200, 10)}

    def test_the_burn_in_is_the_configured_one_and_only_it(
        self, polytope_calls: list[dict[str, Any]]
    ) -> None:
        _, encoder, constraints = mixture()
        config = configWith(polytope_burn_in=7)
        sampleFeasibleEncoded(encoder, constraints, 5, seed=0, config=config)
        assert {(c["n_burnin"], c["n_thinning"]) for c in polytope_calls} == {(7, 10)}

    def test_the_thinning_is_the_configured_one_and_only_it(
        self, polytope_calls: list[dict[str, Any]]
    ) -> None:
        _, encoder, constraints = mixture()
        config = configWith(polytope_thinning=3)
        sampleFeasibleEncoded(encoder, constraints, 5, seed=0, config=config)
        assert {(c["n_burnin"], c["n_thinning"]) for c in polytope_calls} == {(200, 3)}

    def test_the_chain_is_not_used_without_linear_constraints(
        self, polytope_calls: list[dict[str, Any]]
    ) -> None:
        _, encoder, constraints = unitSquare("x * y", "<=", 0.5)
        sampleFeasibleEncoded(encoder, constraints, 5, seed=0)
        assert polytope_calls == []

    def test_the_equality_holds_whatever_the_settings(self) -> None:
        _, encoder, constraints = mixture()
        for burn_in, thinning in ((0, 1), (1, 1), (50, 2), (400, 20)):
            config = configWith(polytope_burn_in=burn_in, polytope_thinning=thinning)
            points = sampleFeasibleEncoded(encoder, constraints, 30, seed=1, config=config)
            sums = points.sum(dim=-1)
            assert torch.allclose(sums, torch.ones(30, dtype=torch.double), atol=1e-6)


class TestFeasibilityTolerance:
    # `x + y <= -0.05` is out of reach in the unit square: its smallest violation is 0.05,
    # at the origin, relative to a scale of 1.
    ORIGIN = torch.zeros(2, dtype=torch.double)

    def test_the_violation_the_tolerance_is_measured_against(self) -> None:
        _, _, constraints = unitSquare("x + y", "<=", -0.05)
        assert float(constraints.violation(self.ORIGIN)) == pytest.approx(0.05)

    def test_the_default_tolerance_is_the_old_one(self) -> None:
        _, _, constraints = unitSquare("x + y", "<=", -0.05)
        assert constraints.config.feasibility_tolerance == 1e-6
        assert not bool(constraints.isFeasible(self.ORIGIN))
        assert bool(constraints.isFeasible(self.ORIGIN, tolerance=0.05 + 1e-9))

    def test_is_feasible_takes_its_default_from_the_configuration(self) -> None:
        space, encoder, _ = unitSquare("x + y", "<=", -0.05)
        lax = EncodedConstraints(space, encoder, configWith(feasibility_tolerance=0.1))
        strict = EncodedConstraints(space, encoder, configWith(feasibility_tolerance=0.01))
        assert bool(lax.isFeasible(self.ORIGIN))
        assert not bool(strict.isFeasible(self.ORIGIN))

    def test_an_argument_of_the_call_overrides_the_configuration(self) -> None:
        space, encoder, _ = unitSquare("x + y", "<=", -0.05)
        lax = EncodedConstraints(space, encoder, configWith(feasibility_tolerance=0.1))
        assert not bool(lax.isFeasible(self.ORIGIN, tolerance=0.0))
        assert bool(EncodedConstraints(space, encoder).isFeasible(self.ORIGIN, tolerance=0.1))

    def test_the_sampler_accepts_candidates_up_to_the_configured_tolerance(self) -> None:
        _, encoder, constraints = unitSquare("x + y", "<=", -0.05)
        with pytest.raises(RuntimeError, match="Found no point"):
            sampleFeasibleEncoded(encoder, constraints, 5, seed=0)
        points = sampleFeasibleEncoded(
            encoder, constraints, 5, seed=0, config=configWith(feasibility_tolerance=0.1)
        )
        assert points.shape == (5, 2)
        assert bool((constraints.violation(points) <= 0.1).all())

    def test_a_tolerance_of_zero_accepts_only_points_that_meet_the_constraint(self) -> None:
        _, encoder, constraints = unitSquare("x * y", "<=", 0.5)
        config = configWith(feasibility_tolerance=0.0)
        points = sampleFeasibleEncoded(encoder, constraints, 20, seed=0, config=config)
        assert bool((constraints.violation(points) == 0.0).all())


class TestWhichConfigurationTheSamplerUses:
    def test_it_uses_the_configuration_the_constraints_were_built_with(self) -> None:
        space, encoder, _ = unitSquare("x + y", "<=", -0.05)
        constraints = EncodedConstraints(space, encoder, configWith(feasibility_tolerance=0.1))
        points = sampleFeasibleEncoded(encoder, constraints, 5, seed=0)
        assert points.shape == (5, 2)

    def test_an_explicit_configuration_wins_over_that_of_the_constraints(self) -> None:
        space, encoder, _ = unitSquare("x + y", "<=", -0.05)
        constraints = EncodedConstraints(space, encoder, configWith(feasibility_tolerance=0.1))
        with pytest.raises(RuntimeError, match="Found no point"):
            sampleFeasibleEncoded(
                encoder, constraints, 5, seed=0, config=configWith(feasibility_tolerance=0.01)
            )

    def test_the_rounds_of_the_constraints_configuration_apply_too(self) -> None:
        space, encoder, _ = unitSquare("x + y", "<=", -1.0)
        constraints = EncodedConstraints(space, encoder, configWith(max_rounds=2))
        with pytest.raises(RuntimeError) as caught:
            sampleFeasibleEncoded(encoder, constraints, 5, seed=0)
        assert "among 192 space-filling" in str(caught.value)


class TestNoModuleConstantOrSignatureLiteral:
    """Roadmap P3-05: no module constant or signature literal remains in these files."""

    FILES = ("sampleFeasibleEncoded.py", "EncodedConstraints.py")

    @staticmethod
    def _tree(file_name: str) -> ast.Module:
        return ast.parse((ENCODING_PACKAGE / file_name).read_text(encoding="utf-8"))

    @staticmethod
    def _checker() -> ModuleType:
        name = "check_policy_values_for_the_sampler_tests"
        spec = importlib.util.spec_from_file_location(
            name, REPO_ROOT / "scripts" / "check_policy_values.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.mark.parametrize("file_name", FILES)
    def test_the_file_holds_no_module_level_assignment(self, file_name: str) -> None:
        assignments = [
            ast.unparse(node)
            for node in self._tree(file_name).body
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign)
        ]
        assert assignments == []

    @pytest.mark.parametrize("file_name", FILES)
    def test_no_function_has_a_number_as_the_default_of_a_parameter(
        self, file_name: str
    ) -> None:
        offenders: list[str] = []
        for node in ast.walk(self._tree(file_name)):
            if not isinstance(node, ast.FunctionDef):
                continue
            defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d)]
            for default in defaults:
                is_number = isinstance(default, ast.Constant) and isinstance(
                    default.value, int | float
                )
                if is_number and not isinstance(default.value, bool):  # type: ignore[attr-defined]
                    offenders.append(f"{node.name}: {ast.unparse(default)}")
        assert offenders == []

    @pytest.mark.parametrize("file_name", FILES)
    def test_the_file_has_no_numeric_literal_that_is_not_allowed_or_structural(
        self, file_name: str
    ) -> None:
        source = (ENCODING_PACKAGE / file_name).read_text(encoding="utf-8")
        found = self._checker().findViolations(source, f"src/planexs/encoding/{file_name}")
        assert [violation.describe() for violation in found] == []

    def test_the_old_names_are_gone_everywhere_in_the_source(self) -> None:
        for path in (REPO_ROOT / "src").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for old in (
                "DEFAULT_POLYTOPE_BURN_IN",
                "DEFAULT_POLYTOPE_THINNING",
                "_OVERSAMPLE",
            ):
                assert old not in source, (path.name, old)
            assert "_MAX_ROUNDS" not in source, path.name

    def test_the_baseline_no_longer_lists_the_two_files(self) -> None:
        baseline = (REPO_ROOT / "scripts" / "policy_values_baseline.txt").read_text(
            encoding="utf-8"
        )
        assert "sampleFeasibleEncoded.py" not in baseline
        assert "EncodedConstraints.py" not in baseline

    def test_the_legacy_strategy_gets_its_defaults_from_the_schema(self) -> None:
        # The strategy keeps its two constructor arguments, and `None` means the schema's.
        from planexs import AlphaAcquisitionStrategy

        strategy = AlphaAcquisitionStrategy()
        assert strategy._sampling == FeasibleSamplingConfig()
        tuned = AlphaAcquisitionStrategy(polytope_burn_in=11, polytope_thinning=4)
        assert tuned._sampling == configWith(polytope_burn_in=11, polytope_thinning=4)
