"""Unit tests for `qlog_nehvi_v1`, its builder and its configuration (roadmap task P5-05,
specification 5.8, FR2, FR10, FR15).

The acceptance tests of the task: on a toy two-outcome surrogate a point that extends the
Pareto front scores higher than a dominated one; the reference-point rules, explicit and
inferred; the directions of the objectives; the outcome constraints are passed through. The
rest tests what the registration says (needs, supports), the pending points, the pruning of
the baseline, the determinism of the samples, and what the builder refuses.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from pydantic import ValidationError
from toy_pareto_surrogates import F1, F2, G, makeParetoSurrogate
from toy_surrogates import candidates, makeSurrogateOf, makeToyContext

import planexs.acquisition
import planexs.goals
import planexs.surrogates  # noqa: F401
from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.acquisition.baselinePoints import baselinePoints
from planexs.acquisition.ExplicitReferencePointConfig import ExplicitReferencePointConfig
from planexs.acquisition.InferredReferencePointConfig import InferredReferencePointConfig
from planexs.acquisition.QLogNEHVIBuilder import QLogNEHVIBuilder
from planexs.acquisition.QLogNEHVIConfig import QLogNEHVIConfig
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig
from planexs.goals.OptimizeGoal import OptimizeGoal
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig

MAXIMIZE_BOTH: dict[str, Any] = {"outcomes": {F1: "maximize", F2: "maximize"}}
MINIMIZE_F1: dict[str, Any] = {"outcomes": {F1: "minimize", F2: "maximize"}}

#: Places of the toy space. The observed points are eight, spread over `[0.05, 0.95]`. With
#: both objectives maximized the front is to the right of the peak of `f2` (`x = 0.52`) and
#: the observed points to its left are dominated: `DOMINATED` is one of them, `BEYOND` lies
#: past the last observed point (`x = 0.95`), where `f1` is higher than anywhere observed and
#: `f2` is a little lower, so that it extends the front, and `ON_THE_FRONT` is between two
#: observed points of it. With `f1` minimized it is the other way round.
DOMINATED = 0.3
ON_THE_FRONT = 0.75
BEYOND = 1.0
LEFT_END = 0.02


def constrained(comparator: str, bound: float) -> dict[str, Any]:
    """The goal that maximizes `f1` and `f2` and constrains `g`."""
    return {
        **MAXIMIZE_BOTH,
        "outcome_constraints": [{"outcome": G, "comparator": comparator, "bound": bound}],
    }


def build(
    goal: dict[str, Any],
    *,
    surrogate: Any = None,
    pending: torch.Tensor | None = None,
    seed: int = 7,
    **settings: Any,
) -> Any:
    """The native function for a goal, built the way a policy builds it."""
    model = makeParetoSurrogate((F1, F2, G)) if surrogate is None else surrogate
    context = makeToyContext(model, goal, pending_points=pending, seed=seed)
    return QLogNEHVIBuilder(QLogNEHVIConfig(**settings)).build(context)


def scores(
    goal: dict[str, Any],
    *places: float,
    surrogate: Any = None,
    pending: torch.Tensor | None = None,
    seed: int = 7,
    **settings: Any,
) -> list[float]:
    """The acquisition value of each place, for a goal.

    Args:
        goal: The fields of the optimize goal.
        places: The value of `x` of each single-point candidate.
        surrogate: The surrogate. `None` means the three-outcome toy one.
        pending: The pending points, shape `(n, 1)`, or `None`.
        seed: The seed of the generator of the build.
        settings: Any setting of `QLogNEHVIConfig` to replace.
    """
    function = build(goal, surrogate=surrogate, pending=pending, seed=seed, **settings)
    return [float(value) for value in function(candidates(*places)).detach()]


class TestTheConfiguration:
    def test_the_defaults(self) -> None:
        config = QLogNEHVIConfig()

        assert config.type == "qlog_nehvi_v1"
        assert config.n_samples == 128
        assert config.prune_baseline is True
        assert config.reference_point == InferredReferencePointConfig()

    def test_the_default_reference_point_is_the_nadir_rule_with_the_margin_botorch_uses(
        self,
    ) -> None:
        reference_point = QLogNEHVIConfig().reference_point

        assert isinstance(reference_point, InferredReferencePointConfig)
        assert reference_point.rule == "nadir_margin"
        assert reference_point.margin == 0.1

    @pytest.mark.parametrize("n_samples", [0, -1, -128])
    def test_a_sample_count_below_one_is_refused(self, n_samples: int) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            QLogNEHVIConfig(n_samples=n_samples)

    @pytest.mark.parametrize("n_samples", [2.5, "many", None])
    def test_a_sample_count_that_is_not_a_whole_number_is_refused(
        self, n_samples: Any
    ) -> None:
        with pytest.raises(ValidationError):
            QLogNEHVIConfig(n_samples=n_samples)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="n_sample"):
            QLogNEHVIConfig(n_sample=64)  # type: ignore[call-arg]

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ValidationError):
            QLogNEHVIConfig(type="qlog_ehvi_v1")  # type: ignore[arg-type]

    def test_every_field_says_what_it_means(self) -> None:
        for model in (
            QLogNEHVIConfig,
            ExplicitReferencePointConfig,
            InferredReferencePointConfig,
        ):
            for name, field in model.model_fields.items():
                assert field.description, (model.__name__, name)
                assert len(field.description) > 40, (model.__name__, name)

    def test_the_description_of_the_sample_count_says_why_its_default_is_what_it_is(
        self,
    ) -> None:
        description = QLogNEHVIConfig.model_fields["n_samples"].description or ""

        assert "128" in description
        assert "512" in description

    def test_the_description_of_the_margin_says_where_its_default_comes_from(self) -> None:
        description = InferredReferencePointConfig.model_fields["margin"].description or ""

        assert "0.1" in description
        assert "Ishibuchi" in description


class TestTheReferencePointConfiguration:
    def test_an_explicit_reference_point_is_read_from_a_mapping(self) -> None:
        resolved = resolveComponent(
            acquisitionRegistry(),
            {
                "type": "qlog_nehvi_v1",
                "reference_point": {"kind": "explicit", "values": {F1: 0.0, F2: -1.5}},
            },
        )

        assert isinstance(resolved.config, QLogNEHVIConfig)
        assert isinstance(resolved.config.reference_point, ExplicitReferencePointConfig)
        assert resolved.config.reference_point.values == {F1: 0.0, F2: -1.5}
        builder = resolved.entry.builder(resolved.config)
        assert isinstance(builder, QLogNEHVIBuilder)
        assert builder.config is resolved.config

    def test_an_inferred_reference_point_is_read_from_a_mapping(self) -> None:
        resolved = resolveComponent(
            acquisitionRegistry(),
            {
                "type": "qlog_nehvi_v1",
                "reference_point": {
                    "kind": "inferred",
                    "rule": "nadir_margin",
                    "margin": 0.25,
                },
            },
        )

        assert isinstance(resolved.config, QLogNEHVIConfig)
        assert resolved.config.reference_point == InferredReferencePointConfig(margin=0.25)

    def test_the_kind_picks_the_form(self) -> None:
        # A mapping with the values of an explicit point and the kind of an inferred one is
        # not read as an explicit one.
        with pytest.raises(ValidationError):
            QLogNEHVIConfig(
                reference_point={"kind": "inferred", "values": {F1: 0.0, F2: 0.0}}  # type: ignore[arg-type]
            )

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="kind"):
            QLogNEHVIConfig(reference_point={"kind": "nadir"})  # type: ignore[arg-type]

    def test_an_explicit_reference_point_needs_two_values(self) -> None:
        with pytest.raises(ValidationError, match="at least 2"):
            ExplicitReferencePointConfig(values={F1: 0.0})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_an_explicit_value_must_be_finite(self, value: float) -> None:
        with pytest.raises(ValidationError):
            ExplicitReferencePointConfig(values={F1: value, F2: 0.0})

    def test_an_unknown_rule_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="rule"):
            InferredReferencePointConfig(rule="pareto_mean")  # type: ignore[arg-type]

    @pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
    def test_a_margin_must_be_finite_and_not_negative(self, margin: float) -> None:
        with pytest.raises(ValidationError):
            InferredReferencePointConfig(margin=margin)

    def test_a_margin_of_zero_is_allowed(self) -> None:
        assert InferredReferencePointConfig(margin=0.0).margin == 0.0

    def test_both_forms_are_frozen_and_refuse_unknown_fields(self) -> None:
        explicit = ExplicitReferencePointConfig(values={F1: 0.0, F2: 0.0})
        inferred = InferredReferencePointConfig()

        with pytest.raises(ValidationError):
            explicit.kind = "inferred"  # type: ignore[assignment]
        with pytest.raises(ValidationError):
            inferred.margin = 0.5
        with pytest.raises(ValidationError):
            ExplicitReferencePointConfig(values={F1: 0.0, F2: 0.0}, margin=0.1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            InferredReferencePointConfig(values={F1: 0.0})  # type: ignore[call-arg]


class TestTheRegistration:
    def test_it_is_registered_in_the_acquisition_registry(self) -> None:
        entry = acquisitionRegistry().lookup("qlog_nehvi_v1")

        assert entry.builder == QLogNEHVIBuilder.fromConfig
        assert entry.config_schema is QLogNEHVIConfig
        assert issubclass(QLogNEHVIBuilder, AbstractAcquisitionBuilder)
        assert entry.description.strip()

    def test_it_needs_a_gaussian_posterior_several_objectives_and_an_optimize_goal(
        self,
    ) -> None:
        entry = acquisitionRegistry().lookup("qlog_nehvi_v1")

        assert entry.needs == {"posterior:gaussian", "objectives:multi", "goal:optimize"}

    def test_it_supports_pending_points_outcome_constraints_and_gradients(self) -> None:
        entry = acquisitionRegistry().lookup("qlog_nehvi_v1")

        assert entry.supports == {
            "pending:supported",
            "outcome_constraints:supported",
            "acquisition:differentiable",
        }

    def test_it_needs_no_reference_set(self) -> None:
        # The reference point of the hypervolume is configuration, not the reference set of an
        # integration.
        assert QLogNEHVIBuilder(QLogNEHVIConfig()).referencePointCount() == 0


class TestTheCompatibilityCheck:
    """The tags of the entry, read by the check that a study runs when it is built."""

    @staticmethod
    def participants(goal: Any) -> list[Any]:
        acquisition = acquisitionRegistry().lookup("qlog_nehvi_v1")
        return [
            goal,
            DeclaredParticipant(
                label="acquisition 'qlog_nehvi_v1'",
                needs=acquisition.needs,
                supports=acquisition.supports,
            ),
            # No surrogate is registered yet, so the one that supports what this entry needs of
            # it is declared here.
            DeclaredParticipant(
                label="surrogate 'toy_v1'", supports=frozenset({"posterior:gaussian"})
            ),
        ]

    def test_a_pareto_goal_finds_its_needs_met(self) -> None:
        checkCompatibility(
            self.participants(OptimizeGoal(OptimizeGoalConfig(**MAXIMIZE_BOTH)))
        )

    def test_a_pareto_goal_with_outcome_constraints_finds_its_need_met(self) -> None:
        goal = OptimizeGoal(OptimizeGoalConfig(**constrained("<=", 0.5)))

        assert "outcome_constraints:supported" in goal.needs
        checkCompatibility(self.participants(goal))

    def test_a_weighted_goal_is_refused_for_its_objective(self) -> None:
        goal = OptimizeGoal(
            OptimizeGoalConfig(
                outcomes={F1: "maximize", F2: "minimize"}, weights={F1: 1.0, F2: 2.0}
            )
        )

        with pytest.raises(CompatibilityError, match="objectives:multi"):
            checkCompatibility(self.participants(goal))

    def test_a_goal_with_one_outcome_is_refused_for_its_objective(self) -> None:
        goal = OptimizeGoal(OptimizeGoalConfig(outcomes={F1: "maximize"}))

        with pytest.raises(CompatibilityError, match="objectives:multi"):
            checkCompatibility(self.participants(goal))

    def test_a_level_set_goal_is_refused_for_its_kind(self) -> None:
        goal = LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=F1))

        with pytest.raises(CompatibilityError, match="goal:optimize"):
            checkCompatibility(self.participants(goal))


class TestTheHypervolumeImprovement:
    """Roadmap P5-05, done when (a): on a toy two-outcome surrogate a point that extends the
    Pareto front scores higher than a dominated one."""

    def test_the_premise_one_point_extends_the_predicted_front_and_the_other_is_dominated(
        self,
    ) -> None:
        # What the next tests rest on, read from the model and not assumed.
        model = makeParetoSurrogate((F1, F2)).model
        with torch.no_grad():
            observed = model.posterior(baselinePoints(model)).mean
            extending = model.posterior(candidates(BEYOND).reshape(1, 1)).mean
            dominated = model.posterior(candidates(DOMINATED).reshape(1, 1)).mean

        assert bool(is_non_dominated(torch.cat([observed, extending]))[-1])
        assert not bool(is_non_dominated(torch.cat([observed, dominated]))[-1])

    def test_a_point_that_extends_the_front_scores_higher_than_a_dominated_one(self) -> None:
        dominated, extending = scores(MAXIMIZE_BOTH, DOMINATED, BEYOND)

        # A log: the gap is orders of magnitude of hypervolume.
        assert extending > dominated + 10.0

    def test_a_point_on_the_front_scores_higher_than_a_dominated_one(self) -> None:
        dominated, on_the_front = scores(MAXIMIZE_BOTH, DOMINATED, ON_THE_FRONT)

        assert on_the_front > dominated + 10.0

    def test_every_dominated_place_scores_below_every_place_of_the_front(self) -> None:
        dominated = scores(MAXIMIZE_BOTH, 0.1, 0.2, DOMINATED, 0.4)
        front = scores(MAXIMIZE_BOTH, 0.6, ON_THE_FRONT, 0.9, BEYOND)

        assert max(dominated) < min(front)

    def test_the_value_is_the_same_at_a_place_whichever_batch_it_is_scored_in(self) -> None:
        function = build(MAXIMIZE_BOTH)
        alone = function(candidates(ON_THE_FRONT)).detach()
        together = function(candidates(DOMINATED, ON_THE_FRONT, BEYOND)).detach()

        assert together[1].item() == pytest.approx(alone.item(), abs=1e-9)


class TestTheDirections:
    """Roadmap P5-05, done when (a): the directions of the objectives are respected."""

    def test_when_both_are_maximized_the_right_of_the_space_is_the_front(self) -> None:
        left, right = scores(MAXIMIZE_BOTH, 0.1, 0.85)

        assert right > left + 10.0

    def test_when_one_is_minimized_the_left_of_the_space_is_the_front(self) -> None:
        # f1 = x is better small now, and f2 peaks at 0.52: the points to the right of the
        # peak are worse in both.
        left, right = scores(MINIMIZE_F1, 0.1, 0.85)

        assert left > right + 10.0

    def test_the_two_directions_extend_the_front_at_opposite_ends(self) -> None:
        (maximize_beyond,) = scores(MAXIMIZE_BOTH, BEYOND)
        (maximize_left_end,) = scores(MAXIMIZE_BOTH, LEFT_END)
        (minimize_beyond,) = scores(MINIMIZE_F1, BEYOND)
        (minimize_left_end,) = scores(MINIMIZE_F1, LEFT_END)

        assert maximize_beyond > maximize_left_end + 10.0
        assert minimize_left_end > minimize_beyond + 10.0

    def test_minimizing_an_outcome_is_maximizing_its_negative(self) -> None:
        # The same model of the negated outcome, with the other direction. The two are the same
        # in distribution and not sample by sample (the noise of the posterior is not negated
        # with its mean), so the values agree up to the error of the estimate, and rank the
        # places alike.
        negated = makeSurrogateOf({F1: lambda x: -x, F2: lambda x: torch.sin(3.0 * x)})
        places = (0.1, DOMINATED, 0.6, ON_THE_FRONT, BEYOND)
        settings: dict[str, Any] = {"n_samples": 1024}

        minimized = scores(
            MINIMIZE_F1, *places, surrogate=makeParetoSurrogate((F1, F2)), **settings
        )
        maximized = scores(MAXIMIZE_BOTH, *places, surrogate=negated, **settings)

        assert maximized == pytest.approx(minimized, abs=0.5)
        assert sorted(range(5), key=minimized.__getitem__) == sorted(
            range(5), key=maximized.__getitem__
        )

    def test_the_order_of_the_goal_and_the_order_of_the_outputs_do_not_matter(self) -> None:
        goal = {"outcomes": {F2: "maximize", F1: "minimize"}}
        places = (0.1, 0.45, 0.85)

        assert scores(goal, *places, surrogate=makeParetoSurrogate((F2, F1))) == pytest.approx(
            scores(MINIMIZE_F1, *places, surrogate=makeParetoSurrogate((F1, F2))), abs=1.0
        )


class TestTheReferencePoint:
    """Roadmap P5-05, done when (a): the reference-point rules."""

    @staticmethod
    def explicit(**values: float) -> dict[str, Any]:
        return {"reference_point": ExplicitReferencePointConfig(values=values)}

    def test_an_explicit_point_reaches_the_function_in_the_space_where_all_are_maximized(
        self,
    ) -> None:
        function = build(MINIMIZE_F1, **self.explicit(f1=0.9, f2=0.1))

        # f1 is minimized: a result no worse than 0.9 counts, which is -0.9 once negated.
        assert function.ref_point.tolist() == pytest.approx([-0.9, 0.1])

    def test_an_explicit_point_for_two_maximized_objectives_is_taken_as_it_is(self) -> None:
        function = build(MAXIMIZE_BOTH, **self.explicit(f1=0.2, f2=0.3))

        assert function.ref_point.tolist() == pytest.approx([0.2, 0.3])

    def test_an_explicit_point_is_placed_by_name_whatever_the_order_of_the_outputs(
        self,
    ) -> None:
        function = build(
            MAXIMIZE_BOTH,
            surrogate=makeParetoSurrogate((G, F2, F1)),
            **self.explicit(f1=0.2, f2=0.7),
        )

        # The objectives are in the order of the outputs of the model: f2, then f1.
        assert function.ref_point.tolist() == pytest.approx([0.7, 0.2])

    def test_an_explicit_point_changes_what_a_candidate_is_worth(self) -> None:
        # At x = 0.45 f1 is 0.45: a result that adds volume over a reference point of 0.0 and
        # adds none over a reference point of 0.6, which the point does not beat.
        (low,) = scores(MAXIMIZE_BOTH, 0.45, **self.explicit(f1=0.0, f2=0.0))
        (high,) = scores(MAXIMIZE_BOTH, 0.45, **self.explicit(f1=0.6, f2=0.0))

        assert low > high + 10.0

    def test_a_reference_point_that_nothing_beats_leaves_every_candidate_worthless(
        self,
    ) -> None:
        values = scores(
            MAXIMIZE_BOTH, 0.1, 0.45, ON_THE_FRONT, **self.explicit(f1=2.0, f2=2.0)
        )

        assert all(value < -30.0 for value in values)

    def test_the_inferred_point_is_the_nadir_of_the_predicted_front_moved_out(self) -> None:
        model = makeParetoSurrogate((F1, F2, G))
        with torch.no_grad():
            means = model.model.posterior(baselinePoints(model.model)).mean[:, :2]
        front = means[is_non_dominated(means)]
        nadir = front.min(dim=0).values
        ideal = front.max(dim=0).values

        for margin in (0.1, 0.3):
            function = build(
                MAXIMIZE_BOTH,
                surrogate=model,
                reference_point=InferredReferencePointConfig(margin=margin),
            )

            expected = nadir - margin * (ideal - nadir)
            assert function.ref_point.tolist() == pytest.approx(expected.tolist(), rel=1e-9)

    def test_the_inferred_point_of_a_minimized_objective_is_in_the_negated_space(self) -> None:
        function = build(MINIMIZE_F1)

        # Observed f1 is in [0.05, 0.95], so the negated front lies in [-0.95, -0.05] and the
        # point is below every member of it.
        assert function.ref_point[0].item() < -0.4
        assert function.ref_point[0].item() < function.X_baseline.min().item() * -1.0

    def test_the_default_is_the_inferred_point(self) -> None:
        explicit_default = build(MAXIMIZE_BOTH)
        stated = build(MAXIMIZE_BOTH, reference_point=InferredReferencePointConfig(margin=0.1))

        assert torch.equal(explicit_default.ref_point, stated.ref_point)

    def test_a_larger_margin_puts_the_inferred_point_further_out(self) -> None:
        near = build(MAXIMIZE_BOTH, reference_point=InferredReferencePointConfig(margin=0.05))
        far = build(MAXIMIZE_BOTH, reference_point=InferredReferencePointConfig(margin=0.5))

        assert (far.ref_point < near.ref_point).all()

    def test_a_point_that_gives_no_value_for_an_objective_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="Missing") as caught:
            build(MAXIMIZE_BOTH, **self.explicit(f1=0.0, g=0.0))

        assert caught.value.field_path == "reference_point.values"
        assert caught.value.component == "acquisition 'qlog_nehvi_v1'"

    def test_a_point_with_a_value_for_an_outcome_that_is_not_an_objective_is_refused(
        self,
    ) -> None:
        with pytest.raises(ConfigError, match="Not objectives"):
            build(MAXIMIZE_BOTH, **self.explicit(f1=0.0, f2=0.0, g=0.0))


class TestOutcomeConstraints:
    """Roadmap P5-05, done when (b): the outcome constraints are passed through."""

    def test_each_constraint_of_the_goal_reaches_the_function(self) -> None:
        goal = {
            **MAXIMIZE_BOTH,
            "outcome_constraints": [
                {"outcome": G, "comparator": "<=", "bound": 0.5},
                {"outcome": G, "comparator": ">=", "bound": 0.1},
            ],
        }

        function = build(goal)

        assert len(function.constraints) == 2

    def test_a_goal_without_constraints_passes_none(self) -> None:
        assert build(MAXIMIZE_BOTH).constraints is None

    def test_the_constraint_reads_the_output_that_has_the_name_of_its_outcome(self) -> None:
        # Negative where the constraint is met, positive where it is violated, on the samples
        # of the three outputs (f1, f2, g).
        (violation,) = build(constrained("<=", 0.5)).constraints
        met = torch.tensor([[[9.0, 9.0, 0.4]]], dtype=torch.float64)
        violated = torch.tensor([[[-9.0, -9.0, 0.6]]], dtype=torch.float64)

        assert violation(met).item() < 0.0
        assert violation(violated).item() > 0.0

    def test_a_lower_bound_is_read_the_other_way_round(self) -> None:
        (violation,) = build(constrained(">=", 0.5)).constraints
        met = torch.tensor([[[0.0, 0.0, 0.6]]], dtype=torch.float64)
        violated = torch.tensor([[[0.0, 0.0, 0.4]]], dtype=torch.float64)

        assert violation(met).item() < 0.0
        assert violation(violated).item() > 0.0

    def test_the_constraint_follows_its_outcome_to_another_output(self) -> None:
        (violation,) = build(
            constrained("<=", 0.5), surrogate=makeParetoSurrogate((G, F2, F1))
        ).constraints
        met = torch.tensor([[[0.4, 9.0, 9.0]]], dtype=torch.float64)
        violated = torch.tensor([[[0.6, -9.0, -9.0]]], dtype=torch.float64)

        assert violation(met).item() < 0.0
        assert violation(violated).item() > 0.0

    def test_a_candidate_predicted_to_violate_a_constraint_scores_lower_than_a_feasible_one(
        self,
    ) -> None:
        # Without the constraint the right of the space is the front. With g = x <= 0.5 it is
        # infeasible, and the best place is the feasible point that is nearest to the front.
        free_feasible, free_violating = scores(MAXIMIZE_BOTH, 0.45, ON_THE_FRONT)
        feasible, violating = scores(constrained("<=", 0.5), 0.45, ON_THE_FRONT)

        assert free_violating > free_feasible + 10.0
        assert feasible > violating + 10.0

    def test_a_lower_bound_prefers_the_other_side(self) -> None:
        feasible, violating = scores(constrained(">=", 0.5), ON_THE_FRONT, 0.45)

        assert feasible > violating + 10.0

    def test_an_observed_point_that_violates_a_constraint_is_not_on_the_front(self) -> None:
        function = build(constrained("<=", 0.5))

        # Only one observed point is feasible and on the front, at x = 0.436.
        assert function.X_baseline.flatten().tolist() == pytest.approx([0.43571428571428567])

    def test_the_unit_of_the_constrained_outcome_does_not_matter(self) -> None:
        metre = scores(constrained("<=", 0.5), 0.45, ON_THE_FRONT)
        millimetre = scores(
            constrained("<=", 500.0),
            0.45,
            ON_THE_FRONT,
            surrogate=makeParetoSurrogate((F1, F2, G), constraint_scale=1000.0),
        )

        assert millimetre == pytest.approx(metre, rel=1e-6)

    def test_the_inferred_point_leaves_out_the_points_predicted_to_violate_a_constraint(
        self,
    ) -> None:
        free = build(MAXIMIZE_BOTH).ref_point
        constrained_point = build(constrained("<=", 0.5)).ref_point

        assert not torch.allclose(free, constrained_point)

    def test_a_constraint_on_an_outcome_the_surrogate_does_not_model_is_refused(self) -> None:
        with pytest.raises(AcquisitionContextError, match="'g'") as caught:
            build(constrained("<=", 0.5), surrogate=makeParetoSurrogate((F1, F2)))

        assert caught.value.field_path == "goal.outcome_constraints[0].outcome"

    def test_an_output_that_the_goal_only_constrains_is_not_an_objective(self) -> None:
        function = build(constrained("<=", 0.5))

        assert function.ref_point.shape == (2,)
        assert function.objective(torch.zeros(1, 1, 3, dtype=torch.float64)).shape[-1] == 2


class TestPendingPoints:
    def test_a_pending_point_changes_the_value(self) -> None:
        pending = candidates(ON_THE_FRONT).reshape(1, 1)

        without = scores(MAXIMIZE_BOTH, 0.6, ON_THE_FRONT, BEYOND)
        with_pending = scores(MAXIMIZE_BOTH, 0.6, ON_THE_FRONT, BEYOND, pending=pending)

        assert all(a != b for a, b in zip(without, with_pending, strict=True))

    def test_a_candidate_at_a_pending_point_is_worth_less_than_it_was(self) -> None:
        # The improvement is incremental over the front of the observed and the pending points,
        # so a point that is already pending has less to add.
        pending = candidates(ON_THE_FRONT).reshape(1, 1)

        (without,) = scores(MAXIMIZE_BOTH, ON_THE_FRONT)
        (at_pending,) = scores(MAXIMIZE_BOTH, ON_THE_FRONT, pending=pending)

        assert at_pending < without - 5.0

    def test_a_pending_point_does_not_lower_the_value_of_a_distant_candidate(self) -> None:
        pending = candidates(ON_THE_FRONT).reshape(1, 1)

        (without,) = scores(MAXIMIZE_BOTH, DOMINATED)
        (with_pending,) = scores(MAXIMIZE_BOTH, DOMINATED, pending=pending)

        assert with_pending == pytest.approx(without, abs=1.0)

    def test_the_pending_points_of_the_context_reach_the_function(self) -> None:
        pending = candidates(0.5, 0.6).reshape(2, 1)

        function = build(MAXIMIZE_BOTH, pending=pending)

        assert torch.equal(function.X_pending, pending)

    def test_no_pending_point_is_the_same_as_an_empty_set(self) -> None:
        empty = torch.empty((0, 1), dtype=candidates(0.5).dtype)

        assert scores(MAXIMIZE_BOTH, 0.6, ON_THE_FRONT, pending=empty) == scores(
            MAXIMIZE_BOTH, 0.6, ON_THE_FRONT
        )

    def test_a_pending_point_set_after_the_build_counts_as_one_of_the_context(
        self,
    ) -> None:
        pending = candidates(ON_THE_FRONT).reshape(1, 1)
        function = build(MAXIMIZE_BOTH)
        before = function(candidates(ON_THE_FRONT)).detach()

        function.set_X_pending(pending)
        during = function(candidates(ON_THE_FRONT)).detach()

        assert during.item() < before.item() - 5.0
        built_with = build(MAXIMIZE_BOTH, pending=pending)
        assert during.item() == pytest.approx(
            built_with(candidates(ON_THE_FRONT)).item(), abs=1e-6
        )

    def test_the_pending_points_of_a_function_are_not_taken_back_by_clearing_them(
        self,
    ) -> None:
        # The engine folds the pending points into the partitioning that it caches when they
        # are set, and clearing them does not undo that. A different set of pending points is a
        # new build, which is how a policy gets one (the pending points are in its context).
        function = build(MAXIMIZE_BOTH)
        before = function(candidates(ON_THE_FRONT)).detach()
        function.set_X_pending(candidates(ON_THE_FRONT).reshape(1, 1))
        during = function(candidates(ON_THE_FRONT)).detach()

        function.set_X_pending(None)
        after = function(candidates(ON_THE_FRONT)).detach()

        assert function.X_pending is None
        assert torch.equal(during, after)
        assert not torch.equal(before, after)
        fresh = build(MAXIMIZE_BOTH)
        assert fresh(candidates(ON_THE_FRONT)).item() == pytest.approx(before.item(), abs=1e-9)


class TestTheSampleCount:
    def test_the_function_samples_as_many_times_as_the_configuration_says(self) -> None:
        for n_samples in (16, 64, 256):
            function = build(MAXIMIZE_BOTH, n_samples=n_samples)

            assert function.sampler.sample_shape == torch.Size([n_samples])

    def test_the_default_is_the_default_of_the_configuration(self) -> None:
        assert build(MAXIMIZE_BOTH).sampler.sample_shape == torch.Size([128])

    def test_the_count_changes_the_estimate_and_not_its_meaning(self) -> None:
        places = (DOMINATED, ON_THE_FRONT)
        few = scores(MAXIMIZE_BOTH, *places, n_samples=16)
        many = scores(MAXIMIZE_BOTH, *places, n_samples=512)

        assert few != many
        assert many.index(max(many)) == few.index(max(few)) == 1

    def test_a_count_that_the_configuration_gives_through_a_mapping_reaches_the_function(
        self,
    ) -> None:
        resolved = resolveComponent(
            acquisitionRegistry(), {"type": "qlog_nehvi_v1", "n_samples": 32}
        )
        function = resolved.entry.builder(resolved.config).build(
            makeToyContext(makeParetoSurrogate((F1, F2, G)), MAXIMIZE_BOTH)
        )

        assert function.sampler.sample_shape == torch.Size([32])  # type: ignore[attr-defined]


class TestThePruningOfTheBaseline:
    def test_pruning_drops_the_observed_points_that_are_dominated(self) -> None:
        pruned = build(MAXIMIZE_BOTH, prune_baseline=True)
        kept = build(MAXIMIZE_BOTH, prune_baseline=False)

        assert kept.X_baseline.shape[0] == 8
        assert pruned.X_baseline.shape[0] < 8
        # The observed points at x < 0.4 are beaten in both objectives by the ones to their
        # right; the ones at x > 0.5 are the front.
        assert (pruned.X_baseline > 0.4).all()

    def test_pruning_is_the_one_that_the_function_is_told_not_to_repeat(self) -> None:
        # The engine does not prune again: the baseline it holds is the one that was kept.
        function = build(MAXIMIZE_BOTH, prune_baseline=True)

        assert (
            function._prune_baseline is False if hasattr(function, "_prune_baseline") else True
        )

    def test_the_value_hardly_depends_on_the_pruning(self) -> None:
        places = (DOMINATED, ON_THE_FRONT, BEYOND)

        pruned = scores(MAXIMIZE_BOTH, *places, prune_baseline=True)
        kept = scores(MAXIMIZE_BOTH, *places, prune_baseline=False)

        assert pruned == pytest.approx(kept, abs=1.0)


class TestTheNativeFunction:
    def test_it_scores_a_batch_of_candidates_of_q_points(self) -> None:
        function = build(MAXIMIZE_BOTH)
        batch = torch.stack(
            [candidates(0.2, 0.5).reshape(2, 1), candidates(0.6, 0.9).reshape(2, 1)]
        )

        values = function(batch)

        assert values.shape == (2,)

    def test_a_batch_of_two_equal_points_adds_nothing_to_a_single_one(self) -> None:
        function = build(MAXIMIZE_BOTH)
        single = function(candidates(ON_THE_FRONT)).detach()
        doubled = function(candidates(ON_THE_FRONT, ON_THE_FRONT).reshape(1, 2, 1)).detach()

        # One point twice improves the front no more than once does.
        assert doubled.item() == pytest.approx(single.item(), abs=0.5)

    def test_two_places_that_trade_off_are_worth_more_together_than_alone(self) -> None:
        function = build(MAXIMIZE_BOTH)
        first = function(candidates(0.6)).detach().exp()
        second = function(candidates(BEYOND)).detach().exp()
        together = function(candidates(0.6, BEYOND).reshape(1, 2, 1)).detach().exp()

        assert together.item() > max(first.item(), second.item())

    def test_it_is_differentiable_in_the_candidate(self) -> None:
        function = build(MAXIMIZE_BOTH)
        candidate = candidates(0.4).clone().requires_grad_(True)

        function(candidate).sum().backward()

        assert candidate.grad is not None
        assert torch.isfinite(candidate.grad).all()
        assert candidate.grad.abs().sum() > 0.0
        # The front is to the right of 0.4, so the value rises with x there.
        assert candidate.grad.item() > 0.0

    def test_the_value_is_finite_far_from_any_improvement(self) -> None:
        # A log, so that it stays finite where the improvement is almost zero.
        values = scores(MAXIMIZE_BOTH, 0.0, 0.05, 0.2)

        assert all(torch.isfinite(torch.tensor(value)) for value in values)

    def test_the_model_is_not_changed_by_building_or_scoring(self) -> None:
        surrogate = makeParetoSurrogate((F1, F2, G))
        before = [parameter.detach().clone() for parameter in surrogate.model.parameters()]

        scores(constrained("<=", 0.5), 0.45, ON_THE_FRONT, surrogate=surrogate)

        after = list(surrogate.model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))

    def test_three_objectives_are_scored_too(self) -> None:
        surrogate = makeSurrogateOf(
            {
                "a": lambda x: x,
                "b": lambda x: torch.sin(3.0 * x),
                "c": lambda x: torch.cos(5.0 * x),
            }
        )
        goal = {"outcomes": {"a": "maximize", "b": "maximize", "c": "minimize"}}

        values = scores(goal, 0.1, 0.5, 0.9, surrogate=surrogate)

        assert all(torch.isfinite(torch.tensor(value)) for value in values)


class TestTheSamplesAreDeterministic:
    def test_the_same_seed_gives_the_same_values(self) -> None:
        assert scores(MAXIMIZE_BOTH, 0.6, BEYOND, seed=3) == scores(
            MAXIMIZE_BOTH, 0.6, BEYOND, seed=3
        )

    def test_another_seed_gives_other_samples(self) -> None:
        assert scores(MAXIMIZE_BOTH, 0.6, BEYOND, seed=3) != scores(
            MAXIMIZE_BOTH, 0.6, BEYOND, seed=4
        )

    def test_the_samples_are_the_same_for_every_call_of_one_function(self) -> None:
        function = build(MAXIMIZE_BOTH)

        assert torch.equal(
            function(candidates(0.6, BEYOND)), function(candidates(0.6, BEYOND))
        )

    def test_the_pruning_of_the_baseline_is_seeded_too_and_leaves_the_global_generator_alone(
        self,
    ) -> None:
        # The engine prunes with the global generator: left to it, builds from one seed would
        # keep different points. Many builds, each after another global seed, all the same.
        surrogate = makeParetoSurrogate((F1, F2, G))
        builder = QLogNEHVIBuilder(QLogNEHVIConfig(n_samples=8))

        results = []
        for global_seed in range(12):
            torch.manual_seed(global_seed)
            before = torch.get_rng_state()
            function = builder.build(
                AcquisitionBuildContext(
                    surrogate=surrogate,
                    goal=OptimizeGoal(OptimizeGoalConfig(**MAXIMIZE_BOTH)),
                    generator=torchGenerator(7, RuntimeConfig()),
                )
            )
            assert torch.equal(before, torch.get_rng_state())
            results.append((function.X_baseline, function(candidates(0.6, BEYOND)).detach()))

        first_baseline, first_values = results[0]
        for baseline, values in results[1:]:
            assert torch.equal(baseline, first_baseline)
            assert torch.equal(values, first_values)

    def test_building_advances_the_generator_of_the_context(self) -> None:
        context = AcquisitionBuildContext(
            surrogate=makeParetoSurrogate((F1, F2, G)),
            goal=OptimizeGoal(OptimizeGoalConfig(**MAXIMIZE_BOTH)),
            generator=torchGenerator(7, RuntimeConfig()),
        )
        builder = QLogNEHVIBuilder(QLogNEHVIConfig())

        first = builder.build(context)(candidates(0.6, BEYOND)).detach()
        second = builder.build(context)(candidates(0.6, BEYOND)).detach()

        assert not torch.equal(first, second)


class TestWhatTheBuilderRefuses:
    def test_a_goal_that_is_not_an_optimize_goal_is_refused(self) -> None:
        context = AcquisitionBuildContext(
            surrogate=makeParetoSurrogate((F1, F2, G)),
            goal=LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=F1)),
            generator=torchGenerator(7, RuntimeConfig()),
        )

        with pytest.raises(ConfigError, match="not an optimize goal") as caught:
            QLogNEHVIBuilder(QLogNEHVIConfig()).build(context)

        assert caught.value.field_path == "goal"

    def test_a_goal_with_weights_is_refused(self) -> None:
        goal = {**MAXIMIZE_BOTH, "weights": {F1: 1.0, F2: 3.0}}

        with pytest.raises(ConfigError, match="weights") as caught:
            build(goal)

        assert caught.value.field_path == "goal.weights"

    def test_a_goal_with_one_outcome_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="at least two objectives") as caught:
            build({"outcomes": {F1: "maximize"}})

        assert caught.value.field_path == "goal.outcomes"

    def test_an_objective_that_the_surrogate_does_not_model_is_refused(self) -> None:
        with pytest.raises(AcquisitionContextError, match="'f2'") as caught:
            build(MAXIMIZE_BOTH, surrogate=makeParetoSurrogate((F1, G)))

        assert caught.value.field_path == "goal.outcomes.f2"

    def test_a_model_with_no_observed_point_is_refused(self) -> None:
        surrogate = makeParetoSurrogate((F1, F2, G))
        surrogate.model.train_inputs = (torch.empty((3, 0, 1), dtype=torch.float64),)

        with pytest.raises(AcquisitionContextError, match="no observed point"):
            build(MAXIMIZE_BOTH, surrogate=surrogate)
