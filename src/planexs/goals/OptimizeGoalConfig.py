"""The configuration schema of the `optimize_v1` goal."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator

from planexs.core.ComponentConfig import ComponentConfig
from planexs.domain.outcome_kinds.outcomeKindRegistry import outcomeKindRegistry
from planexs.goals.OutcomeConstraint import OutcomeConstraint

#: What a caller wants of one outcome: its smallest or its largest value.
Direction = Literal["minimize", "maximize"]

#: A scalarizing weight: a finite number above zero. The direction of an outcome is given by
#: `OptimizeGoalConfig.outcomes` and nowhere else, so a weight only says how much an outcome
#: counts, never which way.
Weight = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]


class OptimizeGoalConfig(ComponentConfig):
    """The configuration of the `optimize_v1` goal: optimize one or several outcomes
    (specification 5.7, FR2, FR15).

    The goal states what is wanted and computes nothing (ADR-0013). Four settings say it:

    - `outcomes` names the outcomes to optimize and gives each a direction. One outcome is the
      plain case, several without `weights` are a Pareto problem, and several with `weights`
      are one scalar preference (a weighted sum of the direction-signed outcomes);
    - `weights` is the scalarization, with one weight for each outcome of `outcomes`;
    - `outcome_constraints` limits what other outcomes, or these, may measure;
    - `outcome_kinds` says which kinds of outcome the goal can optimize or constrain.

    This schema holds the stored form of what the legacy layer spread over `LegacyObjective`,
    `LegacyMetric` and `LegacyOutcomeConstraint`. Two things differ, on purpose. The weights
    are keyed by outcome name where `LegacyObjective` listed them in the order of its metrics,
    so that a weight can never be applied to the wrong outcome (the surrogate's outcomes follow
    the order of the problem, and the goal's own order is only the order it was written in).
    And a constraint may name any outcome of the problem, where a `LegacyOutcomeConstraint`
    had to name a metric of its own objective, which could not state a limit on an outcome
    that is measured but not optimized.

    What a schema can check alone is checked here: at least one outcome, a direction that is
    one of two, weights that match the outcomes, finite constraint bounds and registered
    outcome kinds. That the outcomes exist in a problem, and that they are of an accepted
    kind, is checked against the problem by `OptimizeGoal.validateAgainst`.

    Attributes:
        type: The registered name, always `"optimize_v1"`.
        outcomes: Maps the name of each outcome to optimize to its direction.
        weights: Optionally maps each outcome of `outcomes` to a scalarizing weight.
        outcome_constraints: The limits on observed outcomes, empty if there are none.
        outcome_kinds: The registered names of the outcome kinds the goal accepts.
    """

    type: Literal["optimize_v1"] = Field(
        default="optimize_v1",
        description=(
            "The registered name of this goal, always 'optimize_v1'. It selects the "
            "'optimize_v1' entry of the 'goal' registry, so that a stored goal says which one "
            "it is (specification 5.7). It is the only value the field accepts."
        ),
    )
    outcomes: dict[str, Direction] = Field(
        min_length=1,
        description=(
            "Maps the name of each outcome to optimize, as the problem names it, to its "
            "direction: 'minimize' or 'maximize'. At least one outcome is required. One "
            "outcome is a single-objective goal. Several outcomes are a multi-objective goal "
            "with no preference between them (a Pareto problem), unless 'weights' is given, "
            "which turns them into one scalar preference. The order of the entries is only "
            "the order they were written in: nothing depends on it. There is no default: only "
            "the caller knows what is wanted."
        ),
    )
    weights: dict[str, Weight] | None = Field(
        default=None,
        description=(
            "Optionally maps each outcome of 'outcomes' to a scalarizing weight, a finite "
            "number above zero without a unit, so that several outcomes are optimized as one: "
            "the weighted sum of the outcomes, each taken in its own direction. It needs "
            "exactly one weight for each outcome of 'outcomes', no more and no fewer. A "
            "weight says how much an outcome counts, never which way: the direction is "
            "'outcomes', and a negative weight would silently turn an outcome around, so it "
            "is refused. It is a scalar preference between objectives and does not control "
            "exploration versus exploitation, which belongs to the acquisition. The default "
            "is None: several outcomes then have no preference between them, and one outcome "
            "needs no weight."
        ),
    )
    outcome_constraints: list[OutcomeConstraint] = Field(
        default_factory=list,
        description=(
            "The limits that observed outcomes must respect, each an outcome name, a "
            "comparator ('<=' or '>=') and a bound in the unit of the outcome (FR15). A "
            "constrained outcome need not be one of 'outcomes': a throughput that must stay "
            "above a floor while a yield is maximized is the usual case. Bounds relative to a "
            "baseline trial are not supported. A goal with at least one constraint needs an "
            "acquisition that can enforce it (the tag 'outcome_constraints:supported'). The "
            "default is an empty list: no outcome is limited."
        ),
    )
    outcome_kinds: tuple[str, ...] = Field(
        default=("continuous",),
        min_length=1,
        description=(
            "The registered names of the outcome kinds that this goal accepts, for the "
            "outcomes it optimizes and for those it constrains: an objective has a direction "
            "and a constraint a bound, and both compare values, so the kind must have "
            "ordered numeric values. Each name must be registered in the 'outcome_kind' "
            "registry. At least one is required. The default is ('continuous',): the only "
            "built-in kind whose values are measured numbers. A binary label has no "
            "direction, and is refused. A kind that a plugin registers (a count, an ordinal "
            "grade) is accepted by adding its name here, with no change to planexs."
        ),
    )

    @field_validator("weights")
    @classmethod
    def _checkWeightsMatchOutcomes(
        cls, weights: dict[str, float] | None, info: ValidationInfo
    ) -> dict[str, float] | None:
        """Checks that there is exactly one weight for each outcome.

        Args:
            weights: The weights, or `None`.
            info: Pydantic's view of the fields validated so far, which holds `outcomes`.

        Returns:
            The weights, unchanged.

        Raises:
            ValueError: if an outcome has no weight, or a weight names something that is not
                one of the outcomes. The message names the field and says how to fix it.
        """
        outcomes = info.data.get("outcomes")
        if weights is None or outcomes is None:
            # `outcomes` failed its own validation when it is missing here: that error is
            # reported, and a second one about the weights would only repeat it.
            return weights
        missing = [name for name in outcomes if name not in weights]
        unknown = [name for name in weights if name not in outcomes]
        if missing or unknown:
            problems: list[str] = []
            if missing:
                problems.append(f"no weight for {missing!r}")
            if unknown:
                problems.append(
                    f"a weight for {unknown!r}, which is not an outcome of the goal"
                )
            raise ValueError(
                f"weights has {len(weights)} weight(s) for {len(outcomes)} outcome(s) "
                f"{list(outcomes)!r}: {' and '.join(problems)}. Give exactly one weight for "
                "each outcome, or leave weights out"
            )
        return weights

    @field_validator("outcome_kinds")
    @classmethod
    def _checkKindsAreRegistered(cls, kinds: tuple[str, ...]) -> tuple[str, ...]:
        """Checks that every kind named is registered in the `outcome_kind` registry.

        Args:
            kinds: The names of the kinds.

        Returns:
            The names, unchanged.

        Raises:
            ValueError: if a name is not registered. The message names the field and lists
                the registered kinds, so that a typo shows.
        """
        registry = outcomeKindRegistry()
        unknown = [kind for kind in kinds if kind not in registry]
        if unknown:
            raise ValueError(
                f"outcome_kinds names {unknown!r}, which is not a registered outcome kind. "
                f"Registered kinds: {', '.join(registry.names()) or 'none'}. Check the "
                "spelling, or load the plugin that registers the kind"
            )
        return kinds
