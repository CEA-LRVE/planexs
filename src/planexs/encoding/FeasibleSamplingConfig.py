"""The policy values of the feasible-point sampler and of the feasibility check."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FeasibleSamplingConfig(BaseModel):
    """The values that `sampleFeasibleEncoded` and `EncodedConstraints` decide with
    (specification 3.4, 10.8).

    Six settings that change which points the sampler returns, how long it looks for them and
    when a point counts as satisfying a constraint: how many rounds of candidates it draws, how
    many candidates the first round draws (an oversampling factor and a minimum), the two
    settings of the hit-and-run chain that draws points inside linear constraints, and the
    tolerance of the feasibility check. They are the configuration of the sampler, not of a
    registered component: this class has no `type` field, is not registered in any kind
    and is not a `ComponentConfig`, like `RuntimeConfig` and `EncoderConfig`. A study-level
    model holds it as one of its fields, and `validateConfig` reads it like any other model,
    with the same field paths in its errors (for example `sampling.polytope_burn_in`).

    The defaults are the values that the sampler and the constraints used before they became
    settings, so that a study that sets nothing samples exactly as it did. They are written in
    the fields below and nowhere else, and so is the reasoning that chose the two settings of
    the chain. A study stores its sampling settings in its resolved configuration, so a later
    change of a default never alters how an old study samples.

    A configuration is frozen: it is shared by the sampler and the constraints and must not
    change under them. Build a new one to change a setting.

    Attributes:
        max_rounds: The largest number of rounds of candidates.
        oversampling_factor: The candidates the first round draws per requested point.
        min_draw: The fewest candidates the first round draws.
        polytope_burn_in: The steps of the hit-and-run chain discarded before the first point.
        polytope_thinning: The steps of the hit-and-run chain between two kept points.
        feasibility_tolerance: The largest violation of a constraint that still counts as
            meeting it, relative to the constraint's own scale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_rounds: int = Field(
        default=25,
        ge=1,
        description=(
            "The largest number of rounds of candidates that `sampleFeasibleEncoded` draws "
            "while it looks for points that satisfy the constraints, in rounds. Round k, "
            "counted from 1, draws k times as many candidates as the first round (see "
            "`oversampling_factor` and `min_draw`), so a feasible region that the first "
            "round misses costs more draws and not a failure. The search stops after this "
            "many rounds even if fewer points than asked for were found, and the ones found"
            " are then repeated to fill the request; it raises only when none was found. "
            "The default is 25: the rounds together draw 325 times the candidates of the "
            "first (20800 for a first round of 64), which finds a feasible region about 300"
            " times smaller than the first round alone would, and still gives up quickly on"
            " an empty one."
        ),
    )

    oversampling_factor: int = Field(
        default=8,
        ge=1,
        description=(
            "How many candidates the first round draws for each point that is asked for, in"
            " candidates per requested point. Some candidates violate a nonlinear "
            "constraint, or miss a constraint once they are rounded onto a grid, and are "
            "thrown away, so more are drawn than are wanted. The first round draws the "
            "larger of this factor times the number of points asked for and `min_draw`. The"
            " default is 8: it expects at least one candidate in eight to be feasible, and "
            "a smaller feasible share is found by the later rounds, which draw more."
        ),
    )

    min_draw: int = Field(
        default=64,
        ge=1,
        description=(
            "The fewest candidates that the first round draws, whatever the number of "
            "points asked for, in candidates. It gives a request for one or two points a "
            "fair chance of meeting a constraint that few candidates meet: the first round "
            "draws the larger of `oversampling_factor` times the number of points asked for"
            " and this number. The default is 64: a power of two, which keeps a scrambled "
            "Sobol sequence balanced, and enough that a feasible share of 1 in 64 is "
            "expected to give a point in the first round."
        ),
    )

    polytope_burn_in: int = Field(
        default=200,
        ge=0,
        description=(
            "How many steps of the hit-and-run Markov chain are discarded before the first "
            "kept point, in steps of the chain. The chain draws points inside the polytope "
            "that the linear constraints define, so that an equality such as a mixture that"
            " sums to one is met exactly, which rejection sampling could never do; it is "
            "used only when the search space has linear constraints. It starts at an "
            "arbitrary interior point, which it must forget before its points are kept. "
            "BoTorch's own default for the starting points of `optimize_acqf` is 10000, "
            "which is much more expensive. The default is 200, chosen from a measurement "
            "and not from theory: on a simplex, whose exact law is known, 200 and 10000 "
            "gave indistinguishable means and variances in 3 and 10 dimensions, so the "
            "cheaper value is used."
        ),
    )

    polytope_thinning: int = Field(
        default=10,
        ge=1,
        description=(
            "How many steps of the hit-and-run Markov chain are made between two kept "
            "points, in steps of the chain: consecutive states of the chain are correlated,"
            " and thinning spaces them out. It is used only when the search space has "
            "linear constraints, and it is the expensive setting: on an equality-"
            "constrained mixed problem, 32 (BoTorch's own default for the starting points "
            "of `optimize_acqf`) instead of 10 made suggesting 3 points 2.7 times slower. "
            "The default is 10, chosen from that measurement. Its price is more correlated "
            "samples in high dimension (lag-1 autocorrelation 0.76 in 10 dimensions, "
            "against 0.43 with 32), which was judged acceptable because the samples only "
            "seed the optimizer's restarts and normalize the acquisition terms. Raise it "
            "for a high-dimensional polytope."
        ),
    )

    feasibility_tolerance: float = Field(
        default=1e-6,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "The largest violation of a constraint that still counts as satisfying it, as a"
            " fraction of that constraint's own scale (a number without a unit): "
            "`EncodedConstraints.violation` measures each violation relative to its "
            "constraint, so a single tolerance suits all of them. It is what "
            "`EncodedConstraints.isFeasible` accepts when it is not given another, and what"
            " the sampler checks its candidates with. 0 accepts only points that meet every"
            " constraint exactly. The default is 1e-6: about eight times the machine "
            "epsilon of single precision, so that a float32 study does not reject points "
            "whose only fault is round-off, and far below any violation that matters."
        ),
    )
