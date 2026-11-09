"""The policy values of the search-space encoder, as a configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: The named rules that say what an unset `is_ordered` of a `Choice` means (see
#: `EncoderConfig.unset_choice_ordering`). A rule is a name selected in the configuration, so
#: that the choice is reachable by the caller and stored with the study (specification 3.4).
ChoiceOrderingRule = Literal["numeric_over_two_values", "numeric", "never"]


class EncoderConfig(BaseModel):
    """The values that `SearchSpaceEncoder` takes decisions with (specification 3.4, 10.8).

    Four settings that change how a search space is laid out and rounded, and that a caller
    could reasonably want to change: which choices count as ordered when the parameter does
    not say, how many rounding combinations are listed, how forgiving the grid arithmetic is,
    and how long the random subset of categorical assignments is drawn for. They are the
    configuration of the encoder, not of a registered component: this class has no `type`
    field, is not registered in any kind and is not a `ComponentConfig`, like
    `RuntimeConfig`. A study-level model holds it as one of its fields, and `validateConfig`
    reads it like any other model, with the same field paths in its errors (for example
    `encoder.grid_tolerance`).

    The defaults are the values the encoder used before they became settings, so that a study
    that sets nothing encodes exactly as it did. They are written in the fields below and
    nowhere else. A study stores its encoder settings in its resolved configuration, so a later
    change of a default never alters how an old study encodes.

    The only numbers that stay in the code of the encoder are structural: the midpoint of the
    unit cube, where a 0/1 column is cut and where an unknown conditional parameter is
    encoded (specification 3.4).

    A configuration is frozen: it is shared by everything that holds the encoder and must not
    change under them. Build a new one to change a setting.

    Attributes:
        unset_choice_ordering: What an unset `is_ordered` of a `Choice` means.
        max_ordinal_columns: The largest number of grid or ordered columns whose rounding
            combinations are all listed.
        grid_tolerance: The tolerance, in grid steps, when counting the grid points of a range.
        categorical_subset_attempts_factor: The random draws allowed per requested categorical
            assignment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unset_choice_ordering: ChoiceOrderingRule = Field(
        default="numeric_over_two_values",
        description=(
            "The named rule that decides whether a `Choice` whose `is_ordered` is unset "
            "(None) is encoded as ordered (one rank column) or as categorical (a 0/1 column"
            " for two values, a one-hot block otherwise). A choice that sets `is_ordered` "
            "to True or False is never affected: the rule only fills a gap. The rules are "
            "'numeric_over_two_values' (ordered when every value is a number, a boolean not"
            " being one, and there are more than two of them), 'numeric' (ordered when "
            "every value is a number, whatever their count) and 'never' (an unset flag "
            "always means categorical). The default is 'numeric_over_two_values': a choice "
            "among three or more numbers (batch sizes, layer counts) is almost always "
            "ordered, while two values read as a switch, whose rank column and 0/1 column "
            "would not differ in meaning."
        ),
    )
    max_ordinal_columns: int = Field(
        default=8,
        ge=0,
        description=(
            "The largest number of grid or ordered columns for which "
            "`SearchSpaceEncoder.roundingNeighbors` lists every combination of rounding "
            "down and rounding up, in a number of columns. A point with n such columns has "
            "up to 2 ** n neighbours, each of which the caller checks against the "
            "constraints; with more columns than this only the nearest rounding is listed, "
            "and 0 always lists only that one. The default is 8: at most 256 rows is cheap "
            "to check and covers the usual number of integer parameters in a search space, "
            "and every extra column doubles the cost."
        ),
    )
    grid_tolerance: float = Field(
        default=1e-9,
        ge=0.0,
        lt=0.5,
        description=(
            "The tolerance, in grid steps, with which the grid points of a range with a "
            "step are counted: the quotient (upper - lower) / step is raised by this much "
            "before it is rounded down, so that a quotient that floating-point arithmetic "
            "leaves just under a whole number (0.3 / 0.1 is 2.9999999999999996) still "
            "counts the last grid point. 0 turns the tolerance off. It must stay below 0.5,"
            " so that a bound is never extended by half a step or more. The default is "
            "1e-9: far above the rounding error of a double-precision quotient for a grid "
            "of up to about a hundred thousand steps (about 1e-11), and far below any "
            "fraction of a step that a caller would mean."
        ),
    )
    categorical_subset_attempts_factor: int = Field(
        default=20,
        ge=1,
        description=(
            "How many random draws `SearchSpaceEncoder.categoricalFixedFeatures` may make "
            "for each assignment it is asked for, in draws per requested assignment, when "
            "the unordered choices have more combinations than that and a random subset of "
            "them is drawn. A draw that repeats an earlier one is discarded, and the "
            "drawing stops after this many draws per requested assignment even if fewer "
            "distinct ones were found, so a small value can return a smaller subset. The "
            "default is 20: a wide margin over the ln(N) draws per assignment that "
            "collecting almost all of N assignments needs, for any space that can be "
            "enumerated at all."
        ),
    )
