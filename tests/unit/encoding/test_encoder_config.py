"""Unit tests for `EncoderConfig` and for what it changes in `SearchSpaceEncoder`.

Specification 3.4 and 10.8, roadmap task P3-03. Three groups: the schema (defaults, validation,
the stored form), one test per configured value showing that changing it changes what the
encoder does while the default reproduces what it did before, and the check that no numeric
literal other than a structural one is left in the encoder's files.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, get_args

import pytest
import torch
from pydantic import BaseModel, Field, ValidationError

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.domain.parameters.Categorical import Categorical
from planexs.domain.parameters.Choice import Choice
from planexs.domain.parameters.Integer import Integer
from planexs.domain.parameters.Real import Real
from planexs.domain.SearchSpace import SearchSpace
from planexs.encoding.EncoderConfig import ChoiceOrderingRule, EncoderConfig
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder, _unsetMeansOrdered

REPO_ROOT = Path(__file__).resolve().parents[3]
ENCODING_PACKAGE = REPO_ROOT / "src" / "planexs" / "encoding"

#: The values that the encoder used before they became settings: the rule "numeric with more
#: than two values" of `_encodeChoice`, `max_ordinal=8` of `roundingNeighbors`, `1e-9` in
#: `_snapRaw` and `_floorAndCeil`, and `20 * max_combinations` in `categoricalFixedFeatures`.
#: The defaults of the schema must reproduce them, so a study that sets nothing encodes as it
#: always did.
OLD_BEHAVIOUR: dict[str, Any] = {
    "unset_choice_ordering": "numeric_over_two_values",
    "max_ordinal_columns": 8,
    "grid_tolerance": 1e-9,
    "categorical_subset_attempts_factor": 20,
}

#: How each default is spelled in the description of its field, which has to say what the
#: default is (specification 10.4).
DEFAULT_SPELLINGS: dict[str, str] = {
    "unset_choice_ordering": "the default is 'numeric_over_two_values'",
    "max_ordinal_columns": "the default is 8",
    "grid_tolerance": "the default is 1e-9",
    "categorical_subset_attempts_factor": "the default is 20",
}

MIDPOINT_MARKER = "# structural: unit-cube midpoint"


def failureOf(data: Any) -> ConfigError:
    """The `ConfigError` that validating `data` as an `EncoderConfig` raises."""
    with pytest.raises(ConfigError) as caught:
        validateConfig(EncoderConfig, data)
    return caught.value


class StudyLikeConfig(BaseModel):
    """A study-level model that holds the encoder settings as one of its fields."""

    name: str = Field(default="study", description="The study's name.")
    encoder: EncoderConfig = Field(
        default_factory=EncoderConfig, description="The encoder settings of the study."
    )


class TestDefaults:
    def test_a_bare_configuration_reproduces_the_old_behaviour(self) -> None:
        assert EncoderConfig().model_dump() == OLD_BEHAVIOUR

    def test_the_defaults_are_written_in_the_fields_of_the_schema(self) -> None:
        # Specification 3.4: a default is written in exactly one place, the schema field.
        written = {name: field.default for name, field in EncoderConfig.model_fields.items()}
        assert written == OLD_BEHAVIOUR

    def test_an_empty_mapping_validates_to_the_defaults(self) -> None:
        assert validateConfig(EncoderConfig, {}) == EncoderConfig()

    def test_setting_one_field_leaves_the_others_at_their_defaults(self) -> None:
        config = EncoderConfig(grid_tolerance=0.0)
        assert config.model_dump() == {**OLD_BEHAVIOUR, "grid_tolerance": 0.0}

    def test_an_encoder_built_with_no_configuration_uses_the_defaults(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Real("x", 0.0, 1.0)]))
        assert encoder.config == EncoderConfig()

    def test_an_encoder_keeps_the_configuration_it_was_given(self) -> None:
        config = EncoderConfig(max_ordinal_columns=3)
        encoder = SearchSpaceEncoder(SearchSpace([Real("x", 0.0, 1.0)]), config=config)
        assert encoder.config is config


class TestTheSchema:
    def test_it_has_exactly_the_four_settings_of_the_card(self) -> None:
        assert list(EncoderConfig.model_fields) == list(OLD_BEHAVIOUR)

    def test_every_field_has_a_description(self) -> None:
        # Specification 10.8: a description on every field.
        for name, field in EncoderConfig.model_fields.items():
            assert field.description, f"field {name!r} has no description"
            assert field.description.strip() == field.description, name

    def test_every_description_says_what_the_default_is(self) -> None:
        # Specification 10.4: a description states why the default is what it is. Whether the
        # reason is a good one is for review; that the default is named can be checked.
        for name, field in EncoderConfig.model_fields.items():
            assert DEFAULT_SPELLINGS[name] in (field.description or "").lower(), name

    def test_it_is_a_model_of_its_own_and_not_a_component_configuration(self) -> None:
        # Like `RuntimeConfig`: no `type` field, registered in no kind.
        assert issubclass(EncoderConfig, BaseModel)
        assert not issubclass(EncoderConfig, ComponentConfig)
        assert "type" not in EncoderConfig.model_fields

    def test_a_configuration_is_frozen(self) -> None:
        config = EncoderConfig()
        with pytest.raises(ValidationError):
            config.grid_tolerance = 0.1  # type: ignore[misc]

    def test_a_misspelled_setting_is_an_error_with_its_path(self) -> None:
        error = failureOf({"grid_tolerence": 1e-6})
        assert error.field_path == "grid_tolerence"
        assert "Unknown field" in error.message

    def test_the_suggestion_lists_the_fields_of_the_schema(self) -> None:
        suggestion = failureOf({"tolerance": 1e-6}).suggestion
        assert suggestion is not None
        for name in OLD_BEHAVIOUR:
            assert name in suggestion

    def test_it_is_a_field_of_a_study_level_model_with_the_full_path_in_errors(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(StudyLikeConfig, {"encoder": {"grid_tolerance": -1.0}})
        assert caught.value.field_path == "encoder.grid_tolerance"

    def test_a_study_level_model_without_the_field_gets_the_defaults(self) -> None:
        assert validateConfig(StudyLikeConfig, {}).encoder == EncoderConfig()


class TestValidation:
    @pytest.mark.parametrize("name", ["numeric_over_two_values", "numeric", "never"])
    def test_the_three_named_rules_are_accepted(self, name: str) -> None:
        assert validateConfig(EncoderConfig, {"unset_choice_ordering": name}) == EncoderConfig(
            unset_choice_ordering=name  # type: ignore[arg-type]
        )

    @pytest.mark.parametrize(
        "bad", ["sometimes", "", "Numeric", "numeric ", "always", None, 3, ["never"]], ids=repr
    )
    def test_an_unknown_rule_is_rejected_with_the_field_path(self, bad: Any) -> None:
        error = failureOf({"unset_choice_ordering": bad})
        assert error.field_path == "unset_choice_ordering"
        # The message lists the rules that exist, which is where a typo shows.
        for rule in ("numeric_over_two_values", "numeric", "never"):
            assert repr(rule) in error.message

    @pytest.mark.parametrize("value", [0, 1, 8, 20])
    def test_a_count_of_columns_of_zero_or_more_is_accepted(self, value: int) -> None:
        assert EncoderConfig(max_ordinal_columns=value).max_ordinal_columns == value

    @pytest.mark.parametrize("bad", [-1, 1.5, "eight", None, [8]], ids=repr)
    def test_a_bad_count_of_columns_is_rejected_with_the_field_path(self, bad: Any) -> None:
        assert failureOf({"max_ordinal_columns": bad}).field_path == "max_ordinal_columns"

    @pytest.mark.parametrize("value", [0.0, 1e-12, 1e-9, 0.01, 0.49])
    def test_a_tolerance_below_half_a_step_is_accepted(self, value: float) -> None:
        assert EncoderConfig(grid_tolerance=value).grid_tolerance == value

    @pytest.mark.parametrize(
        "bad", [-1e-9, 0.5, 1.0, float("nan"), float("inf"), "tiny", None], ids=repr
    )
    def test_a_bad_tolerance_is_rejected_with_the_field_path(self, bad: Any) -> None:
        assert failureOf({"grid_tolerance": bad}).field_path == "grid_tolerance"

    @pytest.mark.parametrize("value", [1, 20, 1000])
    def test_a_factor_of_one_or_more_is_accepted(self, value: int) -> None:
        config = EncoderConfig(categorical_subset_attempts_factor=value)
        assert config.categorical_subset_attempts_factor == value

    @pytest.mark.parametrize("bad", [0, -5, 2.5, "many", None], ids=repr)
    def test_a_bad_factor_is_rejected_with_the_field_path(self, bad: Any) -> None:
        error = failureOf({"categorical_subset_attempts_factor": bad})
        assert error.field_path == "categorical_subset_attempts_factor"

    def test_several_problems_are_reported_together(self) -> None:
        error = failureOf({"grid_tolerance": -1.0, "max_ordinal_columns": -1, "bogus": 1})
        for name in ("grid_tolerance", "max_ordinal_columns", "bogus"):
            assert name in error.message
        # `field_path` is the path of the first problem, in the order the model declares.
        assert error.field_path == "max_ordinal_columns"


class TestTheStoredForm:
    """A study stores its resolved configuration (specification 3.4, 5.11)."""

    def test_the_resolved_configuration_writes_every_field_out(self) -> None:
        assert toResolvedConfig(EncoderConfig()) == OLD_BEHAVIOUR

    def test_a_changed_default_in_the_code_could_not_alter_a_stored_configuration(
        self,
    ) -> None:
        stored = toResolvedConfig(EncoderConfig(max_ordinal_columns=3))
        assert stored == {**OLD_BEHAVIOUR, "max_ordinal_columns": 3}
        assert validateConfig(EncoderConfig, stored).max_ordinal_columns == 3

    def test_resolving_then_validating_then_resolving_gives_the_same_dictionary(self) -> None:
        config = EncoderConfig(
            unset_choice_ordering="never",
            max_ordinal_columns=3,
            grid_tolerance=1e-6,
            categorical_subset_attempts_factor=5,
        )
        resolved = toResolvedConfig(config)
        assert toResolvedConfig(validateConfig(EncoderConfig, resolved)) == resolved

    def test_the_resolved_dictionary_survives_json_unchanged(self) -> None:
        resolved = toResolvedConfig(EncoderConfig(grid_tolerance=1e-6))
        assert json.loads(json.dumps(resolved)) == resolved

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_a_configuration_written_to_a_file_format_and_read_back_is_equal(
        self, file_format: Any
    ) -> None:
        # Specification 10.8: Python, JSON and YAML have the same meaning.
        config = EncoderConfig(unset_choice_ordering="numeric", grid_tolerance=1e-6)
        text = dumpConfig(toResolvedConfig(config), file_format)
        assert validateConfig(EncoderConfig, parseConfig(text, file_format)) == config


class TestUnsetChoiceOrdering:
    @staticmethod
    def _kinds(rule: Any) -> dict[str, str]:
        """The encoding kind of each of four choices whose `is_ordered` is unset."""
        space = SearchSpace(
            [
                Choice("three", [1, 2, 3], "int"),
                Choice("two", [1, 2], "int"),
                Choice("text", ["a", "b", "c"], "str"),
                Choice("flag", [True, False], "bool"),
            ]
        )
        encoder = SearchSpaceEncoder(space, config=EncoderConfig(unset_choice_ordering=rule))
        return {encoding.name: encoding.kind for encoding in encoder.encodings}

    def test_the_default_rule_is_ordered_for_more_than_two_numeric_values_only(self) -> None:
        assert self._kinds("numeric_over_two_values") == {
            "three": "ordered",
            "two": "categorical",
            "text": "categorical",
            "flag": "categorical",
        }

    def test_no_configuration_is_the_default_rule(self) -> None:
        space = SearchSpace([Choice("three", [1, 2, 3], "int"), Choice("two", [1, 2], "int")])
        kinds = {e.name: e.kind for e in SearchSpaceEncoder(space).encodings}
        assert kinds == {"three": "ordered", "two": "categorical"}

    def test_the_numeric_rule_also_orders_two_numeric_values_but_not_text_or_booleans(
        self,
    ) -> None:
        assert self._kinds("numeric") == {
            "three": "ordered",
            "two": "ordered",
            "text": "categorical",
            "flag": "categorical",
        }

    def test_the_never_rule_leaves_every_unset_choice_categorical(self) -> None:
        assert set(self._kinds("never").values()) == {"categorical"}

    def test_the_rule_only_fills_a_gap_and_never_overrides_the_parameter(self) -> None:
        space = SearchSpace(
            [
                Choice("ordered", [1, 2, 3], "int", is_ordered=True),
                Choice("unordered", [1, 2, 3], "int", is_ordered=False),
            ]
        )
        never = SearchSpaceEncoder(space, config=EncoderConfig(unset_choice_ordering="never"))
        numeric = SearchSpaceEncoder(
            space, config=EncoderConfig(unset_choice_ordering="numeric")
        )
        for encoder in (never, numeric):
            assert {e.name: e.kind for e in encoder.encodings} == {
                "ordered": "ordered",
                "unordered": "categorical",
            }

    def test_a_two_valued_ordered_choice_round_trips_with_sorted_levels(self) -> None:
        space = SearchSpace([Choice("two", [8, 2], "int")])
        encoder = SearchSpaceEncoder(
            space, config=EncoderConfig(unset_choice_ordering="numeric")
        )
        (encoding,) = encoder.encodings
        assert (encoding.kind, encoding.levels, encoding.n_columns) == ("ordered", (2, 8), 1)
        for value, column in ((2, 0.0), (8, 1.0)):
            row = encoder.encodeParameters({"two": value})
            assert float(row[0]) == column
            assert encoder.decode(row) == {"two": value}


class TestTheOrderingRulesAreExhaustive:
    def test_the_encoder_handles_every_rule_that_the_schema_names(self) -> None:
        # A rule added to `ChoiceOrderingRule` and not to the encoder would be a rule that the
        # configuration accepts and the encoder cannot apply: it must be found here.
        rules = get_args(ChoiceOrderingRule)
        assert set(rules) == {"numeric_over_two_values", "numeric", "never"}
        for rule in rules:
            assert isinstance(_unsetMeansOrdered(rule, True, 3), bool)

    def test_a_rule_the_encoder_does_not_know_fails_loudly(self) -> None:
        with pytest.raises(AssertionError):
            _unsetMeansOrdered("sometimes", True, 3)  # type: ignore[arg-type]


class TestMaxOrdinalColumns:
    @staticmethod
    def _rows(config: EncoderConfig, max_ordinal: int | None = None) -> int:
        """The number of rows `roundingNeighbors` lists for four relaxed integer columns."""
        space = SearchSpace([Integer(f"n{i}", 0, 10) for i in range(4)])
        encoder = SearchSpaceEncoder(space, config=config)
        relaxed = torch.full((4,), 0.34, dtype=torch.double)
        return int(encoder.roundingNeighbors(relaxed, max_ordinal=max_ordinal).shape[0])

    def test_the_default_lists_every_combination_of_four_columns(self) -> None:
        assert self._rows(EncoderConfig()) == 2**4

    def test_a_smaller_limit_than_the_number_of_columns_lists_only_the_nearest_rounding(
        self,
    ) -> None:
        assert self._rows(EncoderConfig(max_ordinal_columns=3)) == 1

    def test_a_limit_equal_to_the_number_of_columns_still_lists_them_all(self) -> None:
        assert self._rows(EncoderConfig(max_ordinal_columns=4)) == 2**4

    def test_a_limit_of_zero_lists_only_the_nearest_rounding(self) -> None:
        assert self._rows(EncoderConfig(max_ordinal_columns=0)) == 1

    def test_the_limit_is_the_one_the_default_of_the_old_signature_was(self) -> None:
        # Eight columns were listed (2 ** 8 rows) and nine were not.
        for count, rows in ((8, 2**8), (9, 1)):
            space = SearchSpace([Integer(f"n{i}", 0, 10) for i in range(count)])
            relaxed = torch.full((count,), 0.34, dtype=torch.double)
            listed = SearchSpaceEncoder(space).roundingNeighbors(relaxed)
            assert listed.shape[0] == rows

    def test_an_argument_of_the_call_overrides_the_configuration(self) -> None:
        assert self._rows(EncoderConfig(max_ordinal_columns=3), max_ordinal=4) == 2**4
        assert self._rows(EncoderConfig(), max_ordinal=3) == 1

    def test_a_limit_of_the_configuration_does_not_change_a_space_without_such_columns(
        self,
    ) -> None:
        encoder = SearchSpaceEncoder(
            SearchSpace([Real("x", 0.0, 1.0)]), config=EncoderConfig(max_ordinal_columns=0)
        )
        rows = encoder.roundingNeighbors(torch.tensor([0.3], dtype=torch.double))
        assert rows.shape == (1, 1)


class TestGridTolerance:
    # 0.3 / 0.1 is 2.9999999999999996 in floating point, so without a tolerance the grid
    # 0.0, 0.1, 0.2, 0.3 seems to end at 0.2.
    SPACE = SearchSpace([Real("t", 0.0, 0.3, step_size=0.1)])

    @classmethod
    def _encoder(cls, tolerance: float | None) -> SearchSpaceEncoder:
        config = (
            EncoderConfig() if tolerance is None else EncoderConfig(grid_tolerance=tolerance)
        )
        return SearchSpaceEncoder(cls.SPACE, config=config)

    def test_the_quotient_that_the_tolerance_exists_for(self) -> None:
        assert 0.3 / 0.1 < 3.0

    def test_the_default_keeps_the_upper_bound_on_the_grid(self) -> None:
        top = torch.tensor([1.0], dtype=torch.double)
        assert self._encoder(None).decode(top) == {"t": 0.3}
        assert self._encoder(None).decode(top) == self._encoder(1e-9).decode(top)

    def test_without_a_tolerance_the_upper_bound_falls_off_the_grid(self) -> None:
        top = torch.tensor([1.0], dtype=torch.double)
        assert self._encoder(0.0).decode(top) == {"t": 0.2}

    def test_snapping_follows_the_tolerance(self) -> None:
        top = torch.tensor([1.0], dtype=torch.double)
        assert float(self._encoder(None).snapColumns(top)[0]) == 1.0
        assert float(self._encoder(0.0).snapColumns(top)[0]) == pytest.approx(0.2 / 0.3)

    def test_rounding_neighbors_follow_the_tolerance(self) -> None:
        relaxed = torch.tensor([0.95], dtype=torch.double)
        with_tolerance = self._encoder(None).roundingNeighbors(relaxed)
        without = self._encoder(0.0).roundingNeighbors(relaxed)
        decoded = {self._encoder(None).decode(row)["t"] for row in with_tolerance}
        assert decoded == {0.2, 0.3}
        assert without.shape[0] == 1
        assert self._encoder(0.0).decode(without[0]) == {"t": 0.2}

    def test_the_tolerance_is_counted_in_grid_steps(self) -> None:
        # Bounds that stop 0.001 of a step short of the fourth grid point: the default keeps it
        # out, a tolerance of 0.01 steps lets it in (the value is then clamped to the bound).
        space = SearchSpace([Real("t", 0.0, 0.2999, step_size=0.1)])
        top = torch.tensor([1.0], dtype=torch.double)
        default = SearchSpaceEncoder(space).decode(top)
        generous = SearchSpaceEncoder(space, config=EncoderConfig(grid_tolerance=0.01)).decode(
            top
        )
        assert default == {"t": 0.2}
        assert generous == {"t": 0.2999}

    def test_an_integer_range_is_not_affected_by_a_tolerance_that_is_below_half_a_step(
        self,
    ) -> None:
        space = SearchSpace([Integer("n", 0, 10)])
        top = torch.tensor([1.0], dtype=torch.double)
        for tolerance in (0.0, 1e-9, 0.4):
            config = EncoderConfig(grid_tolerance=tolerance)
            assert SearchSpaceEncoder(space, config=config).decode(top) == {"n": 10}


class TestCategoricalSubsetAttemptsFactor:
    # Four assignments of one four-level choice, of which a subset of three is asked for: the
    # random subset is drawn, and a repeated draw is discarded.
    SPACE = SearchSpace([Categorical("c", ["a", "b", "c", "d"])])
    SEEDS = range(20)

    def _subsets(self, factor: int) -> list[list[dict[int, float]]]:
        config = EncoderConfig(categorical_subset_attempts_factor=factor)
        encoder = SearchSpaceEncoder(self.SPACE, config=config)
        return [encoder.categoricalFixedFeatures(3, seed=seed) for seed in self.SEEDS]

    def test_the_default_reaches_the_requested_size_for_every_seed(self) -> None:
        encoder = SearchSpaceEncoder(self.SPACE)
        for seed in self.SEEDS:
            assert len(encoder.categoricalFixedFeatures(3, seed=seed)) == 3

    def test_a_generous_factor_reaches_the_requested_size_for_every_seed(self) -> None:
        assert {len(subset) for subset in self._subsets(100)} == {3}

    def test_a_factor_of_one_stops_after_as_many_draws_as_assignments_asked_for(self) -> None:
        sizes = [len(subset) for subset in self._subsets(1)]
        # One draw per requested assignment: a repeated draw is not made up for.
        assert max(sizes) == 3
        assert min(sizes) < 3
        assert all(size >= 1 for size in sizes)

    def test_the_factor_does_not_matter_when_every_combination_is_listed(self) -> None:
        # Under the cap nothing is drawn, so the factor has nothing to change.
        small = SearchSpaceEncoder(
            self.SPACE, config=EncoderConfig(categorical_subset_attempts_factor=1)
        )
        assert len(small.categoricalFixedFeatures(10)) == 4

    def test_the_same_seed_and_factor_give_the_same_subset(self) -> None:
        assert self._subsets(1) == self._subsets(1)


class TestNoPolicyLiteralLeftInTheEncoder:
    """Specification 3.4 and 10.7: the encoder's numbers are settings, or structural."""

    @staticmethod
    def _checker() -> ModuleType:
        # The script is not an importable package, so it is loaded from its path, under a name
        # of its own so that the module of its own tests is left alone.
        name = "check_policy_values_for_the_encoder_tests"
        spec = importlib.util.spec_from_file_location(
            name, REPO_ROOT / "scripts" / "check_policy_values.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @pytest.mark.parametrize("file_name", ["SearchSpaceEncoder.py", "ParameterEncoding.py"])
    def test_the_file_has_no_numeric_literal_that_is_not_allowed_or_structural(
        self, file_name: str
    ) -> None:
        checker = self._checker()
        source = (ENCODING_PACKAGE / file_name).read_text(encoding="utf-8")
        found = checker.findViolations(source, f"src/planexs/encoding/{file_name}")
        assert [violation.describe() for violation in found] == []

    def test_the_configuration_file_is_the_one_place_for_the_defaults(self) -> None:
        # `EncoderConfig.py` holds the numbers, and the checker does not scan a file whose only
        # public symbol is a configuration schema: that is where they belong.
        source = (ENCODING_PACKAGE / "EncoderConfig.py").read_text(encoding="utf-8")
        assert (
            self._checker().findViolations(source, "src/planexs/encoding/EncoderConfig.py")
            == []
        )
        for literal in ("1e-9", "20", "8"):
            assert literal in source

    def test_the_three_unit_cube_midpoints_carry_the_marker_with_its_reason(self) -> None:
        source = (ENCODING_PACKAGE / "SearchSpaceEncoder.py").read_text(encoding="utf-8")
        # Real comments only, as the checker reads them: the word in a docstring is no marker.
        with_reason, without_reason = self._checker().findMarkerLines(source)
        assert without_reason == set()
        lines = source.splitlines()
        assert len(with_reason) == 3
        for number in with_reason:
            assert "0.5" in lines[number - 1], lines[number - 1]
            assert MIDPOINT_MARKER in lines[number - 1]

    def test_no_marker_is_used_for_anything_but_a_midpoint(self) -> None:
        # The marker is for mathematical definitions only: it must not hide a tunable value.
        for file_name in ("SearchSpaceEncoder.py", "ParameterEncoding.py"):
            source = (ENCODING_PACKAGE / file_name).read_text(encoding="utf-8")
            with_reason, _ = self._checker().findMarkerLines(source)
            lines = source.splitlines()
            assert all(MIDPOINT_MARKER in lines[number - 1] for number in with_reason), (
                file_name
            )

    def test_the_baseline_no_longer_lists_the_encoder(self) -> None:
        baseline = (REPO_ROOT / "scripts" / "policy_values_baseline.txt").read_text(
            encoding="utf-8"
        )
        assert "SearchSpaceEncoder.py" not in baseline

    def test_the_old_literals_are_gone_from_the_code_of_the_encoder(self) -> None:
        source = (ENCODING_PACKAGE / "SearchSpaceEncoder.py").read_text(encoding="utf-8")
        for old in ("1e-9", "20 *", "max_ordinal: int = 8"):
            assert old not in source, old
