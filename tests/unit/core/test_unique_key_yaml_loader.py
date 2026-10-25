"""Unit tests for `UniqueKeyYamlLoader`: safe, no duplicate keys, numbers as in JSON."""

from __future__ import annotations

import datetime
import json
import math
import os
from typing import Any

import pytest
import yaml
from yaml.constructor import ConstructorError

from planexs.core.UniqueKeyYamlLoader import UniqueKeyYamlLoader


def load(text: str) -> Any:
    """Loads `text` with the loader under test."""
    return yaml.load(text, Loader=UniqueKeyYamlLoader)


def failureOf(text: str) -> ConstructorError:
    """The `ConstructorError` that loading `text` raises."""
    with pytest.raises(ConstructorError) as caught:
        load(text)
    return caught.value


class TestDuplicateKeys:
    def test_a_key_written_twice_is_an_error(self) -> None:
        error = failureOf("a: 1\nb: 2\na: 3\n")
        assert error.problem == "found duplicate key 'a', first defined on line 1"

    def test_the_mark_is_the_second_key(self) -> None:
        error = failureOf("a: 1\nb: 2\n  \na: 3\n")
        assert error.problem_mark is not None
        # Marks count from 0: the second `a` is on line 4, column 1.
        assert (error.problem_mark.line, error.problem_mark.column) == (3, 0)

    def test_a_duplicate_below_the_top_level(self) -> None:
        error = failureOf("outer:\n  k: 1\n  j: 2\n  k: 3\n")
        assert error.problem == "found duplicate key 'k', first defined on line 2"
        assert error.problem_mark is not None
        assert (error.problem_mark.line, error.problem_mark.column) == (3, 2)

    def test_a_duplicate_in_a_mapping_inside_a_list(self) -> None:
        error = failureOf("- a: 1\n- b: 1\n  b: 2\n")
        assert error.problem == "found duplicate key 'b', first defined on line 2"

    def test_a_duplicate_in_flow_style(self) -> None:
        assert "found duplicate key 'a'" in str(failureOf("{a: 1, b: 2, a: 3}"))

    def test_a_quoted_key_equals_the_same_key_unquoted(self) -> None:
        assert "duplicate key 'a'" in str(failureOf("a: 1\n'a': 2\n"))

    def test_two_keys_that_are_equal_values_are_duplicates(self) -> None:
        assert "duplicate key 1" in str(failureOf("1: a\n1: b\n"))

    def test_the_first_duplicate_in_the_document_is_the_one_reported(self) -> None:
        # In x, y, y, x the second `y` comes before the second `x`.
        error = failureOf("x: 1\ny: 1\ny: 2\nx: 2\n")
        assert error.problem == "found duplicate key 'y', first defined on line 2"

    def test_the_same_key_in_different_mappings_is_fine(self) -> None:
        assert load("p: {a: 1}\nq: {a: 2}\n") == {"p": {"a": 1}, "q": {"a": 2}}

    def test_the_same_key_in_each_element_of_a_list_is_fine(self) -> None:
        assert load("- {a: 1}\n- {a: 2}\n") == [{"a": 1}, {"a": 2}]

    def test_a_parent_and_a_child_may_share_a_key(self) -> None:
        assert load("type: x\nchild:\n  type: y\n") == {"type": "x", "child": {"type": "y"}}

    def test_a_mapping_may_override_what_a_merge_key_brings_in(self) -> None:
        text = "base: &b {a: 1, b: 2}\nchild:\n  <<: *b\n  a: 9\n"
        assert load(text)["child"] == {"a": 9, "b": 2}

    def test_several_merge_keys_are_not_duplicates(self) -> None:
        text = "x: &x {a: 1}\ny: &y {b: 2}\nz:\n  <<: *x\n  <<: *y\n"
        assert load(text)["z"] == {"a": 1, "b": 2}

    def test_a_key_written_twice_next_to_a_merge_key_is_still_an_error(self) -> None:
        text = "base: &b {a: 1}\nchild:\n  <<: *b\n  a: 2\n  a: 3\n"
        assert "duplicate key 'a'" in str(failureOf(text))

    def test_a_key_that_cannot_be_hashed_is_left_to_the_parent_class(self) -> None:
        error = failureOf("? [1, 2]\n: v\n")
        assert error.problem == "found unhashable key"


class TestSafety:
    def test_a_python_tag_is_an_error(self) -> None:
        error = failureOf("a: !!python/object/apply:os.getcwd []\n")
        assert error.problem is not None
        assert "could not determine a constructor for the tag" in error.problem
        assert "python/object/apply:os.getcwd" in error.problem

    def test_a_python_tag_runs_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[object] = []
        monkeypatch.setattr(os, "getcwd", lambda: calls.append("called") or "x")
        failureOf("a: !!python/object/apply:os.getcwd []\n")
        assert calls == []

    def test_a_python_name_tag_is_an_error(self) -> None:
        assert "could not determine a constructor" in str(
            failureOf("a: !!python/name:os.getcwd\n")
        )

    def test_a_syntax_error_is_a_marked_yaml_error_with_a_line(self) -> None:
        with pytest.raises(yaml.MarkedYAMLError) as caught:
            load("a: 1\nb: [1, 2\nc: 3\n")
        assert caught.value.problem_mark is not None
        assert caught.value.problem_mark.line == 2


class TestScalars:
    def test_a_date_and_a_time_stay_strings(self) -> None:
        data = load("d: 2020-01-01\nt: 2020-01-01 10:00:00\nz: 2020-01-01T10:00:00Z\n")
        assert data == {
            "d": "2020-01-01",
            "t": "2020-01-01 10:00:00",
            "z": "2020-01-01T10:00:00Z",
        }

    def test_the_standard_loader_still_makes_a_date(self) -> None:
        # Importing the loader changes nothing for anyone else's `yaml.safe_load`.
        assert yaml.safe_load("d: 2020-01-01") == {"d": datetime.date(2020, 1, 1)}

    def test_the_standard_loader_still_reads_1e5_as_a_string(self) -> None:
        assert yaml.safe_load("a: 1e5") == {"a": "1e5"}

    def test_the_standard_loader_table_still_has_the_timestamp_pattern(self) -> None:
        tags = {tag for tag, _ in yaml.SafeLoader.yaml_implicit_resolvers["2"]}
        assert "tag:yaml.org,2002:timestamp" in tags

    def test_the_float_pattern_is_registered_once(self) -> None:
        for first in "-+0123456789.":
            tags = [tag for tag, _ in UniqueKeyYamlLoader.yaml_implicit_resolvers[first]]
            assert tags.count("tag:yaml.org,2002:float") == 1, first

    @pytest.mark.parametrize(
        "literal",
        [
            "0", "1", "-1", "10", "123456789", "-0", "0.0", "1.5", "-0.5", "3.14159",
            "1e5", "1E5", "1e-3", "1E-3", "1e+3", "1.5e3", "1.5E3", "1.5e-3", "-2.5e+10",
            "0.5e0", "123.456e-7",
        ],
    )  # fmt: skip
    def test_a_number_is_read_as_json_reads_it(self, literal: str) -> None:
        expected = json.loads(literal)
        for text in (literal, f"a: {literal}", f"[{literal}, {literal}]", f"{{k: {literal}}}"):
            got = load(text)
            value = got.get("a", got.get("k")) if isinstance(got, dict) else got
            if isinstance(value, list):
                value = value[0]
            assert value == expected, text
            assert type(value) is type(expected), text

    @pytest.mark.parametrize(
        "literal", ["1e5x", "e5", "1.2.3", "1e", "--1", ".e5", "1:30.5", "1e5.0"]
    )
    def test_what_is_not_a_number_stays_a_string(self, literal: str) -> None:
        assert load(f"a: {literal}") == {"a": literal}

    def test_infinity_and_not_a_number(self) -> None:
        data = load("a: .inf\nb: -.inf\nc: .nan\n")
        assert data["a"] == math.inf
        assert data["b"] == -math.inf
        assert math.isnan(data["c"])

    def test_a_quoted_number_is_a_string(self) -> None:
        assert load("a: '1e5'\nb: \"5\"\n") == {"a": "1e5", "b": "5"}

    def test_the_quirks_of_yaml_1_1_are_kept_and_quoting_avoids_them(self) -> None:
        data = load("a: no\nb: yes\nc: on\nd: off\ne: 010\nf: 12:30\ng: 'no'\n")
        assert data == {
            "a": False,
            "b": True,
            "c": True,
            "d": False,
            "e": 8,
            "f": 750,
            "g": "no",
        }


class TestPlainData:
    def test_a_document_comes_back_as_plain_values_only(self) -> None:
        data = load(
            "name: study\nn: 3\nx: 1.5\nflag: true\nnothing: null\nitems: [1, two, {k: 3.0}]\n"
        )
        assert data == {
            "name": "study",
            "n": 3,
            "x": 1.5,
            "flag": True,
            "nothing": None,
            "items": [1, "two", {"k": 3.0}],
        }

    def test_json_text_is_read_as_json_does(self) -> None:
        text = '{"a": [1, 2.5, null, true, "x"], "b": {"c": -1e-3}, "d": "é☃"}'
        assert load(text) == json.loads(text)

    def test_an_empty_document_is_none(self) -> None:
        assert load("") is None
        assert load("# only a comment\n") is None
