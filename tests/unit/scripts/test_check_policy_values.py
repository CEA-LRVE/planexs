"""Tests of `scripts/check_policy_values.py` (specification 3.4 and 10.7).

The script is not an importable package, so it is loaded from its path. The tests are grouped
by the rule they pin: which numbers are allowed, how signs are read, the `# structural:`
marker, the configuration-schema exclusion, the position of a literal, the baseline (the
ratchet), and the command line.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_policy_values.py"


def _loadChecker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_policy_values", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `dataclass` looks the module up while it builds the class.
    sys.modules["check_policy_values"] = module
    spec.loader.exec_module(module)
    return module


checker = _loadChecker()


def violationsOf(source: str) -> list[Any]:
    """Runs the checker on a snippet; the first code line of the snippet is line 1."""
    return checker.findViolations(textwrap.dedent(source).lstrip("\n"), "pkg/module.py")


def spellingsOf(source: str) -> list[str]:
    return [violation.spelling for violation in violationsOf(source)]


def whereOf(source: str) -> list[tuple[str, str, str | None]]:
    """The `(position, scope, parameter)` of each violation of a snippet."""
    return [(v.position, v.scope, v.parameter) for v in violationsOf(source)]


class TestAllowedLiterals:
    @pytest.mark.parametrize(
        "literal",
        [
            "0",
            "1",
            "2",
            "-1",
            "+1",
            "+2",
            "- 1",
            "0.0",
            "1.0",
            "2.0",
            "2.00",
            "-1.0",
            "-0.0",
            "1e0",
            "2e0",
            "10e-1",
            "0x1",
            "0x2",
            "0b10",
            "0o2",
        ],
    )
    def test_the_values_0_1_minus_1_and_2_are_allowed_in_any_spelling(
        self, literal: str
    ) -> None:
        assert violationsOf(f"x = {literal}\n") == []

    @pytest.mark.parametrize(
        ("literal", "value"),
        [
            ("3", "3"),
            ("10", "10"),
            ("0.5", "0.5"),
            ("2.5", "2.5"),
            ("1e-9", "1e-09"),
            ("0.000000001", "1e-09"),
            ("1_000", "1000"),
            ("1_000_003", "1000003"),
            ("0x10", "16"),
            ("0b11", "3"),
            ("0.5e1", "5.0"),
            ("1j", "1j"),
        ],
    )
    def test_any_other_value_is_reported_under_its_normalized_value(
        self, literal: str, value: str
    ) -> None:
        (violation,) = violationsOf(f"x = {literal}\n")

        assert violation.value == value
        assert violation.spelling == literal

    def test_two_spellings_of_one_value_share_a_baseline_key(self) -> None:
        (first,) = violationsOf("x = 1e-9\n")
        (second,) = violationsOf("x = 0.000000001\n")

        assert first.key == second.key

    def test_numbers_that_are_not_literals_are_not_seen(self) -> None:
        source = """
            name = "0.5"
            flag = True
            other = False
            nothing = None
            raw = b"42"
            dots = ...
            "A docstring with 42 and 0.5 inside."
            text = f"{name:.12g}"  # a comment with 99 in it
        """
        assert violationsOf(source) == []


class TestNegativeNumbers:
    def test_minus_one_is_allowed(self) -> None:
        assert violationsOf("x = -1\ny = -1.0\n") == []

    @pytest.mark.parametrize("literal", ["-2", "-3", "-0.5", "- 3"])
    def test_a_negative_number_other_than_minus_one_is_one_signed_violation(
        self, literal: str
    ) -> None:
        (violation,) = violationsOf(f"x = {literal}\n")

        assert violation.value == f"-{literal.lstrip('- ')}"
        assert violation.spelling == literal.replace(" ", "")

    def test_the_two_inside_minus_two_does_not_make_it_allowed(self) -> None:
        # The naive reading (the literal 2 is allowed, so -2 is too) is wrong: -2 is a tensor
        # axis index and counts.
        assert spellingsOf("rows = x[:-2]\ncolumns = x.mean(dim=-2)\n") == ["-2", "-2"]

    def test_a_binary_minus_is_not_a_sign(self) -> None:
        assert spellingsOf("a = b - 3\n") == ["3"]
        assert violationsOf("a = b - 2\nc = b - 1\n") == []

    def test_a_sign_in_front_of_a_name_is_not_a_literal(self) -> None:
        assert violationsOf("a = -x\nb = +x\n") == []

    def test_a_signed_number_after_a_binary_operator_is_reported_once(self) -> None:
        assert spellingsOf("x = 5 - -3\n") == ["5", "-3"]

    def test_the_column_of_a_signed_number_is_that_of_its_sign(self) -> None:
        (violation,) = violationsOf("x = [1, -3]\n")

        assert (violation.line, violation.column) == (1, 9)


class TestStructuralMarker:
    def test_a_marker_with_a_reason_exempts_its_line(self) -> None:
        assert violationsOf("x = 0.5  # structural: unit-cube midpoint\n") == []

    def test_a_marker_without_a_reason_exempts_nothing_and_says_so(self) -> None:
        (violation,) = violationsOf("x = 0.5  # structural:\n")

        assert violation.marker_without_reason
        assert "no reason" in violation.describe()

    def test_a_marker_whose_reason_is_only_whitespace_exempts_nothing(self) -> None:
        (violation,) = violationsOf("x = 0.5  # structural:    \n")

        assert violation.marker_without_reason

    def test_a_marker_without_a_space_after_the_hash_works(self) -> None:
        assert violationsOf("x = 0.5  #structural: midpoint\n") == []

    def test_a_marker_exempts_only_its_own_line(self) -> None:
        source = """
            a = 0.5  # structural: midpoint
            b = 0.5
        """
        assert [violation.line for violation in violationsOf(source)] == [2]

    def test_a_marker_exempts_every_literal_of_its_line(self) -> None:
        assert violationsOf("pair = (0.5, 0.25)  # structural: both are midpoints\n") == []

    def test_in_a_multi_line_call_the_marker_covers_the_literal_on_its_line_only(self) -> None:
        source = """
            x = f(
                0.5,  # structural: midpoint
                0.25,
            )
        """
        assert [violation.line for violation in violationsOf(source)] == [3]

    def test_a_marker_after_another_comment_works(self) -> None:
        assert violationsOf("x = 0.5  # noqa: E501  # structural: midpoint\n") == []

    def test_a_signature_default_can_carry_a_marker(self) -> None:
        assert violationsOf("def f(a=0.5):  # structural: midpoint\n    return a\n") == []

    def test_text_that_looks_like_a_marker_inside_a_string_is_not_a_marker(self) -> None:
        assert spellingsOf('x = 0.5; text = "# structural: a reason"\n') == ["0.5"]

    def test_a_comment_that_merely_mentions_the_word_is_not_a_marker(self) -> None:
        assert spellingsOf("x = 0.5  # not structural: a reason\n") == ["0.5"]

    def test_the_marker_is_case_sensitive(self) -> None:
        assert spellingsOf("x = 0.5  # Structural: a reason\n") == ["0.5"]

    def test_find_marker_lines_separates_reasoned_and_reasonless_markers(self) -> None:
        source = "a = 1  # structural: why\nb = 2  # structural:\nc = 3  # plain comment\n"

        assert checker.findMarkerLines(source) == ({1}, {2})

    def test_a_marker_that_exempts_a_literal_keeps_it_out_of_the_baseline(self) -> None:
        assert checker.formatBaseline(violationsOf("x = 0.5  # structural: midpoint\n")) == (
            checker._BASELINE_HEADER
        )


class TestConfigSchemaExclusion:
    LITERALS = "    tolerance: float = 1e-9\n    restarts: int = 8\n"

    def test_a_file_with_a_single_public_config_class_is_not_scanned(self) -> None:
        assert violationsOf(f"class EncoderConfig:\n{self.LITERALS}") == []

    def test_a_class_named_exactly_config_is_a_config_class(self) -> None:
        assert violationsOf(f"class Config:\n{self.LITERALS}") == []

    def test_private_helpers_do_not_count_as_public_symbols(self) -> None:
        source = (
            f"class EncoderConfig:\n{self.LITERALS}\ndef _helper() -> int:\n    return 99\n"
        )

        assert violationsOf(source) == []

    def test_module_level_assignments_are_not_public_symbols(self) -> None:
        # The same definition as scripts/check_naming_convention.py: only top-level class and
        # def statements count.
        source = f"LIMIT = 99\n\nclass EncoderConfig:\n{self.LITERALS}"

        assert violationsOf(source) == []

    def test_a_second_public_function_brings_the_file_back_into_scope(self) -> None:
        source = f"class EncoderConfig:\n{self.LITERALS}\ndef build() -> int:\n    return 99\n"

        assert spellingsOf(source) == ["1e-9", "8", "99"]

    def test_a_second_public_async_function_brings_the_file_back_into_scope(self) -> None:
        public = "async def build() -> int:\n    return 99\n"
        source = f"class EncoderConfig:\n{self.LITERALS}\n{public}"

        assert "99" in spellingsOf(source)

    def test_two_public_config_classes_are_not_a_single_public_symbol(self) -> None:
        source = f"class AConfig:\n{self.LITERALS}\nclass BConfig:\n    n: int = 5\n"

        assert spellingsOf(source) == ["1e-9", "8", "5"]

    def test_a_class_whose_name_does_not_end_with_config_is_scanned(self) -> None:
        assert spellingsOf(f"class Encoder:\n{self.LITERALS}") == ["1e-9", "8"]

    def test_config_must_be_the_end_of_the_name_not_just_in_it(self) -> None:
        assert spellingsOf(f"class ConfigHelper:\n{self.LITERALS}") == ["1e-9", "8"]

    def test_a_function_named_like_a_config_is_not_a_config_class(self) -> None:
        assert spellingsOf("def makeConfig() -> int:\n    return 8\n") == ["8"]

    def test_an_empty_file_has_no_violation(self) -> None:
        assert violationsOf("") == []


class TestPositions:
    def test_a_module_constant_is_module_level(self) -> None:
        assert whereOf("LIMIT = 3\n") == [("module level", "<module>", None)]

    def test_a_class_attribute_is_module_level_in_the_scope_of_its_class(self) -> None:
        assert whereOf("class A:\n    limit = 3\n") == [("module level", "A", None)]

    def test_a_function_body_is_a_function_body(self) -> None:
        assert whereOf("def f():\n    return 3\n") == [("function body", "f", None)]

    def test_a_method_body_is_scoped_to_class_and_method(self) -> None:
        source = "class A:\n    def m(self):\n        return 3\n"

        assert whereOf(source) == [("function body", "A.m", None)]

    def test_an_async_function_is_handled_like_a_function(self) -> None:
        assert whereOf("async def f(a=3):\n    return 4\n") == [
            ("signature default", "f", "a"),
            ("function body", "f", None),
        ]

    def test_a_signature_default_names_its_function_and_its_parameter(self) -> None:
        assert whereOf("def f(a, b=3):\n    return a\n") == [("signature default", "f", "b")]

    def test_a_keyword_only_default_is_a_signature_default(self) -> None:
        assert whereOf("def f(a, *, c=4.5):\n    return a\n") == [
            ("signature default", "f", "c")
        ]

    def test_defaults_are_matched_to_the_right_parameters_with_positional_only(self) -> None:
        assert whereOf("def f(a=3, /, b=5, *, c=7):\n    return a\n") == [
            ("signature default", "f", "a"),
            ("signature default", "f", "b"),
            ("signature default", "f", "c"),
        ]

    def test_a_parameter_without_default_is_skipped_in_the_alignment(self) -> None:
        assert whereOf("def f(a, b, c=3):\n    return a\n") == [
            ("signature default", "f", "c")
        ]

    def test_a_method_default_is_scoped_to_class_and_method(self) -> None:
        source = "class A:\n    def __init__(self, n=10):\n        self.n = n\n"

        assert whereOf(source) == [("signature default", "A.__init__", "n")]

    def test_a_lambda_default_and_a_lambda_body_are_told_apart(self) -> None:
        assert whereOf("g = lambda x=3: x\n") == [("signature default", "<lambda>", "x")]
        assert whereOf("g = lambda x: x + 3\n") == [("function body", "<lambda>", None)]

    def test_a_nested_function_has_a_dotted_scope(self) -> None:
        source = "def outer():\n    def inner(k=7):\n        return 9\n"

        assert whereOf(source) == [
            ("signature default", "outer.inner", "k"),
            ("function body", "outer.inner", None),
        ]

    def test_a_lambda_in_a_default_has_a_function_body_of_its_own(self) -> None:
        assert whereOf("def f(a=lambda: 3):\n    return a\n") == [
            ("function body", "f.<lambda>", None)
        ]

    def test_a_decorator_argument_is_evaluated_where_the_function_is_defined(self) -> None:
        source = "@deco(3)\ndef f():\n    return 0\n"

        assert whereOf(source) == [("module level", "<module>", None)]

    def test_a_decorator_of_a_method_belongs_to_the_class_scope(self) -> None:
        source = "class A:\n    @deco(3)\n    def m(self):\n        return 0\n"

        assert whereOf(source) == [("module level", "A", None)]

    def test_an_annotation_is_evaluated_where_the_function_is_defined(self) -> None:
        assert whereOf("def f(x: Limited[3]) -> Limited[5]:\n    return x\n") == [
            ("module level", "<module>", None),
            ("module level", "<module>", None),
        ]

    def test_a_comprehension_in_a_function_is_part_of_the_function_body(self) -> None:
        assert whereOf("def f(xs):\n    return [x * 3 for x in xs]\n") == [
            ("function body", "f", None)
        ]

    def test_line_and_column_are_one_based_and_violations_come_in_file_order(self) -> None:
        source = "def f():\n    a = 3\n    return a + 7\n"

        assert [(v.line, v.column, v.spelling) for v in violationsOf(source)] == [
            (2, 9, "3"),
            (3, 16, "7"),
        ]

    def test_the_description_gives_file_line_value_position_scope_and_parameter(self) -> None:
        (violation,) = violationsOf("def f(a, limit=512):\n    return a\n")

        assert violation.describe() == (
            "pkg/module.py:1:16: 512 [signature default; f, parameter limit]"
        )

    def test_the_baseline_key_has_no_line_number_and_no_column(self) -> None:
        (violation,) = violationsOf("def f(a, limit=512):\n    return a\n")

        assert violation.key == "pkg/module.py :: f :: signature default :: 512"


def makeTree(root: Path, files: dict[str, str]) -> Path:
    """Writes `files` under `<root>/src/planexs` and returns the root."""
    for name, text in files.items():
        target = root / "src" / "planexs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return root


def run(*arguments: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = checker.main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


ONE_VIOLATION = {"a.py": "def f():\n    return 0.5\n"}
ACCEPTED_ENTRY = "src/planexs/a.py :: f :: function body :: 0.5"


class TestReportWithoutBaseline:
    def test_every_violation_is_reported_and_the_exit_status_is_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path, ONE_VIOLATION)

        code, out, _ = run("--root", str(root), capsys=capsys)

        assert code == 1
        assert "1 violation(s) in 1 file(s)" in out
        assert "src/planexs/a.py:2:12: 0.5 [function body; f]" in out

    def test_a_clean_tree_exits_0(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path, {"a.py": "def f():\n    return 1\n"})

        code, out, _ = run("--root", str(root), capsys=capsys)

        assert code == 0
        assert "OK" in out

    def test_a_configuration_schema_file_is_ignored_in_a_tree_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(
            tmp_path, {"EncoderConfig.py": "class EncoderConfig:\n    n: int = 8\n"}
        )

        code, _, _ = run("--root", str(root), capsys=capsys)

        assert code == 0

    def test_files_in_subpackages_are_scanned_and_reported_with_their_relative_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path, {"models/deep/b.py": "LIMIT = 9\n"})

        _, out, _ = run("--root", str(root), capsys=capsys)

        assert "src/planexs/models/deep/b.py:1:9: 9 [module level; <module>]" in out

    def test_only_src_planexs_is_scanned(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path, {"a.py": "x = 1\n"})
        (root / "tests").mkdir()
        (root / "tests" / "test_x.py").write_text("LIMIT = 99\n", encoding="utf-8")
        (root / "scripts").mkdir()
        (root / "scripts" / "tool.py").write_text("LIMIT = 99\n", encoding="utf-8")

        code, _, _ = run("--root", str(root), capsys=capsys)

        assert code == 0

    def test_a_file_that_does_not_parse_exits_2_and_names_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path, {"broken.py": "def f(:\n"})

        code, _, err = run("--root", str(root), capsys=capsys)

        assert code == 2
        assert "src/planexs/broken.py" in err
        assert "cannot parse" in err

    def test_a_missing_source_directory_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run("--root", str(tmp_path), capsys=capsys)

        assert code == 2
        assert "does not exist" in err


class TestBaselineRatchet:
    @staticmethod
    def baseline(tmp_path: Path, *entries: str) -> Path:
        path = tmp_path / "baseline.txt"
        path.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
        return path

    def test_a_baseline_that_lists_every_violation_passes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)
        baseline = self.baseline(tmp_path, ACCEPTED_ENTRY)

        code, out, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert code == 0
        assert "OK (1 violation(s) in 1 file(s)" in out

    def test_a_new_violation_fails_and_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(
            tmp_path / "repo",
            {**ONE_VIOLATION, "b.py": "def g():\n    return 0.25\n"},
        )
        baseline = self.baseline(tmp_path, ACCEPTED_ENTRY)

        code, out, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert code == 1
        assert "1 violation(s) not in the baseline" in out
        assert "src/planexs/b.py:2:12: 0.25" in out
        assert "src/planexs/a.py:2" not in out
        assert "stale" not in out

    def test_a_stale_entry_fails_and_names_the_entry(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", {"a.py": "def f():\n    return 1\n"})
        baseline = self.baseline(tmp_path, ACCEPTED_ENTRY)

        code, out, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert code == 1
        assert "1 stale baseline entry" in out
        assert ACCEPTED_ENTRY in out
        assert "entries are only ever removed" in out
        assert "not in the baseline" not in out

    def test_a_new_violation_and_a_stale_entry_are_both_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", {"b.py": "def g():\n    return 0.25\n"})
        baseline = self.baseline(tmp_path, ACCEPTED_ENTRY)

        code, out, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert code == 1
        assert "not in the baseline" in out
        assert "stale baseline entry" in out

    def test_a_new_violation_message_says_how_to_fix_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)
        baseline = self.baseline(tmp_path)

        _, out, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert "configuration schema" in out
        assert "# structural: <reason>" in out

    def test_editing_code_above_a_literal_does_not_disturb_the_baseline(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)
        baseline = self.baseline(tmp_path, ACCEPTED_ENTRY)
        makeTree(
            tmp_path / "repo",
            {"a.py": '"""A new docstring."""\n\nimport os\n\n\ndef f():\n    return 0.5\n'},
        )

        code, _, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert code == 0

    def test_two_identical_literals_in_one_scope_need_two_entries(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", {"a.py": "def f():\n    return 0.5 + 0.5\n"})

        listed_once = self.baseline(tmp_path, ACCEPTED_ENTRY)
        code, out, _ = run("--root", str(root), "--baseline", str(listed_once), capsys=capsys)
        assert code == 1
        assert "1 violation(s) not in the baseline" in out
        assert "src/planexs/a.py:2:18: 0.5" in out  # the later occurrence is the new one

        listed_twice = self.baseline(tmp_path, ACCEPTED_ENTRY, ACCEPTED_ENTRY)
        code, _, _ = run("--root", str(root), "--baseline", str(listed_twice), capsys=capsys)
        assert code == 0

    def test_an_entry_listed_twice_for_a_literal_that_occurs_once_is_half_stale(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)
        baseline = self.baseline(tmp_path, ACCEPTED_ENTRY, ACCEPTED_ENTRY)

        code, out, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert code == 1
        assert "1 stale baseline entry" in out

    def test_comments_and_blank_lines_of_the_baseline_are_ignored(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)
        baseline = tmp_path / "baseline.txt"
        baseline.write_text(f"# a comment\n\n   \n{ACCEPTED_ENTRY}\n  # indented\n", "utf-8")

        code, _, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert code == 0

    def test_an_empty_baseline_accepts_a_clean_tree(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", {"a.py": "x = 1\n"})
        baseline = self.baseline(tmp_path)

        code, out, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)

        assert code == 0
        assert "OK (0 violation(s)" in out

    def test_a_missing_baseline_file_exits_2_and_says_how_to_create_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)

        code, _, err = run(
            "--root", str(root), "--baseline", str(tmp_path / "nope.txt"), capsys=capsys
        )

        assert code == 2
        assert "does not exist" in err
        assert "--write-baseline" in err

    def test_compare_to_baseline_flags_the_later_occurrences_as_new(self) -> None:
        violations = [
            checker.Violation("a.py", line, 1, "0.5", "0.5", "function body", "f")
            for line in (1, 2, 3)
        ]

        comparison = checker.compareToBaseline(violations, Counter({violations[0].key: 1}))

        assert [violation.line for violation in comparison.new] == [2, 3]
        assert comparison.stale == []

    def test_compare_to_baseline_reports_an_entry_that_matches_nothing_as_stale(self) -> None:
        comparison = checker.compareToBaseline(
            [], Counter({"x :: y :: function body :: 3": 2})
        )

        assert comparison.new == []
        assert comparison.stale == ["x :: y :: function body :: 3"] * 2


class TestWriteBaseline:
    def test_it_writes_a_commented_sorted_baseline_that_the_check_then_accepts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(
            tmp_path / "repo",
            {"b.py": "LIMIT = 9\n", "a.py": "def f():\n    return 0.5\n"},
        )
        baseline = tmp_path / "baseline.txt"

        code, out, err = run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )

        text = baseline.read_text(encoding="utf-8")
        entries = [line for line in text.split("\n") if line and not line.startswith("#")]
        assert (code, err) == (0, "")
        assert "wrote 2 entries" in out
        assert text.startswith("# Policy-value baseline")
        assert "only ever removed" in text
        assert entries == [
            ACCEPTED_ENTRY,
            "src/planexs/b.py :: <module> :: module level :: 9",
        ]
        code, _, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)
        assert code == 0

    def test_writing_twice_gives_identical_bytes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", {"a.py": "def f():\n    return 0.5 + 0.25 + 0.5\n"})
        baseline = tmp_path / "baseline.txt"

        run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )
        first = baseline.read_bytes()
        run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )

        assert baseline.read_bytes() == first

    def test_entries_are_sorted_whatever_the_order_of_the_source(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # In the source 0.9 comes before 0.1; the baseline lists them in sorted order, so that
        # a regenerated file differs from the committed one only where entries changed.
        root = makeTree(tmp_path / "repo", {"a.py": "def f():\n    return 0.9 + 0.1\n"})
        baseline = tmp_path / "baseline.txt"

        run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )

        entries = [x for x in baseline.read_text("utf-8").split("\n") if x and x[0] != "#"]
        assert entries == [
            "src/planexs/a.py :: f :: function body :: 0.1",
            "src/planexs/a.py :: f :: function body :: 0.9",
        ]

    def test_a_duplicate_literal_is_written_once_per_occurrence(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", {"a.py": "def f():\n    return 0.5 + 0.5\n"})
        baseline = tmp_path / "baseline.txt"

        run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )

        assert baseline.read_text(encoding="utf-8").split("\n").count(ACCEPTED_ENTRY) == 2

    def test_a_clean_tree_gives_a_baseline_with_a_header_and_no_entry(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", {"a.py": "x = 1\n"})
        baseline = tmp_path / "baseline.txt"

        run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )

        assert baseline.read_text(encoding="utf-8") == checker._BASELINE_HEADER
        code, _, _ = run("--root", str(root), "--baseline", str(baseline), capsys=capsys)
        assert code == 0

    def test_removing_entries_is_reported_without_a_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", {"a.py": "x = 1\n"})
        baseline = tmp_path / "baseline.txt"
        baseline.write_text(f"{ACCEPTED_ENTRY}\n", encoding="utf-8")

        code, out, err = run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )

        assert code == 0
        assert "(0 added, 1 removed)" in out
        assert err == ""

    def test_adding_entries_to_an_existing_baseline_is_not_done_without_the_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)
        baseline = tmp_path / "baseline.txt"
        baseline.write_text("# empty so far\n", encoding="utf-8")

        code, out, err = run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )

        assert code == 1
        assert out == ""
        assert "baseline not written" in err
        assert "only shrink" in err
        assert "src/planexs/a.py:2:12: 0.5" in err  # what would have been added
        assert "--allow-additions" in err  # the way to do it on purpose
        assert baseline.read_text(encoding="utf-8") == "# empty so far\n"  # nothing written

    def test_allow_additions_accepts_the_new_entries_and_notes_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)
        baseline = tmp_path / "baseline.txt"
        baseline.write_text("# empty so far\n", encoding="utf-8")

        code, out, err = run(
            "--root",
            str(root),
            "--baseline",
            str(baseline),
            "--write-baseline",
            "--allow-additions",
            capsys=capsys,
        )

        assert code == 0
        assert "(1 added, 0 removed)" in out
        assert "note: 1 entry was added" in err
        assert "only shrink" in err
        assert ACCEPTED_ENTRY in baseline.read_text(encoding="utf-8")

    def test_a_write_that_both_adds_and_removes_is_not_done_as_a_whole(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The stale entry would be removed, but the new one is not accepted: nothing changes.
        root = makeTree(tmp_path / "repo", {"b.py": "def g():\n    return 0.25\n"})
        baseline = tmp_path / "baseline.txt"
        baseline.write_text(f"{ACCEPTED_ENTRY}\n", encoding="utf-8")

        code, _, err = run(
            "--root", str(root), "--baseline", str(baseline), "--write-baseline", capsys=capsys
        )

        assert code == 1
        assert "src/planexs/b.py:2:12: 0.25" in err
        assert baseline.read_text(encoding="utf-8") == f"{ACCEPTED_ENTRY}\n"

    def test_the_first_write_needs_no_flag_even_though_every_entry_is_new(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)

        code, _, err = run(
            "--root",
            str(root),
            "--baseline",
            str(tmp_path / "does-not-exist-yet.txt"),
            "--write-baseline",
            capsys=capsys,
        )

        assert code == 0
        assert err == ""

    def test_allow_additions_without_write_baseline_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as stopped:
            checker.main(["--allow-additions", "--baseline", str(tmp_path / "b.txt")])

        assert stopped.value.code == 2
        assert "only makes sense together with --write-baseline" in capsys.readouterr().err

    def test_the_first_write_has_nothing_to_compare_with_and_does_not_warn(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = makeTree(tmp_path / "repo", ONE_VIOLATION)

        code, out, err = run(
            "--root",
            str(root),
            "--baseline",
            str(tmp_path / "new.txt"),
            "--write-baseline",
            capsys=capsys,
        )

        assert code == 0
        assert "added" not in out
        assert err == ""

    def test_read_baseline_keeps_the_multiplicity_of_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.txt"
        path.write_text("# header\na\nb\na\n\n", encoding="utf-8")

        assert checker.readBaseline(path) == Counter({"a": 2, "b": 1})


class TestCommandLine:
    def test_write_baseline_without_a_baseline_file_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as stopped:
            checker.main(["--write-baseline"])

        assert stopped.value.code == 2
        assert "--write-baseline needs --baseline" in capsys.readouterr().err

    def test_the_default_root_is_the_checkout_that_holds_the_script(self) -> None:
        assert checker.DEFAULT_ROOT == REPO_ROOT

    def test_the_script_runs_as_a_program_with_the_documented_exit_statuses(
        self, tmp_path: Path
    ) -> None:
        root = makeTree(tmp_path, ONE_VIOLATION)
        command = [sys.executable, str(SCRIPT), "--root", str(root)]

        failing = subprocess.run(command, capture_output=True, text=True, check=False)
        makeTree(tmp_path, {"a.py": "def f():\n    return 1\n"})
        passing = subprocess.run(command, capture_output=True, text=True, check=False)

        assert failing.returncode == 1
        assert "src/planexs/a.py:2:12: 0.5" in failing.stdout
        assert passing.returncode == 0

    def test_a_reader_that_closes_the_pipe_early_does_not_get_a_traceback(
        self, tmp_path: Path
    ) -> None:
        many = "".join(f"V{number} = {number + 3}\n" for number in range(60))
        root = makeTree(tmp_path, {"a.py": many})
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--root", str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdout.close()  # the reader goes away before the script writes anything

        errors = process.stderr.read()
        process.wait()

        assert "Traceback" not in errors
        assert "BrokenPipeError" not in errors

    def test_help_documents_the_options(self) -> None:
        helped = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert helped.returncode == 0
        for option in ("--root", "--baseline", "--write-baseline"):
            assert option in helped.stdout


class TestModuleDocstring:
    @pytest.mark.parametrize(
        "phrase",
        ["cannot see", "strings", "booleans", "hidden behind an allowed number", "review"],
    )
    def test_it_states_what_the_script_cannot_see_and_that_review_covers_it(
        self, phrase: str
    ) -> None:
        assert phrase in " ".join((checker.__doc__ or "").split())

    def test_it_names_the_rules_a_contributor_needs(self) -> None:
        documented = " ".join((checker.__doc__ or "").split())

        for phrase in ("# structural: <reason>", "ends with `Config`", "--write-baseline"):
            assert phrase in documented
