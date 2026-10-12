"""Typography guard (specification 10.3).

The house style bans the em dash and the unicode right arrow from every tracked `.py`, `.md`,
`.yaml` and `.toml` file: code, comments, docstrings and documentation. Use a period, a colon,
parentheses or two sentences instead of a dash, and write `->` for an arrow.

Occurrences inside an inline code span (text between backticks) are ignored, so that the rule
itself can be documented. A code span must sit on one line to be recognized: a span that wraps
over two lines is not exempt, and a banned character in it is reported.

This file is scanned like every other tracked file, so it never contains a banned character
itself: they are written as escapes below.

Three layers are tested: the scanner on strings, the listing of tracked files on a throwaway
git repository, and the real tree.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

#: The banned characters, as escapes so that this file does not contain them. Adding one here
#: is the whole change needed to ban another character.
EM_DASH = "\u2014"
RIGHT_ARROW = "\u2192"
BANNED_CHARACTERS: dict[str, str] = {
    EM_DASH: "em dash",
    RIGHT_ARROW: "unicode right arrow",
}

#: File types the guard scans.
SCANNED_SUFFIXES = (".py", ".md", ".yaml", ".toml")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: An inline code span, as CommonMark defines it on one line: a run of N backticks, some text,
#: then a run of exactly N backticks. The lookarounds keep a longer run from matching a shorter
#: one, so a span written with double backticks may itself contain a single backtick.
_INLINE_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")

_FIX_HINT = (
    "Use a period, a colon, parentheses or two sentences instead of an em dash, and write -> "
    "instead of a unicode arrow. Text that documents the rule itself belongs in an inline "
    "code span (between backticks)."
)

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@dataclass(frozen=True)
class Violation:
    """One banned character found in a file."""

    path: str
    line: int
    name: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.name}"


def findBannedCharacters(text: str) -> list[tuple[int, str]]:
    """Finds the banned characters of `text` that are outside inline code spans.

    Lines are split on the newline character only, so that a reported line number is the one an
    editor shows (`str.splitlines` also splits on form feeds and unicode separators).

    Args:
        text: The content of a file.

    Returns:
        One `(line_number, character_name)` pair per banned character and per line, with
        1-based line numbers, in file order.
    """
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        visible = _INLINE_CODE_SPAN.sub("", line)
        for character, name in BANNED_CHARACTERS.items():
            if character in visible:
                found.append((number, name))
    return found


def trackedFiles(root: Path) -> list[Path]:
    """Lists the git-tracked files of `root` that the guard scans.

    Files that are in the index but missing on disk (deleted, not yet committed) and non-file
    entries such as submodules are skipped.

    Args:
        root: The root of a git work tree.

    Returns:
        The absolute paths of the tracked files whose suffix is in `SCANNED_SUFFIXES`.

    Raises:
        FileNotFoundError: if git is not installed.
        subprocess.CalledProcessError: if `root` is not a git work tree.
    """
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
    )
    names = listing.stdout.decode("utf-8").split("\0")
    paths = (root / name for name in names if name.endswith(SCANNED_SUFFIXES))
    return [path for path in paths if path.is_file()]


def findViolations(root: Path) -> list[Violation]:
    """Scans every tracked, scanned file of `root` for banned characters.

    Args:
        root: The root of a git work tree.

    Returns:
        The violations, ordered by file and line, with paths relative to `root`.
    """
    violations: list[Violation] = []
    for path in sorted(trackedFiles(root)):
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(root).as_posix()
        violations.extend(
            Violation(relative, number, name) for number, name in findBannedCharacters(text)
        )
    return violations


class TestFindBannedCharacters:
    def test_clean_text_has_no_violation(self) -> None:
        assert findBannedCharacters("Plain text, a colon: (parentheses) and ->.\n") == []

    @pytest.mark.parametrize(
        ("character", "name"),
        [(EM_DASH, "em dash"), (RIGHT_ARROW, "unicode right arrow")],
        ids=["em-dash", "right-arrow"],
    )
    def test_each_banned_character_is_reported_with_its_name(
        self, character: str, name: str
    ) -> None:
        assert findBannedCharacters(f"before {character} after") == [(1, name)]

    def test_both_characters_on_one_line_are_both_reported(self) -> None:
        text = f"a {EM_DASH} b {RIGHT_ARROW} c"
        assert findBannedCharacters(text) == [(1, "em dash"), (1, "unicode right arrow")]

    def test_one_report_per_character_and_line_however_many_times_it_occurs(self) -> None:
        assert findBannedCharacters(f"a {EM_DASH} b {EM_DASH} c") == [(1, "em dash")]

    def test_line_numbers_are_one_based_and_follow_newlines_only(self) -> None:
        # A form feed and a unicode line separator must not shift the numbering: only the
        # newline character does.
        text = f"one\ntwo\x0c still line two\u2028 still line two\nthree {EM_DASH}\n"
        assert findBannedCharacters(text) == [(3, "em dash")]

    @pytest.mark.parametrize(
        "line",
        [
            "A span `{c}` documents the rule.",
            "A double-backtick span ``{c}`` too.",
            "A span with an inner backtick ``a ` {c} b`` is still one span.",
            "Two spans `{c}` and `{c}` on one line.",
        ],
        ids=["single", "double", "inner-backtick", "two-spans"],
    )
    @pytest.mark.parametrize(
        "character", [EM_DASH, RIGHT_ARROW], ids=["em-dash", "right-arrow"]
    )
    def test_inside_an_inline_code_span_is_ignored(self, line: str, character: str) -> None:
        assert findBannedCharacters(line.format(c=character)) == []

    def test_a_character_outside_the_span_is_reported_when_another_is_inside(self) -> None:
        text = f"The rule `{EM_DASH}` is broken here {EM_DASH} on purpose."
        assert findBannedCharacters(text) == [(1, "em dash")]

    @pytest.mark.parametrize(
        "line",
        [
            "A lone backtick ` before {c} is not a span.",
            "``{c}` has mismatched backtick runs.",
            "{c}` closes a span that never opened.",
        ],
        ids=["unclosed", "mismatched-runs", "closing-only"],
    )
    def test_an_unclosed_or_mismatched_span_does_not_exempt(self, line: str) -> None:
        assert findBannedCharacters(line.format(c=EM_DASH)) == [(1, "em dash")]

    def test_a_fenced_code_block_is_scanned_not_exempt(self) -> None:
        # Only inline code spans are exempt: a fenced block is code, and the rule covers code.
        text = f"```python\nx = 1  # a {EM_DASH} b\n```\n"
        assert findBannedCharacters(text) == [(2, "em dash")]

    def test_a_span_wrapped_over_two_lines_is_not_exempt(self) -> None:
        text = f"start of a `span\n{EM_DASH} that wraps` ends here"
        assert findBannedCharacters(text) == [(2, "em dash")]

    @pytest.mark.parametrize("allowed", ["\u2013", "\u2012", "-", "--", "->"])
    def test_en_dash_hyphens_and_the_ascii_arrow_are_allowed(self, allowed: str) -> None:
        assert findBannedCharacters(f"a {allowed} b") == []


@requires_git
class TestTrackedFiles:
    @staticmethod
    def _makeRepository(root: Path) -> None:
        subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)

    @staticmethod
    def _track(root: Path, name: str, text: str) -> None:
        (root / name).write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", name], check=True)

    def test_only_tracked_files_of_the_scanned_types_are_listed(self, tmp_path: Path) -> None:
        self._makeRepository(tmp_path)
        for name in ("a.py", "b.md", "c.yaml", "d.toml", "e.txt", "f.json"):
            self._track(tmp_path, name, "x\n")
        (tmp_path / "untracked.md").write_text("x\n", encoding="utf-8")

        listed = {path.name for path in trackedFiles(tmp_path)}

        assert listed == {"a.py", "b.md", "c.yaml", "d.toml"}

    def test_a_file_in_the_index_but_missing_on_disk_is_skipped(self, tmp_path: Path) -> None:
        self._makeRepository(tmp_path)
        self._track(tmp_path, "gone.md", "x\n")
        (tmp_path / "gone.md").unlink()

        assert trackedFiles(tmp_path) == []

    def test_a_directory_that_is_not_a_work_tree_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stop git from searching upwards, in case the temporary directory is inside a
        # repository.
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

        with pytest.raises(subprocess.CalledProcessError):
            trackedFiles(tmp_path)

    def test_a_tracked_file_with_a_banned_character_is_a_violation(
        self, tmp_path: Path
    ) -> None:
        self._makeRepository(tmp_path)
        self._track(tmp_path, "clean.md", "Nothing to see.\n")
        self._track(tmp_path, "scratch.py", f"# fine\n# not fine {EM_DASH} here\n")
        self._track(tmp_path, "scratch.yaml", f"key: value {RIGHT_ARROW} other\n")

        assert findViolations(tmp_path) == [
            Violation("scratch.py", 2, "em dash"),
            Violation("scratch.yaml", 1, "unicode right arrow"),
        ]

    def test_removing_the_banned_character_clears_the_violation(self, tmp_path: Path) -> None:
        self._makeRepository(tmp_path)
        self._track(tmp_path, "scratch.md", f"a {EM_DASH} b\n")
        assert findViolations(tmp_path) != []

        (tmp_path / "scratch.md").write_text("a: b\n", encoding="utf-8")

        assert findViolations(tmp_path) == []

    def test_an_untracked_file_is_not_scanned(self, tmp_path: Path) -> None:
        self._makeRepository(tmp_path)
        (tmp_path / "scratch.md").write_text(f"a {EM_DASH} b\n", encoding="utf-8")

        assert findViolations(tmp_path) == []

    def test_a_banned_character_in_an_unscanned_file_type_is_not_reported(
        self, tmp_path: Path
    ) -> None:
        self._makeRepository(tmp_path)
        self._track(tmp_path, "data.txt", f"a {EM_DASH} b\n")

        assert findViolations(tmp_path) == []

    def test_a_banned_character_documented_in_a_code_span_is_not_a_violation(
        self, tmp_path: Path
    ) -> None:
        self._makeRepository(tmp_path)
        self._track(
            tmp_path,
            "rule.md",
            f"No em dashes (`{EM_DASH}`) and no arrows (`{RIGHT_ARROW}`).\n",
        )

        assert findViolations(tmp_path) == []

    def test_non_ascii_file_names_are_listed(self, tmp_path: Path) -> None:
        # `git ls-files` quotes such names unless given -z; the guard must still find them.
        self._makeRepository(tmp_path)
        self._track(tmp_path, "\u00e9t\u00e9.md", f"a {EM_DASH} b\n")

        assert [violation.path for violation in findViolations(tmp_path)] == [
            "\u00e9t\u00e9.md"
        ]


@requires_git
def test_tracked_files_have_no_banned_character() -> None:
    try:
        files = trackedFiles(REPO_ROOT)
    except subprocess.CalledProcessError:
        pytest.skip("not a git work tree (for example a source archive): nothing to list")

    # The scan must not pass vacuously because it looked in the wrong place.
    assert REPO_ROOT / "pyproject.toml" in files, f"the scan did not reach {REPO_ROOT}"

    violations = findViolations(REPO_ROOT)

    assert not violations, (
        "Banned typography in tracked files (specification 10.3):\n"
        + "\n".join(f"  {violation}" for violation in violations)
        + f"\n{_FIX_HINT}"
    )
