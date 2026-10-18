"""Documentation entry-point guard (specification 11, roadmap task P0-07).

`mkdocs build --strict` checks the links of the pages under `docs/`, but nothing checks the
README, which lives outside `docs_dir`, and an anchor to a heading that no longer exists is
only an INFO message for mkdocs. This guard checks every relative link of `README.md` and of
every Markdown file under `docs/`: the target file or directory must exist and, when the link
carries an anchor into a Markdown file, that file must have a heading with that anchor. A page
under `docs/` is held to the stricter rule of the built site: it may only link to a file under
`docs/`, because a link to the README, to a script or to a directory exists on disk but is dead
on the site. The guard also checks that the README carries no phase number and no test count,
which go stale: the status of the project lives in the roadmap.

External links (any URL scheme, or a protocol-relative one) are not checked: the tests do not
use the network. Anchors follow GitHub's rule (lowercase, punctuation dropped, spaces turned
into hyphens, a numeric suffix for a repeated heading), which is how the README is read. Only
ATX headings (`## Title`) are recognized.

Three layers are tested: the scanner on strings, the link check on a throwaway tree, and the
real tree.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]

_FENCE = re.compile(r"^\s*(```|~~~)")
#: An inline code span on one line (same definition as `test_typography.py`).
_INLINE_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
#: The `](target)` part of a link or an image, with an optional title. Matching the closing
#: bracket rather than a whole `[text](target)` also finds the target of a link that wraps an
#: image, such as the CI badge.
_INLINK_TARGET = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
#: A reference definition such as `[label]: target`.
_REFERENCE_DEFINITION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
_URL_SCHEME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*:|//)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)(?:\s+#+)?\s*$")

#: What the README must not state: a phase number or a test count (both go stale).
_STALE_STATUS_CLAIMS: dict[str, re.Pattern[str]] = {
    "a phase number": re.compile(r"\bphase\s+\d+", re.IGNORECASE),
    "a test count": re.compile(r"\b\d+\s+(?:unit\s+|integration\s+)?tests?\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class BrokenLink:
    """One relative link that does not resolve."""

    path: str
    line: int
    target: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.target} ({self.reason})"


def extractLinks(text: str) -> list[tuple[int, str]]:
    """Finds the link and image targets of a Markdown text, outside code.

    Fenced code blocks and inline code spans are skipped, so that a link written as an example
    is not checked.

    Args:
        text: The content of a Markdown file.

    Returns:
        One `(line_number, target)` pair per link, with 1-based line numbers, in file order.
        The target is returned as written, anchor and query included.
    """
    found: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(text.split("\n"), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        visible = _INLINE_CODE_SPAN.sub("", line)
        found.extend((number, match.group(1)) for match in _INLINK_TARGET.finditer(visible))
        definition = _REFERENCE_DEFINITION.match(visible)
        if definition:
            found.append((number, definition.group(1)))
    return found


def isExternal(target: str) -> bool:
    """Whether a link target leaves the repository (a URL scheme or a protocol-relative URL).

    Args:
        target: A link target as written.

    Returns:
        `True` for `https://...`, `mailto:...` and `//host/...`.
    """
    return bool(_URL_SCHEME.match(target))


def githubSlug(heading: str) -> str:
    """The anchor GitHub gives a heading, before any numeric suffix for a repeated heading.

    Args:
        heading: The heading text, without its leading `#` characters.

    Returns:
        The heading in lowercase, with links reduced to their text, punctuation dropped and
        spaces turned into hyphens.
    """
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading.strip()).lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def headingAnchors(text: str) -> set[str]:
    """The anchors a Markdown text offers, one per ATX heading outside fenced code.

    Args:
        text: The content of a Markdown file.

    Returns:
        The anchors, with `-1`, `-2`, ... appended for the second and later headings that share
        a slug.
    """
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    in_fence = False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if match is None:
            continue
        slug = githubSlug(match.group(1))
        count = seen.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return anchors


def collectDocumentationFiles(root: Path) -> list[Path]:
    """Lists the Markdown files the guard checks: the README and everything under `docs/`.

    Args:
        root: The repository root.

    Returns:
        The existing files, sorted by path.
    """
    candidates = [root / "README.md", *(root / "docs").rglob("*.md")]
    return sorted(path for path in candidates if path.is_file())


def findBrokenLinks(root: Path, files: Sequence[Path]) -> list[BrokenLink]:
    """Checks the relative links of `files`.

    A link is broken when its target file or directory does not exist, or when it names an
    anchor that no heading of the target Markdown file produces. A link that starts with `/`
    is read from `root`. Anchors into files that are not Markdown are not checked. A file under
    `docs/` is also broken when its link leaves `docs/` or names a directory: the built site
    has no page for either (`mkdocs build --strict` warns about the first).

    Args:
        root: The repository root, which paths in the result are relative to.
        files: The Markdown files to check, all under `root`.

    Returns:
        The broken links, in the order of `files` and then of each file.
    """
    broken: list[BrokenLink] = []
    anchors_of: dict[Path, set[str]] = {}
    docs_dir = root / "docs"
    for file in files:
        shown = file.relative_to(root).as_posix()
        for number, target in extractLinks(file.read_text(encoding="utf-8")):
            if isExternal(target):
                continue
            location, _, anchor = target.partition("#")
            location = unquote(location.partition("?")[0])
            if location == "":
                destination = file
            elif location.startswith("/"):
                destination = root / location.lstrip("/")
            else:
                destination = file.parent / location
            if not destination.exists():
                broken.append(BrokenLink(shown, number, target, "the target does not exist"))
                continue
            if file.is_relative_to(docs_dir):
                inside = Path(os.path.normpath(destination))
                if not (inside.is_relative_to(docs_dir) and inside.is_file()):
                    broken.append(
                        BrokenLink(
                            shown,
                            number,
                            target,
                            "a docs page can only link to a file under docs/",
                        )
                    )
                    continue
            if anchor and destination.is_file() and destination.suffix == ".md":
                if destination not in anchors_of:
                    anchors_of[destination] = headingAnchors(
                        destination.read_text(encoding="utf-8")
                    )
                if unquote(anchor).lower() not in anchors_of[destination]:
                    broken.append(
                        BrokenLink(shown, number, target, "no heading has this anchor")
                    )
    return broken


def findStaleStatusClaims(text: str) -> list[tuple[int, str]]:
    """Finds the phase numbers and test counts of a text, outside code.

    Args:
        text: The content of the README.

    Returns:
        One `(line_number, what)` pair per claim and per line, with 1-based line numbers.
    """
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        visible = _INLINE_CODE_SPAN.sub("", line)
        for what, pattern in _STALE_STATUS_CLAIMS.items():
            if pattern.search(visible):
                found.append((number, what))
    return found


def writeFile(root: Path, relative: str, text: str) -> Path:
    """Writes a file of a throwaway tree, creating its directories."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestExtractLinks:
    def test_an_inline_link_and_an_image_are_found_with_their_line(self) -> None:
        text = "intro\nsee [a](./a.md) and ![logo](img/logo.png) here\n"
        assert extractLinks(text) == [(2, "./a.md"), (2, "img/logo.png")]

    def test_a_link_that_wraps_an_image_gives_both_targets(self) -> None:
        text = "[![CI](https://x.test/badge.svg)](https://x.test/workflow)\n"
        assert extractLinks(text) == [
            (1, "https://x.test/badge.svg"),
            (1, "https://x.test/workflow"),
        ]

    def test_a_title_and_angle_brackets_are_not_part_of_the_target(self) -> None:
        assert extractLinks('[a](./a.md "A title") [b](<./b.md>)') == [
            (1, "./a.md"),
            (1, "./b.md"),
        ]

    def test_the_anchor_is_kept_in_the_target(self) -> None:
        assert extractLinks("[a](./a.md#some-heading)") == [(1, "./a.md#some-heading")]

    def test_a_reference_definition_is_found(self) -> None:
        assert extractLinks("text [a][ref]\n\n[ref]: ./a.md\n") == [(3, "./a.md")]

    def test_a_link_in_a_fenced_block_is_skipped(self) -> None:
        text = "```markdown\n[a](./missing.md)\n```\n[b](./b.md)\n"
        assert extractLinks(text) == [(4, "./b.md")]

    def test_a_link_in_an_inline_code_span_is_skipped(self) -> None:
        assert extractLinks("write `[a](./missing.md)` like this, or [b](./b.md)") == [
            (1, "./b.md")
        ]

    def test_a_code_span_inside_the_link_text_does_not_hide_the_target(self) -> None:
        assert extractLinks("[`spec.md`](./docs/spec.md)") == [(1, "./docs/spec.md")]


class TestIsExternal:
    def test_urls_with_a_scheme_and_protocol_relative_urls_are_external(self) -> None:
        for target in (
            "https://ax.dev",
            "http://x.test",
            "mailto:a@b.test",
            "//cdn.test/x.js",
        ):
            assert isExternal(target), target

    def test_relative_paths_and_bare_anchors_are_not_external(self) -> None:
        for target in ("./a.md", "../a.md", "a.md#x", "#x", "/docs/a.md", "docs/a:b.md"):
            assert not isExternal(target), target


class TestHeadingAnchors:
    def test_a_numbered_heading_matches_githubs_slug(self) -> None:
        assert headingAnchors("## 6. Extension contracts\n") == {"6-extension-contracts"}

    def test_punctuation_and_code_spans_are_dropped(self) -> None:
        text = "# `BayesianOptimizer.ask`: what's new?\n"
        assert headingAnchors(text) == {"bayesianoptimizerask-whats-new"}

    def test_a_link_in_a_heading_is_reduced_to_its_text(self) -> None:
        assert headingAnchors("## See [the spec](./spec.md)\n") == {"see-the-spec"}

    def test_a_closing_hash_sequence_is_not_part_of_the_slug(self) -> None:
        assert headingAnchors("## Title ##\n") == {"title"}

    def test_a_repeated_heading_gets_a_numeric_suffix(self) -> None:
        assert headingAnchors("# Notes\n## Notes\n### Notes\n") == {
            "notes",
            "notes-1",
            "notes-2",
        }

    def test_a_heading_in_a_fenced_block_is_not_a_heading(self) -> None:
        assert headingAnchors("```bash\n# a comment\n```\n## Real\n") == {"real"}


class TestFindBrokenLinks:
    def check(self, root: Path) -> list[BrokenLink]:
        return findBrokenLinks(root, collectDocumentationFiles(root))

    def test_a_resolving_link_to_a_file_and_to_a_directory_is_fine(
        self, tmp_path: Path
    ) -> None:
        writeFile(tmp_path, "README.md", "[a](./docs/a.md) [e](./examples)\n")
        writeFile(tmp_path, "docs/a.md", "# A\n")
        (tmp_path / "examples").mkdir()
        assert self.check(tmp_path) == []

    def test_a_missing_target_is_reported_with_its_file_and_line(self, tmp_path: Path) -> None:
        writeFile(tmp_path, "README.md", "ok\n[spec](./PROJECT_SPECIFICATION.md)\n")
        writeFile(tmp_path, "docs/PROJECT_SPECIFICATION.md", "# Spec\n")
        assert self.check(tmp_path) == [
            BrokenLink(
                "README.md", 2, "./PROJECT_SPECIFICATION.md", "the target does not exist"
            )
        ]

    def test_links_are_resolved_from_the_directory_of_their_own_file(
        self, tmp_path: Path
    ) -> None:
        # `./b.md` is right from docs/ and wrong from docs/sub/, where `../b.md` is right.
        writeFile(tmp_path, "docs/b.md", "# B\n")
        writeFile(tmp_path, "docs/a.md", "[b](./b.md)\n")
        writeFile(tmp_path, "docs/sub/c.md", "[wrong](./b.md) [right](../b.md)\n")
        assert [(b.path, b.target) for b in self.check(tmp_path)] == [
            ("docs/sub/c.md", "./b.md")
        ]

    def test_an_anchor_that_a_heading_produces_is_fine(self, tmp_path: Path) -> None:
        writeFile(tmp_path, "README.md", "[s](./docs/s.md#6-extension-contracts)\n")
        writeFile(tmp_path, "docs/s.md", "## 6. Extension contracts\n")
        assert self.check(tmp_path) == []

    def test_an_anchor_with_no_heading_is_reported(self, tmp_path: Path) -> None:
        # The bug this guard exists for: the README pointed at a section that was renumbered.
        writeFile(tmp_path, "README.md", "[r](./docs/s.md#6-roadmap)\n")
        writeFile(tmp_path, "docs/s.md", "## 6. Extension contracts\n")
        assert self.check(tmp_path) == [
            BrokenLink("README.md", 1, "./docs/s.md#6-roadmap", "no heading has this anchor")
        ]

    def test_a_bare_anchor_is_checked_against_the_same_file(self, tmp_path: Path) -> None:
        writeFile(tmp_path, "README.md", "# Top\n[ok](#top) [bad](#nowhere)\n")
        assert [b.target for b in self.check(tmp_path)] == ["#nowhere"]

    def test_external_links_are_not_checked(self, tmp_path: Path) -> None:
        writeFile(
            tmp_path, "README.md", "[a](https://nowhere.invalid/x) [m](mailto:a@b.test)\n"
        )
        assert self.check(tmp_path) == []

    def test_an_anchor_into_a_file_that_is_not_markdown_is_not_checked(
        self, tmp_path: Path
    ) -> None:
        writeFile(tmp_path, "README.md", "[l](./scripts/run.py#L10)\n")
        writeFile(tmp_path, "scripts/run.py", "print(1)\n")
        assert self.check(tmp_path) == []

    def test_a_query_string_and_percent_encoding_are_read_properly(
        self, tmp_path: Path
    ) -> None:
        writeFile(tmp_path, "README.md", "[a](./my%20file.md?plain=1)\n")
        writeFile(tmp_path, "my file.md", "# A\n")
        assert self.check(tmp_path) == []

    def test_a_root_relative_link_is_read_from_the_repository_root(
        self, tmp_path: Path
    ) -> None:
        writeFile(tmp_path, "README.md", "[a](/docs/a.md) [x](/missing.md)\n")
        writeFile(tmp_path, "docs/a.md", "# A\n")
        assert [b.target for b in self.check(tmp_path)] == ["/missing.md"]

    def test_a_link_written_as_an_example_in_code_is_not_checked(self, tmp_path: Path) -> None:
        writeFile(
            tmp_path, "README.md", "```md\n[a](./missing.md)\n```\nand `[b](./missing.md)`\n"
        )
        assert self.check(tmp_path) == []

    def test_a_docs_page_linking_out_of_docs_is_reported_even_if_the_target_exists(
        self, tmp_path: Path
    ) -> None:
        # Fine for GitHub, dead on the built site: this is what `mkdocs build --strict` warns
        # about for `../README.md` in docs/index.md.
        writeFile(tmp_path, "README.md", "# R\n")
        writeFile(tmp_path, "docs/index.md", "[readme](../README.md)\n")
        assert [(b.path, b.target) for b in self.check(tmp_path)] == [
            ("docs/index.md", "../README.md")
        ]

    def test_a_docs_page_linking_to_a_directory_is_reported(self, tmp_path: Path) -> None:
        writeFile(tmp_path, "docs/index.md", "[adrs](./adr)\n")
        writeFile(tmp_path, "docs/adr/0001.md", "# ADR\n")
        assert [b.target for b in self.check(tmp_path)] == ["./adr"]

    def test_a_docs_page_may_link_to_pages_and_assets_under_docs(self, tmp_path: Path) -> None:
        writeFile(
            tmp_path, "docs/index.md", "[a](adr/0001.md) ![i](img/x.png) [b](./b.md#top)\n"
        )
        writeFile(tmp_path, "docs/adr/0001.md", "[back](../index.md)\n")
        writeFile(tmp_path, "docs/img/x.png", "png")
        writeFile(tmp_path, "docs/b.md", "# Top\n")
        assert self.check(tmp_path) == []

    def test_the_readme_may_link_to_a_directory_and_into_docs(self, tmp_path: Path) -> None:
        writeFile(tmp_path, "README.md", "[e](./examples) [d](./docs/adr)\n")
        (tmp_path / "examples").mkdir()
        (tmp_path / "docs" / "adr").mkdir(parents=True)
        assert self.check(tmp_path) == []


class TestFindStaleStatusClaims:
    def test_the_old_readme_status_is_caught(self) -> None:
        old_status = (
            "**Phase 2 of 4 (the exploration/exploitation acquisition layer)**\n"
            "> all 425 tests pass (399 unit, 26 integration)\n"
        )
        assert findStaleStatusClaims(old_status) == [
            (1, "a phase number"),
            (2, "a test count"),
        ]

    def test_a_short_status_line_that_points_to_the_roadmap_is_fine(self) -> None:
        text = "planexs is being refactored. The [roadmap](./docs/ROADMAP.md) has the plan.\n"
        assert findStaleStatusClaims(text) == []

    def test_unrelated_numbers_and_the_word_phase_alone_are_fine(self) -> None:
        text = "A campaign of 4-5 evaluations, in the design phase, with 2 or 3 workers.\n"
        assert findStaleStatusClaims(text) == []

    def test_a_claim_in_an_inline_code_span_is_ignored(self) -> None:
        assert findStaleStatusClaims("the old wording was `Phase 2 of 4`\n") == []


class TestRealTree:
    def test_the_scan_covers_the_readme_and_the_two_reference_documents(self) -> None:
        names = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in collectDocumentationFiles(REPO_ROOT)
        }
        assert {"README.md", "docs/PROJECT_SPECIFICATION.md", "docs/ROADMAP.md"} <= names

    def test_the_readme_links_to_both_reference_documents_where_they_live(self) -> None:
        targets = [
            target
            for _, target in extractLinks(
                (REPO_ROOT / "README.md").read_text(encoding="utf-8")
            )
        ]
        assert "./docs/PROJECT_SPECIFICATION.md" in targets
        assert "./docs/ROADMAP.md" in targets

    def test_every_relative_link_of_the_readme_and_of_docs_resolves(self) -> None:
        broken = findBrokenLinks(REPO_ROOT, collectDocumentationFiles(REPO_ROOT))
        assert broken == [], "Broken documentation links:\n" + "\n".join(map(str, broken))

    def test_the_readme_states_no_phase_number_and_no_test_count(self) -> None:
        claims = findStaleStatusClaims((REPO_ROOT / "README.md").read_text(encoding="utf-8"))
        assert claims == [], (
            "README.md must not state "
            + ", ".join(f"{what} (line {number})" for number, what in claims)
            + ": the status of the project lives in docs/ROADMAP.md."
        )

    def test_the_specification_and_the_roadmap_are_in_the_navigation(self) -> None:
        navigation = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        targets = set(re.findall(r":\s+(\S+\.md)\s*$", navigation, flags=re.MULTILINE))
        assert {"PROJECT_SPECIFICATION.md", "ROADMAP.md"} <= targets
