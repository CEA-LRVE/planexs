"""Contributor guide guard (specification 10, 12; roadmap task P1-17).

`docs/contributing.md` summarizes the rules of the specification for someone who is about to
write code. A summary that cites a section that does not exist, or that loses one of its
sections, is wrong without anybody noticing, because nothing else reads it. This guard checks
what a reviewer would otherwise check by eye:

- the guide has a section for each rule the roadmap asks it to summarize (the registry rules,
  the policy-value rule with its `# structural:` marker, the options-object rule, the
  new-component checklist, the request check, and how to run each checker), and for the naming
  and typography rules it has always had;
- each of those sections says which section of the specification it summarizes, on a
  `**Specification:**` line, and names the ones the roadmap cites for it;
- every specification section that the guide cites, on those lines or in prose
  (`specification 10.6`), exists in `docs/PROJECT_SPECIFICATION.md`;
- every checker script the guide tells a contributor to run exists, and the guide gives the
  command of every check of the roadmap's verification list (section 3.3);
- the checklist has the eight steps of specification 10.14 and the request check has the six
  questions of 12.2.

Three layers are tested: the scanners on strings, then the checks on the real tree.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "contributing.md"
SPECIFICATION = REPO_ROOT / "docs" / "PROJECT_SPECIFICATION.md"

#: The sections the guide must have, by the start of their heading, and the sections of the
#: specification that each must cite. The first six are the ones added by P1-17, the last two
#: are the ones that task keeps.
REQUIRED_SECTIONS: dict[str, set[str]] = {
    "Running the checks": {"10.1", "10.13"},
    "Before you code: the request check": {"12.2"},
    "Registry rules": {"10.6"},
    "Policy values": {"3.4", "10.7"},
    "Configuration schemas and options objects": {"10.8"},
    "Adding a component: the checklist": {"10.14"},
    "Naming convention": {"10.2"},
    "Typography": {"10.3"},
}

#: The commands of the roadmap's verification list (section 3.3), as the guide must write them.
REQUIRED_COMMANDS = (
    "uv run ruff check .",
    "uv run ruff format --check .",
    "uv run mypy",
    "uv run pytest",
    "uv run python scripts/check_naming_convention.py",
    "uv run python scripts/check_policy_values.py "
    "--baseline scripts/policy_values_baseline.txt",
    "uv run python scripts/check_layering.py",
    "uv run mkdocs build --strict",
)

_FENCE = re.compile(r"^\s*(```|~~~)")
_SPECIFICATION_HEADING = re.compile(r"^#{2,3}\s+(\d+(?:\.\d+)?)\.?\s")
_SECTION_NUMBER = r"\d+(?:\.\d+)?"
_PROSE_CITATION = re.compile(
    rf"\b[Ss]pecification\s+({_SECTION_NUMBER}(?:(?:,\s*|\s+and\s+){_SECTION_NUMBER})*)"
)
_DECLARATION = re.compile(r"^\*\*Specification:\*\*\s*(.+?)\s*$")
_GUIDE_HEADING = re.compile(r"^##\s+(.+?)\s*$")
_NUMBERED_ITEM = re.compile(r"^\d+\.\s+\S")
_SCRIPT_PATH = re.compile(r"scripts/([A-Za-z0-9_]+\.py)")


def specificationSections(text: str) -> set[str]:
    """The section numbers a specification has, from its `##` and `###` headings.

    Args:
        text: The content of the specification.

    Returns:
        Numbers such as `"3"` and `"10.14"`, outside fenced code.
    """
    found: set[str] = set()
    in_fence = False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else _SPECIFICATION_HEADING.match(line)
        if match is not None:
            found.add(match.group(1))
    return found


def citedSections(text: str) -> list[tuple[int, str]]:
    """The specification sections a text cites, on `**Specification:**` lines and in prose.

    Args:
        text: The content of the guide.

    Returns:
        One `(line_number, section_number)` pair per citation, with 1-based line numbers.
    """
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        declaration = _DECLARATION.match(line)
        if declaration is not None:
            found.extend(
                (number, n) for n in re.findall(_SECTION_NUMBER, declaration.group(1))
            )
        for citation in _PROSE_CITATION.finditer(line):
            found.extend((number, n) for n in re.findall(_SECTION_NUMBER, citation.group(1)))
    return found


def guideSections(text: str) -> dict[str, list[str]]:
    """Splits the guide at its `##` headings.

    Args:
        text: The content of the guide.

    Returns:
        Each heading, in order, with the lines under it up to the next `##` heading. A line
        in a fenced block that starts with `#` is not a heading.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    in_fence = False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _GUIDE_HEADING.match(line)
        if heading is not None:
            current = sections.setdefault(heading.group(1), [])
        elif current is not None:
            current.append(line)
    return sections


def declaredSections(lines: list[str]) -> set[str]:
    """The sections a guide section says it summarizes, from its `**Specification:**` line."""
    for line in lines:
        declaration = _DECLARATION.match(line)
        if declaration is not None:
            return set(re.findall(_SECTION_NUMBER, declaration.group(1)))
    return set()


def sectionStartingWith(sections: dict[str, list[str]], start: str) -> list[str]:
    """The lines of the guide section whose heading starts with `start`.

    Raises:
        AssertionError: if no heading does.
    """
    for heading, lines in sections.items():
        if heading.startswith(start):
            return lines
    raise AssertionError(f"the guide has no section titled {start!r}")


class TestSpecificationSections:
    def test_numbered_headings_at_both_levels_are_found(self) -> None:
        text = (
            "## 0. Language\n### 1.1 What it is\n"
            "## 10. Coding standards\n### 10.14 Checklist\n"
        )
        assert specificationSections(text) == {"0", "1.1", "10", "10.14"}

    def test_a_heading_without_a_number_and_one_in_fenced_code_are_ignored(self) -> None:
        text = "## Overview\n```bash\n## 9. not a heading\n```\n#### 3.2.1 too deep\n"
        assert specificationSections(text) == set()


class TestCitedSections:
    def test_a_declaration_line_gives_every_number_on_it(self) -> None:
        assert citedSections("**Specification:** 10.1, 10.13.\n") == [
            (1, "10.1"),
            (1, "10.13"),
        ]

    def test_a_prose_citation_with_a_list_is_read_whole(self) -> None:
        text = "see specification 3.4, 10.7 and 12.2 for it\n"
        assert citedSections(text) == [(1, "3.4"), (1, "10.7"), (1, "12.2")]

    def test_a_capitalized_citation_and_a_trailing_period_are_read(self) -> None:
        assert citedSections("Specification 10.2.\n") == [(1, "10.2")]

    def test_the_word_alone_and_the_old_wording_are_not_a_citation(self) -> None:
        text = "the specification wins\n`PROJECT_SPECIFICATION.md` sections 4.5 and 4.6\n"
        assert citedSections(text) == []

    def test_line_numbers_are_those_of_the_text(self) -> None:
        assert citedSections("a\nb\nspecification 5.1\n") == [(3, "5.1")]


class TestGuideSections:
    def test_the_text_is_split_at_second_level_headings(self) -> None:
        text = "# Title\nintro\n## One\na\n## Two\nb\nc\n"
        assert guideSections(text) == {"One": ["a"], "Two": ["b", "c", ""]}

    def test_a_hash_line_in_a_fenced_block_is_not_a_heading(self) -> None:
        text = "## One\n```bash\n## not a heading\n```\n## Two\n"
        assert list(guideSections(text)) == ["One", "Two"]

    def test_the_declaration_of_a_section_is_read_from_its_lines(self) -> None:
        assert declaredSections(["", "**Specification:** 10.6, 3.3.", "text"]) == {
            "10.6",
            "3.3",
        }
        assert declaredSections(["no declaration here"]) == set()


class TestRealGuide:
    def guide(self) -> str:
        return GUIDE.read_text(encoding="utf-8")

    def test_every_required_section_exists_and_cites_what_the_roadmap_names(self) -> None:
        sections = guideSections(self.guide())
        problems: list[str] = []
        for start, required in REQUIRED_SECTIONS.items():
            declared = declaredSections(sectionStartingWith(sections, start))
            missing = sorted(required - declared)
            if missing:
                problems.append(f"{start!r} does not cite specification {', '.join(missing)}")
        assert problems == [], "\n".join(problems)

    def test_every_cited_specification_section_exists(self) -> None:
        existing = specificationSections(SPECIFICATION.read_text(encoding="utf-8"))
        unknown = [
            f"line {number}: specification {section}"
            for number, section in citedSections(self.guide())
            if section not in existing
        ]
        assert unknown == [], "The guide cites sections the specification does not have:\n" + (
            "\n".join(unknown)
        )

    def test_the_guide_cites_something(self) -> None:
        # A scanner that silently stopped matching would make the check above vacuous.
        assert len(citedSections(self.guide())) >= 20

    def test_the_guide_gives_the_command_of_every_check(self) -> None:
        text = self.guide()
        missing = [command for command in REQUIRED_COMMANDS if command not in text]
        assert missing == [], f"The guide does not give these commands: {missing}"

    def test_every_script_the_guide_names_exists(self) -> None:
        names = set(_SCRIPT_PATH.findall(self.guide()))
        assert {"check_naming_convention.py", "check_policy_values.py"} <= names
        assert {"check_layering.py", "new_adr.py"} <= names
        missing = sorted(n for n in names if not (REPO_ROOT / "scripts" / n).is_file())
        assert missing == [], f"The guide names scripts that do not exist: {missing}"

    def test_every_test_file_the_guide_names_exists(self) -> None:
        names = set(re.findall(r"tests/[A-Za-z0-9_/]+\.py", self.guide()))
        assert "tests/unit/test_typography.py" in names
        missing = sorted(n for n in names if not (REPO_ROOT / n).is_file())
        assert missing == [], f"The guide names test files that do not exist: {missing}"

    def test_the_policy_value_section_documents_the_structural_marker(self) -> None:
        lines = sectionStartingWith(guideSections(self.guide()), "Policy values")
        body = "\n".join(lines)
        assert "# structural: <reason>" in body or "# structural: tensor axis" in body
        for rule in ("--write-baseline", "--allow-additions", "baseline"):
            assert rule in body, rule

    def test_the_checklist_has_the_eight_steps_of_specification_10_14(self) -> None:
        lines = sectionStartingWith(guideSections(self.guide()), "Adding a component")
        assert sum(1 for line in lines if _NUMBERED_ITEM.match(line)) == 8

    def test_the_request_check_has_the_six_questions_of_specification_12_2(self) -> None:
        lines = sectionStartingWith(guideSections(self.guide()), "Before you code")
        assert sum(1 for line in lines if _NUMBERED_ITEM.match(line)) == 6

    def test_the_existing_naming_and_typography_rules_are_kept(self) -> None:
        text = self.guide()
        for kept in ("`camelCase`", "File names match", "Properties", "No em dashes"):
            assert kept in text, kept
