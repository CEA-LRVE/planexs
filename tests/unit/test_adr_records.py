"""ADR records guard (specification 11, roadmap task P1-01).

Architectural decisions live in `docs/adr/`. Specification 11 says that when a decision
changes, a new ADR supersedes the old one, that the old one's status line says so, and that its
text is not edited. This guard checks what a reviewer would otherwise check by eye:

- every ADR is listed in the `mkdocs.yml` navigation;
- every ADR has its `Status`, `Date` and `Deciders` header lines, and its title carries the
  number of its file;
- supersession is recorded in both directions: an ADR whose status is `Superseded by ADR-N`
  is named in the `Supersedes` line of ADR-N, and every ADR named in a `Supersedes` line has
  the status `Superseded by` that ADR;
- a live ADR has every section of the template (`scripts/new_adr.py`) and no placeholder left
  over from it.

A superseded ADR, and a stub still marked "to write", is a historical record: its text is
never edited, so it is exempt from the template checks. A stub is held to them as soon as it
is completed and its status changes.

Three layers are tested: the checks on throwaway trees, then the real tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HEADER_FIELDS = ("Status", "Date", "Deciders")
#: The sections of the template of `scripts/new_adr.py`, in order.
TEMPLATE_SECTIONS = (
    "Context",
    "Decision",
    "Options considered",
    "Trade-off analysis",
    "Consequences",
    "Action items",
)

_FILE_NAME = re.compile(r"^(\d{4})-.+\.md$")
_TITLE = re.compile(r"^# ADR-(\d{4}):")
_HEADER_LINE = re.compile(r"^\*\*([A-Za-z]+):\*\*\s*(.*?)\s*$")
_SECTION = re.compile(r"^## (.+?)\s*$")
_SUPERSEDED_BY = re.compile(r"^Superseded by ADR-(\d{4})\b")
_ADR_ID = re.compile(r"ADR-(\d{4})")
_NAVIGATION_TARGET = re.compile(r":\s+adr/(\S+\.md)\s*$", re.MULTILINE)
#: Leftovers of the template that mean a section was never filled in.
_PLACEHOLDERS: dict[str, re.Pattern[str]] = {
    "an HTML comment": re.compile(r"<!--"),
    "an empty table row": re.compile(r"^\|(?:\s*\|)+\s*$"),
    "an empty checklist item": re.compile(r"^\s*(?:\d+\.|[-*])\s*\[[ xX]\]\s*$"),
}


@dataclass(frozen=True)
class Adr:
    """One ADR file, reduced to what the guard looks at."""

    number: int
    file_name: str
    title_number: int | None
    header: dict[str, str]
    sections: tuple[str, ...]
    lines: tuple[str, ...]

    @property
    def status(self) -> str:
        return self.header.get("Status", "")

    @property
    def is_historical(self) -> bool:
        """Superseded, or a stub marked "to write": its text is not held to the template."""
        return self.status.startswith("Superseded") or "to write" in self.status.lower()

    @property
    def label(self) -> str:
        return f"ADR-{self.number:04d}"


def parseAdr(path: Path) -> Adr:
    """Reads an ADR file.

    Args:
        path: A file named `NNNN-slug.md`.

    Returns:
        The ADR: its header lines (those before the first section), its sections and its
        lines.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    title = _TITLE.match(lines[0]) if lines else None
    header: dict[str, str] = {}
    sections: list[str] = []
    for line in lines[1:]:
        section = _SECTION.match(line)
        if section:
            sections.append(section.group(1))
        elif not sections:
            field = _HEADER_LINE.match(line)
            if field:
                header[field.group(1)] = field.group(2)
    match = _FILE_NAME.match(path.name)
    assert match is not None, path.name
    return Adr(
        number=int(match.group(1)),
        file_name=path.name,
        title_number=int(title.group(1)) if title else None,
        header=header,
        sections=tuple(sections),
        lines=tuple(lines),
    )


def loadAdrs(adr_directory: Path) -> list[Adr]:
    """Reads every `NNNN-slug.md` file of a directory, sorted by number."""
    paths = sorted(p for p in adr_directory.glob("*.md") if _FILE_NAME.match(p.name))
    return [parseAdr(path) for path in paths]


def navigationTargets(mkdocs_text: str) -> set[str]:
    """The ADR file names listed in the navigation of a `mkdocs.yml`."""
    return set(_NAVIGATION_TARGET.findall(mkdocs_text))


def _listedNumbers(supersedes_line: str) -> set[int]:
    """The ADR numbers named in the value of a `Supersedes` header line."""
    return {int(number) for number in _ADR_ID.findall(supersedes_line)}


def findProblems(adr_directory: Path, mkdocs_text: str) -> list[str]:
    """Checks the ADR records of a directory against its navigation.

    Args:
        adr_directory: The directory holding the `NNNN-slug.md` records.
        mkdocs_text: The content of `mkdocs.yml`.

    Returns:
        One message per problem, each starting with the label of the ADR it concerns. Empty
        when everything is consistent.
    """
    adrs = loadAdrs(adr_directory)
    by_number = {adr.number: adr for adr in adrs}
    listed = navigationTargets(mkdocs_text)
    problems: list[str] = []

    for adr in adrs:
        if adr.file_name not in listed:
            problems.append(f"{adr.label}: not listed in the mkdocs navigation")
        for field in HEADER_FIELDS:
            if not adr.header.get(field):
                problems.append(f"{adr.label}: no '{field}' header line")
        if adr.title_number != adr.number:
            problems.append(f"{adr.label}: the title does not carry the number of the file")

        superseder = _SUPERSEDED_BY.match(adr.status)
        if adr.status.startswith("Superseded") and superseder is None:
            problems.append(
                f"{adr.label}: status '{adr.status}' does not say 'Superseded by ADR-NNNN'"
            )
        if superseder is not None:
            target = by_number.get(int(superseder.group(1)))
            if target is None:
                problems.append(f"{adr.label}: superseded by an ADR that does not exist")
            elif adr.number not in _listedNumbers(target.header.get("Supersedes", "")):
                problems.append(
                    f"{adr.label}: {target.label} does not list it under 'Supersedes'"
                )

        for named in sorted(_listedNumbers(adr.header.get("Supersedes", ""))):
            old = by_number.get(named)
            expected = f"Superseded by {adr.label}"
            if old is None:
                problems.append(
                    f"{adr.label}: supersedes ADR-{named:04d}, which does not exist"
                )
            elif old.status != expected:
                problems.append(
                    f"{old.label}: status is '{old.status}', expected '{expected}'"
                )

        if adr.is_historical:
            continue
        for section in TEMPLATE_SECTIONS:
            if section not in adr.sections:
                problems.append(f"{adr.label}: no '{section}' section")
        for number, line in enumerate(adr.lines, start=1):
            for what, pattern in _PLACEHOLDERS.items():
                if pattern.search(line):
                    problems.append(
                        f"{adr.label}: line {number} is {what}, left from the template"
                    )
    return problems


def writeAdr(
    directory: Path,
    number: int,
    status: str = "Accepted",
    supersedes: str | None = None,
    sections: tuple[str, ...] = TEMPLATE_SECTIONS,
    extra: str = "",
) -> Path:
    """Writes a well-formed ADR into a throwaway directory."""
    directory.mkdir(parents=True, exist_ok=True)
    header = f"**Status:** {status}\n**Date:** 2026-10-16\n**Deciders:** project owner\n"
    if supersedes is not None:
        header += f"**Supersedes:** {supersedes}\n"
    body = "".join(f"\n## {name}\n\nSome text.\n" for name in sections)
    path = directory / f"{number:04d}-decision.md"
    path.write_text(f"# ADR-{number:04d}: Decision\n\n{header}{body}{extra}", encoding="utf-8")
    return path


def navigationFor(*numbers: int) -> str:
    """A `mkdocs.yml` navigation listing the ADRs `numbers`."""
    entries = "".join(f'      - "ADR-{n:04d}": adr/{n:04d}-decision.md\n' for n in numbers)
    return "nav:\n  - Architecture Decisions:\n" + entries


class TestFindProblems:
    def test_a_consistent_set_has_no_problem(self, tmp_path: Path) -> None:
        writeAdr(tmp_path, 1, status="Superseded by ADR-0002", sections=("Context",))
        writeAdr(tmp_path, 2, supersedes="ADR-0001")
        assert findProblems(tmp_path, navigationFor(1, 2)) == []

    def test_an_adr_missing_from_the_navigation_is_reported(self, tmp_path: Path) -> None:
        writeAdr(tmp_path, 1)
        writeAdr(tmp_path, 2)
        assert findProblems(tmp_path, navigationFor(1)) == [
            "ADR-0002: not listed in the mkdocs navigation"
        ]

    def test_a_missing_header_line_and_a_wrong_title_number_are_reported(
        self, tmp_path: Path
    ) -> None:
        path = writeAdr(tmp_path, 1)
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace("**Date:** 2026-10-16\n", "").replace("# ADR-0001", "# ADR-0007"),
            encoding="utf-8",
        )
        assert findProblems(tmp_path, navigationFor(1)) == [
            "ADR-0001: no 'Date' header line",
            "ADR-0001: the title does not carry the number of the file",
        ]

    def test_a_live_adr_must_have_every_section_of_the_template(self, tmp_path: Path) -> None:
        writeAdr(tmp_path, 1, sections=("Context", "Decision", "Consequences"))
        assert findProblems(tmp_path, navigationFor(1)) == [
            "ADR-0001: no 'Options considered' section",
            "ADR-0001: no 'Trade-off analysis' section",
            "ADR-0001: no 'Action items' section",
        ]

    def test_a_superseded_adr_and_a_stub_are_exempt_from_the_template(
        self, tmp_path: Path
    ) -> None:
        writeAdr(tmp_path, 1, status="Superseded by ADR-0003", sections=("Context",))
        writeAdr(tmp_path, 2, status="Proposed (to write in full)", sections=("Context",))
        writeAdr(tmp_path, 3, supersedes="ADR-0001")
        assert findProblems(tmp_path, navigationFor(1, 2, 3)) == []

    def test_a_stub_is_held_to_the_template_once_it_is_completed(self, tmp_path: Path) -> None:
        writeAdr(tmp_path, 1, status="Accepted", sections=("Context",))
        assert findProblems(tmp_path, navigationFor(1)) == [
            f"ADR-0001: no '{section}' section" for section in TEMPLATE_SECTIONS[1:]
        ]

    def test_each_kind_of_template_placeholder_is_reported(self, tmp_path: Path) -> None:
        leftovers = {
            "an HTML comment": "<!-- What problem does this decision address? -->",
            "an empty table row": "| | | |",
            "an empty checklist item": "1. [ ]",
        }
        for what, line in leftovers.items():
            directory = tmp_path / what.replace(" ", "_")
            path = writeAdr(directory, 1, extra=f"\n{line}\n")
            line_number = path.read_text(encoding="utf-8").split("\n").index(line) + 1
            assert findProblems(directory, navigationFor(1)) == [
                f"ADR-0001: line {line_number} is {what}, left from the template"
            ], what

    def test_real_content_that_looks_like_the_template_is_not_a_placeholder(
        self, tmp_path: Path
    ) -> None:
        extra = (
            "\n| Option | Description |\n|---|---|\n| A | Real text |\n"
            "\n1. [x] A finished item\n2. [ ] A pending item\n"
        )
        writeAdr(tmp_path, 1, extra=extra)
        assert findProblems(tmp_path, navigationFor(1)) == []

    def test_a_superseded_status_must_name_an_existing_adr(self, tmp_path: Path) -> None:
        writeAdr(tmp_path, 1, status="Superseded by ADR-0009", sections=("Context",))
        assert findProblems(tmp_path, navigationFor(1)) == [
            "ADR-0001: superseded by an ADR that does not exist"
        ]

    def test_a_superseded_status_must_be_in_the_form_the_guard_can_follow(
        self, tmp_path: Path
    ) -> None:
        writeAdr(tmp_path, 1, status="Superseded", sections=("Context",))
        assert findProblems(tmp_path, navigationFor(1)) == [
            "ADR-0001: status 'Superseded' does not say 'Superseded by ADR-NNNN'"
        ]

    def test_the_superseding_adr_must_list_the_superseded_one(self, tmp_path: Path) -> None:
        writeAdr(tmp_path, 1, status="Superseded by ADR-0002", sections=("Context",))
        writeAdr(tmp_path, 2)
        assert findProblems(tmp_path, navigationFor(1, 2)) == [
            "ADR-0001: ADR-0002 does not list it under 'Supersedes'"
        ]

    def test_an_adr_listed_as_superseded_must_have_its_status_updated(
        self, tmp_path: Path
    ) -> None:
        writeAdr(tmp_path, 1, status="Accepted")
        writeAdr(tmp_path, 2, supersedes="ADR-0001")
        assert findProblems(tmp_path, navigationFor(1, 2)) == [
            "ADR-0001: status is 'Accepted', expected 'Superseded by ADR-0002'"
        ]

    def test_an_adr_cannot_supersede_one_that_does_not_exist(self, tmp_path: Path) -> None:
        writeAdr(tmp_path, 2, supersedes="ADR-0001")
        assert findProblems(tmp_path, navigationFor(2)) == [
            "ADR-0002: supersedes ADR-0001, which does not exist"
        ]

    def test_one_adr_can_supersede_several(self, tmp_path: Path) -> None:
        for number in (1, 2, 3):
            writeAdr(tmp_path, number, status="Superseded by ADR-0004", sections=("Context",))
        writeAdr(tmp_path, 4, supersedes="ADR-0001, ADR-0002, ADR-0003")
        assert findProblems(tmp_path, navigationFor(1, 2, 3, 4)) == []


class TestRealTree:
    def realProblems(self) -> list[str]:
        mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        return findProblems(REPO_ROOT / "docs" / "adr", mkdocs)

    def test_the_adr_records_are_consistent(self) -> None:
        problems = self.realProblems()
        assert problems == [], "ADR records are inconsistent:\n" + "\n".join(problems)

    def test_adr_0009_supersedes_the_five_records_the_roadmap_names(self) -> None:
        adrs = {adr.number: adr for adr in loadAdrs(REPO_ROOT / "docs" / "adr")}
        named = {int(n) for n in _ADR_ID.findall(adrs[9].header["Supersedes"])}
        assert named == {1, 5, 6, 7, 8}
        for number in named:
            assert adrs[number].status == "Superseded by ADR-0009", adrs[number].label

    def test_adr_0009_is_a_live_record_and_therefore_complete(self) -> None:
        adr = {a.number: a for a in loadAdrs(REPO_ROOT / "docs" / "adr")}[9]
        assert not adr.is_historical
        assert set(TEMPLATE_SECTIONS) <= set(adr.sections)

    def test_the_scan_sees_every_adr_file_of_the_directory(self) -> None:
        on_disk = {
            p.name for p in (REPO_ROOT / "docs" / "adr").glob("[0-9][0-9][0-9][0-9]-*.md")
        }
        assert {a.file_name for a in loadAdrs(REPO_ROOT / "docs" / "adr")} == on_disk
        assert len(on_disk) >= 9
