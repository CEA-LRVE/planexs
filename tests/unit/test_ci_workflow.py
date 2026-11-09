"""The CI workflow of roadmap task P0-03 and the coverage floor it relies on.

What a test can see of the workflow is its shape: when it runs, which jobs it has, what each
job runs and on which Python versions. Whether the jobs are green is for the CI run itself.
These tests keep the shape from drifting (specification 10.13, decision D-6).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yaml"
PYPROJECT = ROOT / "pyproject.toml"

#: The Python versions of decision D-6. Change them here only when the decision changes.
PYTHON_VERSIONS = ["3.11", "3.13"]

JOBS = ["lint", "types", "tests"]


def workflow() -> dict[str, Any]:
    """The parsed workflow."""
    return dict(yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


def runs(job: str) -> list[str]:
    """The shell commands of the steps of one job, in order."""
    return [str(step["run"]) for step in workflow()["jobs"][job]["steps"] if "run" in step]


def pyproject() -> dict[str, Any]:
    """The parsed `pyproject.toml`."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


class TestTriggers:
    """The workflow runs on every push and on every pull request."""

    def test_it_runs_on_push_and_on_pull_request(self) -> None:
        # YAML 1.1 reads the bare key `on` as the boolean True.
        triggers = workflow().get("on", workflow().get(True))
        assert triggers is not None
        assert {"push", "pull_request"} <= set(triggers)


class TestJobs:
    """The three jobs of the card, on the versions of D-6, installed from the lock file."""

    def test_the_three_jobs_exist(self) -> None:
        assert set(JOBS) <= set(workflow()["jobs"])

    @pytest.mark.parametrize("job", JOBS)
    def test_each_job_runs_on_the_python_versions_of_d6(self, job: str) -> None:
        matrix = workflow()["jobs"][job]["strategy"]["matrix"]
        assert matrix["python-version"] == PYTHON_VERSIONS

    @pytest.mark.parametrize("job", JOBS)
    def test_each_job_installs_from_the_lock_file(self, job: str) -> None:
        installs = [run for run in runs(job) if "uv sync" in run]
        assert installs, job
        assert all("--locked" in run for run in installs), installs

    @pytest.mark.parametrize("job", JOBS)
    def test_a_failure_on_one_version_does_not_hide_the_other(self, job: str) -> None:
        assert workflow()["jobs"][job]["strategy"]["fail-fast"] is False

    @pytest.mark.parametrize(
        ("job", "command"),
        [
            ("lint", "ruff check ."),
            ("lint", "ruff format --check ."),
            ("lint", "scripts/check_naming_convention.py"),
            ("types", "mypy"),
            ("tests", "pytest"),
        ],
    )
    def test_each_job_runs_what_the_card_names(self, job: str, command: str) -> None:
        assert any(run.endswith(command) for run in runs(job)), (job, command, runs(job))


class TestCoverageFloor:
    """The floor is written in one place, `pyproject.toml`, and it only goes up."""

    def test_it_is_a_number_in_the_coverage_table(self) -> None:
        floor = pyproject()["tool"]["coverage"]["report"]["fail_under"]
        assert isinstance(floor, (int, float)) and not isinstance(floor, bool)
        assert 0 < floor <= 100

    def test_the_workflow_does_not_repeat_it(self) -> None:
        # Comments are not part of the parsed workflow, so the ones that explain where the
        # floor lives do not count: only commands and environment variables do.
        text = " ".join(run for job in JOBS for run in runs(job))
        text += " " + str(workflow().get("env", ""))
        assert "fail-under" not in text and "fail_under" not in text

    def test_the_pytest_options_do_not_repeat_it(self) -> None:
        addopts = pyproject()["tool"]["pytest"]["ini_options"]["addopts"]
        assert "fail-under" not in addopts

    def test_the_comment_above_it_says_that_it_may_only_go_up(self) -> None:
        text = PYPROJECT.read_text(encoding="utf-8")
        comment = text.split("[tool.coverage.report]", 1)[0].rsplit("\n\n", 1)[-1]
        assert "may only go up" in comment, comment

    def test_the_comment_records_the_history_of_the_value(self) -> None:
        history = floorHistory()
        assert history, "the comment above the floor lists no value"
        assert history == sorted(set(history)), f"the floor went down or repeated: {history}"
        assert history[-1] == pyproject()["tool"]["coverage"]["report"]["fail_under"]


def floorHistory() -> list[float]:
    """The values listed in the comment above `fail_under`, in the order written.

    Each entry of the history starts a comment line with the value, two decimals and a gap,
    for example `#   93.55  task P0-03: ...`.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    above = text.split("[tool.coverage.report]", 1)[0]
    comment = above[above.rindex("# The coverage floor.") :]
    return [
        float(m.group(1)) for m in re.finditer(r"^#\s+(\d+\.\d{2})\s", comment, re.MULTILINE)
    ]
