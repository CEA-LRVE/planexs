"""Equivalence of `trialsToTensors` with the legacy `encodeTrials` (roadmap task P3-08;
roadmap 2.3: a legacy implementation is deleted only after a parity test has been run).

Both functions are given the same data, written once as a list of `(point, values, noise)` and
turned into `LegacyTrial`s for the old function and into `Trial`s for the new one. Delete this
file with the legacy code (roadmap P11-02); the unit tests of the new function stay.

The tolerances are stated where they are used. The inputs and the outcomes must be *equal*,
bit for bit: both read the same encoder and the same numbers, in double precision. The noise
needs a tolerance only because the two functions return different things: the legacy one the
variance (`std ** 2`, computed in Python), the new one the standard deviation, which is what a
`SurrogateBuildContext` carries and which the builder squares.

**Documented differences** (all of them checked below, none of them hidden):

- the new function returns standard deviations and the old one variances (see above);
- the old function stops with a `ValueError` for a missing metric, the new one with a
  `SurrogateContextError` that names the trial (it is not a parity matter, it is tested in
  `test_trials_to_tensors.py`);
- the old function only knows noise that every trial gives for every metric, so when each trial
  gives it for some metrics only it returns `None` *without a warning*. The new function counts
  per value, sees that known noise is being dropped, and logs the warning (and, under the
  `error` policy, refuses). The tensors are the same.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from surrogate_fakes import POINTS, makeSpace

from planexs import LegacyMetric, LegacyObjective, LegacyTrial
from planexs.domain.Observation import Observation
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.models.encodeTrials import encodeTrials
from planexs.surrogates.trialsToTensors import trialsToTensors

LEGACY_LOGGER = "planexs.models.encodeTrials"
NEW_LOGGER = "planexs.surrogates.trialsToTensors"

#: Relative tolerance on the noise: `std ** 2` in torch against `std ** 2` in Python, both
#: double precision, so they differ by rounding at most. The inputs and outcomes use none.
NOISE_RTOL = 1e-14

# One observation of a history: the index of a point of `POINTS`, the values, the noise.
Row = tuple[int, dict[str, float], dict[str, float] | None]

ONE = ["yield"]
TWO = ["yield", "cost"]

HISTORIES: dict[str, tuple[list[str], list[Row]]] = {
    "one-outcome-no-noise": (
        ONE,
        [(0, {"yield": 1.0}, None), (1, {"yield": 2.5}, None), (2, {"yield": -3.0}, None)],
    ),
    "two-outcomes-no-noise": (
        TWO,
        [(i, {"yield": i + 0.5, "cost": 10.0 * i}, None) for i in range(len(POINTS))],
    ),
    "one-outcome-all-noise": (
        ONE,
        [(0, {"yield": 1.0}, {"yield": 0.5}), (1, {"yield": 2.0}, {"yield": 2.0})],
    ),
    "two-outcomes-all-noise": (
        TWO,
        [
            (0, {"yield": 1.0, "cost": 10.0}, {"yield": 0.5, "cost": 2.0}),
            (1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.25, "cost": 4.0}),
            (2, {"yield": 3.0, "cost": 30.0}, {"yield": 0.125, "cost": 8.0}),
        ],
    ),
    "a-replicated-point": (
        TWO,
        [
            (1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.1, "cost": 1.0}),
            (1, {"yield": 2.2, "cost": 19.0}, {"yield": 0.1, "cost": 1.0}),
        ],
    ),
}

#: Noise known for some trials only: both functions ignore it and both warn.
PARTIAL_BY_TRIAL: dict[str, tuple[list[str], list[Row]]] = {
    "one-trial-without-noise": (
        TWO,
        [
            (0, {"yield": 1.0, "cost": 10.0}, {"yield": 0.5, "cost": 2.0}),
            (1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.25, "cost": 4.0}),
            (2, {"yield": 3.0, "cost": 30.0}, None),
        ],
    ),
    "one-outcome-one-trial-without-noise": (
        ONE,
        [(0, {"yield": 1.0}, {"yield": 0.5}), (1, {"yield": 2.0}, None)],
    ),
}

#: Noise known for some metrics of every trial: both functions return no noise, but only the
#: new one says so (see the module docstring).
PARTIAL_BY_OUTCOME: dict[str, tuple[list[str], list[Row]]] = {
    "every-trial-gives-one-of-two": (
        TWO,
        [
            (0, {"yield": 1.0, "cost": 10.0}, {"yield": 0.5}),
            (1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.25}),
        ],
    ),
    "a-single-trial-gives-one-of-two": (
        TWO,
        [(0, {"yield": 1.0, "cost": 10.0}, {"cost": 2.0})],
    ),
}


def problemOf(names: Sequence[str]) -> Problem:
    return Problem(
        search_space=makeSpace(), outcomes=[Outcome(name=n, type="continuous") for n in names]
    )


def legacyTrials(rows: Sequence[Row]) -> list[LegacyTrial]:
    return [
        LegacyTrial(dict(POINTS[point]), dict(values), None if noise is None else dict(noise))
        for point, values, noise in rows
    ]


def newTrials(problem: Problem, rows: Sequence[Row], interleave: bool) -> list[Trial]:
    """The same history as `Trial`s. With `interleave`, trials that are not completed are
    placed between them: the new function must leave them out, where the legacy one is only
    ever given completed ones.
    """
    trials: list[Trial] = []
    for position, (point, values, noise) in enumerate(rows):
        if interleave:
            trials.append(Trial(id=100 + position, parameters=POINTS[point]))  # pending
        trials.append(
            Trial(
                id=position,
                parameters=POINTS[point],
                state=TrialState.completed,
                observation=Observation.create(problem.outcomes, values, noise),
            )
        )
        if interleave:
            trials.append(
                Trial(
                    id=200 + position,
                    parameters=POINTS[point],
                    state=TrialState.failed,
                    reason="instrument down",
                )
            )
    return trials


def runBoth(
    names: list[str], rows: list[Row], interleave: bool, caplog: pytest.LogCaptureFixture
) -> tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None], list[str], list[str]]:
    """Runs both functions on one history, with what each one logged as a warning."""
    problem = problemOf(names)
    encoder = SearchSpaceEncoder(problem.search_space)
    objective = LegacyObjective([LegacyMetric(name, minimize=False) for name in names])

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        legacy = encodeTrials(encoder, objective, legacyTrials(rows))
        legacy_logged = [r.getMessage() for r in caplog.records if r.name == LEGACY_LOGGER]
        caplog.clear()
        new = trialsToTensors(problem, encoder, newTrials(problem, rows, interleave))
        new_logged = [r.getMessage() for r in caplog.records if r.name == NEW_LOGGER]
    return new, legacy, legacy_logged, new_logged


def assertSame(
    new: Any, legacy: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
) -> None:
    """The inputs and outcomes are equal; the noise is the square root of the variance."""
    legacy_x, legacy_y, legacy_variance = legacy
    torch.testing.assert_close(new.train_inputs, legacy_x, rtol=0.0, atol=0.0)
    torch.testing.assert_close(new.train_outcomes, legacy_y, rtol=0.0, atol=0.0)
    assert new.train_inputs.dtype == legacy_x.dtype == torch.float64
    assert new.train_outcomes.dtype == legacy_y.dtype == torch.float64
    if legacy_variance is None:
        assert new.train_noise_std is None
    else:
        assert new.train_noise_std is not None
        torch.testing.assert_close(
            new.train_noise_std**2, legacy_variance, rtol=NOISE_RTOL, atol=0.0
        )


@pytest.mark.parametrize("interleave", [False, True], ids=["completed-only", "interleaved"])
class TestTheSameTensors:
    @pytest.mark.parametrize("name", list(HISTORIES))
    def test_complete_histories(
        self, name: str, interleave: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        names, rows = HISTORIES[name]

        new, legacy, legacy_logged, new_logged = runBoth(names, rows, interleave, caplog)

        assertSame(new, legacy)
        assert legacy_logged == [] and new_logged == []

    def test_the_noise_of_a_complete_history_is_the_square_root_of_the_variance(
        self, interleave: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        names, rows = HISTORIES["two-outcomes-all-noise"]

        new, legacy, _, _ = runBoth(names, rows, interleave, caplog)

        assert legacy[2] is not None and new.train_noise_std is not None
        assert legacy[2].tolist() == [[0.25, 4.0], [0.0625, 16.0], [0.015625, 64.0]]
        assert new.train_noise_std.tolist() == [[0.5, 2.0], [0.25, 4.0], [0.125, 8.0]]

    @pytest.mark.parametrize("name", list(PARTIAL_BY_TRIAL))
    def test_noise_known_for_some_trials_is_ignored_by_both_with_a_warning(
        self, name: str, interleave: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        names, rows = PARTIAL_BY_TRIAL[name]

        new, legacy, legacy_logged, new_logged = runBoth(names, rows, interleave, caplog)

        assertSame(new, legacy)
        assert new.train_noise_std is None
        assert len(legacy_logged) == 1 and "ignoring result_std" in legacy_logged[0]
        assert len(new_logged) == 1 and "ignoring them and inferring" in new_logged[0]

    @pytest.mark.parametrize("name", list(PARTIAL_BY_OUTCOME))
    def test_noise_known_for_some_outcomes_gives_the_same_tensors_but_only_the_new_one_warns(
        self, name: str, interleave: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        names, rows = PARTIAL_BY_OUTCOME[name]

        new, legacy, legacy_logged, new_logged = runBoth(names, rows, interleave, caplog)

        assertSame(new, legacy)
        assert new.train_noise_std is None
        assert legacy_logged == []  # the documented difference: the old function is silent
        assert len(new_logged) == 1
        assert "noise standard deviations are known" in new_logged[0]
