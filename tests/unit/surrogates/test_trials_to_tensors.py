"""Unit tests for `trialsToTensors` (roadmap task P3-08, specification 5.4, 5.6).

The function turns the completed trials of a study into `train_inputs`, `train_outcomes` and
`train_noise_std`, the tensors that fill a `SurrogateBuildContext`. Four things are tested:
which trials become rows and in which order, what the columns hold, what happens with the known
noise (three situations, two policies), and that the result is what the build context accepts.

The equivalence with the legacy `encodeTrials` is in `test_trials_to_tensors_parity.py`, which
is deleted with the legacy code (roadmap P11-02).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from surrogate_fakes import OUTCOME_NAMES, POINTS, makeProblem

from planexs.core.ConfigError import ConfigError
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Observation import Observation
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchGenerator
from planexs.surrogates.NoisePolicyConfig import NoisePolicyConfig
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext
from planexs.surrogates.SurrogateContextError import SurrogateContextError
from planexs.surrogates.TrainingTensors import TrainingTensors
from planexs.surrogates.trialsToTensors import trialsToTensors

LOGGER = "planexs.surrogates.trialsToTensors"
N_COLUMNS = 5  # the encoded columns of the search space of `surrogate_fakes`
ERROR = NoisePolicyConfig(partial="error")
IGNORE = NoisePolicyConfig(partial="ignore_with_warning")


def encoderOf(problem: Problem, runtime: RuntimeConfig | None = None) -> SearchSpaceEncoder:
    return SearchSpaceEncoder(problem.search_space, runtime)


def completed(
    trial_id: int,
    point: int,
    values: Mapping[str, Any],
    noise: Mapping[str, Any] | None = None,
    problem: Problem | None = None,
) -> Trial:
    """A completed trial at `POINTS[point]`, its observation checked against the problem."""
    outcomes = (problem or makeProblem()).outcomes
    return Trial(
        id=trial_id,
        parameters=POINTS[point],
        state=TrialState.completed,
        observation=Observation.create(outcomes, values, noise),
    )


def pending(trial_id: int, point: int) -> Trial:
    return Trial(id=trial_id, parameters=POINTS[point])


def failed(trial_id: int, point: int) -> Trial:
    return Trial(
        id=trial_id,
        parameters=POINTS[point],
        state=TrialState.failed,
        reason="instrument down",
    )


def abandoned(trial_id: int, point: int) -> Trial:
    return Trial(id=trial_id, parameters=POINTS[point], state=TrialState.abandoned)


def mixedProblem() -> Problem:
    """One continuous and one binary outcome, over the search space of the fakes."""
    return Problem(
        search_space=makeProblem().search_space,
        outcomes=[
            Outcome(name="yield", type="continuous"),
            Outcome(name="phase", type="binary"),
        ],
    )


def binaryProblem() -> Problem:
    return Problem(
        search_space=makeProblem().search_space,
        outcomes=[Outcome(name="phase", type="binary")],
    )


def warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING]


class TestWhichTrialsBecomeRows:
    def test_only_completed_trials_are_used(self) -> None:
        problem = makeProblem()
        history = [
            pending(0, 0),
            completed(1, 1, {"yield": 2.0, "cost": 20.0}),
            failed(2, 2),
            abandoned(3, 3),
            completed(4, 3, {"yield": 4.0, "cost": 40.0}),
        ]

        result = trialsToTensors(problem, encoderOf(problem), history)

        assert result.train_inputs.shape == (2, N_COLUMNS)
        assert result.train_outcomes.tolist() == [[2.0, 20.0], [4.0, 40.0]]

    @pytest.mark.parametrize(
        "excluded", [pending(0, 0), failed(0, 0), abandoned(0, 0)], ids=lambda t: t.state.value
    )
    def test_each_state_that_is_not_completed_is_left_out(self, excluded: Trial) -> None:
        problem = makeProblem()
        kept = completed(1, 1, {"yield": 2.0, "cost": 20.0})

        result = trialsToTensors(problem, encoderOf(problem), [excluded, kept])

        assert result.train_outcomes.tolist() == [[2.0, 20.0]]

    def test_the_rows_keep_the_order_of_the_history_not_the_order_of_the_identifiers(
        self,
    ) -> None:
        problem = makeProblem()
        encoder = encoderOf(problem)
        history = [
            completed(9, 2, {"yield": 3.0, "cost": 30.0}),
            completed(2, 0, {"yield": 1.0, "cost": 10.0}),
            completed(5, 1, {"yield": 2.0, "cost": 20.0}),
        ]

        result = trialsToTensors(problem, encoder, history)

        assert result.train_outcomes.tolist() == [[3.0, 30.0], [1.0, 10.0], [2.0, 20.0]]
        for row, point in zip(result.train_inputs, [2, 0, 1], strict=True):
            assert torch.equal(row, encoder.encodeParameters(POINTS[point]))

    def test_a_replicate_is_a_row_of_its_own(self) -> None:
        problem = makeProblem()
        history = [
            completed(0, 1, {"yield": 2.0, "cost": 20.0}),
            completed(1, 1, {"yield": 2.5, "cost": 21.0}),
        ]

        result = trialsToTensors(problem, encoderOf(problem), history)

        assert result.train_outcomes.tolist() == [[2.0, 20.0], [2.5, 21.0]]
        assert torch.equal(result.train_inputs[0], result.train_inputs[1])

    def test_any_sequence_of_trials_is_accepted(self) -> None:
        problem = makeProblem()
        history = (completed(0, 0, {"yield": 1.0, "cost": 10.0}),)

        assert trialsToTensors(problem, encoderOf(problem), history).train_outcomes.shape == (
            1,
            2,
        )

    @pytest.mark.parametrize(
        "history",
        [[], [pending(0, 0), failed(1, 1), abandoned(2, 2)]],
        ids=["empty-history", "nothing-completed"],
    )
    def test_a_history_without_a_completed_trial_has_no_row(
        self, history: list[Trial]
    ) -> None:
        problem = makeProblem()

        result = trialsToTensors(problem, encoderOf(problem), history)

        assert result.train_inputs.shape == (0, N_COLUMNS)
        assert result.train_outcomes.shape == (0, 2)
        assert result.train_noise_std is None


class TestWhatTheColumnsHold:
    def test_the_inputs_are_the_encoding_of_the_parameters(self) -> None:
        problem = makeProblem()
        encoder = encoderOf(problem)
        history = [completed(i, i, {"yield": 1.0, "cost": 2.0}) for i in range(len(POINTS))]

        result = trialsToTensors(problem, encoder, history)

        expected = torch.stack([encoder.encodeParameters(point) for point in POINTS])
        assert torch.equal(result.train_inputs, expected)
        assert bool(((result.train_inputs >= 0.0) & (result.train_inputs <= 1.0)).all())

    def test_the_outcome_columns_follow_the_problem_and_not_the_observation(self) -> None:
        problem = makeProblem()
        assert OUTCOME_NAMES == ("yield", "cost")
        # The observation lists `cost` first: the stored mapping order must not matter.
        trial = Trial(
            id=0,
            parameters=POINTS[0],
            state=TrialState.completed,
            observation=Observation(values={"cost": 10.0, "yield": 1.0}),
        )

        result = trialsToTensors(problem, encoderOf(problem), [trial])

        assert result.train_outcomes.tolist() == [[1.0, 10.0]]

    def test_a_value_the_problem_does_not_name_is_ignored(self) -> None:
        problem = makeProblem()
        trial = Trial(
            id=0,
            parameters=POINTS[0],
            state=TrialState.completed,
            observation=Observation(values={"yield": 1.0, "cost": 10.0, "extra": 99.0}),
        )

        result = trialsToTensors(problem, encoderOf(problem), [trial])

        assert result.train_outcomes.tolist() == [[1.0, 10.0]]

    def test_it_is_a_frozen_record_of_three_tensors(self) -> None:
        problem = makeProblem()

        result = trialsToTensors(problem, encoderOf(problem), [])

        assert isinstance(result, TrainingTensors)
        with pytest.raises(AttributeError):
            result.train_inputs = torch.zeros(1)  # type: ignore[misc]


class TestABinaryOutcome:
    def test_its_labels_are_zeros_and_ones(self) -> None:
        problem = binaryProblem()
        history = [
            completed(0, 0, {"phase": False}, problem=problem),
            completed(1, 1, {"phase": True}, problem=problem),
            completed(2, 2, {"phase": 1}, problem=problem),
        ]

        result = trialsToTensors(problem, encoderOf(problem), history)

        assert result.train_outcomes.shape == (3, 1)
        assert result.train_outcomes.flatten().tolist() == [0.0, 1.0, 1.0]

    def test_it_has_no_noise_and_nothing_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        problem = binaryProblem()
        history = [completed(0, 0, {"phase": True}, problem=problem)]

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

        assert result.train_noise_std is None
        assert warnings(caplog) == []

    def test_a_stored_deviation_for_a_kind_without_noise_is_never_read(self) -> None:
        # `Observation.create` refuses it, but a stored document is read back unchecked.
        problem = binaryProblem()
        trial = Trial(
            id=0,
            parameters=POINTS[0],
            state=TrialState.completed,
            observation=Observation(values={"phase": 1.0}, noise_std={"phase": 0.3}),
        )

        result = trialsToTensors(problem, encoderOf(problem), [trial], noise_policy=ERROR)

        assert result.train_noise_std is None

    def test_next_to_a_continuous_outcome_the_columns_follow_the_problem(self) -> None:
        problem = mixedProblem()
        history = [
            completed(0, 0, {"yield": 1.5, "phase": False}, problem=problem),
            completed(1, 1, {"yield": 2.5, "phase": True}, problem=problem),
        ]

        result = trialsToTensors(problem, encoderOf(problem), history)

        assert result.train_outcomes.tolist() == [[1.5, 0.0], [2.5, 1.0]]


class TestTheKnownNoise:
    def test_when_none_is_known_it_is_inferred_and_nothing_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        problem = makeProblem()
        history = [completed(i, i, {"yield": 1.0, "cost": 2.0}) for i in range(3)]

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

        assert result.train_noise_std is None
        assert warnings(caplog) == []

    def test_when_all_is_known_the_tensor_holds_standard_deviations_not_variances(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        problem = makeProblem()
        history = [
            completed(0, 0, {"yield": 1.0, "cost": 10.0}, {"yield": 0.5, "cost": 2.0}),
            completed(1, 1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.25, "cost": 4.0}),
        ]

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

        assert result.train_noise_std is not None
        assert result.train_noise_std.tolist() == [[0.5, 2.0], [0.25, 4.0]]
        assert result.train_noise_std.shape == result.train_outcomes.shape
        assert warnings(caplog) == []

    def test_a_deviation_of_zero_is_a_known_deviation(self) -> None:
        problem = makeProblem()
        history = [completed(0, 0, {"yield": 1.0, "cost": 10.0}, {"yield": 0.0, "cost": 0.0})]

        result = trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

        assert result.train_noise_std is not None
        assert result.train_noise_std.tolist() == [[0.0, 0.0]]

    def test_trials_that_are_not_completed_never_make_the_noise_partial(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A pending, a failed and an abandoned trial have no observation, so no deviation: they
        # are left out before the noise is looked at, and must not be counted as unknown.
        problem = makeProblem()
        history = [
            pending(0, 0),
            completed(1, 1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.1, "cost": 1.0}),
            failed(2, 2),
            abandoned(3, 3),
        ]

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

        assert result.train_noise_std is not None
        assert result.train_noise_std.tolist() == [[0.1, 1.0]]
        assert warnings(caplog) == []


#: Histories whose noise is known for some of the values only, with how many of how many.
#: Every history has two outcomes, so `n` completed trials make `2n` values.
PARTIAL_HISTORIES = [
    pytest.param(
        [
            completed(0, 0, {"yield": 1.0, "cost": 10.0}, {"yield": 0.1, "cost": 1.0}),
            completed(1, 1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.2, "cost": 2.0}),
            completed(2, 2, {"yield": 3.0, "cost": 30.0}),
        ],
        "4 of 6",
        id="one-trial-without-noise",
    ),
    pytest.param(
        [
            completed(0, 0, {"yield": 1.0, "cost": 10.0}, {"yield": 0.1}),
            completed(1, 1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.2}),
            completed(2, 2, {"yield": 3.0, "cost": 30.0}, {"yield": 0.3}),
        ],
        "3 of 6",
        id="one-outcome-without-noise",
    ),
    pytest.param(
        [completed(0, 0, {"yield": 1.0, "cost": 10.0}, {"cost": 1.0})],
        "1 of 2",
        id="one-value-of-one-trial",
    ),
]


class TestPartialNoise:
    @pytest.mark.parametrize(("history", "counts"), PARTIAL_HISTORIES)
    def test_it_is_ignored_with_a_warning_by_default(
        self, history: Sequence[Trial], counts: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        problem = makeProblem()

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = trialsToTensors(problem, encoderOf(problem), history)

        assert result.train_noise_std is None
        [record] = warnings(caplog)
        assert record.levelno == logging.WARNING
        assert f"Only {counts} noise standard deviations are known" in record.getMessage()

    @pytest.mark.parametrize(("history", "counts"), PARTIAL_HISTORIES)
    def test_ignoring_it_is_what_the_explicit_rule_does_too(
        self, history: Sequence[Trial], counts: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        problem = makeProblem()

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            default = trialsToTensors(problem, encoderOf(problem), history)
            explicit = trialsToTensors(
                problem, encoderOf(problem), history, noise_policy=IGNORE
            )

        assert default.train_noise_std is None and explicit.train_noise_std is None
        assert len(warnings(caplog)) == 2
        assert torch.equal(default.train_outcomes, explicit.train_outcomes)

    @pytest.mark.parametrize(("history", "counts"), PARTIAL_HISTORIES)
    def test_the_error_rule_refuses_it_and_says_how_much_was_known(
        self, history: Sequence[Trial], counts: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        problem = makeProblem()

        with (
            caplog.at_level(logging.WARNING, logger=LOGGER),
            pytest.raises(SurrogateContextError) as caught,
        ):
            trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

        error = caught.value
        known, total = counts.split(" of ")
        assert f"only {known} of the {total} values" in error.message
        assert error.component == "noise policy"
        assert error.field_path == "partial"
        assert "ignore_with_warning" in (error.suggestion or "")
        assert warnings(caplog) == []  # it is refused, not worked around

    def test_the_error_is_a_planexs_error(self) -> None:
        assert issubclass(SurrogateContextError, PlanexsError)

    def test_known_noise_dropped_by_the_warning_is_dropped_for_every_value(self) -> None:
        # All the deviations go, the known ones included: a surrogate cannot mix the two.
        problem = makeProblem()
        history = PARTIAL_HISTORIES[0].values[0]

        result = trialsToTensors(problem, encoderOf(problem), history)

        assert result.train_noise_std is None
        assert result.train_outcomes.shape == (3, 2)


class TestNoiseNextToABinaryOutcome:
    """Only the values of outcomes that can have a deviation count towards what is known."""

    def test_complete_noise_on_the_continuous_outcome_is_complete(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        problem = mixedProblem()
        history = [
            completed(0, 0, {"yield": 1.5, "phase": False}, {"yield": 0.1}, problem),
            completed(1, 1, {"yield": 2.5, "phase": True}, {"yield": 0.2}, problem),
        ]

        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            result = trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

        assert result.train_noise_std is not None
        # `0.0` in the binary column is not a claim that the label is exact: it is never read.
        assert result.train_noise_std.tolist() == [[0.1, 0.0], [0.2, 0.0]]
        assert warnings(caplog) == []

    def test_partial_noise_on_the_continuous_outcome_counts_only_its_values(self) -> None:
        problem = mixedProblem()
        history = [
            completed(0, 0, {"yield": 1.5, "phase": False}, {"yield": 0.1}, problem),
            completed(1, 1, {"yield": 2.5, "phase": True}, None, problem),
        ]

        with pytest.raises(SurrogateContextError, match="only 1 of the 2 values"):
            trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

    def test_no_noise_on_the_continuous_outcome_is_not_partial(self) -> None:
        problem = mixedProblem()
        history = [completed(0, 0, {"yield": 1.5, "phase": True}, None, problem)]

        result = trialsToTensors(problem, encoderOf(problem), history, noise_policy=ERROR)

        assert result.train_noise_std is None


class TestPlacement:
    """Every tensor has the dtype and device that the runtime names, as a context requires."""

    @staticmethod
    def history() -> list[Trial]:
        return [
            completed(0, 0, {"yield": 1.0, "cost": 10.0}, {"yield": 0.5, "cost": 2.0}),
            completed(1, 1, {"yield": 2.0, "cost": 20.0}, {"yield": 0.25, "cost": 4.0}),
        ]

    def test_the_default_runtime_is_double_precision_on_the_cpu(self) -> None:
        problem = makeProblem()

        result = trialsToTensors(problem, encoderOf(problem), self.history())

        assert result.train_noise_std is not None
        for tensor in (result.train_inputs, result.train_outcomes, result.train_noise_std):
            assert tensor.dtype == torch.float64
            assert tensor.device.type == "cpu"

    @pytest.mark.parametrize("encoder_dtype", ["float32", "float64"])
    def test_the_runtime_decides_the_dtype_whatever_the_encoder_uses(
        self, encoder_dtype: str
    ) -> None:
        problem = makeProblem()
        encoder = encoderOf(problem, RuntimeConfig(dtype=encoder_dtype))  # type: ignore[arg-type]

        result = trialsToTensors(
            problem, encoder, self.history(), runtime=RuntimeConfig(dtype="float32")
        )

        assert result.train_noise_std is not None
        for tensor in (result.train_inputs, result.train_outcomes, result.train_noise_std):
            assert tensor.dtype == torch.float32

    def test_an_empty_history_has_the_dtype_of_the_runtime_too(self) -> None:
        problem = makeProblem()

        result = trialsToTensors(
            problem, encoderOf(problem), [], runtime=RuntimeConfig(dtype="float32")
        )

        assert result.train_inputs.dtype == result.train_outcomes.dtype == torch.float32

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_the_result_is_exactly_what_a_build_context_accepts(self, dtype: str) -> None:
        problem = makeProblem()
        runtime = RuntimeConfig(dtype=dtype)  # type: ignore[arg-type]
        encoder = encoderOf(problem, runtime)

        result = trialsToTensors(problem, encoder, self.history(), runtime=runtime)
        context = SurrogateBuildContext(
            problem=problem,
            encoder=encoder,
            train_inputs=result.train_inputs,
            train_outcomes=result.train_outcomes,
            train_noise_std=result.train_noise_std,
            generator=torchGenerator(0, runtime),
            runtime=runtime,
        )

        assert context.n_observations == 2
        assert context.train_noise_std is not None

    def test_a_device_that_torch_cannot_use_is_a_config_error(self) -> None:
        problem = makeProblem()

        with pytest.raises(ConfigError, match="device"):
            trialsToTensors(
                problem, encoderOf(problem), [], runtime=RuntimeConfig(device="notadevice")
            )


class TestErrors:
    def test_a_completed_trial_without_a_value_for_an_outcome_is_refused(self) -> None:
        problem = makeProblem()
        trial = Trial(
            id=3,
            parameters=POINTS[0],
            state=TrialState.completed,
            observation=Observation(values={"yield": 1.0}),
        )

        with pytest.raises(SurrogateContextError) as caught:
            trialsToTensors(problem, encoderOf(problem), [trial])

        error = caught.value
        assert "Trial 3" in error.message and "'cost'" in error.message
        assert error.component == "trial 3"
        assert error.field_path == "observation.values"

    def test_parameters_that_cannot_be_encoded_are_refused_with_the_trial(self) -> None:
        problem = makeProblem()
        wrong = Trial(
            id=8,
            parameters={"x": 1.0, "rate": 0.01, "solvent": "not-a-solvent"},
            state=TrialState.completed,
            observation=Observation(values={"yield": 1.0, "cost": 2.0}),
        )

        with pytest.raises(SurrogateContextError) as caught:
            trialsToTensors(problem, encoderOf(problem), [wrong])

        error = caught.value
        assert "Trial 8 cannot be encoded" in error.message
        assert error.component == "trial 8"
        assert error.field_path == "parameters"
        assert isinstance(error.__cause__, ValueError)

    def test_a_missing_parameter_is_refused_with_the_trial(self) -> None:
        problem = makeProblem()
        wrong = Trial(
            id=4,
            parameters={"x": 1.0},
            state=TrialState.completed,
            observation=Observation(values={"yield": 1.0, "cost": 2.0}),
        )

        with pytest.raises(SurrogateContextError, match="Trial 4 cannot be encoded"):
            trialsToTensors(problem, encoderOf(problem), [wrong])

    def test_a_trial_that_is_not_completed_is_never_checked(self) -> None:
        # Its parameters are wrong, but it is left out before anything reads them.
        problem = makeProblem()
        wrong = Trial(id=1, parameters={"x": 1.0})
        kept = completed(2, 1, {"yield": 2.0, "cost": 20.0})

        result = trialsToTensors(problem, encoderOf(problem), [wrong, kept])

        assert result.train_outcomes.tolist() == [[2.0, 20.0]]
