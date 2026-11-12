"""Unit tests for `SurrogateBuildContext` and `SurrogateContextError` (roadmap task P3-07,
specification 5.6, 6).

The context is "the read-only information a builder receives". Three things are tested: that a
valid one holds what it was given, that it is read-only, and that every way in which its data
can be malformed is refused when it is built, with the name of the field and a way to fix it
(specification 3.6), and not deep inside an engine call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
import torch
from surrogate_fakes import N_COLUMNS, OUTCOME_NAMES, makeContext, makeProblem, makeSpace

from planexs.core.ConfigError import ConfigError
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchGenerator
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext, _isOnDevice
from planexs.surrogates.SurrogateContextError import SurrogateContextError

FIELDS = [
    "problem",
    "encoder",
    "train_inputs",
    "train_outcomes",
    "generator",
    "runtime",
    "train_noise_std",
]

FLOAT32 = RuntimeConfig(dtype="float32")


def tensors(runtime: RuntimeConfig | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The inputs and outcomes of a valid context, to be modified into invalid ones."""
    context = makeContext(runtime)
    return context.train_inputs.clone(), context.train_outcomes.clone()


class TestAValidContext:
    def test_it_holds_what_it_was_given(self) -> None:
        problem = makeProblem()
        encoder = SearchSpaceEncoder(problem.search_space)
        inputs = torch.zeros(2, N_COLUMNS, dtype=torch.double)
        outcomes = torch.ones(2, 2, dtype=torch.double)
        noise = torch.full((2, 2), 0.1, dtype=torch.double)
        generator = torchGenerator(3, RuntimeConfig())
        runtime = RuntimeConfig()

        context = SurrogateBuildContext(
            problem=problem,
            encoder=encoder,
            train_inputs=inputs,
            train_outcomes=outcomes,
            train_noise_std=noise,
            generator=generator,
            runtime=runtime,
        )

        assert context.problem is problem
        assert context.encoder is encoder
        assert context.train_inputs is inputs
        assert context.train_outcomes is outcomes
        assert context.train_noise_std is noise
        assert context.generator is generator
        assert context.runtime is runtime

    def test_it_reports_the_number_of_observations_and_the_outcome_names(self) -> None:
        context = makeContext()

        assert context.n_observations == 4
        assert context.outcome_names == OUTCOME_NAMES

    def test_the_noise_is_optional(self) -> None:
        assert makeContext().train_noise_std is None

    def test_a_noise_of_zero_is_a_known_noise(self) -> None:
        # A deterministic observation has a standard deviation of zero: it is not an error.
        zeros = torch.zeros(4, 2, dtype=torch.double)

        assert makeContext(train_noise_std=zeros).train_noise_std is zeros

    def test_an_empty_history_is_legal(self) -> None:
        # Whether it is enough to fit a model is the builder's call (ModelNotReadyError).
        context = makeContext(
            train_inputs=torch.empty(0, N_COLUMNS, dtype=torch.double),
            train_outcomes=torch.empty(0, 2, dtype=torch.double),
        )

        assert context.n_observations == 0

    def test_the_inputs_may_touch_the_faces_of_the_unit_cube(self) -> None:
        inputs = torch.tensor([[0.0] * N_COLUMNS, [1.0] * N_COLUMNS], dtype=torch.double)
        outcomes = torch.zeros(2, 2, dtype=torch.double)

        assert makeContext(train_inputs=inputs, train_outcomes=outcomes).n_observations == 2

    def test_a_single_precision_runtime_takes_single_precision_tensors(self) -> None:
        context = makeContext(FLOAT32, train_noise_std=torch.zeros(4, 2, dtype=torch.float32))

        assert context.train_inputs.dtype == torch.float32
        assert context.train_outcomes.dtype == torch.float32
        assert context.train_noise_std is not None
        assert context.train_noise_std.dtype == torch.float32

    def test_it_is_built_with_keywords(self) -> None:
        context = makeContext()

        with pytest.raises(TypeError):
            SurrogateBuildContext(  # type: ignore[misc]
                context.problem,
                context.encoder,
                context.train_inputs,
                context.train_outcomes,
                context.generator,
                context.runtime,
            )

    def test_a_missing_field_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="train_outcomes"):
            SurrogateBuildContext(  # type: ignore[call-arg]
                problem=makeProblem(),
                encoder=SearchSpaceEncoder(makeSpace()),
                train_inputs=torch.zeros(1, N_COLUMNS, dtype=torch.double),
                generator=torchGenerator(1, RuntimeConfig()),
                runtime=RuntimeConfig(),
            )

    def test_contexts_are_compared_by_identity(self) -> None:
        # Equality of tensors is a tensor, not a boolean: a dataclass `==` would raise.
        first = makeContext()
        second = makeContext()

        assert first != second
        assert first == first
        assert len({first, second}) == 2

    def test_the_generator_is_the_one_member_that_advances(self) -> None:
        context = makeContext()
        generator = context.generator

        first = torch.rand(1, generator=generator)
        second = torch.rand(1, generator=generator)

        assert context.generator is generator
        assert not torch.equal(first, second)


class TestTheContextIsReadOnly:
    @pytest.mark.parametrize("field", FIELDS)
    def test_no_field_can_be_assigned(self, field: str) -> None:
        context = makeContext()

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(context, field, None)

    def test_no_field_can_be_added(self) -> None:
        context = makeContext()

        with pytest.raises(dataclasses.FrozenInstanceError):
            context.extra = 1  # type: ignore[attr-defined]

    @pytest.mark.parametrize("field", FIELDS)
    def test_no_field_can_be_deleted(self, field: str) -> None:
        context = makeContext()

        with pytest.raises(dataclasses.FrozenInstanceError):
            delattr(context, field)

    @pytest.mark.parametrize("name", ["n_observations", "outcome_names"])
    def test_the_derived_properties_cannot_be_assigned(self, name: str) -> None:
        context = makeContext()

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            setattr(context, name, 1)

    def test_the_tensors_are_shared_and_not_copied(self) -> None:
        # Documented: a frozen dataclass cannot freeze a tensor, so a builder must not
        # modify it in place. Sharing is what makes the context cheap.
        inputs, outcomes = tensors()
        context = makeContext(train_inputs=inputs, train_outcomes=outcomes)

        assert context.train_inputs is inputs
        assert context.train_outcomes is outcomes


def withInputs(change: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
    """Overrides for a context whose inputs are a changed copy of the valid ones."""
    inputs, _ = tensors()
    return {"train_inputs": change(inputs)}


def withOutcomes(change: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
    """Overrides for a context whose outcomes are a changed copy of the valid ones."""
    _, outcomes = tensors()
    return {"train_outcomes": change(outcomes)}


def withNoise(noise: torch.Tensor) -> dict[str, Any]:
    """Overrides for a context with the given noise."""
    return {"train_noise_std": noise}


def nan(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor with its first value replaced by `nan`."""
    tensor[0, 0] = float("nan")
    return tensor


def infinity(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor with its first value replaced by an infinity."""
    tensor[0, 0] = float("inf")
    return tensor


def ones(rows: int = 4, columns: int = 2) -> torch.Tensor:
    """A tensor of ones, of the dtype of the default runtime."""
    return torch.ones(rows, columns, dtype=torch.double)


#: One malformed context per rule: the overrides that break it, the field the error names and
#: a sentence it contains. The runtime is the default one unless `runtime` is given.
MALFORMED: list[tuple[dict[str, Any], str, str]] = [
    # Shapes.
    (withInputs(lambda t: t[:, 0]), "train_inputs", "must have shape (n, 5)"),
    (withInputs(lambda t: t[:, :-1]), "train_inputs", "must have shape (n, 5)"),
    (withInputs(lambda t: t.unsqueeze(0)), "train_inputs", "must have shape (n, 5)"),
    (withOutcomes(lambda t: t[:-1]), "train_outcomes", "must have shape (4, 2)"),
    (withOutcomes(lambda t: t[:, :1]), "train_outcomes", "must have shape (4, 2)"),
    (withOutcomes(lambda t: t[:, 0]), "train_outcomes", "must have shape (4, 2)"),
    (withNoise(ones(3)), "train_noise_std", "must have the shape of train_outcomes"),
    (withNoise(ones(4, 1)), "train_noise_std", "must have the shape of train_outcomes"),
    # Precision: the runtime says float64.
    (withInputs(lambda t: t.float()), "train_inputs", "dtype torch.float32"),
    (withOutcomes(lambda t: t.float()), "train_outcomes", "dtype torch.float32"),
    (withNoise(ones().float()), "train_noise_std", "dtype torch.float32"),
    # Device: the runtime says cpu, and a tensor on the meta device is not.
    (withInputs(lambda t: t.to("meta")), "train_inputs", "device meta"),
    (withOutcomes(lambda t: t.to("meta")), "train_outcomes", "device meta"),
    (withNoise(ones().to("meta")), "train_noise_std", "device meta"),
    # Values.
    (withInputs(nan), "train_inputs", "not finite"),
    (withInputs(infinity), "train_inputs", "not finite"),
    (withOutcomes(nan), "train_outcomes", "not finite"),
    (withOutcomes(infinity), "train_outcomes", "not finite"),
    (withNoise(nan(ones())), "train_noise_std", "not finite"),
    (withNoise(infinity(ones())), "train_noise_std", "not finite"),
    (withNoise(-ones()), "train_noise_std", "negative value"),
    (withInputs(lambda t: t - 0.5), "train_inputs", "outside [0, 1]"),
    (withInputs(lambda t: t + 0.5), "train_inputs", "outside [0, 1]"),
]

IDS = [
    "inputs-1d",
    "inputs-too-few-columns",
    "inputs-3d",
    "outcomes-too-few-rows",
    "outcomes-too-few-columns",
    "outcomes-1d",
    "noise-too-few-rows",
    "noise-too-few-columns",
    "inputs-float32",
    "outcomes-float32",
    "noise-float32",
    "inputs-meta-device",
    "outcomes-meta-device",
    "noise-meta-device",
    "inputs-nan",
    "inputs-infinity",
    "outcomes-nan",
    "outcomes-infinity",
    "noise-nan",
    "noise-infinity",
    "noise-negative",
    "inputs-below-zero",
    "inputs-above-one",
]


class TestAMalformedContextIsRefused:
    @pytest.mark.parametrize(("overrides", "field", "sentence"), MALFORMED, ids=IDS)
    def test_each_rule_is_enforced_and_names_the_field(
        self, overrides: dict[str, Any], field: str, sentence: str
    ) -> None:
        with pytest.raises(SurrogateContextError) as caught:
            makeContext(**overrides)

        error = caught.value
        assert error.field_path == field
        assert error.component == "surrogate build context"
        assert sentence in error.message
        assert error.suggestion

    def test_an_encoder_that_encodes_a_parameter_the_problem_lacks_is_refused(self) -> None:
        # The encoder is built for a space with one more parameter than the problem has.
        other = SearchSpaceEncoder(makeSpace("elsewhere"))

        with pytest.raises(SurrogateContextError) as caught:
            makeContext(encoder=other)

        assert caught.value.field_path == "encoder"
        assert "'elsewhere'" in caught.value.message
        assert "another search space" in caught.value.message

    def test_an_encoder_of_a_space_with_fewer_parameters_is_refused_by_its_shape(self) -> None:
        # No parameter of the problem is missing from the *encoder's* point of view, but the
        # inputs no longer fit its columns: the first rule that fails is reported.
        smaller = SearchSpaceEncoder(
            makeSpace().model_copy(update={"parameters": makeSpace().parameters[:2]})
        )

        with pytest.raises(SurrogateContextError) as caught:
            makeContext(encoder=smaller)

        assert caught.value.field_path == "train_inputs"

    def test_the_error_is_a_planexs_error_with_a_rendering_that_says_how_to_fix_it(
        self,
    ) -> None:
        inputs, _ = tensors()

        with pytest.raises(PlanexsError) as caught:
            makeContext(train_inputs=inputs + 5.0)

        text = str(caught.value)
        assert isinstance(caught.value, SurrogateContextError)
        assert "Component: surrogate build context" in text
        assert "Field: train_inputs" in text
        assert "Suggestion: encode each parameterization with" in text

    def test_a_double_precision_tensor_is_refused_by_a_single_precision_runtime(self) -> None:
        inputs, outcomes = tensors()  # float64, as the default runtime says

        with pytest.raises(SurrogateContextError, match="'float32'") as caught:
            makeContext(FLOAT32, train_inputs=inputs, train_outcomes=outcomes)

        assert caught.value.field_path == "train_inputs"
        assert "convert it with .to(torch.float32)" in str(caught.value)

    def test_the_noise_is_judged_like_the_other_tensors(self) -> None:
        with pytest.raises(SurrogateContextError) as caught:
            makeContext(train_noise_std=ones(4, 2).to(torch.float32))

        assert caught.value.field_path == "train_noise_std"


class TestTheFirstFaultIsReportedInASensibleOrder:
    def test_shapes_are_judged_before_values(self) -> None:
        inputs, _ = tensors()
        inputs = nan(inputs)[:, :-1]  # wrong shape and not finite

        with pytest.raises(SurrogateContextError) as caught:
            makeContext(train_inputs=inputs)

        assert "must have shape" in caught.value.message

    def test_the_precision_is_judged_before_the_values(self) -> None:
        inputs, _ = tensors()
        inputs = nan(inputs).float()  # wrong dtype and not finite

        with pytest.raises(SurrogateContextError) as caught:
            makeContext(train_inputs=inputs)

        assert "dtype" in caught.value.message

    def test_the_device_is_judged_before_the_values(self) -> None:
        # A value check on a meta tensor could not even be computed.
        inputs, _ = tensors()

        with pytest.raises(SurrogateContextError, match="device meta"):
            makeContext(train_inputs=inputs.to("meta"))

    def test_non_finite_values_are_judged_before_the_unit_cube(self) -> None:
        inputs, _ = tensors()
        inputs[0, 0] = float("inf")  # also outside the cube

        with pytest.raises(SurrogateContextError, match="not finite"):
            makeContext(train_inputs=inputs)


class TestARuntimeThatCannotBeUsedIsAConfigError:
    def test_a_device_type_that_torch_does_not_know(self) -> None:
        # The syntax of the name is valid, so `RuntimeConfig` accepts it. Only torch knows that
        # no such device exists, and the error is the one of `torchDevice`.
        runtime = RuntimeConfig(device="nonexistent")
        inputs, outcomes = tensors()

        with pytest.raises(ConfigError, match="'nonexistent'") as caught:
            makeContext(runtime, train_inputs=inputs, train_outcomes=outcomes)

        assert caught.value.field_path == "device"


class TestWhereATensorLives:
    """`_isOnDevice` judges a tensor's device against the runtime's, by type and, when both
    have one, by index. Device objects are only descriptors, so no GPU is needed.
    """

    @pytest.mark.parametrize(
        ("actual", "expected", "result"),
        [
            ("cpu", "cpu", True),
            ("cuda:0", "cuda", True),  # the runtime says "any CUDA device"
            ("cuda:1", "cuda", True),
            ("cuda:0", "cuda:0", True),
            ("cuda:1", "cuda:0", False),  # two different devices
            ("cuda:0", "cuda:1", False),
            ("cpu", "cuda", False),
            ("cuda:0", "cpu", False),
            ("cpu", "cpu:0", True),  # a CPU tensor reports no index at all
            ("meta", "cpu", False),
        ],
    )
    def test_the_type_must_match_and_the_index_only_when_both_have_one(
        self, actual: str, expected: str, result: bool
    ) -> None:
        assert _isOnDevice(torch.device(actual), torch.device(expected)) is result

    def test_a_runtime_that_names_the_cpu_with_an_index_takes_cpu_tensors(self) -> None:
        # The false positive that the index rule exists to avoid.
        context = makeContext(RuntimeConfig(device="cpu:0"))

        assert context.train_inputs.device == torch.device("cpu")
