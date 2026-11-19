"""Unit tests for `LikelihoodBuildContext` (roadmap task P3-11, specification 5.6, 6)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from likelihood_fakes import makeContext

from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.surrogates.likelihoods.LikelihoodBuildContext import LikelihoodBuildContext
from planexs.surrogates.SurrogateContextError import SurrogateContextError

RUNTIME = RuntimeConfig()


def build(**overrides: object) -> LikelihoodBuildContext:
    """A context with one continuous outcome and the default runtime, with replacements."""
    values: dict[str, object] = {
        "outcomes": (Outcome(name="y", type="continuous"),),
        "runtime": RUNTIME,
    }
    values.update(overrides)
    return LikelihoodBuildContext(**values)  # type: ignore[arg-type]


class TestAContext:
    def test_it_holds_what_it_is_given(self) -> None:
        noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.double)
        context = makeContext("continuous", "continuous", noise=noise, batch_shape=(2,))

        assert [outcome.name for outcome in context.outcomes] == ["y0", "y1"]
        assert context.runtime == RUNTIME
        assert context.batch_shape == torch.Size([2])
        assert context.noise_variance is noise

    def test_the_noise_is_inferred_and_the_batch_is_one_model_by_default(self) -> None:
        context = build()

        assert context.noise_variance is None
        assert context.batch_shape == torch.Size()

    def test_it_stores_a_sequence_of_outcomes_as_a_tuple_and_a_shape_as_a_size(self) -> None:
        context = build(outcomes=[Outcome(name="y", type="continuous")], batch_shape=(3,))

        assert isinstance(context.outcomes, tuple)
        assert isinstance(context.batch_shape, torch.Size)

    @pytest.mark.parametrize("field", ["outcomes", "runtime", "batch_shape", "noise_variance"])
    def test_no_field_can_be_assigned(self, field: str) -> None:
        context = build()

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(context, field, None)

    def test_it_is_built_with_keywords(self) -> None:
        with pytest.raises(TypeError):
            LikelihoodBuildContext((Outcome(name="y", type="continuous"),), RUNTIME)  # type: ignore[misc]

    def test_it_accepts_a_zero_variance(self) -> None:
        # A deterministic observation: the likelihood decides what to do with it.
        context = makeContext(noise=[0.0, 0.5])

        assert context.noise_variance is not None


class TestTheOutcomes:
    def test_there_must_be_at_least_one(self) -> None:
        with pytest.raises(SurrogateContextError, match="at least one outcome") as error:
            build(outcomes=())

        assert error.value.field_path == "outcomes"
        assert error.value.component == "likelihood build context"

    def test_each_is_named_once(self) -> None:
        twice = (Outcome(name="y", type="continuous"), Outcome(name="y", type="binary"))

        with pytest.raises(SurrogateContextError, match="more than once") as error:
            build(outcomes=twice)

        assert error.value.field_path == "outcomes"


class TestTheNoise:
    def test_it_has_the_precision_of_the_runtime(self) -> None:
        single = torch.tensor([0.1, 0.2], dtype=torch.float32)

        with pytest.raises(SurrogateContextError, match="dtype") as error:
            build(noise_variance=single)

        assert error.value.field_path == "noise_variance"
        assert error.value.suggestion is not None

    def test_it_is_on_the_device_of_the_runtime(self) -> None:
        elsewhere = torch.empty(3, dtype=torch.double, device="meta")

        with pytest.raises(SurrogateContextError, match="device") as error:
            build(noise_variance=elsewhere)

        assert error.value.field_path == "noise_variance"

    @pytest.mark.parametrize(
        ("batch_shape", "values"),
        [
            ((), [[0.1, 0.2]]),  # one axis too many
            ((2,), [0.1, 0.2, 0.3]),  # one axis too few
            ((2,), [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),  # the batch has 3 models, not 2
        ],
    )
    def test_its_shape_is_the_batch_shape_and_the_observations(
        self, batch_shape: tuple[int, ...], values: list[list[float]] | list[float]
    ) -> None:
        noise = torch.tensor(values, dtype=torch.double)

        with pytest.raises(SurrogateContextError, match="must have the shape") as error:
            build(noise_variance=noise, batch_shape=batch_shape)

        assert error.value.field_path == "noise_variance"
        assert "'n'" in error.value.message

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_it_is_finite(self, bad: float) -> None:
        with pytest.raises(SurrogateContextError, match="not finite"):
            build(noise_variance=torch.tensor([0.1, bad], dtype=torch.double))

    def test_a_variance_is_not_negative(self) -> None:
        with pytest.raises(SurrogateContextError, match="negative") as error:
            build(noise_variance=torch.tensor([0.1, -0.2], dtype=torch.double))

        assert error.value.suggestion is not None
        assert "square" in error.value.suggestion

    def test_a_batch_of_models_takes_one_row_of_variances_each(self) -> None:
        context = build(
            noise_variance=torch.ones(2, 5, dtype=torch.double), batch_shape=torch.Size([2])
        )

        assert context.noise_variance is not None
        assert tuple(context.noise_variance.shape) == (2, 5)

    def test_single_precision_noise_goes_with_single_precision_runtime(self) -> None:
        runtime = RuntimeConfig(dtype="float32")

        context = makeContext(noise=[0.1, 0.2], runtime=runtime)

        assert context.noise_variance is not None
        assert context.noise_variance.dtype == torch.float32
