"""Unit tests for `AbstractKernelBuilder`, the contract of the `kernel` kind (roadmap task
P3-10, specification 5.6, 6).
"""

from __future__ import annotations

from typing import Any

import pytest
from gpytorch.kernels import Kernel
from kernel_fakes import RecordingKernelBuilder, fakeConfig

from planexs.core.ConfigError import ConfigError
from planexs.surrogates.kernels.AbstractKernelBuilder import AbstractKernelBuilder
from planexs.surrogates.kernels.BotorchDefaultKernelBuilder import BotorchDefaultKernelBuilder
from planexs.surrogates.kernels.MaternKernelBuilder import MaternKernelBuilder
from planexs.surrogates.kernels.RBFKernelBuilder import RBFKernelBuilder

CONFIG = fakeConfig("fake_kernel_v1")


class NoKernelBuilder(RecordingKernelBuilder):
    """A kernel that leaves the choice to the engine."""

    def makeKernel(self, n_columns: int) -> Kernel | None:
        self.received.append(n_columns)
        return None


class TestTheContract:
    def test_the_contract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractKernelBuilder(CONFIG())  # type: ignore[abstract]

    def test_a_kernel_that_does_not_define_makeKernel_cannot_be_instantiated(self) -> None:
        class Incomplete(AbstractKernelBuilder[Any]):
            pass

        with pytest.raises(TypeError, match="makeKernel"):
            Incomplete(CONFIG())  # type: ignore[abstract]

    def test_fromConfig_makes_an_instance_of_the_class_it_is_called_on(self) -> None:
        config = CONFIG(scale=2.0)

        builder = RecordingKernelBuilder.fromConfig(config)

        assert type(builder) is RecordingKernelBuilder
        assert builder.config is config

    def test_the_configuration_is_a_read_only_attribute(self) -> None:
        builder = RecordingKernelBuilder(CONFIG())

        with pytest.raises(AttributeError):
            builder.config = CONFIG(scale=3.0)  # type: ignore[misc]

    @pytest.mark.parametrize(
        "builder", [BotorchDefaultKernelBuilder, MaternKernelBuilder, RBFKernelBuilder]
    )
    def test_no_real_kernel_overrides_the_entry_point(self, builder: type) -> None:
        # `build` checks `n_columns` for every kernel in the same way: it is not overridden.
        assert builder.build is AbstractKernelBuilder.build


class TestWhatBuildDoes:
    def test_it_gives_makeKernel_the_number_of_columns_and_returns_its_kernel(self) -> None:
        builder = RecordingKernelBuilder(CONFIG())

        kernel = builder.build(3)

        assert isinstance(kernel, Kernel)
        assert builder.received == [3]

    def test_no_kernel_is_a_legal_answer_and_is_passed_on_as_it_is(self) -> None:
        builder = NoKernelBuilder(CONFIG())

        assert builder.build(3) is None
        assert builder.received == [3]

    def test_every_call_makes_a_new_kernel(self) -> None:
        builder = RecordingKernelBuilder(CONFIG())

        first, second = builder.build(3), builder.build(3)

        assert first is not second

    @pytest.mark.parametrize("n_columns", [1, 3, 40])
    def test_any_number_of_columns_of_one_or_more_is_accepted(self, n_columns: int) -> None:
        builder = RecordingKernelBuilder(CONFIG())

        builder.build(n_columns)

        assert builder.received == [n_columns]


class TestABadNumberOfColumns:
    @pytest.mark.parametrize("bad", [0, -1, -40, True, False, 2.5, 3.0, "3", None])
    def test_it_is_refused_before_the_kernel_is_made(self, bad: Any) -> None:
        builder = RecordingKernelBuilder(CONFIG())

        with pytest.raises(ConfigError) as caught:
            builder.build(bad)

        error = caught.value
        assert "whole number of one or more" in error.message
        assert repr(bad) in error.message
        assert error.component == "kernel 'fake_kernel_v1'"
        assert error.field_path == "n_columns"
        assert "n_columns of the encoder" in (error.suggestion or "")
        assert builder.received == []  # nothing was made
