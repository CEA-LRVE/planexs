"""Unit tests for the `botorch_default_v1` kernel (roadmap task P3-10, specification 5.6).

The kernel makes no kernel: it leaves the choice to BoTorch. What is tested is that it does
exactly that, that the engine behaves as the documentation of the kernel says, and that a
surrogate can use it as it uses any other.
"""

from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from kernel_fakes import N_COLUMNS, fitAndPredict, threeColumnEncoder
from pydantic import ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.surrogates.kernels.BotorchDefaultKernelBuilder import BotorchDefaultKernelBuilder
from planexs.surrogates.kernels.BotorchDefaultKernelConfig import BotorchDefaultKernelConfig
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry


def builder() -> BotorchDefaultKernelBuilder:
    return BotorchDefaultKernelBuilder.fromConfig(BotorchDefaultKernelConfig())


class TestWhatItBuilds:
    def test_it_builds_no_kernel_on_a_three_column_encoder(self) -> None:
        assert builder().build(threeColumnEncoder().n_columns) is None

    @pytest.mark.parametrize("n_columns", [1, 3, 40])
    def test_it_builds_no_kernel_whatever_the_number_of_columns(self, n_columns: int) -> None:
        assert builder().build(n_columns) is None

    def test_the_registry_entry_makes_it(self) -> None:
        resolved = resolveComponent(kernelRegistry(), {"type": "botorch_default_v1"})

        made = resolved.entry.builder(resolved.config)

        assert isinstance(made, BotorchDefaultKernelBuilder)
        assert made.build(N_COLUMNS) is None


class TestItsConfiguration:
    def test_it_has_no_setting_but_its_type(self) -> None:
        assert set(BotorchDefaultKernelConfig.model_fields) == {"type"}
        assert toResolvedConfig(BotorchDefaultKernelConfig()) == {"type": "botorch_default_v1"}

    def test_a_setting_of_another_kernel_is_refused_with_its_name(self) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(kernelRegistry(), {"type": "botorch_default_v1", "ard": True})

        assert caught.value.field_path == "ard"
        assert "Unknown field" in caught.value.message

    def test_the_type_can_only_be_its_own_name(self) -> None:
        with pytest.raises(ValidationError, match="botorch_default_v1"):
            BotorchDefaultKernelConfig.model_validate({"type": "matern_v1"})


class TestTheEngineDefaultIsWhatTheDocumentationSays:
    """The description of the kernel says what BoTorch builds when it is given none. That is a
    fact about the installed version, so it is pinned: a BoTorch that changes it fails here,
    and the documentation is corrected, instead of going stale in silence.
    """

    def test_the_default_is_an_ard_rbf_kernel_with_a_lengthscale_prior_and_no_scale(
        self,
    ) -> None:
        model, _ = fitAndPredict(None)
        default = model.covar_module

        assert isinstance(default, RBFKernel)
        assert not isinstance(default, ScaleKernel)
        assert default.ard_num_dims == N_COLUMNS
        assert [name for name, *_ in default.named_priors()] == ["lengthscale_prior"]

    def test_the_default_has_hyperparameters_of_its_own_for_each_output(self) -> None:
        model, _ = fitAndPredict(None, n_outputs=2)

        assert tuple(model.covar_module.raw_lengthscale.shape) == (2, 1, N_COLUMNS)


class TestInASurrogate:
    def test_a_model_given_no_kernel_fits_and_predicts(self) -> None:
        kernel = builder().build(N_COLUMNS)

        model, mean = fitAndPredict(kernel)

        assert isinstance(model, SingleTaskGP)
        assert mean.shape == (20, 1) and bool(torch.isfinite(mean).all())

    def test_the_model_it_gives_is_the_one_that_botorch_builds_by_itself(self) -> None:
        # `None` is not a degraded mode: it is the same model as leaving the argument out.
        _, by_the_kernel = fitAndPredict(builder().build(N_COLUMNS))
        _, by_the_engine = fitAndPredict(None)

        assert torch.equal(by_the_kernel, by_the_engine)
