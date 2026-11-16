"""Unit tests for the `matern_v1` kernel (roadmap task P3-10, specification 5.6, 3.4, 10.8).

The tests of the card: the kernel builds on a three-column encoder, and an unsupported `nu` is
rejected with the field path. Around them: what is built (the plain GPyTorch kernel, with what
it leaves out on purpose) and what the set of accepted smoothness values is, which is a fact
about the engine and is checked against it.
"""

from __future__ import annotations

import math
from typing import Annotated, Any

import pytest
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from kernel_fakes import N_COLUMNS, fitAndPredict, threeColumnEncoder
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf
from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry
from planexs.surrogates.kernels.MaternKernelBuilder import MaternKernelBuilder
from planexs.surrogates.kernels.MaternKernelConfig import (
    SUPPORTED_SMOOTHNESS,
    MaternKernelConfig,
)


def build(n_columns: int = N_COLUMNS, **settings: Any) -> MaternKernel:
    kernel = MaternKernelBuilder.fromConfig(MaternKernelConfig(**settings)).build(n_columns)
    assert isinstance(kernel, MaternKernel)
    return kernel


class Holder(BaseModel):
    """What a surrogate's configuration will be: a model with a `kernel` component field."""

    model_config = ConfigDict(extra="forbid")
    kernel: Annotated[ComponentConfig, ComponentOf("kernel")] = Field(
        default_factory=lambda: {"type": "matern_v1"},  # type: ignore[arg-type,return-value]
        validate_default=True,
        description="The kernel.",
    )


class TestItBuildsOnAThreeColumnEncoder:
    def test_it_builds_a_matern_kernel_with_one_lengthscale_per_column_by_default(
        self,
    ) -> None:
        encoder = threeColumnEncoder()

        kernel = MaternKernelBuilder.fromConfig(MaternKernelConfig()).build(encoder.n_columns)

        assert isinstance(kernel, MaternKernel)
        assert kernel.nu == 2.5
        assert kernel.ard_num_dims == encoder.n_columns == 3
        assert tuple(kernel.lengthscale.shape) == (1, 3)

    def test_the_registry_entry_makes_it(self) -> None:
        resolved = resolveComponent(kernelRegistry(), {"type": "matern_v1", "nu": 1.5})

        kernel = resolved.entry.builder(resolved.config).build(N_COLUMNS)

        assert isinstance(kernel, MaternKernel) and kernel.nu == 1.5

    @pytest.mark.parametrize("n_columns", [1, 3, 40])
    def test_each_column_has_a_lengthscale_of_its_own(self, n_columns: int) -> None:
        assert tuple(build(n_columns).lengthscale.shape) == (1, n_columns)

    @pytest.mark.parametrize("n_columns", [1, 3, 40])
    def test_without_ard_one_lengthscale_is_shared(self, n_columns: int) -> None:
        kernel = build(n_columns, ard=False)

        assert kernel.ard_num_dims is None
        assert tuple(kernel.lengthscale.shape) == (1, 1)


class TestTheSmoothness:
    @pytest.mark.parametrize("nu", SUPPORTED_SMOOTHNESS)
    def test_every_supported_value_is_kept_by_the_kernel(self, nu: float) -> None:
        assert build(nu=nu).nu == nu

    def test_the_default_is_the_matern_five_halves_kernel(self) -> None:
        assert MaternKernelConfig().nu == 2.5

    def test_the_smoothness_changes_the_covariance(self) -> None:
        # Not a label: three different kernels, giving three different covariance matrices.
        points = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1).repeat(1, 3)
        matrices = [build(nu=nu).double()(points).to_dense() for nu in SUPPORTED_SMOOTHNESS]

        assert not torch.allclose(matrices[0], matrices[1])
        assert not torch.allclose(matrices[1], matrices[2])
        for matrix in matrices:
            assert bool(torch.isfinite(matrix).all())
            assert torch.allclose(matrix, matrix.T)

    def test_the_accepted_set_is_exactly_the_set_the_installed_gpytorch_accepts(self) -> None:
        # `SUPPORTED_SMOOTHNESS` is a fact about the engine (specification 10.9), pinned
        # here: a GPyTorch that supports another value, or drops one, fails this test.
        candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0]

        accepted = []
        for nu in candidates:
            try:
                MaternKernel(nu=nu)
            except RuntimeError:
                continue
            accepted.append(nu)

        assert tuple(accepted) == SUPPORTED_SMOOTHNESS


class TestAnUnsupportedNuIsRejectedWithTheFieldPath:
    @pytest.mark.parametrize(
        "nu", [0.0, 1.0, 2.0, 3.5, 5.0, -0.5, 2, 3, math.nan, math.inf, True], ids=repr
    )
    def test_the_field_path_is_nu(self, nu: object) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(kernelRegistry(), {"type": "matern_v1", "nu": nu})

        error = caught.value
        assert error.field_path == "nu"
        assert error.component == "kernel 'matern_v1'"
        assert "nu must be one of 0.5, 1.5, 2.5" in error.message

    def test_the_message_names_the_value_that_was_given(self) -> None:
        with pytest.raises(ConfigError, match=r"\(got 3\.5\)"):
            resolveComponent(kernelRegistry(), {"type": "matern_v1", "nu": 3.5})

    def test_the_path_is_the_full_path_when_the_kernel_is_nested(self) -> None:
        # How a surrogate holds a kernel: the field path starts at the surrogate's own field.
        with pytest.raises(ConfigError) as caught:
            validateConfig(Holder, {"kernel": {"type": "matern_v1", "nu": 3.5}})

        assert caught.value.field_path == "kernel.nu"

    def test_a_value_that_is_not_a_number_is_refused_with_the_same_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(kernelRegistry(), {"type": "matern_v1", "nu": "smooth"})

        assert caught.value.field_path == "nu"

    def test_the_schema_refuses_it_directly_too(self) -> None:
        with pytest.raises(ValidationError, match="nu must be one of"):
            MaternKernelConfig(nu=3.5)

    def test_a_number_written_as_text_is_read_as_the_number_as_configurations_do(self) -> None:
        # A YAML or JSON file may quote it: the same meaning in Python, JSON and YAML.
        config = MaternKernelConfig.model_validate({"type": "matern_v1", "nu": "1.5"})

        assert config.nu == 1.5


class TestItsConfiguration:
    def test_the_fields_and_their_defaults(self) -> None:
        assert toResolvedConfig(MaternKernelConfig()) == {
            "type": "matern_v1",
            "nu": 2.5,
            "ard": True,
        }

    def test_an_unknown_field_is_refused_with_its_name(self) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(kernelRegistry(), {"type": "matern_v1", "lengthscale": 1.0})

        assert caught.value.field_path == "lengthscale"

    def test_resolving_and_validating_again_gives_the_same_configuration(self) -> None:
        config = MaternKernelConfig(nu=0.5, ard=False)

        resolved = toResolvedConfig(config)

        assert validateConfig(MaternKernelConfig, resolved) == config

    def test_the_nested_default_of_a_holder_is_matern_when_it_says_so(self) -> None:
        assert validateConfig(Holder, {}).kernel == MaternKernelConfig()


class TestWhatItBuildsAndWhatItLeavesOut:
    def test_every_call_makes_a_new_kernel_that_shares_nothing(self) -> None:
        builder = MaternKernelBuilder.fromConfig(MaternKernelConfig())

        first, second = builder.build(N_COLUMNS), builder.build(N_COLUMNS)

        assert first is not second
        assert isinstance(first, MaternKernel) and isinstance(second, MaternKernel)
        first.lengthscale = 5.0
        assert not torch.allclose(first.lengthscale, second.lengthscale)

    def test_the_same_configuration_gives_the_same_initial_kernel(self) -> None:
        # No random number is drawn: nothing to seed (specification 6).
        a, b = build(nu=1.5), build(nu=1.5)

        assert torch.equal(a.raw_lengthscale, b.raw_lengthscale)

    def test_it_carries_no_prior_so_a_map_fit_of_it_is_a_maximum_likelihood_fit(self) -> None:
        assert list(build().named_priors()) == []

    def test_it_is_not_wrapped_in_an_output_scale(self) -> None:
        assert not isinstance(build(), ScaleKernel)


class TestInASurrogate:
    @pytest.mark.parametrize("nu", SUPPORTED_SMOOTHNESS)
    @pytest.mark.parametrize("ard", [True, False])
    def test_a_model_given_the_kernel_fits_and_predicts(self, nu: float, ard: bool) -> None:
        model, mean = fitAndPredict(build(nu=nu, ard=ard))

        assert mean.shape == (20, 1) and bool(torch.isfinite(mean).all())
        assert isinstance(model.covar_module, MaternKernel)
        assert model.covar_module.nu == nu

    def test_the_model_converts_the_kernel_to_its_own_precision(self) -> None:
        # The kernel is made in single precision, as GPyTorch makes it, and the model of a
        # study that works in double precision moves it to its own dtype: the builder needs
        # no dtype.
        assert build().raw_lengthscale.dtype == torch.float32

        model, _ = fitAndPredict(build(), dtype=torch.float64)

        assert model.covar_module.raw_lengthscale.dtype == torch.float64

    def test_one_batched_model_of_several_outputs_cannot_be_fitted_with_it(self) -> None:
        # The contract of the kind gives a kernel `n_columns` and nothing else, so it has no
        # batch shape, and BoTorch cannot fit one batched model of several outputs with it:
        # pinned so that it is not discovered later, by the surrogate that needs it (roadmap
        # P3-14). BoTorch's own default kernel is built with the batch shape and has no such
        # limit (see `test_botorch_default_kernel.py`). If a BoTorch release lifts it, this
        # test fails: correct the notes of `MaternKernelBuilder` and of the contract then.
        with pytest.raises(RuntimeError, match="invalid for input of size"):
            fitAndPredict(build(), n_outputs=2)

    def test_one_model_for_each_outcome_with_a_kernel_of_its_own_is_the_way(self) -> None:
        builder = MaternKernelBuilder.fromConfig(MaternKernelConfig())

        fitted = [fitAndPredict(builder.build(N_COLUMNS)) for _ in range(2)]

        (first, first_mean), (second, second_mean) = fitted
        assert first.covar_module is not second.covar_module
        for mean in (first_mean, second_mean):
            assert mean.shape == (20, 1) and bool(torch.isfinite(mean).all())
