"""Unit tests for the `rbf_v1` kernel (roadmap task P3-10, specification 5.6).

The kernel is the plain `gpytorch.kernels.RBFKernel`, with or without one lengthscale for each
encoded column. It has the notes of the Matern kernel (no prior, no output scale, no batch
shape), which `test_matern_kernel.py` tests at length: the tests here are those that are about
the RBF kernel, and the few that make sure the notes hold for it too.
"""

from __future__ import annotations

import pytest
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from kernel_fakes import N_COLUMNS, fitAndPredict, threeColumnEncoder

from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry
from planexs.surrogates.kernels.RBFKernelBuilder import RBFKernelBuilder
from planexs.surrogates.kernels.RBFKernelConfig import RBFKernelConfig


def build(n_columns: int = N_COLUMNS, **settings: bool) -> RBFKernel:
    kernel = RBFKernelBuilder.fromConfig(RBFKernelConfig(**settings)).build(n_columns)
    assert isinstance(kernel, RBFKernel)
    return kernel


class TestItBuildsOnAThreeColumnEncoder:
    def test_it_builds_an_rbf_kernel_with_one_lengthscale_per_column_by_default(self) -> None:
        encoder = threeColumnEncoder()

        kernel = RBFKernelBuilder.fromConfig(RBFKernelConfig()).build(encoder.n_columns)

        assert isinstance(kernel, RBFKernel)
        assert kernel.ard_num_dims == encoder.n_columns == 3
        assert tuple(kernel.lengthscale.shape) == (1, 3)

    def test_the_registry_entry_makes_it(self) -> None:
        resolved = resolveComponent(kernelRegistry(), {"type": "rbf_v1", "ard": False})

        kernel = resolved.entry.builder(resolved.config).build(N_COLUMNS)

        assert isinstance(kernel, RBFKernel) and kernel.ard_num_dims is None

    @pytest.mark.parametrize("n_columns", [1, 3, 40])
    def test_each_column_has_a_lengthscale_of_its_own(self, n_columns: int) -> None:
        assert tuple(build(n_columns).lengthscale.shape) == (1, n_columns)

    @pytest.mark.parametrize("n_columns", [1, 3, 40])
    def test_without_ard_one_lengthscale_is_shared(self, n_columns: int) -> None:
        kernel = build(n_columns, ard=False)

        assert kernel.ard_num_dims is None
        assert tuple(kernel.lengthscale.shape) == (1, 1)

    def test_the_lengthscales_count_the_encoded_columns_and_not_the_parameters(self) -> None:
        # ARD is per column of the encoding: a one-hot block has a lengthscale for each level.
        from planexs import Categorical, Real, SearchSpace
        from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder

        space = SearchSpace([Real("x", 0.0, 1.0), Categorical("kind", ["a", "b", "c"])])
        encoder = SearchSpaceEncoder(space)

        kernel = RBFKernelBuilder.fromConfig(RBFKernelConfig()).build(encoder.n_columns)

        assert isinstance(kernel, RBFKernel)
        assert len(space.parameters) == 2 and encoder.n_columns == 4
        assert tuple(kernel.lengthscale.shape) == (1, 4)


class TestItsConfiguration:
    def test_the_fields_and_their_defaults(self) -> None:
        assert toResolvedConfig(RBFKernelConfig()) == {"type": "rbf_v1", "ard": True}

    def test_it_has_no_smoothness(self) -> None:
        # `nu` belongs to the Matern kernel; the RBF kernel is its limit and has no such field.
        with pytest.raises(ConfigError) as caught:
            resolveComponent(kernelRegistry(), {"type": "rbf_v1", "nu": 2.5})

        assert caught.value.field_path == "nu"
        assert "Unknown field" in caught.value.message

    def test_resolving_and_validating_again_gives_the_same_configuration(self) -> None:
        config = RBFKernelConfig(ard=False)

        assert validateConfig(RBFKernelConfig, toResolvedConfig(config)) == config

    def test_it_is_not_the_matern_kernel(self) -> None:
        assert not isinstance(build(), MaternKernel)


class TestWhatItBuildsAndWhatItLeavesOut:
    def test_every_call_makes_a_new_kernel_that_shares_nothing(self) -> None:
        builder = RBFKernelBuilder.fromConfig(RBFKernelConfig())

        first, second = builder.build(N_COLUMNS), builder.build(N_COLUMNS)

        assert first is not second
        assert isinstance(first, RBFKernel) and isinstance(second, RBFKernel)
        first.lengthscale = 5.0
        assert not torch.allclose(first.lengthscale, second.lengthscale)

    def test_it_carries_no_prior_unlike_the_kernel_botorch_builds_by_itself(self) -> None:
        # `rbf_v1` with ARD is the engine's default kernel without its lengthscale prior.
        engine_default, _ = fitAndPredict(None)

        assert list(build().named_priors()) == []
        assert isinstance(engine_default.covar_module, RBFKernel)
        assert len(list(engine_default.covar_module.named_priors())) == 1

    def test_it_is_not_wrapped_in_an_output_scale(self) -> None:
        assert not isinstance(build(), ScaleKernel)


class TestInASurrogate:
    @pytest.mark.parametrize("ard", [True, False])
    def test_a_model_given_the_kernel_fits_and_predicts(self, ard: bool) -> None:
        model, mean = fitAndPredict(build(ard=ard))

        assert mean.shape == (20, 1) and bool(torch.isfinite(mean).all())
        assert isinstance(model.covar_module, RBFKernel)

    def test_one_model_for_each_outcome_with_a_kernel_of_its_own_is_the_way(self) -> None:
        builder = RBFKernelBuilder.fromConfig(RBFKernelConfig())

        (first, _), (second, _) = [fitAndPredict(builder.build(N_COLUMNS)) for _ in range(2)]

        assert first.covar_module is not second.covar_module

    def test_one_batched_model_of_several_outputs_cannot_be_fitted_with_it(self) -> None:
        # The same limit as the Matern kernel's, for the same reason: no batch shape.
        with pytest.raises(RuntimeError, match="invalid for input of size"):
            fitAndPredict(build(), n_outputs=2)
