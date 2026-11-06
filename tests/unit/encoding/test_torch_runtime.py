"""Unit tests for `planexs.encoding.torch_runtime` (specification 3.4, 3.7, 8)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch

from planexs.core.ConfigError import ConfigError
from planexs.core.RngTree import RngTree
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import (
    deterministicAlgorithms,
    torchDevice,
    torchDtype,
    torchGenerator,
)

#: The largest seed `RngTree.seed` can return: its streams are 64-bit words.
LARGEST_RNG_TREE_SEED = 2**64 - 1


@pytest.fixture(autouse=True)
def restoreTorchDeterminism() -> Iterator[None]:
    """Gives every test the process-wide deterministic mode it found, even when it fails."""
    mode = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)


def deterministicState() -> tuple[bool, bool]:
    """torch's deterministic mode and its warn-only flag, as one value."""
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


class TestTorchDtype:
    @pytest.mark.parametrize(
        ("name", "expected"), [("float32", torch.float32), ("float64", torch.float64)]
    )
    def test_each_name_gives_the_torch_type_of_that_name(
        self, name: Any, expected: torch.dtype
    ) -> None:
        assert torchDtype(RuntimeConfig(dtype=name)) is expected

    def test_the_default_runtime_is_double_precision(self) -> None:
        assert torchDtype(RuntimeConfig()) is torch.float64

    @pytest.mark.parametrize("name", ["bogus", "device", "Tensor", "zeros"])
    def test_a_name_that_torch_does_not_know_as_a_dtype_is_reported(self, name: str) -> None:
        # `RuntimeConfig` only lets two names through, so build one around its validation.
        runtime = RuntimeConfig.model_construct(dtype=name)
        with pytest.raises(ConfigError) as caught:
            torchDtype(runtime)
        assert caught.value.component == "runtime"
        assert caught.value.field_path == "dtype"
        assert repr(name) in caught.value.message
        assert caught.value.suggestion == "use 'float32' or 'float64'"


class TestTorchDevice:
    def test_the_default_runtime_is_the_cpu(self) -> None:
        assert torchDevice(RuntimeConfig()) == torch.device("cpu")

    def test_a_device_type_that_torch_does_not_know_is_reported(self) -> None:
        with pytest.raises(ConfigError) as caught:
            torchDevice(RuntimeConfig(device="cudaa"))
        assert caught.value.component == "runtime"
        assert caught.value.field_path == "device"
        assert "'cudaa' cannot be used" in caught.value.message
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_a_well_formed_device_that_is_not_on_this_machine_is_reported(self) -> None:
        # Written correctly, so `RuntimeConfig` accepts it: only torch knows it is absent.
        runtime = RuntimeConfig(device="cuda:99")
        with pytest.raises(ConfigError) as caught:
            torchDevice(runtime)
        assert caught.value.field_path == "device"
        assert "'cuda:99' cannot be used" in caught.value.message

    def test_a_build_of_torch_without_support_for_the_device_is_reported_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A CPU-only build of torch raises AssertionError ("not compiled with CUDA").
        def refuse(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("Torch not compiled with CUDA enabled")

        monkeypatch.setattr(torch, "empty", refuse)
        with pytest.raises(ConfigError, match="not compiled with CUDA"):
            torchDevice(RuntimeConfig())

    def test_only_the_first_line_of_torchs_explanation_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("first line\nsecond line\nthird line")

        monkeypatch.setattr(torch, "empty", refuse)
        with pytest.raises(ConfigError) as caught:
            torchDevice(RuntimeConfig())
        assert "first line" in caught.value.message
        assert "second line" not in caught.value.message

    def test_an_explanation_with_no_text_falls_back_to_the_error_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError()

        monkeypatch.setattr(torch, "empty", refuse)
        with pytest.raises(ConfigError, match="RuntimeError"):
            torchDevice(RuntimeConfig())


class TestTorchGenerator:
    def draw(self, generator: torch.Generator) -> torch.Tensor:
        return torch.rand(8, generator=generator, dtype=torch.float64)

    def test_it_is_a_torch_generator_on_the_runtime_device(self) -> None:
        generator = torchGenerator(7, RuntimeConfig())
        assert isinstance(generator, torch.Generator)
        assert generator.device == torch.device("cpu")
        assert generator.initial_seed() == 7

    def test_the_same_seed_gives_the_same_stream(self) -> None:
        first = self.draw(torchGenerator(7, RuntimeConfig()))
        second = self.draw(torchGenerator(7, RuntimeConfig()))
        assert torch.equal(first, second)

    def test_a_different_seed_gives_a_different_stream(self) -> None:
        first = self.draw(torchGenerator(7, RuntimeConfig()))
        second = self.draw(torchGenerator(8, RuntimeConfig()))
        assert not torch.equal(first, second)

    def test_every_generator_starts_at_the_beginning_of_its_stream(self) -> None:
        used = torchGenerator(7, RuntimeConfig())
        self.draw(used)
        self.draw(used)
        fresh = torchGenerator(7, RuntimeConfig())
        assert torch.equal(self.draw(fresh), self.draw(torchGenerator(7, RuntimeConfig())))

    def test_the_streams_of_one_rng_tree_are_reproducible_and_independent(self) -> None:
        tree = RngTree(2024)
        runtime = RuntimeConfig()
        same = [
            self.draw(torchGenerator(tree.seed(3, "policy.surrogate"), runtime))
            for _ in range(2)
        ]
        other_path = self.draw(torchGenerator(tree.seed(3, "policy.design"), runtime))
        other_ask = self.draw(torchGenerator(tree.seed(4, "policy.surrogate"), runtime))
        assert torch.equal(same[0], same[1])
        assert not torch.equal(same[0], other_path)
        assert not torch.equal(same[0], other_ask)

    def test_a_tree_rebuilt_from_its_recorded_entropy_gives_the_same_streams(self) -> None:
        tree = RngTree()
        rebuilt = RngTree(tree.entropy)
        first = self.draw(torchGenerator(tree.seed(0, "encoder"), RuntimeConfig()))
        second = self.draw(torchGenerator(rebuilt.seed(0, "encoder"), RuntimeConfig()))
        assert torch.equal(first, second)

    @pytest.mark.parametrize("seed", [0, 1, LARGEST_RNG_TREE_SEED])
    def test_every_seed_an_rng_tree_can_give_is_accepted(self, seed: int) -> None:
        assert torchGenerator(seed, RuntimeConfig()).initial_seed() == seed

    def test_drawing_from_it_leaves_the_global_random_state_alone(self) -> None:
        state = torch.get_rng_state()
        try:
            torch.manual_seed(123)
            expected = torch.rand(3)
            torch.manual_seed(123)
            self.draw(torchGenerator(5, RuntimeConfig()))
            assert torch.equal(torch.rand(3), expected)
        finally:
            torch.set_rng_state(state)

    @pytest.mark.parametrize("seed", [2**64, -(2**63) - 1])
    def test_a_seed_that_torch_refuses_is_reported(self, seed: int) -> None:
        with pytest.raises(ConfigError) as caught:
            torchGenerator(seed, RuntimeConfig())
        assert caught.value.field_path == "seed"
        assert f"does not accept {seed}" in caught.value.message
        assert caught.value.suggestion == "use a seed from RngTree.seed"

    @pytest.mark.parametrize("seed", [True, False, 1.5, "7", None])
    def test_a_seed_that_is_not_an_integer_is_reported(self, seed: Any) -> None:
        with pytest.raises(ConfigError, match="must be an integer") as caught:
            torchGenerator(seed, RuntimeConfig())
        assert caught.value.field_path == "seed"

    def test_a_device_that_cannot_be_used_is_reported_as_the_device(self) -> None:
        with pytest.raises(ConfigError) as caught:
            torchGenerator(7, RuntimeConfig(device="cuda:99"))
        assert caught.value.field_path == "device"


class TestDeterministicAlgorithms:
    @pytest.mark.parametrize("previous", [False, True])
    @pytest.mark.parametrize("configured", [False, True])
    def test_the_mode_is_the_configured_one_inside_and_the_previous_one_after(
        self, previous: bool, configured: bool
    ) -> None:
        torch.use_deterministic_algorithms(previous)
        runtime = RuntimeConfig(deterministic_algorithms=configured)
        with deterministicAlgorithms(runtime):
            assert torch.are_deterministic_algorithms_enabled() is configured
        assert torch.are_deterministic_algorithms_enabled() is previous

    def test_the_configuration_wins_over_what_the_process_did_before(self) -> None:
        torch.use_deterministic_algorithms(True)
        with deterministicAlgorithms(RuntimeConfig(deterministic_algorithms=False)):
            assert not torch.are_deterministic_algorithms_enabled()

    @pytest.mark.parametrize("configured", [False, True])
    def test_the_warn_only_flag_is_restored_too(self, configured: bool) -> None:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with deterministicAlgorithms(RuntimeConfig(deterministic_algorithms=configured)):
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert deterministicState() == (True, True)

    def test_the_previous_mode_is_restored_when_the_block_raises(self) -> None:
        torch.use_deterministic_algorithms(False)
        with (
            pytest.raises(ZeroDivisionError),
            deterministicAlgorithms(RuntimeConfig(deterministic_algorithms=True)),
        ):
            assert torch.are_deterministic_algorithms_enabled()
            raise ZeroDivisionError
        assert deterministicState() == (False, False)

    def test_blocks_can_be_nested(self) -> None:
        torch.use_deterministic_algorithms(False)
        with deterministicAlgorithms(RuntimeConfig(deterministic_algorithms=True)):
            with deterministicAlgorithms(RuntimeConfig(deterministic_algorithms=False)):
                assert not torch.are_deterministic_algorithms_enabled()
            assert torch.are_deterministic_algorithms_enabled()
        assert not torch.are_deterministic_algorithms_enabled()

    def test_it_yields_nothing(self) -> None:
        with deterministicAlgorithms(RuntimeConfig()) as value:
            assert value is None
