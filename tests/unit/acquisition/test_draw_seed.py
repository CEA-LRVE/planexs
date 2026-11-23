"""Unit tests for `drawSeed`, the seed of a sampler drawn from the generator of a build
(roadmap task P5-05, specification 5.8)."""

from __future__ import annotations

import torch

from planexs.acquisition.drawSeed import drawSeed
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator


class TestDrawSeed:
    def test_it_gives_a_whole_number_that_a_sampler_accepts(self) -> None:
        seed = drawSeed(torchGenerator(7, RuntimeConfig()))

        assert isinstance(seed, int)
        assert 0 <= seed < torch.iinfo(torch.int32).max

    def test_the_same_generator_state_gives_the_same_seed(self) -> None:
        first = drawSeed(torchGenerator(7, RuntimeConfig()))
        second = drawSeed(torchGenerator(7, RuntimeConfig()))

        assert first == second

    def test_another_generator_seed_gives_another_seed(self) -> None:
        assert drawSeed(torchGenerator(7, RuntimeConfig())) != drawSeed(
            torchGenerator(8, RuntimeConfig())
        )

    def test_each_draw_advances_the_generator(self) -> None:
        generator = torchGenerator(7, RuntimeConfig())

        seeds = [drawSeed(generator) for _ in range(5)]

        assert len(set(seeds)) == 5

    def test_the_global_generator_is_not_used(self) -> None:
        generator = torchGenerator(7, RuntimeConfig())
        state = torch.get_rng_state()

        drawSeed(generator)

        assert torch.equal(state, torch.get_rng_state())
