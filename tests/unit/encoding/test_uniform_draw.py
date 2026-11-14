"""Unit tests for the `draw` hook of `SearchSpaceEncoder.sampleEncoded` and
`sampleFeasibleEncoded` (roadmap task P4-06, `UniformDraw`).

A design supplies the raw points that the sampler turns into encoded ones, so that the
constraint handling is the sampler's and the choice of points is the design's. Tested: what the
draw is called with, that giving none keeps the scrambled Sobol points the sampler has always
drawn, that the rounds of a constrained search call the draw again, and that the hit-and-run
chain, which has no raw points to draw, never does.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from planexs import Categorical, LinearConstraint, NonlinearConstraint, Real, SearchSpace
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.FeasibleSamplingConfig import FeasibleSamplingConfig
from planexs.encoding.sampleFeasibleEncoded import sampleFeasibleEncoded
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder


def unitSquare() -> SearchSpace:
    """Two reals in `[0, 1]`: the encoding is the identity."""
    return SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)])


def mixedSpace() -> SearchSpace:
    """Two reals and a three-level categorical: three searched parameters, five columns."""
    return SearchSpace(
        [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Categorical("kind", ["u", "v", "w"])]
    )


class Recorder:
    """A draw that returns a constant and records what it was called with."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[int, int]] = []

    def __call__(self, n: int, dimension: int) -> Tensor:
        self.calls.append((n, dimension))
        return torch.full((n, dimension), self.value, dtype=torch.double)


class TestSampleEncoded:
    def test_the_draw_is_called_once_with_the_count_and_the_number_of_searched_parameters(
        self,
    ) -> None:
        encoder = SearchSpaceEncoder(mixedSpace())
        draw = Recorder(0.5)

        encoder.sampleEncoded(4, draw=draw)

        # Three searched parameters, and five encoded columns: the draw is in the first space.
        assert draw.calls == [(4, 3)]
        assert encoder.n_columns == 5

    def test_the_raw_points_are_turned_into_the_encoding_of_each_parameter(self) -> None:
        encoder = SearchSpaceEncoder(mixedSpace())

        points = encoder.sampleEncoded(2, draw=Recorder(0.5))

        # The reals keep 0.5, and 0.5 of three categories is the second one: a one-hot block.
        assert points.tolist() == [[0.5, 0.5, 0.0, 1.0, 0.0]] * 2

    def test_what_the_draw_returns_is_what_is_encoded(self) -> None:
        encoder = SearchSpaceEncoder(unitSquare())
        fixed = torch.tensor([[0.25, 0.75], [0.5, 0.125]])

        points = encoder.sampleEncoded(2, draw=lambda n, dimension: fixed)

        assert torch.equal(points, fixed.to(torch.float64))

    def test_the_seed_is_ignored_when_a_draw_is_given(self) -> None:
        # A draw owns its randomness: here it has none, so the seed can change nothing.
        encoder = SearchSpaceEncoder(unitSquare())

        first = encoder.sampleEncoded(4, seed=1, draw=Recorder(0.3))
        second = encoder.sampleEncoded(4, seed=2, draw=Recorder(0.3))

        assert torch.equal(first, second)

    def test_the_points_are_in_the_dtype_of_the_encoder(self) -> None:
        encoder = SearchSpaceEncoder(unitSquare(), RuntimeConfig(dtype="float32"))

        # The draw returns double precision: the encoder converts.
        points = encoder.sampleEncoded(
            3, draw=lambda n, d: torch.zeros(n, d, dtype=torch.double)
        )

        assert points.dtype == torch.float32

    def test_the_points_are_snapped_unless_told_otherwise(self) -> None:
        space = SearchSpace([Real("a", 0.0, 1.0, step_size=0.5)])
        encoder = SearchSpaceEncoder(space)

        snapped = encoder.sampleEncoded(1, draw=Recorder(0.3))
        relaxed = encoder.sampleEncoded(1, snap=False, draw=Recorder(0.3))

        assert snapped.tolist() == [[0.5]]
        assert relaxed[0, 0].item() == pytest.approx(0.3, abs=1e-6)

    @pytest.mark.parametrize(
        "shape", [(3, 3), (5, 2), (4,), (4, 2, 1)], ids=["rows", "columns", "1d", "3d"]
    )
    def test_a_draw_of_the_wrong_shape_is_refused_and_says_what_was_expected(
        self, shape: tuple[int, ...]
    ) -> None:
        encoder = SearchSpaceEncoder(unitSquare())

        with pytest.raises(ValueError, match=r"need the shape \(4, 2\)") as caught:
            encoder.sampleEncoded(4, draw=lambda n, dimension: torch.zeros(shape))

        assert str(tuple(shape)) in str(caught.value)


class TestTheDefaultIsUnchanged:
    """No draw means the scrambled Sobol sequence that the sampler has always used."""

    def test_it_is_the_scrambled_sobol_sequence_of_the_seed(self) -> None:
        encoder = SearchSpaceEncoder(unitSquare())

        points = encoder.sampleEncoded(8, seed=5)

        expected = SobolEngine(dimension=2, scramble=True, seed=5).draw(8).to(torch.float64)
        assert torch.equal(points, expected)

    def test_it_depends_on_the_seed(self) -> None:
        encoder = SearchSpaceEncoder(unitSquare())

        assert not torch.equal(
            encoder.sampleEncoded(8, seed=1), encoder.sampleEncoded(8, seed=2)
        )

    def test_giving_none_is_giving_nothing(self) -> None:
        encoder = SearchSpaceEncoder(mixedSpace())

        assert torch.equal(
            encoder.sampleEncoded(6, seed=3), encoder.sampleEncoded(6, seed=3, draw=None)
        )


class TestSampleFeasibleEncoded:
    def test_without_constraints_the_draw_supplies_the_points(self) -> None:
        space = unitSquare()
        encoder = SearchSpaceEncoder(space)
        draw = Recorder(0.25)

        points = sampleFeasibleEncoded(
            encoder, EncodedConstraints(space, encoder), 3, draw=draw
        )

        assert draw.calls == [(3, 2)]
        assert points.tolist() == [[0.25, 0.25]] * 3

    def test_with_a_nonlinear_constraint_each_round_calls_the_draw_again(self) -> None:
        # `a * b >= 0.5` on the unit square. The first draw is all zeros, so no candidate is
        # feasible, and the second is all 0.9, so every one is: two rounds, and the second
        # asks for twice as many candidates as the first.
        space = SearchSpace(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)],
            [NonlinearConstraint("a * b", ">=", 0.5)],
        )
        encoder = SearchSpaceEncoder(space)
        sampling = FeasibleSamplingConfig(oversampling_factor=1, min_draw=1)
        constraints = EncodedConstraints(space, encoder, sampling)
        values = iter([0.0, 0.9])
        calls: list[tuple[int, int]] = []

        def draw(n: int, dimension: int) -> Tensor:
            calls.append((n, dimension))
            return torch.full((n, dimension), next(values), dtype=torch.double)

        points = sampleFeasibleEncoded(encoder, constraints, 4, draw=draw)

        assert calls == [(4, 2), (8, 2)]
        assert points.tolist() == [[0.9, 0.9]] * 4

    def test_the_hit_and_run_chain_never_calls_the_draw(self) -> None:
        # Linear constraints are met by a chain that has no raw points to draw: only the seed
        # governs it, and the draw is not consulted.
        space = SearchSpace(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)],
            [LinearConstraint({"a": 1.0, "b": 1.0}, 1.0, "=")],
        )
        encoder = SearchSpaceEncoder(space)
        constraints = EncodedConstraints(space, encoder)
        draw = Recorder(0.5)

        points = sampleFeasibleEncoded(encoder, constraints, 3, seed=2, draw=draw)

        assert draw.calls == []
        assert torch.allclose(points.sum(dim=1), torch.ones(3, dtype=points.dtype))

    def test_the_default_is_unchanged_with_and_without_constraints(self) -> None:
        space = SearchSpace(
            [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)],
            [NonlinearConstraint("a * b", ">=", 0.1)],
        )
        encoder = SearchSpaceEncoder(space)
        constraints = EncodedConstraints(space, encoder)

        plain = sampleFeasibleEncoded(encoder, constraints, 5, seed=4)
        explicit = sampleFeasibleEncoded(encoder, constraints, 5, seed=4, draw=None)

        assert torch.equal(plain, explicit)
