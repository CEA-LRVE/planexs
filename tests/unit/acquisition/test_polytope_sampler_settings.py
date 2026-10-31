"""The polytope sampler's burn-in and thinning are configuration, not hidden constants."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

from planexs import AlphaAcquisitionStrategy, LinearConstraint, Real, SearchSpace
from planexs.acquisition.EncodedConstraints import EncodedConstraints
from planexs.acquisition.sampleFeasibleEncoded import (
    DEFAULT_POLYTOPE_BURN_IN,
    DEFAULT_POLYTOPE_THINNING,
    sampleFeasibleEncoded,
)
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder

_SAMPLER_MODULE = "planexs.acquisition.sampleFeasibleEncoded"


def _mixture() -> SearchSpace:
    return SearchSpace(
        [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
        constraints=[LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")],
    )


@pytest.fixture
def sampler_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Records the arguments the hit-and-run sampler is called with."""
    module = importlib.import_module(_SAMPLER_MODULE)
    calls: list[dict[str, Any]] = []
    original = module.sample_q_batches_from_polytope

    def recording(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append({"n_burnin": kwargs["n_burnin"], "n_thinning": kwargs["n_thinning"]})
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "sample_q_batches_from_polytope", recording)
    return calls


class TestDefaults:
    def test_the_documented_defaults_are_the_ones_used(self) -> None:
        assert (DEFAULT_POLYTOPE_BURN_IN, DEFAULT_POLYTOPE_THINNING) == (200, 10)

    def test_the_strategy_defaults_to_them(self, sampler_calls: list[dict[str, Any]]) -> None:
        AlphaAcquisitionStrategy().suggestSpaceFilling(_mixture(), 3, seed=0)

        assert sampler_calls
        assert {(c["n_burnin"], c["n_thinning"]) for c in sampler_calls} == {
            (DEFAULT_POLYTOPE_BURN_IN, DEFAULT_POLYTOPE_THINNING)
        }


class TestSamplerFunction:
    def test_forwards_the_requested_values(self, sampler_calls: list[dict[str, Any]]) -> None:
        space = _mixture()
        encoder = SearchSpaceEncoder(space)

        sampleFeasibleEncoded(
            encoder, EncodedConstraints(space, encoder), 5, seed=0, n_burnin=7, n_thinning=3
        )

        assert {(c["n_burnin"], c["n_thinning"]) for c in sampler_calls} == {(7, 3)}

    def test_does_not_use_the_chain_without_linear_constraints(
        self, sampler_calls: list[dict[str, Any]]
    ) -> None:
        space = SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)])
        encoder = SearchSpaceEncoder(space)

        sampleFeasibleEncoded(encoder, EncodedConstraints(space, encoder), 5, seed=0)

        assert sampler_calls == []

    def test_equality_holds_whatever_the_settings(self) -> None:
        space = _mixture()
        encoder = SearchSpaceEncoder(space)
        constraints = EncodedConstraints(space, encoder)

        for burn_in, thinning in ((1, 1), (50, 2), (400, 20)):
            points = sampleFeasibleEncoded(
                encoder, constraints, 30, seed=1, n_burnin=burn_in, n_thinning=thinning
            )
            assert torch.allclose(
                points.sum(dim=-1), torch.ones(30, dtype=torch.double), atol=1e-6
            ), (burn_in, thinning)


class TestStrategyConfiguration:
    def test_every_use_of_the_sampler_gets_the_configured_values(
        self, sampler_calls: list[dict[str, Any]]
    ) -> None:
        strategy = AlphaAcquisitionStrategy(polytope_burn_in=11, polytope_thinning=4)

        strategy.suggestSpaceFilling(_mixture(), 4, seed=0)

        assert sampler_calls
        assert {(c["n_burnin"], c["n_thinning"]) for c in sampler_calls} == {(11, 4)}

    def test_the_optimizers_own_starting_points_get_them_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from planexs import LegacyMetric, LegacyObjective, LegacyTrial
        from planexs.models.buildSurrogateModel import buildSurrogateModel
        from planexs.models.encodeTrials import encodeTrials

        module = importlib.import_module("planexs.acquisition.AlphaAcquisitionStrategy")
        seen: list[dict[str, Any]] = []
        original = module.optimize_acqf

        def recording(*args: Any, **kwargs: Any) -> Any:
            seen.append(dict(kwargs["options"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "optimize_acqf", recording)
        space = _mixture()
        objective = LegacyObjective([LegacyMetric("f", minimize=True)])
        strategy = AlphaAcquisitionStrategy(
            num_restarts=2,
            raw_samples=32,
            n_reference_points=32,
            n_mc_samples=16,
            polytope_burn_in=13,
            polytope_thinning=5,
        )
        points = strategy.suggestSpaceFilling(space, 5, seed=0)
        trials = [LegacyTrial(parameters=p, results={"f": p["a"]}) for p in points]
        model = buildSurrogateModel(
            *encodeTrials(SearchSpaceEncoder(space), objective, trials)
        )

        strategy.suggest(model, objective, space, 0.5, 1, seed=0)

        assert seen
        assert all(o["n_burnin"] == 13 and o["n_thinning"] == 5 for o in seen)


class TestSeeding:
    """`manual_seed(None)` is a no-op, so an unseeded call needs no special case."""

    @staticmethod
    def _model_and_space():  # type: ignore[no-untyped-def]
        from planexs import LegacyMetric, LegacyObjective, LegacyTrial
        from planexs.models.buildSurrogateModel import buildSurrogateModel
        from planexs.models.encodeTrials import encodeTrials

        space = SearchSpace([Real("x", 0.0, 1.0)])
        objective = LegacyObjective([LegacyMetric("f", minimize=True)])
        trials = [
            LegacyTrial(parameters={"x": x}, results={"f": (x - 0.3) ** 2})
            for x in (0.0, 0.5, 1.0)
        ]
        model = buildSurrogateModel(
            *encodeTrials(SearchSpaceEncoder(space), objective, trials)
        )
        return model, objective, space

    def test_an_unseeded_call_works(self) -> None:
        model, objective, space = self._model_and_space()
        strategy = AlphaAcquisitionStrategy(
            num_restarts=2, raw_samples=32, n_reference_points=32
        )

        (point,) = strategy.suggest(model, objective, space, 0.5, 1, seed=None)

        assert 0.0 <= point["x"] <= 1.0  # type: ignore[operator]

    def test_a_seeded_call_leaves_the_global_random_state_untouched(self) -> None:
        model, objective, space = self._model_and_space()
        strategy = AlphaAcquisitionStrategy(
            num_restarts=2, raw_samples=32, n_reference_points=32
        )
        before = torch.random.get_rng_state()

        strategy.suggest(model, objective, space, 0.5, 1, seed=123)

        assert torch.equal(torch.random.get_rng_state(), before)
