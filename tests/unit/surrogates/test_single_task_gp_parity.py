"""Parity of `single_task_gp_v1` with the legacy surrogate (roadmap task P3-14, section 2.3,
specification 10.11: "parity tests whenever a component replaces another").

The legacy `buildSurrogateModel` is fitted on the same data as the new builder with its default
configuration, and their posteriors are compared at fixed points. The legacy function and its
tests are deleted by roadmap task P11-02, and this test with them.

**Tolerances.** Posterior means and variances are compared with `rtol=1e-6` and `atol=1e-8`.
Both fits start from the same point and run the same optimizer on the same objective, so on the
machine where this was written the two posteriors are equal to the last bit; the tolerance is
only room for a different BLAS or thread count, and is far below any difference that a change
of the model would make (the negative control below differs by more than 1e-2).
"""

from __future__ import annotations

import pytest
import torch
from botorch.models.model import Model
from single_task_gp_fakes import builderOf, fixedPoints, makeContext

from planexs.models.buildSurrogateModel import buildSurrogateModel
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext

RTOL = 1e-6
ATOL = 1e-8

#: `(observations, outcomes, known noise)`: inferred and known noise, one and several
#: outcomes, and a low-data fit.
CASES = {
    "inferred-one": (12, 1, False),
    "inferred-several": (12, 3, False),
    "known-one": (12, 1, True),
    "known-several": (12, 3, True),
    "low-data": (3, 1, False),
}


def legacyModel(context: SurrogateBuildContext) -> Model:
    """What the legacy function builds and fits on the data of the context."""
    variance = None if context.train_noise_std is None else context.train_noise_std**2
    return buildSurrogateModel(context.train_inputs, context.train_outcomes, variance)


class TestPosteriorsMatchTheLegacySurrogate:
    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("observation_noise", [False, True], ids=["latent", "noisy"])
    def test_means_and_variances_at_fixed_points(
        self, case: str, observation_noise: bool
    ) -> None:
        n_points, n_outcomes, noise = CASES[case]
        context = makeContext(n_points, n_outcomes, noise=noise)
        legacy = legacyModel(context)

        new = builderOf().build(context).model

        with torch.no_grad():
            expected = legacy.posterior(fixedPoints(), observation_noise=observation_noise)
            found = new.posterior(fixedPoints(), observation_noise=observation_noise)
        assert new is not legacy
        torch.testing.assert_close(found.mean, expected.mean, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(found.variance, expected.variance, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("case", ["inferred-several", "known-several"])
    def test_the_hyperparameters_match_too(self, case: str) -> None:
        n_points, n_outcomes, noise = CASES[case]
        context = makeContext(n_points, n_outcomes, noise=noise)
        legacy = legacyModel(context)

        new = builderOf().build(context).model

        expected = dict(legacy.named_parameters())
        found = dict(new.named_parameters())
        assert found.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(found[name], value, rtol=RTOL, atol=ATOL)


class TestTheComparisonCanFail:
    """A parity test that nothing can break proves nothing: these are its negative controls."""

    def test_posteriors_of_different_data_differ_by_more_than_the_tolerance(self) -> None:
        context = makeContext(12, 1)
        shifted = makeContext(12, 1, train_outcomes=context.train_outcomes * 1.1 + 0.2)
        legacy = legacyModel(shifted)

        new = builderOf().build(context).model

        with torch.no_grad():
            difference = (
                new.posterior(fixedPoints()).mean - legacy.posterior(fixedPoints()).mean
            )
        assert float(difference.abs().max()) > 1e-2

    def test_a_noise_that_is_ignored_gives_a_different_posterior(self) -> None:
        # The legacy fit is given the known noise and the new one is not: the variances must
        # differ, so that the known-noise cases above do test that the noise arrives.
        known = makeContext(12, 1, noise=True)
        legacy = legacyModel(known)

        new = builderOf().build(makeContext(12, 1, noise=False)).model

        with torch.no_grad():
            difference = (
                new.posterior(fixedPoints()).variance
                - legacy.posterior(fixedPoints()).variance
            )
        assert float(difference.abs().max()) > ATOL * 1e4

    def test_the_posterior_varies_across_the_points_so_that_the_match_is_not_trivial(
        self,
    ) -> None:
        model = builderOf().build(makeContext(12, 1)).model

        with torch.no_grad():
            posterior = model.posterior(fixedPoints())

        assert float(posterior.mean.std()) > 0.1
        assert float(posterior.variance.std()) > 0.0
