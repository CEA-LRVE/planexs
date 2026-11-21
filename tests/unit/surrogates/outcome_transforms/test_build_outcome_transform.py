"""Unit tests for `buildOutcomeTransform` (roadmap task P3-12, specification 5.6).

It turns the list of transforms of a surrogate's configuration into the one native transform
that a model takes. The card asks for "chaining order": that the transforms are applied in the
order given, and their effect on the predictions undone in the reverse order.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    OutcomeTransform,
    Power,
    Standardize,
)
from outcome_transform_fakes import defineFake, fakeConfig, fitModel, toyData

import planexs.surrogates  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.buildOutcomeTransform import buildOutcomeTransform
from planexs.surrogates.outcome_transforms.OutcomeTransformError import OutcomeTransformError
from planexs.surrogates.outcome_transforms.outcomeTransformRegistry import (
    outcomeTransformRegistry,
)
from planexs.surrogates.outcome_transforms.registerOutcomeTransform import (
    registerOutcomeTransform,
)
from planexs.surrogates.outcome_transforms.StrictlyPositiveLog import StrictlyPositiveLog

DOUBLE = torch.float64

# The toy model of a logarithm is fitted on targets that are not standardized, and BoTorch
# warns about it.
pytestmark = pytest.mark.filterwarnings("ignore::botorch.exceptions.warnings.InputDataWarning")


def config(name: str, **settings: Any) -> ComponentConfig:
    """The validated configuration of a registered transform, as a surrogate's holds it."""
    return resolveComponent(outcomeTransformRegistry(), {"type": name, **settings}).config


def standardize(**settings: Any) -> ComponentConfig:
    return config("standardize_v1", **settings)


def log() -> ComponentConfig:
    return config("log_v1")


def none() -> ComponentConfig:
    return config("none_v1")


def defineSquare() -> None:
    """Registers `square_v1`, which squares the outcome (`Power(2)`). It stands for a plugin's
    transform that does not commute with the logarithm, and is valid for any outcome.
    """

    @registerOutcomeTransform(
        "square_v1", config=fakeConfig("square_v1"), description="Squares each outcome."
    )
    class SquareBuilder(AbstractOutcomeTransformBuilder[Any]):
        def makeTransform(self, n_outcomes: int) -> OutcomeTransform:
            return Power(power=2.0)


class TestWhatIsLeftOfTheList:
    def test_an_empty_list_is_no_transform(self) -> None:
        assert buildOutcomeTransform([], 1) is None

    @pytest.mark.parametrize("count", [1, 2, 5])
    def test_a_list_of_transforms_that_are_none_is_no_transform(self, count: int) -> None:
        assert buildOutcomeTransform([none()] * count, 2) is None

    def test_one_transform_is_returned_as_it_is_and_not_wrapped(self) -> None:
        # A list of one `standardize_v1` is the `Standardize` that BoTorch builds by default.
        transform = buildOutcomeTransform([standardize()], 3)

        assert type(transform) is Standardize
        assert transform.means.shape == (1, 3)

    def test_one_log_is_returned_as_it_is(self) -> None:
        assert type(buildOutcomeTransform([log()], 1)) is StrictlyPositiveLog

    def test_a_transform_that_is_none_is_left_out_of_the_chain(self) -> None:
        transform = buildOutcomeTransform([none(), standardize(), none()], 1)

        assert type(transform) is Standardize

    def test_several_transforms_are_chained(self) -> None:
        transform = buildOutcomeTransform([log(), standardize()], 1)

        assert isinstance(transform, ChainedOutcomeTransform)
        assert [type(step) for step in transform.values()] == [
            StrictlyPositiveLog,
            Standardize,
        ]

    def test_the_settings_of_each_transform_reach_it(self) -> None:
        transform = buildOutcomeTransform([log(), standardize(min_std=0.5)], 1)
        assert isinstance(transform, ChainedOutcomeTransform)
        spread = torch.tensor(
            [[1.0], [2.0], [4.0]], dtype=DOUBLE
        )  # log has a deviation of 0.69

        transform(spread)

        # Under a floor of 0.5 the log of the outcome (deviation 0.69) is scaled...
        assert float(transform["1_standardize_v1"].stdvs) == pytest.approx(0.6931, abs=1e-3)
        # ...and under a floor of 1 it would not be.
        other = buildOutcomeTransform([log(), standardize(min_std=1.0)], 1)
        assert isinstance(other, ChainedOutcomeTransform)
        other(spread)
        assert float(other["1_standardize_v1"].stdvs) == 1.0

    def test_the_number_of_outcomes_reaches_every_transform(self) -> None:
        transform = buildOutcomeTransform([log(), standardize()], 4)
        assert isinstance(transform, ChainedOutcomeTransform)

        assert transform["1_standardize_v1"].means.shape == (1, 4)


class TestTheNamesInTheChain:
    """A chain is a `ModuleDict`: its keys are what a state dict and a repr show."""

    def test_the_position_in_the_list_is_in_the_name(self) -> None:
        transform = buildOutcomeTransform([log(), standardize()], 1)
        assert isinstance(transform, ChainedOutcomeTransform)

        assert list(transform.keys()) == ["0_log_v1", "1_standardize_v1"]

    def test_the_same_transform_twice_is_a_legal_list(self) -> None:
        transform = buildOutcomeTransform([standardize(), standardize()], 1)
        assert isinstance(transform, ChainedOutcomeTransform)

        assert list(transform.keys()) == ["0_standardize_v1", "1_standardize_v1"]
        assert transform["0_standardize_v1"] is not transform["1_standardize_v1"]

    def test_a_transform_left_out_does_not_shift_the_names_of_the_others(self) -> None:
        # The names say where in the user's list a transform was, `none_v1` included.
        transform = buildOutcomeTransform([none(), log(), standardize()], 1)
        assert isinstance(transform, ChainedOutcomeTransform)

        assert list(transform.keys()) == ["1_log_v1", "2_standardize_v1"]


class TestTheOrder:
    """The transforms are applied in the order given, and undone in the reverse order."""

    def test_the_first_transform_is_applied_first(self) -> None:
        y = toyData()[1]
        transform = buildOutcomeTransform([log(), standardize()], 1)
        assert transform is not None

        mapped, _ = transform(y)

        # The logarithm first, then the standardization of the logarithm.
        expected, _ = Standardize(m=1)(torch.log(y))
        assert torch.allclose(mapped, expected)
        assert float(mapped.mean()) == pytest.approx(0.0, abs=1e-9)
        assert float(mapped.std()) == pytest.approx(1.0, abs=1e-9)

    def test_the_other_order_is_another_transform_and_refuses_what_it_cannot_take(
        self,
    ) -> None:
        # A standardized outcome has negative values, which have no logarithm: the reversed
        # list fails, with a named error and not with a `nan`.
        y = toyData()[1]
        reversed_chain = buildOutcomeTransform([standardize(), log()], 1)
        assert reversed_chain is not None

        with pytest.raises(OutcomeTransformError, match="strictly positive"):
            reversed_chain(y)

    def test_two_transforms_that_do_not_commute_give_different_results_in_each_order(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineSquare()
        y = torch.tensor([[1.5], [2.0], [4.0]], dtype=DOUBLE)
        square_then_log = buildOutcomeTransform([config("square_v1"), log()], 1)
        log_then_square = buildOutcomeTransform([log(), config("square_v1")], 1)
        assert square_then_log is not None and log_then_square is not None

        first, _ = square_then_log(y)
        second, _ = log_then_square(y)

        assert torch.allclose(first, 2 * torch.log(y))  # log(y^2)
        assert torch.allclose(second, torch.log(y) ** 2)  # (log y)^2
        assert not torch.allclose(first, second)

    @pytest.mark.parametrize("first", ["log_then_square", "square_then_log"])
    def test_the_transforms_are_undone_in_the_reverse_order(
        self, isolatedOutcomeTransformKind: KindTable, first: str
    ) -> None:
        # If `untransform` ran the transforms in the order they were applied, it would take the
        # square root of a logarithm, or the exponential of a square root, and not give y back.
        defineSquare()
        y = torch.tensor(
            [[1.5], [2.0], [4.0]], dtype=DOUBLE
        )  # y > 1: log y > 0, so (log y)^2 inverts
        steps = [log(), config("square_v1")]
        transform = buildOutcomeTransform(
            steps if first == "log_then_square" else steps[::-1], 1
        )
        assert transform is not None

        mapped, _ = transform(y)
        back, _ = transform.untransform(mapped)

        assert torch.allclose(back, y, rtol=1e-10)

    def test_a_chain_of_the_two_real_transforms_gives_the_outcome_back(self) -> None:
        y = toyData(n_outputs=2)[1]
        transform = buildOutcomeTransform([log(), standardize()], 2)
        assert transform is not None

        mapped, _ = transform(y)
        back, _ = transform.untransform(mapped)

        assert torch.allclose(back, y, rtol=1e-10)


class TestEachCallMakesNewTransforms:
    def test_two_builds_share_nothing(self) -> None:
        steps = [log(), standardize()]
        first = buildOutcomeTransform(steps, 1)
        second = buildOutcomeTransform(steps, 1)
        assert isinstance(first, ChainedOutcomeTransform)
        assert isinstance(second, ChainedOutcomeTransform)

        first(torch.tensor([[1.0], [2.0], [3.0]], dtype=DOUBLE))
        second(torch.tensor([[10.0], [20.0], [900.0]], dtype=DOUBLE))

        assert first is not second
        assert first["1_standardize_v1"] is not second["1_standardize_v1"]
        assert not torch.equal(
            first["1_standardize_v1"].means, second["1_standardize_v1"].means
        )

    def test_the_configurations_are_not_changed(self) -> None:
        steps = [log(), standardize(min_std=0.5)]
        before = [step.model_dump() for step in steps]

        buildOutcomeTransform(steps, 1)

        assert [step.model_dump() for step in steps] == before


class TestWhatIsRefused:
    def test_a_single_configuration_is_not_a_list(self) -> None:
        with pytest.raises(ConfigError) as caught:
            buildOutcomeTransform(standardize(), 1)  # type: ignore[arg-type]

        assert "single outcome transform configuration" in caught.value.message
        assert caught.value.suggestion == "wrap it in a list: [config]"

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, None, "2"])
    @pytest.mark.parametrize("which", ["empty", "none", "standardize"])
    def test_a_bad_number_of_outcomes_is_refused_whatever_the_list(
        self, bad: Any, which: str
    ) -> None:
        # Even for the empty list, where no transform is built to refuse it: a caller that
        # passes a wrong count finds out at once, and not when its list stops being empty.
        configs = {"empty": [], "none": [none()], "standardize": [standardize()]}[which]

        with pytest.raises(ConfigError, match="whole number of one or more"):
            buildOutcomeTransform(configs, bad)

    def test_a_transform_that_is_not_registered_lists_the_ones_that_are(self) -> None:
        unknown = ComponentConfig(type="warp_v1")

        with pytest.raises(
            RegistryError, match="Unknown outcome_transform 'warp_v1'"
        ) as caught:
            buildOutcomeTransform([log(), unknown], 1)

        for name in ("log_v1", "none_v1", "standardize_v1"):
            assert name in caught.value.message

    def test_a_faulty_plugin_transform_is_reported_by_name(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        @registerOutcomeTransform(
            "faulty_v1", config=fakeConfig("faulty_v1"), description="Makes no transform."
        )
        class FaultyBuilder(AbstractOutcomeTransformBuilder[Any]):
            def makeTransform(self, n_outcomes: int) -> Any:
                return object()

        with pytest.raises(OutcomeTransformError, match="made a object") as caught:
            buildOutcomeTransform([standardize(), config("faulty_v1")], 1)

        assert caught.value.component == "outcome_transform 'faulty_v1'"


class TestAPluginTransformInTheChain:
    def test_it_is_built_through_the_registry_like_the_real_ones(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake("winsorize_v1")

        transform = buildOutcomeTransform([log(), config("winsorize_v1")], 2)

        assert isinstance(transform, ChainedOutcomeTransform)
        assert list(transform.keys()) == ["0_log_v1", "1_winsorize_v1"]
        assert type(transform["1_winsorize_v1"]) is Standardize


class TestOnAToyModel:
    """The chain handed to a model, as a surrogate does."""

    # Measured on this toy: the posterior mean of the chain `log_v1`, `standardize_v1` is
    # within 0.07 of the observations at the training inputs, whose standard deviation is 2.6.
    # The tolerance is 10% of that deviation, as for each transform on its own.
    TOLERANCE = 0.25

    def test_the_model_is_fitted_to_the_standardized_logarithm(self) -> None:
        train_x, train_y, _, _ = toyData()
        transform = buildOutcomeTransform([log(), standardize()], 1)

        model = fitModel(transform, train_x, train_y)

        assert float(model.train_targets.mean()) == pytest.approx(0.0, abs=1e-9)
        assert float(model.train_targets.std()) == pytest.approx(1.0, abs=1e-9)
        expected, _ = Standardize(m=1)(torch.log(train_y))
        assert torch.allclose(model.train_targets, expected.squeeze(-1))

    def test_the_predictions_are_brought_back_through_both_transforms(self) -> None:
        train_x, train_y, _, _ = toyData()
        model = fitModel(buildOutcomeTransform([log(), standardize()], 1), train_x, train_y)
        far = torch.tensor([[-5.0, 7.0], [10.0, -10.0]], dtype=DOUBLE)

        with torch.no_grad():
            at_the_data = model.posterior(train_x)
            away = model.posterior(far)

        assert float((at_the_data.mean - train_y).abs().max()) < self.TOLERANCE
        assert bool((away.mean > 0).all())  # still a log-normal outcome after the chain
        assert bool((away.variance > 0).all())

    def test_the_predictions_of_two_outcomes(self) -> None:
        train_x, train_y, _, _ = toyData(n_outputs=2)
        model = fitModel(buildOutcomeTransform([log(), standardize()], 2), train_x, train_y)

        with torch.no_grad():
            posterior = model.posterior(train_x)

        assert posterior.mean.shape == train_y.shape
        assert float((posterior.mean - train_y).abs().max()) < self.TOLERANCE

    def test_a_list_with_nothing_to_apply_gives_a_model_that_does_not_rescale(self) -> None:
        train_x, train_y, _, _ = toyData()

        model = fitModel(buildOutcomeTransform([none()], 1), train_x, train_y)

        assert torch.equal(model.train_targets, train_y.squeeze(-1))
