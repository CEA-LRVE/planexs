"""Unit tests for `Prediction` (roadmap task P3-09, specification 5.6, 3.11)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from planexs.surrogates.Prediction import Prediction


def fields(**overrides: Any) -> dict[str, Any]:
    """The fields of a valid prediction of two outcomes at three points, with replacements."""
    values: dict[str, Any] = {
        "outcomes": ("yield", "cost"),
        "mean": {"yield": (1.0, 2.0, 3.0), "cost": (10.0, 20.0, 30.0)},
        "std": {"yield": (0.5, 0.25, 0.0), "cost": (1.0, 2.0, 3.0)},
        "space": "latent",
        "includes_observation_noise": False,
    }
    values.update(overrides)
    return values


class TestAPrediction:
    def test_it_holds_what_it_is_given(self) -> None:
        prediction = Prediction(**fields())

        assert prediction.outcomes == ("yield", "cost")
        assert prediction.mean["cost"] == (10.0, 20.0, 30.0)
        assert prediction.std["yield"] == (0.5, 0.25, 0.0)
        assert prediction.space == "latent"
        assert prediction.includes_observation_noise is False
        assert prediction.n_points == 3

    def test_its_variance_is_the_standard_deviation_squared(self) -> None:
        prediction = Prediction(**fields())

        assert prediction.variance == {
            "yield": (0.25, 0.0625, 0.0),
            "cost": (1.0, 4.0, 9.0),
        }

    def test_its_variance_is_a_new_dictionary_each_time(self) -> None:
        prediction = Prediction(**fields())

        prediction.variance["yield"] = (9.0,)

        assert prediction.variance["yield"] == (0.25, 0.0625, 0.0)

    def test_a_response_without_noise_is_legal(self) -> None:
        # The probability of a label (roadmap task P3-16) is a response with no noise.
        prediction = Prediction(**fields(space="response"))

        assert prediction.space == "response"
        assert prediction.includes_observation_noise is False

    def test_a_response_with_noise_is_legal(self) -> None:
        prediction = Prediction(**fields(space="response", includes_observation_noise=True))

        assert prediction.includes_observation_noise is True

    def test_it_accepts_lists_for_the_numbers_as_a_json_document_gives_them(self) -> None:
        prediction = Prediction(
            **fields(mean={"yield": [1.0], "cost": [2.0]}, std={"yield": [0.0], "cost": [0.0]})
        )

        assert prediction.mean["yield"] == (1.0,)


class TestAPredictionIsImmutable:
    @pytest.mark.parametrize("field", ["outcomes", "mean", "std", "space"])
    def test_no_field_can_be_assigned(self, field: str) -> None:
        prediction = Prediction(**fields())

        with pytest.raises(ValidationError):
            setattr(prediction, field, None)

    @pytest.mark.parametrize("field", ["mean", "std"])
    def test_the_mappings_are_read_only(self, field: str) -> None:
        prediction = Prediction(**fields())

        with pytest.raises(TypeError):
            getattr(prediction, field)["yield"] = (9.0, 9.0, 9.0)

    def test_the_caller_cannot_change_it_through_the_mapping_it_gave(self) -> None:
        mean = {"yield": [1.0, 2.0, 3.0], "cost": [10.0, 20.0, 30.0]}
        prediction = Prediction(**fields(mean=mean))

        mean["yield"] = [0.0]
        mean["extra"] = [0.0]

        assert prediction.mean["yield"] == (1.0, 2.0, 3.0)
        assert set(prediction.mean) == {"yield", "cost"}

    def test_equal_predictions_are_equal_and_hash_alike(self) -> None:
        first = Prediction(**fields())
        second = Prediction(**fields())

        assert first == second
        assert hash(first) == hash(second)
        assert first != Prediction(**fields(space="response"))
        assert len({first, second}) == 1


class TestAPredictionRoundTrips:
    def test_it_survives_json(self) -> None:
        prediction = Prediction(**fields(space="response", includes_observation_noise=True))

        text = prediction.model_dump_json()
        again = Prediction.model_validate_json(text)

        assert again == prediction
        assert json.loads(text)["mean"]["yield"] == [1.0, 2.0, 3.0]

    def test_its_dump_is_plain_data(self) -> None:
        dumped = Prediction(**fields()).model_dump()

        assert dumped["mean"] == {"yield": [1.0, 2.0, 3.0], "cost": [10.0, 20.0, 30.0]}
        assert type(dumped["mean"]) is dict
        assert Prediction.model_validate(dumped) == Prediction(**fields())


class TestAPredictionIsChecked:
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"outcomes": ()}, "outcomes is empty"),
            ({"outcomes": ("yield", "yield")}, "more than once"),
            ({"mean": {"yield": (1.0, 2.0, 3.0)}}, "mean has the outcomes"),
            (
                {"std": {"yield": (1.0,) * 3, "cost": (1.0,) * 3, "other": (1.0,) * 3}},
                "std has the outcomes",
            ),
            (
                {"mean": {"yield": (1.0, 2.0), "cost": (1.0, 2.0, 3.0)}},
                "same number of points",
            ),
            (
                {"std": {"yield": (1.0, 2.0), "cost": (1.0, 2.0)}},
                "same number of points",
            ),
            (
                {"mean": {"yield": (1.0, float("nan"), 3.0), "cost": (1.0,) * 3}},
                "mean['yield'] holds a value that is not finite",
            ),
            (
                {"std": {"yield": (float("inf"), 1.0, 1.0), "cost": (1.0,) * 3}},
                "std['yield'] holds a value that is not finite",
            ),
            (
                {"std": {"yield": (-0.1, 1.0, 1.0), "cost": (1.0,) * 3}},
                "std['yield'] holds a negative value",
            ),
            (
                {"space": "latent", "includes_observation_noise": True},
                "latent space includes no observation noise",
            ),
            ({"space": "posterior"}, "space"),
            ({"includes_observation_noise": "yes"}, "includes_observation_noise"),
        ],
    )
    def test_it_refuses_what_does_not_hold_together(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValidationError) as error:
            Prediction(**fields(**overrides))

        assert message in str(error.value)

    def test_it_refuses_an_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Prediction(**fields(sem={"yield": (1.0,)}))

    def test_it_needs_the_space_and_the_noise_flag(self) -> None:
        values = fields()
        del values["space"]
        with pytest.raises(ValidationError, match="space"):
            Prediction(**values)

        values = fields()
        del values["includes_observation_noise"]
        with pytest.raises(ValidationError, match="includes_observation_noise"):
            Prediction(**values)
