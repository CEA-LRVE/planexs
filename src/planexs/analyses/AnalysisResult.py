"""The typed, serializable result of an analysis."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class AnalysisResult(BaseModel):
    """What an analysis returns (specification 5.10): a typed, immutable, serializable value.

    The base class holds what every result says about itself: the registered name of the
    analysis that made it (`analysis`) and the version of the study's history that it was
    computed on (`history_version`), so a reader who keeps a result can tell what it is and
    whether the study has changed since. An analysis defines a subclass with the fields of its
    own answer, each typed: a best point has a trial identifier, a Pareto front a tuple of
    them, a prediction grid tuples of numbers. Several analyses may share one result class,
    which is why the result names the analysis and not the class.

    **Serialization.** A result is data. `toDict` gives a plain dictionary that holds only
    dicts, lists, strings, numbers, booleans and `None`, so `json.dumps(result.toDict())`
    always works, and `fromDict`, called on the class of the result, gives back a result that
    is equal to the original (the round trip is exact). A number that is not finite is refused
    when the result is built, wherever it is, in a tuple or in a nested model as well, because
    JSON has none and a stored result would not be the one that was computed. Arrays are
    tuples of numbers, not tensors: a result never holds an object that needs code to be read
    back.

    **Immutability.** A result is frozen, and so is everything in it: assigning a field is an
    error, and so is a field that holds a list, a dictionary or a set, or a model that is not
    frozen itself, at any depth, because a frozen model forbids assigning a field and not
    changing a list that the field holds. Use tuples, and nested models that are frozen. A
    study may keep a result and hand it to several callers: none of them can change what the
    others read.

    Results are built with keywords, `BestPointResult(analysis="...", history_version=3, ...)`.
    A result is compared by value.

    Attributes:
        analysis: The registered name of the analysis that computed the result.
        history_version: The version of the study's history that the result was computed on:
            the `history_version` of the state the analysis was given. A non-negative integer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    analysis: str = Field(
        strict=True,
        description=(
            "The registered name of the analysis that computed this result, for example "
            "'best_observed_v1'. It is what tells a reader of a stored result which analysis "
            "it belongs to."
        ),
    )
    history_version: int = Field(
        ge=0,
        strict=True,
        description=(
            "The version of the study's history that the result was computed on. Two states "
            "with the same version hold the same trials in the same states, so a result is "
            "out of date exactly when the study's version is another one."
        ),
    )

    @model_validator(mode="after")
    def _requireSoundValues(self) -> Self:
        """Refuses a field that holds something that can be changed, or a number that JSON
        cannot hold, at any depth.

        Returns:
            The result itself.

        Raises:
            ValueError: naming the field, if one holds a list, a dictionary, a set or a model
                that is not frozen, or a float that is not finite, inside tuples and nested
                models as well.
        """
        for name in type(self).model_fields:
            parts = list(_parts(getattr(self, name)))
            mutable = next((what for part in parts if (what := _mutable(part))), None)
            if mutable is not None:
                raise ValueError(
                    f"The field {name!r} of {type(self).__name__} holds {mutable}, which can "
                    f"be changed. A result is immutable: use a tuple, and nested models that "
                    f"are frozen."
                )
            bad = next(
                (
                    part
                    for part in parts
                    if isinstance(part, float) and not math.isfinite(part)
                ),
                None,
            )
            if bad is not None:
                raise ValueError(
                    f"The field {name!r} of {type(self).__name__} holds {bad!r}, which is "
                    f"not finite. JSON has no such number, so a stored result would not be "
                    f"the one that was computed."
                )
        return self

    def toDict(self) -> dict[str, JsonValue]:
        """The result as plain JSON-compatible data.

        Returns:
            A new dictionary that holds only dictionaries, lists, strings, numbers, booleans
            and `None`, with finite numbers. A tuple is written as a list.
        """
        return self.model_dump(mode="json")

    @classmethod
    def fromDict(cls, data: Mapping[str, Any]) -> Self:
        """A result back from the data that `toDict` gave, or that went through JSON.

        Call it on the class of the result: `BestPointResult.fromDict(data)`. A list is read
        as a tuple where the field is a tuple, so the result is equal to the original.

        Args:
            data: The mapping to read.

        Returns:
            The result.

        Raises:
            ValidationError: if the data is not a result of this class: a field is missing, of
                the wrong type or not finite, there is a field the class does not have, or the
                data holds something that can be changed. It names the field.
        """
        return cls.model_validate(dict(data))


def _parts(value: object) -> Iterator[object]:
    """A value and everything inside it that a result holds: the items of a tuple and the
    fields of a nested model, at any depth. A list, a dictionary or a set is not opened: it is
    refused as it is.

    Args:
        value: The value of a field.

    Yields:
        The value, then each part of it, depth first.
    """
    yield value
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _parts(getattr(value, name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _parts(item)


def _mutable(value: object) -> str | None:
    """Says what a value is if it can be changed.

    Args:
        value: One part of a field.

    Returns:
        A short description (`a list`, `a Band that is not frozen`), or `None` if the value
        cannot be changed.
    """
    if isinstance(value, (list, dict, set, bytearray)):
        return f"a {type(value).__name__}"
    if isinstance(value, BaseModel) and not value.model_config.get("frozen", False):
        return f"a {type(value).__name__} that is not frozen"
    return None
