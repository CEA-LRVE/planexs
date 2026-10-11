"""Sugar for a `Choice` of strings with no meaningful order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from planexs.domain.parameters.Choice import Choice, ChoiceValue


class Categorical(Choice):
    """`Categorical(name, categories)` is sugar for
    `Choice(name, categories, parameter_type="str", is_ordered=False)`.
    """

    def __init__(
        self,
        name: str,
        categories: list[str],
        dependent_parameters: Mapping[str, Sequence[str]] | None = None,
        default: str | None = None,
    ) -> None:
        """Creates a categorical parameter.

        Args:
            name: The parameter's name.
            categories: The legal values.
            dependent_parameters: Optional map from a category to the names of parameters that
                only exist when it is chosen.
            default: Optional default category.
        """
        # `Mapping` is invariant in its key type, so a Mapping[str, ...] is not a
        # Mapping[ChoiceValue, ...]: copy it into one.
        dependents: dict[ChoiceValue, Sequence[str]] | None = (
            {value: names for value, names in dependent_parameters.items()}
            if dependent_parameters is not None
            else None
        )
        super().__init__(
            name=name,
            values=categories,
            parameter_type="str",
            is_ordered=False,
            dependent_parameters=dependents,
            default=default,
        )
