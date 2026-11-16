"""The observations of a study, as the tensors a surrogate is fitted on."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, eq=False, kw_only=True)
class TrainingTensors:
    """The three tensors that `trialsToTensors` makes from the completed trials of a study.

    Its fields have the names, shapes and meanings of the fields of the same names of
    `SurrogateBuildContext`, which is what they are meant to fill: `n` is the number of
    completed trials, `d` the number of encoded columns of the encoder and `m` the number of
    outcomes of the problem. Row `i` of every tensor is the `i`-th completed trial of the
    history, in the order it was given.

    A frozen record of tensors: no field can be assigned, but the tensors themselves are not
    copied, and a reader must not modify them in place. Records are compared by identity
    (`eq=False`), as the build context is: equality of tensors is not a boolean.

    Attributes:
        train_inputs: The encoded parameterizations, shape `(n, d)`, in the unit cube.
        train_outcomes: The observed values, shape `(n, m)`, one column per outcome in the
            order of `problem.outcomes`, in the outcomes' own units. A label of a binary
            outcome is `0.0` or `1.0`.
        train_noise_std: The known noise standard deviations, shape `(n, m)`, in the units of
            the outcomes, or `None` when the noise is not known (or was left out, as the
            noise policy says). In the column of an outcome whose kind has no noise it holds
            `0.0`, which is not a claim that the value is exact: a surrogate of such an
            outcome never reads it.
    """

    train_inputs: Tensor
    train_outcomes: Tensor
    train_noise_std: Tensor | None
