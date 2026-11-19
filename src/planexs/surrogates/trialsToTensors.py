"""Turns the completed trials of a study into the tensors a surrogate is fitted on."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchDevice, torchDtype
from planexs.surrogates.NoisePolicyConfig import NoisePolicyConfig
from planexs.surrogates.SurrogateContextError import SurrogateContextError
from planexs.surrogates.TrainingTensors import TrainingTensors

logger = logging.getLogger(__name__)


def trialsToTensors(
    problem: Problem,
    encoder: SearchSpaceEncoder,
    trials: Sequence[Trial],
    runtime: RuntimeConfig | None = None,
    noise_policy: NoisePolicyConfig | None = None,
) -> TrainingTensors:
    """Encodes the completed trials of a history for a surrogate (specification 5.4, 5.6).

    This is where the observations of a study become the `train_inputs`, `train_outcomes` and
    `train_noise_std` of a `SurrogateBuildContext`, which checks them again when it is built.

    **Which trials.** Only completed trials are used: a pending, a failed and an abandoned
    trial has no observation to learn from. The rows are in the order the trials are given,
    not sorted by identifier, and a replicated evaluation of one parameterization is a row of
    its own (specification 5.4). A history with no completed trial gives tensors with no row.

    **Which columns.** The inputs are the trial's parameters as `encoder` encodes them. The
    outcomes are one column per outcome of `problem`, in the order of `problem.outcomes`
    (never in the order of an observation's own mapping), in the outcomes' own units; the
    direction in which an outcome is wanted is a matter of the goal, applied later. A value
    the problem does not name is ignored. A binary outcome is a column of `0.0` and `1.0`,
    as `Observation` stores it.

    **Noise.** The noise tensor holds the known standard deviations, not variances: the
    surrogate squares them if it wants variances. What is known is read per value, and only
    for the outcomes whose kind allows noise (`OutcomeKind.allows_noise`: a continuous
    outcome has a deviation, a binary one cannot). The three situations are:

    - no deviation is known (including when no outcome can have one): the tensor is `None`,
      and the surrogate infers the noise. This is not a problem and nothing is logged;
    - every deviation that can be known is: the tensor is returned, with `0.0` in the columns
      of outcomes that have no noise;
    - only some are (*partial* knowledge): `noise_policy.partial` decides. A Gaussian process
      cannot mix fixed and inferred noise, so under `ignore_with_warning` the known deviations
      are all dropped, the tensor is `None` and a warning is logged, and under `error` the
      data is refused.

    **Placement.** Every tensor has the dtype and lives on the device that `runtime` names,
    whatever the dtype of the encoder: that is what a build context requires.

    Args:
        problem: What can be varied and what is measured. It gives the outcomes, in order.
        encoder: The encoder of the problem's search space.
        trials: The history, in any state and in the order to keep.
        runtime: The precision and device of the tensors. `None` means `RuntimeConfig()`.
        noise_policy: What to do with partial knowledge of noise. `None` means
            `NoisePolicyConfig()`.

    Returns:
        The three tensors, with `n` rows, `n` being the number of completed trials.

    Raises:
        SurrogateContextError: if a completed trial has no value for an outcome of the
            problem, a trial's parameters cannot be encoded (a parameter is missing, a value
            is not one of the levels), or the noise is partial under the `error` policy. The
            message names the trial or the setting and says how to fix it.
        ConfigError: if torch has no type or no usable device of the names that `runtime`
            gives (see `torchDtype` and `torchDevice`).
    """
    runtime = RuntimeConfig() if runtime is None else runtime
    noise_policy = NoisePolicyConfig() if noise_policy is None else noise_policy
    dtype = torchDtype(runtime)
    device = torchDevice(runtime)

    names = [outcome.name for outcome in problem.outcomes]
    has_noise = [outcome.kind().allows_noise for outcome in problem.outcomes]
    completed = [trial for trial in trials if trial.state is TrialState.completed]

    rows: list[torch.Tensor] = []
    values: list[list[float]] = []
    deviations: list[list[float | None]] = []
    for trial in completed:
        observation = trial.observation
        assert observation is not None  # a completed trial always holds its observation
        missing = [name for name in names if name not in observation.values]
        if missing:
            raise SurrogateContextError(
                f"Trial {trial.id} is completed but its observation has no value for the "
                f"outcome(s) {missing!r}; the problem measures {names!r}.",
                component=f"trial {trial.id}",
                field_path="observation.values",
                suggestion="give each completed trial a value for every outcome",
            )
        try:
            rows.append(encoder.encodeParameters(trial.parameters))
        except ValueError as error:
            raise SurrogateContextError(
                f"Trial {trial.id} cannot be encoded: {error}",
                component=f"trial {trial.id}",
                field_path="parameters",
                suggestion="check the parameters of the trial against the search space",
            ) from error
        values.append([observation.values[name] for name in names])
        deviations.append(
            [
                observation.noise_std.get(name) if has_noise[position] else None
                for position, name in enumerate(names)
            ]
        )

    n_rows = len(completed)
    train_inputs = (
        torch.stack(rows).to(dtype=dtype, device=device)
        if rows
        else torch.empty(0, encoder.n_columns, dtype=dtype, device=device)
    )
    train_outcomes = torch.tensor(values, dtype=dtype, device=device).reshape(
        n_rows, len(names)
    )
    noise = _noiseStd(deviations, has_noise, noise_policy, dtype, device)
    return TrainingTensors(
        train_inputs=train_inputs, train_outcomes=train_outcomes, train_noise_std=noise
    )


def _noiseStd(
    deviations: Sequence[Sequence[float | None]],
    has_noise: Sequence[bool],
    policy: NoisePolicyConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Decides what to do with the known noise standard deviations, as the policy says.

    Args:
        deviations: For each completed trial and each outcome, the known standard deviation
            or `None`. It is `None` for every outcome whose kind has no noise.
        has_noise: For each outcome, whether its kind allows a noise standard deviation.
        policy: What to do with partial knowledge.
        dtype: The dtype of the tensor.
        device: The device of the tensor.

    Returns:
        A tensor of shape `(n, m)` if the noise is known for every value that can have it, and
        `None` if it is known for none, or only for some and the policy says to ignore that.

    Raises:
        SurrogateContextError: if the knowledge is partial and the policy says `error`.
    """
    n_values = len(deviations) * sum(has_noise)
    n_known = sum(deviation is not None for trial in deviations for deviation in trial)
    if n_known == 0:
        return None
    if n_known < n_values:
        if policy.partial == "error":
            raise SurrogateContextError(
                f"The observations give a noise standard deviation for only {n_known} of the "
                f"{n_values} values that can have one (one per outcome with noise, in each of "
                f"the {len(deviations)} completed trials): a surrogate cannot mix known and "
                "inferred noise.",
                component="noise policy",
                field_path="partial",
                suggestion=(
                    "give a noise standard deviation for every such value, or for none, or "
                    "set the `partial` field of the noise policy to 'ignore_with_warning' to "
                    "infer the noise instead"
                ),
            )
        logger.warning(
            "Only %d of %d noise standard deviations are known (one per outcome with noise, "
            "in each of the %d completed trials): ignoring them and inferring the "
            "observation noise instead, because a surrogate cannot mix known and inferred "
            "noise.",
            n_known,
            n_values,
            len(deviations),
        )
        return None
    filled = [[0.0 if item is None else item for item in trial] for trial in deviations]
    return torch.tensor(filled, dtype=dtype, device=device)
