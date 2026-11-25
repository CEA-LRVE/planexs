"""Binds a fitted native model to the encoding and the outcomes it was fitted for."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.ParameterizationError import ParameterizationError
from planexs.domain.Problem import Problem
from planexs.encoding.ParameterEncoding import LevelValue
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchDevice, torchDtype
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.Prediction import Prediction
from planexs.surrogates.PredictionConfig import PredictionConfig
from planexs.surrogates.ResponseLink import ResponseLink
from planexs.surrogates.SamplingConfig import SamplingConfig
from planexs.surrogates.SurrogateModelError import SurrogateModelError

if TYPE_CHECKING:
    from botorch.models.model import Model

#: Who the errors of this module are about.
_COMPONENT = "surrogate"


class Surrogate:
    """A fitted native model, bound to the encoding of a problem and to the outcomes it models
    (specification 5.6, 3.11; FR7, FR8, FR22).

    A surrogate builder returns a `BuiltSurrogate`: a native model and how it was fitted. This
    wrapper adds what a caller needs to use it without knowing the engine: it takes
    parameterizations (what the caller sees, on natural values) and encodes them, and it gives
    back `Prediction` records that say what they describe. It works without a study: it needs
    only the problem, the encoder and the runtime settings that the model was fitted with.

    **What it offers.**

    - `predict`: the mean and standard deviation of each outcome at each parameterization, as
      a `Prediction`.
    - `sample`: joint posterior samples at the parameterizations, drawn from a generator that
      the caller supplies (specification 3.7: no global random state).
    - `latentLevel`: a level that is given in the response space, as the probability of a
      classifier, converted into the threshold on the latent function that a level-set
      acquisition or analysis works with.
    - `model`: the native model itself, never a copy or a different one (specification 5.13).
    - `status`: how the model was fitted, low-data flag included (specification 3.11).
    - `outcome_names` and `outcomes`: the outcomes it covers, in the order of the outputs of
      the model. A surrogate may cover only some of the outcomes of the problem.

    **What it assumes of the model.** The wrapper never imports a model class, so any native
    model that honours BoTorch's `Model` interface works, a fake one in a test included. It
    needs `num_outputs`, and `posterior(X, observation_noise=...)` returning a posterior that
    has `mean`, `variance`, `base_sample_shape` and `rsample_from_base_samples`. A model that
    answers otherwise, or with a number that is not finite, is reported as a
    `SurrogateModelError` and never passed on.

    **Space and noise.** Without observation noise a prediction describes the latent function
    of the model, its epistemic uncertainty only (the default, FR8). With it, the prediction
    describes a new noisy measurement and its space is `"response"`. Noise is refused for an
    outcome whose kind has none, a binary label. The probability that a classifier reads from
    its latent function through a link is a second kind of response: with `through_link` the
    prediction describes the outcome as the link of the surrogate reads it (the predictive
    probability and the standard deviation of the probability, for a probit link), its space
    is `"response"` and it includes no observation noise. A surrogate whose model reads its
    outcomes through a link says so in the `link` of its `BuiltSurrogate`; asking for the link
    of one that has none is an error, and so is combining it with noise, which a label has no
    meaning for.

    **Cost.** `predict` asks the model about each point on its own, so its cost grows
    linearly with the number of points. `sample` draws the points jointly, as one sample of a
    function must be, so the model builds the covariance of all of them and the cost grows
    with the cube of their number: keep a joint draw to a few thousand points.

    Every property of this class is a stored value that is read without touching the model,
    so they are all in snake_case; what asks the model something is a method
    (specification 10.2).

    Args:
        built: The fitted native model and its fit status, as a builder returns them.
        problem: The problem the model was fitted for. Parameterizations are checked against
            it.
        encoder: The encoder of the problem's search space that produced the training
            inputs. It must have been built with the same runtime settings.
        runtime: The runtime settings the model was fitted with: they choose the precision and
            the device of the tensors that the model is given.
        outcome_names: The names of the outcomes the model covers, in the order of its
            outputs. Each must be an outcome of the problem, named once.

    Raises:
        ConfigError: if `outcome_names` is empty, names an outcome twice or one that the
            problem does not have, the encoder encodes a parameter that the problem does not
            have, or its precision is not that of the runtime settings.
        SurrogateModelError: if the model has a number of outputs other than the number of
            outcomes.
    """

    def __init__(
        self,
        built: BuiltSurrogate,
        *,
        problem: Problem,
        encoder: SearchSpaceEncoder,
        runtime: RuntimeConfig,
        outcome_names: Sequence[str],
    ) -> None:
        """Binds the model, checking that everything fits together.

        Arguments and errors are described in the class docstring.
        """
        outcomes = _resolveOutcomes(problem, outcome_names)
        self._dtype = torchDtype(runtime)
        self._device = torchDevice(runtime)
        _checkEncoder(problem, encoder, self._dtype)
        n_outputs = built.model.num_outputs
        if n_outputs != len(outcomes):
            raise SurrogateModelError(
                f"The native model has {n_outputs} output(s), but the surrogate is bound to "
                f"{len(outcomes)} outcome(s) {[outcome.name for outcome in outcomes]!r}: the "
                "outputs of the model must be those outcomes, in the same order.",
                component=_COMPONENT,
                field_path="outcome_names",
                suggestion="bind the surrogate to the outcomes the model was fitted on, in "
                "the order of its outputs",
            )
        self._built = built
        self._problem = problem
        self._encoder = encoder
        self._outcomes = outcomes

    @property
    def model(self) -> Model:
        """The native model: the one that predictions are read from, never a copy (a stored
        value: cheap, snake_case). It is outside what planexs validates (specification 5.13).
        """
        return self._built.model

    @property
    def status(self) -> FitStatus:
        """How the model was fitted, with its low-data flag (a stored value: cheap,
        snake_case).
        """
        return self._built.status

    @property
    def link(self) -> ResponseLink | None:
        """How the outcomes are read from the latent function of the model, or `None` when the
        model's own output is the outcome (a stored value: cheap, snake_case).
        """
        return self._built.link

    @property
    def problem(self) -> Problem:
        """The problem the model was fitted for (a stored value: cheap, snake_case)."""
        return self._problem

    @property
    def encoder(self) -> SearchSpaceEncoder:
        """The encoder that binds the problem to the inputs of the model (a stored value:
        cheap, snake_case).
        """
        return self._encoder

    @property
    def outcomes(self) -> tuple[Outcome, ...]:
        """The outcomes the model covers, in the order of its outputs (a stored value: cheap,
        snake_case).
        """
        return self._outcomes

    @property
    def outcome_names(self) -> tuple[str, ...]:
        """The names of the outcomes the model covers, in the order of its outputs (a short
        tuple built from the stored outcomes: cheap, snake_case).
        """
        return tuple(outcome.name for outcome in self._outcomes)

    def predict(
        self,
        parameterizations: Sequence[Mapping[str, LevelValue]],
        options: PredictionConfig | None = None,
    ) -> Prediction:
        """Predicts the outcomes at parameterizations.

        Each parameterization is checked against the problem, encoded, and given to the model
        as a point of its own, so the standard deviations are those of each point and not of
        a joint distribution.

        Args:
            parameterizations: The points, each a mapping from parameter name to value on
                natural values, legal for the problem (every constraint included). At least
                one.
            options: Whether the observation noise is included and whether the prediction is
                read through the link. `None` means `PredictionConfig()`, the defaults of the
                schema.

        Returns:
            The prediction, in the order of the points. It states its space (`"latent"`
            by default, `"response"` with noise or through the link) and whether the noise
            is included.

        Raises:
            ConfigError: if `parameterizations` is empty or is a single mapping, noise is
                asked for an outcome whose kind has none, or the link is asked for of a
                surrogate that has none or together with noise.
            ParameterizationError: if a parameterization is not legal for the problem. The
                message names its position.
            SurrogateModelError: if the model answers with a shape that does not fit the
                points, or with a number that is not finite.
        """
        settings = PredictionConfig() if options is None else options
        noisy = settings.include_observation_noise
        linked = settings.through_link
        self._requireLink(linked, noisy)
        self._requireNoiseMeaningful(noisy)
        inputs = self._encodeAll(parameterizations)
        expected = (inputs.shape[0], 1, len(self._outcomes))
        with torch.no_grad():
            # One batch of single points: `(n, 1, d)` is n batches of one point, so the model
            # gives the marginal of each point, and the cost stays linear in n.
            posterior: Any = self._built.model.posterior(
                inputs.unsqueeze(1), observation_noise=noisy
            )
            mean = self._checkedTable(posterior.mean, expected, "mean")
            variance = self._checkedTable(posterior.variance, expected, "variance")
        if linked:
            mean, std = self._requiredLink().marginal(mean, variance.clamp_min(0.0))
        else:
            std = variance.clamp_min(0.0).sqrt()
        return Prediction(
            outcomes=self.outcome_names,
            mean={name: tuple(mean[:, column].tolist()) for column, name in self._columns()},
            std={name: tuple(std[:, column].tolist()) for column, name in self._columns()},
            space="response" if noisy or linked else "latent",
            includes_observation_noise=noisy,
        )

    def sample(
        self,
        parameterizations: Sequence[Mapping[str, LevelValue]],
        options: SamplingConfig | None = None,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Draws joint posterior samples of the outcomes at parameterizations.

        Each sample is one draw of the outcomes at all the points together, so it keeps the
        correlation between points that the model believes in. The draws are made from base
        samples that the given generator produces, so the same generator state gives the same
        samples, and the random state of torch is never read or changed (specification 3.7).
        The generator advances: a second call with it gives other samples.

        What the samples describe is what `options` says: the latent function without
        observation noise, a new noisy measurement with it, or the response that the link
        reads from each draw of the latent function with `through_link` (see
        `PredictionConfig`).

        Args:
            parameterizations: The points, as for `predict`. At least one.
            options: The number of samples, whether the noise is included and whether the
                draws are read through the link. `None` means `SamplingConfig()`, the
                defaults of the schema.
            generator: The seeded generator the base samples are drawn from. The study gives
                each call one that it derived for the ask and the component.

        Returns:
            A tensor of shape `(n_samples, n_points, n_outcomes)`, in the units of the
            outcomes (probabilities through a probit link), with the precision and on the
            device of the runtime settings.

        Raises:
            ConfigError: if `parameterizations` is empty or is a single mapping, noise is
                asked for an outcome whose kind has none, the link is asked for of a
                surrogate that has none or together with noise, or `generator` is not a
                `torch.Generator`.
            ParameterizationError: if a parameterization is not legal for the problem.
            SurrogateModelError: if the model answers with a shape that does not fit the
                points, or the samples hold a number that is not finite.
        """
        if not isinstance(generator, torch.Generator):
            raise ConfigError(
                f"sample needs a seeded torch.Generator; got {type(generator).__name__}. "
                "Drawing from torch's global random state would make the samples depend on "
                "whatever ran before.",
                component=_COMPONENT,
                field_path="generator",
                suggestion="pass torchGenerator(seed, runtime), with a seed from RngTree.seed",
            )
        settings = SamplingConfig() if options is None else options
        self._requireLink(settings.through_link, settings.include_observation_noise)
        self._requireNoiseMeaningful(settings.include_observation_noise)
        inputs = self._encodeAll(parameterizations)
        shape = torch.Size([settings.n_samples])
        expected = (settings.n_samples, inputs.shape[0], len(self._outcomes))
        with torch.no_grad():
            posterior: Any = self._built.model.posterior(
                inputs, observation_noise=settings.include_observation_noise
            )
            base_samples = torch.randn(
                torch.Size([*shape, *posterior.base_sample_shape]),
                generator=generator,
                dtype=self._dtype,
                device=generator.device,
            ).to(self._device)
            draws = posterior.rsample_from_base_samples(shape, base_samples)
        samples = self._checkedTable(draws, expected, "samples")
        return self._requiredLink().inverse(samples) if settings.through_link else samples

    def latentLevel(self, level: float) -> float:
        """Converts a level of the response into the threshold on the latent function.

        A level-set goal gives the level in the space of the outcome: in the outcome's units
        for a continuous one, as a probability for a binary one. A level-set acquisition or
        analysis compares the posterior of the latent function with it, so the level has to be
        read in that space first: the latent threshold is the value of the latent function at
        which the response equals the level, `link.forward(level)` (for a probit link,
        `Phi^-1(level)`, which is 0 for the level 0.5). A surrogate with no link models the
        outcome itself, and its latent function is in the units of the outcome, so the level
        is returned as it is.

        Args:
            level: The level in the response space. For a surrogate with a link it must be
                strictly between 0 and 1: the response never reaches the ends, so no latent
                value has a response of 0 or 1.

        Returns:
            The threshold, in the units of the latent function.

        Raises:
            ConfigError: if the level is not a finite number, or, for a surrogate with a link,
                is not strictly between 0 and 1.
        """
        link = self._built.link
        if isinstance(level, bool) or not math.isfinite(level):
            raise ConfigError(
                f"The level must be a finite number; got {level!r}.",
                component=_COMPONENT,
                field_path="level",
                suggestion="give the level as a number, a probability for a binary outcome",
            )
        if link is None:
            return float(level)
        if not 0.0 < level < 1.0:
            raise ConfigError(
                f"The level {level!r} is not strictly between 0 and 1, so no value of the "
                f"latent function has it as its response: the {link.name} link reads the "
                "latent function as a probability, which never reaches 0 or 1.",
                component=_COMPONENT,
                field_path="level",
                suggestion="give a level for the probability of the label, between 0 and 1 "
                "exclusive, such as 0.5",
            )
        probability = torch.tensor(level, dtype=self._dtype, device=self._device)
        return float(link.forward(probability))

    def _requiredLink(self) -> ResponseLink:
        """The link of the surrogate, known to exist because `_requireLink` has checked.

        Returns:
            The link.
        """
        link = self._built.link
        assert link is not None  # `_requireLink` refused the request otherwise
        return link

    def _requireLink(self, through_link: bool, include_noise: bool) -> None:
        """Refuses a request for the response through a link that this surrogate cannot give.

        Args:
            through_link: Whether the caller asked for the response through the link.
            include_noise: Whether the caller asked for the observation noise.

        Raises:
            ConfigError: if the link is asked for of a surrogate that has none, or together
                with noise.
        """
        if not through_link:
            return
        if self._built.link is None:
            raise ConfigError(
                "The response through a link was asked for, but this surrogate has no link: "
                "its model gives the outcome itself, so there is no further space to read "
                "it in.",
                component=_COMPONENT,
                field_path="through_link",
                suggestion="leave through_link False; the observation noise is what turns the "
                "latent function of a Gaussian outcome into a measurement",
            )
        if include_noise:
            raise ConfigError(
                "through_link and include_observation_noise were both asked for: the "
                "response that a link gives is not a noisy measurement of the latent "
                "function, and a label has no observation noise.",
                component=_COMPONENT,
                field_path="through_link",
                suggestion="ask for one of them",
            )

    def _columns(self) -> list[tuple[int, str]]:
        """The position of each outcome among the outputs of the model, with its name."""
        return list(enumerate(self.outcome_names))

    def _requireNoiseMeaningful(self, include_noise: bool) -> None:
        """Refuses observation noise for an outcome whose kind has none.

        The kind says it (`allows_noise`), and no kind is named here, so a kind that a plugin
        registers is judged like the built-in ones.

        Args:
            include_noise: Whether the caller asked for the noise to be included.

        Raises:
            ConfigError: if noise is asked and an outcome's kind has none.
        """
        if not include_noise:
            return
        without = [
            outcome.name for outcome in self._outcomes if not outcome.kind().allows_noise
        ]
        if without:
            raise ConfigError(
                f"Observation noise was asked for, but it is not meaningful for the "
                f"outcome(s) {without!r}: their kind has no observation noise.",
                component=_COMPONENT,
                field_path="include_observation_noise",
                suggestion="leave include_observation_noise False for this surrogate",
            )

    def _encodeAll(self, parameterizations: Sequence[Mapping[str, LevelValue]]) -> Tensor:
        """Checks and encodes the points, in order.

        Args:
            parameterizations: The points on natural values.

        Returns:
            A tensor of shape `(n, d)` with the precision of the encoder, on the device of
            the runtime settings.

        Raises:
            ConfigError: if there is no point, or a single mapping was given.
            ParameterizationError: if a point is not legal for the problem.
        """
        if isinstance(parameterizations, Mapping):
            raise ConfigError(
                "A single parameterization was given where a sequence of them is expected.",
                component=_COMPONENT,
                field_path="parameterizations",
                suggestion="wrap it in a list: [parameters]",
            )
        points = list(parameterizations)
        if not points:
            raise ConfigError(
                "No parameterization was given: there is nothing to ask the model about.",
                component=_COMPONENT,
                field_path="parameterizations",
                suggestion="give at least one parameterization",
            )
        rows: list[Tensor] = []
        for position, parameters in enumerate(points):
            try:
                self._problem.validateParameterization(parameters)
            except ParameterizationError as error:
                raise ParameterizationError(
                    f"Parameterization {position} of {len(points)}: {error.message}",
                    component=error.component,
                    field_path=error.field_path,
                    suggestion=error.suggestion,
                ) from error
            rows.append(self._encoder.encodeParameters(parameters))
        return torch.stack(rows).to(self._device)

    def _checkedTable(self, values: Tensor, expected: tuple[int, ...], what: str) -> Tensor:
        """Checks what the model returned, and lays it out as `(points, outcomes)` or, for
        samples, as it is.

        Args:
            values: A posterior mean or variance, or a tensor of samples.
            expected: The shape it must have.
            what: What it is, for the message: `mean`, `variance` or `samples`.

        Returns:
            The values with the leading singleton axis of a prediction dropped. Samples are
            returned unchanged.

        Raises:
            SurrogateModelError: if the shape is not the expected one, or a value is not
                finite.
        """
        if tuple(values.shape) != expected:
            raise SurrogateModelError(
                f"The posterior {what} has the shape {tuple(values.shape)}, but the points "
                f"asked about need {expected}: one entry per point and per outcome.",
                component=_COMPONENT,
                field_path=what,
                suggestion="check that the model answers posterior(X) for a batch of single "
                "points, with one output for each outcome the surrogate is bound to",
            )
        if not bool(torch.isfinite(values).all()):
            raise SurrogateModelError(
                f"The posterior {what} holds a value that is not finite (nan or an "
                "infinity): a surrogate never passes such a number on.",
                component=_COMPONENT,
                field_path=what,
                suggestion="refit the model, or check that the points are inside the "
                "problem's bounds",
            )
        if what == "samples":
            return values
        return values.reshape(expected[0], expected[-1])


def _resolveOutcomes(problem: Problem, outcome_names: Sequence[str]) -> tuple[Outcome, ...]:
    """Finds the outcomes of the problem that a surrogate is bound to, in the given order.

    Args:
        problem: The problem.
        outcome_names: The names the surrogate is bound to.

    Returns:
        The outcomes.

    Raises:
        ConfigError: if no name is given, a name repeats, or a name is not an outcome of the
            problem.
    """
    names = list(outcome_names)
    by_name = {outcome.name: outcome for outcome in problem.outcomes}
    if not names:
        raise ConfigError(
            "A surrogate covers at least one outcome, and outcome_names is empty.",
            component=_COMPONENT,
            field_path="outcome_names",
            suggestion=f"name the outcomes the model was fitted on, among {list(by_name)!r}",
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigError(
            f"outcome_names names {repeated!r} more than once: each output of the model is "
            "one outcome.",
            component=_COMPONENT,
            field_path="outcome_names",
            suggestion="name each outcome once",
        )
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ConfigError(
            f"outcome_names has {unknown!r}, which {'is' if len(unknown) == 1 else 'are'} "
            f"not among the outcomes of the problem {list(by_name)!r}.",
            component=_COMPONENT,
            field_path="outcome_names",
            suggestion="check the spelling, or add the outcome to the problem",
        )
    return tuple(by_name[name] for name in names)


def _checkEncoder(problem: Problem, encoder: SearchSpaceEncoder, dtype: torch.dtype) -> None:
    """Checks that the encoder belongs to the problem and has the precision of the runtime.

    Args:
        problem: The problem.
        encoder: The encoder.
        dtype: The floating-point type that the runtime settings name.

    Raises:
        ConfigError: if the encoder encodes a parameter that the problem does not have, or
            builds tensors of another precision (BoTorch does not mix precisions).
    """
    known = set(problem.search_space.parameter_names)
    foreign = sorted({encoding.name for encoding in encoder.encodings} - known)
    if foreign:
        raise ConfigError(
            f"The encoder encodes {foreign!r}, which "
            f"{'is' if len(foreign) == 1 else 'are'} not among the parameters of the problem "
            f"{sorted(known)!r}: it was built for another search space.",
            component=_COMPONENT,
            field_path="encoder",
            suggestion="build the encoder from problem.search_space",
        )
    if encoder.dtype != dtype:
        raise ConfigError(
            f"The encoder builds {encoder.dtype} tensors, but the runtime settings say "
            f"{dtype}: the model was fitted in one precision and would be asked in another.",
            component=_COMPONENT,
            field_path="encoder",
            suggestion="build the encoder with the same RuntimeConfig as the surrogate",
        )
