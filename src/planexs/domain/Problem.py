"""A search space together with the outcomes that are measured."""

from __future__ import annotations

import math
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from planexs.core.ConfigError import ConfigError
from planexs.domain.Outcome import Outcome
from planexs.domain.ParameterizationCheckConfig import ParameterizationCheckConfig
from planexs.domain.ParameterizationError import ParameterizationError
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.ParameterValueError import ParameterValueError
from planexs.domain.ProblemFacts import ProblemFacts
from planexs.domain.SearchSpace import SearchSpace

#: What is wrong with a parameterization, as `(parameter name or None, sentence, suggestion)`.
_Fault = tuple[str | None, str, str]


class Problem(BaseModel):
    """What can be varied and what is measured (specification 4, 5.3).

    A problem is a search space (its parameters and the constraints on them) together with the
    outcomes that are measured. It is the thing that a goal is validated against, that a
    parameterization is checked against, and whose `facts` say which capabilities a study of it
    needs (specification 5.12). It is pure data: it knows nothing of any engine.

    A problem holds its own copy of the search space it is given, so that adding a parameter to
    that search space afterwards cannot change a problem that something has already checked
    parameterizations against. It is frozen: treat `search_space` as read-only too, since a
    `SearchSpace` is a mutable object (it is built up by `addParameter`) and a problem cannot
    freeze it.

    It is checked when it is built: there is at least one outcome, outcome names are unique,
    and every constraint is about parameters of the search space. A constraint that names a
    parameter that does not exist is a typo, and a constraint that is never checked is worse
    than none, so it is refused here and not found by chance later.

    It is built with keywords, `Problem(search_space=space, outcomes=[...])`, and `outcomes`
    may be any sequence of outcomes. Anything that is wrong with the arguments is reported as
    pydantic reports it, with its location, and the three checks above raise `ConfigError`.

    Serialization is pydantic's: `model_dump_json` and `model_validate_json` round-trip it, the
    parameters and constraints through their registered types and the outcomes through theirs,
    so a type that a plugin registers is saved and reloaded like the built-in ones. A sugar
    constructor (`Real`, `Fixed`, ...) comes back as the type it is sugar for (specification
    5.3), so a reloaded problem is equal to the original once it has been loaded once, and its
    dump is equal from the start. A document that names a type that is not registered fails
    with a `ConfigError` that lists the registered ones, and one with a field that does not
    exist fails with the name of the field.

    Attributes:
        search_space: The parameters and the constraints on them.
        outcomes: The measured quantities, in order. At least one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    search_space: SearchSpace
    outcomes: tuple[Outcome, ...]

    @field_validator("search_space")
    @classmethod
    def _copySearchSpace(cls, value: SearchSpace) -> SearchSpace:
        """Takes a copy of the search space, so that the caller's own cannot change this one.

        Args:
            value: The search space as given.

        Returns:
            A deep copy of it.
        """
        return value.model_copy(deep=True)

    @model_validator(mode="after")
    def _validateOutcomes(self) -> Problem:
        """Checks that there is an outcome and that outcome names are unique.

        Returns:
            The problem itself.

        Raises:
            ConfigError: if either check fails.
        """
        if not self.outcomes:
            raise ConfigError(
                "A problem needs at least one outcome: without a measured quantity there is "
                "nothing to learn from a trial.",
                component="problem",
                field_path="outcomes",
                suggestion="add an Outcome, e.g. Outcome(name='y', type='continuous')",
            )
        names = [outcome.name for outcome in self.outcomes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigError(
                f"A problem has duplicate outcome names: {duplicates!r}.",
                component="problem",
                field_path="outcomes",
                suggestion="give every outcome a name of its own",
            )
        return self

    @model_validator(mode="after")
    def _validateConstraintNames(self) -> Problem:
        """Checks that every constraint is about parameters of the search space.

        Returns:
            The problem itself.

        Raises:
            ConfigError: if a constraint names a parameter that the search space does not have.
        """
        known = set(self.search_space.parameter_names)
        for constraint in self.search_space.constraints:
            unknown = sorted(constraint.referencedNames() - known)
            if unknown:
                raise ConfigError(
                    f"The constraint {constraint!r} is about {unknown!r}, which "
                    f"{'is' if len(unknown) == 1 else 'are'} not among the parameters of the "
                    f"search space {sorted(known)!r}.",
                    component="problem",
                    field_path="search_space.constraints",
                    suggestion="check the spelling, or add the parameter to the search space",
                )
        return self

    def facts(self) -> ProblemFacts:
        """What this problem contains, as plain data.

        Returns:
            The facts: the number of outcomes, the types that occur, and whether there are
            conditional parameters, equality constraints or categorical parameters.
        """
        return ProblemFacts.of(self)

    def validateParameterization(
        self,
        parameters: Mapping[str, object],
        options: ParameterizationCheckConfig | None = None,
    ) -> None:
        """Checks that a parameterization is legal for this problem.

        Legal means, for a parameterization on natural values, the one that the caller sees:

        - every name in it is a parameter of the search space, and every value is legal for
          its parameter (each parameter type says what it accepts: `validateValue`);
        - every parameter that is active has a value, except a computed one (`Derived`), which
          the caller does not have to give; and a parameter that is not active is absent. A
          parameter is active if no choice switches it on or off, or if a choice that is
          itself active has been given a value that switches it on (conditional parameters,
          FR14);
        - every constraint holds, within `options.constraint_tolerance`.

        The three are judged in this order, and the faults of the first stage that finds any
        are all reported together: a constraint is not evaluated on a value that is not legal,
        and what is missing is not known until every given value is legal. A constraint is
        judged only if every parameter it names is there. One that names a parameter that is
        inactive does not apply to this parameterization, and one that names a computed
        parameter that was not given cannot be judged, which is reported.

        Args:
            parameters: Parameter name to value, as the caller gives it.
            options: The tolerances of the check. `None` means the defaults of
                `ParameterizationCheckConfig`.

        Raises:
            ParameterizationError: if the parameterization is not legal. The message names
                every fault of the first stage that finds any, and says how to fix them.
        """
        if not isinstance(parameters, Mapping):
            raise ParameterizationError(
                f"A parameterization must be a mapping from parameter names to values; got "
                f"{type(parameters).__name__}.",
                component="problem",
                suggestion="give a dictionary such as {'temperature': 50.0}",
            )
        checked = options if options is not None else ParameterizationCheckConfig()
        faults = self._valueFaults(parameters, checked)
        if not faults:
            active = self._activeNames(parameters)
            faults = self._presenceFaults(parameters, active)
            if not faults:
                faults = self._constraintFaults(parameters, active, checked)
        if faults:
            raise _error(faults)

    def _valueFaults(
        self, parameters: Mapping[str, object], options: ParameterizationCheckConfig
    ) -> list[_Fault]:
        """The names that are not parameters and the values that are not legal for theirs.

        Args:
            parameters: The parameterization.
            options: The tolerances.

        Returns:
            The faults: the unknown names first, in the order given, then the illegal values
            in the order the parameters are declared.
        """
        known = self.search_space.parameter_names
        faults: list[_Fault] = [
            (
                str(name),
                f"The name {name!r} is not a parameter of the search space {known!r}.",
                "check the spelling, or leave it out",
            )
            for name in parameters
            if name not in known
        ]
        for parameter in self.search_space.parameters:
            if parameter.name not in parameters:
                continue
            try:
                parameter.validateValue(parameters[parameter.name], options)
            except ParameterValueError as error:
                faults.append(
                    (parameter.name, error.message, error.suggestion or "fix the value")
                )
        return faults

    def _activeNames(self, parameters: Mapping[str, object]) -> set[str]:
        """The names of the parameters that are active, given the values of the others.

        A parameter that no choice mentions is always active. One that a choice mentions is
        active if that choice is active and has been given a value that switches it on, which
        can go several levels deep. Only values that are already known to be legal are looked
        up.

        Args:
            parameters: The parameterization, whose given values are all legal.

        Returns:
            The active names.
        """
        declared = self.search_space.parameters
        dependents = {
            name
            for parameter in declared
            for names in parameter.activations().values()
            for name in names
        }
        active = {parameter.name for parameter in declared} - dependents
        changed = True
        while changed:
            changed = False
            for parameter in declared:
                if parameter.name not in active or parameter.name not in parameters:
                    continue
                switched_on = parameter.activations().get(parameters[parameter.name], ())
                for name in switched_on:
                    if name not in active:
                        active.add(name)
                        changed = True
        return active

    def _presenceFaults(
        self, parameters: Mapping[str, object], active: set[str]
    ) -> list[_Fault]:
        """The active parameters that have no value and the inactive ones that have one.

        Args:
            parameters: The parameterization.
            active: The active names.

        Returns:
            The faults, in the order the parameters are declared.
        """
        faults: list[_Fault] = []
        for parameter in self.search_space.parameters:
            name = parameter.name
            if name in active:
                if name not in parameters and not parameter.is_computed:
                    faults.append(
                        (name, f"The parameter {name!r} has no value.", "give it a value")
                    )
            elif name in parameters:
                faults.append(
                    (
                        name,
                        f"The parameter {name!r} is given, but it only exists "
                        f"{self._whenActive(parameter)}.",
                        "leave it out, or change the parameter that switches it on",
                    )
                )
        return faults

    def _whenActive(self, parameter: Parameter) -> str:
        """Says which values of which parameters switch a parameter on, for a message.

        Args:
            parameter: A parameter that is the dependent of at least one choice.

        Returns:
            For example `when 'model' is 'tree' or when 'kind' is 'deep'`.
        """
        causes = [
            f"when {other.name!r} is {value!r}"
            for other in self.search_space.parameters
            for value, names in other.activations().items()
            if parameter.name in names
        ]
        return " or ".join(causes)

    def _constraintFaults(
        self,
        parameters: Mapping[str, object],
        active: set[str],
        options: ParameterizationCheckConfig,
    ) -> list[_Fault]:
        """The constraints that do not hold, or cannot be judged.

        Args:
            parameters: The parameterization, whose values are all legal and complete.
            active: The active names.
            options: The tolerances.

        Returns:
            The faults, in the order the constraints are declared.
        """
        faults: list[_Fault] = []
        for constraint in self.search_space.constraints:
            names = constraint.referencedNames()
            absent = sorted(names - parameters.keys())
            if absent and not set(absent) <= active:
                continue  # about a parameter that does not exist in this parameterization
            if absent:
                faults.append(
                    (
                        absent[0],
                        f"The constraint {constraint!r} cannot be judged: it is about the "
                        f"computed parameter(s) {absent!r}, which were not given.",
                        "give a value for them",
                    )
                )
                continue
            try:
                violation = constraint.violation(parameters)
            except ValueError as error:
                faults.append(
                    (
                        min(names),
                        f"The constraint {constraint!r} cannot be evaluated: {error}",
                        "make the parameters it names numbers",
                    )
                )
                continue
            if violation > options.constraint_tolerance:
                faults.append(
                    (
                        min(names),
                        f"The constraint {constraint!r} does not hold: "
                        + (
                            "its value is undefined here."
                            if math.isinf(violation)
                            else f"it is violated by {violation:.3g} of its scale, above the "
                            f"tolerance {options.constraint_tolerance:.3g}."
                        ),
                        f"change the parameters it names: {sorted(names)!r}",
                    )
                )
        return faults


def _error(faults: list[_Fault]) -> ParameterizationError:
    """Builds the one error that reports a list of faults.

    Args:
        faults: What is wrong, at least one.

    Returns:
        The error. For one fault its sentence and its own suggestion; for several, a numbered
        count and a line each. Its field path is that of the first fault.
    """
    first_name, first_sentence, first_suggestion = faults[0]
    if len(faults) == 1:
        message = f"The parameterization is not legal for the problem: {first_sentence}"
        suggestion = first_suggestion
    else:
        lines = "\n".join(f"  - {sentence}" for _, sentence, _ in faults)
        count = f"{len(faults)} problems."
        message = f"The parameterization is not legal for the problem: {count}\n{lines}"
        suggestion = "fix the problems listed above"
    return ParameterizationError(
        message, component="problem", field_path=first_name, suggestion=suggestion
    )
