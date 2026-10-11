"""Maps a planexs `SearchSpace` to and from the unit-cube tensors a GP works on."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from planexs.domain.constraints.validateExpression import validateExpression
from planexs.domain.parameters.Choice import Choice
from planexs.domain.parameters.Derived import Derived
from planexs.domain.parameters.Range import Range
from planexs.domain.SearchSpace import SearchSpace
from planexs.models.compileExpression import compileExpression
from planexs.models.ParameterEncoding import LevelValue, ParameterEncoding

_DTYPE = torch.double


class SearchSpaceEncoder:
    """Encodes a `SearchSpace` into the unit cube `[0, 1]^d` and decodes points
    of it back into parameter dictionaries.

    The surrogate model and the acquisition function both work on this
    encoding, so they never need to know about log scales, integer grids, or
    categorical values:

    - A `Range` is one column, linear or log scaled onto `[0, 1]`. An `int`
      range or one with a `step_size` is a grid: the encoding is a continuous
      relaxation that `snapColumns`/`decode` round onto the grid.
    - A `Choice` with an order (`is_ordered`, or unset with more than two
      numeric values) is one column holding the rank of its value, relaxed and
      rounded like a grid.
    - An unordered `Choice` is a 0/1 column (two values) or a one-hot block.
      These are never relaxed: the acquisition layer enumerates them.
    - A `Choice` with a single value (`Fixed`) and a `Derived` parameter are not
      searched: they occupy no column, but `decode` still reports them.

    Conditional parameters (`Choice.dependent_parameters`) are encoded as
    ordinary columns (the model sees the flattened space), and a trial that
    lacks an inactive parameter is encoded with that parameter's column at its
    midpoint. `decode` removes the parameters that are inactive given the
    chosen values, because Ax rejects a parameterization that contains them.

    Args:
        search_space: The space to encode.

    Raises:
        ValueError: if the space has no searched parameter, an integer range
            with no integer inside its bounds, or a `Derived` parameter of
            type `"str"` (the custom layer can only compute numeric ones).
        TypeError: if a parameter is of an unknown kind.
    """

    def __init__(self, search_space: SearchSpace) -> None:
        """Lays out the columns of `search_space`.

        Arguments and errors are described in the class docstring.
        """
        self._search_space = search_space
        self._encodings: list[ParameterEncoding] = []
        self._fixed_values: dict[str, LevelValue] = {}
        self._derived: list[tuple[Derived, frozenset[str], Callable[..., Tensor]]] = []
        self._choices_with_dependents: list[Choice] = []
        self._dependent_names: set[str] = set()

        all_names = search_space.parameter_names
        column = 0
        for parameter in search_space.parameters:
            if isinstance(parameter, Range):
                encoding = _encodeRange(parameter, column)
            elif isinstance(parameter, Choice):
                if parameter.dependent_parameters:
                    self._choices_with_dependents.append(parameter)
                    for names in parameter.dependent_parameters.values():
                        self._dependent_names.update(names)
                if len(parameter.values) == 1:
                    self._fixed_values[parameter.name] = parameter.values[0]
                    continue
                encoding = _encodeChoice(parameter, column)
            elif isinstance(parameter, Derived):
                if parameter.parameter_type == "str":
                    raise ValueError(
                        f"Derived parameter {parameter.name!r} has parameter_type='str': "
                        "the custom acquisition layer can only compute numeric derived "
                        "parameters."
                    )
                function = compileExpression(parameter.expression, all_names)
                referenced = validateExpression(parameter.expression)
                self._derived.append((parameter, referenced, function))
                continue
            else:
                raise TypeError(
                    f"No encoding is defined for parameter kind {type(parameter).__name__!r} "
                    f"(parameter {parameter.name!r})."
                )
            self._encodings.append(encoding)
            column += encoding.n_columns

        self._n_columns = column
        if column == 0:
            raise ValueError(
                "The search space has no searched parameter (every parameter is Fixed or "
                "Derived), so there is nothing to optimize."
            )
        self._by_name = {encoding.name: encoding for encoding in self._encodings}
        self._range_encodings = [e for e in self._encodings if e.kind == "range"]

    @property
    def n_columns(self) -> int:
        """Number of columns of an encoded point (stored value, snake_case)."""
        return self._n_columns

    @property
    def encodings(self) -> list[ParameterEncoding]:
        """The per-parameter layout, in column order (a copy of a held list)."""
        return list(self._encodings)

    @property
    def unit_bounds(self) -> Tensor:
        """A `2 x d` tensor of lower and upper bounds: zeros and ones (cheap
        to build, snake_case).
        """
        return torch.stack(
            [
                torch.zeros(self._n_columns, dtype=_DTYPE),
                torch.ones(self._n_columns, dtype=_DTYPE),
            ]
        )

    @property
    def range_names(self) -> list[str]:
        """Names of the `Range` parameters, in the column order
        `rawRangeValues` uses (cheap, snake_case).
        """
        return [encoding.name for encoding in self._range_encodings]

    @property
    def numeric_fixed_values(self) -> dict[str, float]:
        """Values of the `Fixed` parameters that are numeric, for use as
        constants in constraint expressions (cheap, snake_case).
        """
        return {
            name: float(value)
            for name, value in self._fixed_values.items()
            if not isinstance(value, str)
        }

    def getEncoding(self, name: str) -> ParameterEncoding | None:
        """Returns the encoding of the searched parameter `name`, or `None` if
        it is not searched (unknown, `Fixed`, or `Derived`).
        """
        return self._by_name.get(name)

    def encodeParameters(self, parameters: Mapping[str, LevelValue]) -> Tensor:
        """Encodes one parameterization (e.g. a `Trial.parameters`).

        Args:
            parameters: Parameter name to value. A parameter that is inactive
                because of `dependent_parameters` may be absent.

        Returns:
            A tensor of shape `(d,)` in `[0, 1]`.

        Raises:
            ValueError: if a searched, non-conditional parameter is missing, or
                a choice value is not among the parameter's levels.
        """
        row = torch.zeros(self._n_columns, dtype=_DTYPE)
        for encoding in self._encodings:
            if encoding.name not in parameters:
                if encoding.name not in self._dependent_names:
                    raise ValueError(
                        f"Parameterization {dict(parameters)!r} is missing the parameter "
                        f"{encoding.name!r}."
                    )
                _writeCenter(row, encoding)
                continue
            _writeValue(row, encoding, parameters[encoding.name])
        return row

    def decode(self, x: Tensor) -> dict[str, LevelValue]:
        """Decodes one encoded point into a parameterization Ax accepts.

        The point is first snapped (grid ranges and ordered choices rounded,
        categorical blocks resolved to their largest entry). Fixed values are
        included, derived parameters are computed, and parameters made inactive
        by `dependent_parameters` are left out.

        Args:
            x: A tensor of shape `(d,)`.

        Returns:
            Parameter name to value, in the search space's declaration order.
        """
        snapped = self.snapColumns(x.detach().to(_DTYPE))
        chosen: dict[str, LevelValue] = {
            encoding.name: _readValue(snapped, encoding) for encoding in self._encodings
        }
        chosen.update(self._fixed_values)
        active = self._activeNames(chosen)

        derived_values = self._computeDerived({n: v for n, v in chosen.items() if n in active})
        result: dict[str, LevelValue] = {}
        for parameter in self._search_space.parameters:
            if parameter.name in chosen and parameter.name in active:
                result[parameter.name] = chosen[parameter.name]
            elif parameter.name in derived_values:
                result[parameter.name] = derived_values[parameter.name]
        return result

    def snapColumns(self, x: Tensor) -> Tensor:
        """Rounds a (possibly relaxed) encoded point onto valid values.

        Continuous columns are clipped to `[0, 1]`, grid ranges and ordered
        choices are rounded to the nearest level (and re-encoded), and each
        categorical block is resolved to a single level.

        Args:
            x: A tensor of shape `(..., d)`.

        Returns:
            A new tensor of the same shape.
        """
        out = x.clone().clamp(0.0, 1.0)
        for encoding in self._encodings:
            column = encoding.column
            if encoding.kind == "range":
                if encoding.step is not None:
                    raw = _unitToRaw(encoding, out[..., column])
                    out[..., column] = _rawToUnit(encoding, _snapRaw(encoding, raw))
            elif encoding.kind == "ordered":
                n_levels = len(encoding.levels)
                rank = torch.round(out[..., column] * (n_levels - 1))
                out[..., column] = rank / (n_levels - 1)
            elif encoding.n_columns == 1:
                out[..., column] = (out[..., column] >= 0.5).to(_DTYPE)
            else:
                block = out[..., column : column + encoding.n_columns]
                winner = block.argmax(dim=-1, keepdim=True)
                out[..., column : column + encoding.n_columns] = torch.zeros_like(
                    block
                ).scatter_(-1, winner, 1.0)
        return out

    def sampleEncoded(self, n: int, seed: int | None = None, snap: bool = True) -> Tensor:
        """Draws `n` space-filling (scrambled Sobol) points of the space.

        Args:
            n: Number of points.
            seed: Seed of the Sobol scrambling.
            snap: Whether to round the points onto valid values (`snapColumns`).
                Leave `False` to get continuous relaxations, as constrained
                optimization needs for its starting points.

        Returns:
            A tensor of shape `(n, d)`.
        """
        engine = SobolEngine(  # type: ignore[no-untyped-call]
            dimension=len(self._encodings), scramble=True, seed=seed
        )
        uniform = engine.draw(n).to(_DTYPE)
        points = torch.zeros(n, self._n_columns, dtype=_DTYPE)
        for latent, encoding in enumerate(self._encodings):
            if encoding.kind != "categorical":
                points[:, encoding.column] = uniform[:, latent]
                continue
            n_levels = len(encoding.levels)
            index = (uniform[:, latent] * n_levels).floor().clamp(max=n_levels - 1).long()
            if encoding.n_columns == 1:
                points[:, encoding.column] = index.to(_DTYPE)
            else:
                points[:, encoding.column : encoding.column + n_levels] = (
                    torch.nn.functional.one_hot(index, n_levels).to(_DTYPE)
                )
        return self.snapColumns(points) if snap else points

    def rawRangeValues(self, x: Tensor) -> Tensor:
        """The natural (un-normalized) values of the `Range` parameters at
        encoded points, without rounding, and differentiable in `x`.

        Args:
            x: A tensor of shape `(..., d)`.

        Returns:
            A tensor of shape `(..., n_ranges)`, columns in `range_names` order.
        """
        if not self._range_encodings:
            return x.new_zeros(*x.shape[:-1], 0)
        return torch.stack(
            [
                _unitToRaw(encoding, x[..., encoding.column])
                for encoding in self._range_encodings
            ],
            dim=-1,
        )

    def roundingNeighbors(self, x: Tensor, max_ordinal: int = 8) -> Tensor:
        """All ways of rounding the relaxed grid/ordered columns of `x` down or
        up (at most `2 ** max_ordinal` rows), with categorical blocks resolved.

        Rounding a relaxed optimum to the nearest grid point can leave a
        constrained region; this lists the neighbouring roundings so a
        feasible one can be picked.

        Args:
            x: A tensor of shape `(d,)`.
            max_ordinal: If more grid/ordered columns than this exist, only the
                nearest rounding is returned.

        Returns:
            A tensor of shape `(m, d)`, `m >= 1`.
        """
        base = self.snapColumns(x)
        ordinal = [e for e in self._encodings if e.is_discrete_ordinal]
        if not ordinal or len(ordinal) > max_ordinal:
            return base.unsqueeze(0)
        options = [_floorAndCeil(encoding, x[encoding.column]) for encoding in ordinal]
        combinations = list(itertools.product(*options))
        rows = base.repeat(len(combinations), 1)
        for row_index, combination in enumerate(combinations):
            for encoding, unit_value in zip(ordinal, combination, strict=True):
                rows[row_index, encoding.column] = unit_value
        return rows

    def categoricalFixedFeatures(
        self, max_combinations: int, seed: int | None = None
    ) -> list[dict[int, float]]:
        """The categorical assignments to optimize over, as BoTorch
        `fixed_features_list` entries (column index to value).

        All combinations of the unordered choices are returned when there are at
        most `max_combinations` of them; otherwise a uniform random subset of
        that size.

        Args:
            max_combinations: Cap on the number of assignments.
            seed: Seed of the random subset.

        Returns:
            One `{column: value}` dict per assignment; empty if the space has
            no unordered choice.
        """
        blocks = [e for e in self._encodings if e.kind == "categorical"]
        if not blocks:
            return []
        sizes = [len(block.levels) for block in blocks]
        total = math.prod(sizes)
        if total <= max_combinations:
            assignments: list[tuple[int, ...]] = list(
                itertools.product(*(range(k) for k in sizes))
            )
        else:
            generator = torch.Generator().manual_seed(0 if seed is None else seed)
            chosen: set[tuple[int, ...]] = set()
            attempts = 0
            while len(chosen) < max_combinations and attempts < 20 * max_combinations:
                draw = tuple(int(torch.randint(k, (1,), generator=generator)) for k in sizes)
                chosen.add(draw)
                attempts += 1
            assignments = sorted(chosen)
        result: list[dict[int, float]] = []
        for assignment in assignments:
            fixed: dict[int, float] = {}
            for block, index in zip(blocks, assignment, strict=True):
                if block.n_columns == 1:
                    fixed[block.column] = float(index)
                else:
                    for offset in range(block.n_columns):
                        fixed[block.column + offset] = 1.0 if offset == index else 0.0
            result.append(fixed)
        return result

    def _activeNames(self, chosen: Mapping[str, LevelValue]) -> set[str]:
        """Names of the parameters that are active given `chosen` values: those
        that are nobody's dependent, plus those a chosen value of an active
        choice switches on.
        """
        active = {name for name in self._search_space.parameter_names} - self._dependent_names
        changed = True
        while changed:
            changed = False
            for choice in self._choices_with_dependents:
                if choice.name not in active or choice.name not in chosen:
                    continue
                switched_on = (choice.dependent_parameters or {}).get(chosen[choice.name], ())
                for name in switched_on:
                    if name not in active:
                        active.add(name)
                        changed = True
        return active

    def _computeDerived(self, values: Mapping[str, LevelValue]) -> dict[str, LevelValue]:
        """Evaluates the derived parameters from the values of the active ones.

        A derived parameter may use another derived one; a derived parameter that depends on a
        parameter absent from `values` (an inactive conditional one) is left out.

        Args:
            values: Active parameter name to value.

        Returns:
            Derived parameter name to its computed value, cast to the parameter's type.
        """
        environment = {
            name: torch.tensor(float(value), dtype=_DTYPE)
            for name, value in values.items()
            if not isinstance(value, str)
        }
        computed: dict[str, LevelValue] = {}
        pending = list(self._derived)
        progressed = True
        while pending and progressed:
            progressed = False
            for entry in list(pending):
                derived, names, function = entry
                if not names <= environment.keys():
                    continue
                number = float(function(environment))
                environment[derived.name] = torch.tensor(number, dtype=_DTYPE)
                computed[derived.name] = _castDerived(derived.parameter_type, number)
                pending.remove(entry)
                progressed = True
        return computed


def _castDerived(parameter_type: str, number: float) -> LevelValue:
    """Casts a computed number to a derived parameter's declared type.

    Args:
        parameter_type: `"float"`, `"int"` or `"bool"`.
        number: The computed value.

    Returns:
        The value as a `float`, a rounded `int`, or a `bool` (non-zero is `True`).
    """
    if parameter_type == "int":
        return round(number)
    if parameter_type == "bool":
        return number != 0.0
    return number


def _isNumeric(value: object) -> bool:
    """Whether `value` is an `int` or a `float` (a `bool` does not count)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _encodeRange(parameter: Range, column: int) -> ParameterEncoding:
    """Describes how a `Range` is laid out: one column, linear or log scaled, maybe on a grid.

    Args:
        parameter: The range to encode.
        column: The index of the column it will occupy.

    Returns:
        Its encoding. An `int` range is snapped to the integers inside its bounds.

    Raises:
        ValueError: if an integer range does not contain two distinct integers.
    """
    lower, upper = parameter.bounds
    is_int = parameter.parameter_type == "int"
    step = parameter.step_size if parameter.step_size else (1.0 if is_int else None)
    if is_int:
        lower, upper = float(math.ceil(lower)), float(math.floor(upper))
        if lower >= upper:
            raise ValueError(
                f"Integer range {parameter.name!r} with bounds {parameter.bounds!r} does "
                "not contain two distinct integers."
            )
    return ParameterEncoding(
        name=parameter.name,
        kind="range",
        column=column,
        n_columns=1,
        lower=float(lower),
        upper=float(upper),
        log_scale=parameter.scaling == "log",
        step=step,
        is_int=is_int,
    )


def _encodeChoice(parameter: Choice, column: int) -> ParameterEncoding:
    """Describes how a `Choice` is laid out: one rank column if ordered, otherwise a 0/1 column
    (two values) or a one-hot block.

    An unset `is_ordered` means ordered only for more than two numeric values.

    Args:
        parameter: The choice to encode.
        column: The index of its first column.

    Returns:
        Its encoding. Numeric ordered values are sorted; others keep the order given.
    """
    values: list[LevelValue] = list(parameter.values)
    numeric = all(_isNumeric(value) for value in values)
    ordered = parameter.is_ordered
    if ordered is None:
        ordered = numeric and len(values) > 2
    if ordered:
        levels = tuple(sorted(values)) if numeric else tuple(values)
        return ParameterEncoding(
            name=parameter.name, kind="ordered", column=column, n_columns=1, levels=levels
        )
    n_columns = 1 if len(values) == 2 else len(values)
    return ParameterEncoding(
        name=parameter.name,
        kind="categorical",
        column=column,
        n_columns=n_columns,
        levels=tuple(values),
    )


def _rawToUnit(encoding: ParameterEncoding, raw: Tensor) -> Tensor:
    """Maps natural values of a range onto `[0, 1]`, on a log scale if the range uses one.

    Args:
        encoding: The range's encoding.
        raw: Natural values.

    Returns:
        Values in `[0, 1]`.
    """
    if encoding.log_scale:
        low, high = math.log(encoding.lower), math.log(encoding.upper)
        return ((torch.log(raw) - low) / (high - low)).clamp(0.0, 1.0)
    return ((raw - encoding.lower) / (encoding.upper - encoding.lower)).clamp(0.0, 1.0)


def _unitToRaw(encoding: ParameterEncoding, unit: Tensor) -> Tensor:
    """Maps values in `[0, 1]` back to natural values, differentiably and without rounding.

    Args:
        encoding: The range's encoding.
        unit: Values in `[0, 1]`.

    Returns:
        Natural values.
    """
    if encoding.log_scale:
        low, high = math.log(encoding.lower), math.log(encoding.upper)
        return torch.exp(low + unit * (high - low))
    return encoding.lower + unit * (encoding.upper - encoding.lower)


def _snapRaw(encoding: ParameterEncoding, raw: Tensor) -> Tensor:
    """Rounds natural values of a range onto its grid, or clips them if it has no grid.

    Args:
        encoding: The range's encoding.
        raw: Natural values.

    Returns:
        Values on the grid and inside the bounds.
    """
    if encoding.step is None:
        return raw.clamp(encoding.lower, encoding.upper)
    max_index = math.floor((encoding.upper - encoding.lower) / encoding.step + 1e-9)
    index = torch.round((raw - encoding.lower) / encoding.step).clamp(0, max_index)
    return encoding.lower + index * encoding.step


def _floorAndCeil(encoding: ParameterEncoding, unit: Tensor) -> tuple[float, ...]:
    """The two nearest legal unit values around a relaxed value: rounded down and rounded up.

    Args:
        encoding: An ordered choice or a range with a grid.
        unit: The relaxed value in `[0, 1]`.

    Returns:
        One or two unit values (one when the relaxed value is already on a level).
    """
    if encoding.kind == "ordered":
        position = float(unit.clamp(0.0, 1.0)) * (len(encoding.levels) - 1)
        low, high = math.floor(position), math.ceil(position)
        denominator = len(encoding.levels) - 1
        return tuple(sorted({low / denominator, high / denominator}))
    assert encoding.step is not None
    max_index = math.floor((encoding.upper - encoding.lower) / encoding.step + 1e-9)
    raw = float(_unitToRaw(encoding, unit.clamp(0.0, 1.0)))
    position = (raw - encoding.lower) / encoding.step
    indices = {
        min(max(math.floor(position), 0), max_index),
        min(max(math.ceil(position), 0), max_index),
    }
    return tuple(
        sorted(
            float(
                _rawToUnit(
                    encoding, torch.tensor(encoding.lower + i * encoding.step, dtype=_DTYPE)
                )
            )
            for i in indices
        )
    )


def _writeCenter(row: Tensor, encoding: ParameterEncoding) -> None:
    """Writes the "unknown" value of an inactive conditional parameter into `row`.

    A rank or numeric column gets its midpoint, a one-hot block gets equal weights.

    Args:
        row: The encoded row to fill in place.
        encoding: The parameter's encoding.
    """
    if encoding.kind == "categorical" and encoding.n_columns > 1:
        row[encoding.column : encoding.column + encoding.n_columns] = 1.0 / encoding.n_columns
    else:
        row[encoding.column] = 0.5


def _writeValue(row: Tensor, encoding: ParameterEncoding, value: LevelValue) -> None:
    """Writes one parameter value into the encoded row.

    Args:
        row: The encoded row to fill in place.
        encoding: The parameter's encoding.
        value: The parameter's value.

    Raises:
        ValueError: if a choice value is not among the parameter's levels.
    """
    if encoding.kind == "range":
        raw = torch.tensor(float(value), dtype=_DTYPE)
        row[encoding.column] = _rawToUnit(encoding, raw)
        return
    try:
        index = encoding.levels.index(value)
    except ValueError:
        raise ValueError(
            f"Value {value!r} is not one of the levels {list(encoding.levels)!r} of "
            f"parameter {encoding.name!r}."
        ) from None
    if encoding.kind == "ordered":
        row[encoding.column] = index / (len(encoding.levels) - 1)
    elif encoding.n_columns == 1:
        row[encoding.column] = float(index)
    else:
        row[encoding.column + index] = 1.0


def _readValue(snapped: Tensor, encoding: ParameterEncoding) -> LevelValue:
    """Reads one parameter's value out of a snapped encoded point.

    Args:
        snapped: A point already rounded by `snapColumns`.
        encoding: The parameter's encoding.

    Returns:
        The value, as a Python `int`, `float`, `bool` or `str`.
    """
    column = encoding.column
    if encoding.kind == "range":
        raw = float(_snapRaw(encoding, _unitToRaw(encoding, snapped[column])))
        if encoding.is_int:
            return round(raw)
        if encoding.step is not None:
            raw = float(f"{raw:.12g}")
        return min(max(raw, encoding.lower), encoding.upper)
    if encoding.kind == "ordered":
        return encoding.levels[round(float(snapped[column]) * (len(encoding.levels) - 1))]
    if encoding.n_columns == 1:
        return encoding.levels[int(float(snapped[column]) >= 0.5)]
    return encoding.levels[int(snapped[column : column + encoding.n_columns].argmax())]
