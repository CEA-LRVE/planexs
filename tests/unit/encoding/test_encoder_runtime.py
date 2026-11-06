"""Unit tests for the dtype of `SearchSpaceEncoder` (specification 3.4, 8; roadmap P3-02)."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.parameters.Boolean import Boolean
from planexs.domain.parameters.Categorical import Categorical
from planexs.domain.parameters.Choice import Choice
from planexs.domain.parameters.Derived import Derived
from planexs.domain.parameters.Fixed import Fixed
from planexs.domain.parameters.Integer import Integer
from planexs.domain.parameters.Real import Real
from planexs.domain.SearchSpace import SearchSpace
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder

ENCODING_PACKAGE = Path(__file__).resolve().parents[3] / "src" / "planexs" / "encoding"

DTYPES = [("float32", torch.float32), ("float64", torch.float64)]
DTYPE_IDS = [name for name, _ in DTYPES]

#: Every dtype torch has, read from torch itself so that the list cannot go stale.
TORCH_DTYPE_NAMES = {
    name for name in dir(torch) if isinstance(getattr(torch, name), torch.dtype)
}


def richSpace() -> SearchSpace:
    """A space with every kind of column: linear, log, integer, grid, one-hot, ordered."""
    return SearchSpace(
        [
            Real("x", 0.0, 1.0),
            Real("rate", 0.001, 1.0, scaling="log"),
            Integer("n", 1, 10),
            Real("stepped", 0.0, 1.0, step_size=0.25),
            Categorical("kind", ["A", "B", "C"]),
            Boolean("flag"),
            Choice("level", [1, 10, 100, 1000], parameter_type="int", is_ordered=True),
            Fixed("mode", "fast"),
            Derived("total", "x + n", "float"),
        ]
    )


#: Values that single precision holds exactly. The log-scaled `rate` is the exception: a log
#: scale goes through `exp(log(...))`, which is exact in neither precision (see
#: `withoutLogValue`).
EXACT_POINT: dict[str, Any] = {
    "x": 0.25,
    "rate": 0.5,
    "n": 7,
    "stepped": 0.75,
    "kind": "B",
    "flag": True,
    "level": 100,
    "mode": "fast",
}


def encoderFor(name: Any) -> SearchSpaceEncoder:
    return SearchSpaceEncoder(richSpace(), RuntimeConfig(dtype=name))


def withoutLogValue(point: dict[str, Any]) -> dict[str, Any]:
    """`point` without the log-scaled parameter, which only round-trips approximately."""
    return {key: value for key, value in point.items() if key != "rate"}


class TestDtype:
    @pytest.mark.parametrize(("name", "expected"), DTYPES, ids=DTYPE_IDS)
    def test_the_dtype_follows_the_runtime_settings(
        self, name: Any, expected: torch.dtype
    ) -> None:
        assert encoderFor(name).dtype is expected

    def test_without_runtime_settings_the_encoder_is_double_precision(self) -> None:
        assert SearchSpaceEncoder(richSpace()).dtype is torch.float64

    def test_no_settings_and_the_default_settings_give_the_same_encoder(self) -> None:
        implicit = SearchSpaceEncoder(richSpace())
        explicit = SearchSpaceEncoder(richSpace(), RuntimeConfig())
        assert implicit.dtype is explicit.dtype
        assert torch.equal(
            implicit.encodeParameters(EXACT_POINT), explicit.encodeParameters(EXACT_POINT)
        )

    def test_a_dtype_that_torch_does_not_know_is_reported_when_the_encoder_is_built(
        self,
    ) -> None:
        runtime = RuntimeConfig.model_construct(dtype="bogus")
        with pytest.raises(ConfigError) as caught:
            SearchSpaceEncoder(richSpace(), runtime)
        assert caught.value.field_path == "dtype"


@pytest.mark.parametrize(("name", "expected"), DTYPES, ids=DTYPE_IDS)
class TestEveryTensorHasTheEncodersDtype:
    def test_the_unit_bounds(self, name: Any, expected: torch.dtype) -> None:
        assert encoderFor(name).unit_bounds.dtype is expected

    def test_an_encoded_parameterization(self, name: Any, expected: torch.dtype) -> None:
        assert encoderFor(name).encodeParameters(EXACT_POINT).dtype is expected

    def test_an_encoded_parameterization_with_an_inactive_conditional_parameter(
        self, name: Any, expected: torch.dtype
    ) -> None:
        # The column of a parameter that is absent is written at its neutral value.
        space = SearchSpace(
            [
                Categorical("mode", ["a", "b"], dependent_parameters={"a": ["gain"]}),
                Real("gain", 0.0, 1.0),
            ]
        )
        encoder = SearchSpaceEncoder(space, RuntimeConfig(dtype=name))
        row = encoder.encodeParameters({"mode": "b"})
        assert row.dtype is expected
        assert float(row[1]) == 0.5

    @pytest.mark.parametrize("snap", [True, False])
    def test_a_sample(self, name: Any, expected: torch.dtype, snap: bool) -> None:
        assert encoderFor(name).sampleEncoded(16, seed=3, snap=snap).dtype is expected

    def test_a_snapped_point(self, name: Any, expected: torch.dtype) -> None:
        encoder = encoderFor(name)
        relaxed = encoder.sampleEncoded(4, seed=3, snap=False)
        assert encoder.snapColumns(relaxed).dtype is expected

    def test_the_rounding_neighbours(self, name: Any, expected: torch.dtype) -> None:
        encoder = encoderFor(name)
        relaxed = encoder.sampleEncoded(1, seed=3, snap=False)[0]
        assert encoder.roundingNeighbors(relaxed).dtype is expected

    def test_the_natural_values_of_the_ranges(self, name: Any, expected: torch.dtype) -> None:
        encoder = encoderFor(name)
        assert encoder.rawRangeValues(encoder.encodeParameters(EXACT_POINT)).dtype is expected


@pytest.mark.parametrize(("name", "expected"), DTYPES, ids=DTYPE_IDS)
class TestRoundTrip:
    def test_values_that_the_type_holds_exactly_come_back_exactly(
        self, name: Any, expected: torch.dtype
    ) -> None:
        encoder = encoderFor(name)
        decoded = encoder.decode(encoder.encodeParameters(EXACT_POINT))
        assert withoutLogValue(decoded) == withoutLogValue({**EXACT_POINT, "total": 7.25})
        assert math.isclose(decoded["rate"], EXACT_POINT["rate"], rel_tol=1e-6)

    def test_integers_choices_and_booleans_are_exact_over_a_wide_range(
        self, name: Any, expected: torch.dtype
    ) -> None:
        space = SearchSpace(
            [
                Integer("n", 0, 10_000),
                Integer("m", -50, 50, step_size=5),
                Choice("o", [1, 2, 3, 5, 8, 13, 21], parameter_type="int", is_ordered=True),
                Boolean("b"),
            ]
        )
        encoder = SearchSpaceEncoder(space, RuntimeConfig(dtype=name))
        for point in (
            {"n": 0, "m": -50, "o": 1, "b": False},
            {"n": 7_777, "m": 35, "o": 13, "b": True},
            {"n": 10_000, "m": 50, "o": 21, "b": True},
        ):
            assert encoder.decode(encoder.encodeParameters(point)) == point

    def test_a_real_value_comes_back_to_the_precision_of_the_type(
        self, name: Any, expected: torch.dtype
    ) -> None:
        encoder = encoderFor(name)
        point = {**EXACT_POINT, "x": 0.37, "rate": 0.0123}
        decoded = encoder.decode(encoder.encodeParameters(point))
        # About seven digits in single precision, and (far) better than that in double.
        tolerance = 1e-6 if expected is torch.float32 else 1e-12
        assert math.isclose(decoded["x"], 0.37, rel_tol=tolerance)
        assert math.isclose(decoded["rate"], 0.0123, rel_tol=tolerance)

    def test_derived_and_fixed_parameters_are_still_reported(
        self, name: Any, expected: torch.dtype
    ) -> None:
        encoder = encoderFor(name)
        decoded = encoder.decode(encoder.encodeParameters(EXACT_POINT))
        assert decoded["mode"] == "fast"
        assert decoded["total"] == 7.25


class TestBothPrecisions:
    def test_a_real_range_on_a_step_that_is_not_a_power_of_two_is_only_approximate_in_float32(
        self,
    ) -> None:
        # The limit that the class docstring states, written down so that it cannot be lost.
        space = SearchSpace([Real("x", 0.0, 1.0, step_size=0.1)])
        single = SearchSpaceEncoder(space, RuntimeConfig(dtype="float32"))
        double = SearchSpaceEncoder(space, RuntimeConfig(dtype="float64"))
        assert double.decode(double.encodeParameters({"x": 0.3})) == {"x": 0.3}
        back = single.decode(single.encodeParameters({"x": 0.3}))["x"]
        assert back != 0.3
        assert math.isclose(back, 0.3, rel_tol=1e-6)

    def test_the_two_precisions_agree_to_single_precision(self) -> None:
        single, double = encoderFor("float32"), encoderFor("float64")
        point = {**EXACT_POINT, "x": 0.37, "rate": 0.0123}
        low = single.encodeParameters(point).to(torch.float64)
        high = double.encodeParameters(point)
        assert torch.allclose(low, high, rtol=0.0, atol=1e-6)

    def test_the_same_seed_samples_the_same_points_in_both_precisions(self) -> None:
        low = encoderFor("float32").sampleEncoded(16, seed=5, snap=False)
        high = encoderFor("float64").sampleEncoded(16, seed=5, snap=False)
        assert torch.equal(low, high.to(torch.float32))

    def test_a_sample_decodes_to_legal_parameterizations_in_both_precisions(self) -> None:
        for name in ("float32", "float64"):
            encoder = encoderFor(name)
            for point in encoder.sampleEncoded(8, seed=5):
                decoded = encoder.decode(point)
                assert decoded["kind"] in {"A", "B", "C"}
                assert 1 <= decoded["n"] <= 10
                assert decoded["stepped"] in {0.0, 0.25, 0.5, 0.75, 1.0}
                assert decoded["level"] in {1, 10, 100, 1000}

    def test_a_point_in_the_other_precision_is_accepted_by_decode_and_by_snapping(
        self,
    ) -> None:
        single, double = encoderFor("float32"), encoderFor("float64")
        expected = withoutLogValue({**EXACT_POINT, "total": 7.25})
        for decoded in (
            single.decode(double.encodeParameters(EXACT_POINT)),
            double.decode(single.encodeParameters(EXACT_POINT)),
        ):
            assert withoutLogValue(decoded) == expected
            assert math.isclose(decoded["rate"], EXACT_POINT["rate"], rel_tol=1e-6)
        assert single.snapColumns(double.encodeParameters(EXACT_POINT)).shape == (9,)

    def test_decode_reads_a_point_in_the_precision_of_its_own_encoder(self) -> None:
        # A span of 3 makes the product `unit * span` differ between the two precisions.
        space = SearchSpace([Real("x", 0.0, 3.0)])
        single = SearchSpaceEncoder(space, RuntimeConfig(dtype="float32"))
        double = SearchSpaceEncoder(space, RuntimeConfig(dtype="float64"))
        unit = 0.37 / 3.0
        unit_single = torch.tensor([unit], dtype=torch.float32)
        unit_double = torch.tensor([unit], dtype=torch.float64)
        # A float32 encoder reads even a double-precision point in single precision...
        assert single.decode(unit_double)["x"] == float(unit_single * 3.0)
        # ...and a float64 encoder reads even a single-precision point in double precision.
        assert double.decode(unit_single)["x"] == float(unit_single.to(torch.float64) * 3.0)

    def test_two_encoders_of_different_precision_do_not_affect_each_other(self) -> None:
        single = encoderFor("float32")
        double = encoderFor("float64")
        assert single.dtype is torch.float32
        assert double.dtype is torch.float64
        assert single.unit_bounds.dtype is torch.float32


def moduleLevelDtypeConstants(source: str) -> list[str]:
    """The names a module assigns, at its top level, a value that mentions a torch dtype."""
    found: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        mentions_a_dtype = any(
            isinstance(part, ast.Attribute)
            and part.attr in TORCH_DTYPE_NAMES
            and isinstance(part.value, ast.Name)
            and part.value.id == "torch"
            for part in ast.walk(value)
        )
        if mentions_a_dtype:
            found.extend(ast.unparse(target) for target in targets)
    return found


class TestNoModuleLevelDtypeConstant:
    @pytest.mark.parametrize(
        ("source", "names"),
        [
            ("import torch\n_DTYPE = torch.double\n", ["_DTYPE"]),
            ("import torch\nDTYPE: torch.dtype = torch.float32\n", ["DTYPE"]),
            ("import torch\n_PAIR = (torch.float32, torch.float64)\n", ["_PAIR"]),
            ("import torch\n_BY_NAME = {'a': torch.float64}\n", ["_BY_NAME"]),
            ("import torch\nA = B = torch.half\n", ["A", "B"]),
        ],
    )
    def test_the_guard_finds_a_dtype_constant(self, source: str, names: list[str]) -> None:
        assert moduleLevelDtypeConstants(source) == names

    @pytest.mark.parametrize(
        "source",
        [
            "import torch\ndef f():\n    return torch.zeros(3, dtype=torch.double)\n",
            "import torch\nclass C:\n    def m(self):\n        d = torch.float32\n",
            "import torch\n_FUNCTIONS = {'abs': torch.abs, 'min': torch.minimum}\n",
            "_NAME = 'float64'\n",
            "import torch\nx = torch.zeros(3)\n",
        ],
    )
    def test_the_guard_ignores_a_dtype_used_inside_a_function_or_something_else(
        self, source: str
    ) -> None:
        assert moduleLevelDtypeConstants(source) == []

    def test_no_file_of_the_encoding_package_has_a_module_level_dtype_constant(self) -> None:
        files = sorted(ENCODING_PACKAGE.glob("*.py"))
        assert files, "the encoding package was not found"
        offenders = {
            file.name: names
            for file in files
            if (names := moduleLevelDtypeConstants(file.read_text(encoding="utf-8")))
        }
        assert offenders == {}
