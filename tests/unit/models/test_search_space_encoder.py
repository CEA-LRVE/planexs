"""Unit tests for `SearchSpaceEncoder`."""

from __future__ import annotations

import math

import pytest
import torch

from planexs import (
    Boolean,
    Categorical,
    Choice,
    Derived,
    Fixed,
    Integer,
    Range,
    Real,
    SearchSpace,
)
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder


def _mixedSpace() -> SearchSpace:
    return SearchSpace(
        [
            Real("lr", 1e-5, 1e-1, scaling="log"),
            Integer("layers", 1, 8),
            Real("temp", 20.0, 120.0, step_size=5.0),
            Categorical("solvent", ["water", "ethanol", "toluene"]),
            Boolean("flag"),
            Choice("batch", [16, 32, 64, 128], parameter_type="int"),
            Fixed("seed", 7),
            Derived("area", "temp * layers", "float"),
        ]
    )


class TestLayout:
    def test_columns_follow_parameter_kinds(self) -> None:
        encoder = SearchSpaceEncoder(_mixedSpace())

        layout = [(e.name, e.kind, e.column, e.n_columns) for e in encoder.encodings]

        assert layout == [
            ("lr", "range", 0, 1),
            ("layers", "range", 1, 1),
            ("temp", "range", 2, 1),
            ("solvent", "categorical", 3, 3),
            ("flag", "categorical", 6, 1),
            ("batch", "ordered", 7, 1),
        ]
        assert encoder.n_columns == 8
        assert encoder.range_names == ["lr", "layers", "temp"]

    def test_fixed_and_derived_take_no_column(self) -> None:
        encoder = SearchSpaceEncoder(_mixedSpace())
        assert encoder.getEncoding("seed") is None
        assert encoder.getEncoding("area") is None
        assert encoder.numeric_fixed_values == {"seed": 7.0}

    def test_unit_bounds(self) -> None:
        bounds = SearchSpaceEncoder(_mixedSpace()).unit_bounds
        assert bounds.shape == (2, 8)
        assert bool((bounds[0] == 0).all()) and bool((bounds[1] == 1).all())

    def test_unordered_numeric_choice_is_categorical_when_flagged(self) -> None:
        space = SearchSpace([Choice("k", [1, 2, 3], "int", is_ordered=False)])
        assert SearchSpaceEncoder(space).encodings[0].kind == "categorical"

    def test_unset_is_ordered_means_ordered_only_for_more_than_two_numeric_values(
        self,
    ) -> None:
        many = SearchSpace([Choice("k", [1, 2, 3], "int")])
        two = SearchSpace([Choice("k", [1, 2], "int")])
        strings = SearchSpace([Choice("k", ["a", "b", "c"], "str")])
        assert SearchSpaceEncoder(many).encodings[0].kind == "ordered"
        assert SearchSpaceEncoder(two).encodings[0].kind == "categorical"
        assert SearchSpaceEncoder(strings).encodings[0].kind == "categorical"

    def test_ordered_numeric_levels_are_sorted_and_strings_keep_given_order(self) -> None:
        numeric = SearchSpace([Choice("k", [8, 2, 4], "int", is_ordered=True)])
        text = SearchSpace([Choice("k", ["low", "high", "mid"], "str", is_ordered=True)])
        assert SearchSpaceEncoder(numeric).encodings[0].levels == (2, 4, 8)
        assert SearchSpaceEncoder(text).encodings[0].levels == ("low", "high", "mid")

    def test_rejects_a_space_with_nothing_to_search(self) -> None:
        with pytest.raises(ValueError, match="nothing to optimize"):
            SearchSpaceEncoder(SearchSpace([Fixed("a", 1.0)]))

    def test_rejects_an_integer_range_without_two_integers(self) -> None:
        with pytest.raises(ValueError, match="two distinct integers"):
            SearchSpaceEncoder(SearchSpace([Range("r", (0.2, 0.8), parameter_type="int")]))

    def test_rejects_a_string_derived_parameter(self) -> None:
        space = SearchSpace([Real("a", 0.0, 1.0), Derived("d", "a * 2", "str")])
        with pytest.raises(ValueError, match="numeric derived"):
            SearchSpaceEncoder(space)


class TestRoundTrip:
    def test_decode_of_encode_recovers_sampled_points(self) -> None:
        encoder = SearchSpaceEncoder(_mixedSpace())
        for point in encoder.sampleEncoded(16, seed=0):
            parameters = encoder.decode(point)
            searched = {k: v for k, v in parameters.items() if k not in ("seed", "area")}

            assert torch.allclose(
                encoder.encodeParameters(searched), encoder.snapColumns(point)
            )

    def test_decode_returns_python_native_types_in_declaration_order(self) -> None:
        encoder = SearchSpaceEncoder(_mixedSpace())

        parameters = encoder.decode(encoder.sampleEncoded(1, seed=1)[0])

        assert list(parameters) == [
            "lr", "layers", "temp", "solvent", "flag", "batch", "seed", "area",
        ]  # fmt: skip
        assert type(parameters["lr"]) is float
        assert type(parameters["layers"]) is int
        assert type(parameters["temp"]) is float
        assert type(parameters["solvent"]) is str
        assert type(parameters["flag"]) is bool
        assert type(parameters["batch"]) is int
        assert parameters["seed"] == 7

    def test_log_scale_midpoint_is_the_geometric_mean(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Real("lr", 1e-4, 1e-2, scaling="log")]))
        value = encoder.decode(torch.tensor([0.5], dtype=torch.double))["lr"]
        assert value == pytest.approx(1e-3)

    def test_bounds_are_hit_exactly(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Real("x", -2.0, 3.0), Integer("n", 2, 9)]))
        low = encoder.decode(torch.tensor([0.0, 0.0], dtype=torch.double))
        high = encoder.decode(torch.tensor([1.0, 1.0], dtype=torch.double))
        assert (low["x"], low["n"]) == (-2.0, 2)
        assert (high["x"], high["n"]) == (3.0, 9)

    def test_step_size_grid_is_respected(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Real("t", 20.0, 120.0, step_size=5.0)]))
        for u in torch.linspace(0, 1, 37, dtype=torch.double):
            value = encoder.decode(u.reshape(1))["t"]
            assert (value - 20.0) % 5.0 == pytest.approx(0.0, abs=1e-9)
            assert 20.0 <= value <= 120.0

    def test_step_grid_that_does_not_divide_the_span_never_exceeds_the_upper_bound(
        self,
    ) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Real("t", 0.0, 10.0, step_size=4.0)]))
        values = {encoder.decode(u.reshape(1))["t"] for u in torch.linspace(0, 1, 41)}
        assert values == {0.0, 4.0, 8.0}

    def test_integer_with_explicit_step(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Integer("n", 0, 20, step_size=5)]))
        values = {encoder.decode(u.reshape(1))["n"] for u in torch.linspace(0, 1, 41)}
        assert values == {0, 5, 10, 15, 20}

    def test_snap_is_idempotent(self) -> None:
        encoder = SearchSpaceEncoder(_mixedSpace())
        relaxed = torch.rand(20, encoder.n_columns, dtype=torch.double)
        once = encoder.snapColumns(relaxed)
        assert torch.equal(encoder.snapColumns(once), once)

    def test_snap_resolves_a_one_hot_block_to_its_largest_entry(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Categorical("c", ["a", "b", "c"])]))
        snapped = encoder.snapColumns(torch.tensor([0.2, 0.9, 0.6], dtype=torch.double))
        assert snapped.tolist() == [0.0, 1.0, 0.0]

    def test_snap_clips_out_of_range_values(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Real("x", 0.0, 1.0)]))
        assert encoder.snapColumns(torch.tensor([1.7, -0.3], dtype=torch.double)).tolist() == [
            1.0,
            0.0,
        ]

    def test_decode_does_not_mutate_its_input(self) -> None:
        encoder = SearchSpaceEncoder(_mixedSpace())
        point = torch.rand(encoder.n_columns, dtype=torch.double)
        before = point.clone()
        encoder.decode(point)
        assert torch.equal(point, before)


class TestDerivedAndFixed:
    def test_derived_values_are_computed_and_cast(self) -> None:
        space = SearchSpace(
            [
                Integer("a", 1, 10),
                Real("b", 0.0, 1.0),
                Derived("total", "a + b", "float"),
                Derived("doubled", "total * 2", "float"),
                Derived("rounded", "a * 1.4", "int"),
                Derived("big", "a - 5", "bool"),
            ]
        )
        encoder = SearchSpaceEncoder(space)

        parameters = encoder.decode(torch.tensor([1.0, 0.0], dtype=torch.double))

        assert parameters["a"] == 10
        assert parameters["total"] == 10.0
        assert parameters["doubled"] == 20.0  # a derived parameter may use another
        assert parameters["rounded"] == 14 and type(parameters["rounded"]) is int
        assert parameters["big"] is True

    def test_derived_may_use_a_fixed_parameter(self) -> None:
        space = SearchSpace(
            [Real("a", 0.0, 1.0), Fixed("k", 3.0), Derived("d", "a * k", "float")]
        )
        parameters = SearchSpaceEncoder(space).decode(torch.tensor([1.0], dtype=torch.double))
        assert parameters == {"a": 1.0, "k": 3.0, "d": 3.0}


class TestConditionalParameters:
    @staticmethod
    def _space() -> SearchSpace:
        return SearchSpace(
            [
                Categorical(
                    "model",
                    ["mlp", "cnn"],
                    dependent_parameters={"mlp": ["hidden"], "cnn": ["filters"]},
                ),
                Integer("hidden", 8, 512),
                Integer("filters", 8, 256),
                Real("lr", 1e-4, 1e-1, scaling="log"),
            ]
        )

    def test_decode_drops_parameters_that_the_chosen_value_switches_off(self) -> None:
        encoder = SearchSpaceEncoder(self._space())
        mlp = torch.tensor([0.0, 0.5, 0.5, 0.5], dtype=torch.double)
        cnn = torch.tensor([1.0, 0.5, 0.5, 0.5], dtype=torch.double)

        assert set(encoder.decode(mlp)) == {"model", "hidden", "lr"}
        assert set(encoder.decode(cnn)) == {"model", "filters", "lr"}

    def test_encode_imputes_the_midpoint_for_an_inactive_parameter(self) -> None:
        encoder = SearchSpaceEncoder(self._space())

        row = encoder.encodeParameters({"model": "cnn", "filters": 8, "lr": 1e-4})

        assert row[encoder.getEncoding("hidden").column] == 0.5  # type: ignore[union-attr]

    def test_a_missing_unconditional_parameter_is_an_error(self) -> None:
        encoder = SearchSpaceEncoder(self._space())
        with pytest.raises(ValueError, match="missing the parameter 'lr'"):
            encoder.encodeParameters({"model": "cnn", "filters": 8})

    def test_nested_dependencies_activate_transitively(self) -> None:
        # A dependency chain: use_a -> a_kind -> a_size.
        space = SearchSpace(
            [
                Choice(
                    "use_a", [True, False], "bool", dependent_parameters={True: ["a_kind"]}
                ),
                Choice("a_kind", ["x", "y"], "str", dependent_parameters={"x": ["a_size"]}),
                Integer("a_size", 1, 5),
            ]
        )
        encoder = SearchSpaceEncoder(space)
        columns = {e.name: e.column for e in encoder.encodings}

        def point(use_a: float, a_kind: float) -> torch.Tensor:
            row = torch.full((encoder.n_columns,), 0.5, dtype=torch.double)
            row[columns["use_a"]], row[columns["a_kind"]] = use_a, a_kind
            return row

        assert set(encoder.decode(point(0.0, 0.0))) == {"use_a", "a_kind", "a_size"}
        assert set(encoder.decode(point(0.0, 1.0))) == {"use_a", "a_kind"}
        assert set(encoder.decode(point(1.0, 0.0))) == {"use_a"}  # a_kind off, so a_size off


class TestSampling:
    def test_is_reproducible_for_a_seed_and_shaped_n_by_d(self) -> None:
        encoder = SearchSpaceEncoder(_mixedSpace())

        first = encoder.sampleEncoded(10, seed=4)
        second = encoder.sampleEncoded(10, seed=4)
        other = encoder.sampleEncoded(10, seed=5)

        assert first.shape == (10, encoder.n_columns)
        assert torch.equal(first, second)
        assert not torch.equal(first, other)

    def test_snapped_samples_are_valid_and_unsnapped_ones_are_relaxed(self) -> None:
        encoder = SearchSpaceEncoder(_mixedSpace())
        snapped = encoder.sampleEncoded(30, seed=0, snap=True)
        relaxed = encoder.sampleEncoded(30, seed=0, snap=False)

        assert torch.equal(encoder.snapColumns(snapped), snapped)
        assert not torch.equal(encoder.snapColumns(relaxed), relaxed)

    def test_covers_every_category(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Categorical("c", ["a", "b", "c", "d"])]))
        seen = {encoder.decode(x)["c"] for x in encoder.sampleEncoded(64, seed=0)}
        assert seen == {"a", "b", "c", "d"}


class TestRawRangeValues:
    def test_returns_natural_values_of_range_columns_only(self) -> None:
        encoder = SearchSpaceEncoder(
            SearchSpace(
                [
                    Real("x", 10.0, 20.0),
                    Categorical("c", ["a", "b", "c"]),
                    Real("y", 1e-2, 1.0, scaling="log"),
                ]
            )
        )
        point = torch.tensor([0.5, 1.0, 0.0, 0.0, 0.5], dtype=torch.double)

        raw = encoder.rawRangeValues(point)

        assert raw.tolist() == pytest.approx([15.0, 0.1])

    def test_does_not_round_and_is_differentiable(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Integer("n", 0, 10)]))
        point = torch.tensor([0.33], dtype=torch.double, requires_grad=True)

        raw = encoder.rawRangeValues(point)
        raw.sum().backward()

        assert float(raw) == pytest.approx(3.3)
        assert float(point.grad) == pytest.approx(10.0)

    def test_is_empty_without_range_parameters(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Categorical("c", ["a", "b", "c"])]))
        assert encoder.rawRangeValues(torch.zeros(4, 3, dtype=torch.double)).shape == (4, 0)

    def test_supports_leading_batch_dimensions(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Real("x", 0.0, 2.0), Real("y", 0.0, 4.0)]))
        raw = encoder.rawRangeValues(torch.full((3, 5, 2), 0.5, dtype=torch.double))
        assert raw.shape == (3, 5, 2)
        assert torch.allclose(raw[0, 0], torch.tensor([1.0, 2.0], dtype=torch.double))


class TestRoundingNeighbors:
    def test_lists_floor_and_ceiling_of_every_relaxed_column(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Integer("a", 0, 10), Integer("b", 0, 10)]))

        rows = encoder.roundingNeighbors(torch.tensor([0.34, 0.76], dtype=torch.double))

        decoded = {(encoder.decode(r)["a"], encoder.decode(r)["b"]) for r in rows}
        assert decoded == {(3, 7), (3, 8), (4, 7), (4, 8)}

    def test_an_already_integral_column_has_a_single_option(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Integer("a", 0, 10)]))
        assert encoder.roundingNeighbors(torch.tensor([0.3], dtype=torch.double)).shape == (
            1,
            1,
        )

    def test_ordered_choices_are_included(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Choice("k", [1, 2, 3, 4], "int")]))
        rows = encoder.roundingNeighbors(torch.tensor([0.4], dtype=torch.double))
        assert {encoder.decode(r)["k"] for r in rows} == {2, 3}

    def test_falls_back_to_the_nearest_rounding_beyond_the_cap(self) -> None:
        space = SearchSpace([Integer(f"n{i}", 0, 10) for i in range(4)])
        encoder = SearchSpaceEncoder(space)
        relaxed = torch.full((4,), 0.34, dtype=torch.double)
        assert encoder.roundingNeighbors(relaxed, max_ordinal=3).shape == (1, 4)

    def test_categorical_blocks_are_resolved_in_every_row(self) -> None:
        encoder = SearchSpaceEncoder(
            SearchSpace([Integer("a", 0, 10), Categorical("c", ["x", "y", "z"])])
        )
        relaxed = torch.tensor([0.34, 0.1, 0.8, 0.2], dtype=torch.double)
        rows = encoder.roundingNeighbors(relaxed)
        assert all(encoder.decode(r)["c"] == "y" for r in rows)


class TestCategoricalFixedFeatures:
    def test_is_empty_without_unordered_choices(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Real("x", 0, 1), Integer("n", 1, 5)]))
        assert encoder.categoricalFixedFeatures(10) == []

    def test_enumerates_every_combination_when_under_the_cap(self) -> None:
        encoder = SearchSpaceEncoder(
            SearchSpace([Categorical("c", ["a", "b", "c"]), Boolean("f")])
        )

        assignments = encoder.categoricalFixedFeatures(100)

        assert len(assignments) == 6
        assert {tuple(sorted(a.items())) for a in assignments} == {
            tuple(
                sorted(
                    {0: float(i == 0), 1: float(i == 1), 2: float(i == 2), 3: float(j)}.items()
                )
            )
            for i in range(3)
            for j in range(2)
        }

    def test_samples_a_distinct_subset_when_over_the_cap(self) -> None:
        encoder = SearchSpaceEncoder(
            SearchSpace([Categorical(f"c{i}", ["a", "b", "c", "d"]) for i in range(4)])
        )

        assignments = encoder.categoricalFixedFeatures(10, seed=3)

        assert len(assignments) == 10
        assert len({tuple(sorted(a.items())) for a in assignments}) == 10
        assert assignments == encoder.categoricalFixedFeatures(10, seed=3)

    def test_every_assignment_is_one_hot_per_block(self) -> None:
        encoder = SearchSpaceEncoder(SearchSpace([Categorical("c", ["a", "b", "c"])]))
        for assignment in encoder.categoricalFixedFeatures(10):
            assert sum(assignment.values()) == 1.0
            assert math.isclose(sum(v * v for v in assignment.values()), 1.0)
