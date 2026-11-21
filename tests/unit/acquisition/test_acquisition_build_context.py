"""Unit tests for `AcquisitionBuildContext` (roadmap task P5-01, specification 5.8, 6).

What is tested: what the context holds and derives, that it is immutable, and every way in
which its sets of points can be malformed, each reported with the name of the field when the
context is built, and not as a shape error inside the engine (specification 3.6).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
import torch
from acquisition_fakes import N_COLUMNS, makeContext, makeGoal, makeSurrogate, points

from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator

FIELDS = ["pending_points", "reference_points"]
FLOAT32 = RuntimeConfig(dtype="float32")


def malformed(field: str, value: Any) -> AcquisitionContextError:
    """The error that a context with this value in this field raises."""
    with pytest.raises(AcquisitionContextError) as caught:
        makeContext(**{field: value})
    return caught.value


class TestWhatTheContextHolds:
    def test_it_holds_the_surrogate_the_goal_and_the_generator(self) -> None:
        surrogate = makeSurrogate()
        goal = makeGoal()
        generator = torchGenerator(11, RuntimeConfig())

        context = AcquisitionBuildContext(surrogate=surrogate, goal=goal, generator=generator)

        assert context.surrogate is surrogate
        assert context.goal is goal
        assert context.generator is generator

    def test_there_are_no_pending_points_and_no_reference_set_unless_they_are_given(
        self,
    ) -> None:
        context = makeContext()

        assert context.pending_points is None
        assert context.reference_points is None

    def test_the_problem_and_the_encoder_are_the_ones_of_the_surrogate(self) -> None:
        # They are read from the surrogate and not stored beside it, so they can never
        # disagree with the model that the acquisition is built on.
        surrogate = makeSurrogate()

        context = makeContext(surrogate=surrogate)

        assert context.problem is surrogate.problem
        assert context.encoder is surrogate.encoder
        assert context.encoder.n_columns == N_COLUMNS

    def test_the_native_model_is_reachable_through_the_surrogate(self) -> None:
        surrogate = makeSurrogate()

        assert makeContext(surrogate=surrogate).surrogate.model is surrogate.model

    def test_the_fields_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            AcquisitionBuildContext(makeSurrogate(), makeGoal())  # type: ignore[misc]

    def test_the_sets_of_points_are_shared_with_the_caller_and_not_copied(self) -> None:
        pending = points([0.25, 0.75])
        reference = points([0.1, 0.1], [0.9, 0.9])

        context = makeContext(pending_points=pending, reference_points=reference)

        assert context.pending_points is pending
        assert context.reference_points is reference

    def test_contexts_are_compared_by_identity(self) -> None:
        surrogate = makeSurrogate()
        goal = makeGoal()
        generator = torchGenerator(7, RuntimeConfig())
        one = AcquisitionBuildContext(surrogate=surrogate, goal=goal, generator=generator)
        same_pieces = AcquisitionBuildContext(
            surrogate=surrogate, goal=goal, generator=generator
        )

        assert one == one
        assert one != same_pieces

    def test_the_boundaries_of_the_unit_cube_are_legal_points(self) -> None:
        corners = points([0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0])

        context = makeContext(pending_points=corners, reference_points=corners)

        assert context.pending_points is corners


class TestTheContextIsImmutable:
    def test_no_field_can_be_assigned(self) -> None:
        context = makeContext()

        for field in ("surrogate", "goal", "generator", "pending_points", "reference_points"):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(context, field, None)

    def test_no_field_can_be_deleted(self) -> None:
        context = makeContext()

        with pytest.raises(dataclasses.FrozenInstanceError):
            del context.goal

    def test_no_field_can_be_added(self) -> None:
        context = makeContext()

        with pytest.raises(dataclasses.FrozenInstanceError):
            context.extra = 1  # type: ignore[attr-defined]

    def test_the_derived_properties_cannot_be_assigned_either(self) -> None:
        context = makeContext()

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            context.encoder = None  # type: ignore[misc, assignment]

    def test_an_assignment_that_fails_leaves_the_context_as_it_was(self) -> None:
        pending = points([0.25, 0.75])
        context = makeContext(pending_points=pending)

        with pytest.raises(dataclasses.FrozenInstanceError):
            context.pending_points = None

        assert context.pending_points is pending

    def test_the_generator_is_the_one_thing_meant_to_change(self) -> None:
        context = makeContext()
        reference = torchGenerator(7, RuntimeConfig())

        first = torch.rand(3, generator=context.generator, dtype=torch.double)
        second = torch.rand(3, generator=context.generator, dtype=torch.double)

        assert not torch.equal(first, second)  # drawing advances it
        assert torch.equal(first, torch.rand(3, generator=reference, dtype=torch.double))


class TestNoPendingPoint:
    def test_an_empty_set_is_read_as_none(self) -> None:
        # A caller that collects the pending points into a tensor gets this when none is
        # pending, and the engine's own `X_pending` wants None: one representation.
        context = makeContext(pending_points=torch.empty(0, N_COLUMNS, dtype=torch.double))

        assert context.pending_points is None

    def test_a_set_with_one_point_is_kept(self) -> None:
        pending = points([0.5, 0.5])

        assert makeContext(pending_points=pending).pending_points is pending

    def test_an_empty_set_still_has_to_fit_the_encoder(self) -> None:
        error = malformed("pending_points", torch.empty(0, N_COLUMNS + 1, dtype=torch.double))

        assert error.field_path == "pending_points"


class TestTheReferenceSet:
    def test_an_empty_reference_set_is_refused_because_it_would_normalize_nothing(
        self,
    ) -> None:
        error = malformed("reference_points", torch.empty(0, N_COLUMNS, dtype=torch.double))

        assert error.field_path == "reference_points"
        assert "empty" in error.message
        assert error.suggestion is not None
        assert "None" in error.suggestion  # the way to say that none is needed

    def test_a_set_with_one_point_is_a_legal_reference_set(self) -> None:
        reference = points([0.5, 0.5])

        assert makeContext(reference_points=reference).reference_points is reference


@pytest.mark.parametrize("field", FIELDS)
class TestEverySetOfPointsIsChecked:
    def test_a_one_dimensional_tensor_is_refused(self, field: str) -> None:
        error = malformed(field, torch.tensor([0.5, 0.5], dtype=torch.double))

        assert error.field_path == field
        assert f"{field} must have shape (n, {N_COLUMNS})" in error.message
        assert "(2,)" in error.message  # what was given

    def test_a_three_dimensional_tensor_is_refused(self, field: str) -> None:
        error = malformed(field, torch.rand(2, 1, N_COLUMNS, dtype=torch.double))

        assert error.field_path == field
        assert "(2, 1, 2)" in error.message

    @pytest.mark.parametrize("columns", [1, 3, 5])
    def test_a_wrong_number_of_columns_is_refused(self, field: str, columns: int) -> None:
        error = malformed(field, torch.rand(2, columns, dtype=torch.double))

        assert error.field_path == field
        assert f"(n, {N_COLUMNS})" in error.message
        assert f"(2, {columns})" in error.message

    def test_a_list_is_not_a_tensor(self, field: str) -> None:
        error = malformed(field, [[0.5, 0.5]])

        assert error.field_path == field
        assert "must be a tensor or None" in error.message
        assert "list" in error.message

    def test_a_different_precision_than_the_encoder_is_refused(self, field: str) -> None:
        error = malformed(field, torch.rand(2, N_COLUMNS, dtype=torch.float32))

        assert error.field_path == field
        assert "float32" in error.message
        assert "float64" in error.message
        assert "precisions" in error.message

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_value_that_is_not_finite_is_refused(self, field: str, bad: float) -> None:
        tensor = points([0.5, 0.5], [0.5, 0.5])
        tensor[1, 0] = bad

        error = malformed(field, tensor)

        assert error.field_path == field
        assert "not finite" in error.message

    @pytest.mark.parametrize("bad", [-0.01, 1.01, -5.0, 100.0])
    def test_a_value_outside_the_unit_cube_is_refused(self, field: str, bad: float) -> None:
        error = malformed(field, points([0.5, bad]))

        assert error.field_path == field
        assert "outside [0, 1]" in error.message
        assert "natural values" in error.message

    def test_the_error_names_the_context_and_says_how_to_fix_it(self, field: str) -> None:
        error = malformed(field, torch.rand(2, N_COLUMNS + 1, dtype=torch.double))

        assert error.component == "acquisition build context"
        assert error.suggestion
        assert "encodeParameters" in error.suggestion


class TestThePrecisionFollowsTheEncoder:
    def test_a_single_precision_surrogate_takes_single_precision_points(self) -> None:
        pending = points([0.25, 0.75], runtime=FLOAT32)

        context = makeContext(runtime=FLOAT32, pending_points=pending)

        assert context.pending_points is pending
        assert context.encoder.dtype == torch.float32

    def test_a_single_precision_surrogate_refuses_double_precision_points(self) -> None:
        with pytest.raises(AcquisitionContextError, match="float64"):
            makeContext(runtime=FLOAT32, pending_points=points([0.25, 0.75]))


class TestTheErrorIsAPlanexsError:
    def test_it_derives_from_the_one_base_class_and_renders_its_hints(self) -> None:
        error = malformed("pending_points", points([0.5, 2.0]))

        assert isinstance(error, PlanexsError)
        rendered = str(error)
        assert rendered.splitlines()[0] == error.message
        assert "Component: acquisition build context" in rendered
        assert "Field: pending_points" in rendered
        assert "Suggestion:" in rendered

    def test_the_fault_in_the_first_set_is_reported_before_the_second(self) -> None:
        with pytest.raises(AcquisitionContextError) as caught:
            makeContext(
                pending_points=points([0.5, 2.0]),
                reference_points=torch.rand(2, 5, dtype=torch.double),
            )

        assert caught.value.field_path == "pending_points"
