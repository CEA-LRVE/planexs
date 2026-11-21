"""Unit tests for `AbstractAcquisitionBuilder` (roadmap task P5-01, specification 5.8, 6).

What is tested: what the contract is (what is abstract, what has a default), the flow from a
configuration mapping to a native acquisition function through the registry, the pending points
passing through the context, and the reference set: how the size is read from the
configuration, what the guard says when a builder that needs a set is not given one, and that
a set drawn by the feasible sampler is what the context accepts.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from acquisition_fakes import N_COLUMNS, defineFake, makeContext, makeProblem, points
from botorch.acquisition.acquisition import AcquisitionFunction

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.sampleFeasibleEncoded import sampleFeasibleEncoded
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder


def builderFor(mapping: dict[str, Any]) -> AbstractAcquisitionBuilder[Any]:
    """The builder that the registry makes from a configuration mapping."""
    resolved = resolveComponent(acquisitionRegistry(), mapping)
    builder = resolved.entry.builder(resolved.config)
    assert isinstance(builder, AbstractAcquisitionBuilder)
    return builder


class TestTheContract:
    def test_the_contract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractAcquisitionBuilder(object())  # type: ignore[abstract, type-var]

    def test_only_build_is_abstract(self) -> None:
        assert AbstractAcquisitionBuilder.__abstractmethods__ == {"build"}

    def test_a_subclass_that_does_not_build_cannot_be_instantiated(self) -> None:
        class Lazy(AbstractAcquisitionBuilder[Any]):
            pass

        with pytest.raises(TypeError, match="build"):
            Lazy(object())  # type: ignore[abstract]

    def test_by_default_a_builder_needs_no_reference_set(self) -> None:
        assert AbstractAcquisitionBuilder.needs_reference_points is False

    def test_by_default_the_reference_set_has_no_points(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("plain_v1")

        assert builderFor({"type": "plain_v1"}).referencePointCount() == 0

    def test_the_flag_is_readable_from_the_class_without_building_anything(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        # The policy asks the class, before it has a configured builder.
        assert defineFake("plain_v1").needs_reference_points is False
        assert defineFake("sized_v1", with_reference_set=True).needs_reference_points is True

    def test_from_config_makes_an_instance_of_the_class_it_is_called_on(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        cls = defineFake("plain_v1")
        config = acquisitionRegistry().lookup("plain_v1").config_schema()

        builder = cls.fromConfig(config)

        assert type(builder) is cls
        assert builder.config is config

    def test_the_configuration_of_a_builder_is_read_only(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("plain_v1")
        builder = builderFor({"type": "plain_v1"})

        with pytest.raises(AttributeError):
            builder.config = None  # type: ignore[misc]


class TestFromAMappingToANativeAcquisition:
    def test_the_registry_builds_a_native_acquisition_function_of_the_engine(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("plain_v1")
        context = makeContext()

        acquisition = builderFor({"type": "plain_v1"}).build(context)

        assert isinstance(acquisition, AcquisitionFunction)
        assert acquisition.model is context.surrogate.model  # the model the policy uses

    def test_the_acquisition_scores_candidates_of_the_encoded_space(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("plain_v1")
        acquisition = builderFor({"type": "plain_v1"}).build(makeContext())

        candidates = points([0.1, 0.2], [0.5, 0.5], [0.9, 0.9]).unsqueeze(1)

        scores = acquisition(candidates)

        assert tuple(scores.shape) == (3,)
        assert torch.allclose(scores, torch.tensor([0.3, 1.0, 1.8], dtype=scores.dtype))

    def test_the_pending_points_of_the_context_reach_the_acquisition(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("plain_v1")
        pending = points([0.25, 0.75], [0.5, 0.5])

        acquisition = builderFor({"type": "plain_v1"}).build(
            makeContext(pending_points=pending)
        )

        # The engine keeps its own detached copy of the points: the content is what is passed.
        assert torch.equal(acquisition.X_pending, pending)

    def test_without_pending_points_the_acquisition_has_none(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("plain_v1")

        acquisition = builderFor({"type": "plain_v1"}).build(makeContext())

        assert acquisition.X_pending is None

    def test_a_builder_is_configured_once_and_builds_as_often_as_it_is_asked(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("plain_v1")
        builder = builderFor({"type": "plain_v1"})
        first_pending = points([0.1, 0.1])
        second_pending = points([0.9, 0.9], [0.8, 0.8])

        first = builder.build(makeContext(pending_points=first_pending))
        second = builder.build(makeContext(pending_points=second_pending))

        assert first is not second
        assert torch.equal(first.X_pending, first_pending)  # nothing of the first build leaks
        assert torch.equal(second.X_pending, second_pending)

    def test_the_builder_changes_nothing_in_the_context(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("sized_v1", with_reference_set=True)
        pending = points([0.25, 0.75])
        reference = points([0.1, 0.1], [0.9, 0.9], [0.5, 0.5])
        before = (pending.clone(), reference.clone())
        context = makeContext(pending_points=pending, reference_points=reference)

        builderFor({"type": "sized_v1"}).build(context)

        assert torch.equal(context.pending_points, before[0])  # type: ignore[arg-type]
        assert torch.equal(context.reference_points, before[1])  # type: ignore[arg-type]


class TestTheReferenceSet:
    def test_the_size_is_read_from_the_configuration_of_the_builder(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("sized_v1", with_reference_set=True)

        assert builderFor({"type": "sized_v1"}).referencePointCount() == 16  # the default
        assert (
            builderFor({"type": "sized_v1", "n_reference_points": 4}).referencePointCount()
            == 4
        )

    def test_the_guard_returns_the_set_of_the_context_itself(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("sized_v1", with_reference_set=True)
        reference = points([0.1, 0.1], [0.9, 0.9])

        found = builderFor({"type": "sized_v1"}).requireReferencePoints(
            makeContext(reference_points=reference)
        )

        assert found is reference

    def test_the_guard_names_the_acquisition_the_field_and_the_way_to_fix_it(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("sized_v1", with_reference_set=True)

        with pytest.raises(AcquisitionContextError) as caught:
            builderFor({"type": "sized_v1"}).requireReferencePoints(makeContext())

        assert "'sized_v1'" in caught.value.message
        assert caught.value.component == "acquisition 'sized_v1'"
        assert caught.value.field_path == "reference_points"
        assert caught.value.suggestion is not None
        assert "referencePointCount" in caught.value.suggestion

    def test_a_builder_that_needs_a_set_fails_to_build_without_one(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("sized_v1", with_reference_set=True)

        with pytest.raises(AcquisitionContextError, match="reference set"):
            builderFor({"type": "sized_v1"}).build(makeContext())

    def test_a_builder_that_needs_a_set_is_given_it_when_it_builds(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("sized_v1", with_reference_set=True)
        reference = points([0.1, 0.1], [0.9, 0.9])

        acquisition = builderFor({"type": "sized_v1"}).build(
            makeContext(reference_points=reference)
        )

        assert acquisition.reference is reference  # type: ignore[attr-defined]

    def test_a_builder_that_needs_none_ignores_a_set_it_is_given(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("plain_v1")
        reference = points([0.1, 0.1])

        acquisition = builderFor({"type": "plain_v1"}).build(
            makeContext(reference_points=reference)
        )

        assert acquisition.reference is None  # type: ignore[attr-defined]


class TestAReferenceSetDrawnTheWayThePolicyWillDrawIt:
    """The policy reads the flag, asks for the size, draws that many points with the feasible
    sampler and hands them over: what the sampler returns is what the context must accept.
    """

    def test_the_protocol_end_to_end(self, isolatedAcquisitionKind: KindTable) -> None:
        cls = defineFake("sized_v1", with_reference_set=True)
        builder = builderFor({"type": "sized_v1", "n_reference_points": 12})
        problem = makeProblem()
        encoder = SearchSpaceEncoder(problem.search_space)

        reference = None
        if cls.needs_reference_points:
            reference = sampleFeasibleEncoded(
                encoder,
                EncodedConstraints(problem.search_space, encoder),
                builder.referencePointCount(),
                seed=3,
            )
        context = makeContext(reference_points=reference)
        acquisition = builder.build(context)

        assert tuple(acquisition.reference.shape) == (12, N_COLUMNS)  # type: ignore[attr-defined]
        assert acquisition.reference is context.reference_points  # type: ignore[attr-defined]
