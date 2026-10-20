"""Unit tests for the error hierarchy of `planexs.core` (specification 3.6, 10.10)."""

from __future__ import annotations

import copy
import importlib
import inspect
import itertools
import pickle
import pkgutil
from typing import Any

import pytest

import planexs.core
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ConfigError import ConfigError
from planexs.core.ModelNotReadyError import ModelNotReadyError
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RegistryError import RegistryError

SUBCLASSES = [ConfigError, RegistryError, CompatibilityError, ModelNotReadyError]
ALL_ERRORS = [PlanexsError, *SUBCLASSES]

#: Shorthand that keeps the parametrized signatures on one line.
ErrorClass = type[PlanexsError]


class TestInheritance:
    def test_the_base_class_is_an_exception(self) -> None:
        assert issubclass(PlanexsError, Exception)

    @pytest.mark.parametrize("error_class", SUBCLASSES)
    def test_every_subclass_derives_from_the_base_class(self, error_class: ErrorClass) -> None:
        assert issubclass(error_class, PlanexsError)

    @pytest.mark.parametrize("error_class", SUBCLASSES)
    def test_every_subclass_derives_directly_from_the_base_class(
        self, error_class: ErrorClass
    ) -> None:
        # A flat hierarchy: catching one kind never catches another by accident.
        assert error_class.__bases__ == (PlanexsError,)

    @pytest.mark.parametrize("error_class", SUBCLASSES)
    def test_one_handler_for_the_base_class_catches_every_subclass(
        self, error_class: ErrorClass
    ) -> None:
        with pytest.raises(PlanexsError) as caught:
            raise error_class("boom")
        assert type(caught.value) is error_class

    @pytest.mark.parametrize(
        ("raised", "handler"), list(itertools.permutations(SUBCLASSES, 2))
    )
    def test_a_subclass_handler_does_not_catch_a_sibling(
        self, raised: ErrorClass, handler: ErrorClass
    ) -> None:
        assert not isinstance(raised("boom"), handler)

    def test_every_exception_defined_in_core_derives_from_the_base_class(self) -> None:
        # Specification 10.10: every error the library raises derives from one base class.
        # This walks the real package, so an error added to `core` later is covered.
        defined: list[type[BaseException]] = []
        for module_info in pkgutil.iter_modules(planexs.core.__path__):
            module = importlib.import_module(f"planexs.core.{module_info.name}")
            for _, member in inspect.getmembers(module, inspect.isclass):
                if issubclass(member, BaseException) and member.__module__ == module.__name__:
                    defined.append(member)
        assert set(ALL_ERRORS) <= set(defined)
        stray = [error.__name__ for error in defined if not issubclass(error, PlanexsError)]
        assert stray == []


@pytest.mark.parametrize("error_class", ALL_ERRORS)
class TestRendering:
    def test_a_message_alone_renders_as_the_message(self, error_class: ErrorClass) -> None:
        assert str(error_class("Something went wrong.")) == "Something went wrong."

    def test_a_component_hint_goes_on_its_own_line(self, error_class: ErrorClass) -> None:
        error = error_class("Bad.", component="surrogate")
        assert str(error) == "Bad.\nComponent: surrogate"

    def test_a_field_path_hint_goes_on_its_own_line(self, error_class: ErrorClass) -> None:
        error = error_class("Bad.", field_path="policy.surrogate.kernel.nu")
        assert str(error) == "Bad.\nField: policy.surrogate.kernel.nu"

    def test_a_suggestion_hint_goes_on_its_own_line(self, error_class: ErrorClass) -> None:
        error = error_class("Bad.", suggestion="use one of: a, b")
        assert str(error) == "Bad.\nSuggestion: use one of: a, b"

    def test_all_hints_render_in_a_fixed_order_one_per_line(
        self, error_class: ErrorClass
    ) -> None:
        error = error_class(
            "Bad.",
            # Given out of order on purpose: the rendering order is not the call order.
            suggestion="use one of: a, b",
            field_path="policy.surrogate.type",
            component="surrogate",
        )
        assert str(error).splitlines() == [
            "Bad.",
            "Component: surrogate",
            "Field: policy.surrogate.type",
            "Suggestion: use one of: a, b",
        ]

    def test_two_hints_leave_out_the_third(self, error_class: ErrorClass) -> None:
        error = error_class("Bad.", component="surrogate", suggestion="fix it")
        assert str(error).splitlines() == [
            "Bad.",
            "Component: surrogate",
            "Suggestion: fix it",
        ]

    def test_there_is_no_trailing_newline(self, error_class: ErrorClass) -> None:
        assert not str(error_class("Bad.", component="surrogate")).endswith("\n")
        assert not str(error_class("Bad.")).endswith("\n")

    def test_a_multi_line_message_is_rendered_as_given(self, error_class: ErrorClass) -> None:
        error = error_class("First line.\nSecond line.", suggestion="fix it")
        assert str(error).splitlines() == ["First line.", "Second line.", "Suggestion: fix it"]

    def test_the_hints_are_kept_as_attributes(self, error_class: ErrorClass) -> None:
        error = error_class("Bad.", component="c", field_path="a.b", suggestion="s")
        assert (error.message, error.component, error.field_path, error.suggestion) == (
            "Bad.",
            "c",
            "a.b",
            "s",
        )

    def test_hints_default_to_none(self, error_class: ErrorClass) -> None:
        error = error_class("Bad.")
        assert (error.component, error.field_path, error.suggestion) == (None, None, None)

    def test_hints_are_keyword_only(self, error_class: ErrorClass) -> None:
        positional: tuple[Any, ...] = ("Bad.", "surrogate")
        with pytest.raises(TypeError):
            error_class(*positional)

    def test_the_rendering_is_what_a_traceback_shows(self, error_class: ErrorClass) -> None:
        with pytest.raises(error_class, match=r"Bad\.\nField: a\.b") as caught:
            raise error_class("Bad.", field_path="a.b")
        assert caught.value.args == ("Bad.",)

    def test_survives_pickling_and_copying_with_its_hints(
        self, error_class: ErrorClass
    ) -> None:
        # Errors cross process boundaries (parallel test runs, worker pools).
        error = error_class("Bad.", component="c", field_path="a.b", suggestion="s")
        for clone in (pickle.loads(pickle.dumps(error)), copy.deepcopy(error)):
            assert type(clone) is error_class
            assert str(clone) == str(error)
            assert (clone.component, clone.field_path, clone.suggestion) == ("c", "a.b", "s")
