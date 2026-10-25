"""Unit tests for `loadPlugins` (specification 5.12, ADR-0010).

The plugins here are real: each test writes modules and `.dist-info` directories of temporary
distributions to disk and puts them on `sys.path`, and `importlib.metadata` finds their entry
points the way it finds those of an installed package.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.KindTable import KindTable
from planexs.core.loadPlugins import PLUGIN_ENTRY_POINT_GROUP, loadPlugins
from planexs.core.PlanexsError import PlanexsError
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError

#: A plugin that registers the components named in `{registrations}` in the `test_kind` kind.
REGISTERING = """\
from pydantic import BaseModel

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.KindTable import KindTable


class Config(BaseModel):
    pass


def build():
    return None


def register(name):
    KindTable.default().registry("test_kind").register(
        ComponentEntry(
            name=name, builder=build, config_schema=Config, description="From a plugin."
        )
    )


{registrations}
"""

MISSING_DEPENDENCY = "import a_dependency_that_is_not_installed_anywhere\n"


def recording(name: str) -> str:
    """The source of a plugin that notes in `sys.t_import_order` that it has been imported."""
    return f"import sys\nsys.t_import_order.append({name!r})\n"


def registering(*names: str, then: str = "") -> str:
    """The source of a plugin that registers components `names`, then runs `then`."""
    registrations = "\n".join(f"register({name!r})" for name in names)
    return REGISTERING.format(registrations=registrations) + then


def ours(names: tuple[str, ...]) -> tuple[str, ...]:
    """The plugins of a test among the names `loadPlugins` returned: those that start `t_`.

    The environment may hold plugins of its own, and they are loaded too.
    """
    return tuple(name for name in names if name.startswith("t_"))


class PluginSite:
    """Writes plugin modules and distributions to directories that are put on `sys.path`.

    Every call writes to a directory of its own: `importlib.metadata` remembers what it found
    in a directory, and a second distribution written to the same one in the same instant
    could go unseen.
    """

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._root = root
        self._monkeypatch = monkeypatch
        self._count = 0

    def _newDirectory(self) -> Path:
        self._count += 1
        directory = self._root / f"site{self._count}"
        directory.mkdir()
        self._monkeypatch.syspath_prepend(str(directory))
        return directory

    def module(self, name: str, source: str) -> None:
        """Writes the module `name`, importable but declared by no distribution."""
        (self._newDirectory() / f"{name}.py").write_text(source, encoding="utf-8")
        importlib.invalidate_caches()

    def distribution(
        self,
        name: str,
        version: str,
        entry_points: Mapping[str, str],
        modules: Mapping[str, str] | None = None,
        group: str = PLUGIN_ENTRY_POINT_GROUP,
    ) -> None:
        """Writes a distribution: its modules, and entry points `{name: value}` in `group`."""
        directory = self._newDirectory()
        for module, source in (modules or {}).items():
            (directory / f"{module}.py").write_text(source, encoding="utf-8")
        info = directory / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
        )
        lines = [f"[{group}]", *(f"{key} = {value}" for key, value in entry_points.items())]
        (info / "entry_points.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        importlib.invalidate_caches()


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PluginSite]:
    """A place to write plugins; the modules they leave in `sys.modules` are removed after."""
    yield PluginSite(tmp_path, monkeypatch)
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", "")).startswith(str(tmp_path)):
            del sys.modules[name]


@pytest.fixture
def components(isolatedKindTable: KindTable) -> Registry[ComponentEntry]:
    """The `test_kind` registry, in an isolated kind table, where test plugins register."""
    registry: Registry[ComponentEntry] = isolatedKindTable.define("test_kind")
    return registry


def failureOf(*arguments: Any) -> RegistryError:
    """The `RegistryError` that `loadPlugins(*arguments)` raises."""
    with pytest.raises(RegistryError) as caught:
        loadPlugins(*arguments)
    return caught.value


class TestTheGroup:
    def test_the_group_is_the_one_of_decision_d_10(self) -> None:
        assert PLUGIN_ENTRY_POINT_GROUP == "planexs.plugins"

    def test_it_returns_a_tuple_of_names(self, components: Registry[ComponentEntry]) -> None:
        names = loadPlugins()
        assert isinstance(names, tuple)
        assert all(isinstance(name, str) for name in names)


class TestEntryPoints:
    def test_an_installed_plugin_registers_its_components(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "plugin-a", "1.0", {"t_a": "t_mod_a"}, {"t_mod_a": registering("a_v1")}
        )
        assert "a_v1" not in components
        names = loadPlugins()
        assert "t_a" in names
        assert components.lookup("a_v1").description == "From a plugin."

    def test_nothing_is_imported_before_the_call(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "plugin-a", "1.0", {"t_a": "t_mod_a"}, {"t_mod_a": registering("a_v1")}
        )
        assert "t_mod_a" not in sys.modules
        loadPlugins()
        assert "t_mod_a" in sys.modules

    def test_the_entry_points_are_imported_in_the_order_of_their_names(
        self,
        site: PluginSite,
        components: Registry[ComponentEntry],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "t_import_order", [], raising=False)
        # Written out of order, so that the order of discovery is not the order of names.
        for name in ("b", "c", "a"):
            site.distribution(
                f"plugin-{name}",
                "1.0",
                {f"t_{name}": f"t_mod_{name}"},
                {f"t_mod_{name}": recording(name)},
            )
        assert ours(loadPlugins()) == ("t_a", "t_b", "t_c")
        # What was imported, in the order it was imported, and not only what was returned.
        assert sys.t_import_order == ["a", "b", "c"]  # type: ignore[attr-defined]

    def test_a_distribution_may_declare_several_entry_points(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "plugin-multi",
            "2.0",
            {"t_one": "t_mod_one", "t_two": "t_mod_two"},
            {"t_mod_one": registering("one_v1"), "t_mod_two": registering("two_v1")},
        )
        assert ours(loadPlugins()) == ("t_one", "t_two")
        assert components.names() == ["one_v1", "two_v1"]

    def test_an_entry_point_of_another_group_is_not_imported(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "other-plugin", "1.0", {"t_other": "t_mod_other"},
            {"t_mod_other": "raise RuntimeError('must not be imported')\n"},
            group="some.other.group",
        )  # fmt: skip
        assert "t_other" not in loadPlugins()
        assert "t_mod_other" not in sys.modules

    def test_an_entry_point_may_name_an_attribute_of_the_module(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        source = registering("attr_v1", then="\nmarker = 1\n")
        site.distribution(
            "plugin-attr", "1.0", {"t_attr": "t_mod_attr:marker"}, {"t_mod_attr": source}
        )
        assert "t_attr" in loadPlugins()
        assert "attr_v1" in components


class TestExplicitModules:
    def test_a_module_is_imported_by_its_name(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.module("t_explicit", registering("explicit_v1"))
        assert ours(loadPlugins(["t_explicit"])) == ("t_explicit",)
        assert "explicit_v1" in components

    def test_a_single_name_is_one_module_and_not_a_list_of_letters(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.module("t_explicit", registering("explicit_v1"))
        assert ours(loadPlugins("t_explicit")) == ("t_explicit",)
        assert "explicit_v1" in components

    def test_any_iterable_of_names_is_accepted(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.module("t_one", registering("one_v1"))
        site.module("t_two", registering("two_v1"))
        assert ours(loadPlugins(("t_one", "t_two"))) == ("t_one", "t_two")
        site.module("t_three", registering("three_v1"))
        assert ours(loadPlugins(name for name in ["t_three"])) == ("t_three",)

    def test_the_modules_are_imported_in_the_order_given(
        self,
        site: PluginSite,
        components: Registry[ComponentEntry],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "t_import_order", [], raising=False)
        for name in ("t_m1", "t_m2", "t_m3"):
            site.module(name, recording(name))
        assert ours(loadPlugins(["t_m3", "t_m1", "t_m2"])) == ("t_m3", "t_m1", "t_m2")
        assert sys.t_import_order == ["t_m3", "t_m1", "t_m2"]  # type: ignore[attr-defined]

    def test_the_modules_come_after_the_entry_points(
        self,
        site: PluginSite,
        components: Registry[ComponentEntry],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "t_import_order", [], raising=False)
        site.distribution(
            "plugin-z", "1.0", {"t_z": "t_mod_z"}, {"t_mod_z": recording("entry")}
        )
        site.module("t_a_module", recording("module"))
        names = loadPlugins(["t_a_module"])
        # The module sorts first by name, but it is an explicit module: it goes last.
        assert names.index("t_z") < names.index("t_a_module")
        assert sys.t_import_order == ["entry", "module"]  # type: ignore[attr-defined]

    def test_a_module_that_is_also_an_entry_point_is_imported_once(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "plugin-s", "1.0", {"t_s": "t_shared"}, {"t_shared": registering("s_v1")}
        )
        names = loadPlugins(["t_shared"])
        assert "t_s" in names and "t_shared" in names
        assert components.names() == ["s_v1"]

    def test_no_module_loads_only_the_entry_points(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "plugin-a", "1.0", {"t_a": "t_mod_a"}, {"t_mod_a": registering("a_v1")}
        )
        assert ours(loadPlugins([])) == ("t_a",)

    def test_a_module_that_registers_nothing_is_still_imported(
        self, components: Registry[ComponentEntry]
    ) -> None:
        assert "json" in loadPlugins(["json"])
        assert components.names() == []

    @pytest.mark.parametrize(
        "given", [3, None, sys, b"t_x"], ids=["int", "None", "module", "bytes"]
    )
    def test_a_module_must_be_given_by_its_name(self, given: object) -> None:
        error = failureOf([given])
        assert error.message == (
            f"A plugin module must be given by its name, a string; got {type(given).__name__}."
        )
        assert error.component == "plugin"
        assert error.suggestion is not None


class TestImportingTwice:
    def test_a_second_call_gives_the_same_names_and_registers_nothing_twice(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "plugin-a", "1.0", {"t_a": "t_mod_a"}, {"t_mod_a": registering("a_v1")}
        )
        site.module("t_explicit", registering("explicit_v1"))
        first = loadPlugins(["t_explicit"])
        second = loadPlugins(["t_explicit"])
        assert second == first
        assert components.names() == ["a_v1", "explicit_v1"]

    def test_a_plugin_runs_once_however_often_it_is_loaded(
        self, site: PluginSite, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "t_plugin_runs", 0, raising=False)
        source = "import sys\nsys.t_plugin_runs += 1\n"
        site.distribution(
            "plugin-count", "1.0", {"t_count": "t_mod_count"}, {"t_mod_count": source}
        )
        site.module("t_explicit_count", source)
        for _ in range(5):
            loadPlugins(["t_explicit_count"])
        assert sys.t_plugin_runs == 2  # type: ignore[attr-defined]  # once each, not five times

    def test_a_plugin_that_appears_between_two_calls_is_loaded_by_the_second(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "plugin-a", "1.0", {"t_a": "t_mod_a"}, {"t_mod_a": registering("a_v1")}
        )
        assert ours(loadPlugins()) == ("t_a",)
        site.distribution(
            "plugin-b", "1.0", {"t_b": "t_mod_b"}, {"t_mod_b": registering("b_v1")}
        )
        assert ours(loadPlugins()) == ("t_a", "t_b")
        assert components.names() == ["a_v1", "b_v1"]

    def test_loading_the_same_module_twice_in_one_call_is_harmless(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.module("t_twice", registering("twice_v1"))
        assert ours(loadPlugins(["t_twice", "t_twice"])) == ("t_twice", "t_twice")
        assert components.names() == ["twice_v1"]


class TestAPluginThatFails:
    def test_a_missing_dependency_raises_a_registry_error_that_names_everything(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "bad-dist", "0.3", {"t_bad": "t_mod_bad"}, {"t_mod_bad": MISSING_DEPENDENCY}
        )
        error = failureOf()
        assert isinstance(error, PlanexsError)
        assert str(error) == (
            "Cannot load the plugin 't_bad' (entry point 't_mod_bad' of bad-dist 0.3): "
            "ModuleNotFoundError: "
            "No module named 'a_dependency_that_is_not_installed_anywhere'\n"
            "Component: plugin 't_bad'\n"
            "Suggestion: fix the plugin, or uninstall the package that provides it"
        )

    def test_the_original_error_is_the_cause(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "bad-dist", "0.3", {"t_bad": "t_mod_bad"}, {"t_mod_bad": MISSING_DEPENDENCY}
        )
        assert isinstance(failureOf().__cause__, ModuleNotFoundError)

    def test_an_error_in_the_code_of_the_plugin(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        source = "x = 1\nraise RuntimeError('boom')\n"
        site.distribution("bad-dist", "0.3", {"t_bad": "t_mod_bad"}, {"t_mod_bad": source})
        error = failureOf()
        assert error.message.endswith("RuntimeError: boom")
        assert isinstance(error.__cause__, RuntimeError)

    def test_a_syntax_error_in_the_plugin(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution("bad-dist", "0.3", {"t_bad": "t_mod_bad"}, {"t_mod_bad": "def (:\n"})
        assert "SyntaxError" in failureOf().message

    def test_an_error_without_a_message_is_named_by_its_type(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "bad-dist", "0.3", {"t_bad": "t_mod_bad"}, {"t_mod_bad": "raise RuntimeError()\n"}
        )
        assert failureOf().message.endswith("): RuntimeError")

    def test_an_entry_point_that_names_a_module_that_does_not_exist(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution("ghost-dist", "1.0", {"t_ghost": "t_no_such_module"})
        error = failureOf()
        assert "plugin 't_ghost'" in error.message
        assert "ModuleNotFoundError: No module named 't_no_such_module'" in error.message

    def test_an_entry_point_that_names_an_attribute_that_does_not_exist(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "attr-dist", "1.0", {"t_attr": "t_mod_attr:nope"}, {"t_mod_attr": "x = 1\n"}
        )
        error = failureOf()
        assert "plugin 't_attr'" in error.message
        assert "AttributeError" in error.message
        assert "nope" in error.message

    def test_a_component_registered_under_a_name_that_is_taken(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "a-dist", "1.0", {"t_a": "t_mod_a"}, {"t_mod_a": registering("same_v1")}
        )
        site.distribution(
            "b-dist", "2.5", {"t_b": "t_mod_b"}, {"t_mod_b": registering("same_v1")}
        )
        error = failureOf()
        assert "plugin 't_b'" in error.message
        assert "b-dist 2.5" in error.message
        assert "A test_kind named 'same_v1' is already registered." in error.message
        assert isinstance(error.__cause__, RegistryError)

    def test_a_plugin_that_registers_in_a_kind_that_does_not_exist(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        source = registering("x_v1").replace('"test_kind"', '"no_such_kind"')
        site.distribution("kind-dist", "1.0", {"t_kind": "t_mod_kind"}, {"t_mod_kind": source})
        assert "Unknown component kind 'no_such_kind'" in failureOf().message

    def test_an_explicit_module_that_fails(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.module("t_bad_module", MISSING_DEPENDENCY)
        error = failureOf(["t_bad_module"])
        assert error.component == "plugin module 't_bad_module'"
        assert error.message.startswith("Cannot load the plugin module 't_bad_module': ")
        assert "ModuleNotFoundError" in error.message
        assert error.suggestion == (
            "check the module name, and that the module can be imported on its own"
        )

    def test_an_explicit_module_that_does_not_exist(
        self, components: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(["t_no_such_package.kernels"])
        assert "plugin module 't_no_such_package.kernels'" in error.message
        assert "No module named 't_no_such_package'" in error.message

    @pytest.mark.parametrize("name", [".relative", ""], ids=["relative", "empty"])
    def test_a_name_that_cannot_be_imported_at_all(
        self, components: Registry[ComponentEntry], name: str
    ) -> None:
        error = failureOf([name])
        assert f"plugin module {name!r}" in error.message
        assert error.__cause__ is not None

    def test_the_failure_is_logged_as_a_warning(
        self,
        site: PluginSite,
        components: Registry[ComponentEntry],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        site.distribution(
            "bad-dist", "0.3", {"t_bad": "t_mod_bad"}, {"t_mod_bad": MISSING_DEPENDENCY}
        )
        with caplog.at_level(logging.WARNING, logger="planexs.core.loadPlugins"):
            failureOf()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "plugin 't_bad'" in warnings[0].getMessage()
        assert "ModuleNotFoundError" in warnings[0].getMessage()

    def test_a_plugin_that_loads_logs_no_warning(
        self,
        site: PluginSite,
        components: Registry[ComponentEntry],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        site.distribution(
            "plugin-a", "1.0", {"t_a": "t_mod_a"}, {"t_mod_a": registering("a_v1")}
        )
        with caplog.at_level(logging.WARNING, logger="planexs.core.loadPlugins"):
            loadPlugins()
        assert caplog.records == []


class TestWhatHappensAfterAFailure:
    def test_the_first_failing_plugin_in_the_order_of_names_is_the_one_reported(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution("two", "1.0", {"t_x2": "t_mod_x2"}, {"t_mod_x2": MISSING_DEPENDENCY})
        site.distribution("one", "1.0", {"t_x1": "t_mod_x1"}, {"t_mod_x1": MISSING_DEPENDENCY})
        assert failureOf().component == "plugin 't_x1'"

    def test_the_plugins_before_the_failing_one_stay_loaded_and_those_after_are_not_imported(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution("a", "1.0", {"t_a": "t_mod_a"}, {"t_mod_a": registering("a_v1")})
        site.distribution("b", "1.0", {"t_b": "t_mod_b"}, {"t_mod_b": MISSING_DEPENDENCY})
        site.distribution("c", "1.0", {"t_c": "t_mod_c"}, {"t_mod_c": registering("c_v1")})
        failureOf()
        assert components.names() == ["a_v1"]
        assert "t_mod_c" not in sys.modules

    def test_explicit_modules_are_not_imported_after_an_entry_point_failed(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution("b", "1.0", {"t_b": "t_mod_b"}, {"t_mod_b": MISSING_DEPENDENCY})
        site.module("t_explicit", registering("explicit_v1"))
        failureOf(["t_explicit"])
        assert "t_explicit" not in sys.modules

    def test_a_second_call_raises_again_and_still_names_the_plugin(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution("b", "1.0", {"t_b": "t_mod_b"}, {"t_mod_b": MISSING_DEPENDENCY})
        assert failureOf().component == "plugin 't_b'"
        assert failureOf().component == "plugin 't_b'"

    def test_a_plugin_that_failed_after_registering_leaves_its_components_registered(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        source = registering("early_v1", then="raise RuntimeError('late failure')\n")
        site.distribution("p", "1.0", {"t_p": "t_mod_p"}, {"t_mod_p": source})
        assert "late failure" in failureOf().message
        assert "early_v1" in components
        # Running it again registers `early_v1` a second time: the error says so, and still
        # names the plugin (documented on `loadPlugins`).
        second = failureOf()
        assert second.component == "plugin 't_p'"
        assert "already registered" in second.message

    def test_an_exception_that_is_not_an_exception_is_not_caught(
        self, site: PluginSite, components: Registry[ComponentEntry]
    ) -> None:
        site.distribution(
            "k", "1.0", {"t_k": "t_mod_k"}, {"t_mod_k": "raise KeyboardInterrupt\n"}
        )
        with pytest.raises(KeyboardInterrupt):
            loadPlugins()
