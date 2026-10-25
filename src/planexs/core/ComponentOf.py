"""The annotation that makes a configuration field hold another component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.configErrors import toValidationError
from planexs.core.KindTable import KindTable
from planexs.core.Registry import Registry
from planexs.core.selectComponent import selectComponent


@dataclass(frozen=True)
class ComponentOf:
    """Declares that a configuration field holds a component of one kind (specification 5.12).

    It goes in the `Annotated` metadata of the field, whose declared type is `ComponentConfig`.
    The kind names the registry the component is looked up in, so the schema of a surrogate can
    hold a kernel, and the schema of a composed policy can hold a design, a surrogate, an
    acquisition and an optimizer::

        class SingleTaskGPConfig(ComponentConfig):
            type: Literal["single_task_gp_v1"] = "single_task_gp_v1"
            kernel: Annotated[ComponentConfig, ComponentOf("kernel")] = Field(
                default_factory=lambda: {"type": "botorch_default_v1"},
                validate_default=True,
                description="The kernel of the GP: the component of kind 'kernel' to use.",
            )

    A field holds one component, or a list of components of the kind if the marker is put on
    the element: `list[Annotated[ComponentConfig, ComponentOf("analysis")]]`. It also works
    inside `| None` and any other container pydantic validates.

    When the parent is validated, each value is resolved as `resolveComponent` does: its `type`
    selects the registered component, and the whole mapping is validated against that
    component's schema. The field then holds an instance of that schema, with every default
    filled in, and nested components are resolved the same way to any depth. A value that is
    already a configuration object is accepted if its class is the schema registered under its
    `type`.

    A problem anywhere below the field is reported with its full path (`surrogate.kernel.nu`,
    `analyses[1].level`), and a `type` that is missing or names nothing is reported at
    `<path>.type` with the registered names.

    Dumping a parent writes each nested component in full, with its own fields: the field is
    serialized as the component it holds, not as `ComponentConfig`, which has only `type`.

    A default component is written as a mapping, as above, so that the schema does not import
    the component it names: `validate_default=True` resolves it like any other value, and the
    name stays in the schema field, the one place a default may live (specification 3.4). The
    registry is looked up each time a value is validated, so the schema can be declared before
    the kind it names is defined, but the kind must be defined, and its components registered
    (the modules that do so imported), by then.

    Args:
        kind: The kind of component the field holds, the name `defineKind` was given.

    Attributes:
        kind: The kind of component the field holds.
    """

    kind: str

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Tells pydantic how to validate and serialize a field marked with this kind.

        Pydantic calls this when it builds the model that holds the field. The schema of the
        declared type (`ComponentConfig`) is replaced, not wrapped: which schema applies is
        only known once the value, and its `type`, has been read. For the same reason the
        field is serialized as what it holds: a schema with no serializer of its own is
        serialized by the runtime type of the value, where the schema of `ComponentConfig`
        would write only its `type`.

        Args:
            source_type: The declared type of the field. Not used.
            handler: Pydantic's schema generator. Not used.

        Returns:
            A schema that resolves a value through the registry.
        """
        return core_schema.no_info_plain_validator_function(self._resolve)

    def _resolve(self, value: Any) -> ComponentConfig:
        """Resolves one value of the field: a mapping, or a configuration object.

        Args:
            value: The value being validated.

        Returns:
            The validated configuration of the component the value names.

        Raises:
            ValidationError: if the value does not name a registered component, or does not
                validate against its schema. Pydantic merges it into the errors of the parent.
            RegistryError: if the kind is not defined, or the component is registered with a
                schema that does not derive from `ComponentConfig`.
        """
        registry = KindTable.default().registry(self.kind)
        if isinstance(value, ComponentConfig):
            return self._accept(registry, value)
        try:
            _, schema = selectComponent(registry, value)
        except ConfigError as error:
            raise toValidationError(error, value) from error
        # Pydantic merges the location of any problem in here into that of the field.
        return schema.model_validate(dict(value))

    def _accept(self, registry: Registry[Any], config: ComponentConfig) -> ComponentConfig:
        """Accepts a configuration object built in Python if it fits its `type`.

        Args:
            registry: The registry of the kind.
            config: The object.

        Returns:
            `config` itself: it was validated when it was built.

        Raises:
            ValidationError: if its `type` names nothing in the registry, or it is not an
                instance of the schema registered under that name.
        """
        try:
            entry, schema = selectComponent(registry, {"type": config.type})
            if not isinstance(config, schema):
                raise ConfigError(
                    f"A {type(config).__name__} is not the configuration schema of the "
                    f"{self.kind} {entry.name!r}, which is {schema.__name__}.",
                    component=self.kind,
                )
        except ConfigError as error:
            raise toValidationError(error, config) from error
        return config
