"""A safe YAML loader that reads configuration files as plain data."""

from __future__ import annotations

import re
from collections.abc import Hashable
from typing import Any

import yaml
from yaml.constructor import ConstructorError

_MERGE_TAG = "tag:yaml.org,2002:merge"
_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"
_FLOAT_TAG = "tag:yaml.org,2002:float"
_REPLACED_TAGS = frozenset({_TIMESTAMP_TAG, _FLOAT_TAG})

#: Every number JSON writes is read as the same number. PyYAML's own float pattern needs a dot
#: and a signed exponent, so it reads `1e-3` and `1.5e3` as strings, where JSON reads numbers.
#: YAML's base 60 numbers (`1:30.5`) are not accepted as floats.
_JSON_FLOAT = re.compile(
    r"""^(?:[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
    |[-+]?\.[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.VERBOSE,
)


class UniqueKeyYamlLoader(yaml.SafeLoader):
    """Loads a YAML document as plain data: safe, with no duplicate keys, numbers as in JSON.

    It is what `parseConfig` reads YAML with, so that a configuration file means what the same
    configuration in JSON or Python means (specification 10.8). It differs from
    `yaml.SafeLoader` in three ways, and from nothing else:

    - A key written twice in one mapping is an error, where `SafeLoader` keeps the last value
      and drops the first without a word. The error is a `ConstructorError`, whose mark is the
      line and column of the second key and whose message gives the line of the first. The
      keys that a merge key (`<<`) brings in are not counted: a mapping may override them.
    - A date or a time (`2020-01-01`) stays a string, where `SafeLoader` makes a `date` or a
      `datetime`: those are not plain data, and JSON has no such value.
    - A number is read as JSON reads it: `1e-3` and `1.5E3` are floats.

    Like `SafeLoader`, it builds only plain values, so reading a file never runs code: a tag
    such as `!!python/object` is an error. The rest is YAML 1.1 as PyYAML reads it, which
    differs from JSON for a few unquoted scalars: `yes`, `no`, `on` and `off` are booleans, a
    number with a leading zero is octal and `12:30` is a base 60 integer. Quote such a value to
    keep it a string.

    This class changes only itself: `yaml.SafeLoader` and `yaml.safe_load` behave as before.
    """

    # `SafeLoader`'s patterns, less those for dates and floats (floats are replaced below).
    # PyYAML's own extension point: a subclass replaces this table, which is why it is a
    # mutable class attribute (RUF012) that its stubs declare as an instance one.
    # A new dictionary of new lists: `SafeLoader` itself is left as it was.
    yaml_implicit_resolvers = {  # noqa: RUF012
        first: [(tag, pattern) for tag, pattern in resolvers if tag not in _REPLACED_TAGS]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        """Builds a mapping, refusing a key that appears twice in it.

        Args:
            node: The mapping node.
            deep: Whether to build the values at once, as `SafeLoader` does for a mapping that
                is itself a key.

        Returns:
            The mapping.

        Raises:
            ConstructorError: if a key is written twice, or by the parent class for a key that
                cannot be hashed.
        """
        first_lines: dict[Hashable, int] = {}
        for key_node, _ in node.value:
            if key_node.tag == _MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                continue  # the parent class reports a key that cannot be hashed
            if key in first_lines:
                raise ConstructorError(
                    None,
                    None,
                    f"found duplicate key {key!r}, first defined on line {first_lines[key]}",
                    key_node.start_mark,
                )
            first_lines[key] = key_node.start_mark.line + 1
        return super().construct_mapping(node, deep=deep)


# The float pattern above replaces PyYAML's: a float can only start with one of these.
UniqueKeyYamlLoader.add_implicit_resolver(_FLOAT_TAG, _JSON_FLOAT, list("-+0123456789."))
