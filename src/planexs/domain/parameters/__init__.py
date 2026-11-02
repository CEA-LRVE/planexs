"""Parameter kinds: `Range`/`Real`/`Integer`, `Choice`/`Categorical`/
`Boolean`/`Fixed`, and `Derived`. Re-exported at the top-level `planexs`
package; import from there in application code.

The three base kinds are the registered types of the `parameter` kind of components
(`registerParameter`, specification 5.3, 5.12): a type is registered when its module is
imported, so every module that holds one is imported here. The convenience constructors
(`Real`, `Integer`, `Categorical`, `Boolean`, `Fixed`) are not registered: they build the
base kinds.
"""

from planexs.domain.parameters import (
    Choice,
    Derived,
    Range,
)

__all__ = ["Choice", "Derived", "Range"]
