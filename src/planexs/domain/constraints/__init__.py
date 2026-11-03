"""Parameter-level constraints: `Constraint` (ABC), `LinearConstraint` (enforced
natively by the optimization backend) and `NonlinearConstraint` (FR17, enforced
only by planexs' own acquisition layer, see ADR-0006).

The two are the registered types of the `constraint` kind of components
(`registerConstraint`, specification 5.3, 5.12): a type is registered when its module is
imported, so every module that holds one is imported here.
"""

from planexs.domain.constraints import (
    LinearConstraint,
    NonlinearConstraint,
)

__all__ = ["LinearConstraint", "NonlinearConstraint"]
