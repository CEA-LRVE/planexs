"""Pure domain models: zero ML dependencies (design philosophy, section 4.1).

Knows no optimization engine: the mapping of a domain object to an engine's own
vocabulary lives with that engine's adapter (specification 3.9), so that planexs'
public surface stays stable even if the backend underneath it changes.
"""

# Importing the subpackages registers the parameter and constraint types and the outcome kinds
# (specification 5.12, "Population"), so that the domain can rebuild them from a stored
# document whichever module of it was imported first.
from planexs.domain import constraints, outcome_kinds, parameters

__all__ = ["constraints", "outcome_kinds", "parameters"]
