"""Pure domain models: zero ML dependencies (design philosophy, section 4.1).

Knows no optimization engine: the mapping of a domain object to an engine's own
vocabulary lives with that engine's adapter (specification 3.9), so that planexs'
public surface stays stable even if the backend underneath it changes.
"""
