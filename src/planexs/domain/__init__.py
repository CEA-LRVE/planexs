"""Pure domain models: zero ML dependencies (design philosophy, section 4.1).

Mirrors `ax.api.configs`' own parameter shapes field for field wherever one
exists, so that planexs' public surface stays stable even if the backend
underneath it changes.
"""
