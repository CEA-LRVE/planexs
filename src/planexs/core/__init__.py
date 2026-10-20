"""The `core` layer: the mechanics that every other layer of planexs builds on.

It holds the error hierarchy, registries and the kind table, component metadata, capability
tags and the compatibility check, the component configuration base, random-number derivation,
runtime settings and plugin discovery (specification 5.1).

The rule of this layer: it imports nothing from the rest of planexs (only from `planexs.core`
itself) and no machine-learning library (torch, BoTorch, GPyTorch or Ax). Every other layer may
depend on `core`, and `core` depends on none of them. The rule is what lets `domain` and the
other pure layers use errors, registries and configuration without pulling an engine in.
"""
