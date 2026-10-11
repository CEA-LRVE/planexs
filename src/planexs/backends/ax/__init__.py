"""The Ax-backed `OptimizationBackend` implementation: search space and
optimization-config mapping, trial bookkeeping, the default (no-manual-
tuning) generation strategy, sensitivity analysis, prediction, Pareto-
frontier/best-trial lookup, and JSON persistence, all delegated to Ax's own
`ax.api.client.Client`.
"""
