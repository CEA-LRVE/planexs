"""The custom BoTorch acquisition layer implementing the alpha dial (FR5) and
enforcing `NonlinearConstraint` (FR17): `ExplorationExploitationAcquisition` and
its multi-objective variant, the `AlphaAcquisitionStrategy` that optimizes them
(batches built sequentially, FR10), and the `toBotorchNonlinearConstraints`
compiler.
"""
