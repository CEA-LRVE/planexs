"""The custom BoTorch acquisition layer implementing the alpha dial (FR5) and
enforcing `NonlinearConstraint` (FR17): `ExplorationExploitationAcquisition` and
its multi-objective variant, and the `AlphaAcquisitionStrategy` that optimizes them
(batches built sequentially, FR10).

The constraints it enforces are rewritten on the encoded space by the `encoding` layer
(`EncodedConstraints`, `toBotorchNonlinearConstraints`, `sampleFeasibleEncoded`).
"""
