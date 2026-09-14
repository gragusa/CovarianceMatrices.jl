module PDMatsExt

using CovarianceMatrices: CovarianceMatrix
using LinearAlgebra: LinearAlgebra, Symmetric, isposdef
using PDMats: PDMats, PDMat

"""
    PDMat(V::CovarianceMatrix)

Convert an estimate to a `PDMat`, giving access to the PDMats interface
(`invquad`, `whiten`, `logdet`, …).

Throws `LinearAlgebra.PosDefException` when the estimate is not positive definite.
Most estimators in this package produce positive definite estimates, but not all do:
the `Truncated` kernel is consistent without being positive semi-definite, the
multiway cluster-robust estimator is an alternating sum that can be indefinite in
finite samples, and a rank-deficient model produces `NaN` entries for the
non-estimable parameters. Use `isposdef(V)` to check first.
"""
PDMats.PDMat(V::CovarianceMatrix) = PDMat(Symmetric(parent(V)))

LinearAlgebra.isposdef(V::CovarianceMatrix) = isposdef(Symmetric(parent(V)))

end
