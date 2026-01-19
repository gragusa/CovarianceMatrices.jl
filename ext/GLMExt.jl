module GLMExt

using CovarianceMatrices, GLM, LinearAlgebra, StatsBase, StatsModels, StatsAPI
using Statistics

const FAM = Union{GLM.Gamma, GLM.Bernoulli, GLM.InverseGaussian}
const CM = CovarianceMatrices

# Define GLM-specific union types to avoid type piracy on abstract RegressionModel
const GLMLinearModel = GLM.LinearModel
const GLMGeneralizedLinearModel = GLM.GeneralizedLinearModel
const GLMLinPredModel = GLM.LinPredModel
const GLMTableModel = StatsModels.TableRegressionModel{<:GLM.LinPredModel}

##=================================================
## numobs: Actual observation count (not sum of weights)
##=================================================

"""
    CM.numobs(m::GLM.LinPredModel) -> Int

Return the actual number of observations (rows) in the GLM model.

This is different from `nobs(m)` for weighted models:
- `nobs(m)`: sum of weights (effective sample size)
- `numobs(m)`: actual row count (for DOF calculations)

Uses internal GLM field `.pp.X` for efficiency.
"""
CM.numobs(m::GLMTableModel) = CM.numobs(m.model)
CM.numobs(m::GLMLinPredModel) = size(modelmatrix(m), 1)

##=================================================
## Dispersion Parameter (GLM-specific)
##=================================================

"""
    _dispersion(m) -> Float64

Compute dispersion parameter for GLM models.

For linear models: always 1.0
For GLMs: depends on family
  - Binomial, Poisson: dispersion = 1.0 (fixed)
  - Gamma, Bernoulli, InverseGaussian: estimated from residuals

Uses internal GLM fields `.rr.wrkwt` and `.rr.wrkresid`.
"""
_dispersion(m::GLMLinearModel) = 1.0
_dispersion(m::GLMGeneralizedLinearModel) = _dispersion(m.rr)

_dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2, T3} = 1.0
_dispersion(rr::GLM.LmResp) = 1.0
function _dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2 <: FAM, T3}
    sum(abs2, rr.wrkwt .* rr.wrkresid) / sum(rr.wrkwt)
end

##=================================================
## bread: Optimized using GLM internals
##=================================================

"""
    CM.bread(m::GLM.LinearModel) -> Matrix

Compute bread matrix (X'X)^(-1) efficiently using LM
Compute bread matrix (X'WX)^(-1) * dispersion for GLM.

Uses internal `GLM.invchol(m.pp)` for efficiency instead of recomputing.
"""

CM.bread(m::GLMTableModel) = CM.bread(m.model)
CM.hessian_objective(m::GLMTableModel) = CM.hessian_objective(m.model)

CM.bread(m::GLMLinearModel) = GLM.invchol(m.pp)
CM.bread(m::GLMGeneralizedLinearModel) = GLM.invchol(m.pp) .* _dispersion(m)

CM.hessian_objective(m::GLMLinearModel) = Matrix(m.pp)
CM.hessian_objective(m::GLMGeneralizedLinearModel) = Matrix(m.pp) ./ _dispersion(m)

##=================================================
## residuals: Override for working residuals
##=================================================

function CM._residuals(m::GLMLinearModel)
    u = GLM.residuals(m)
    w = m.rr.wts
    isempty(w) ? u : u .* w
end

function CM._residuals(m::GLMGeneralizedLinearModel)
    u = m.rr.wrkresid
    w = m.rr.wrkwt
    (u .* w) ./ _dispersion(m)
end

CM._residuals(m::GLMTableModel) = CM._residuals(m.model)
##=================================================
## mask: Rank deficiency detection (GLM-specific)
##=================================================

"""
    CM.mask(m) -> Vector{Bool}

Return boolean mask indicating which parameters are estimable.

For rank-deficient designs, uses GLM's pivoted Cholesky decomposition
to determine which parameters are aliased.

Uses internal fields `.pp.chol.rank` and `.pp.chol.p`.
"""
CM.mask(m::GLMLinearModel) = CM.mask(m.pp)
CM.mask(m::GLMGeneralizedLinearModel) = CM.mask(m.pp)
CM.mask(m::GLMTableModel) = CM.mask(m.model)

function CM.mask(pp::GLM.DensePredChol{F, C}) where {F, C <: LinearAlgebra.CholeskyPivoted}
    k = size(pp.X, 2)
    rnk = pp.chol.rank
    p = pp.chol.p
    if rnk == k
        return ones(Bool, k)
    else
        mask = zeros(Bool, k)
        mask[p[1:rnk]] .= true
        return mask
    end
end

function CM.mask(pp::GLM.DensePredChol{F, C}) where {F, C <: LinearAlgebra.Cholesky}
    k = size(pp.X, 2)
    return ones(Bool, k)
end

##=================================================
## momentmatrix: Optimized using GLM internals
##=================================================

"""
    CM.momentmatrix(m::GLM.LinearModel) -> Matrix

Compute moment matrix for linear model efficiently.

Returns X .* residuals .* weights (if weighted).
Uses GLM's internal scratch buffer `.pp.scratchm1` for efficiency.
"""
CM.momentmatrix(m::GLMTableModel) = CM.momentmatrix(m.model)

function momentmatrix!(M::AbstractMatrix, m::GLMGeneralizedLinearModel)
    X = modelmatrix(m)
    wrkwt = m.rr.wrkwt
    wrkrs = m.rr.wrkresid
    d = _dispersion(m)
    @. M = (X * wrkwt * wrkrs) / d
    return M
end

function momentmatrix!(M::AbstractMatrix, m::GLMLinearModel)
    X = modelmatrix(m)
    wrkresid = CM._residuals(m)
    @. M = X * wrkresid
    return M
end

function CM.momentmatrix(m::GLMLinearModel)
    M = similar(modelmatrix(m))
    return momentmatrix!(M, m)
end

function CM.momentmatrix(m::GLMGeneralizedLinearModel)
    M = similar(modelmatrix(m))
    return momentmatrix!(M, m)
end

##=================================================
## leverage: Optimized using GLM internals
##=================================================

"""
    CM.leverage(m::GLM.LinearModel) -> Vector

Compute hat matrix diagonals efficiently using GLM's Cholesky decomposition.

For unweighted: h = diag(X * (X'X)^(-1) * X')
For weighted: h = diag(X * (X'WX)^(-1) * X' * W)
For GLMs: h = diag(X * (X'WX)^(-1) * X' * W) where W are working weights.

Uses internal fields `.pp.chol` for efficiency and handles rank deficiency
via pivoted Cholesky.
"""
function CM.leverage(r::GLMLinearModel)
    X = modelmatrix(r)
    scratch = copy(X)
    if !isempty(r.rr.wts)
        scratch .*= sqrt.(r.rr.wts)
    end
    return _leverage(r.pp, scratch)
end

function CM.leverage(r::GLMGeneralizedLinearModel)
    X = modelmatrix(r) .* sqrt.(r.rr.wrkwt)
    return _leverage(r.pp, X)
end

CM.leverage(m::GLMTableModel) = CM.leverage(m.model)

function _leverage(
        pp::GLM.DensePredChol{F, C},
        X
) where {F, C <: LinearAlgebra.CholeskyPivoted}
    ch = pp.chol
    rnk = rank(ch)
    p = ch.p
    idx = invperm(p)[1:rnk]
    return vec(sum(abs2, view(X, :, 1:rnk) / view(ch.U, 1:rnk, idx), dims = 2))
end

function _leverage(pp::GLM.DensePredChol{F, C}, X) where {F, C <: LinearAlgebra.Cholesky}
    return vec(sum(abs2, X / pp.chol.U, dims = 2))
end

##=================================================
## weights: Observation weights (not working weights)
##=================================================

function StatsBase.weights(m::GLMLinPredModel)
    m.rr.wts
end

StatsBase.weights(m::GLMTableModel) = StatsBase.weights(m.model)

##=================================================
## aVar for CR estimators (uses crmomentmatrix)
##=================================================

# """
#     CM.aVar(k::CR, m::GLMLinPredModel; kwargs...)

# Compute asymptotic variance for cluster-robust estimators on GLM models.

# Uses cluster-adjusted residuals via residual_adjustment and forms
# moment matrix via crmomentmatrix.
# """
# function CM.aVar(
#         k::K,
#         m::Union{GLMLinPredModel, GLMTableModel};
#         demean = false,
#         prewhite = false,
#         scale = true,
#         kwargs...
# ) where {K <: CM.CR}
#     # Get cluster-adjusted residuals
#     u = CM.residual_adjustment(k, m)
#     mm = momentmatrix(m)

#     # Handle rank deficiency
#     midx = CM.mask(m)
#     Σ = if sum(midx) == size(mm, 2)
#         CM.aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)
#     else
#         CM.aVar(k, mm[:, midx]; demean = demean, prewhite = prewhite, scale = scale)
#     end

#     return Σ
# end

end # module
