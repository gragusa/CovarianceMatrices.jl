#=========
Requires
=========#

import .GLM
import StatsModels: TableRegressionModel, RegressionModel
import StatsBase: modelmatrix, vcov, stderror

const INNERMOD = Union{GLM.GeneralizedLinearModel, GLM.LinearModel}
const LINMOD = GLM.LinearModel

cache(k, m::TableRegressionModel; kwargs...) = cache(k, m.model)
cache(k, m::T; kwargs...) where T<:INNERMOD = cache(k, modelmatrix(m))

vcov(k::RobustVariance, m::RegressionModel; kwargs...) = vcov(k, m, cache(k, m), Matrix, Nothing; kwargs...)

function vcov(k::RobustVariance, m::RegressionModel, ::Type{T}; kwargs...) where T<:Matrix
    vcov(k, m, cache(k, m), Matrix, Nothing; kwargs...)
end

function vcov(k::RobustVariance, m::RegressionModel, ::Type{T}; kwargs...) where T<:CovarianceMatrix
    vcov(k, m, cache(k, m), T, SVD; kwargs...)
end

function vcov(k::RobustVariance, m::RegressionModel, ::Type{T}; kwargs...) where T<:Factorization
    vcov(k, m, cache(k, m), CovarianceMatrix, T; kwargs...)
end

function vcov(k::RobustVariance, m::RegressionModel, ::Type{T}, ::Type{F}; kwargs...) where T<:CovarianceMatrix where F<:Factorization
    vcov(k, m, cache(k, m), CovarianceMatrix, F; kwargs...)
end

function vcov(k::RobustVariance, m::RegressionModel, cache::AbstractCache, ::Type{T}; kwargs...) where T<:Factorization
    vcov(k, m, cache, CovarianceMatrix, T; kwargs...)
end

function vcov(k::RobustVariance, m::RegressionModel, cache::AbstractCache, ::Type{T}; kwargs...) where T<:CovarianceMatrix
    vcov(k, m, cache, T, SVD; kwargs...)
end

function vcov(k::RobustVariance, m::RegressionModel, cache::AbstractCache, ::Type{T}; kwargs...) where T<:Matrix
    vcov(k, m, cache, Matrix, Nothing; kwargs...)
end

#============
General GLM methods
=============#
modelmatrix(m::TableRegressionModel{T}) where T<:INNERMOD = m.mm.m
modelmatrix(m::T) where T<:INNERMOD = m.pp.X


residuals(m::TableRegressionModel{T}) where T<:INNERMOD = m.model.rr.y .- m.model.rr.mu
residuals(m::GLM.GeneralizedLinearModel) = m.rr.y .- m.rr.mu
residuals(m::TableRegressionModel{T}) where T<:LINMOD = m.model.rr.y .- m.model.rr.mu
residuals(m::GLM.LinearModel) = m.rr.y .- m.rr.mu

modelweights(m::TableRegressionModel{T}) where T<:INNERMOD = m.model.rr.wrkwt
modelweights(m::TableRegressionModel{T}) where T<:LINMOD = m.model.rr.wts
modelweights(m::GLM.GeneralizedLinearModel) = m.rr.wrkwt
modelweights(m::GLM.LinearModel) = m.rr.wts

smplweights(m::TableRegressionModel{T}) where T<:INNERMOD = m.model.rr.wts
# smplweights(m::GLM.GeneralizedLinearModel) = m.rr.wts
# smplweights(m::GLM.LinearModel) = m.rr.wts
smplweights(m::RegressionModel) = m.rr.wts

choleskyfactor(m::TableRegressionModel{T}) where T<:INNERMOD = m.model.pp.chol.UL
choleskyfactor(m::T) where T<:INNERMOD = m.pp.chol.UL

unweighted_nobs(m::TableRegressionModel{T}) where T<:INNERMOD = size(modelmatrix(m), 1)
unweighted_nobs(m::T) where T<:INNERMOD = size(modelmatrix(m), 1)

unweighted_dof_residual(m::TableRegressionModel{T}) where T<:INNERMOD = unweighted_nobs(m) - length(coef(m))
unweighted_dof_residual(m::T) where T<:INNERMOD = unweighted_nobs(m) - length(coef(m))

installxuw!(cache, m::T) where T<:TableRegressionModel = installxuw!(cache, m.model)
function installxuw!(cache, m::T) where T<:INNERMOD
    copyto!(cache.X, modelmatrix(m))
    copyto!(cache.u, residuals(m))
    if !isempty(m.rr.wts)
        broadcast!(*, cache.u, cache.u, smplweights(m))
    end
    nothing
end

function esteq!(cache, m::RegressionModel, k::T) where T<:Union{HC, CRHC}
    broadcast!(*, cache.q, cache.X, cache.u)
    return cache.q
end

function esteq!(cache, m::RegressionModel, k::HAC)
    copyto!(cache.q, modelmatrix(m))
    u = copy(residuals(m))
    if !isempty(smplweights(m))
        broadcast!(*, u, u, smplweights(m))
    end
    broadcast!(*, cache.q, cache.q, u)
end

pseudohessian(m::TableRegressionModel{T}) where T<:INNERMOD = GLM.invchol(m.model.pp)
pseudohessian(m::T) where T<:INNERMOD = GLM.invchol(m.pp)

#==============
HAC GLM Methods
===============#
function vcov(k::T, m, cache, returntype, factortype; demean::Bool=false, dof_adjustment::Bool=true) where T<:HAC
    mf = esteq!(cache, m, k)
    B = pseudohessian(m)
    set_bw_weights!(k, m)
    Ω = covariance(k, mf, cache, Matrix, demean = demean, scale = unweighted_nobs(m))
    cache.V .= B*Ω*B'
    rmul!(cache.V, size(mf, 1)^2)
    scale = dof_adjustment ? unweighted_dof_residual(m) : unweighted_nobs(m)
    finalize(cache, returntype, factortype, k, scale)
end

function set_bw_weights!(k, m::TableRegressionModel{T}) where T<:INNERMOD
    β = coef(m)
    resize!(k.weights, length(β))
    "(Intercept)" ∈ coefnames(m) ? (k.weights .= 1.0; k.weights[1] = 0.0) : k.weights .= 1.0
end

function set_bw_weights!(k, m::T) where T<:INNERMOD
    β = coef(m)
    resize!(k.weights, length(β))
    k.weights .= 1.0
end

#==============
HC GLM Methods
===============#
function vcov(k::T, m, cache, returntype, factortype;
              demean::Bool=false, dof_adjustment=false) where T<:HC
    CovarianceMatrices.installxuw!(cache, m)
    mf = CovarianceMatrices.esteq!(cache, m, k)
    demean!(cache, Val{demean})
    B = CovarianceMatrices.pseudohessian(m)
    CovarianceMatrices.adjfactor!(cache, m, k)
    V = mul!(cache.V, mf', broadcast!(*, cache.X, mf, cache.η))
    V .= B*V*B'
    finalize(cache, returntype, factortype, k, 1)
end

## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----
hatmatrix(cache, m::TableRegressionModel{T}) where T<:INNERMOD = hatmatrix(cache, m.model)
function hatmatrix(cache, m::T) where T<:INNERMOD
      X = cache.X  ## THIS ASSUME THAT X IS WEIGHTED BY SQRT(W)
      if !isempty(modelweights(m))
          X .= X.*sqrt.(modelweights(m))
      end
      cf = choleskyfactor(m)::UpperTriangular
      rdiv!(X, cf)
      sum!(cache.v, X.^2)
 end

  adjfactor!(cache, m::RegressionModel, k::HC0) = cache.η .= one(eltype(cache.u))
  adjfactor!(cache, m::RegressionModel, k::HC1) = cache.η .= unweighted_nobs(m)./unweighted_dof_residual(m)
  adjfactor!(cache, m::RegressionModel, k::HC2) = cache.η .= one(eltype(cache.u))./(one(eltype(cache.u)).-hatmatrix(cache, m))
  adjfactor!(cache, m::RegressionModel, k::HC3) = cache.η .= one(eltype(cache.u))./(one(eltype(cache.u)).-hatmatrix(cache, m)).^2

  function adjfactor!(cache, m::RegressionModel, k::HC4)
      n, p = size(cache.X)
      tone = one(eltype(cache.u))
      h = hatmatrix(cache, m)
      @inbounds for j in eachindex(h)
          delta = min(4, n*h[j]/p)
          cache.η[j] = tone/(tone-h[j])^delta
      end
      cache.η
  end

  function adjfactor!(cache, m::RegressionModel, k::HC4m)
      n, p = size(cache.X)
      tone = one(eltype(cache.u))
      h = hatmatrix(cache, m)
      @inbounds for j in eachindex(h)
          delta = min(tone, n*h[j]/p) + min(1.5, n*h[j]/p)
          cache.η[j] = tone/(tone-h[j])^delta
      end
      cache.η
  end

  function adjfactor!(cache, m::RegressionModel, k::HC5)
      n, p = size(cache.X)
      tone = one(eltype(cache.u))
      h     = hatmatrix(cache, m)
      mx    = max(n*0.7*maximum(h)/p, 4)
      @inbounds for j in eachindex(h)
          alpha =  min(n*h[j]/p, mx)
          cache.η[j] = sqrt(tone/(tone-h[j])^alpha)
      end
      cache.η
  end

#==========
Cluster GLM
=========#

function vcov(k::T, m, cache, returntype, factortype;
              demean::Bool=false, sorted::Bool=false) where T<:CRHC
    B = CovarianceMatrices.pseudohessian(m)
    CovarianceMatrices.installsortedxuw!(cache, m, k, Val{sorted})
    bstarts = (searchsorted(cache.clus, j[2]) for j in enumerate(unique(cache.clus)))
    CovarianceMatrices.adjresid!(k, cache, B, bstarts)
    CovarianceMatrices.esteq!(cache, m, k)
    demean!(cache, Val{demean})
    V = CovarianceMatrices.clusterize!(cache, bstarts)
    df = dof_adjustment(cache, k, bstarts)
    cache.V .= df.*(B*V*B)
    finalize(cache, returntype, factortype, k, 1)
end

stderror(k::RobustVariance, m, args...; kwargs...) = sqrt.(diag(vcov(k, m, args...; kwargs...)))
stderror(cm::CovarianceMatrix) = sqrt.(diag(cm))