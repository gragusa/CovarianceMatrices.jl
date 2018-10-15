#=========
Requires
=========#

using .GLM
using .DataFrames
using .StatsBase
using .StatsModels
import .StatsBase: residuals

#============
General GLM methods
=============#


StatsModels.modelmatrix(m::T) where T<:StatsModels.DataFrameRegressionModel = m.mm.m
StatsModels.modelmatrix(m::T) where T<:GLM.LinearModel = m.pp.X

function modelmatrix!(cache, m::T, weighted::Type{Val{false}}) where T<:Union{StatsModels.DataFrameRegressionModel, GLM.LinearModel}
    return cache.X
end

residuals(m::StatsModels.DataFrameRegressionModel{T}) where T<:GLM.GeneralizedLinearModel = m.model.rr.wrkresid
getweights(m::StatsModels.DataFrameRegressionModel) = m.model.rr.wts
getweights(m::GLM.LinearModel) = m.rr.wts

function installxuw!(cache, m::T) where T<:Union{StatsModels.DataFrameRegressionModel, GLM.LinearModel}
    copyto!(cache.X, modelmatrix(m))
    copyto!(cache.u, residuals(m))
    if !isempty(getweights(m))
        cache.w .= sqrt.(getweights(m))
        broadcast!(*, cache.X, cache.X, cache.w)
        broadcast!(*, cache.u, cache.u, cache.w)
    end
end

choleskyfactor(m::StatsModels.DataFrameRegressionModel) = m.model.pp.chol.UL
choleskyfactor(m::GLM.LinearModel) = m.pp.chol.UL

unweighted_nobs(m::T) where T<:Union{StatsModels.DataFrameRegressionModel, GLM.LinearModel} = size(modelmatrix(m), 1)
unweighted_dof_residual(m::T) where T<:Union{StatsModels.DataFrameRegressionModel, GLM.LinearModel} = size(modelmatrix(m), 1) - length(coef(m))

## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----

function esteq!(cache, m::StatsModels.RegressionModel)
    broadcast!(*, cache.q, cache.X, cache.u)
    return cache.q
end

pseudohessian(cache, m::StatsModels.DataFrameRegressionModel) = GLM.invchol(m.model.pp)
pseudohessian(m::StatsModels.DataFrameRegressionModel) = GLM.invchol(m.model.pp)

function pseudohessian(cache, m::GLM.LinearModel)
    X = cache.X
    mul!(cache.x, X', X)
    inv(cholesky!(cache.x))
end


#==============
HAC GLM Methods
===============#

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::HAC, cache::CovarianceMatrices.HACCache; demean = Val{false}, dof_adjustment::Bool = true)
    mf = esteq_hac!(cache, m)
    br = pseudohessian(m)
    n = unweighted_nobs(m)
    V = variance(mf, k, cache, demean = demean, dof_adjustment = dof_adjustment)
    V = br*V*br'
    dof_adjustment ? rmul!(V, unweighted_dof_residual(m)) : rmul!(V, unweighted_nobs(m))
end

function StatsBase.vcov(m::GLM.LinearModel, k::HAC, cache::CovarianceMatrices.HACCache; demean = Val{false}, dof_adjustment = dof_adjustment)
    mf = esteq_hac!(cache, m)
    br = pseudohessian(m)
    V = variance(mf, k, cache, demean = demean, kwargs...).*size(cache.X_demean,1)
    V = br*V*br'
    dof_adjustment ? rmul!(V, unweighted_dof_residual(m)) : rmul!(V, unweighted_nobs(m))
end

function StatsBase.vcov(m::GLM.LinearModel, k::HAC; kwargs...)
    cache = HACCache(modelmatrix(m), k)
    vcov(m, k, cache, demean=demean, kwargs...)
end

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::HAC; kwargs...)
    cache = HACCache(modelmatrix(m), k)
    vcov(m, k, cache, kwargs...)
end


function esteq_hac!(cache, m::RegressionModel)
    X = copy(modelmatrix(m))
    u = copy(residuals(m))
    if !isempty(getweights(m))
        broadcast!(*, X, X, sqrt.(getweights(m)))
        broadcast!(*, u, u, sqrt.(getweights(m)))
    end
    broadcast!(*, X, X, u)
    return X
end

#==============
HC GLM Methods
===============#

function HCCache(m::StatsModels.DataFrameRegressionModel; returntype::Type{T1} = Float64) where T1
    s = size(ModelMatrix(m.mf).m)
    HCCache(similar(Array{T1, 2}(undef, s...)); returntype = returntype)
end

function HCCache(m::GLM.LinearModel; returntype::Type{T1} = Float64) where T1
    s = size(m.pp.X)
    HCCache(similar(Array{T1, 2}(undef, s...)); returntype = T1)
end


StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::HC; kwargs...) = vcov(m, k, HCCache(m); kwargs...)
StatsBase.vcov(m::GLM.LinearModel, k::HC, kwargs...) = vcov(m, k, HCCache(m); kwargs...)

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::HC, cache::CovarianceMatrices.HCCache)
    installxuw!(cache, m)
    mf = esteq!(cache, m)
    br = pseudohessian(cache, m)
    adjfactor!(cache, m, k)
    mul!(cache.x, mf', broadcast!(*, cache.X, mf, cache.η))
    br*cache.x*br'
end

function StatsBase.vcov(m::GLM.LinearModel, k::HC, cache::CovarianceMatrices.HCCache; kwargs...)
    installxuw!(cache, m)
    mf = esteq!(cache, m)
    br = pseudohessian(cache, m)
    adjfactor!(cache, m, k)
    mul!(cache.x, mf', broadcast!(*, cache.X, mf, cache.η))
    br*cache.x*br'
end

StatsBase.stderror(m::StatsModels.DataFrameRegressionModel, k::HC, args...; kwargs...) = sqrt.(diag(vcov(m, k, args...)))
StatsBase.stderror(m::GLM.LinearModel, k::HC, args...; kwargs...) = sqrt.(diag(vcov(m, k, args...; kwrags...)))

## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----

function hatmatrix(cache, m::T) where T<:Union{StatsModels.DataFrameRegressionModel, GLM.LinearModel}
      z = cache.X  ## THIS ASSUME THAT X IS WEIGHTED BY SQRT(W)
      cf = choleskyfactor(m)::UpperTriangular
      rdiv!(z, cf)
      sum!(cache.v, z.^2)
 end

  adjfactor!(cache, m::StatsModels.RegressionModel, k::HC0) = cache.η .= one(eltype(cache.u))
  adjfactor!(cache, m::StatsModels.RegressionModel, k::HC1) = cache.η .= unweighted_nobs(m)./unweighted_dof_residual(m)
  adjfactor!(cache, m::StatsModels.RegressionModel, k::HC2) = cache.η .= one(eltype(cache.u))./(one(eltype(cache.u)).-hatmatrix(cache, m))
  adjfactor!(cache, m::StatsModels.RegressionModel, k::HC3) = cache.η .= one(eltype(cache.u))./(one(eltype(cache.u)).-hatmatrix(cache, m)).^2

  function adjfactor!(cache, m::StatsModels.RegressionModel, k::HC4)
      n, p = size(cache.X)
      tone = one(eltype(cache.u))
      h = hatmatrix(cache, m)
      @inbounds for j in eachindex(h)
          delta = min(4, n*h[j]/p)
          cache.η[j] = tone/(tone-h[j])^delta
      end
      cache.η
  end

  function adjfactor!(cache, m::StatsModels.RegressionModel, k::HC4m)
      n, p = size(cache.X)
      tone = one(eltype(cache.u))
      h = hatmatrix(cache, m)
      @inbounds for j in eachindex(h)
          delta = min(tone, n*h[j]/p) + min(1.5, n*h[j]/p)
          cache.η[j] = tone/(tone-h[j])^delta
      end
      cache.η
  end

  function adjfactor!(cache, m::StatsModels.RegressionModel, k::HC5)
      n, p = size(cache.X)
      tone = one(eltype(cache.u))
      h     = hatmatrix(cache, m)
      mx    = max(n*0.7*maximum(h)/p, 4)
      @inbounds for j in eachindex(h)
          alpha =  min(n*h[j]/p, mx)
          cache.η[j] = tone/(tone-h[j])^alpha
      end
      cache.η
  end

#=========
Cluster GLM
=========#

function CRHCCache(m::StatsModels.DataFrameRegressionModel; returntype::Type{T1} = Float64) where T1
    s = size(ModelMatrix(m.mf).m)
    CRHCCache(similar(Array{T1, 2}(undef, s...)); returntype = returntype)
end

function CRHCCache(m::GLM.LinearModel; returntype::Type{T1} = Float64) where T1
    s = size(m.pp.X)
    CRHCCache(similar(Array{T1, 2}(undef, s...)); returntype = T1)
end

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::CRHC, cache; sorted::Bool = false)
    B = CovarianceMatrices.pseudohessian(cache, m)
    CovarianceMatrices.installsortedxuw!(cache, m, k, Val{sorted})
    bstarts = (searchsorted(cache.clus, j[2]) for j in enumerate(unique(cache.clus)))
    CovarianceMatrices.adjresid!(k, cache, B, bstarts)
    CovarianceMatrices.esteq!(cache, m)
    CovarianceMatrices.clusterize!(cache, bstarts)
    return B*cache.M*B
end

function StatsBase.vcov(m::GLM.LinearModel, k::CRHC, cache; sorted::Bool = false)
    B = CovarianceMatrices.pseudohessian(cache, m)
    CovarianceMatrices.installsortedxuw!(cache, m, k, Val{sorted})
    bstarts = (searchsorted(cache.clus, j[2]) for j in enumerate(unique(cache.clus)))
    CovarianceMatrices.adjresid!(k, cache, B, bstarts)
    CovarianceMatrices.esteq!(cache, m)
    CovarianceMatrices.clusterize!(cache, bstarts)
    return B*cache.M*B
end


export vcov, stderror
