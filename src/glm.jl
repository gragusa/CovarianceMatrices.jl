#=========
Requires
=========#

using GLM
using DataFrames
using StatsBase
using StatsModels
#import StatsBase: residuals

const INNERMOD = Union{GLM.GeneralizedLinearModel, GLM.LinearModel}
#const UNIONC = Union{Type{Nothing}, Type{LinearAlgebra.Cholesky}, Type{PositiveFactorizations.Positive}}
#============
General GLM methods
=============#
StatsModels.modelmatrix(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD = m.mm.m
StatsModels.modelmatrix(m::T) where T<:INNERMOD = m.pp.X


residuals(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD = m.model.rr.wrkresid
residuals(m::T) where T<:INNERMOD = m.rr.y .- m.rr.mu

modelweights(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD = m.model.rr.wrkwt
modelweights(m::GLM.GeneralizedLinearModel) = m.rr.wrkwt
modelweights(m::GLM.LinearModel) = m.rr.wts

smplweights(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD = m.model.rr.wts
smplweights(m::GLM.GeneralizedLinearModel) = m.rr.wts
smplweights(m::GLM.LinearModel) = m.rr.wts


choleskyfactor(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD = m.model.pp.chol.UL
choleskyfactor(m::T) where T<:INNERMOD = m.pp.chol.UL

unweighted_nobs(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD = size(modelmatrix(m), 1)
unweighted_nobs(m::T) where T<:INNERMOD = size(modelmatrix(m), 1)

unweighted_dof_residual(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD = unweighted_nobs(m) - length(coef(m))
unweighted_dof_residual(m::T) where T<:INNERMOD = unweighted_nobs(m) - length(coef(m))

function installxuw!(cache, m::T) where T<:INNERMOD
    copyto!(cache.X, copy(modelmatrix(m)))
    copyto!(cache.u, copy(residuals(m)))
    if !isempty(m.rr.wts)
        broadcast!(*, cache.u, cache.u, smplweights(m))
    end
end

function esteq!(cache, m::StatsModels.RegressionModel, k::T) where T<:Union{HC, CRHC}
    broadcast!(*, cache.q, cache.X, cache.u)
    return cache.q
end

function esteq!(cache, m::RegressionModel, k::HAC)
    X = copy(modelmatrix(m))
    u = copy(residuals(m))
    if !isempty(smplweights(m))
        broadcast!(*, u, u, smplweights(m))
    end
    broadcast!(*, X, X, u)
end

pseudohessian(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD = GLM.invchol(m.model.pp)
pseudohessian(m::T) where T<:INNERMOD = GLM.invchol(m.pp)

#==============
HAC GLM Methods
===============#

function HACCache(m::StatsModels.DataFrameRegressionModel{T}; prewhiten = false) where T<:INNERMOD
    HACCache(m.model, prewhiten = prewhiten)
end

function HACCache(m::T; prewhiten = false) where T<:INNERMOD
    HACCache(copy(modelmatrix(m)), prewhiten = prewhiten)
end

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel{T}, k::HAC, args...; kwargs...) where T<:INNERMOD
    set_bw_weights!(k, m)
    vcov(m.model, k, args...; kwargs...)
end

function StatsBase.vcov(m::T, k::K, returntype::Type{T1} = CovarianceMatrix, factortype::Type{T2} = Cholesky; kwargs...) where {T<:INNERMOD, K<:HAC, T1<:Union{CovarianceMatrix, Matrix}, T2<:Union{Nothing, Factorization}}
    cache = HACCache(m, prewhiten = k.prewhiten)
    vcov(m, k, cache, returntype, factortype; kwargs...)
end

function StatsBase.vcov(m::T, k::K, cache::HACCache, returntype::Type{T1} = CovarianceMatrix, factortype::Type{T2} = Cholesky; demean::Bool = false, dof_adjustment::Bool = true) where {T<:INNERMOD, K<:HAC, T1<:Union{CovarianceMatrix, Matrix}, T2<:Union{Nothing, Factorization}}
    mf = esteq!(cache, m, k)
    br = pseudohessian(m)
    set_bw_weights!(k, m)
    Ω = covariance(mf, k, cache, Matrix, demean = demean, scale = 1)
    V = br*Ω*br'
    rmul!(V, size(mf, 1)^2)
    scale = dof_adjustment ? unweighted_dof_residual(m) : unweighted_nobs(m)
    finalize(V, T2, T1, k, scale)
end

# setcholesky!(cache, cholesky::Type{Nothing}, V) = nothing

# function setcholesky!(cache, cholesky::Type{Cholesky}, V)
#     chol = LinearAlgebra.cholesky(Symmetric(V))
#     copyto!(cache.chol.UL.data, chol.UL.data)
#     copyto!(cache.chol.U.data, chol.U.data)
#     copyto!(cache.chol.L.data, chol.L.data)
# end

# function setcholesky!(cache, cholesky::Type{PositiveFactorizations.Positive}, V)
#     chol = LinearAlgebra.cholesky(Positive, V)
#     copyto!(cache.chol.UL.data, chol.UL.data)
#     copyto!(cache.chol.U.data, chol.U.data)
#     copyto!(cache.chol.L.data, chol.L.data)
# end

function set_bw_weights!(k::HAC, m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD
    if isempty(k.weights)
        for j in eachindex(coef(m))
            push!(k.weights, 1.0)
        end
    end
    if length(k.weights) == length(coef(m))
        "(Intercept)" ∈ coefnames(m) ? (k.weights .= 1.0; k.weights[1] = 0.0) : k.weights .= 1.0
    else
        error("Bandwidth weights have wrong dimension for the problem")
    end
end

function set_bw_weights!(k::HAC, m::T) where T<:INNERMOD
    if isempty(k.weights)
        for j in eachindex(coef(m))
            push!(k.weights, 1.0)
        end
    end
end

#==============
HC GLM Methods
===============#
function HCCache(m::StatsModels.DataFrameRegressionModel{T}) where T<:INNERMOD
    HCCache(m.model)
end

function HCCache(m::T) where {T<:INNERMOD}
    n, p = unweighted_nobs(m), length(coef(m))
    HCCache(similar(Array{eltype(modelmatrix(m)), 2}(undef, n, p)))
end

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel{T}, 
                        k::K, returntype::Type{T1} = CovarianceMatrix,
                        factortype::Type{T2} = Cholesky; 
                        kwargs...) where {T<:INNERMOD,
                                          K<:HC,
                                          T1<:Union{CovarianceMatrix, Matrix},
                                          T2<:Union{Nothing, Factorization}}
    vcov(m.model, k, HCCache(m.model), returntype, factortype; kwargs...)
end

function StatsBase.vcov(m::T, 
    k::K, returntype::Type{T1} = CovarianceMatrix,
    factortype::Type{T2} = Cholesky; 
    kwargs...) where {T<:INNERMOD,
                      K<:HC,
                      T1<:Union{CovarianceMatrix, Matrix},
                      T2<:Union{Nothing, Factorization}}
    vcov(m, k, HCCache(m), returntype, factortype; kwargs...)
end



# function StatsBase.vcov(m::T,
#                         k::K,
#                         cache::HCCache,
#                         returntype::Type{T1} = CovarianceMatrix,
#                         factortype::Type{T2} = Cholesky; kwargs...) where {T<:INNERMOD,
#                                                                            K<:HC,
#                                                                            T1<:Union{CovarianceMatrix, Matrix},
#                                                                            T2<:Union{Nothing, Factorization}}

#     vcov(m, k, HCCache(m), returntype, factortype; kwargs...)
# end

function StatsBase.vcov(m::T,
                        k::K,
                        cache::HCCache,
                        returntype::Type{T1} = CovarianceMatrix,
                        factortype::Type{T2} = Cholesky;
                        demean::Bool = false) where {T<:INNERMOD,
                                                            K<:HC,
                                                            T1<:Union{CovarianceMatrix, Matrix},
                                                            T2<:Union{Nothing, Factorization}}
    CovarianceMatrices.installxuw!(cache, m)
    mf = CovarianceMatrices.esteq!(cache, m, k)
    br = CovarianceMatrices.pseudohessian(m)
    CovarianceMatrices.adjfactor!(cache, m, k)
    mul!(cache.x, mf', broadcast!(*, cache.X, mf, cache.η))
    V = br*cache.x*br'    
    finalize(V, T2, T1, k, 1.0)
end

## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----

function hatmatrix(cache, m::T) where T<:INNERMOD
      z = cache.X  ## THIS ASSUME THAT X IS WEIGHTED BY SQRT(W)
      if !isempty(modelweights(m))
          z .= z.*sqrt.(modelweights(m))
      end
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
          cache.η[j] = sqrt(tone/(tone-h[j])^alpha)
      end
      cache.η
  end

#==========
Cluster GLM
=========#

function CRHCCache(m::StatsModels.DataFrameRegressionModel{T}, cl::AbstractArray{F}) where {T<:INNERMOD, F}
    CRHCCache(m.model, cl, returntype=returntype)
end

function CRHCCache(m::T, cl::AbstractVector{F}) where {T<:INNERMOD, F}
    n, p = unweighted_nobs(m), length(coef(m))
    CRHCCache(similar(modelmatrix(m)), cl)
end

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel{T}, 
                        k::K, 
                        returntype::Type{T1} = CovarianceMatrix,
                        factortype::Type{T2} = Cholesky; kwargs...) where {T<:INNERMOD,
                                                                           K<:CRHC,
                                                                           T1<:Union{CovarianceMatrix, Matrix},
                                                                           T2<:Union{Nothing, Factorization}}
    vcov(m.model, k, CRHCCache(m.model, k.cl), returntype, factortype; kwargs...)
end

function StatsBase.vcov(m::T, 
                        k::K, 
                        returntype::Type{T1} = CovarianceMatrix, 
                        factortype::Type{T2} = Cholesky; 
                        kwargs...) where {T<:INNERMOD,
                                          K<:CRHC,
                                          T1<:Union{CovarianceMatrix, Matrix},
                                          T2<:Union{Nothing, Factorization}}
    vcov(m, k, CRHCCache(m, k.cl), returntype, factortype; kwargs...)
end


function StatsBase.vcov(m::T, 
                        k::CRHC, 
                        cache::CRHCCache,
                        returntype::Type{T1} = CovarianceMatrix, 
                        factortype::Type{T2} = Cholesky; 
                        sorted::Bool = false) where {T<:INNERMOD,
                                                     K<:HC,
                                                     T1<:Union{CovarianceMatrix, Matrix},
                                                     T2<:Union{Nothing, Factorization}}
    B = CovarianceMatrices.pseudohessian(m)
    CovarianceMatrices.installsortedxuw!(cache, m, k, Val{sorted})
    bstarts = (searchsorted(cache.clus, j[2]) for j in enumerate(unique(cache.clus)))
    CovarianceMatrices.adjresid!(k, cache, B, bstarts)
    CovarianceMatrices.esteq!(cache, m, k)
    CovarianceMatrices.clusterize!(cache, bstarts)
    V = dof_adjustment(cache, k, bstarts).*(B*cache.M*B)
    finalize(V, T2, T1, k, 1.0)
end

StatsBase.stderror(m::StatsModels.DataFrameRegressionModel{T}, k::RobustVariance; kwargs...) where T<:INNERMOD = sqrt.(diag(vcov(m, k, Matrix; kwargs...)))

