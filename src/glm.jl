#=========
Requires
=========#
import .GLM
import StatsModels: TableRegressionModel, RegressionModel
import StatsBase: modelmatrix, vcov, stderror

const INNERMOD = Union{GLM.GeneralizedLinearModel, GLM.LinearModel}
const GLMMOD = Union{INNERMOD,
                     TableRegressionModel{GLM.GeneralizedLinearModel},
                     TableRegressionModel{GLM.LinearModel}}
#============
General GLM methods
=============#

# TODO: Find a good name for it
# pseudolikhess
# likhess
# bread
# ....
pseudohessian(m::TableRegressionModel{T}) where T<:INNERMOD = GLM.invchol(m.model.pp).*dispersion(m.model.rr)
pseudohessian(m::T) where T<:INNERMOD = GLM.invchol(m.pp).*dispersion(m.rr)

chol(m::TableRegressionModel{T}) where T<:INNERMOD = chol(m.model)
chol(m::T) where T<:INNERMOD = m.pp.chol

modmatrix(m::TableRegressionModel{T}) where T<:INNERMOD = modmatrix(m.model)
modmatrix(m::T) where T<:GLM.GeneralizedLinearModel = sqrt.(m.rr.wrkwt).*modelmatrix(m)
function modmatrix(m::T) where T<:GLM.LinearModel
    X = modelmatrix(m)
    if !isempty(m.rr.wts)
        sqrt.(m.rr.wts).*X
    else
        copy(X)
    end
end

numobs(m::TableRegressionModel) = length(m.model.rr.y)
numobs(m::INNERMOD) = length(m.rr.y)
dof_resid(m::TableRegressionModel) = numobs(m) - length(coef(m))
dof_resid(m::INNERMOD) = numobs(m) - length(coef(m))

StatsModels.hasintercept(m::TableRegressionModel) = "(Intercept)" ∈ coefnames(m)
interceptindex(m::INNERMOD) = findfirst(map(x->allequal(x), eachcol(modmatrix(m))))

dispersion(m::TableRegressionModel{T}) where T<:GLM.GeneralizedLinearModel = dispersion(m.model.rr)
dispersion(m::TableRegressionModel{T}) where T<:GLM.LinearModel = 1
dispersion(m::GLM.GeneralizedLinearModel) = dispersion(m.rr)
dispersion(m::GLM.LinearModel) = 1
dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2, T3} = 1
dispersion(rr::GLM.LmResp) = 1
function dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2<:Union{GLM.Gamma, GLM.Bernoulli, GLM.InverseGaussian}, T3}
    sum(abs2, rr.wrkwt.*rr.wrkresid)/sum(rr.wrkwt)
end

function StatsModels.hasintercept(m::INNERMOD)
    hasint = findfirst(map(x->allequal(x), eachcol(modmatrix(m))))
    hasint === nothing ? false : true
end

resid(m::TableRegressionModel{T}) where T<:INNERMOD = resid(m.model)
function resid(m::T) where T<:GLM.LinearModel
    if !isempty(m.rr.wts)
        sqrt.(m.rr.wts).*residuals(m)
    else
        copy(residuals(m))
    end
end
function resid(m::T) where T<:GLM.GeneralizedLinearModel
    sqrt.(m.rr.wrkwt).*m.rr.wrkresid
end

momentmatrix(m::TableRegressionModel{T}) where T<:INNERMOD = momentmatrix(m.model)
momentmatrix(m::INNERMOD) = (modmatrix(m).*resid(m))./dispersion(m)

# TODO: move to the interface file
hasresiduals(m::INNERMOD) = true
hasmodelmatrix(m::TableRegressionModel{T}) where T<:INNERMOD = true
#================
Caching mechanism
=================#

#==============
HAC GLM Methods
===============#
function set_bw_weights!(k, m::TableRegressionModel{T}) where T<:INNERMOD
    β = coef(m)
    resize!(k.weights, length(β))
    "(Intercept)" ∈ coefnames(m) ? (k.weights .= 1.0; k.weights[1] = 0.0) : k.weights .= 1.0
end
function set_bw_weights!(k, m::T) where T<:INNERMOD
    cf = coef(m)
    resize!(k.weights, length(cf))
    fill!(k.weights, 1)
    i = interceptindex(m)
    i !== nothing && (k.weights[i] = 0)
end
function __vcov(k, m, rt, ft, pre, scale)
    B  = pseudohessian(m)
    mm = momentmatrix(m)
    set_bw_weights!(k, m)
    A = __covariance(k, mm, rt, ft, pre, 1)
    V = B*A*B
    return finalize(k, V, rt, ft, inv(scale))
end

function vcov(k::T, m; returntype=Matrix, factortype=Cholesky, prewhite=false,
              dof_adjustment::Bool=true) where T<:HAC
    adj = dof_resid(m)/numobs(m)
    scale = dof_adjustment ? adj : one(adj)
    return __vcov(k, m, returntype, factortype, prewhite, scale)
end

#==============
HC GLM Methods
===============#
hatmatrix(m::TableRegressionModel{T}, x) where T<:INNERMOD = hatmatrix(m.model, x)
function hatmatrix(m::T, x) where T<:INNERMOD
    cf = m.pp.chol.UL::UpperTriangular
    rdiv!(x, cf)
    return sum(x.^2, dims = 2)
 end

adjfactor(k::HC0, m::RegressionModel, x) = 1
adjfactor(k::HC1, m::RegressionModel, x) = numobs(m)./dof_resid(m)
adjfactor(k::HC2, m::RegressionModel, x) = 1.0 ./(1.0 .- hatmatrix(m, x))
adjfactor(k::HC3, m::RegressionModel, x) = 1.0 ./(1.0 .- hatmatrix(m, x)).^2

function adjfactor(k::HC4, m::RegressionModel, x)
    n, p = size(x)
    tone = one(eltype(x))
    h = hatmatrix(m, x)
    #η = similar(h)
    @inbounds for j in eachindex(h)
        delta = min(4, n*h[j]/p)
        h[j] = tone/(tone-h[j])^delta
    end
    return h
end

function adjfactor(k::HC4m, m::RegressionModel, x)
    n, p = size(x)
    tone = one(eltype(x))
    h = hatmatrix(m, x)
    @inbounds for j in eachindex(h)
        delta = min(tone, n*h[j]/p) + min(1.5, n*h[j]/p)
        h[j] = tone/(tone-h[j])^delta
    end
    return h
end

function adjfactor(k::HC5, m::RegressionModel, x)
    n, p = size(x)
    tone = one(eltype(x))
    h = hatmatrix(m, x)
    mx = max(n*0.7*maximum(h)/p, 4)
    @inbounds for j in eachindex(h)
        alpha = min(n*h[j]/p, mx)
        h[j] = tone/sqrt((tone-h[j])^alpha)
    end
    return h
end

function vcov(k::HC, m::RegressionModel; returntype::UnionAll=Matrix, factortype::UnionAll=Cholesky)
    B  = pseudohessian(m)
    mm = momentmatrix(m)
    adj = adjfactor(k, m, modmatrix(m))
    mm .= length(adj) > 1 ? mm.*sqrt.(adj) : mm
    scale = length(adj) > 1 ? 1 : adj
    A = covariance(k, mm; returntype = returntype, factortype = Cholesky, demean = false, scale = scale)
    V = B*A*B
    return finalize(k, V, returntype, factortype)
end

# #==========
# Cluster GLM
# =========#

# CRHCCache(m::TableRegressionModel{T}) where T = CRHCCache(m.model)
function install_cache(k::CRHC, m::RegressionModel)
    X = modmatrix(m)
    res = resid(m)
    f = categorize(k.cl)
    (X, res), sf = bysort((X, res), f)
    ci = clusters_intervals(sf)
    p = size(X, 2)
    cf = chol(m)
    Shat = Matrix{eltype(res)}(undef,p,p)
    return CRHCCache(similar(X), X, res, similar(res, (0,0)), cf, Shat, ci, sf)
end

function vcov(k::CRHC, m::RegressionModel; returntype = Matrix, factortype = Cholesky, dof_adjustment::Union{Nothing, Real} = nothing)
    cache = install_cache(k, m)
    df = dof_adjustment === nothing ? float(dofadjustment(k, cache)) : float(dof_adjustment)
    return __vcov(k, m, cache, returntype, factortype, df)
end

function __vcov(k::CRHC, m::RegressionModel, cache::CRHCCache, rt, ft, df)
    B = pseudohessian(m)
    res = adjust_resid!(k, cache)
    cache.momentmatrix .= cache.modelmatrix.*res
    Shat = clusterize!(cache)
    finalize(k, Symmetric(B*Shat*B), rt, ft, df)
end

## Standard errors
stderror(k::RobustVariance, m::RegressionModel; kwargs...) = sqrt.(diag(vcov(k, m; kwargs...)))

## Optimal bandwidth
function optimal_bandwidth(k::HAC{Optimal{T}}, m::TableRegressionModel{F}; kwargs...) where {T<:OptimalBandwidth, F<:INNERMOD}
    optimal_bandwidth(k, m.model; kwargs...)
end

function optimal_bandwidth(k::HAC{Optimal{T}}, m::F; prewhite=false) where {T<:OptimalBandwidth, F<:INNERMOD}
    set_bw_weights!(k, m)
    mm = momentmatrix(m)
    mmm, D = prewhiter(mm, Val{prewhite})
    bw = _optimal_bandwidth(k, mmm, prewhite)
    return bw
end