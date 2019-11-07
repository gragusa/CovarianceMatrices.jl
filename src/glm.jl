# --------------------------------------------------------------------
# Requires
# --------------------------------------------------------------------
import .GLM
using StatsModels
using StatsBase
using Tables: columntable, istable

import StatsModels: TableRegressionModel, RegressionModel
import StatsBase: modelmatrix

const INNERMOD = Union{GLM.GeneralizedLinearModel, GLM.LinearModel}
# --------------------------------------------------------------------
# GLM Methods
# --------------------------------------------------------------------

# TODO: Find a good name for it
# invinvpseudohessian
# ....
invpseudohessian(m::TableRegressionModel{T}) where T<:INNERMOD = GLM.invchol(m.model.pp).*dispersion(m.model.rr)
invpseudohessian(m::T) where T<:INNERMOD = GLM.invchol(m.pp).*dispersion(m.rr)

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
function StatsModels.hasintercept(m::INNERMOD)
    hasint = findfirst(map(x->allequal(x), eachcol(modmatrix(m))))
    hasint === nothing ? false : true
end

dispersion(m::TableRegressionModel{T}) where T<:GLM.GeneralizedLinearModel = dispersion(m.model.rr)
dispersion(m::TableRegressionModel{T}) where T<:GLM.LinearModel = 1
dispersion(m::GLM.GeneralizedLinearModel) = dispersion(m.rr)
dispersion(m::GLM.LinearModel) = 1
dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2, T3} = 1
dispersion(rr::GLM.LmResp) = 1
function dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2<:Union{GLM.Gamma, GLM.Bernoulli, GLM.InverseGaussian}, T3}
    sum(abs2, rr.wrkwt.*rr.wrkresid)/sum(rr.wrkwt)
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
# hasresiduals(m::INNERMOD) = true
# hasmodelmatrix(m::TableRegressionModel{T}) where T<:INNERMOD = true

# --------------------------------------------------------------------
# HAC GLM Methods
# --------------------------------------------------------------------
function StatsBase.vcov(k::HAC, m; prewhite=false, dof_adjustment::Bool=true, scale::Real=1)
    return _vcov(k, m, prewhite, dof_adjustment, scale)
end

function _vcov(k::HAC, m, prewhite::Bool, dof_adjustment::Bool, scale::Real)
    setupkernelweights!(k, m)
    B  = invpseudohessian(m)
    mm = momentmatrix(m)
    A = lrvar(k, mm; prewhite=prewhite, demean=false)
    scale *= (dof_adjustment ? numobs(m)^2/dof_resid(m) : numobs(m))
    V = Symmetric((B*A*B).*scale)
    return V
end

vcovmatrix(
    k::HAC,
    m::RegressionModel,
    factorization=Cholesky;
    prewhite=false,
    dof_adjustment::Bool=true,
    scale::Real=1,
) = _vcovmatrix(k, m, prewhite, dof_adjustment, scale, factorization)

function _vcovmatrix(
    k::HAC,
    m::RegressionModel,
    prewhite::Bool,
    dof_adjustment::Bool,
    scale::Real,
    ::Type{Cholesky},
)
    V = _vcov(k, m, prewhite, dof_adjustment, scale)
    return CovarianceMatrix(cholesky(V, check=true), k, V)
end

function _vcovmatrix(
    k::HAC,
    m::RegressionModel,
    prewhite::Bool,
    dof_adjustment::Bool,
    scale::Real,
    ::Type{SVD},
)
    V = _vcov(k, m, prewhite, dof_adjustment, scale)
    return CovarianceMatrix(svd(V.data), k, V)
end

# --------------------------------------------------------------------
# HC GLM Methods 
# --------------------------------------------------------------------
hatmatrix(m::TableRegressionModel{T}, x) where T<:INNERMOD = hatmatrix(m.model, x)

function hatmatrix(m::T, x) where T<:INNERMOD
    cf = m.pp.chol.UL::UpperTriangular
    rdiv!(x, cf)
    return sum(x.^2, dims = 2)
 end

adjfactor(k::HC0, m::RegressionModel, x) = one(first(x))
adjfactor(k::HC1, m::RegressionModel, x) = numobs(m)./dof_resid(m)
adjfactor(k::HC2, m::RegressionModel, x) = one(first(x)) ./(one(first(x)).-hatmatrix(m, x))
adjfactor(k::HC3, m::RegressionModel, x) = one(first(x))./(one(first(x)).-hatmatrix(m, x)).^2

function adjfactor(k::HC4, m::RegressionModel, x)
    n, p = size(x)
    tone = one(eltype(x))
    h = hatmatrix(m, x)
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

adjust!(m, adj::AbstractFloat) = m
adjust!(m, adj::AbstractMatrix) = m.*sqrt.(adj)

StatsBase.vcov(k::HC, m::RegressionModel; scale::Real=1) = _vcov(k, m, scale)

function _vcov(k::HC, m::RegressionModel, scale)
    B  = invpseudohessian(m)
    mm = momentmatrix(m)
    adj = adjfactor(k, m, modmatrix(m))
    mm = adjust!(mm, adj)
    scale *= length(adj) > 1 ? one(first(adj)) : adj
    A = mm'*mm
    return Symmetric((B*A*B).*scale)
end

vcovmatrix(k::HC, m::RegressionModel, factorization=Cholesky; scale::Real=1) =
    _vcovmatrix(k, m, scale, Val{:factorization})

function _vcovmatrix(k::HC, m::RegressionModel, scale::Real, ::Type{Cholesky})
    V = _vcov(k, m, scale)
    CovarianceMatrix(cholesky(V, check=true), k, V)
end

function _vcovmatrix(k::HC, m::RegressionModel, scale::Real, ::Type{SVD})
    V = _vcov(k, m, scale)
    CovarianceMatrix(svd(V), k, V)
end

# --------------------------------------------------------------------
# CRHC GLM Methods
# --------------------------------------------------------------------

function installcache(k::CRHC, m::RegressionModel)
    X = modmatrix(m)
    res = resid(m)
    f = categorize(k.cl)
    (X, res), sf = bysort((X, res), f)
    ci = clustersintervals(sf)
    p = size(X, 2)
    cf = chol(m)
    Shat = Matrix{eltype(res)}(undef,p,p)
    return CRHCCache(similar(X), X, res, similar(X, (0,0)), cf, Shat, ci, sf)
end

function StatsBase.vcov(k::CRHC, m::RegressionModel; scale::Real=1)
    knew = recast(k, m)
    length(knew.cl) == numobs(m) || throw(ArgumentError(k, "the length of the cluster variable must be $(numobs(m))"))
    cache = installcache(knew, m)
    return _vcov(knew, m, cache, scale)
end

function vcovmatrix(k::CRHC, m::RegressionModel, factorization = Cholesky; scale::Real=1)
    knew = recast(k, m)
    cache = installcache(knew, m)
    df = dofadjustment(knew, cache)
    return _vcovmatrix(knew, m, cache, scale, factorization)
end

function _vcov(k::CRHC, m::RegressionModel, cache::CRHCCache, scale::Real)
    B = invpseudohessian(m)
    res = adjustresid!(k, cache)
    cache.momentmatrix .= cache.modelmatrix.*res
    df = dofadjustment(k, cache)
    scale *= df
    Shat = clusterize!(cache)
    return Symmetric((B*Shat*B).*scale)
end

function _vcovmatrix(
    k::HC,
    m::RegressionModel,
    cache::CRHCCache,
    scale::Real,
    ::Type{Cholesky}
)
    V = _vcov(k, m, cache, scale)
    CovarianceMatrix(cholesky(V, check=true), k, V)
end

function _vcovmatrix(
    k::HC,
    m::RegressionModel,
    cache::CRHCCache,
    scale::Real,
    ::Type{SVD}
)
    V = _vcov(k, m, cache, cache)
    CovarianceMatrix(svd(V.data), k, V)
end

# -----------------------------------------------------------------------------
# CRHC GLM - Trick to use vcov(CRHC1(:cluster, df), ::GLM)
# -----------------------------------------------------------------------------
StatsBase.stderror(k::RobustVariance, m::RegressionModel; kwargs...) = sqrt.(diag(vcov(k, m; kwargs...)))
StatsBase.stderror(v::CovarianceMatrix) = sqrt.(diag(v.V))

# -----------------------------------------------------------------------------
# CRHC GLM - Trick to use vcov(CRHC1(:cluster, df), ::GLM)
# -----------------------------------------------------------------------------
recast(k::CRHC{T,D}, m::INNERMOD) where {T<:AbstractVector, D<:Nothing} = k
recast(k::CRHC{T,D}, m::TableRegressionModel) where {T<:AbstractVector, D<:Nothing} = k
reterm(k::CRHC{T,D}, m::TableRegressionModel) where {T, D} = tuple(k.cl...)

function recast(k::CRHC{T,D}, m::TableRegressionModel) where {T<:Symbol, D}
    @assert istable(k.df) "`df` must be a DataFrames"
    t = k.cl
    if length(k.df[!, t]) == length(m.mf.data[1])
        ## The dimension fit
        id = compress(categorical(k.df[:,t]))
        return renew(k, id)
    else
        f = m.mf.f
        frm = f.lhs ~ tuple(f.rhs.terms..., Term(t))
        nt = NamedTuple{tuple(StatsModels.termvars(frm)...)}(columntable(k.df))
        idx = StatsModels.missing_omit(nt)[2]
        id = compress(categorical(k.df[idx,t]))
        return renew(k, id)
    end
    # ct = columntable(clus)
    # length_unique = map(x->length(unique(x)), ct)
    # fg = 1:prod(length_unique)
    # #cl = map(x->compress(categorical(x)), eachcol(x))
    # clus[!, :clusid] .= size(clus, 2) > 1 ? zero(Int) : clus[!, tterms[1]]
    # if length(tterms) > 1
    #     for (i,j) in enumerate(groupby(clus, [tterms...]))
    #         j[:, :clusid] .= fg[i]
    #     end
    # end
    # id = compress(categorical(clus[!, :clusid]))    
end


# -----------------------------------------------------------------------------
# optimalbandwidth method
# -----------------------------------------------------------------------------
function optimalbandwidth(
    k::HAC,
    m::TableRegressionModel{F};
    kwargs...
) where F<:INNERMOD
    setupkernelweights!(k, m)
    optimalbandwidth(k, m.model; kwargs...)
end

function optimalbandwidth(k::HAC, m::F; prewhite=false) where F<:INNERMOD    
    mm = momentmatrix(m)
    setupkernelweights!(k, m)
    mmm, D = prewhiter(mm, Val{prewhite})
    return optimalbandwidth(k, mmm; prewhite=prewhite)
end

function setupkernelweights!(k, m::TableRegressionModel{T}) where T<:INNERMOD
    β = coef(m)
    resize!(k.weights, length(β))
    "(Intercept)" ∈ coefnames(m) ? (k.weights .= 1.0; k.weights[1] = 0.0) : k.weights .= 1.0
    return nothing
end

function setupkernelweights!(k, m::T) where T<:INNERMOD
    cf = coef(m)
    resize!(k.weights, length(cf))
    fill!(k.weights, 1)
    i = interceptindex(m)
    i !== nothing && (k.weights[i] = 0)
    return nothing
end
