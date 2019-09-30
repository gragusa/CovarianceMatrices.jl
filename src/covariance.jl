covariance(k, m; kwargs...) = covariance(k, m, cache(k, m), Matrix, Nothing; kwargs...)

covariance(k, m::Vector, args...; kwargs...) = covariance(k, reshape(m, (length(m),1)), args...; kwargs...)

function covariance(k, m::AbstractMatrix, ::Type{T}; kwargs...) where T<:Matrix
    covariance(k, m, cache(k, m), Matrix, Nothing; kwargs...)
end

function covariance(k, m::AbstractMatrix, ::Type{T}; kwargs...) where T<:CovarianceMatrix
    covariance(k, m, cache(k, m), T, SVD; kwargs...)
end

function covariance(k, m::AbstractMatrix, ::Type{T}; kwargs...) where T<:Factorization
    covariance(k, m, cache(k, m), CovarianceMatrix, T; kwargs...)
end

function covariance(k, m::AbstractMatrix, cache::AbstractCache, ::Type{T}; kwargs...) where T<:Factorization
    covariance(k, m, cache, CovarianceMatrix, T; kwargs...)
end

function covariance(k, m::AbstractMatrix, cache::AbstractCache, ::Type{T}; kwargs...) where T<:Matrix
    covariance(k, m, cache, Matrix, Nothing; kwargs...)
end

#======
Demeaning methods
=======#

function demean!(cache, ::Type{Val{true}})
    mean!(cache.μ, cache.q)
    cache.q .= cache.q .- cache.μ
end
demean!(cache, ::Type{Val{false}}) = nothing

function demean!(cache, X, ::Type{Val{true}})
    copy!(cache.q, X)
    mean!(cache.μ, cache.q)
    cache.q .= cache.q .- cache.μ
end
demean!(cache, X, ::Type{Val{false}}) = copy!(cache.q, X); nothing

#======
A. HAC
=======#
function covariance(k::T, X, cache, returntype, factortype;
                    demean::Bool=true, scale::Int=size(X,1)) where T<:HAC
    check_cache_consistenty(k, cache)
    demean!(cache, X, Val{demean})
    prewhiten!(cache)
    _covariance!(cache, k)
    finalize(cache, returntype, factortype, k, scale)
end

#=--
Implementation
--=#
function _covariance!(cache::HACCache, k::HAC{Optimal{T}}) where {T<:OptimalBandwidth}
    n, p = size(cache.XX)
    setupkernelweights!(k, p, eltype(cache.XX))
    optimal_bw!(cache, k, T())
    __covariance!(cache, k)
end

_covariance!(cache, k::HAC{T}) where {T<:Fixed} = __covariance!(cache, k)

function __covariance!(cache, k::HAC)
    n, p = size(cache.XX)
    #fill!(cache.V, zero(eltype(cache.XX)))
    bw = first(k.bw)
    mul!(cache.V, cache.XX', cache.XX)
    triu!(cache.V)
    idxs = getcovindeces(k, n)
    @inbounds for j in idxs
        k_j = CovarianceMatrices.kernel(k, j/bw)
        LinearAlgebra.axpy!(k_j, CovarianceMatrices.Γ!(cache, j), cache.V)
    end
    LinearAlgebra.copytri!(cache.V, 'U')
    swhiten!(cache)
    nothing
end

#=======
B. VARHC
=======#
function covariance(k::T, X, cache, returntype, factortype;
    demean::Bool=true, scale::Int=1) where T<:VARHAC
    maxlag, lagstrategy, selectionstrategy = k.maxlag, k.lagstrategy, k.selectionstrategy
    demean!(cache, X, Val{demean})
    strategy = selectionstrategy == :aic ? 1 : (selectionstrategy == :bic ? 2 : 3)
    cache.V .= varhac(cache.q,maxlag,lagstrategy,strategy)
    finalize(cache, returntype, factortype, k, scale)
end


#=======
B. HC
=======#
function covariance(k::K, X, cache, returntype, factortype;
                    demean::Bool=true, scale::Int=size(X, 1)) where K<:HC
    demean!(cache, X, Val{demean})
    cache.V .= cache.q'*cache.q
    finalize(cache, returntype, factortype, k, scale)
end

#======
CRHC
=======#

function covariance(k::T, X, cache, returntype, factortype;
                    demean::Bool=true, scale::Int=size(X, 1), sorted::Bool=false) where T<:CRHC
    #check_cache_consistenty(k, cache)
    demean!(cache, X, Val{demean})
    _covariance!(cache, X, k, sorted)
    finalize(cache, returntype, factortype, k, scale)
end

#=---
Implementation
---=#

function _covariance!(cache, X, k::CRHC, sorted)
    CovarianceMatrices.installsortedX!(cache, X, k, Val{sorted})
    bstarts = (searchsorted(cache.clus, j[2]) for j in enumerate(unique(cache.clus)))
    V = CovarianceMatrices.clusterize!(cache, bstarts)
    rmul!(V, dof_adjustment(cache, k, bstarts))
end

function installsortedX!(cache, X, k, ::Type{Val{true}})
    copyto!(cache.q, X)
    copyto!(cache.clus, k.cl)
    nothing
end

function installsortedX!(cache, X, k, ::Type{Val{false}})
    n, p = size(cache.X)
    sortperm!(cache.clusidx, k.cl)
    cidx = cache.clusidx
    c  = k.cl
    cc = cache.clus
    XX = cache.q
    @inbounds for j in 1:p, i in eachindex(cache.clusidx)
        XX[i,j] = X[cidx[i], j]
    end
    @inbounds for i in eachindex(cache.clusidx)
        cc[i] = c[cidx[i]]
    end
    nothing
end

#======
Finalizers
======#

factorizer(::Type{SVD}) = svd
factorizer(::Type{Cholesky}) = x->cholesky(Hermitian(x), check = false)

function finalize(cache, ::Type{M}, T, k, scale) where M<:Matrix
    return copy(rmul!(cache.V, 1/scale))
end

function finalize(cache, ::Type{M}, T, k, scale) where M<:CovarianceMatrix
    rmul!(cache.V, 1/scale)
    CovarianceMatrix(factorizer(T)(cache.V), k, copy(cache.V))
end