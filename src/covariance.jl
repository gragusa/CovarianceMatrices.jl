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

covariance(X, k::HAC, returntype::Type{T1} = CovarianceMatrix, factortype::Type{T2} = Cholesky; kwargs...) where {T1, T2}= covariance(X, k, HACCache(X, k), returntype, factortype; kwargs...) 

function covariance(X::T,
                    k::HAC{F},
                    cache::HACCache = HACCache(X, k),
                    returntype::Type{T1} = CovarianceMatrix,
                    factortype::Type{T2} = Cholesky;
                    demean::Bool = true,
                    scale::Int = size(X,1)) where {T<:AbstractMatrix,
                                                   F,                                    
                                                   T1<:Union{CovarianceMatrix, Matrix},
                                                   T2<:Union{Nothing, Factorization}}
    check_cache_consistenty(k, cache)
    demean!(cache, X, Val{demean})
    prewhiten!(cache)
    _covariance!(cache, k)
    finalize(cache, T1, T2, k, scale)
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

_covariance!(cache::HACCache, k::HAC{T}) where {T<:Fixed} = __covariance!(cache, k)

function __covariance!(cache::HACCache, k::HAC)
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
B. HC
=======#
function covariance(X::T, k::K, returntype::Type{T1} = CovarianceMatrix, 
                    factortype::Type{T2} = Cholesky; demean::Bool = true, 
                    scale::Int = size(X,1)) where {T<:AbstractMatrix, K<:HC, 
                                                   T1<:Union{CovarianceMatrix, Matrix}, 
                                                   T2<:Union{Nothing, Factorization}}
    cache = HCCache(X)
    covariance(X, k, cache, returntype, factortype, demean = demean, scale = scale)
end

function covariance(X::T, k::K, cache::HCCache, returntype::Type{T1} = CovarianceMatrix, 
                    factortype::Type{T2} = Cholesky; 
                    demean::Bool = true, scale::Int = size(X, 1)) where 
                        {T<:AbstractMatrix, K<:HC, T2<:Union{Nothing, Factorization}, 
                         T1<:Union{CovarianceMatrix, Matrix}}

    demean!(cache, X, Val{demean})
    cache.V .= cache.q'*cache.q
    finalize(cache, T1, T2, k, scale)
end

#======
CRHC
=======#
function covariance(X::T, k::K, returntype::Type{T1} = CovarianceMatrix, factortype::Type{T2} = Cholesky; demean::Bool = true, scale::Int = size(X,1), sorted::Bool = false) where {T<:AbstractMatrix, K<:CRHC, T1<:Union{CovarianceMatrix, Matrix}, T2<:Union{Nothing, Factorization}}
    cache = CRHCCache(X, k.cl)
    covariance(X, k, cache, factortype, returntype, demean = demean, scale = scale, sorted = sorted)
end


function demean!(cache::CRHCCache, X, ::Type{Val{true}})
    sum!(cache.μ, X)
    rmul!(cache.μ, 1/size(X,1))
    cache.X .= X .- cache.μ
end

function covariance(X::T, k::K, cache::CRHCCache, returntype::Type{T1} = CovarianceMatrix, factortype::Type{T2} = Cholesky; demean::Bool = true, scale::Int = size(X, 1), sorted::Bool = false ) where {T<:AbstractMatrix, K<:CRHC, T1<:Union{CovarianceMatrix, Matrix}, T2<:Union{Nothing, Factorization}}
    #check_cache_consistenty(k, cache)
    demean!(cache, X, Val{demean})
    _covariance!(cache, X, k, sorted)
    finalize(cache.V, T2, T1, k, scale)
end

#=---
Implementation
---=#

function _covariance!(cache, X, k::CRHC, sorted::Bool)
    CovarianceMatrices.installsortedX!(cache, X, k, Val{sorted})
    bstarts = (searchsorted(cache.clus, j[2]) for j in enumerate(unique(cache.clus)))
    CovarianceMatrices.clusterize!(cache, bstarts)
    rmul!(cache.V, dof_adjustment(cache, k, bstarts))
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

## TODO: Fix the finilizer. The now all store variance in cache.V
##
## finalizer(cache, F, M, K, scale)

factorizer(::Type{SVD}) = svd
factorizer(::Type{Cholesky}) = x->cholesky(Hermitian(x), check = false)

function finalize(cache, ::Type{M}, T, k, scale) where M<:Matrix
    return copy(rmul!(cache.V, 1/scale))
end

function finalize(cache, ::Type{M}, T, k, scale) where M<:CovarianceMatrix
    rmul!(cache.V, 1/scale)
    CovarianceMatrix(factorizer(T)(cache.V), k, copy(cache.V))
end