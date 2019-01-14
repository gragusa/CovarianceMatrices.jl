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
    finalize(T2, T1, k, cache, scale)
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
    fill!(cache.V, zero(eltype(cache.XX)))
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
    rmul!(cache.V, 1/(n+isprewhiten(k)))
    nothing
end

#=======
B. HC
=======#
function covariance(X::T, k::K, returntype::Type{T1} = CovarianceMatrix, factortype::Type{T2} = Cholesky; demean::Bool = true, scale::Int = size(X,1)) where {T<:AbstractMatrix, K<:HC, T1<:Union{CovarianceMatrix, Matrix}, T2<:Union{Nothing, Factorization}}
    V = X'X
    finalize(V, T2, T1, k, scale)
end

function covariance(X::T, k::K, cache::HCCache, factortype::Type{T1} = Cholesky, returntype::Type{T2} = CovarianceMatrix; demean::Bool = true, scale::Int = size(X, 1)) where {T<:AbstractMatrix, K<:HC, T1<:Union{Nothing, Factorization}, T2<:Union{CovarianceMatrix, Matrix}}
    check_cache_consistenty(k, cache)
    _covariance!(cache, k)
    finalize(T2, T1, k, cache, scale)
end

#======
CRHC
=======#
function covariance(X::T, k::K, returntype::Type{T1} = CovarianceMatrix, factortype::Type{T2} = Cholesky; demean::Bool = true, scale::Int = size(X,1), sorted::Bool = false) where {T<:AbstractMatrix, K<:CRHC, T1<:Union{CovarianceMatrix, Matrix}, T2<:Union{Nothing, Factorization}}
    cache = CRHCCache(X, k.cl)
    covariance(X, k, cache, factortype, returntype, demean = demean, scale = scale, sorted = sorted)
end

function covariance(X::T, k::K, cache::CRHCCache, returntype::Type{T1} = CovarianceMatrix, factortype::Type{T2} = Cholesky; demean::Bool = true, scale::Int = size(X, 1), sorted::Bool = false ) where {T<:AbstractMatrix, K<:CRHC, T1<:Union{CovarianceMatrix, Matrix}, T2<:Union{Nothing, Factorization}}
    #check_cache_consistenty(k, cache)
    _covariance!(cache, X, k, sorted)
    finalize(cache.M, T2, T1, k, scale)
end

#=---
Implementation
---=#

function _covariance!(cache, X, k::CRHC, sorted::Bool)
    CovarianceMatrices.installsortedX!(cache, X, k, Val{sorted})
    bstarts = (searchsorted(cache.clus, j[2]) for j in enumerate(unique(cache.clus)))
    CovarianceMatrices.clusterize!(cache, bstarts)
    rmul!(cache.M, dof_adjustment(cache, k, bstarts))

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
finalize(V, ::Type{T}, ::Type{Matrix}, k, scale::Real) where T = rmul!(V, 1/scale)
finalize(::Type{T}, ::Type{Matrix}, k, cache, scale::Real) where T = rmul!(cache.V, 1/scale)

function finalize(V, ::Type{SVD}, ::Type{CovarianceMatrix}, k, scale::Real)
    rmul!(V, 1/scale)
    CovarianceMatrix(svd(V), k, V)
end

function finalize(V, ::Type{Cholesky}, ::Type{CovarianceMatrix},  k, scale::Real)
    rmul!(V, 1/scale)
    CovarianceMatrix(cholesky(Hermitian(V)), k, V)
end

function finalize(::Type{SVD}, ::Type{CovarianceMatrix}, k, cache::AbstractCache, scale::Real)
    rmul!(cache.V, 1/scale)
    CovarianceMatrix(svd(cache.V), k, copy(cache.V))
end

function finalize(::Type{Cholesky}, ::Type{CovarianceMatrix},  k, cache::HACCache, scale::Real)
    rmul!(cache.V, 1/scale)
    CovarianceMatrix(cholesky(Hermitian(cache.V)), k, copy(cache.V))
end
