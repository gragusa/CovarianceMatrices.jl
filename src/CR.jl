# The scatter loops below index `g.groups` by row under `@inbounds`.
function _checkclusterlength(X, g::Clustering)
    length(g.groups) == size(X, 1) || throw(DimensionMismatch(
        "the cluster variable has $(length(g.groups)) observations but the moment matrix has $(size(X, 1)) rows"))
    return nothing
end

function clusterize(X::AbstractVecOrMat, g::Clustering)
    Base.require_one_based_indexing(X)
    _checkclusterlength(X, g)
    X2 = zeros(eltype(X), g.ngroups, size(X, 2))
    for j in axes(X, 2)
        @inbounds @simd for i in axes(X, 1)
            X2[g.groups[i], j] += X[i, j]
        end
    end
    return X2' * X2
end

"""
    clusterize!(X2::Matrix, S::Matrix, X::Matrix, g::Clustering)

In-place version of `clusterize` that uses preallocated buffers.
Aggregates observations to cluster level and computes X2'X2.

# Arguments
- `X2`: Preallocated buffer (ngroups × ncols), will be zeroed and filled
- `S`: Preallocated output buffer (ncols × ncols), result is ADDED to S
- `X`: Input moment matrix (nobs × ncols)
- `g`: Clustering defining cluster assignments
"""
function clusterize!(X2::Matrix{T}, S::Matrix{T}, X::Matrix{T}, g::Clustering) where {T}
    _checkclusterlength(X, g)
    # Zero the aggregation buffer
    fill!(X2, zero(T))

    # Aggregate to cluster level
    for j in axes(X, 2)
        @inbounds @simd for i in axes(X, 1)
            X2[g.groups[i], j] += X[i, j]
        end
    end

    # Compute X2'X2 and add to S (using BLAS for efficiency)
    # S += X2' * X2
    LinearAlgebra.BLAS.syrk!('U', 'T', one(T), X2, one(T), S)
    # Copy upper triangle to lower
    @inbounds for j in axes(S, 2)
        for i in (j + 1):size(S, 1)
            S[i, j] = S[j, i]
        end
    end
    return S
end

"""
    clusterize_add!(X2::Matrix, S::Matrix, X::Matrix, g::Clustering, sign::Int)

In-place version that aggregates and adds sign * (X2'X2) to S.
Used for multi-way clustering with inclusion-exclusion.
"""
function clusterize_add!(
        X2::Matrix{T}, S::Matrix{T}, X::Matrix{T}, g::Clustering, sign::Int) where {T}
    _checkclusterlength(X, g)
    fill!(X2, zero(T))

    for j in axes(X, 2)
        @inbounds @simd for i in axes(X, 1)
            X2[g.groups[i], j] += X[i, j]
        end
    end

    # S += sign * X2' * X2
    LinearAlgebra.BLAS.syrk!('U', 'T', T(sign), X2, one(T), S)
    @inbounds for j in axes(S, 2)
        for i in (j + 1):size(S, 1)
            S[i, j] = S[j, i]
        end
    end
    return S
end

"""
    clusterize_gather_add!(X2::Matrix, S::Matrix, X::Matrix, indices::Vector{Vector{Int}}, sign::Int)

Fast gather-based aggregation using precomputed cluster indices.
Aggregates observations by cluster using sequential reads (cache-friendly),
then adds sign * (X2'X2) to S.

This is ~10x faster than scatter-add for large matrices because it converts
random writes to sequential reads.
"""
function clusterize_gather_add!(X2::Matrix{T}, S::Matrix{T}, X::Matrix{T},
        indices::Vector{Vector{Int}}, sign::Int) where {T}
    ngroups = length(indices)
    ncols = size(X, 2)

    # Gather-sum: for each cluster, sum observations belonging to it
    @inbounds for gc in 1:ngroups
        idx = indices[gc]
        for j in 1:ncols
            s = zero(T)
            @simd for k in eachindex(idx)
                s += X[idx[k], j]
            end
            X2[gc, j] = s
        end
    end

    # S += sign * X2' * X2
    LinearAlgebra.BLAS.syrk!('U', 'T', T(sign), X2, one(T), S)
    @inbounds for j in axes(S, 2)
        for i in (j + 1):size(S, 1)
            S[i, j] = S[j, i]
        end
    end
    return S
end

function clusterize_mean(X::Matrix, g::Clustering)
    _checkclusterlength(X, g)
    X2 = zeros(eltype(X), g.ngroups, size(X, 2))
    idx = 0
    for j in axes(X, 2)
        idx += 1
        @inbounds @simd for i in axes(X, 1)
            X2[g.groups[i], idx] += X[i, j]
        end
    end
    return X2
end

# Helper to get lengths from tuple of Clustering in type-stable way
@inline _get_lengths(f::Tuple{Clustering}) = (length(f[1]),)
@inline _get_lengths(f::Tuple{Clustering, Clustering}) = (length(f[1]), length(f[2]))
@inline _get_lengths(f::Tuple{
    Clustering, Clustering, Clustering}) = (length(f[1]), length(f[2]), length(f[3]))
@inline _get_lengths(f::NTuple{N, Clustering}) where {N} = ntuple(i -> length(f[i]), Val(N))

function _checkclusters(lens, X)
    allequal(lens) || throw(DimensionMismatch(
        "all cluster variables must have the same length, got lengths $(lens)"))
    n = first(lens)
    size(X, 1) == n || throw(DimensionMismatch(
        "the cluster variables have $n observations but the moment matrix has $(size(X, 1)) rows"))
    n > 1 || throw(ArgumentError(
        "clustering needs at least 2 observations, got $n"))
    return nothing
end

# Single-cluster fast path (most common case) - fully type-stable
function avar(k::CR, X::AbstractVecOrMat{F}; kwargs...) where {F <: Real}
    f = k.g
    return _avar_impl(f, X)
end

function avar_with_info(k::CR, X; weights = nothing, kwargs...)
    V = avar(k, X; kwargs...)
    return V, (; nclusters = nclusters(k))
end

# `Cluster`: the mean/matrix-interface cluster estimator. It reuses the same
# cluster-sum machinery as the CR family but is a distinct, explicitly named type
# for the raw (uncorrected) cluster covariance of a data/moment matrix.
#
# Note on `CR0`–`CR3` on a bare matrix: the `avar(k::CR, X)` method above applies
# no finite-sample correction, so on a matrix all four CR variants reduce to this
# same raw cluster sum (the DOF/leverage factors are formed on the model path via
# `residual_adjustment`, and are relied on there and by `CachedCR`). `Cluster` is
# the recommended way to request that raw cluster covariance for the mean case.
function avar(k::Cluster, X::AbstractVecOrMat{F}; kwargs...) where {F <: Real}
    return _avar_impl(k.g, X)
end

# Dispatch on tuple length for type stability
@inline function _avar_impl(f::Tuple{Clustering}, X::AbstractVecOrMat{F}) where {F <: Real}
    _checkclusters(_get_lengths(f), X)
    return clusterize(X, f[1])
end

# Two-way clustering - type-stable
@inline function _avar_impl(f::Tuple{Clustering, Clustering}, X::AbstractVecOrMat{F}) where {F <:
                                                                                             Real}
    _checkclusters(_get_lengths(f), X)

    # Inclusion-exclusion: S = S1 + S2 - S12
    S1 = clusterize(X, f[1])
    S2 = clusterize(X, f[2])
    g12 = Clustering(f[1], f[2])
    S12 = clusterize(X, g12)
    return S1 + S2 - S12
end

# General N-way clustering fallback
function _avar_impl(f::NTuple{N, Clustering}, X::AbstractVecOrMat{F}) where {N, F <: Real}
    _checkclusters(_get_lengths(f), X)

    # Multi-way clustering: use inclusion-exclusion
    S = zeros(eltype(X), (size(X, 2), size(X, 2)))
    for c in combinations(1:N)
        isempty(c) && continue
        if length(c) == 1
            g = f[c[1]]
        else
            g = _merge_clusterings(f, c)
        end
        S += (-1)^(length(c) - 1) * clusterize(X, g)
    end
    return S
end

# Type-stable clustering merge helper
@inline function _merge_clusterings(f::NTuple{N, Clustering}, indices::Vector{Int}) where {N}
    if length(indices) == 2
        return Clustering(f[indices[1]], f[indices[2]])
    elseif length(indices) == 3
        return Clustering(f[indices[1]], f[indices[2]], f[indices[3]])
    else
        # Fallback for 4+ way clustering (rare)
        return Clustering(ntuple(i -> f[indices[i]], length(indices))...)
    end
end

# CachedCR fast path: uses preallocated buffers and precomputed cluster indices. A
# `CachedCR` built from a model has no buffers and takes the uncached `avar(::CR, X)`.
function avar(k::CachedCR{<:CR, <:CRCache}, X::Union{Matrix{F}, Vector{F}};
        kwargs...) where {F <: Real}
    cache = k.cache
    ncols = size(X, 2)

    # Validate cache dimensions
    ncols == cache.ncols || throw(DimensionMismatch(
        "the moment matrix has $ncols columns but the cache was built for $(cache.ncols)"))
    _checkclusters(_get_lengths(k.g), X)

    # Zero the output buffer
    S = cache.S_buffer
    fill!(S, zero(F))

    # Use precomputed cluster indices for fast gather-based aggregation
    Xp = parent(X)
    @inbounds for i in eachindex(cache.cluster_indices)
        clusterize_gather_add!(
            cache.X2_buffers[i], S, Xp, cache.cluster_indices[i], cache.signs[i])
    end

    return S
end

function avar_tuple(k::CR, X; kwargs...)
    f = k.g
    return _avar_tuple_impl(f, X)
end

# Type-stable implementation for single clustering
function _avar_tuple_impl(f::Tuple{Clustering}, X)
    return (clusterize(X[1], f[1]),)
end

# Type-stable implementation for two-way clustering
function _avar_tuple_impl(f::Tuple{Clustering, Clustering}, X)
    g12 = Clustering(f[1], f[2])
    return (clusterize(X[1], f[1]), clusterize(X[2], f[2]), clusterize(X[3], g12))
end

# Fallback for N-way
function _avar_tuple_impl(f::NTuple{N, Clustering}, X) where {N}
    combs = Iterators.filter(!isempty, combinations(1:N))
    map(zip(X, combs)) do (Z, c)
        if length(c) == 1
            g = f[c[1]]
        else
            g = _merge_clusterings(f, c)
        end
        clusterize(Z, g)
    end
end

"""
    nclusters(k)

Return the number of groups in each clustering dimension of a cluster-robust
estimator, as a tuple with one entry per dimension.

```julia
nclusters(CR0(firm))         # (n_firms,)
nclusters(CR0(firm, year))   # (n_firms, n_years)
```
"""
nclusters(k::CR) = _nclusters(_check_resolved(k).g)
@inline _nclusters(f::Tuple{Clustering}) = (f[1].ngroups,)
@inline _nclusters(f::Tuple{Clustering, Clustering}) = (f[1].ngroups, f[2].ngroups)
@inline _nclusters(f::Tuple{
    Clustering, Clustering, Clustering}) = (f[1].ngroups, f[2].ngroups, f[3].ngroups)
@inline _nclusters(f::NTuple{N, Clustering}) where {N} = ntuple(i -> f[i].ngroups, Val(N))

# Type-stable total_num_clusters
function total_num_clusters(k::CR)
    f = k.g
    return _total_num_clusters(f)
end

@inline function _total_num_clusters(f::Tuple{Clustering})
    return (f[1].ngroups,)
end

@inline function _total_num_clusters(f::Tuple{Clustering, Clustering})
    g12 = Clustering(f[1], f[2])
    return (f[1].ngroups, f[2].ngroups, g12.ngroups)
end

function _total_num_clusters(f::NTuple{N, Clustering}) where {N}
    combs = Iterators.filter(!isempty, combinations(1:N))
    map(c -> begin
            if length(c) == 1
                f[c[1]].ngroups
            else
                _merge_clusterings(f, c).ngroups
            end
        end,
        combs)
end
