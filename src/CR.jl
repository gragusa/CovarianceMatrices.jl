function clusterize(X::Matrix, g::Clustering)
    X2 = zeros(eltype(X), g.ngroups, size(X, 2))
    idx = 0
    for j in axes(X, 2)
        idx += 1
        @inbounds @simd for i in axes(X, 1)
            X2[g.groups[i], idx] += X[i, j]
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
@inline _get_lengths(f::Tuple{Clustering, Clustering, Clustering}) = (length(f[1]), length(f[2]), length(f[3]))
@inline _get_lengths(f::NTuple{N, Clustering}) where {N} = ntuple(i -> length(f[i]), Val(N))

# Single-cluster fast path (most common case) - fully type-stable
function avar(k::CR, X::Union{Matrix{F}, Vector{F}}; kwargs...) where {F <: Real}
    f = k.g
    return _avar_impl(f, X)
end

# Dispatch on tuple length for type stability
@inline function _avar_impl(f::Tuple{Clustering}, X::Union{Matrix{F}, Vector{F}}) where {F <: Real}
    n = length(f[1])
    @assert n > 1 "All groups must have at least 2 observations"
    @assert size(X, 1) == n "X must have the same number of observations as the groups"
    return clusterize(parent(X), f[1])
end

# Two-way clustering - type-stable
@inline function _avar_impl(f::Tuple{Clustering, Clustering}, X::Union{Matrix{F}, Vector{F}}) where {F <: Real}
    n1, n2 = length(f[1]), length(f[2])
    @assert n1 == n2 "All groups must have the same size"
    @assert n1 > 1 "All groups must have at least 2 observations"
    @assert size(X, 1) == n1 "X must have the same number of observations as the groups"

    Xp = parent(X)
    # Inclusion-exclusion: S = S1 + S2 - S12
    S1 = clusterize(Xp, f[1])
    S2 = clusterize(Xp, f[2])
    g12 = Clustering(f[1], f[2])
    S12 = clusterize(Xp, g12)
    return S1 + S2 - S12
end

# General N-way clustering fallback
function _avar_impl(f::NTuple{N, Clustering}, X::Union{Matrix{F}, Vector{F}}) where {N, F <: Real}
    lens = _get_lengths(f)
    gmin, gmax = minimum(lens), maximum(lens)
    @assert gmin == gmax "All groups must have the same size"
    @assert gmin > 1 "All groups must have at least 2 observations"
    @assert size(X, 1) == gmin "X must have the same number of observations as the groups"

    # Multi-way clustering: use inclusion-exclusion
    S = zeros(eltype(X), (size(X, 2), size(X, 2)))
    Xp = parent(X)
    @inbounds for c in combinations(1:N)
        isempty(c) && continue
        if length(c) == 1
            g = f[c[1]]
        else
            g = _merge_clusterings(f, c)
        end
        S += (-1)^(length(c) - 1) * clusterize(Xp, g)
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

# CachedCR fast path: uses preallocated buffers and precomputed cluster indices
function avar(k::CachedCR, X::Union{Matrix{F}, Vector{F}}; kwargs...) where {F <: Real}
    cache = k.cache
    ncols = size(X, 2)

    # Validate cache dimensions
    @assert ncols == cache.ncols "Moment matrix columns ($ncols) must match cache columns ($(cache.ncols))"
    @assert size(X, 1) == length(k.g[1]) "X must have the same number of observations as the groups"

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
    return (clusterize(X[1], f[1]), clusterize(X[2], f[2]), -clusterize(X[3], g12))
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
        (-1)^(length(c) - 1) * clusterize(Z, g)
    end
end

# Type-stable nclusters
nclusters(k::CR) = _nclusters(k.g)
@inline _nclusters(f::Tuple{Clustering}) = (f[1].ngroups,)
@inline _nclusters(f::Tuple{Clustering, Clustering}) = (f[1].ngroups, f[2].ngroups)
@inline _nclusters(f::Tuple{Clustering, Clustering, Clustering}) = (f[1].ngroups, f[2].ngroups, f[3].ngroups)
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

rescale!(k::T, S::Matrix) where {T <: CR0} = nothing

function rescale!(k::T, S::AbstractMatrix) where {T <: CR1}
    # scale total vcov estimate by ((N-1)/(N-K)) * (G/(G-1))
    nc = nclusters(k)
    G = minimum(nc)
    N = length(k.g[1])
    K = size(S, 1)
    @. S *= ((N - 1) / (N - K)) * (G / (G - 1))
end

rescale!(k::T, S::AbstractMatrix) where {T <: CR2} = nothing
rescale!(k::T, S::AbstractMatrix) where {T <: CR3} = nothing
