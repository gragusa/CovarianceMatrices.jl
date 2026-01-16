function clusterize(X::Matrix, g::GroupedArray)
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
    clusterize!(X2::Matrix, S::Matrix, X::Matrix, g::GroupedArray)

In-place version of `clusterize` that uses preallocated buffers.
Aggregates observations to cluster level and computes X2'X2.

# Arguments
- `X2`: Preallocated buffer (ngroups × ncols), will be zeroed and filled
- `S`: Preallocated output buffer (ncols × ncols), result is ADDED to S
- `X`: Input moment matrix (nobs × ncols)
- `g`: GroupedArray defining cluster assignments
"""
function clusterize!(X2::Matrix{T}, S::Matrix{T}, X::Matrix{T}, g::GroupedArray) where {T}
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
    clusterize_add!(X2::Matrix, S::Matrix, X::Matrix, g::GroupedArray, sign::Int)

In-place version that aggregates and adds sign * (X2'X2) to S.
Used for multi-way clustering with inclusion-exclusion.
"""
function clusterize_add!(
        X2::Matrix{T}, S::Matrix{T}, X::Matrix{T}, g::GroupedArray, sign::Int) where {T}
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

function clusterize_mean(X::Matrix, g::GroupedArray)
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

# Single-cluster fast path: skip combinations loop entirely
function avar(k::CR, X::Union{Matrix{F}, Vector{F}}; kwargs...) where {F <: Real}
    f = k.g
    gmin, gmax = minimum(length.(f)), maximum(length.(f))
    @assert gmin == gmax "All groups must have the same size"
    @assert gmin > 1 "All groups must have at least 2 observations"
    @assert size(X, 1) == gmin "X must have the same number of observations as the groups"

    # Fast path for single cluster variable - skip combinations machinery
    if length(f) == 1
        return clusterize(parent(X), f[1])
    end

    # Multi-way clustering: use inclusion-exclusion
    S = zeros(eltype(X), (size(X, 2), size(X, 2)))
    @inbounds for c in combinations(1:length(f))
        isempty(c) && continue  # skip empty combinations (Combinatorics.jl < 1.1 includes empty set)
        if length(c) == 1
            g = GroupedArray(f[c[1]])
        else
            g = GroupedArray((f[i] for i in c)...; sort = nothing)
        end
        S += (-1)^(length(c) - 1) * clusterize(parent(X), g)
    end
    return S
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
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(f)))
    map(zip(X, combs)) do (Z, c)
        begin
            if length(c) == 1
                g = GroupedArray(f[c[1]])
            else
                g = GroupedArray((f[i] for i in c)...; sort = nothing)
            end
            (-1)^(length(c) - 1) * clusterize(Z, g)
        end
    end
end

nclusters(k::CR) = (map(x -> x.ngroups, k.g))

function total_num_clusters(k::CR)
    f = k.g
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(f)))
    map(c -> begin
            if length(c) == 1
                g = GroupedArray(f[c[1]])
            else
                g = GroupedArray((f[i] for i in c)...; sort = nothing)
            end
            length(unique(g))
        end,
        combs)
end

rescale!(k::T, S::Matrix) where {T <: CR0} = nothing

function rescale!(k::T, S::AbstractMatrix) where {T <: CR1}
    # scale total vcov estimate by ((N-1)/(N-K)) * (G/(G-1))
    G = minimum(nclusters(k))
    N = length(k.g[1])
    K = size(S, 1)
    @. S *= ((N - 1) / (N - K)) * (G / (G - 1))
end

rescale!(k::T, S::AbstractMatrix) where {T <: CR2} = nothing
rescale!(k::T, S::AbstractMatrix) where {T <: CR3} = nothing
