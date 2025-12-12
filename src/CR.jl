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

function avar(k::CR, X::Union{Matrix{F}, Vector{F}}; kwargs...) where {F <: Real}
    f = k.g
    S = zeros(eltype(X), (size(X, 2), size(X, 2)))
    gmin, gmax = minimum(length.(f)), maximum(length.(f))
    @assert gmin==gmax "All groups must have the same size"
    @assert gmin>1 "All groups must have at least 2 observations"
    @assert size(X, 1)==gmin "X must have the same number of observations as the groups"

    @inbounds for c in combinations(1:length(f))
        isempty(c) && continue  # skip empty combinations (Combinatorics.jl < 1.1 includes empty set)
        if length(c) == 1
            g = GroupedArray(f[c[1]])
        else
            g = GroupedArray((f[i] for i in c)...; sort = nothing)
        end
        S += (-1)^(length(c) - 1) * clusterize(parent(X), g)
    end
    #rescale!(k, S)
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
