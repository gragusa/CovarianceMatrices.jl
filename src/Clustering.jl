"""
    Clustering

Lightweight internal struct for grouping observations into clusters.

Maps arbitrary cluster identifiers to contiguous integer indices 1:ngroups.
This is a minimal replacement for GroupedArrays.jl, providing only the subset
of functionality needed by CovarianceMatrices.jl.

# Fields
- `groups::Vector{Int}`: Integer cluster assignments for each observation (1:ngroups)
- `ngroups::Int`: Total number of unique clusters

# Usage
```julia
# Single clustering dimension
cl = Clustering([1, 1, 2, 2, 3, 3])
cl.groups  # [1, 1, 2, 2, 3, 3] - already contiguous
cl.ngroups # 3

# Works with any element type
cl = Clustering(["A", "A", "B", "B", "C"])
cl.groups  # [1, 1, 2, 2, 3]
cl.ngroups # 3

# Multi-way clustering
cl = Clustering([1, 1, 2, 2], ["A", "B", "A", "B"])
cl.groups  # Unique combinations mapped to 1:ngroups
cl.ngroups # Number of unique (id1, id2) combinations
```
"""
struct Clustering
    groups::Vector{Int}
    ngroups::Int
end

"""
    Clustering(v::AbstractVector)

Create a Clustering from a single vector of cluster identifiers.

Maps arbitrary identifiers to contiguous integers 1:ngroups in encounter order.
Uses a Dict for O(n) grouping.

# Example
```julia
cl = Clustering([3, 3, 1, 1, 2])
cl.groups  # [1, 1, 2, 2, 3] - mapped in encounter order
cl.ngroups # 3
```
"""
function Clustering(v::AbstractVector)
    n = length(v)
    groups = Vector{Int}(undef, n)
    groupmap = Dict{eltype(v), Int}()
    ngroups = 0
    @inbounds for i in eachindex(v)
        val = v[i]
        gid = get(groupmap, val, 0)
        if gid == 0
            ngroups += 1
            groupmap[val] = ngroups
            gid = ngroups
        end
        groups[i] = gid
    end
    return Clustering(groups, ngroups)
end

"""
    Clustering(v1::AbstractVector, vs::AbstractVector...; sort=nothing)

Create a Clustering from multiple vectors (multi-way clustering).

Combines multiple clustering dimensions by creating a unique group for each
combination of values. The `sort` keyword is accepted for compatibility but ignored.

# Example
```julia
firm_ids = [1, 1, 2, 2]
year_ids = [2020, 2021, 2020, 2021]
cl = Clustering(firm_ids, year_ids)
cl.ngroups # 4 (one for each firm-year combination)
```
"""
function Clustering(v1::AbstractVector, vs::AbstractVector...; sort=nothing)
    n = length(v1)
    # Verify all vectors have the same length
    for v in vs
        length(v) == n || throw(DimensionMismatch("All clustering vectors must have the same length"))
    end

    groups = Vector{Int}(undef, n)
    # Use a tuple as the key type for type stability
    nvecs = length(vs) + 1
    KeyType = NTuple{nvecs, Any}
    groupmap = Dict{KeyType, Int}()
    ngroups = 0

    @inbounds for i in eachindex(v1)
        # Build tuple key from all vectors
        key = (v1[i], ntuple(j -> vs[j][i], length(vs))...)::KeyType
        gid = get(groupmap, key, 0)
        if gid == 0
            ngroups += 1
            groupmap[key] = ngroups
            gid = ngroups
        end
        groups[i] = gid
    end
    return Clustering(groups, ngroups)
end

# Pass-through constructor: if already a Clustering, return as-is
Clustering(c::Clustering; sort=nothing) = c

"""
    Clustering(c1::Clustering, cs::Clustering...; sort=nothing)

Merge multiple Clustering objects into a single Clustering (multi-way clustering).

Creates a unique group for each combination of group assignments across the
input Clustering objects. The `sort` keyword is accepted for compatibility but ignored.

# Example
```julia
firms = Clustering([1, 1, 2, 2])
years = Clustering([1, 2, 1, 2])
cl = Clustering(firms, years)
cl.ngroups # 4 (one for each firm-year combination)
```
"""
function Clustering(c1::Clustering, cs::Clustering...; sort=nothing)
    n = length(c1.groups)
    # Verify all Clustering objects have the same length
    for c in cs
        length(c.groups) == n || throw(DimensionMismatch("All Clustering objects must have the same length"))
    end

    groups = Vector{Int}(undef, n)
    nvecs = length(cs) + 1
    KeyType = NTuple{nvecs, Int}
    groupmap = Dict{KeyType, Int}()
    ngroups = 0

    @inbounds for i in 1:n
        # Build tuple key from all group assignments
        key = (c1.groups[i], ntuple(j -> cs[j].groups[i], length(cs))...)::KeyType
        gid = get(groupmap, key, 0)
        if gid == 0
            ngroups += 1
            groupmap[key] = ngroups
            gid = ngroups
        end
        groups[i] = gid
    end
    return Clustering(groups, ngroups)
end

Base.length(c::Clustering) = length(c.groups)

# Iteration support (for unique() to work)
Base.iterate(c::Clustering) = iterate(c.groups)
Base.iterate(c::Clustering, state) = iterate(c.groups, state)
Base.eltype(::Type{Clustering}) = Int
