"""
    Clustering

Grouping of observations into clusters. Cluster estimators (`CR0`–`CR3`,
`Cluster`) store one `Clustering` per clustering dimension in their `g` field.

Maps arbitrary cluster identifiers to contiguous integer group codes `1:G`, in the
order the identifiers are first encountered.

# Accessors
- [`groups`](@ref CovarianceMatrices.groups)`(c)`: the group code of each observation.
- [`nclusters`](@ref)`(c)`: the number of groups `G`.

# Usage
```julia
using CovarianceMatrices: Clustering, groups, nclusters

# Single clustering dimension
cl = Clustering([1, 1, 2, 2, 3, 3])
groups(cl)     # [1, 1, 2, 2, 3, 3] - already contiguous
nclusters(cl)  # 3

# Works with any element type
cl = Clustering(["A", "A", "B", "B", "C"])
groups(cl)     # [1, 1, 2, 2, 3]
nclusters(cl)  # 3

# Multi-way clustering
cl = Clustering([1, 1, 2, 2], ["A", "B", "A", "B"])
groups(cl)     # Unique combinations mapped to 1:G
nclusters(cl)  # Number of unique (id1, id2) combinations
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
groups(cl)     # [1, 1, 2, 2, 3] - mapped in encounter order
nclusters(cl)  # 3
```
"""
function Clustering(v::AbstractVector)
    Base.require_one_based_indexing(v)
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
    Clustering(v1::AbstractVector, vs::AbstractVector...)

Create a Clustering from multiple vectors (multi-way clustering).

Combines multiple clustering dimensions by creating a unique group for each
combination of values.

# Example
```julia
firm_ids = [1, 1, 2, 2]
year_ids = [2020, 2021, 2020, 2021]
cl = Clustering(firm_ids, year_ids)
nclusters(cl)  # 4 (one for each firm-year combination)
```
"""
function Clustering(v1::AbstractVector, vs::AbstractVector...)
    # Delegate to single-vector constructors and merge for type stability
    # This avoids NTuple{N, Any} keys in the Dict
    c1 = Clustering(v1)
    cs = map(Clustering, vs)
    return Clustering(c1, cs...)
end

# Pass-through constructor: if already a Clustering, return as-is
Clustering(c::Clustering) = c

"""
    Clustering(c1::Clustering, cs::Clustering...)

Merge multiple Clustering objects into a single Clustering (multi-way clustering).

Creates a unique group for each combination of group assignments across the
input Clustering objects.

# Example
```julia
firms = Clustering([1, 1, 2, 2])
years = Clustering([1, 2, 1, 2])
cl = Clustering(firms, years)
nclusters(cl)  # 4 (one for each firm-year combination)
```
"""
function Clustering(c1::Clustering, cs::Clustering...)
    n = length(c1.groups)
    # Verify all Clustering objects have the same length
    for c in cs
        length(c.groups) == n ||
            throw(DimensionMismatch("All Clustering objects must have the same length"))
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

"""
    groups(c::Clustering) -> Vector{Int}

Return the group code of each observation, an integer in `1:nclusters(c)`.
"""
groups(c::Clustering) = c.groups

Base.length(c::Clustering) = length(c.groups)

# Iteration support (for unique() to work)
Base.iterate(c::Clustering) = iterate(c.groups)
Base.iterate(c::Clustering, state) = iterate(c.groups, state)
Base.eltype(::Type{Clustering}) = Int
