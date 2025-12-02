# Implementation Notes

This document describes key implementation details and optimizations in CovarianceMatrices.jl.

## CachedCR: Optimized Cluster-Robust Variance

The `CachedCR` wrapper provides significant performance improvements for repeated cluster-robust variance calculations with the same cluster structure.

### The Problem

The standard cluster-robust variance calculation in `clusterize` uses a **scatter-add** pattern:

```julia
function clusterize(X::Matrix, g::GroupedArray)
    X2 = zeros(eltype(X), g.ngroups, size(X, 2))
    for j in axes(X, 2)
        @inbounds @simd for i in axes(X, 1)
            X2[g.groups[i], j] += X[i, j]  # Random write to X2
        end
    end
    return X2' * X2
end
```

This pattern has poor cache performance because:
1. **Random write access**: `X2[g.groups[i], j]` writes to unpredictable memory locations
2. **Poor vectorization**: The compiler cannot effectively vectorize due to potential write conflicts
3. **Cache thrashing**: Non-sequential memory access patterns cause frequent cache misses

Profiling showed that the scatter-add loop consumes **~99.7%** of the total computation time.

### The Solution: Gather-Based Aggregation

`CachedCR` uses a **gather-sum** pattern with precomputed cluster indices:

```julia
function clusterize_gather_add!(X2, S, X, indices, sign)
    ngroups = length(indices)
    ncols = size(X, 2)

    # Gather-sum: sequential reads, sequential writes
    @inbounds for gc in 1:ngroups
        idx = indices[gc]  # Precomputed: which rows belong to cluster gc
        for j in 1:ncols
            s = zero(eltype(X))
            @simd for k in eachindex(idx)
                s += X[idx[k], j]  # Sequential accumulation
            end
            X2[gc, j] = s  # Sequential write
        end
    end

    # Efficient symmetric rank-k update
    LinearAlgebra.BLAS.syrk!('U', 'T', sign, X2, one(T), S)
    # ... symmetrize S ...
end
```

Key improvements:
1. **Sequential writes**: Each cluster sum is written once to a known location
2. **Better vectorization**: The inner accumulation loop can be SIMD-vectorized
3. **Cache-friendly reads**: While reads from X are still indirect, the accumulation pattern is more predictable
4. **BLAS acceleration**: Uses optimized `syrk!` for the matrix multiply

### Cache Structure

The `CRCache` struct precomputes and stores:

```julia
struct CRCache{T<:Real}
    X2_buffers::Vector{Matrix{T}}            # Preallocated aggregation buffers
    S_buffer::Matrix{T}                       # Preallocated output buffer
    grouped_arrays::Vector{GroupedArray}      # Precomputed for each combination
    cluster_indices::Vector{Vector{Vector{Int}}}  # [combination][cluster] -> obs indices
    signs::Vector{Int}                        # Inclusion-exclusion signs
    ncols::Int
end
```

For multi-way clustering (e.g., firm × year), the cache precomputes:
- GroupedArrays for each subset of clustering variables
- Cluster indices for each GroupedArray
- Signs for the inclusion-exclusion formula

### Performance Results

| Scenario | Standard | CachedCR | Speedup |
|----------|----------|----------|---------|
| Single cluster (10k obs, 100 clusters) | 420 μs | 105 μs | **4.0x** |
| Two-way clustering (2k obs) | 178 μs | 45 μs | **3.9x** |
| Wild bootstrap (100 iterations) | 56 ms | 22 ms | **2.6x** |

### Trade-offs

1. **Memory overhead**: Cache stores precomputed indices (~8 bytes per observation per combination)
2. **Setup cost**: Building the cache has one-time overhead
3. **AD incompatibility**: In-place operations break automatic differentiation
4. **Fixed dimensions**: Cache must be rebuilt if column count changes

### When to Use

Use `CachedCR` when:
- Same cluster structure is used repeatedly (bootstrap, Monte Carlo)
- Performance is critical
- AD is not required

Use standard CR estimators when:
- Single calculation or few repetitions
- AD compatibility needed
- Memory is constrained

## Single-Cluster Fast Path

For single-cluster CR estimators, the code skips the combinations loop entirely:

```julia
function avar(k::CR, X; kwargs...)
    f = k.g

    # Fast path: single cluster variable
    if length(f) == 1
        return clusterize(parent(X), f[1])  # Direct call, no combinations
    end

    # Multi-way: use inclusion-exclusion
    S = zeros(eltype(X), (size(X, 2), size(X, 2)))
    for c in combinations(1:length(f))
        # ... compute each combination ...
    end
    return S
end
```

This avoids:
- Tuple wrapping overhead
- `combinations(1:1)` iterator creation
- Unnecessary branching in the loop

## BLAS Optimization

The cached implementation uses BLAS level-3 operations for the symmetric matrix multiply:

```julia
# Instead of: S += sign * X2' * X2
LinearAlgebra.BLAS.syrk!('U', 'T', T(sign), X2, one(T), S)
```

Benefits:
- Exploits matrix symmetry (only computes upper triangle)
- Uses optimized BLAS routines (OpenBLAS, MKL, Apple Accelerate)
- Avoids temporary allocation for `X2' * X2`

## File Organization

Key implementation files:

| File | Purpose |
|------|---------|
| `src/types.jl` | Type definitions for CRCache, CachedCR |
| `src/CR.jl` | Cluster-robust variance implementation |
| `src/aVar.jl` | Generic variance API |
| `benchmark/benchmark_cached_cr.jl` | Performance benchmarks |

## Testing

The implementation is validated against R's sandwich package:

```julia
@testset "Cluster Robust - R sandwich validation" begin
    # Simulated clustered data with known structure
    # Reference values from: vcovCL(lm_1, cluster = df$cl, type = "HC0/1/2/3")
    # Tests CR0, CR1, CR2, CR3 for both weighted and unweighted regression
end
```

See `test/test_core.jl` for the full test suite.
