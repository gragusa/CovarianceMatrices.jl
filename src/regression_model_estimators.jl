"""
Generic variance estimators for RegressionModel types.

This file implements generic HC, HAC, and CR variance estimators that work
with RegressionModel.

"""

"""
    residual_adjustment(estimator, model)

Compute the residual adjustment factor for heteroskedasticity-robust estimators.

Returns a scalar or vector that multiplies the moment matrix rows to implement
different HC/HR adjustments. The generic implementations use the protocol methods
and work with any RegressionModel.

# HC/HR Estimators

- `HC0/HR0`: No adjustment (returns 1.0)
- `HC1/HR1`: DOF adjustment √(n/(n-p))
- `HC2/HR2`: Leverage adjustment 1/√(1-h)
- `HC3/HR3`: Squared leverage adjustment 1/(1-h)
- `HC4/HR4`: Adaptive leverage adjustment with cutoff
- `HC4m/HR4m`: Modified HC4 with different cutoff
- `HC5/HR5`: Maximum leverage adjustment

# HAC Estimators

- Returns 1.0 (no per-observation adjustment for HAC)

# CR Estimators

- `CR0/CR1`: Cluster-level residual aggregation
- `CR2/CR3`: Cluster-level leverage adjustments

# Arguments
- `estimator`: The variance estimator type (HC0, HAC, CR1, etc.)
- `model`: A RegressionModel implementing the protocol

# Returns
- Scalar or vector of adjustment factors
"""
function residual_adjustment end
function numobs end
function mask end
function leverage end
function _residuals end
function bread end

@noinline residual_adjustment(k::HAC, r::RegressionModel) = 1.0
@noinline residual_adjustment(k::EWC, r::RegressionModel) = 1.0

# HC0/HR0: No adjustment
@noinline residual_adjustment(k::HR0, r::RegressionModel) = 1.0

# HC1/HR1: Degrees of freedom adjustment
@noinline function residual_adjustment(k::HR1, r::RegressionModel)
    n = numobs(r)
    #dof = dof_residual(r)
    dof = n - length(coef(r))
    return √(n / dof)
end

# HC2/HR2: Leverage adjustment
@noinline function residual_adjustment(k::HR2, r::RegressionModel)
    h = leverage(r)
    return 1.0 ./ sqrt.(1 .- h)
end

# HC3/HR3: Squared leverage adjustment
@noinline function residual_adjustment(k::HR3, r::RegressionModel)
    h = leverage(r)
    return 1.0 ./ (1 .- h)
end

# HC4/HR4: Adaptive leverage adjustment
@noinline function residual_adjustment(k::HR4, r::RegressionModel)
    n = numobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    delta = similar(h)
    @inbounds for j in eachindex(h)
        delta[j] = min(4.0, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta[j] / 2)
    end
    return h
end

# HC4m/HR4m: Modified HC4
@noinline function residual_adjustment(k::HR4m, r::RegressionModel)
    n = numobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    delta = similar(h)
    @inbounds for j in eachindex(h)
        delta[j] = min(1, n * h[j] / p) + min(1.5, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta[j] / 2)
    end
    return h
end

# HC5/HR5: Maximum leverage adjustment
@noinline function residual_adjustment(k::HR5, r::RegressionModel)
    n = numobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    mx = max(n * 0.7 * maximum(h) / p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n * h[j] / p, mx)
        h[j] = 1 / (1 - h[j])^(alpha / 4)
    end
    return h
end

"""

Compute residual adjustment for basic cluster-robust estimators.

    - CR0: No adjustment (returns 1.0)
    - CR1: DOF adjustment √((N-1)/(N-K)) * (G/(G-1))
    - CR2:
    - CR3:
"""

#residual_adjustment(k::CR, m::GLMTableModel) = residual_adjustment(k, m.model)

"""
    precompute_cluster_indices(g::Clustering)

Compute observation indices for each cluster in a single O(n) pass.
Returns `Vector{Vector{Int}}` where `result[cluster_id]` contains all row indices for that cluster.

This is much faster than repeated `findall()` calls which would be O(n*G).
"""
function precompute_cluster_indices(g::Clustering)
    indices = [Int[] for _ in 1:g.ngroups]
    @inbounds for i in eachindex(g.groups)
        push!(indices[g.groups[i]], i)
    end
    return indices
end

function residual_adjustment(k::CR0, m::RegressionModel)
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(k.g)))
    [1 for x in combs]
end

function residual_adjustment(k::CR1, m::RegressionModel)
    G = total_num_clusters(k)
    N = numobs(m)
    K = length(coef(m))
    map(g -> sqrt.((N - 1) / (N - K)) * (g / (g - 1)), G)
end

function residual_adjustment(k::CR2, m::RegressionModel)
    X = modelmatrix(m)
    XX = bread(m)
    wts = weights(m)
    f = k.g

    # Fast path: single cluster variable
    if length(f) == 1
        g = f[1]  # Already a Clustering
        cluster_indices = precompute_cluster_indices(g)
        blocks = Vector{Matrix{eltype(X)}}(undef, g.ngroups)
        @inbounds for gc in 1:g.ngroups
            ind = cluster_indices[gc]
            Xg = view(X, ind, :)
            tmp = Xg * XX * Xg'
            !isempty(wts) && (tmp .*= view(wts, ind)')
            blocks[gc] = Symmetric(I - tmp)^(-1 / 2)
        end
        return [BlockDiagonal(blocks)]
    end

    # Multi-way clustering: use inclusion-exclusion
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(f)))
    map(combs) do c
        begin
            if length(c) == 1
                g = Clustering(f[c[1]])
            else
                g = Clustering((f[i] for i in c)...; sort = nothing)
            end
            cluster_indices = precompute_cluster_indices(g)
            blocks = Vector{Matrix{eltype(X)}}(undef, g.ngroups)
            @inbounds for gc in 1:g.ngroups
                ind = cluster_indices[gc]
                Xg = view(X, ind, :)
                tmp = Xg * XX * Xg'
                !isempty(wts) && (tmp .*= view(wts, ind)')
                blocks[gc] = Symmetric(I - tmp)^(-1 / 2)
            end
            BlockDiagonal(blocks)
        end
    end
end

function residual_adjustment(k::CR3, m::RegressionModel)
    X = modelmatrix(m)
    XX = bread(m)
    wts = weights(m)
    f = k.g

    # Fast path: single cluster variable
    if length(f) == 1
        g = f[1]  # Already a Clustering
        cluster_indices = precompute_cluster_indices(g)
        blocks = Vector{Matrix{eltype(X)}}(undef, g.ngroups)
        @inbounds for gc in 1:g.ngroups
            ind = cluster_indices[gc]
            Xg = view(X, ind, :)
            tmp = Xg * XX * Xg'
            !isempty(wts) && (tmp .*= view(wts, ind)')
            blocks[gc] = inv(Symmetric(I - tmp))
        end
        return [BlockDiagonal(blocks)]
    end

    # Multi-way clustering: use inclusion-exclusion
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(f)))
    map(combs) do c
        begin
            if length(c) == 1
                g = Clustering(f[c[1]])
            else
                g = Clustering((f[i] for i in c)...; sort = nothing)
            end
            cluster_indices = precompute_cluster_indices(g)
            blocks = Vector{Matrix{eltype(X)}}(undef, g.ngroups)
            @inbounds for gc in 1:g.ngroups
                ind = cluster_indices[gc]
                Xg = view(X, ind, :)
                tmp = Xg * XX * Xg'
                !isempty(wts) && (tmp .*= view(wts, ind)')
                blocks[gc] = inv(Symmetric(I - tmp))
            end
            BlockDiagonal(blocks)
        end
    end
end

"""
    aVar(estimator, model::RegressionModel; kwargs...)

Compute the asymptotic variance matrix for the parameters of a `RegressionModel`.

This is the generic implementation that works with any model implementing the RegressionModel protocol.

"""
function aVar(
        k::AbstractAsymptoticVarianceEstimator,
        m::StatsBase.RegressionModel;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...)
    # Set kernel weights if needed (for HAC with automatic bandwidth)
    setkernelweights!(k, m)
    # Lock weights to prevent changes
    wlock = unlock_kernel!(k)
    # Get moment matrix with residual adjustment
    a = residual_adjustment(k, m)
    X = modelmatrix(m)
    u = _residuals(m)
    if length(a) == 1
        M = X .* (a[1] * u)
    else
        M = X .* (a .* u)
    end
    # Handle rank deficiency
    midx = mask(m)
    Σ = if sum(midx) == size(M, 2)
        aVar(k, M; demean = demean, prewhite = prewhite, scale = scale)
    else
        aVar(k, M[:, midx]; demean = demean, prewhite = prewhite, scale = scale)
    end
    ## Reset lock
    lock_kernel!(k, wlock)
    return Σ
end

function aVar(
        k::CR,
        m::RegressionModel;
        scale = true,
        kwargs...)
    H = residual_adjustment(k, m)
    X = modelmatrix(m)
    u = _residuals(m)
    M = map(h->X .* (h*u), H)
    V = avar_tuple(k, M)
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(k.g)))
    Σ = mapreduce(+, zip(combs, V)) do (c, v)
        (-1)^(length(c) - 1)*v
    end
    scale ? rdiv!(Σ, numobs(m)) : Σ
end

unlock_kernel!(k::AbstractAsymptoticVarianceEstimator) = return false
function unlock_kernel!(k::HAC{T}) where {T <: Union{NeweyWest, Andrews}}
    wlock = k.wlock[1]
    k.wlock .= true
    return wlock
end

lock_kernel!(k::AbstractAsymptoticVarianceEstimator, wlock) = nothing
function lock_kernel!(k::HAC{T}, wlock) where {T <: Union{NeweyWest, Andrews}}
    k.wlock .= wlock
end

"""
    vcov(estimator, model; dofadjust=true, kwargs...)

Compute the sandwich variance-covariance matrix for a RegressionModel.

"""
function StatsAPI.vcov(
        k::AbstractAsymptoticVarianceEstimator,
        m::RegressionModel;
        dofadjust = true,
        kwargs...
)
    # Compute meat
    A = aVar(k, m; kwargs...)

    # Get bread and scale
    n = numobs(m)
    B = bread(m)
    p = size(B, 2)

    # Handle rank deficiency
    midx = mask(m)
    Bm = sum(midx) < p ? B[midx, midx] : B

    # Sandwich: V = n * B * A * B
    V = n .* Bm * A * Bm

    # Reconstruct full matrix with NaN for non-estimable parameters
    if sum(midx) < p
        Vo = similar(A, (p, p))
        Vo[midx, midx] .= V
        Vo[.!midx, :] .= NaN
        Vo[:, .!midx] .= NaN
    else
        Vo = V
    end

    # Apply DOF correction if requested
    dofadjust && dofcorrect!(Vo, k, m)

    return Vo
end

"""
    stderror(estimator, model; kwargs...)

Compute robust standard errors for a RegressionModel.

"""
function StatsAPI.stderror(k::AbstractAsymptoticVarianceEstimator, m; kwargs...)
    sqrt.(diag(vcov(k, m; kwargs...)))
end

"""
    dofcorrect!(V, estimator, model)

Apply degrees-of-freedom correction to variance matrix.

## Note

Only `HAC` estimators apply DOF correction by default. Other estimators
(`HC`, `CR`) incorporate DOF adjustments in their residual adjustments.

"""
dofcorrect!(V, k::AbstractAsymptoticVarianceEstimator, m) = nothing

function dofcorrect!(V, k::HAC, m::RegressionModel)
    k = length(coef(m))
    n = numobs(m)
    dof = n - k
    rmul!(V, n / dof)
end

#=========
CachedCRModel - Cached CR for RegressionModel/GLM
=========#

"""
    CachedCRModel(k::CR, m::RegressionModel)

Create a cached cluster-robust estimator by precomputing leverage adjustments
from the model. The cached leverage adjustments can then be reused for repeated
variance calculations with different residuals (e.g., wild bootstrap).

# Arguments
- `k`: A cluster-robust estimator (CR0, CR1, CR2, or CR3)
- `m`: A fitted RegressionModel (e.g., from GLM.lm)

# Returns
- `CachedCRModel`: Wrapper with precomputed leverage adjustments

# Example
```julia
using CovarianceMatrices, GLM, DataFrames

df = DataFrame(y=randn(1000), x1=randn(1000), cl=repeat(1:50, 20))
model = lm(@formula(y ~ x1), df)

# Create cached estimator (one-time cost)
cached = CachedCRModel(CR2(df.cl), model)

# Fast variance calculations
V1 = vcov(cached, model)  # Uses cached leverage adjustments
```

# Performance
For CR2/CR3 with single-cluster, expect 10-50x speedup in bootstrap scenarios.
"""
function CachedCRModel(k::CR, m::RegressionModel)
    X = modelmatrix(m)
    XX = bread(m)
    wts = weights(m)
    f = k.g
    T = eltype(X)
    n = numobs(m)
    p = size(X, 2)

    # Precompute Clusterings, cluster indices, and signs
    grouped_arrays = Clustering[]
    cluster_indices = Vector{Vector{Int}}[]
    signs = Int[]

    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(f)))
    for c in combs
        if length(c) == 1
            g = Clustering(f[c[1]])
        else
            g = Clustering((f[i] for i in c)...; sort = nothing)
        end
        push!(grouped_arrays, g)
        push!(signs, (-1)^(length(c) - 1))
        push!(cluster_indices, precompute_cluster_indices(g))
    end

    # Compute leverage adjustments based on estimator type
    H = _compute_leverage_adjustments(k, X, XX, wts, grouped_arrays, cluster_indices)

    cache = CRModelCache{T, typeof(H)}(
        grouped_arrays, cluster_indices, signs, XX, H, p, n
    )
    return CachedCRModel(k, cache)
end

# Dispatch on CR type to compute appropriate leverage adjustments
function _compute_leverage_adjustments(k::CR0, X, XX, wts, grouped_arrays, cluster_indices)
    # CR0: No adjustment, just return vector of 1s (one per combination)
    [1 for _ in grouped_arrays]
end

function _compute_leverage_adjustments(k::CR1, X, XX, wts, grouped_arrays, cluster_indices)
    # CR1: DOF adjustment - scalar per combination
    N = size(X, 1)
    K = size(X, 2)
    map(g -> sqrt((N - 1) / (N - K)) * (g.ngroups / (g.ngroups - 1)), grouped_arrays)
end

function _compute_leverage_adjustments(k::CR2, X, XX, wts, grouped_arrays, cluster_indices)
    # CR2: Precompute BlockDiagonal leverage adjustments
    T = eltype(X)
    map(zip(grouped_arrays, cluster_indices)) do (g, indices)
        blocks = Vector{Matrix{T}}(undef, g.ngroups)
        @inbounds for gc in 1:g.ngroups
            ind = indices[gc]
            Xg = view(X, ind, :)
            tmp = Xg * XX * Xg'
            !isempty(wts) && (tmp .*= view(wts, ind)')
            blocks[gc] = Symmetric(I - tmp)^(-1 / 2)
        end
        BlockDiagonal(blocks)
    end
end

function _compute_leverage_adjustments(k::CR3, X, XX, wts, grouped_arrays, cluster_indices)
    # CR3: Precompute BlockDiagonal leverage adjustments (inverse instead of sqrt)
    T = eltype(X)
    map(zip(grouped_arrays, cluster_indices)) do (g, indices)
        blocks = Vector{Matrix{T}}(undef, g.ngroups)
        @inbounds for gc in 1:g.ngroups
            ind = indices[gc]
            Xg = view(X, ind, :)
            tmp = Xg * XX * Xg'
            !isempty(wts) && (tmp .*= view(wts, ind)')
            blocks[gc] = inv(Symmetric(I - tmp))
        end
        BlockDiagonal(blocks)
    end
end

"""
    residual_adjustment(k::CachedCRModel, m::RegressionModel)

Return cached leverage adjustments. This is O(1) since adjustments are precomputed.
"""
function residual_adjustment(k::CachedCRModel, m::RegressionModel)
    return k.cache.leverage_adjustments
end

"""
    aVar(k::CachedCRModel, m::RegressionModel; scale=true, kwargs...)

Compute asymptotic variance using cached leverage adjustments.
Only the residual-dependent parts are computed; leverage adjustments are reused from cache.
"""
function aVar(
        k::CachedCRModel,
        m::RegressionModel;
        scale = true,
        kwargs...)
    cache = k.cache
    H = cache.leverage_adjustments
    X = modelmatrix(m)
    u = _residuals(m)

    # Compute moment matrices using cached leverage adjustments
    M = map(h -> X .* (h * u), H)

    # Use cached cluster indices for aggregation
    V = _avar_cached(cache, M)

    # Apply inclusion-exclusion
    Σ = mapreduce(+, zip(cache.signs, V)) do (sign, v)
        sign * v
    end

    scale ? rdiv!(Σ, numobs(m)) : Σ
end

"""
    _avar_cached(cache::CRModelCache, M::Vector)

Compute cluster-aggregated variance using cached cluster indices.
"""
function _avar_cached(cache::CRModelCache, M::Vector)
    map(zip(cache.grouped_arrays, cache.cluster_indices, M)) do (g, indices, m)
        T = eltype(m)
        ncols = size(m, 2)
        X2 = zeros(T, g.ngroups, ncols)

        # Gather-based aggregation using precomputed indices
        @inbounds for gc in 1:g.ngroups
            idx = indices[gc]
            for j in 1:ncols
                s = zero(T)
                @simd for k in eachindex(idx)
                    s += m[idx[k], j]
                end
                X2[gc, j] = s
            end
        end

        X2' * X2
    end
end

"""
    bread(k::CachedCRModel)

Return cached bread matrix.
"""
bread(k::CachedCRModel) = k.cache.bread_matrix

"""
    stderror(k::CachedCRModel, m::RegressionModel; kwargs...)

Compute robust standard errors using cached leverage adjustments.
"""
function StatsAPI.stderror(k::CachedCRModel, m; kwargs...)
    sqrt.(diag(vcov(k, m; kwargs...)))
end

"""
    vcov(k::CachedCRModel, m::RegressionModel; dofadjust=true, kwargs...)

Compute variance-covariance matrix using cached leverage adjustments.
"""
function StatsAPI.vcov(
        k::CachedCRModel,
        m::RegressionModel;
        dofadjust = true,
        kwargs...
)
    # Compute meat using cached adjustments
    A = aVar(k, m; kwargs...)

    # Use cached bread
    n = numobs(m)
    B = k.cache.bread_matrix
    p = size(B, 2)

    # Handle rank deficiency
    midx = mask(m)
    Bm = sum(midx) < p ? B[midx, midx] : B

    # Sandwich: V = n * B * A * B
    V = n .* Bm * A * Bm

    # Reconstruct full matrix with NaN for non-estimable parameters
    if sum(midx) < p
        Vo = similar(A, (p, p))
        Vo[midx, midx] .= V
        Vo[.!midx, :] .= NaN
        Vo[:, .!midx] .= NaN
    else
        Vo = V
    end

    # Apply DOF correction if requested
    dofadjust && dofcorrect!(Vo, k.estimator, m)

    return Vo
end
