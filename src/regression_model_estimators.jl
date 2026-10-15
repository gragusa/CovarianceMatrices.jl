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

# Regression-model protocol. `vcov(k, model)` and `aVar(k, model)` on a
# `RegressionModel` use `coef`, `modelmatrix`, `bread`, `momentresiduals`, `mask`,
# `leverage` (HC2–HC5) and `weights` (CR2/CR3).

"""
    bread(model::RegressionModel) -> AbstractMatrix

Return the `k × k` bread of the sandwich, scaled so that
`vcov(k, model) = n · bread(model) · aVar(k, model) · bread(model)` with `n` the number of
rows of `modelmatrix(model)`. For least squares this is `inv(X'X)`; for a GLM it is
`inv(X'WX)` times the dispersion. Rows and columns of non-estimable coefficients (see
[`mask`](@ref CovarianceMatrices.mask)) are ignored. Required: there is no fallback.
"""
function bread end

"""
    momentresiduals(model::RegressionModel) -> AbstractVector

Return the per-observation residuals `u` for which `modelmatrix(model) .* u` is the
matrix of score (moment) contributions. For unweighted least squares these are the
ordinary residuals; with weights `w` they are `w .* residuals(model)`; for a GLM they are
the working residuals times the working weights, divided by the dispersion.

The fallback calls `StatsAPI.residuals(model)`, which is correct only for unweighted least
squares. Models that define the internal `_residuals` instead are still supported.
"""
momentresiduals(m) = _residuals(m)
_residuals(m) = StatsAPI.residuals(m)

"""
    mask(model::RegressionModel) -> AbstractVector{Bool}

Return a vector with one entry per coefficient, `true` for estimable coefficients and
`false` for coefficients dropped because the design is rank deficient. The variance of a
dropped coefficient is reported as `NaN`. The fallback marks every coefficient estimable.
"""
mask(m) = trues(size(modelmatrix(m), 2))

"""
    leverage(model::RegressionModel) -> AbstractVector

Return the diagonal of the hat matrix, one entry per observation. Needed only by the
leverage-based estimators `HC2`–`HC5` (`HR2`–`HR5`). The fallback calls
`StatsAPI.leverage(model)`.
"""
leverage(m) = StatsAPI.leverage(m)

numobs(m) = size(modelmatrix(m), 1)

@noinline residual_adjustment(k::HAC, r::RegressionModel) = 1.0
@noinline residual_adjustment(k::EWC, r::RegressionModel) = 1.0
@noinline residual_adjustment(k::DriscollKraay, r::RegressionModel) = 1.0
# VARHAC and the smoothed-moment estimators are full long-run-variance estimators
# that need no per-observation adjustment (mirrors HAC/EWC above). Without these
# methods `vcov(VARHAC(), model)` / `vcov(UniformSmoother(m), model)` would fail
# with a MethodError.
@noinline residual_adjustment(k::VARHAC, r::RegressionModel) = 1.0
@noinline residual_adjustment(k::MomentSmoother, r::RegressionModel) = 1.0

# `Uncorrelated` behaves like HC0/HR0 on a fitted model (White's estimator).
@noinline residual_adjustment(k::Uncorrelated, r::RegressionModel) = 1.0

# HC0/HR0: No adjustment
@noinline residual_adjustment(k::HR0, r::RegressionModel) = 1.0

# HC1/HR1: Degrees of freedom adjustment
@noinline function residual_adjustment(k::HR1, r::RegressionModel)
    n = numobs(r)
    dof = _residual_dof(n, length(coef(r)))
    return √(n / dof)
end

# Degrees-of-freedom corrections divide by `n - k` and by `G - 1`; a nonpositive
# divisor would produce an infinite or complex variance.
function _residual_dof(n, k)
    n > k || throw(ArgumentError(
        "the degrees-of-freedom correction needs more observations than coefficients, got n = $n and k = $k"))
    return n - k
end

function _cluster_dof(G)
    G > 1 || throw(ArgumentError(
        "the CR1 degrees-of-freedom correction needs at least 2 clusters, got $G"))
    return G - 1
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

    - CR0, Cluster: No adjustment (returns 1.0)
    - CR1: DOF adjustment √((N-1)/(N-K)) * (G/(G-1))
    - CR2:
    - CR3:
"""

#residual_adjustment(k::CR, m::GLMTableModel) = residual_adjustment(k, m.model)

function residual_adjustment(k::Union{CR0, Cluster}, m::RegressionModel)
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(k.g)))
    [1 for x in combs]
end

function residual_adjustment(k::CR1, m::RegressionModel)
    G = total_num_clusters(k)
    N = numobs(m)
    K = length(coef(m))
    dof = _residual_dof(N, K)
    map(g -> sqrt((N - 1) / dof) * (g / _cluster_dof(g)), G)
end

# Leverage transform dispatch for CR2/CR3
_leverage_transform(::CR2, M) = Symmetric(M)^(-1 / 2)
_leverage_transform(::CR3, M) = inv(Symmetric(M))

function _residual_adjustment_cr(k::Union{CR2, CR3}, m::RegressionModel)
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
            blocks[gc] = _leverage_transform(k, I - tmp)
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
                g = Clustering((f[i] for i in c)...)
            end
            cluster_indices = precompute_cluster_indices(g)
            blocks = Vector{Matrix{eltype(X)}}(undef, g.ngroups)
            @inbounds for gc in 1:g.ngroups
                ind = cluster_indices[gc]
                Xg = view(X, ind, :)
                tmp = Xg * XX * Xg'
                !isempty(wts) && (tmp .*= view(wts, ind)')
                blocks[gc] = _leverage_transform(k, I - tmp)
            end
            BlockDiagonal(blocks)
        end
    end
end

residual_adjustment(k::CR2, m::RegressionModel) = _residual_adjustment_cr(k, m)
residual_adjustment(k::CR3, m::RegressionModel) = _residual_adjustment_cr(k, m)

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
        scaleby::Union{Nothing, Real} = nothing,
        kwargs...)
    _check_resolved(k)
    # Get moment matrix with residual adjustment
    a = residual_adjustment(k, m)
    X = modelmatrix(m)
    u = momentresiduals(m)
    if length(a) == 1
        M = X .* (a[1] * u)
    else
        M = X .* (a .* u)
    end
    # Handle rank deficiency
    midx = mask(m)
    Mm = sum(midx) == size(M, 2) ? M : M[:, midx]
    # Bandwidth selection weights come from the model matrix, not from the moment
    # matrix: they must give the intercept weight 0, and the intercept column of
    # the moment matrix is not constant.
    kw = _kernelweights(k, X)
    kw === nothing || (kw = kw[midx])
    return _aVar(k, Mm; demean = demean, prewhite = prewhite, scale = scale,
        scaleby = scaleby, weights = kw)
end

function aVar(
        k::CR,
        m::RegressionModel;
        scale = true,
        scaleby::Union{Nothing, Real} = nothing,
        prewhite::Bool = false,
        kwargs...)
    _check_resolved(k)
    _checkscale(scale)
    _checkprewhite(k, prewhite)
    H = residual_adjustment(k, m)
    X = modelmatrix(m)
    u = momentresiduals(m)
    M = map(h->X .* (h*u), H)
    V = avar_tuple(k, M)
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(k.g)))
    Σ = mapreduce(+, zip(combs, V)) do (c, v)
        (-1)^(length(c) - 1)*v
    end
    scalevar!(Σ, scale, scaleby, numobs(m))
    return CovarianceMatrix(Σ, k, (; nclusters = nclusters(k)))
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

    return CovarianceMatrix(Vo, k, information(A))
end

# Finite-sample correction factor for Driscoll-Kraay, matching the scalar
# `type` options of plm::vcovSCC. `n` is the number of observations, `k` the
# number of estimated coefficients, `T` the number of time periods.
function _dk_correction(correction::Symbol, n::Int, k::Int, T::Int)
    correction === :HC0 && return 1.0
    dof = _residual_dof(n, k)
    correction === :HC1 && return n / dof
    T > 1 || throw(ArgumentError(
        "the Driscoll-Kraay :sss correction needs at least 2 time periods, got $T"))
    return (n - 1) / dof * T / (T - 1)
end

# The correction is a `vcov` factor, so the meat on the model path is uncorrected;
# the result still records the estimator as given.
function aVar(k::DriscollKraay, m::RegressionModel; kwargs...)
    k0 = DriscollKraay{typeof(k.K), typeof(k.tis), typeof(k.iis)}(k.K, k.tis, k.iis)
    V = invoke(aVar, Tuple{AbstractAsymptoticVarianceEstimator, RegressionModel}, k0, m;
        kwargs...)
    return CovarianceMatrix(parent(V), k, information(V))
end

"""
    vcov(k::DriscollKraay, model::RegressionModel; dofadjust = true)

Driscoll-Kraay variance-covariance matrix for a `RegressionModel`.

With bread `B = (X'X)^-1` and `A = aVar(k, model)`, the sandwich is
`V = c · n · B · A · B`: the generic sandwich times the finite-sample factor `c`
selected by the `correction` of `k` (`n` observations, `p` estimable
coefficients, `T` time periods). The options reproduce the scalar `type` options of
`plm::vcovSCC`:

- `:HC0` (default): `c = 1`, no correction.
- `:HC1`: `c = n / (n - p)`.
- `:sss`: `c = (n - 1) / (n - p) · T / (T - 1)`, the Stata-style small-sample
  correction.

`dofadjust = false` leaves out `c`. The leverage-based `:HC2`/`:HC3`/`:HC4` options of
`plm::vcovSCC` are not supported. The keyword form `vcov(k, model; type = :HC1)` is
deprecated in favor of `DriscollKraay(...; correction = :HC1)`.
"""
function StatsAPI.vcov(k::DriscollKraay, m::RegressionModel; type = nothing, kwargs...)
    if type !== nothing
        Base.depwarn(
            "`vcov(k::DriscollKraay, model; type = $(repr(type)))` is deprecated; " *
            "construct the estimator with `DriscollKraay(...; correction = $(repr(type)))`.",
            :vcov)
        k.correction === :HC0 || k.correction === type ||
            throw(ArgumentError(
                "`type = $(repr(type))` conflicts with the estimator's correction $(repr(k.correction))"))
        k = DriscollKraay{typeof(k.K), typeof(k.tis), typeof(k.iis)}(k.K, k.tis, k.iis, type)
    end
    return invoke(StatsAPI.vcov,
        Tuple{AbstractAsymptoticVarianceEstimator, RegressionModel}, k, m; kwargs...)
end

function dofcorrect!(V, k::DriscollKraay, m::RegressionModel)
    c = _dk_correction(k.correction, numobs(m), sum(mask(m)), k.tis.ngroups)
    return rmul!(V, c)
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
    n = numobs(m)
    dof = _residual_dof(n, length(coef(m)))
    rmul!(V, n / dof)
end

#=========
CachedCR on a RegressionModel
=========#

function CachedCR(k::CR, m::RegressionModel)
    _check_resolved(k)
    X = modelmatrix(m)
    XX = bread(m)
    grouped_arrays, cluster_indices, signs = _cluster_combinations(k.g)
    H = _compute_leverage_adjustments(k, X, XX, weights(m), grouped_arrays, cluster_indices)
    cache = CRModelCache{eltype(X), typeof(H)}(
        grouped_arrays, cluster_indices, signs, XX, H, size(X, 2), numobs(m))
    return CachedCR(k, cache)
end

# Dispatch on CR type to compute appropriate leverage adjustments
function _compute_leverage_adjustments(
        k::Union{CR0, Cluster}, X, XX, wts, grouped_arrays, cluster_indices)
    # CR0: No adjustment, just return vector of 1s (one per combination)
    [1 for _ in grouped_arrays]
end

function _compute_leverage_adjustments(k::CR1, X, XX, wts, grouped_arrays, cluster_indices)
    # CR1: DOF adjustment - scalar per combination
    N = size(X, 1)
    K = size(X, 2)
    dof = _residual_dof(N, K)
    map(g -> sqrt((N - 1) / dof) * (g.ngroups / _cluster_dof(g.ngroups)), grouped_arrays)
end

function _compute_leverage_adjustments(
        k::Union{CR2, CR3}, X, XX, wts, grouped_arrays, cluster_indices)
    # Precompute BlockDiagonal leverage adjustments using type-dispatched transform
    T = eltype(X)
    map(zip(grouped_arrays, cluster_indices)) do (g, indices)
        blocks = Vector{Matrix{T}}(undef, g.ngroups)
        @inbounds for gc in 1:g.ngroups
            ind = indices[gc]
            Xg = view(X, ind, :)
            tmp = Xg * XX * Xg'
            !isempty(wts) && (tmp .*= view(wts, ind)')
            blocks[gc] = _leverage_transform(k, I - tmp)
        end
        BlockDiagonal(blocks)
    end
end

residual_adjustment(k::CachedCR, m::RegressionModel) = residual_adjustment(k.estimator, m)

# The cached adjustments belong to the model the cache was built from.
function residual_adjustment(k::CachedCR{<:CR, <:CRModelCache}, m::RegressionModel)
    c = k.cache
    n, p = numobs(m), size(modelmatrix(m), 2)
    (n, p) == (c.nobs, c.ncols) || throw(DimensionMismatch(
        "the model has $n observations and $p coefficients, but the cache was built " *
        "for $(c.nobs) and $(c.ncols)"))
    return c.leverage_adjustments
end

"""
    aVar(k::CachedCR{<:CR, <:CRModelCache}, m::RegressionModel; scale=true, scaleby=nothing, kwargs...)

Compute asymptotic variance using cached leverage adjustments.
Only the residual-dependent parts are computed; leverage adjustments are reused from cache.
"""
function aVar(
        k::CachedCR{<:CR, <:CRModelCache},
        m::RegressionModel;
        scale = true,
        scaleby::Union{Nothing, Real} = nothing,
        prewhite::Bool = false,
        kwargs...)
    _checkscale(scale)
    _checkprewhite(k, prewhite)
    cache = k.cache
    H = residual_adjustment(k, m)
    X = modelmatrix(m)
    u = momentresiduals(m)

    # Compute moment matrices using cached leverage adjustments
    M = map(h -> X .* (h * u), H)

    # Use cached cluster indices for aggregation
    V = _avar_cached(cache, M)

    # Apply inclusion-exclusion
    Σ = mapreduce(+, zip(cache.signs, V)) do (sign, v)
        sign * v
    end

    scalevar!(Σ, scale, scaleby, numobs(m))
    return CovarianceMatrix(Σ, k, (; nclusters = nclusters(k)))
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
