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

function residual_adjustment(k::CR0, m::RegressionModel)
    [1 for x in combinations(1:length(k.g))]
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
    Hᵧᵧ = Vector{Matrix{eltype(X)}}(undef, 0)
    f = k.g
    map(combinations(1:length(f))) do c
        begin
            if length(c) == 1
                g = GroupedArray(f[c[1]])
            else
                g = GroupedArray((f[i] for i in c)...; sort = nothing)
            end
            BlockDiagonal(map(gg -> begin
                ind = findall(x -> x == gg, g)
                Xg = view(X, ind, :)
                tmp = (Xg * XX * Xg')
                !isempty(wts) && (tmp .*= view(wts, ind)')
                Symmetric(I - tmp)^(-1 / 2)
            end, unique(g)))
        end
    end
end

function residual_adjustment(k::CR3, m::RegressionModel)
    X = modelmatrix(m)
    XX = bread(m)
    wts = weights(m)
    Hᵧᵧ = Vector{Matrix{eltype(X)}}(undef, 0)
    f = k.g
    map(combinations(1:length(f))) do c
        begin
            if length(c) == 1
                g = GroupedArray(f[c[1]])
            else
                g = GroupedArray((f[i] for i in c)...; sort = nothing)
            end
            BlockDiagonal(map(gg -> begin
                ind = findall(x -> x == gg, g)
                Xg = view(X, ind, :)
                tmp = (Xg * XX * Xg')
                !isempty(wts) && (tmp .*= view(wts, ind)')
                inv(Symmetric(I - tmp))
            end, unique(g)))
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
    M = map(h->X.*(h*u), H)
    V = avar_tuple(k, M)
    Σ = mapreduce(+, zip(combinations(1:length(k.g)), V)) do (c, v)
        (-1)^(length(c) - 1)*v
    end
    scale ? rdiv!(Σ, numobs(m)) : Σ
end




unlock_kernel!(k::AbstractAsymptoticVarianceEstimator) = return false
function unlock_kernel!(k::HAC{T}) where {T<:Union{NeweyWest, Andrews}}
    wlock = k.wlock[1]
    k.wlock .= true
    return wlock
end

lock_kernel!(k::AbstractAsymptoticVarianceEstimator, wlock) = nothing
function lock_kernel!(k::HAC{T}, wlock) where {T<:Union{NeweyWest, Andrews}}
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