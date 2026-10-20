"""
Generic variance estimators for RegressionModel types.

This file implements generic HC, HAC, and CR variance estimators that work
with RegressionModel.

"""

"""
    residual_adjustment(k::HR, model::RegressionModel) -> Real or AbstractVector
    residual_adjustment(k, model::RegressionModel) -> Real

Return the factor `a` that multiplies the residuals of `model` before the moments are
formed: `aVar(k, model)` is computed from `modelmatrix(model) .* (a .* u)` with
`u = momentresiduals(model)`. `a` is a scalar or a vector with one entry per
observation.

A package that builds its own moment matrix calls this to apply the finite-sample
adjustments of `HR1`–`HR5`, and then passes the moments to `aVar` with `HR0`, which is
the estimator `aVar` accepts on a matrix:

```julia
a = CovarianceMatrices.residual_adjustment(HC3(), model)
M = modelmatrix(model) .* (a .* CovarianceMatrices.momentresiduals(model))
aVar(HC0(), M)
```

With `n = numobs(model)`, `K = length(coef(model))` and `h = leverage(model)`:

- `HR0`/`HC0`: `1`.
- `HR1`/`HC1`: `√(n / (n - K))`.
- `HR2`/`HC2`: `1 ./ √(1 .- h)`.
- `HR3`/`HC3`: `1 ./ (1 .- h)`.
- `HR4`/`HC4`, `HR4m`/`HC4m`, `HR5`/`HC5`: `(1 .- h) .^ (-δ / 2)` with the exponents
  `δ` of Cribari-Neto (2004), Cribari-Neto and da Silva (2011), and Cribari-Neto et
  al. (2007).
- `HAC` kernels, `EWC`, `VARHAC`, `DriscollKraay` and the moment smoothers: `1`.
  Their degrees-of-freedom correction is a factor of `vcov`, not of the residuals.

The leverage-based adjustments do not modify the vector `leverage(model)` returns, so
a model may return a stored vector. For cluster estimators the return value is
internal.
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
squares.
"""
momentresiduals(m) = StatsAPI.residuals(m)

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

"""
    numobs(model::RegressionModel) -> Integer

Return the number of observations `n` of `model`: the number of rows of its moment
matrix, which divides the variance in `aVar` and enters the finite-sample corrections
(`n / (n - K)` for `HR1` and the long-run estimators, `(n - 1) / (n - K)` for `CR1`).
The fallback is `size(modelmatrix(model), 1)`. A model whose `modelmatrix` is
expensive to form, or whose `StatsAPI.nobs` counts weighted observations, can define
it.
"""
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

# HC4/HR4: Adaptive leverage adjustment. HC4, HC4m and HC5 return new vectors:
# `leverage(r)` may return an array the model stores.
@noinline function residual_adjustment(k::HR4, r::RegressionModel)
    n = numobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    return @. 1 / (1 - h)^(min(4.0, n * h / p) / 2)
end

# HC4m/HR4m: Modified HC4
@noinline function residual_adjustment(k::HR4m, r::RegressionModel)
    n = numobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    return @. 1 / (1 - h)^((min(1, n * h / p) + min(1.5, n * h / p)) / 2)
end

# HC5/HR5: Maximum leverage adjustment
@noinline function residual_adjustment(k::HR5, r::RegressionModel)
    n = numobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    mx = max(n * 0.7 * maximum(h) / p, 4.0)
    return @. 1 / (1 - h)^(min(n * h / p, mx) / 4)
end

"""

Compute residual adjustment for basic cluster-robust estimators.

    - CR0, Cluster: No adjustment (returns 1.0)
    - CR1: DOF adjustment √((N-1)/(N-K) * G/(G-1))
    - CR2: residuals of cluster g multiplied by (I - H_gg)^(-1/2)
    - CR3: residuals of cluster g multiplied by (I - H_gg)^(-1)
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
    map(g -> sqrt((N - 1) / dof * g / _cluster_dof(g)), G)
end

# CR2 and CR3 replace the residuals `u_g` of each cluster by `(I - H_gg)^p * u_g`, with
# `p = -1/2` (CR2) or `p = -1` (CR3). `H_gg = X_g A X_g' W_g` is the cluster's diagonal
# block of the hat matrix, `A = inv(X'WX) = B B'`, and `W_g` holds the hat weights (see
# `_hatweights`) of the cluster's rows.
#
# `I - H_gg = W_g^(-1/2) (I - C C') W_g^(1/2)` with `C = W_g^(1/2) X_g B`, and every
# adjustment has the form
#
#     (I - H_gg)^p * u_g = u_g + X_g F (r .* (F' X_g' W_g u_g)),
#
# with a `k × k` factor `F` and a `k`-vector `r`; the `n_g × n_g` matrix `I - H_gg` is
# never formed.
#
# - CR2: with the eigendecomposition `C'C = Q Λ Q'`, `F = B Q` and
#   `r = ((1 .- λ).^(-1/2) .- 1) ./ λ`. An eigenvector with `λ = 0` has `C Q[:, j] = 0`,
#   so its finite `r[j]` adds nothing.
# - CR3: by the Woodbury identity, `(I - C C')^(-1) = I + C (I - C'C)^(-1) C'`. With the
#   Cholesky factorization `I - C'C = R'R`, `F = B R⁻¹` and `r = 1`.
#
# A cluster with a single row `i` has `H_gg = h_i`, the row's leverage, and its residual
# is multiplied by `(1 - h_i)^p`.
#
# `indices` holds the clusters with more than one row and `singles` the rows of the
# single-row clusters. `factors[g]` is `F` with zero rows for the non-estimable
# coefficients, so it multiplies the full model matrix.
struct ClusterLeverage{T}
    indices::Vector{Vector{Int}}
    factors::Vector{Matrix{T}}
    ratios::Vector{Vector{T}}
    singles::Vector{Int}
    single_scales::Vector{T}
    weights::Vector{T}
end

# `(1 - h)^p`.
_leverage_scale(::CR2, h) = 1 / sqrt(1 - h)
_leverage_scale(::CR3, h) = 1 / (1 - h)

# Weights `w` of the hat matrix `X inv(X'WX) X'W` of CR2 and CR3, empty for an unweighted
# model.
_hatweights(m) = weights(m)

# An upper triangular `B` with `B B' = inv(X'WX)`.
function _hat_bread_factor(X, w)
    isempty(w) || length(w) == size(X, 1) ||
        throw(DimensionMismatch(
            "the model has $(length(w)) weights but $(size(X, 1)) observations"))
    G = isempty(w) ? X' * X : X' * (w .* X)
    return Matrix(inv(cholesky!(Symmetric(G)).U))
end

# `gap` is the smallest eigenvalue of `I - H_gg`, or a lower bound for it.
function _check_leverage(k, gap, row, tol)
    gap > tol || throw(ArgumentError(
        "$(nameof(typeof(k))) needs I - H_gg to be nonsingular for every cluster, " *
        "but the cluster containing row $row has a hat-matrix eigenvalue of 1 " *
        "(to within $tol). This happens when a regressor is nonzero only within " *
        "one cluster, as with a cluster fixed effect."))
    return nothing
end

# Eigenvalues and eigenvectors of the symmetric matrix `G`. For the small matrices of CR2,
# `syev!` is faster than the `syevr!` behind `eigen(::Symmetric)`.
_symeigen(G::Matrix{<:Union{Float32, Float64}}) = LAPACK.syev!('V', 'U', copy(G))
function _symeigen(G)
    E = eigen(Symmetric(G))
    return E.values, E.vectors
end

# The factor `P` with `F = B P`, and `r`, for the cross product `CC = C'C` of a cluster.
function _cluster_factor(k::CR2, CC, row, tol)
    λ, Q = _symeigen(CC)
    _check_leverage(k, 1 - maximum(λ), row, tol)
    r = map(λ) do λᵢ
        s = sqrt(1 - λᵢ)
        1 / (s * (1 + s))
    end
    return Q, r
end

function _cluster_factor(k::CR3, CC, row, tol)
    ch = cholesky!(Symmetric(I - CC); check = false)
    issuccess(ch) || _check_leverage(k, zero(eltype(CC)), row, tol)
    Rinv = Matrix(inv(ch.U))
    # `1 / sum(abs2, R⁻¹)` is a lower bound for the smallest eigenvalue of `R'R`, within a
    # factor `size(R, 1)` of it.
    _check_leverage(k, 1 / sum(abs2, Rinv), row, tol)
    return Rinv, ones(eltype(Rinv), size(Rinv, 2))
end

# `Xm` holds the estimable columns `midx` of the model matrix, `B B' = inv(Xm'W Xm)`.
function ClusterLeverage(
        k::Union{CR2, CR3}, Xm::AbstractMatrix, B::AbstractMatrix, w::AbstractVector,
        midx::AbstractVector{Bool}, indices::Vector{Vector{Int}})
    Base.require_one_based_indexing(Xm, w)
    T = float(promote_type(eltype(Xm), eltype(B), eltype(w)))
    tol = sqrt(eps(real(T)))

    singles = [first(ind) for ind in indices if length(ind) == 1]
    h = vec(sum(abs2, Xm[singles, :] * B; dims = 2))
    isempty(w) || (h .*= view(w, singles))
    for (i, hi) in zip(singles, h)
        _check_leverage(k, 1 - hi, i, tol)
    end
    single_scales = _leverage_scale.(Ref(k), h)

    multi = filter(ind -> length(ind) > 1, indices)
    factors = Vector{Matrix{T}}(undef, length(multi))
    ratios = Vector{Vector{T}}(undef, length(multi))
    for (gc, ind) in pairs(multi)
        Xg = Xm[ind, :]
        isempty(w) || (Xg .*= sqrt.(view(w, ind)))
        C = Xg * B
        P, r = _cluster_factor(k, C' * C, first(ind), tol)
        if all(midx)
            factors[gc] = B * P
        else
            Fg = zeros(T, length(midx), size(B, 2))
            Fg[midx, :] = B * P
            factors[gc] = Fg
        end
        ratios[gc] = r
    end
    return ClusterLeverage{T}(multi, factors, ratios, singles, single_scales, w)
end

# One `ClusterLeverage` per clustering in `cluster_indices`.
function _cluster_leverage(k::Union{CR2, CR3}, m::RegressionModel, cluster_indices)
    midx = mask(m)
    Xm = modelmatrix(m)[:, midx]
    w = _hatweights(m)
    B = _hat_bread_factor(Xm, w)
    return [ClusterLeverage(k, Xm, B, w, midx, ind) for ind in cluster_indices]
end

function residual_adjustment(k::Union{CR2, CR3}, m::RegressionModel)
    _, cluster_indices, _ = _cluster_combinations(k.g)
    return _cluster_leverage(k, m, cluster_indices)
end

# The residuals `u` after the adjustment `h` of one clustering: a scalar for CR0/CR1, a
# `ClusterLeverage` for CR2/CR3.
_adjusted_residuals(h::Number, X, u) = h * u

function _adjusted_residuals(h::ClusterLeverage, X, u)
    Base.require_one_based_indexing(X, u)
    adj = copyto!(similar(u, promote_type(eltype(u), eltype(h.single_scales))), u)
    view(adj, h.singles) .*= h.single_scales
    wu = isempty(h.weights) ? u : h.weights .* u
    for (gc, ind) in pairs(h.indices)
        Xg = X[ind, :]
        F = h.factors[gc]
        c = h.ratios[gc] .* (F' * (Xg' * wu[ind]))
        view(adj, ind) .+= Xg * (F * c)
    end
    return adj
end

"""
    aVar(estimator, model::RegressionModel; kwargs...)

Compute the asymptotic variance matrix for the parameters of a `RegressionModel`.

This is the generic implementation that works with any model implementing the RegressionModel protocol.

For `Uncorrelated()` the result is the meat of the iid variance, `s² · inv(bread(model))`
divided by `n`, with `s² = RSS / n` (see `vcov`); `n · bread · aVar · bread` then equals
`vcov(Uncorrelated(), model; dofadjust = false)`. `demean` and `prewhite` do not apply
and throw an `ArgumentError` when set.

"""
function aVar(
        k::AbstractAsymptoticVarianceEstimator,
        m::StatsBase.RegressionModel;
        demean = false,
        prewhite = false,
        scale = true,
        scaleby::Union{Nothing, Real} = nothing,
        kwargs...)
    k isa Uncorrelated &&
        return _iid_aVar(k, m; demean, prewhite, scale, scaleby, kwargs...)
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
    kw = bandwidthweights(k, X)
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
    # A model cache is handled here rather than by a method on `CachedCR`, which
    # would be ambiguous with every `aVar(::CR, ::MyModel)` a modeling package defines.
    k isa CachedCR && cache(k) isa CRModelCache &&
        return _cached_model_aVar(k, m; scale, scaleby, prewhite)
    _check_resolved(k)
    _checkscale(scale)
    _checkprewhite(k, prewhite)
    H = residual_adjustment(k, m)
    X = modelmatrix(m)
    u = momentresiduals(m)
    M = map(h -> X .* _adjusted_residuals(h, X, u), H)
    V = avar_tuple(k, M)
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    combs = Iterators.filter(!isempty, combinations(1:length(k.g)))
    Σ = mapreduce(+, zip(combs, V)) do (c, v)
        (-1)^(length(c) - 1)*v
    end
    scalevar!(Σ, scale, scaleby, numobs(m))
    return CovarianceMatrix(Σ, k, (; nclusters = nclusters(k)))
end

#=========
Uncorrelated on a RegressionModel: the iid (model-based) variance
=========#

# The factor `s²` with `vcov(Uncorrelated(), m) = s² · bread(m)`. For least squares
# `bread(m)` carries no residual variance, so `s²` is `deviance / (n - K)`, or
# `deviance / n` without the degrees-of-freedom correction. `n` is the number of rows
# and `K` the number of estimable coefficients, as in R's `vcov(lm(...))` with or
# without weights; `StatsAPI.dof_residual` is not used because GLM.jl counts weighted
# observations as `sum(weights)`. A model whose `bread` already includes an estimated
# dispersion (a GLM) defines its own method.
function _iid_divisor(m, dofadjust::Bool)
    dofadjust ? _residual_dof(numobs(m), sum(mask(m))) : numobs(m)
end

function _iid_dispersion(m::RegressionModel, dofadjust::Bool)
    StatsAPI.deviance(m) / _iid_divisor(m, dofadjust)
end

# Meat of the iid variance, `s² · inv(bread(m))` with `s² = deviance / n`, so that
# `n · bread · aVar · bread` equals the iid variance without the degrees-of-freedom
# correction. The generic `aVar` and `vcov` branch to these two functions rather
# than adding `Uncorrelated`-specific methods, so a package that defines `aVar` or
# `vcov` for every estimator on its own model type does not get ambiguities.
function _iid_aVar(
        k::Uncorrelated,
        m::RegressionModel;
        demean = false,
        prewhite = false,
        scale = true,
        scaleby::Union{Nothing, Real} = nothing,
        kwargs...)
    _check_iid_keywords(kwargs)
    (demean || prewhite) && throw(ArgumentError(
        "`Uncorrelated` on a fitted model is the iid variance; `demean` and `prewhite` " *
        "do not apply to it"))
    _checkscale(scale)
    B = bread(m)
    midx = mask(m)
    Bm = all(midx) ? B : B[midx, midx]
    Σ = _iid_dispersion(m, false) .* inv(Bm)
    scalevar!(Σ, scale, scaleby, numobs(m))
    return CovarianceMatrix(Σ, k)
end

function _check_iid_keywords(kwargs)
    isempty(kwargs) || throw(ArgumentError(
        "`Uncorrelated` on a fitted model does not take the keyword(s) " *
        join(("`$(n)`" for n in keys(kwargs)), ", ")))
    return nothing
end

# The iid variance `s² · bread(m)`, with `NaN` rows and columns for non-estimable
# coefficients. `dofadjust` is its only keyword.
function _iid_vcov(k::Uncorrelated, m::RegressionModel; dofadjust::Bool = true, kwargs...)
    _check_iid_keywords(kwargs)
    V = _iid_dispersion(m, dofadjust) .* bread(m)
    midx = mask(m)
    if !all(midx)
        V[.!midx, :] .= NaN
        V[:, .!midx] .= NaN
    end
    return CovarianceMatrix(V, k)
end

"""
    vcov(estimator, model; dofadjust=true, kwargs...)

Compute the sandwich variance-covariance matrix for a RegressionModel.

For `Uncorrelated()` the result is the iid (non-robust) variance `s² · bread(model)`
instead of a sandwich. For least squares `s² = RSS / (n - K)`, with `n` rows and `K`
estimable coefficients, so the result equals R's `vcov(lm(...))`; with weights, RSS is
the weighted sum of squares. For a GLM the result is the dispersion times `inv(X'WX)`,
which equals R's `vcov(glm(...))`; the dispersion is the Pearson statistic divided by
`n - K` for families that estimate one, and 1 for families with a fixed dispersion
(Binomial, Bernoulli, Poisson). `dofadjust = false` divides by `n` instead of `n - K`;
no other keyword applies. Unweighted, the result is also GLM.jl's `vcov`; with weights,
GLM.jl divides by `sum(weights) - K` instead of `n - K`.

"""
function StatsAPI.vcov(
        k::AbstractAsymptoticVarianceEstimator,
        m::RegressionModel;
        dofadjust = true,
        kwargs...
)
    k isa Uncorrelated && return _iid_vcov(k, m; dofadjust, kwargs...)
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
    _check_resolved(k)
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
`plm::vcovSCC` are not supported. The correction is set only on the estimator;
`vcov` throws an `ArgumentError` if it is given a `type` keyword.
"""
function StatsAPI.vcov(k::DriscollKraay, m::RegressionModel; type = nothing, kwargs...)
    # The generic method ignores unknown keywords; a `type` passed here would
    # otherwise be dropped and the uncorrected matrix returned.
    type === nothing || throw(ArgumentError(
        "`vcov` takes no `type` keyword for a Driscoll-Kraay estimator; construct it " *
        "with `DriscollKraay(...; correction = $(repr(type)))`"))
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

The long-run variance estimators (`HAC` kernels, `EWC`, `VARHAC`, and the moment
smoothers `UniformSmoother` and `TriangularSmoother`) multiply the variance by
`n / (n - K)`, with `K` the number of coefficients. `DriscollKraay` applies the factor
selected by its `correction`. The other estimators (`HC`, `CR`) incorporate their
corrections in their residual adjustments.

"""
dofcorrect!(V, k::AbstractAsymptoticVarianceEstimator, m) = nothing

function dofcorrect!(V, k::Union{HAC, EWC, VARHAC, MomentSmoother}, m::RegressionModel)
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
    grouped_arrays, cluster_indices, signs = _cluster_combinations(k.g)
    H = _compute_leverage_adjustments(k, m, grouped_arrays, cluster_indices)
    cache = CRModelCache{typeof(H)}(
        grouped_arrays, cluster_indices, signs, H, size(X, 2), numobs(m))
    return CachedCR(k, cache)
end

# Dispatch on CR type to compute appropriate leverage adjustments
function _compute_leverage_adjustments(
        k::Union{CR0, Cluster}, m, grouped_arrays, cluster_indices)
    # CR0: No adjustment, just return vector of 1s (one per combination)
    [1 for _ in grouped_arrays]
end

function _compute_leverage_adjustments(k::CR1, m, grouped_arrays, cluster_indices)
    # CR1: DOF adjustment - scalar per combination
    N, K = size(modelmatrix(m))
    dof = _residual_dof(N, K)
    map(g -> sqrt((N - 1) / dof * g.ngroups / _cluster_dof(g.ngroups)), grouped_arrays)
end

function _compute_leverage_adjustments(
        k::Union{CR2, CR3}, m, grouped_arrays, cluster_indices)
    return _cluster_leverage(k, m, cluster_indices)
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

# `aVar(k, m)` for a cache built by `CachedCR(k, model)`: the leverage adjustments
# come from the cache, and only the residual-dependent parts are computed.
function _cached_model_aVar(
        k::CachedCR{<:CR, <:CRModelCache}, m::RegressionModel;
        scale, scaleby::Union{Nothing, Real}, prewhite::Bool)
    _checkscale(scale)
    _checkprewhite(k, prewhite)
    cache = k.cache
    H = residual_adjustment(k, m)
    X = modelmatrix(m)
    u = momentresiduals(m)

    # Compute moment matrices using cached leverage adjustments
    M = map(h -> X .* _adjusted_residuals(h, X, u), H)

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
