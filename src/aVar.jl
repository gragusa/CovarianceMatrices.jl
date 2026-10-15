"""
Asymptotic Variance Estimators

aVar(k::AbstractAsymptoticVarianceEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1, means::Union{Nothing, AbstractArray}=nothing, prewhite::Bool=false, scale::Bool=true, scaleby::Union{Nothing, Real}=nothing)

The asymptotic variance is the matrix `Σ` of the asymptotic approximation:

```math
\\sqrt{n}\\Sigma^{-1/2} (\\bar{X} - \\mu) \\xrightarrow{d} N(0, I_p)
```

where `X̄` is the sample mean of the observations in `m` (averaged along `dims`) and `μ` is the population mean.

## Note

- The element type of `m` must be `Real`.
- `prewhite = true` _prewhitens_ the matrix with a VAR(1) model. It applies to `HAC`,
  `EWC` and `MomentSmoother` estimators, and to `DriscollKraay`, which prewhitens the
  series of period sums; other estimators do not prewhiten and emit a deprecation
  warning when it is requested.
- On a matrix, `HR1`–`HR5`/`HC1`–`HC5` and `CR1`–`CR3` apply no finite-sample
  correction and equal `HR0` and `CR0`. Their corrections need a fitted model; passing
  them with a matrix is deprecated.
- `scale` selects whether the variance is divided by the number of observations.
- `scaleby` divides the variance by an explicit positive divisor instead, which is
  convenient for a degrees-of-freedom correction. It takes precedence over `scale`.
- `scale = true` produces the same per-observation scale for every estimator. `VARHAC`
  estimates the spectral density at frequency zero, which already carries that scaling,
  so `scale = false` multiplies it by the number of observations rather than leaving it
  untouched.
"""
function aVar(k::AbstractAsymptoticVarianceEstimator, m::AbstractMatrix{<:Real}; kwargs...)
    _check_matrix_path(_check_resolved(k))
    return _aVar(k, m; kwargs...)
end

# The finite-sample adjustments of these estimators are formed from a fitted model's
# design matrix. On a bare matrix they apply none, so the result equals `HR0` or `CR0`.
const _ModelOnlyAdjustment = Union{HR1, HR2, HR3, HR4, HR4m, HR5, CR1, CR2, CR3}

_check_matrix_path(k) = nothing
_check_matrix_path(k::CachedCR) = _check_matrix_path(k.estimator)
function _check_matrix_path(k::_ModelOnlyAdjustment)
    name = nameof(typeof(k))
    base = k isa HR ? "`HR0` (`HC0`)" : "`CR0` or `Cluster`"
    Base.depwarn(
        "`aVar` with `$name` on a matrix applies no finite-sample correction, so it " *
        "equals $base. Passing `$name` with a matrix is deprecated and will throw an " *
        "error in a future release: use $base on a matrix, or pass a fitted model.",
        :aVar)
    return nothing
end

# Matrix-path computation shared by `aVar(k, X)` and the generic model path, which
# applies the model-based adjustment to the moments before calling it.
function _aVar(
        k::AbstractAsymptoticVarianceEstimator,
        m::AbstractMatrix{T};
        demean::Bool = true,
        dims::Int = 1,
        means::Union{Nothing, AbstractArray} = nothing,
        prewhite::Bool = false,
        scale = true,
        scaleby::Union{Nothing, Real} = nothing,
        weights = nothing
) where {T <: Real}
    _check_resolved(k)
    Base.require_one_based_indexing(m)
    scale, scaleby = _scale_arguments(scale, scaleby)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat, info = avar_with_info(k, X; prewhite = _checkprewhite(k, prewhite), weights)
    scalevar!(Shat, scale, scaleby, size(X, dims))
    return CovarianceMatrix(Shat, k, info)
end

"""
    avar_with_info(k, X; prewhite=false)

Compute the estimate and the quantities selected from the data.

Returns `(V, info)`. Estimators that select nothing from the data return an empty
`info`; `HAC` kernels report the bandwidth and kernel weights, `VARHAC` the selected
lag orders and information criteria.
"""
function avar_with_info(k, X; weights = nothing, kwargs...)
    return avar(k, X; kwargs...), NamedTuple()
end

# Estimators that implement VAR(1) prewhitening.
_prewhitens(k) = false
_prewhitens(::HAC) = true
_prewhitens(::EWC) = true
_prewhitens(::MomentSmoother) = true

function _checkprewhite(k, prewhite::Bool)
    prewhite && !_prewhitens(k) &&
        Base.depwarn(
            "`prewhite = true` is deprecated for $(nameof(typeof(k))), which does not " *
            "prewhiten: it has no effect and will throw an error in a future release.", :aVar)
    return prewhite && _prewhitens(k)
end

"""
    _scale_arguments(scale, scaleby) -> (scale::Bool, scaleby)

Normalize the scaling keywords to a `Bool` switch and an optional divisor, accepting
the deprecated numeric `scale` as a divisor.
"""
function _scale_arguments(scale, scaleby)
    scale isa Bool && return scale, scaleby
    scale isa Real ||
        throw(ArgumentError("`scale` must be a `Bool`, got $(repr(scale)); pass a divisor as `scaleby`."))
    scaleby === nothing ||
        throw(ArgumentError("`scale` was given a divisor and `scaleby` was also given; pass the divisor as `scaleby` alone."))
    Base.depwarn(
        "`scale=$scale` as a divisor is deprecated: use `scaleby=$scale` to divide by an explicit value.",
        :aVar)
    return true, scale
end

function _checkdivisor(d)
    (isfinite(d) && d > 0) ||
        throw(ArgumentError("`scaleby` must be a positive finite number, got $d."))
end

# A divisor supersedes the `scale` switch: the variance is divided once.
function scalevar!(Shat, scale::Bool, scaleby, n)
    _checkdivisor(scaleby)
    return rdiv!(Shat, scaleby)
end
scalevar!(Shat, scale::Bool, ::Nothing, n) = scale ? rdiv!(Shat, n) : Shat

# Counterpart of `scalevar!` for estimators whose result already carries the `1/n`.
# `scale=true` is then a no-op and `scale=false` must undo it.
function unscalevar!(Shat, scale::Bool, scaleby, n)
    _checkdivisor(scaleby)
    return rmul!(Shat, n / scaleby)
end
unscalevar!(Shat, scale::Bool, ::Nothing, n) = scale ? Shat : rmul!(Shat, n)

function _aVar(
        k::VARHAC,
        m::AbstractMatrix{T};
        demean::Bool = true,
        dims::Int = 1,
        means::Union{Nothing, AbstractArray} = nothing,
        scale = true,
        scaleby::Union{Nothing, Real} = nothing,
        prewhite::Bool = false,
        weights = nothing,
        kwargs...
) where {T <: Real}
    Base.require_one_based_indexing(m)
    scale, scaleby = _scale_arguments(scale, scaleby)
    _checkprewhite(k, prewhite)
    isempty(kwargs) || Base.depwarn(
        "`aVar(::VARHAC, X)` does not use the keyword(s) $(join(keys(kwargs), ", ")); " *
        "passing them is deprecated and will throw an error in a future release.", :aVar)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat, info = avar_with_info(k, X)
    # VARHAC estimates the spectral density at frequency zero, which is already on the
    # per-observation scale that the other estimators reach through `scale=true`.
    # Reaching the unscaled convention therefore multiplies by `n` rather than dividing.
    unscalevar!(Shat, scale, scaleby, size(X, dims))
    return CovarianceMatrix(Shat, k, info)
end

Base.@deprecate_binding a𝕍ar aVar
