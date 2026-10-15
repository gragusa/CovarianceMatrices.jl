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
- `scale` selects whether the variance is divided by the number of observations.
- `scaleby` divides the variance by an explicit positive divisor instead, which is
  convenient for a degrees-of-freedom correction. It takes precedence over `scale`.
- `scale = true` produces the same per-observation scale for every estimator. `VARHAC`
  estimates the spectral density at frequency zero, which already carries that scaling,
  so `scale = false` multiplies it by the number of observations rather than leaving it
  untouched.
"""
function aVar(
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
    _checkscale(scale)
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

function _checkscale(scale)
    scale isa Bool ||
        throw(ArgumentError("`scale` must be a `Bool`, got $(repr(scale)); pass a divisor as `scaleby`."))
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

function aVar(
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
    _checkscale(scale)
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

const a𝕍ar = aVar
