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
- `prewhite` argument is only relevant for `HAC` estimator in which case the matrix is _prewhitened_ using a VAR(1) model.
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
    Base.require_one_based_indexing(m)
    scale, scaleby = _scale_arguments(scale, scaleby)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat, info = avar_with_info(k, X; prewhite = isa(k, HAC) ? prewhite : false, weights)
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

"""
    _scale_arguments(scale, scaleby) -> (scale::Bool, scaleby)

Normalize the scaling keywords to a `Bool` switch and an optional divisor, accepting
the deprecated numeric `scale` as a divisor.
"""
function _scale_arguments(scale, scaleby)
    if !isa(scale, Bool)
        isa(scale, Real) ||
            throw(ArgumentError("`scale` must be a `Bool`; pass a divisor as `scaleby`."))
        Base.depwarn(
            "`scale=$scale` as a divisor is deprecated: use `scaleby=$scale` to divide by an explicit value.",
            :aVar)
        scaleby === nothing ||
            throw(ArgumentError("`scale` was given a divisor and `scaleby` was also given; pass the divisor as `scaleby` alone."))
        return true, scale
    end
    return scale, scaleby
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
        kwargs...
) where {T <: Real}
    Base.require_one_based_indexing(m)
    scale, scaleby = _scale_arguments(scale, scaleby)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat, info = avar_with_info(k, X)
    # VARHAC estimates the spectral density at frequency zero, which is already on the
    # per-observation scale that the other estimators reach through `scale=true`.
    # Reaching the unscaled convention therefore multiplies by `n` rather than dividing.
    unscalevar!(Shat, scale, scaleby, size(X, dims))
    return CovarianceMatrix(Shat, k, info)
end

const a𝕍ar = aVar
