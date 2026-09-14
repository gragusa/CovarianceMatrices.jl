"""
Asymptotic Variance Estimators

aVar(k::AbstractAsymptoticVarianceEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1, means::Union{Nothing, AbstractArray}=nothing, prewhite::Bool=false, scale=true)

The asymptotic variance is the matrix `Σ` of the asymptotic approximation:

```math
\\sqrt{n}\\Sigma^{-1/2} (\\bar{X} - \\mu) \\xrightarrow{d} N(0, I_p)
```

where `X̄` is the sample mean of the observations in `m` (averaged along `dims`) and `μ` is the population mean.

## Note

- The element type of `m` must be `Real`.
- `prewhite` argument is only relevant for `HAC` estimator in which case the matrix is _prewhitened_ using a VAR(1) model.
- The `scale` parameter should indicate whether the variance be scaled by the number of observations. If `scale` is an `Int` that value is used to scale the variance. This is convenient for degrees of freedom correction or in cases where the variance is needed without scaling.
"""
function aVar(
        k::AbstractAsymptoticVarianceEstimator,
        m::AbstractMatrix{T};
        demean::Bool = true,
        dims::Int = 1,
        means::Union{Nothing, AbstractArray} = nothing,
        prewhite::Bool = false,
        scale = true,
        weights = nothing
) where {T <: Real}
    Base.require_one_based_indexing(m)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat, info = avar_with_info(k, X; prewhite = isa(k, HAC) ? prewhite : false, weights)
    scalevar!(Shat, scale, size(X, dims))
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

scalevar!(Shat, scale::Bool, n::Int) = scale ? rdiv!(Shat, n) : Shat
scalevar!(Shat, scale::Int, n::Int) = rdiv!(Shat, scale)
function scalevar!(Shat, scale, n)
    throw(ArgumentError("`scale` should be either an Int or a Bool."))
end
function scalevar!(Shat, scale::Int, n)
    @warn "The variance is being scaled by an AbstractFloat"
    rdiv!(X, scale)
end

function aVar(
        k::VARHAC,
        m::AbstractMatrix{T};
        demean::Bool = true,
        dims::Int = 1,
        means::Union{Nothing, AbstractArray} = nothing,
        scale = true,
        kwargs...
) where {T <: Real}
    Base.require_one_based_indexing(m)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat, info = avar_with_info(k, X)
    # VARHAC returns the spectral density at frequency zero, which already carries
    # the variance scaling; `scale` is handled where it is honored or rejected.
    return CovarianceMatrix(Shat, k, info)
end

const a𝕍ar = aVar
