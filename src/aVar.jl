"""
Asymptotic Variance Estimators

aVar(k::AbstractAsymptoticVarianceEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1, means::Union{Nothing, AbstractArray}=nothing, prewhite::Bool=false, scale=true)

The asymptotic variance is the matrix `Σ` of the asymptotic approximation:

```math
\\sqrt{n}\\Sigma^{-1/2} (\\bar{X} - \\mu) \\xrightarrow{d} N(0, I_p)
```

where `X̄` is the sample mean of the observations in `m` (averaged along `dims`) and `μ` is the population mean.

## Note

- `prewhite` argument is only relevant for `HAC` estimator in which case the matrix is _prewhitened_ using a VAR(1) model.
- The `scale` parameter should indicate whether the variance be scaled by the number of observations. If `scale` is an `Int` that value is used to scale the variance. This is convenient for degrees of freedom correction or in cases where the variance is needed without scaling.
"""
function aVar(k::AbstractAsymptoticVarianceEstimator, m::AbstractMatrix; kwargs...)
    aVar(k, float.(m), kwargs...)
end

function aVar(
        k::AbstractAsymptoticVarianceEstimator,
        m::AbstractMatrix{T};
        demean::Bool = true,
        dims::Int = 1,
        means::Union{Nothing, AbstractArray} = nothing,
        prewhite::Bool = false,
        scale = true
) where {T <: Real}
    Base.require_one_based_indexing(m)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat = avar(k, X; prewhite = isa(k, HAC) ? prewhite : false)
    scalevar!(Shat, scale, size(X, dims))
    return Shat
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
    Shat = avar(k, X)
    # VARHAC returns spectral density at frequency zero, which is already
    # properly scaled for variance estimation, so no additional scaling needed
    # However, maintain API consistency for user expectations
    if scale === false
        # User explicitly requested no scaling, but VARHAC is already properly scaled
        return Shat
    else
        # VARHAC already provides proper variance scaling
        return Shat
    end
end

const a𝕍ar = aVar
