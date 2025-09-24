"""
Asymptotic Variance Estimators

aVar(k::AVarEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1, means::Union{Nothing, AbstractArray}=nothing, prewhite::Bool=false, scale=true)

## Arguments

    - k::AVarEstimator 
    - `demean=false`: whether the data should be demeaned.    
    - `dims=1`: the dimension along which the demeaning should be performed.
    - `means=nothing`: the means to be used for demeaning.
    - `prewhite=false`: should the data be prewithened. Relevant for `HAC` estimator.
    - `scale`: should the variance be scaled by the number of observations. If scle is an 
               `Int` that value is used to scale the variance. This is convenient for degrees
               of freedom correction or in cases where the variance is needed without 
               scaling. 
"""
aVar(k::AVarEstimator, m::AbstractMatrix; kwargs...) = aVar(k, float.(m), kwargs...)

function aVar(
    k::AVarEstimator,
    m::AbstractMatrix{T};
    demean::Bool = true,
    dims::Int = 1,
    means::Union{Nothing,AbstractArray} = nothing,
    prewhite::Bool = false,
    scale = true,
) where {T<:Real}
    Base.require_one_based_indexing(m)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat = avar(k, X; prewhite = isa(k, HAC) ? prewhite : false)
    scalevar!(Shat, scale, size(X, dims))
    return Shat
end

scalevar!(Shat, scale::Bool, n::Int) = scale ? rdiv!(Shat, n) : Shat
scalevar!(Shat, scale::Int, n::Int) = rdiv!(Shat, scale)
scalevar!(Shat, scale, n) =
    throw(ArgumentError("`scale` should be either an Int or a Bool."))
function scalevar!(Shat, scale::Int, n)
    @warn "The variance is being scaled by an AbstractFloat"
    rdiv!(X, scale)
end

function aVar(
    k::VARHAC,
    m::AbstractMatrix{T};
    demean::Bool = true,
    dims::Int = 1,
    means::Union{Nothing,AbstractArray} = nothing,
    scale = true,
    kwargs...,
) where {T<:Real}
    Base.require_one_based_indexing(m)
    X = demean ? demeaner(m; means = means, dims = dims) : m
    Shat = avar(k, X)
    return Shat
end

const a𝕍ar = aVar
