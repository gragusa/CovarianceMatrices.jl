"""
Asymptotic Variance Estimators

aVar(k::AVarEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1, means::Union{Nothing, AbstractArray}=nothing, prewhite::Bool=false, scale=true)

## Arguments

    - k::AVarEstimator 
    - `demean=false`: whether the data should be demeaned.    
    - `prewhite=false`: should the data be prewithened. Relevant for `HAC` estimator.
"""
aVar(k::AVarEstimator, m::AbstractMatrix; kwargs...) = aVar(k, float.(m), kwargs...)

function aVar(k::AVarEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1,
              means::Union{Nothing,AbstractArray}=nothing, prewhite::Bool=false,
              scale=true) where {T<:AbstractFloat}
    Base.require_one_based_indexing(m)
    X = demean ? demeaner(m; means=means, dims=dims) : m
    Shat = avar(k, X; prewhite=prewhite)
    return scale ? Shat ./ size(X, dims) : Shat
end

const a𝕍ar = aVar

# function lrvar(k::AVarEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1,
#                means::Union{Nothing,AbstractArray}=nothing,
#                prewhite::Bool=false) where {T<:AbstractFloat}
#     Shat = aVar(k, m; demean=demean, dims=dims, means=means, prewhite=prewhite, scale=true)
#     return Shat ./ size(m, dims)
# end
