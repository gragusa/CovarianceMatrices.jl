"""
Asymptotic Variance Estimators

aVar(k::AVarEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1, means::Union{Nothing, AbstractArray}=nothing, prewhiten::Bool=false, scale=true)

Caclulate the asymptotic variance of \\bar{X}_{,j}=\\frac{1}{b_n}\\sum{i=1}^n X_{i,j}, where \\bar{X}_{,j} is the j-th column of `X`.

## Arguments 
    - k::AVarEstimator 
    - `demean=false`: whether the data should be demeaned.    
    - `prewhite=false`: should the data be prewithened. Relevant for `HAC` estimator.
    
"""

aVar(k::AVarEstimator, m::AbstractMatrix; kwargs...) = aVar(k, float.(m), kwargs...)

function aVar(k::AVarEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1, means::Union{Nothing, AbstractArray}=nothing, prewhiten::Bool=false, scale=true) where T<:AbstractFloat
    Base.require_one_based_indexing(m)
    X = demean ? demeaner(m; means=means, dims=dims) : m
    Shat = avar(k, X; prewhiten=prewhiten)
    return scale ? Shat./size(X,dims) : Shat
end

const a𝕍ar = aVar


