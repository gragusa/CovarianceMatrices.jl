# """


#  aVar(k::AVarEstimators, M::Matrix{T}; demean=true, prewhite=false, dof=0.0)

# Caclulate the asymptotic variance of \bar{X}_{,j}=\frac{1}{b_n}\sum{i=1}^n X_{i,j}, where \bar{X}_{,j} is the j-th column of `X`.
#  The assumption is b_n is 

#  ## Arguments 
#     k::AVarEstimator 
#     - `demean=false`: whether the data should be demeaned.    
#     - `prewhite=false`: should the data be prewithened. Relevant for `HAC` estimator.
    
# """
 
function aVar(k::AVarEstimator, m::AbstractMatrix{T}; demean::Bool=true, dims::Int=1, means::Union{Nothing, AbstractArray}=nothing, prewhiten::Bool=false) where T<:AbstractFloat
    X = demean ? demeaner(float.(m), means; dims=dims) : float.(m)
    Shat = avar(k, X; prewhiten=prewhiten)
    return Shat
end

aVar(k::AVarEstimator, m::AbstractMatrix{T}; kwargs...) where T<:Integer = aVar(k, float.(m); kwargs...)


const a𝕍ar = aVar
    

