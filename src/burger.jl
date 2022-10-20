## The bun and the meat!

function burger(K::AVarEstimator, m::AbstractMatrix{T}, bun::Union{AbstractMatrix{F}, Factorization{F}}; demean=false, prewhiten::Bool=false, dof::Int64=0) where {T<:Real, F<:Real}
    kwargs = (demean=demean, prewhiten=prewhiten, dof=dof, unscaled=true)
    P = patty(K, m; kwargs...)
    ## Form A^{-1}BA^{-1}'
    return (bun\P)/bun
end

patty(K::AVarEstimator, m::AbstractMatrix; kwargs...) = aVar(K, m; kwargs...)

## And with this, we rule the world, not really...
"""
    burger(K::AVarEstimator, residuals::AbstractVector, modelmatrix::AbstractMatrix, bun::Union{AbstractMatrix{F}, Factorization{F}}; demean=false, prewhiten::Bool=false, dof::Int64=0)

Burger function for generalized linear model. 
    
    ## Arguments
    - K
    - residuals: the residuals; typically obtained by residuals(obj; weighted=true)
    - modelmatrix: the model matrix; typically obtained by modelmatrix(obj; weighted=true)   

"""


function burger(K::AVarEstimator, residuals::AbstractVector, modelmatrix::AbstractMatrix, bun::Union{AbstractMatrix{F}, Factorization{F}}; demean=false, prewhiten::Bool=false, dof::Int64=0)
    kwargs = (demean=demean, prewhiten=prewhiten, dof=dof, unscaled=true)
    P = patty(K, residuals, modelmatrix; kwargs...)
    ## Form A^{-1}BA^{-1}'
    return (bun\P)/bun
end


function patty(K, residuals, modelmatrix; kwargs...) 
    ## implement this
end




