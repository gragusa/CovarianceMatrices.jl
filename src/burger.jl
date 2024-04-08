function burger(K::AVarEstimator, momentmatrix::AbstractMatrix{T}, bun::Union{AbstractMatrix{F}, Factorization{F}}; demean=false, prewhite::Bool=false, dof::Int64=0) where {T<:Real, F<:Real}
    kwargs = (demean=demean, prewhite=prewhite, dof=dof, unscaled=true)
    P = patty(K, m; kwargs...)
    ## Form A^{-1}BA^{-1}'
    return (bun\P)/bun
end

patty(K::AVarEstimator, m::AbstractMatrix; kwargs...) = aVar(K, m; kwargs...)






