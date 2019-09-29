"""
Provide an API for the interface defined in StatsBase.


cov(ce::CovarianceEstimator, X::AbstractMatrix, [w::AbstractWeights]; mean=nothing, dims::Int=1)

Compute the covariance matrix of the matrix `X` along dimension `dims`
using estimator `ce`. A weighting vector `w` can be specified.

The keyword argument `mean` can be:
* `nothing` (default) in which case the mean is estimated and subtracted
  from the data `X`,
* a precalculated mean in which case it is subtracted from the data `X`.
  Assuming `size(X)` is `(N,M)`, `mean` can either be:
  * when `dims=1`, an `AbstractMatrix` of size `(1,M)`,
  * when `dims=2`, an `AbstractVector` of length `N` or an `AbstractMatrix`
    of size `(N,1)`.
"""


# StatsBase.cov(ce::RobustVariance, X::Vector, args...; kwargs...) = 
#         StatsBase.cov(ce, reshape(X, (length(X),1)), args...; kwargs...)

# function StatsBase.cov(ce::RobustVariance, X::AbstractMatrix; mean=nothing, dims::Int=1, corrected::Bool=false)
#     dims ∈ (1, 2) || throw(ArgumentError("Argument dims can only be 1 or 2 (given: $dims)"))
#     Z = dims == 1 ? X : X'
#     if mean === nothing
#         covariance(Z, ce, Matrix, scale=size(X, dims)-corrected, demean=true)
#     else
#         @assert length(mean) == size(X, dims == 1 ? 2 : 1)
#         covariance(Z .- mean, ce, Matrix, scale=size(Z, dims)-corrected, demean=false)
#     end
# end

# function StatsBase.cov(ce::RobustVariance, X::AbstractMatrix, cache::AbstractCache; mean=nothing, dims::Int=1, corrected::Bool=false)
#     if mean === nothing
#         covariance(X, ce, cache, Matrix, SVD, scale = size(X, dims)-corrected, demean = true)
#     else
#         @assert length(mean) == size(X, dims == 1 ? 2 : 1)
#         covariance(X .- mean, ce, Matrix, SVD, scale = size(X, dims)-corrected, demean = false)
#     end
# end