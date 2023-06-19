"""
DriscolKraay variance covariance matrix estimator

Driscol and Kraay (1998) 
"""

function avar(k::T, X::Matrix{R}; kwargs...) where {T<:DriscollKraay, R<:Real}
  i = CovarianceMatrices.clusterintervals(k.i)
  t = CovarianceMatrices.clusterintervals(k.t)

  ## 1.
  ## Sum over i dimension
  h = Array{R}(undef, length(unique(k.t)), size(X,2))
  fill!(h, zero(R))
  @inbounds for (t, ind_ti) in enumerate(t)
    for i in ind_ti
      h[t,:] += X[i, :]
    end
  end
  a𝕍ar(k.K, h; kwargs...)
end


