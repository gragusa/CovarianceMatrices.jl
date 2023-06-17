"""
DriscolKraay variance covariance matrix estimator

Driscol and Kraay (1998) 
"""

function avar(k::T, X::Matrix{R}; kwargs...) where {T<:DriscolKraay, R<:Real}    
  i = clusterindicator(k.i)
  t = clusterindicator(k.t)

  ## 1.
  ## Sum over i dimension
  h = Array{R}(undef, size(X,2), size(X,2))
  fill!(h, zero(R))
  for (t, ind_ti) in enumerate(k.t)
    for i in ind_ti
      h[t,:] += X[i, :]
    end
  end

  a𝕍ar(k, h; kwargs...)
  
end


