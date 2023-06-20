module ClusterVectorizedExt

using CovarianceMatrices, LoopVectorization, Folds, CategoricalArrays, LinearAlgebra


function CovarianceMatrices.clustersum(X::Vector{T}, cl, ::CovarianceMatrices.Threaded) where T<:AbstractFloat
  Shat = fill!(similar(X, (1, 1)), zero(T))
  s = Vector{T}(undef, size(Shat, 1))
  CovarianceMatrices.clustersum!(Shat, s, X[:,:], cl, CovarianceMatrices.Threaded())
  vec(Shat)
end

function CovarianceMatrices.clustersum(X::Matrix{T}, cl, ::CovarianceMatrices.Threaded) where T<:AbstractFloat
  _, p = size(X)
  Shat = fill!(similar(X, (p, p)), zero(T))
  s = Vector{T}(undef, size(Shat, 1))
  CovarianceMatrices.clustersum!(Shat, s, X, cl, CovarianceMatrices.Threaded())
end

function CovarianceMatrices.clustersum!(Shat::Matrix{T}, s::Vector{T}, X::Matrix{T}, cl, ::CovarianceMatrices.Threaded) where T<:AbstractFloat
  for m in CovarianceMatrices.clusterintervals(cl)
      @inbounds fill!(s, zero(T))
      CovarianceMatrices.innerXiXi!(s, m, X, CovarianceMatrices.Sequential())
      CovarianceMatrices.innerXiXj!(Shat, s, CovarianceMatrices.Sequential())
  end
  return LinearAlgebra.copytri!(Shat, 'U')
end

function CovarianceMatrices.innerXiXi!(s, m, X, ::CovarianceMatrices.Threaded)
  @tturbo for j in eachindex(s)
      for i in eachindex(m)
          s[j] += X[m[i], j]
      end
  end
end

function CovarianceMatrices.innerXiXj!(Shat, s, ::CovarianceMatrices.Threaded)
    @inbounds for j in eachindex(s)
      @tturbo for i in 1:j
          Shat[i, j] += s[i]*s[j]
      end
  end
end



end
