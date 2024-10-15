# #======
# A. HAC
# =======#
# function lrvar(
#     k::HAC,
#     m::AbstractMatrix{T};
#     prewhite::Bool = false,
#     demean::Bool=true,
#     scale::Real=one(T),
# ) where T<:Real
#     mm = demean ? m .- mean(m, dims = 1) : m
#     scale *= inv(size(mm,1))
#     return _lrvar(k, mm, prewhite, scale)
# end

# function _lrvar(k::HAC, mm::AbstractMatrix{T}, prewhite::Bool, scale) where T<:Real
#     n, _ = size(mm)
#     m, D = prewhiter(mm, Val{prewhite})
#     bw = optimalbandwidth(k, m; prewhite=prewhite)
#     V = triu!(m'*m)
#     @inbounds for j in covindices(k, n)
#         κⱼ = kernel(k, j/bw)
#         LinearAlgebra.axpy!(κⱼ, Γ(m, j), V)
#     end
#     LinearAlgebra.copytri!(V, 'U')
#     dewhiter!(V, m, D, Val{prewhite})    
#     if eltype(V) <: AbstractFloat
#         Symmetric(V.*convert(eltype(V), scale))
#     else
#         Symmetric(V.*scale)
#     end

# end

# function lrvarmatrix(
#     k::HAC,
#     m::AbstractMatrix{T},
#     factorization=Cholesky;
#     prewhite::Bool=false,
#     demean::Bool=true,
#     scale::Real=one(T),
# ) where T<:Real
#     mm = demean ? m .- mean(m, dims = 1) : m
#     scale *= inv(size(mm,1))
#     return _lrvarmatrix(k, mm, prewhite, scale, factorization)
# end

# function _lrvarmatrix(
#     k::HAC,
#     mm::AbstractMatrix,
#     prewhite::Bool,
#     scale::Real,
#     ::Type{Cholesky},
# )
#     V = _lrvar(k, mm, prewhite, scale)
#     CovarianceMatrix(cholesky(V, check=true), k, V)
# end

# function _lrvarmatrix(
#     k::HAC,
#     mm::AbstractMatrix,
#     prewhite::Bool,
#     scale::Real,
#     ::Type{SVD},
# )
#     V = _lrvar(k, mm, prewhite, scale)
#     CovarianceMatrix(svd(V.data), k, V)
# end

# #=======
# B. VARHC
# =======#

# function lrvar(
#     k::VARHAC,
#     m::AbstractMatrix{T},
#     demean::Bool=true,
#     scale::Real=one(T)
# ) where T<:Real
#     mm = demean ? m .- mean(m, dims = 1) : m
#     return _lrvar(k, mm, scale)
# end

# function _lrvar(k::VARHAC, m, scale)
#     maxlag, lagstrategy, selectionstrategy = k.maxlag, k.lagstrategy, k.selectionstrategy
#     strategy = selectionstrategy == :aic ? 1 : (selectionstrategy == :bic ? 2 : 3)
#     return Symmetric(varhac(m,maxlag,lagstrategy,strategy).*scale)
# end

# #=======
# C. HC
# =======#
# function lrvar(
#     k::HC,
#     m::AbstractMatrix{T};
#     demean::Bool=true,
#     scale::Real=one(T)
# ) where T<:Real
#     mm = demean ? m .- mean(m, dims=1) : m
#     scale *= inv(size(mm, 1))
#     return _lrvar(k, mm, scale)
# end

# function _lrvar(k::HC, m::AbstractMatrix{T}, scale) where T<:Real
#     P = parent(m)
#     V = P'*P
#     if T <: AbstractFloat
#         Symmetric(V.*convert(T, scale))
#     else
#         Symmetric(V.*scale)
#     end
# end

# function lrvarmatrix(
#     k::HC,
#     m::AbstractMatrix{T},
#     factorization=Cholesky;
#     demean::Bool=true,
#     scale::Real=one(T)
# ) where T<:Real    
#     mm = demean ? m .- mean(m, dims=1) : m
#     scale *= inv(size(mm, 1))
#     return _lrvarmatrix(k, mm, scale, factorization)
# end

# function _lrvarmatrix(k::HC, m::AbstractMatrix, scale, ::Type{Cholesky})
#     V = _lrvar(k, m, scale)
#     return CovarianceMatrix(cholesky(V, check=true), k, V)
# end

# function _lrvarmatrix(k::HC, m::AbstractMatrix, scale, ::Type{SVD})
#     V = _lrvar(k, m, scale)
#     return CovarianceMatrix(svd(V.data), k, V)
# end
# #======
# D. CRHC
# =======#
# function lrvar(
#     k::CRHC,
#     m::AbstractMatrix{T};
#     demean::Bool=true,
#     scale::Real=one(T),
# ) where T
#     mm = demean ? m .- mean(m, dims=1) : m
#     scale *= inv(size(mm,1)) ### *scale  <- ????
#     return _lrvar(k, mm, scale)
# end

# function _lrvar(k::CRHC, m::AbstractMatrix{T}, scale) where T<:Real
#     cache = installcache(k, m)
#     Shat = clusterize!(cache)
#     if T <: AbstractFloat
#         Symmetric(Shat.*convert(T, scale))
#     else
#         Symmetric(Shat.*scale)
#     end
# end

# function lrvarmatrix(
#     k::CRHC,
#     m::AbstractMatrix{T},
#     factorization=Cholesky;
#     demean::Bool=true,
#     scale::Real=one(T),
# ) where T<:Real
#     scale *= inv(size(m, 1))
#     mm = demean ? m .- mean(m, dims=1) : m
#     return _lrvarmatrix(k, mm, scale, factorization)
# end

# function _lrvarmatrix(k::CRHC, m::AbstractMatrix, scale, ::Type{Cholesky})
#     V = _lrvar(k, m, scale)
#     CovarianceMatrix(cholesky(V, check=true), k, V)
# end

# function _lrvarmatrix(k::CRHC, m::AbstractMatrix, scale, ::Type{SVD})
#     V = _lrvar(k, m, scale)
#     CovarianceMatrix(SVD(V.data), k, V)
# end
