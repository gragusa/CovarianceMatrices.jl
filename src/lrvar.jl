#======
A. HAC
=======#

function lrvar(
    k::HAC,
    m::AbstractMatrix;
    prewhite::Bool = false,
    demean::Bool=true,
    scale::Real=1,
)
    mm = demean ? m .- mean(m, dims = 1) : m
    scale *= inv(size(m,1))
    return _lrvar(k, mm, prewhite, scale)
end

function _lrvar(k::HAC, mm, prewhite::Bool, scale)
    n, p = size(mm)
    m, D = prewhiter(mm, Val{prewhite})
    bw = optimalbandwidth(k, m; prewhite=prewhite)
    V = triu!(m'*m)
    @inbounds for j in covindices(k, n)
        κⱼ = kernel(k, j/bw)
        LinearAlgebra.axpy!(κⱼ, Γ(m, j), V)
    end
    LinearAlgebra.copytri!(V, 'U')
    dewhiter!(V, m, D, Val{prewhite})
    return Symmetric(V.*convert(eltype(V), scale))
end

function lrvarmatrix(
    k::HAC,
    m::AbstractMatrix,
    factorization=Cholesky;
    prewhite::Bool=false,
    demean::Bool=true,
    scale::Real=1,
)
    mm = demean ? m .- mean(m, dims = 1) : m
    scale *= inv(size(m,1))
    return _lrvarmatrix(k, mm, prewhite, scale, factorization)
end

function _lrvarmatrix(
    k::HAC,
    mm::AbstractMatrix,
    prewhite::Bool,
    scale::Real,
    ::Type{Cholesky},
)
    V = _lrvar(k, mm, prewhite, scale)
    CovarianceMatrix(cholesky(V, check=true), k, V)
end

function _lrvarmatrix(
    k::HAC,
    mm::AbstractMatrix,
    prewhite::Bool,
    scale::Real,
    ::Type{SVD},
)
    V = _lrvar(k, mm, prewhite, scale)
    CovarianceMatrix(svd(V.data), k, V)
end

#=======
B. VARHC
=======#
"""

"""
function lrvar(
    k::VARHAC,
    m::AbstractMatrix,
    demean::Bool=true,
    scale::Real=1
)
    mm = demean ? m .- mean(m, dims = 1) : m
    return _lrvar(k, mm, scale)
end

function _lrvar(k::VARHAC, m, scale)
    maxlag, lagstrategy, selectionstrategy = k.maxlag, k.lagstrategy, k.selectionstrategy
    strategy = selectionstrategy == :aic ? 1 : (selectionstrategy == :bic ? 2 : 3)
    return Symmetric(varhac(m,maxlag,lagstrategy,strategy).*scale)
end

#=======
C. HC
=======#
function lrvar(
    k::HC,
    m::AbstractMatrix;
    demean::Bool=true,
    scale::Real=1
)
    mm = demean ? m .- mean(m, dims=1) : m
    return _lrvar(k, mm, scale)
end

function _lrvar(k::HC, m::AbstractMatrix{T}, scale) where T
    V = m'*m
    F = promote_type(T, eltype(V))
    return Symmetric(V.*convert(F, inv(size(m,1))*scale))
end

function lrvarmatrix(
    k::HC,
    m::AbstractMatrix,
    factorization=Cholesky;
    demean::Bool=true,
    scale::Real=1
)
    mm = demean ? m .- mean(m, dims=1) : m
    return _lrvarmatrix(k, mm, scale, factorization)
end

function _lrvarmatrix(k::HC, m::AbstractMatrix, scale, ::Type{Cholesky})
    V = _lrvar(k, m, scale)
    return CovarianceMatrix(cholesky(V, check=true), k, V)
end

function _lrvarmatrix(k::HC, m::AbstractMatrix, scale, ::Type{SVD})
    V = _lrvar(k, m, scale)
    return CovarianceMatrix(svd(V.data), k, V)
end
#======
D. CRHC
=======#
function lrvar(
    k::CRHC,
    m::AbstractMatrix;
    demean::Bool=true,
    scale::Real=1,
)
    mm = demean ? m .- mean(m, dims=1) : m
    scale *= inv(size(m,1))*scale
    return _lrvar(k, mm, scale)
end

function _lrvar(k::CRHC, m::AbstractMatrix{T}, scale) where T
    cache = installcache(k, m)
    Shat = clusterize!(cache)
    F = promote_type(T, eltype(Shat))
    return Symmetric(Shat.*convert(F, scale))
end

function lrvarmatrix(
    k::CRHC,
    m::AbstractMatrix,
    factorization=Cholesky;
    demean::Bool=true,
    scale::Real=1,
)
    mm = demean ? m .- mean(m, dims=1) : m
    return _lrvarmatrix(k, mm, scale, factorization)
end

function _lrvarmatric(k::CRHC, m::AbstractMatrix, scale, ::Type{Cholesky})
    V = _lrvar(k, m, scale)
    CovarianceMatric(cholesky(V, check=true), k, V)
end

function _lrvarmatric(k::CRHC, m::AbstractMatrix, scale, ::Type{SVD})
    V = _lrvar(k, m, scale)
    CovarianceMatric(SVD(V.data), k, V)
end
#=========
Finalizers
==========#
# factorizer(::Type{SVD}, V) = svd(V.data)
# factorizer(::Type{Cholesky}, V) = cholesky(V, check = false)

# finalize(k, V, M, F) = finalize(k, V, M, F, 1)

# function finalize(k, V, ::Type{M}, F, scale) where M<:Matrix
#     return Symmetric(V.*scale)
# end

# function finalize(k, V, ::Type{M}, F, scale) where M<:CovarianceMatrix
#     V .= V.*scale
#     CovarianceMatrix(factorizer(F, V), k, Symmetric(V))
# end
