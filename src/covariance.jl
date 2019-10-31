#======
A. HAC
=======#
function covariance(
    k::HAC, m::AbstractMatrix;
    returntype=Matrix,
    factortype=Cholesky,
    prewhite::Bool = false,
    demean::Bool=true,
    scale::Real=inv(size(m,1)),
)
    mm = demean ? m .- mean(m, dims = 1) : m
    return __covariance(k, mm, returntype, factortype, prewhite, scale)
end

dewhiter!(V, mm, D, ::Type{Val{false}}) = V
function dewhiter!(V, mm, D, ::Type{Val{true}})
    ## TODO: This can be done better!
    ## V = ldiv!((I-D'), V)
    ## return V*v'
    v = inv(I-D')
    V .= v*V*v'
    return V
end
prewhiter(mm, ::Type{Val{false}}) = (mm, similar(mm, (0,0)))
prewhiter(mm, ::Type{Val{true}}) = fit_var(mm)

function __covariance(k, mm, rt, ft, pre, scale)
    n, p = size(mm)
    m, D = prewhiter(mm, Val{pre})
    bw = _optimal_bandwidth(k, m, pre)
    V = triu!(m'*m)
    @inbounds for j in covindices(k, n)
        κⱼ = kernel(k, j/bw)
        LinearAlgebra.axpy!(κⱼ, Γ(m, j), V)
    end
    LinearAlgebra.copytri!(V, 'U')
    dewhiter!(V, m, D, Val{pre})
    finalize(k, V, rt, ft, scale)
end


#=======
B. VARHC
=======#
function covariance(
    k::VARHAC, m::AbstractMatrix;
    returntype=Matrix,
    factortype=Cholesky,
    demean::Bool=true,
    scale::Real=1
)
    mm = demean ? m .- mean(m, dims = 1) : m
    maxlag, lagstrategy, selectionstrategy = k.maxlag, k.lagstrategy, k.selectionstrategy
    strategy = selectionstrategy == :aic ? 1 : (selectionstrategy == :bic ? 2 : 3)
    V = varhac(mm,maxlag,lagstrategy,strategy)
    finalize(k, V, returntype, factortype, scale)
end

#=======
C. HC
=======#
function covariance(
    k::HC, m::AbstractMatrix;
    returntype=Matrix,
    factortype=Cholesky,
    demean::Bool=true,
    scale::Real=inv(size(m, 1))
)
    mm = demean ? m .- mean(m, dims=1) : m
    V = mm'*mm
    finalize(k, V, returntype, factortype, scale)
end

#======
D. CRHC
=======#
function covariance(
    k::CRHC,
    m::AbstractMatrix{T};
    returntype=Matrix,
    factortype=Cholesky,
    demean::Bool=true,
    scale::Real=inv(size(m,1))
) where T
    mm = demean ? m .- mean(m, dims=1) : m    
    cache = install_cache(k, mm)
    Shat = clusterize!(cache)
    return finalize(k, Shat, returntype, factortype, convert(T, scale))
end

#=========
Finalizers
==========#
factorizer(::Type{SVD}) = svd
factorizer(::Type{Cholesky}) = x->cholesky(Symmetric(x), check = false)

finalize(k, V, M, F) = finalize(k, V, M, F, 1)

function finalize(k, V, ::Type{M}, F, scale) where M<:Matrix
    return Symmetric(V.*scale)
end

function finalize(k, V, ::Type{M}, F, scale) where M<:CovarianceMatrix
    V .= V.*scale
    CovarianceMatrix(factorizer(F)(V), k, Symmetric(V))
end
