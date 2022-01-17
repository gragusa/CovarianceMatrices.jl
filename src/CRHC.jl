mutable struct CRHCCache{M<:AbstractMatrix, V<:AbstractVector, C, IN}
    momentmatrix::M
    "For regression type"
    modelmatrix::M
    residuals::V
    "Scratch matrix"
    matscr::M
    "Factorization of X'X"
    crossx::C
    "Meat"
    Shat::M
    "The cluster indices (*sorted*)"
    clustersindices::IN
    "The *sorted* cluster indicator"
    f::CategoricalArray
end

"""
    bysort(x, f)

Sort each element of `x` according to f (a categorical).

# Arguments
- `x` an iterable whose elements are arrays.
- `f::CategoricalArray` a categorical array defining the sorting order

# Returns
- `Tuple`: a tuple (xs, fs) containing the sorted element of x and f
"""
function bysort(x, f)
    issorted(f) && return x, f
    @assert all(map(x->isa(x,AbstractArray), x))
    @assert all(map(x->size(x,1)==length(f), x))
    idx = sortperm(f)
    rt = map(x->similar(x), x)
    for ix in eachindex(rt)
        xin = x[ix]
        xout= rt[ix]
        for j in 1:size(xin,2)
            for i in eachindex(idx)
                xout[i,j] = xin[idx[i],j]
            end
        end
    end
    return rt, f[idx]
end

categorize(a::AbstractArray) = categorical(a, compress=true)
categorize(f::CategoricalArray) = f

function clustersintervals(f::CategoricalArray)
    ## This operation was deprecated oby CategoricalArrays
    ## s = CategoricalArrays.order(f.pool)[f.refs]
    ## I think f.refs is equivalent....
    s = f.refs
    ci = collect(searchsorted(s, j) for j in unique(s))
    return ci
end

function installcache(k::CRHC, X::AbstractMatrix{T}) where T
    f = categorize(k.cl)
    (X, ), sf = bysort((X,), f)
    ci = clustersintervals(sf)
    n, p = size(X)
    chol = cholesky(diagm(0=>repeat([one(T)], inner=[p])))
    em = similar(X)
    ev = T[]
    Shat= Matrix{T}(undef,p,p)
    Smat = Matrix{T}(undef, 0, 0)
    CRHCCache(X, em, ev, Smat, chol, Shat, ci, sf)
end

"""
    clustersindices(c::CRHCCache)

Return an array whose element `i` is a `Range{Int}` with indeces of the i-th cluster. Since
the data is sorted when cached, the indices are contigous.
"""
clustersindices(c::CRHCCache) = c.clustersindices
StatsModels.modelmatrix(c::CRHCCache) = c.modelmatrix
momentmatrix(c::CRHCCache) = c.momentmatrix
resid(c::CRHCCache) = c.residuals
invcrossx(c::CRHCCache) = inv(crossx(c))
crossx(c::CRHCCache) = c.crossx
#clusters_categorical(c::CRHCCache) = c.clustersindices ## TODO: remove it (??)

"""
    dofadjustment(k::CRHC, ::CRHCCache)

Calculate the default degrees-of-freedom adjsutment for `CRHC`

# Arguments
- `k::CRHC`: cluster robust variance type
- `c::CRHCCache`: the `CRHCCache` from which to extract the information
# Return
- `Float`: the degrees-of-fredom adjustment
# Note: the adjustment is a multyplicative factor.
"""
function dofadjustment(k::CRHC0, c::CRHCCache)
    g = length(clustersindices(c))::Int64
    return g/(g-1)
end

function dofadjustment(k::CRHC1, c::CRHCCache)
    g, (n, p) = length(clustersindices(c)), size(modelmatrix(c))
    return ((n-1)/(n-p) * g/(g-1))
end

dofadjustment(k::CRHC2, c::CRHCCache) = 1

function dofadjustment(k::CRHC3, c::CRHCCache)
     g, (n, p) = length(clustersindices(c)), size(modelmatrix(c))
    return (g/(g-1))
end

adjustresid!(k::CRHC0, c::CRHCCache) = resid(c)
adjustresid!(k::CRHC1, c::CRHCCache) = resid(c)

function adjustresid!(v::CRHC2, c::CRHCCache)
    n, p = size(momentmatrix(c))
    X, u = modelmatrix(c), resid(c)
    invxx, indices = invcrossx(c), clustersindices(c)
    Threads.@threads for index in indices
        Xv = view(X, index, :)
        uv = view(u, index, :)
        xAx = Xv*invxx*Xv'
        ldiv!(cholesky!(Symmetric(I - xAx); check=false).L, uv)
    end
    return u
end

Base.@propagate_inbounds function adjustresid!(k::CRHC3, c::CRHCCache)
    X, u = modelmatrix(c), resid(c)
    n, p = size(X)
    invxx, indices = invcrossx(c), clustersindices(c)
    Threads.@threads for index in indices
        Xv = view(X, index, :)
        uv = view(u, index, :)
        xAx = Xv*invxx*Xv'
        ldiv!(cholesky!(Symmetric(I - xAx); check=false), uv)
    end
    return rmul!(u, 1/sqrt(dofadjustment(k, c)))
end

Base.@propagate_inbounds function clusterize!(c::CRHCCache)
    U = momentmatrix(c)
    zeroM = zero(eltype(U))
    M = fill!(c.Shat, zeroM)
    p = size(M, 1)
    s = Array{eltype(U)}(undef, p)
    for m in clustersindices(c)
        fill!(s, zeroM)
        for j in 1:p
            for i in m
                s[j] += U[i, j]
            end
        end
        for j in 1:p
            for i in 1:j
                M[i, j] += s[i]*s[j]
            end
        end
    end
    return LinearAlgebra.copytri!(M, 'U')
end

renew(::CRHC0, id) = CRHC0(id, nothing)
renew(::CRHC1, id) = CRHC1(id, nothing)
renew(::CRHC2, id) = CRHC2(id, nothing)
renew(::CRHC3, id) = CRHC3(id, nothing)
