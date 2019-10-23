mutable struct CRHCCache{M<:AbstractMatrix, V<:AbstractVector, C, IN}
    momentmatrix::M       # nxm
    "For regression type"
    modelmatrix::M
    residuals::V
    "Scratch matrix"
    matscr::M
    "Factorization of X'X"
    crossx::C         # Factorization of X'*X
    "Meat"
    Shat::M
    "The cluster indices (*sorted*)"
    clusters_indices::IN
    "The *sorted* cluster indicator"
    f::CategoricalArray
end

"""
    bysort(x, f)

Sort each element of `x` according to f (a categorical).
# Arguments

- `x` an iterable whose elements are arrays.
- `f::CategoricalArray` a categorical array defining the sorting order

Note: If the element of x are sorted, return x. Otherwise, returns a tuple
whose elements are sorted according to f.

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

categorize(a::AbstractArray) = categorical(a)
categorize(f::CategoricalArray) = f

function clusters_intervals(f::CategoricalArray)
    s = CategoricalArrays.order(f.pool)[f.refs]
    ci = collect(searchsorted(s, j) for j in unique(s))
    return ci
end


"""
When only a matrix is passed, this is assumed to be a moment function
"""
function install_cache(k::CRHC, X::AbstractMatrix{T}) where T
    f = categorize(k.cl)
    (X, ), sf = bysort((X,), f)
    ci = clusters_intervals(sf)
    n, p = size(X)
    chol = cholesky(diagm(0=>repeat([1], inner=[p])))
    em = Matrix{Float64}(undef,0,0)
    ev = T[]
    Shat= Matrix{T}(undef,p,p)
    CRHCCache(X, em, ev, em, chol, Shat, ci, sf)
end

# function validate_cache(k::CRHC, X::AbstractMatrix{T}, cache::CRHCCache) where T
#     @assert size(X) == size(cache.momentmatrix) "CRHCCache: wrong dimension"
#     @assert k.cl == cache.f "CRHCCache: wrong order"
#     # The cache can only be applied to problems that are presorted
#     # so that we can directly copy into it without resorting.
#     Base.unsafe_copyto!(cache.momentmatrix, X)
#     return cache
# end

"""
clusters_indices(c::CRHCCache)

Return an array whose element gives the indices (as a Range{Int}) of the i-th cluster. Since the data is sorted when cached, the indices are contigous.
"""
clusters_indices(c::CRHCCache) = c.clusters_indices

StatsModels.modelmatrix(c::CRHCCache) = c.modelmatrix
momentmatrix(c::CRHCCache) = c.momentmatrix

# Cannot call this function residuals because
# it is only exported by GLM
resid(c::CRHCCache) = c.residuals
#residscr(c::CRHCCache) = c.residualscr
invcrossx(c::CRHCCache) = inv(crossx(c))
crossx(c::CRHCCache) = c.crossx
clusters_categorical(c::CRHCCache) = c.clusters_indices

function dofadjustment(k::CRHC0, c::CRHCCache)
    g = length(clusters_indices(c))
    return g/(g-1)
end

function dofadjustment(k::CRHC1, c::CRHCCache)
    g, (n, p) = length(clusters_indices(c)), size(modelmatrix(c))
    return (n-1)/(n-p) * g/(g-1)
end

dofadjustment(k::CRHC2, c::CRHCCache) = 1

function dofadjustment(k::CRHC3, c::CRHCCache)
     g, (n, p) = length(clusters_indices(c)), size(modelmatrix(c))
    return g/(g-1)
end

adjust_resid!(k::CRHC0, c::CRHCCache) = resid(c)
adjust_resid!(k::CRHC1, c::CRHCCache) = resid(c)
function adjust_resid!(v::CRHC2, c::CRHCCache)
    n, p = size(momentmatrix(c))
    X, u = modelmatrix(c), resid(c)
    invxx, indices = invcrossx(c), clusters_indices(c)
    for index in indices
        Xv = view(X, index, :)
        uv = view(u, index, :)
        xAx = Xv*invxx*Xv'
        ldiv!(cholesky!(Symmetric(I - xAx)).L, uv)
    end
    return u
end
function adjust_resid!(k::CRHC3, c::CRHCCache)
    X, u = modelmatrix(c), resid(c)
    n, p = size(X)
    invxx, indices = invcrossx(c), clusters_indices(c)
    for index in indices
        Xv = view(X, index, :)
        uv = view(u, index, :)
        xAx = Xv*invxx*Xv'
        ldiv!(cholesky!(Symmetric(I - xAx)), uv)
    end
    return rmul!(u, 1/sqrt(dofadjustment(k, c)))
end

Base.@propagate_inbounds function clusterize!(c::CRHCCache)
    U = momentmatrix(c)
    zeroM = zero(eltype(U))
    M = fill!(c.Shat, zeroM)
    p = size(M, 1)
    s = Array{eltype(U)}(undef, p)
    for m in clusters_indices(c)
        fill!(s, zeroM)
        for j in eachindex(s), i in m
            s[j] += U[i, j]
        end
        for j in eachindex(s), i in eachindex(s)
            M[i, j] += s[i]*s[j]
        end
    end
    return M
end

## Generic __vcov - Used by GLM, but more generic
function __vcov(k::CRHC, cache, df)
    B = Matrix(cache.crossx)
    res = adjust_resid!(k, cache)
    cache.momentmatrix .= cache.modelmatrix.*res
    Shat = clusterize!(cache)
    return Symmetric(B*Shat*B).*df
end
