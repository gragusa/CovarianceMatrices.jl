struct CRHCCache{F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector, IN<:AbstractVector}
    q::F1
    X::F1
    x::F2
    v::V
    w::V
    η::V
    u::V
    M::F1
    clusidx::IN
    clus::IN
end

function CRHCCache(X::AbstractMatrix{T1}; returntype::Type{T1} = eltype(X)) where T1
    n, p = size(X)
    CRHCCache(similar(X), similar(X), Array{T1, 2}(undef, p, p),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 2}(undef, p, p), Array{Int, 1}(undef, n),
             Array{Int, 1}(undef, n))
end

function installsortedxuw!(cache, m, k, ::Type{Val{true}})
    installxuw!(cache, m)
    copyto!(cache.clus, k.cl)
end

function installsortedxuw!(cache, m, k, ::Type{Val{false}})
    n, p = size(cache.X)
    sortperm!(cache.clusidx, k.cl)
    cidx = cache.clusidx
    u = residuals(m)
    w = getweights(m)
    X = modelmatrix(m)
    uu = cache.u
    XX = cache.X
    ww = cache.w
    c  = k.cl
    cc = cache.clus

    @inbounds for i in eachindex(cidx)
        uu[i] = u[cidx[i]]
    end
    @inbounds for j in 1:p, i in eachindex(cache.clusidx)
        XX[i,j] = X[cidx[i], p]
    end
    @inbounds for i in eachindex(cache.clusidx)
        cc[i] = c[cidx[i]]
    end

    if !isempty(ww)
        @inbounds for i in eachindex(cidx)
            ww[i] = sqrt.(w[cidx[i]])
        end
        broadcast!(*, XX, XX, ww)
        broadcast!(*, uu, uu, ww)
    end
end

adjresid!(k::CRHC0, cache, ichol, bstarts) = nothing
adjresid!(k::CRHC1, cache, ichol, bstarts) = cache.u .= scalaradjustment(cache, bstarts)*cache.u
adjresid!(k::CRHC2, cache, ichol, bstarts) = getqii(k, cache, ichol, bstarts)
adjresid!(k::CRHC3, cache, ichol, bstarts) = scalaradjustment(cache, bstarts).*getqii(k, cache, ichol, bstarts)

function scalaradjustment(cache, bstarts)
    n, p = size(cache.X)
    g    = length(bstarts)
    sqrt.((n-1)/(n-p) * g/(g-1))
end


function getqii(v::CRHC2, cache, A, bstarts)
    X, u  = cache.X, cache.u
    @inbounds for rnge in bstarts
        uv = view(u, rnge)
        xv = view(X, rnge, :)
        uu = copy(uv)
        xx = copy(xv)
        BB = Symmetric(I - xx*A*xx')
        uv .= cholesky!(BB).U\uu
    end
    return u
end

function getqii(v::CRHC3, cache, A, bstarts)
    X, u  = cache.X, cache.u
    @inbounds for rnge in bstarts
        uv = view(u, rnge)
        xv = view(X, rnge, :)
        uu = copy(uv)
        xx = copy(xv)
        ldiv!(cholesky!(Symmetric(I - xx*A*xx')), uu)
        uv .= uu
    end
    return u
end

function clusterize!(cache, bstarts)
    M = cache.M
    fill!(M, zero(Float64))
    U = cache.q
    p, p = size(M)
    s = Array{Float64}(undef, p)
    for m in bstarts
        for i = 1:p
            @inbounds s[i] = zero(Float64)
        end
        for j = 1:p, i in m
            @inbounds s[j] += U[i, j]
        end
        for j = 1:p, i = 1:p
            @inbounds M[i, j] += s[i]*s[j]
        end
    end
    return M
end
