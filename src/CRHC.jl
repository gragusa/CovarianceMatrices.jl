function installsortedxuw!(cache, m, k, ::Type{Val{true}})
    copyto!(cache.X, modelmatrix(m))
    copyto!(cache.u, residuals(m))
    copyto!(cache.clus, k.cl)
    if !isempty(smplweights(m))
        cache.w .= sqrt.(smplweights(m))
        broadcast!(*, cache.u, cache.u, cache.w)
        broadcast!(*, cache.u, cache.u, cache.w)
    end
end

function installsortedxuw!(cache, m, k, ::Type{Val{false}})
    n, p = size(cache.X)
    sortperm!(cache.clusidx, k.cl)
    cidx = cache.clusidx
    u = residuals(m)
    w = smplweights(m)
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
        XX[i,j] = X[cidx[i], j]
    end
    @inbounds for i in eachindex(cache.clusidx)
        cc[i] = c[cidx[i]]
    end
    if !isempty(w)
        @inbounds for i in eachindex(cidx)
            ww[i] = sqrt(w[cidx[i]])
        end
        broadcast!(*, XX, XX, ww)
        broadcast!(*, uu, uu, ww)
    end
end

adjresid!(k::CRHC0, cache, ichol, bstarts) = nothing
adjresid!(k::CRHC1, cache, ichol, bstarts) = nothing
adjresid!(k::CRHC2, cache, ichol, bstarts) = getqii(k, cache, ichol, bstarts)
adjresid!(k::CRHC3, cache, ichol, bstarts) = getqii(k, cache, ichol, bstarts)

function dof_adjustment(cache, k::CRHC0, bstarts)
    g = length(bstarts)
    g/(g-1)
end

function dof_adjustment(cache, k::CRHC1, bstarts)
    g, (n, p) = length(bstarts), size(cache.X)
    (n-1)/(n-p) * g/(g-1)
end

dof_adjustment(cache, k::T, bstarts) where T<:Union{CRHC2} = 1.0

function dof_adjustment(cache, k::CRHC3, bstarts)
    g, (n, p) = length(bstarts), size(cache.X)
    g/(g-1)
end

function getqii(v::CRHC2, cache, A, bstarts)
    X, u  = cache.X, cache.u
    @inbounds for rnge in bstarts
        uv = view(u, rnge)
        xv = view(X, rnge, :)
        uu = copy(uv)
        xx = copy(xv)
        BB = Symmetric(I - xx*A*xx')
        uv .= cholesky!(BB).L\uu
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
    M = cache.V
    fill!(M, zero(eltype(M)))
    U = cache.q
    p, p = size(M)
    s = Array{eltype(M)}(undef, p)
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