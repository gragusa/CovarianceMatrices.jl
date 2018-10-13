struct CRHCConfig{F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector, IN<:AbstractVector}
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

function CRHCConfig(X::AbstractMatrix{T1}; returntype::Type{T1} = eltype(X)) where T1
    n, p = size(X)
    CRHCConfig(similar(X), similar(X), Array{T1, 2}(undef, p, p),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 2}(undef, p, p), Array{Int, 1}(undef, n),
             Array{Int, 1}(undef, n))
end

function CRHCConfig(m::StatsModels.DataFrameRegressionModel; returntype::Type{T1} = Float64) where T1
    s = size(ModelMatrix(m.mf).m)
    CRHCConfig(similar(Array{T1, 2}(undef, s...)); returntype = returntype)
end

function CRHCConfig(m::GLM.LinearModel; returntype::Type{T1} = Float64) where T1
    s = size(m.pp.X)
    CRHCConfig(similar(Array{T1, 2}(undef, s...)); returntype = T1)
end

function vcov(m::StatsModels.DataFrameRegressionModel, k::CRHC, cfg; sorted::Bool = false)
    B = CovarianceMatrices.pseudohessian(cfg, m)
    CovarianceMatrices.installsortedxuw!(cfg, m, k, Val{sorted})
    bstarts = (searchsorted(cfg.clus, j[2]) for j in enumerate(unique(cfg.clus)))
    CovarianceMatrices.adjresid!(k, cfg, B, bstarts)
    CovarianceMatrices.esteq!(cfg, m)
    CovarianceMatrices.clusterize!(cfg, bstarts)
    return B*cfg.M*B
end

function vcov(m::GLM.LinearModel, k::CRHC, cfg; sorted::Bool = false)
    B = CovarianceMatrices.pseudohessian(cfg, m)
    CovarianceMatrices.installsortedxuw!(cfg, m, k, Val{sorted})
    bstarts = (searchsorted(cfg.clus, j[2]) for j in enumerate(unique(cfg.clus)))
    CovarianceMatrices.adjresid!(k, cfg, B, bstarts)
    CovarianceMatrices.esteq!(cfg, m)
    CovarianceMatrices.clusterize!(cfg, bstarts)
    return B*cfg.M*B
end

function installsortedxuw!(cfg, m, k, ::Type{Val{true}})
    installxuw!(cfg, m)
    copyto!(cfg.clus, k.cl)
end

function installsortedxuw!(cfg, m, k, ::Type{Val{false}})
    n, p = size(cfg.X)
    sortperm!(cfg.clusidx, k.cl)
    cidx = cfg.clusidx
    u = residuals(m)
    w = getweights(m)
    X = modelmatrix(m)
    uu = cfg.u
    XX = cfg.X
    ww = cfg.w
    c  = k.cl
    cc = cfg.clus

    @inbounds for i in eachindex(cidx)
        uu[i] = u[cidx[i]]
    end
    @inbounds for j in 1:p, i in eachindex(cfg.clusidx)
        XX[i,j] = X[cidx[i], p]
    end
    @inbounds for i in eachindex(cfg.clusidx)
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

adjresid!(k::CRHC0, cfg, ichol, bstarts) = nothing
adjresid!(k::CRHC1, cfg, ichol, bstarts) = cfg.u .= scalaradjustment(cfg, bstarts)*cfg.u
adjresid!(k::CRHC2, cfg, ichol, bstarts) = getqii(k, cfg, ichol, bstarts)
adjresid!(k::CRHC3, cfg, ichol, bstarts) = scalaradjustment(cfg, bstarts).*getqii(k, cfg, ichol, bstarts)

function scalaradjustment(cfg, bstarts)
    n, p = size(cfg.X)
    g    = length(bstarts)
    sqrt.((n-1)/(n-p) * g/(g-1))
end


function getqii(v::CRHC2, cfg, A, bstarts)
    X, u  = cfg.X, cfg.u
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

function getqii(v::CRHC3, cfg, A, bstarts)
    X, u  = cfg.X, cfg.u
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

function clusterize!(cfg, bstarts)
    M = cfg.M
    fill!(M, zero(Float64))
    U = cfg.q
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
