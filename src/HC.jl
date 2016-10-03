function vcov(X::AbstractMatrix, v::HC)
    N, p = size(X)
    XX = Base.LinAlg.At_mul_B(X, X)
    return scale!(XX, 1/N)
end

typealias GenLinMod GeneralizedLinearModel

adjfactor!(u, l::LinPredModel, k::HC0) = u[:] = one(Float64)
adjfactor!(u, l::LinPredModel, k::HC1) = u[:] = _nobs(l)./_df_residual(l)
adjfactor!(u, l::LinPredModel, k::HC2) = u[:] = 1./(1.-hatmatrix(l))
adjfactor!(u, l::LinPredModel, k::HC3) = u[:] = 1./(1.-hatmatrix(l)).^2


_nobs(l::LinPredModel) = length(l.rr.y)
_df_residual(l::LinPredModel) = _nobs(l) - length(coef(l))

function adjfactor!(u, lp::LinPredModel, k::HC4)
    h = hatmatrix(lp)
    n = _nobs(lp)
    p = npars(lp)
    @inbounds for j in eachindex(h)
        delta = min(4, n*h[j]/p)
        u[j] = 1/(1-h[j])^delta
    end
end

function adjfactor!(u, lp::LinPredModel, k::HC4m)
    h = hatmatrix(lp)
    n = _nobs(lp)
    p = npars(lp)
    @inbounds for j in eachindex(h)
        delta = min(1.0, n*h[j]/p) + min(1.5, n*h[j]/p)
        u[j] = 1/(1-h[j])^delta
    end
end

function adjfactor!(u, lp::LinPredModel, k::HC5)
    h     = hatmatrix(lp)
    n     = _nobs(lp)
    p     = npars(lp)
    mx    = max(n*0.7*maximum(h)/p, 4)
    @inbounds for j in eachindex(h)
        alpha =  min(n*h[j]/p, mx)
        u[j] = 1/(1-h[j])^alpha
    end
end

nclus(k::CRHC) = length(unique(k.cl))
npars(x::LinPredModel) = length(x.pp.beta0)

function bread(lp::LinPredModel)
    A = inv(cholfact(lp.pp))::Array{Float64,2}
    scale!(A, nobs(lp))
end

residuals(l::LinPredModel, k::HC) = residuals(l)
residuals(l::LinPredModel, k::HAC) = residuals(l)

function residuals(r::GLM.ModResp)
    a = wrkwts(r)
    u = copy(wrkresid(r))
    length(a) == 0 ? u : broadcast!(*, u, u, a)
end

function wrkresidwts(r::GLM.ModResp)
    a = wrkwts(r)
    u = copy(wrkresid(r))
    length(a) == 0 ? u : broadcast!(*, u, u, a)
end

wrkresid(r::GLM.ModResp) = r.wrkresid
wrkwts(r::GLM.ModResp) = r.wrkwts
wrkwts(r::GLM.LmResp) = r.wts
wrkresid(r::GLM.LmResp) = r.y-r.mu

function weightedModelMatrix(l::LinPredModel)
    w = wrkwts(l.rr)
    length(w) > 0 ? ModelMatrix(l).*sqrt(w) : copy(ModelMatrix(l))
end

function hatmatrix(l::LinPredModel)
    z = weightedModelMatrix(l)
    cf = cholfact(l.pp)[:UL]
    Base.LinAlg.A_rdiv_B!(z, cf)
    diag(Base.LinAlg.A_mul_Bt(z, z))
end

function vcov(ll::LinPredModel, k::HC)
    B = meat(ll, k)
    A = bread(ll)
    scale!(A*B*A, 1/nobs(ll))
end

function meat(l::LinPredModel,  k::HC)
    u = residuals(l, k)
    X = ModelMatrix(l)
    z = X.*u
    adjfactor!(u, l, k)
    scale!(Base.LinAlg.At_mul_B(z, z.*u), 1/nobs(l))
end

vcov(x::DataFrameRegressionModel, k::RobustVariance) = vcov(x.model, k)
stderr(x::DataFrameRegressionModel, k::RobustVariance) = sqrt(diag(vcov(x, k)))
stderr(x::LinPredModel, k::RobustVariance) = sqrt(diag(vcov(x, k)))

################################################################################
## Cluster
################################################################################

function clusterize!(M, U, bstarts)
    k, k = size(M)
    s = Array(Float64, k)
    for m = 1:length(bstarts)
        @simd for i = 1:k
            @inbounds s[i] = zero(Float64)
        end
        for j = 1:k, i = bstarts[m]
            @inbounds s[j] += U[i, j]
        end
        for j = 1:k, i = 1:k
            @inbounds M[i, j] += s[i]*s[j]
        end
    end
end

function getqii(v::CRHC3, e, X, A, bstarts)
    @inbounds for j in 1:length(bstarts)
        rnge = bstarts[j]
        se = view(e, rnge)
        sx = view(X, rnge, :)
        e[rnge] =  (I - sx*A*sx')\se
    end
    return e
end

if VERSION < v"0.5.0"
    Base.cholfact(A::Symmetric, args...) = cholfact(A.data, Symbol(A.uplo), args...)
end

function getqii(v::CRHC2, e, X, A, bstarts)
    @inbounds for j in 1:length(bstarts)
        rnge = bstarts[j]
        se = view(e, rnge)
        sx = view(X, rnge,:)
        BB = Symmetric(I - sx*A*sx')
        e[rnge] =  cholfact(BB)\se
    end
    return e
end

_adjresid!(v::CRHC, X, e, chol, bstarts) =  getqii(v, e, X, chol, bstarts)
_adjresid!(v::CRHC, X, e, ichol, bstarts, c::Float64) = scale!(c, _adjresid!(v::CRHC, X, e, ichol, bstarts))

function scalar_adjustment(X, bstarts)
    n, k = size(X)
    g    = length(bstarts)
    sqrt((n-1)/(n-k) * g/(g-1))
end

adjresid!(v::CRHC0, X, e, ichol, bstarts) = identity(e)
adjresid!(v::CRHC1, X, e, ichol, bstarts) = e[:] = scalar_adjustment(X, bstarts)*e
adjresid!(v::CRHC2, X, e, ichol, bstarts) = _adjresid!(v, X, e, ichol, bstarts, 1.0)
adjresid!(v::CRHC3, X, e, ichol, bstarts) = _adjresid!(v, X, e, ichol, bstarts, scalar_adjustment(X, bstarts))

function meat(x::LinPredModel, v::CRHC)
    idx = sortperm(v.cl)
    cls = v.cl[idx]
    #ichol = inv(x.pp.chol)
    ichol  = inv(cholfact(x.pp))::Array{Float64,2}
    X = ModelMatrix(x)[idx,:]
    e = wrkresid(x.rr)[idx]
    w = wrkwts(x.rr)
    if !isempty(w)
        w = w[idx]
        broadcast!(*, X, X, sqrt(w))
        broadcast!(*, e, e, sqrt(w))
    end
    bstarts = [searchsorted(cls, j[2]) for j in enumerate(unique(cls))]
    adjresid!(v, X, e, ichol, bstarts)
    M = zeros(size(X, 2), size(X, 2))
    clusterize!(M, X.*e, bstarts)
    return scale!(M, 1/nobs(x))
end

function vcov(x::LinPredModel, v::CRHC)
    B = bread(x)::Array{Float64,2}
    M = meat(x, v)::Array{Float64,2}
    scale!(B*M*B, 1/nobs(x))
end
