function vcov(X::AbstractMatrix, v::HC)
    N, p = size(X)
    XX = Base.LinAlg.At_mul_B(X, X)
    return scale!(XX, 1/N)
end

function Xt_A_X!(x, A)
    n = length(x)
    y = 0.0
    k = 0
    for i = 1 : n
        z = 0.0
        for j = 1 : n
            z += A[k + j] * x[j];
            z *= x[i]
            y += A[k + i] * x[i] * x[i] + z + z;
            k += n
        end
    end 
end 

typealias GenLinMod GeneralizedLinearModel

adjfactor!(u, l::LinPredModel, k::HC0) = u[:] = one(Float64)
adjfactor!(u, l::LinPredModel, k::HC1) = u[:] = nobs(l)./df_residual(l)
adjfactor!(u, l::LinPredModel, k::HC2) = u[:] = 1./(1.-hatmatrix(l))
adjfactor!(u, l::LinPredModel, k::HC3) = u[:] = 1./(1.-hatmatrix(l)).^2

function adjfactor!(u, lp::LinPredModel, k::HC4)
    h = hatmatrix(lp)
    n = nobs(lp)
    p = npars(lp)    
    for j = 1:n
        delta = min(4, n*h[j]/p)
        @inbounds u[j] = 1/(1-h[j])^delta
    end
end

function adjfactor!(u, lp::LinPredModel, k::HC4m)
    h = hatmatrix(lp)
    n = nobs(lp)
    p = npars(lp)    
    for j = 1:n
        delta = min(1.0, n*h[j]/p) + min(1.5, n*h[j]/p)
        @inbounds u[j] = 1/(1-h[j])^delta
    end
end

function adjfactor!(u, lp::LinPredModel, k::HC5)
    h     = hatmatrix(lp)
    n     = nobs(lp)
    p     = npars(lp)
    mx    = max(n*0.7*maximum(h)/p, 4)
    for j = 1:n
        alpha =  min(n*h[j]/p, mx)
        @inbounds u[j] = 1/(1-h[j])^alpha
    end 
end

nclus(k::CRHC) = length(unique(k.cl))
npars(x::LinPredModel) = length(x.pp.beta0)

function bread(lp::LinPredModel)
    A = inv(cholfact(lp.pp))
    scale!(A, nobs(lp))
end

function residuals(l::LinPredModel, k::HC)
    wrkresidwts(l.rr)
end 

residuals(l::LinPredModel, k::HAC) = wrkresidwts(l.rr)

function wrkresidwts(r::GLM.ModResp)
    a = r.wrkwts
    u = r.wrkresid
    return u.*a
end


wrkresid(r::GLM.ModResp) = r.wrkresid
wrkwts(r::GLM.ModResp) = r.wrkwts


function hatmatrix(l::LinPredModel)
    w = l.rr.wrkwts
    z = ModelMatrix(l).*sqrt(w)
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
    V = Array(Float64, k, k)
    for m = 1:length(bstarts)
        for j = 1:k, i = 1:k
                @inbounds V[i, j] = zero(Float64)
        end
        for i = 1:k
            @inbounds s[i] = zero(Float64)
        end
        for j = 1:k, i = bstarts[m]
            @inbounds s[j] += U[i, j]
        end        
        for j = 1:k, i = 1:k
            @inbounds V[i, j] += s[i]*s[j]
        end         
        for j = 1:k, i = 1:k
            @inbounds M[i, j] += V[i, j]
        end
    end
end 

function getqii(v::CRHC3, e, X, A, bstarts)
    ## A is inv(cholfac())
    for j in 1:length(bstarts)
        rnge = bstarts[j]
        se = sub(e, rnge)
        sx = sub(X, rnge,:)
        In = eye(length(rnge))
        ##gbmv!(trans, m, kl, ku, alpha, A, x, beta, y)
        e[rnge] =  (In - sx*A*sx')\se
    end
    return e
end 

function getqii(v::CRHC2, e, X, A, bstarts)
    ## A is inv(cholfac())
    for j in 1:length(bstarts)
        rnge = bstarts[j]
        se = sub(e, rnge)
        sx = sub(X, rnge,:)
        In = eye(length(rnge))
        e[rnge] =  chol(In - sx*A*sx')\se
    end
    return e
end

function _adjresid!(v::CRHC, X, e, chol, bstarts)
    getqii(v, e, X, chol, bstarts)    
end 

_adjresid!(v::CRHC, X, e, ichol, bstarts, c::Float64) = scale!(c, _adjresid!(v::CRHC, X, e, ichol, bstarts))

function scalar_adjustment(X, bstarts)
    n, k = size(X);
    g    = length(bstarts);
    sqrt((n-1)/(n-k) * g/(g-1))
end

adjresid!(v::CRHC0, X, e, ichol, bstarts) = identity(e)
adjresid!(v::CRHC1, X, e, ichol, bstarts) = e[:] = scalar_adjustment(X, bstarts)*e
adjresid!(v::CRHC2, X, e, ichol, bstarts) = _adjresid!(v, X, e, ichol, bstarts, 1.0)
adjresid!(v::CRHC3, X, e, ichol, bstarts) = _adjresid!(v, X, e, ichol, bstarts, scalar_adjustment(X, bstarts))

function meat(x::LinPredModel, v::CRHC)
    idx = sortperm(v.cl)
    cls = v.cl[idx]
    ichol = inv(x.pp.chol)
    X = ModelMatrix(x)[idx,:]
    e = wrkresid(x.rr)[idx]
    w = wrkwts(x.rr)[idx]
    if length(w) > 0
        X = X.*sqrt(w)
        e = e.*sqrt(w)
    end
    bstarts = [searchsorted(cls, j[2]) for j in enumerate(unique(cls))]
    adjresid!(v, X, e, ichol, bstarts)
    M = zeros(size(X, 2), size(X, 2))
    clusterize!(M, X.*e, bstarts)
    return scale!(M, 1/nobs(x))
end 



function vcov(x::LinPredModel, v::CRHC)
    B = bread(x)
    M = meat(x, v)
    scale!(B*M*B, 1/nobs(x))
end


    
        
        




