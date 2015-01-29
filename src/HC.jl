function vcov(X::AbstractMatrix, k::HC)
	N, p = size(X)
	return X'X/N
end

typealias GenLinMod GeneralizedLinearModel

adjfactor(l::LinPredModel, k::HC0) = one(Float64)
adjfactor(l::LinPredModel, k::HC1) = sqrt(nobs(l)./df_residual(l))
adjfactor(l::LinPredModel, k::HC2) = sqrt(1./(1.-hatmatrix(l)))
adjfactor(l::LinPredModel, k::HC3) = 1./(1.-hatmatrix(l))

adjfactor(l::LinPredModel, k::CRHC0) = one(Float64)
adjfactor(l::LinPredModel, k::CRHC1) = ( ( nobs(l)-1 ) / ( nobs(l)-npars(l) ) ) * ( nclus(k)/(nclus(k)-1 ) )


function adjfactor(lp::LinPredModel, k::HC4)
    hᵢ = hatmatrix(lp)
    h  = mean(hᵢ)
    δ  = min(4, hᵢ/h)
    sqrt(1./(1.-hᵢ).^δ)
end

nclus(k::CRHC) = length(unique(k.cl))
npars(x::LinPredModel) = length(x.pp.beta0)


## To be removed?
##StatsBase.nobs(lp::LinPredModel) = length(lp.rr.y)

function bread(lp::LinPredModel)
    nobs(lp)*inv(cholfact(lp.pp))
end

residuals(l::LinPredModel, k::HC)  = diagm(wrkresidwts(l.rr).*adjfactor(l, k))
residuals(l::LinPredModel, k::HAC) = diagm(wrkresidwts(l.rr))

wrkresidwts(r::GLM.ModResp) = r.wrkwts.*r.wrkresid
wrkresidwts(r::IVResp) = length(r.wts) == 0 ? r.wrkresid : r.wrkresid.*r.wrkwts


wrkresid(r::GLM.ModResp) = r.wrkresid

wrkwts(r::GLM.ModResp) = r.wrkwts
wrkwts(r::IVResp) = r.wts

function hatmatrix(l::LinPredModel) 
    w = wrkwts(l.rr)
    X = (length(w) == 0 ? ModelMatrix(l) : ModelMatrix(l).*sqrt(w))
    diag(X*inv(cholfact(l.pp))*X')
end 

vcov(ll::LinPredModel, k::HC) = (Ω = meat(ll, k); Q = bread(ll); Q*Ω*Q/nobs(ll))
meat(l::LinPredModel,  k::HC) =  vcov(residuals(l, k)*ModelMatrix(l), k)
meat(l::LinearIVModel, k::HC) = vcov(residuals(l, k)*l.pp.Xp, k)

function hatmatrix(l::LinearIVModel)
    w = wrkwts(l.rr)
    if length(w) == 0
        Xp = l.pp.Xp
    else
        Xp = l.pp.Xp.*sqrt(w)
    end
    diag(Xp*inv(cholfact(l.pp))*Xp') 
end

vcov(x::DataFrameRegressionModel, k::RobustVariance) = vcov(x.model, k)
stderr(x::DataFrameRegressionModel, k::RobustVariance) = sqrt(diag(vcov(x, k)))

## Cluster


function block_crossprod!(out, X, bstarts)
    for j in 1:length(bstarts)
        Base.BLAS.syrk!('U', 'T', 1.,
                        sub(X, bstarts[j],:), 1., out)
    end
end

getQ(v::CRHC2, X, ichol) = inv(chol(eye(size(X)[1])-X*ichol*X'))
getQ(v::CRHC3, X, ichol) = inv(eye(size(X)[1])-X*ichol*X')

function _adjresid!(v::CRHC, X, e, ichol, bstarts)
    for j in 1:length(bstarts)        
        rnge = bstarts[j]
        ## println(rnge)
        ## es = sub(e, rnge)
        ## println(length(es))
        ## QQ = getQ(v, sub(X, rnge, :), ichol)
        ## println(size(QQ))
        ## println(es.*QQ)
        e[rnge] = getQ(v, sub(X, rnge, :), ichol)*sub(e, rnge)
    end
    return e
end 

_adjresid!(v::CRHC, X, e, ichol, bstarts, c::Float64) = scale!(c, _adjresid!(v::CRHC, X, e, ichol, bstarts))

function scalar_adjustment(X, bstarts)
    n, k = size(X);
    g    = length(bstarts);
    (n-1)/(n-k) * g/(g-1)
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
    w = wrkwts(x.rr)
    if length(w) > 0
        X = X[idx,:].*sqrt(w[idx])
    end
    bstarts = [searchsorted(cls, j[2]) for j in enumerate(unique(cls))]
    e = wrkresid(x.rr)[idx]
    adjresid!(v, X, e, ichol, bstarts)
    M = similar(X, size(X)[2], size(X)[2])
    block_crossprod!(M, e.*X, bstarts)
    return M/nobs(x)
end 

function vcov(x::LinPredModel, v::CRHC)
    B = bread(x)
    M = meat(x, v)
    B*M*B/nobs(x)
end

    
    
    

    

    
        
        




