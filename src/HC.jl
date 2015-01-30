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

function clusterize!(M, U, bstarts)
    k, k = size(M)
    s = Array(Float64, k)
    V = Array(Float64, k, k)
    for m = 1:length(bstarts)
        for j = 1:k
            for i = 1:k
                @inbounds V[i, j] = zero(Float64)
            end
        end

         for i = 1:k
             @inbounds s[i] = zero(Float64)
         end

        for j = 1:k
            for i = [bstarts[m]]
                @inbounds s[j] += U[i, j]
            end 
        end
        
        for j = 1:k
            for i = 1:k
                @inbounds V[i, j] += s[i]*s[j]
            end
        end 
        
        for j = 1:k
            for i = 1:k
                @inbounds M[i, j] += V[i, j]
            end
        end
    end
end 

getQ(v::CRHC2, X, ichol) = inv(chol(eye(size(X)[1])-X*ichol*X'))
getQ(v::CRHC3, X, ichol) = inv(eye(size(X)[1])-X*ichol*X')

function getQ(v::CRHC3, e, X, A, bstarts)
    for j in 1:length(bstarts)        
        rnge = bstarts[j]
        se = sub(e, rnge)
        se = sub(X, rnge, :)
        e[rnge] = se - sx*A*sx'\se
    end
end

function getQ(v::CRHC3, e, X, A, bstarts)
    Ai = inv(A)    
    for j in 1:length(bstarts)
        rnge = bstarts[j]
        se = sub(e, rnge)
        sx = sub(X, rnge,:)
        In = eye(length(rnge))
        e[rnge] =  (In - sx*Ai*sx')\se
    end
end

function _adjresid!(v::CRHC, X, e, chol, bstarts)
    getQ(v, e, X, chol, bstarts)
    
    ## for j in 1:length(bstarts)        
    ##     rnge = bstarts[j]
    ##     e[rnge] = getQ(v, sub(X, rnge, :), ichol)*sub(e, rnge)
    ## end
    ## return e
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
    M = zeros(size(X, 2), size(X, 2))
    #block_crossprod!(M, e.*X, bstarts)
    clusterize!(M, X.*e, bstarts)
    return M/nobs(x)
end 

function vcov(x::LinPredModel, v::CRHC)
    B = bread(x)
    M = meat(x, v)
    B*M*B/nobs(x)
end


    
        
        




