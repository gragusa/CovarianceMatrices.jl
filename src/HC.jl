function vcov(X::AbstractMatrix, k::HC)
	N, p = size(X)
	return X'X/N
end

typealias GenLinMod GeneralizedLinearModel

adjfactor(l::LinPredModel, k::HC0) = one(one(Float64))
adjfactor(l::LinPredModel, k::HC1) = sqrt(nobs(l)./df_residual(l))
adjfactor(l::LinPredModel, k::HC2) = sqrt(1./(1.-hatmatrix(l)))
adjfactor(l::LinPredModel, k::HC3) = 1./(1.-hatmatrix(l))

function adjfactor(lp::LinPredModel, k::HC4)
    hᵢ = hatmatrix(lp)
    h  = mean(hᵢ)
    δ  = min(4, hᵢ/h)
    sqrt(1./(1.-hᵢ).^δ)
end

## To be removed?
##StatsBase.nobs(lp::LinPredModel) = length(lp.rr.y)

function bread(lp::LinPredModel)
	nobs(lp)*inv(cholfact(lp.pp))
end

residuals(lp::LinPredModel, k::HC)  = diagm(residuals(lp.rr).*adjfactor(lp, k))
residuals(lp::LinPredModel, k::HAC) = diagm(residuals(lp.rr))

wrkwts(r::GLM.ModResp) = r.wrkwts
wrkwts(r::IVResp)  = r.wts

function hatmatrix(l::LinPredModel) 
    w = wrkwts(l.rr)
    X = (length(w) == 0 ? ModelMatrix(l) : ModelMatrix(l).*sqrt(w))
    diag(X*inv(cholfact(l.pp))*X')
end 

meat(l::LinPredModel, k::RobustVariance) =  vcov(residuals(l, k)*ModelMatrix(l), k)
vcov(ll::LinPredModel, k::RobustVariance) = (Ω = meat(ll, k); Q = bread(ll); Q*Ω*Q/nobs(ll))

meat(l::LinearIVModel, k::RobustVariance) = vcov(residuals(l, k)*l.pp.Xp, k)

function hatmatrix(l::LinearIVModel)
    w = wrkwts(l.rr)
    if length(w) == 0
        Xp = l.pp.Xp
    else
        Xp = l.pp.Xp.*sqrt(w)
    end
    diag(Xp*inv(cholfact(l.pp))*Xp') 
end

vcov(lp::DataFrameRegressionModel, k::RobustVariance) = vcov(lp.model, k)
stderr(x::DataFrameRegressionModel, k::RobustVariance) = sqrt(diag(vcov(x, k)))





