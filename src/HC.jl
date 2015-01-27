function vcov(X::AbstractMatrix, k::HC)
	N, p = size(X)
	return X'X/N
end

typealias GenLinMod GeneralizedLinearModel

adjfactor(lp::LinPredModel, k::HC0) = one(one(Float64))
adjfactor(lp::LinPredModel, k::HC1) = sqrt(nobs(lp)./df_residual(lp))
adjfactor(lp::LinPredModel, k::HC2) = sqrt(1./(1.-hatmatrix(lp)))
adjfactor(lp::LinPredModel, k::HC3) = 1./(1.-hatmatrix(lp))


function adjfactor(lp::LinPredModel, k::HC4)
	hᵢ = hatmatrix(lp)
	h  = mean(hᵢ)
	δ  = min(4, hᵢ/h)
	sqrt(1./(1.-hᵢ).^δ)
end

## To be removed?
StatsBase.nobs(lp::LinPredModel) = length(lp.rr.y)

function bread(lp::LinPredModel)
	nobs(lp)*inv(cholfact(lp.pp))
end

function residuals(lp::LinPredModel, k::HC)
	ε = (lp.rr.wrkresid.*lp.rr.wrkwts).*adjfactor(lp, k)
	diagm(ε)
end

function residuals(lp::LinPredModel, k::HAC)
	ε = (lp.rr.wrkresid.*lp.rr.wrkwts)
	diagm(ε)
end

function hatmatrix(lp::LinPredModel)
	X = ModelMatrix(lp).*sqrt(lp.rr.wrkwts)
	diag(X*inv(cholfact(lp.pp))*X')
end

function meat(lp::LinPredModel, k::RobustVariance)
	ε = residuals(lp, k)
	X = ModelMatrix(lp)
	vcov(ε*X, k)
end

function vcov(ll::LinearPredModel, k::RobustVariance)
	Ω = meat(ll, k)
	Q = bread(ll)
	Q*Ω*Q/nobs(ll)
end


vcov(lp::DataFrameRegressionModel, k::RobustVariance) = vcov(lp.model, k)
stderr(x::DataFrameRegressionModel, k::RobustVariance) = sqrt(diag(vcov(x, k)))





