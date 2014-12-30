function vcov(X::AbstractMatrix, k::HC)
	N, p = size(X)
	return X'X/N
end

typealias GenLinMod GeneralizedLinearModel

adjfactor(LM::GenLinMod, k::HC0) = one(one(Float64))
adjfactor(LM::GenLinMod, k::HC1) = sqrt(nobs(LM)./df_residual(LM))
adjfactor(LM::GenLinMod, k::HC2) = sqrt(1./(1.-hatmatrix(LM)))
adjfactor(LM::GenLinMod, k::HC3) = 1./(1.-hatmatrix(LM))


function adjfactor(LM::GenLinMod, k::HC4)
	hᵢ = hatmatrix(LM)
	h  = mean(hᵢ)
	δ  = min(4, hᵢ/h)
	sqrt(1./(1.-hᵢ).^δ)
end

nobs(LM::GenLinMod) = length(LM.rr.y)

function bread(LM::GenLinMod)
	nobs(LM)*inv(cholfact(LM.pp))
end

function residuals(LM::GenLinMod, k::HC)
	ε = (LM.rr.wrkresid.*LM.rr.wrkwts).*adjfactor(LM, k)
	diagm(ε)
end

function residuals(LM::GenLinMod, k::HAC)
	ε = (LM.rr.wrkresid.*LM.rr.wrkwts)
	diagm(ε)
end

function hatmatrix(LM::GenLinMod)
	X = ModelMatrix(LM).*sqrt(LM.rr.wrkwts)
	diag(X*inv(cholfact(LM.pp))*X')
end

function meat(LM::GenLinMod, k::RobustVariance)
	ε = residuals(LM, k)
	X = ModelMatrix(LM)
	vcov(ε*X, k)
end

function vcov(LM::DataFrameRegressionModel, k::RobustVariance)
	Ω = meat(LM.model, k)
	Q = bread(LM.model)
	Q*Ω*Q/nobs(LM)
end

stderr(LM::DataFrameRegressionModel, k::RobustVariance) = sqrt(diag(vcov(LM, k)))






