module GLMExt

using CovarianceMatrices, GLM, LinearAlgebra
using Statistics

##=================================================
## Moment Matrix 
##=================================================
const FAM = Union{GLM.Gamma, GLM.Bernoulli, GLM.InverseGaussian}

# modelweights(r::GLM.RegressionModel) = modelweights(r.model)
# modelweights(m::GLM.LinearModel) = isempty(m.rr.wts) ? ones(eltype(m.rr.wts), 1) : m.rr.wts
# modelweights(m::GLM.GeneralizedLinearModel) = m.rr.wts

# CovarianceMatrices.modelresiduals(r::GLM.RegressionModel) = CovarianceMatrices.modelresiduals(r.model)
# CovarianceMatrices.modelresiduals(m::GLM.LinearModel) = GLM.residuals(m).*sqrt.(weights(m))
# CovarianceMatrices.modelresiduals(m::GLM.GeneralizedLinearModel) = m.rr.wrkresid.*sqrt.(m.rr.wrkwt).*sqrt.(modelweights(m))



numobs(r::GLM.RegressionModel) = size(r.model.pp.X, 1)
numobs(m::GLM.LinPredModel) = size(m.pp.X, 1)

_dispersion(r::GLM.RegressionModel) = _dispersion(r.model)
_dispersion(m::GLM.LinearModel) = 1
_dispersion(m::GLM.GeneralizedLinearModel) = _dispersion(m.rr)

_dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2, T3} = 1
_dispersion(rr::GLM.LmResp) = 1
function _dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2 <: FAM, T3}
    sum(abs2, rr.wrkwt .* rr.wrkresid) / sum(rr.wrkwt)
end

CovarianceMatrices.bread(r::RegressionModel) = CovarianceMatrices.bread(r.model)
CovarianceMatrices.bread(m::GLM.LinearModel) = GLM.invchol(m.pp)
CovarianceMatrices.bread(m::GLM.GeneralizedLinearModel) = GLM.invchol(m.pp) .* _dispersion(m)

GLM.residuals(m::GLM.GeneralizedLinearModel) = m.rr.wrkresid

mask(r::RegressionModel) = mask(r.model)
mask(m::GLM.LinPredModel) = mask(m.pp)

function mask(pp::GLM.DensePredChol)
    k = size(pp.X, 2)
    rnk = pp.chol.rank
    p = pp.chol.p
    rnk == k ? mask = ones(Bool, k) : begin
    mask = zeros(Bool, k)
    mask[p[1:rnk]] .= true
end
return mask
end

setglmkernelweights!(k::CovarianceMatrices.AVarEstimator, m::AbstractMatrix) = nothing

function setglmkernelweights!(k::HAC, m::AbstractMatrix)
    n, p = size(m)
    kw = CovarianceMatrices.kernelweights(k)
    resize!(kw, p)
    idx = map(x -> allequal(x), eachcol(m))
    kw .= 1.0 .- idx
end


CovarianceMatrices.momentmatrix(m::RegressionModel) = momentmatrix(m.model)

function CovarianceMatrices.momentmatrix(m::GLM.GeneralizedLinearModel)
    X = modelmatrix(m)
    M = m.pp.scratchm1
    wrkwt = m.rr.wrkwt
    d = _dispersion(m)
    @. M = (X * wrkwt * m.rr.wrkresid)/d
    M
end

function CovarianceMatrices.momentmatrix(m::GLM.LinearModel)
    X = modelmatrix(m)
    M = m.pp.scratchm1
    wrkwt = m.rr.wts
    wrkresid = GLM.residuals(m)
    @. M = X * wrkresid
    !isempty(wrkwt) && @. M *= wrkwt
    M
end

function CovarianceMatrices.aVar(k::K, m::RegressionModel; demean = false, prewhiten = false, kwargs...) where K <: CovarianceMatrices.AVarEstimator
    mm = begin
        M = momentmatrix(m)
        u = residualadjustment(k, m)
        @. M = M * u
        setglmkernelweights!(k, modelmatrix(m))
        midx = mask(m)
        @view M[:, midx]
    end
    Σ = aVar(k, mm; demean = demean, prewhiten = prewhiten)
    Ω = if sum(midx) > 0
        O = similar(Σ, (size(midx, 1), size(midx, 1)))
        O[midx, midx] .= Σ
        O[.!midx, :] .= NaN
        O[:, .!midx] .= NaN
        O
    else
        Σ
    end
end

CovarianceMatrices.leverage(r::StatsModels.TableRegressionModel) = CovarianceMatrices.leverage(r.model)

function CovarianceMatrices.leverage(r::GLM.RegressionModel)
    X = modelmatrix(r)
    !isempty(r.rr.wts) && @. r.pp.scratchm1 .= X .* sqrt(r.rr.wts)
    _leverage(r.pp, r.pp.scratchm1)
end

function CovarianceMatrices.leverage(r::GLM.GeneralizedLinearModel)
    X = modelmatrix(r)
    !isempty(r.rr.wts) && @. r.pp.scratchm1 .= X .* sqrt(r.rr.wts)
    _leverage(r.pp, r.pp.scratchm1)
end


function _leverage(pp::GLM.DensePredChol{F, C}, X) where {F, C <: LinearAlgebra.CholeskyPivoted}
    ch = pp.chol
    rnk = rank(ch)
    p = ch.p
    idx = invperm(p)[1:rnk]
    sum(x -> x^2, view(X, :, 1:rnk) / ch.U[1:rnk, idx], dims = 2)
end

function _leverage(pp::GLM.DensePredChol{F, C}, X) where {F, C <: LinearAlgebra.Cholesky}
    sum(x -> x^2, X / pp.chol.U, dims = 2)
end

dofresiduals(r::RegressionModel) = numobs(r) - rank(modelmatrix(r))
#residuals(r::GeneralizedLinearModel) = residuals(r)

residualadjustment(k::CovarianceMatrices.AVarEstimator, r::RegressionModel) = 1.0
residualadjustment(k::HR0, r::RegressionModel) = 1.0
residualadjustment(k::HR1, r::RegressionModel) = ((√numobs(r)) / √dofresiduals(r))
residualadjustment(k::HR2, r::RegressionModel) = (1.0 ./ (1 .- CovarianceMatrices.leverage(r)) .^ 0.5)
residualadjustment(k::HR3, r::RegressionModel) = (1.0 ./ (1 .- CovarianceMatrices.leverage(r)))
function residualadjustment(k::HR4, r::RegressionModel)
    n = length(response(r))
    h = CovarianceMatrices.leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(4.0, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end
function residualadjustment(k::HR4m, r::RegressionModel)
    n = length(response(r))
    h = CovarianceMatrices.leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(1, n * h[j] / p) + min(1.5, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end
function residualadjustment(k::HR5, r::RegressionModel)
    n = length(response(r))
    h = CovarianceMatrices.leverage(r)
    p = round(Int, sum(h))
    mx = max(n * 0.7 * maximum(h) / p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n * h[j] / p, mx)
        h[j] = 1 / (1 - h[j])^(alpha / 4)
    end
    return h
end

##=================================================
## ## RegressionModel - CR
## ##=================================================
function residualadjustment(k::CR2, r::RegressionModel)
    @assert length(k.g) == 1
    g = k.g[1]
    X, u = modelmatrix(r) .* sqrt.(weights(r)), copy(GLM.residuals(r))
    XX   = bread(r)
    for groups in 1:g.ngroups
        ind = findall(x -> x .== groups, g)        
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        Hᵧᵧ = Xg * XX * Xg'
        ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check=false).L, ug)
    end    
    return u
end

function residualadjustment(k::CR3, r::RegressionModel)
    @assert length(k.g) == 1
    g = k.g[1]
    X, u = modelmatrix(r) .* sqrt.(weights(r)), copy(GLM.residuals(r))
    XX  = bread(r)
    for groups in 1:g.ngroups
        ind = findall(g .== groups)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        Hᵧᵧ = Xg * XX * Xg'
        ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check=false), ug)
    end 
    return u
end

function CovarianceMatrices.vcov(k::CovarianceMatrices.AVarEstimator, m::RegressionModel; dofcorrection = true, kwargs...)
    A = aVar(k, m; kwargs...)
    T = numobs(m)
    B = CovarianceMatrices.bread(m)
    p = size(B, 2)
    midx = mask(m)
    Bm = sum(midx) > 0 ? Bm = B[midx, midx] : B
    V = T .* Bm * A * Bm
    if sum(midx) > 0
        Vo = similar(A, (p, p))
        Vo[midx, midx] .= V
        Vo[.!midx, :] .= NaN
        Vo[:, .!midx] .= NaN
    else
        Vo = V
    end
    if haskey(kwargs, :dofcorrection)
        dofcorrect!(Vo, k, m)
    end
    return Vo
end


## Make df correction - only useful for HAC - for other estimator HR CR it depends on the type
dofcorrect!(V, k::CovarianceMatrices.AVarEstimator, m) = nothing

function dofcorrect!(V, k::HAC, m)
    dof = dofresiduals(m)
    n = numobs(m)
    k = size(V, 2)
    rmul!(V, n/dof)
end


end
