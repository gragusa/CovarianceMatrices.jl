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

function mask(pp::GLM.DensePredChol{F, C}) where {F, C <: LinearAlgebra.CholeskyPivoted}
    k = size(pp.X, 2)
    rnk = pp.chol.rank
    p = pp.chol.p
    rnk == k ? mask = ones(Bool, k) : begin
    mask = zeros(Bool, k)
    mask[p[1:rnk]] .= true
end
return mask
end

function mask(pp::GLM.DensePredChol{F, C}) where {F, C <: LinearAlgebra.Cholesky}
    k = size(pp.X, 2)
    return ones(Bool, k)
end

setglmkernelweights!(k::CovarianceMatrices.AVarEstimator, m) = nothing

function setglmkernelweights!(k::HAC, m::GLM.RegressionModel)
    kw = CovarianceMatrices.kernelweights(k)
    k = size(modelmatrix(m), 2)
    if isempty(kw)
        [push!(kw, 1.0) for i in 1:k]
        idx = map(i -> all(==(1), view(modelmatrix(m), :, i)), 1:k)
        kw[idx] .= 0.0
    else
        @assert length(kw) == k "The length of kernelweights should be equal to the number of columns in the model matrix."
    end
end

CovarianceMatrices.momentmatrix(m::RegressionModel) = momentmatrix(m.model)
CovarianceMatrices.momentmatrix!(M::AbstractMatrix, m::RegressionModel) = CovarianceMatrices.momentmatrix!(M, m.model)

CovarianceMatrices.momentmatrix(m::GLM.GeneralizedLinearModel) = CovarianceMatrices.momentmatrix!(m.pp.scratchm1, m)
CovarianceMatrices.momentmatrix(m::GLM.LinearModel) = CovarianceMatrices.momentmatrix!(m.pp.scratchm1, m)

function CovarianceMatrices.momentmatrix!(M::AbstractMatrix, m::GLM.GeneralizedLinearModel)
    X = modelmatrix(m)
    wrkwt = m.rr.wrkwt
    d = _dispersion(m)
    @. M = (X * wrkwt * m.rr.wrkresid)/d
    M
end

function CovarianceMatrices.momentmatrix!(M::AbstractMatrix, m::GLM.LinearModel)
    X = modelmatrix(m)
    wrkwt = m.rr.wts
    wrkresid = GLM.residuals(m)
    @. M = X * wrkresid
    !isempty(wrkwt) && @. M *= wrkwt
    return M
end
scratchm1(m::StatsModels.TableRegressionModel{T}) where T = scratchm1(m.model)
scratchm1(m::GLM.LinPredModel) = m.pp.scratchm1
scratchm1(m::GLM.GeneralizedLinearModel) = m.pp.scratchm1

function CovarianceMatrices.aVar(k::K, m::RegressionModel; demean = false, prewhiten = false, scale=true, kwargs...) where K <: CovarianceMatrices.AVarEstimator
    setglmkernelweights!(k, m)
    mm = begin
        u = residualadjustment(k, m)
        ## Important:
        ## ---------------------------------------------------------------------------
        ## This call should come afer `residualadjustment` as the `scratchm1` used to 
        ## store the momentmatrix is also used by `leverage` which is called by 
        ## `residualadjustment`.
        M = CovarianceMatrices.momentmatrix!(scratchm1(m), m)
        @. M = M * u
        M
    end
    midx = mask(m)
    Σ = aVar(k, mm[:, midx]; demean = demean, prewhiten = prewhiten, scale=scale)
    Σ
end

crmomentmatrix!(M, res, m::RegressionModel) = crmomentmatrix!(M, res, m.model)

function crmomentmatrix!(M::AbstractMatrix, res, m::GLM.GeneralizedLinearModel)
    X = modelmatrix(m)
    wrkwt = m.rr.wrkwt
    d = _dispersion(m)
    @. M = (X * wrkwt * res)/d
    M
end

function crmomentmatrix!(M::AbstractMatrix, res, m::GLM.LinearModel)
    X = modelmatrix(m)
    wrkwt = m.rr.wts
    wrkresid = res
    @. M = X * wrkresid
    !isempty(wrkwt) && @. M *= sqrt(wrkwt)
    return M
end

function CovarianceMatrices.aVar(k::K, m::RegressionModel; demean = false, prewhiten = false, scale=true, kwargs...) where K <: CovarianceMatrices.CR
    mm = begin
        u = residualadjustment(k, m)
        crmomentmatrix!(scratchm1(m), u, m)
    end
    midx = mask(m)
    Σ =  if sum(midx) == size(mm, 2)
        aVar(k, mm; demean = demean, prewhiten = prewhiten, scale=scale)
    else
        aVar(k, mm[:, midx]; demean = demean, prewhiten = prewhiten, scale=scale)
    end
    return Σ
end

CovarianceMatrices.leverage(r::StatsModels.TableRegressionModel) = CovarianceMatrices.leverage(r.model)

function CovarianceMatrices.leverage(r::GLM.RegressionModel)
    X = modelmatrix(r)
    @inbounds copy!(r.pp.scratchm1, X)
    @inbounds if !isempty(r.rr.wts)
        @. r.pp.scratchm1 *= sqrt(r.rr.wts)
    end
    _leverage(r.pp, r.pp.scratchm1)
end

function CovarianceMatrices.leverage(r::GLM.GeneralizedLinearModel)
    X = modelmatrix(r).*sqrt.(r.rr.wrkwt)
    @inbounds copy!(r.pp.scratchm1, X)
    # @inbounds if !isempty(r.rr.wts)
    #     @. r.pp.scratchm1 *= sqrt(r.rr.wts)
    # end
    _leverage(r.pp, r.pp.scratchm1)
end

function _leverage(pp::GLM.DensePredChol{F, C}, X) where {F, C <: LinearAlgebra.CholeskyPivoted}
    ch = pp.chol
    rnk = rank(ch)
    p = ch.p
    idx = invperm(p)[1:rnk]
    sum(abs2, view(X, :, 1:rnk) / view(ch.U, 1:rnk, idx), dims = 2)
end

function _leverage(pp::GLM.DensePredChol{F, C}, X) where {F, C <: LinearAlgebra.Cholesky}
    sum(abs2, X / pp.chol.U, dims = 2)
end

dofresiduals(r::RegressionModel) = numobs(r) - rank(modelmatrix(r))

@noinline residualadjustment(k::CovarianceMatrices.AVarEstimator, r::RegressionModel) = 1.0
@noinline residualadjustment(k::HR0, r::RegressionModel) = 1.0
@noinline residualadjustment(k::HR1, r::RegressionModel) = √numobs(r) / √dofresiduals(r)
@noinline residualadjustment(k::HR2, r::RegressionModel) = 1.0 ./ (1 .- CovarianceMatrices.leverage(r)).^ 0.5
@noinline residualadjustment(k::HR3, r::RegressionModel) = 1.0 ./ (1 .- CovarianceMatrices.leverage(r))

@noinline function residualadjustment(k::HR4, r::RegressionModel)
    n = length(response(r))
    h = CovarianceMatrices.leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(4.0, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function residualadjustment(k::HR4m, r::RegressionModel)
    n = length(response(r))
    h = CovarianceMatrices.leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(1, n * h[j] / p) + min(1.5, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function residualadjustment(k::HR5, r::RegressionModel)
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
function residualadjustment(k::Union{CR0, CR1}, r::RegressionModel) 
    wts = r.model.rr.wts
    if isempty(wts)
        GLM.residuals(r)
    else
        GLM.residuals(r).*sqrt.(wts)
    end
end

function residualadjustment(k::CR2, r::RegressionModel)
    wts = r.model.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u = copy(GLM.residuals(r))
    !isempty(wts) && @. u *= sqrt(wts)
    XX = bread(r)
    for groups in 1:g.ngroups
        ind = findall(x -> x .== groups, g)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check=false).L, ug)
        else
            Hᵧᵧ = (Xg * XX * Xg').*view(wts, ind)'
            ug .= matrixpowbysvd(I - Hᵧᵧ, -0.5)*ug
        end
    end
    return u
end

function matrixpowbysvd(A, p; tol = eps()^(1/1.5))
    s = svd(A)
    V = s.S
    V[V .< tol] .= 0
    return s.V*diagm(0=>V.^p)*s.Vt
end

function residualadjustment(k::CR3, r::RegressionModel)
    wts = r.model.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u = copy(GLM.residuals(r))    
    !isempty(wts) && @. u *= sqrt(wts)
    XX  = bread(r)
    for groups in 1:g.ngroups
        ind = findall(g .== groups)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check=false), ug)
        else
            Hᵧᵧ = (Xg * XX * Xg').*view(wts, ind)'
            ug .= (I - Hᵧᵧ)^(-1)*ug
        end
    end 
    return u
end

function CovarianceMatrices.vcov(k::CovarianceMatrices.AVarEstimator, m::RegressionModel; dofadjust=true, kwargs...)
    ## dofadjust = true only does something for HAC (EWC?) (VARHAC?) (Driskol?), for other estimators it depends on the type
    A = aVar(k, m; kwargs...)
    T = numobs(m)
    B = CovarianceMatrices.bread(m)
    p = size(B, 2)
    midx = mask(m)
    Bm = sum(midx) < p ? Bm = B[midx, midx] : B
    V = T .* Bm * A * Bm
    if sum(midx) > 0
        Vo = similar(A, (p, p))
        Vo[midx, midx] .= V
        Vo[.!midx, :] .= NaN
        Vo[:, .!midx] .= NaN
    else
        Vo = V
    end
    dofadjust && dofcorrect!(Vo, k, m)
    return Vo
end


## Make df correction - only useful for HAC - for other estimator HR CR it depends on the type
dofcorrect!(V, k::CovarianceMatrices.AVarEstimator, m) = nothing


## Add method for Other if needed
function dofcorrect!(V, k::HAC, m)
    dof = dofresiduals(m)
    n = numobs(m)
    rmul!(V, n/dof)
end


end
