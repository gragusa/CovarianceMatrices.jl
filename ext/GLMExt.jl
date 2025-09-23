module GLMExt

using CovarianceMatrices, GLM, LinearAlgebra, StatsBase
using Statistics

##=================================================
## Moment Matrix 
##=================================================
const FAM = Union{GLM.Gamma,GLM.Bernoulli,GLM.InverseGaussian}
const CM = CovarianceMatrices

numobs(r::GLM.RegressionModel) = size(r.model.pp.X, 1)
numobs(m::GLM.LinPredModel) = size(m.pp.X, 1)

_dispersion(r::GLM.RegressionModel) = _dispersion(r.model)
_dispersion(m::GLM.LinearModel) = 1
_dispersion(m::GLM.GeneralizedLinearModel) = _dispersion(m.rr)

_dispersion(rr::GLM.GlmResp{T1,T2,T3}) where {T1,T2,T3} = 1
_dispersion(rr::GLM.LmResp) = 1
function _dispersion(rr::GLM.GlmResp{T1,T2,T3}) where {T1,T2<:FAM,T3}
    sum(abs2, rr.wrkwt .* rr.wrkresid) / sum(rr.wrkwt)
end

CM.bread(r::RegressionModel) = CM.bread(r.model)
CM.bread(m::GLM.LinearModel) = GLM.invchol(m.pp)
CM.bread(m::GLM.GeneralizedLinearModel) = GLM.invchol(m.pp) .* _dispersion(m)

GLM.residuals(m::GLM.GeneralizedLinearModel) = m.rr.wrkresid

mask(r::RegressionModel) = mask(r.model)
mask(m::GLM.LinearModel) = mask(m.pp)
mask(m::GLM.GeneralizedLinearModel) = mask(m.pp)

function mask(pp::GLM.DensePredChol{F,C}) where {F,C<:LinearAlgebra.CholeskyPivoted}
    k = size(pp.X, 2)
    rnk = pp.chol.rank
    p = pp.chol.p
    rnk == k ? mask = ones(Bool, k) : begin
        mask = zeros(Bool, k)
        mask[p[1:rnk]] .= true
    end
    return mask
end

function mask(pp::GLM.DensePredChol{F,C}) where {F,C<:LinearAlgebra.Cholesky}
    k = size(pp.X, 2)
    return ones(Bool, k)
end

CM.momentmatrix(m::RegressionModel) = momentmatrix(m.model)
CM.momentmatrix!(M::AbstractMatrix, m::RegressionModel) = CM.momentmatrix!(M, m.model)

CM.momentmatrix(m::GLM.GeneralizedLinearModel) = CM.momentmatrix!(m.pp.scratchm1, m)
CM.momentmatrix(m::GLM.LinearModel) = CM.momentmatrix!(m.pp.scratchm1, m)

function CM.momentmatrix!(M::AbstractMatrix, m::GLM.GeneralizedLinearModel)
    X = modelmatrix(m)
    wrkwt = m.rr.wrkwt
    d = _dispersion(m)
    @. M = (X * wrkwt * m.rr.wrkresid)/d
    M
end

function CM.momentmatrix!(M::AbstractMatrix, m::GLM.LinearModel)
    X = modelmatrix(m)
    wrkwt = m.rr.wts
    wrkresid = GLM.residuals(m)
    @. M = X * wrkresid
    !isempty(wrkwt) && @. M *= wrkwt
    return M
end
scratchm1(m::StatsModels.TableRegressionModel{T}) where {T} = scratchm1(m.model)
scratchm1(m::GLM.LinPredModel) = m.pp.scratchm1
scratchm1(m::GLM.GeneralizedLinearModel) = m.pp.scratchm1

function CM.aVar(
    k::K,
    m::RegressionModel;
    demean = false,
    prewhite = false,
    scale = true,
    kwargs...,
) where {K<:CM.AVarEstimator}
    CM.setkernelweights!(k, m)
    mm = begin
        u = CM.residualadjustment(k, m)
        ## Important:
        ## ---------------------------------------------------------------------------
        ## This call should come afer `residualadjustment` as the `scratchm1` used to 
        ## store the momentmatrix is also used by `leverage` which is called by 
        ## `residualadjustment`.
        M = CM.momentmatrix!(scratchm1(m), m)
        @. M = M * u
        M
    end
    midx = mask(m)
    Σ = if sum(midx) == size(mm, 2)
        aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)
    else
        aVar(k, mm[:, midx]; demean = demean, prewhite = prewhite, scale = scale)
    end
    return Σ
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

function CM.aVar(
    k::K,
    m::RegressionModel;
    demean = false,
    prewhite = false,
    scale = true,
    kwargs...,
) where {K<:CM.CR}
    mm = begin
        u = CM.residualadjustment(k, m)
        crmomentmatrix!(scratchm1(m), u, m)
    end
    midx = mask(m)
    Σ = if sum(midx) == size(mm, 2)
        aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)
    else
        aVar(k, mm[:, midx]; demean = demean, prewhite = prewhite, scale = scale)
    end
    return Σ
end

CM.leverage(r::StatsModels.TableRegressionModel) = CM.leverage(r.model)

function CM.leverage(r::GLM.RegressionModel)
    X = modelmatrix(r)
    @inbounds copy!(r.pp.scratchm1, X)
    @inbounds if !isempty(r.rr.wts)
        @. r.pp.scratchm1 *= sqrt(r.rr.wts)
    end
    _leverage(r.pp, r.pp.scratchm1)
end

function CM.leverage(r::GLM.GeneralizedLinearModel)
    X = modelmatrix(r) .* sqrt.(r.rr.wrkwt)
    @inbounds copy!(r.pp.scratchm1, X)
    # @inbounds if !isempty(r.rr.wts)
    #     @. r.pp.scratchm1 *= sqrt(r.rr.wts)
    # end
    _leverage(r.pp, r.pp.scratchm1)
end

function _leverage(pp::GLM.DensePredChol{F,C}, X) where {F,C<:LinearAlgebra.CholeskyPivoted}
    ch = pp.chol
    rnk = rank(ch)
    p = ch.p
    idx = invperm(p)[1:rnk]
    sum(abs2, view(X, :, 1:rnk) / view(ch.U, 1:rnk, idx), dims = 2)
end

function _leverage(pp::GLM.DensePredChol{F,C}, X) where {F,C<:LinearAlgebra.Cholesky}
    sum(abs2, X / pp.chol.U, dims = 2)
end

dofresiduals(r::RegressionModel) = numobs(r) - rank(modelmatrix(r))

@noinline CM.residualadjustment(k::HAC, r::Any) = 1.0

@noinline CM.residualadjustment(k::HR0, r::GLM.RegressionModel) = 1.0
@noinline CM.residualadjustment(k::HR1, r::GLM.RegressionModel) =
    √numobs(r) / √dofresiduals(r)
@noinline CM.residualadjustment(k::HR2, r::GLM.RegressionModel) =
    1.0 ./ (1 .- CM.leverage(r)) .^ 0.5
@noinline CM.residualadjustment(k::HR3, r::GLM.RegressionModel) =
    1.0 ./ (1 .- CM.leverage(r))

@noinline function CM.residualadjustment(k::HR4, r::RegressionModel)
    n = length(response(r))
    h = CM.leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(4.0, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function CM.residualadjustment(k::HR4m, r::RegressionModel)
    n = length(response(r))
    h = CM.leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(1, n * h[j] / p) + min(1.5, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function CM.residualadjustment(k::HR5, r::RegressionModel)
    n = length(response(r))
    h = CM.leverage(r)
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
function CM.residualadjustment(k::Union{CR0,CR1}, r::StatsModels.TableRegressionModel)
    wts = r.model.rr.wts
    if isempty(wts)
        GLM.residuals(r)
    else
        GLM.residuals(r) .* sqrt.(wts)
    end
end

function CM.residualadjustment(k::CR2, r::StatsModels.TableRegressionModel)
    wts = r.model.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u = copy(GLM.residuals(r))
    !isempty(wts) && @. u *= sqrt(wts)
    XX = bread(r)
    for groups = 1:g.ngroups
        ind = findall(x -> x .== groups, g)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check = false).L, ug)
        else
            Hᵧᵧ = (Xg * XX * Xg') .* view(wts, ind)'
            ug .= matrixpowbysvd(I - Hᵧᵧ, -0.5)*ug
        end
    end
    return u
end

function matrixpowbysvd(A, p; tol = eps()^(1/1.5))
    s = svd(A)
    V = s.S
    V[V .< tol] .= 0
    return s.V*diagm(0=>V .^ p)*s.Vt
end

function CM.residualadjustment(k::CR3, r::StatsModels.TableRegressionModel)
    wts = r.model.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u = copy(GLM.residuals(r))
    !isempty(wts) && @. u *= sqrt(wts)
    XX = bread(r)
    for groups = 1:g.ngroups
        ind = findall(g .== groups)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check = false), ug)
        else
            Hᵧᵧ = (Xg * XX * Xg') .* view(wts, ind)'
            ug .= (I - Hᵧᵧ)^(-1)*ug
        end
    end
    return u
end

function CM.vcov(k::CM.AVarEstimator, m::RegressionModel; dofadjust = true, kwargs...)
    ## dofadjust = true only does something for HAC (EWC?) (VARHAC?) (Driskol?), for other estimators it depends on the type
    A = aVar(k, m; kwargs...)
    T = numobs(m)
    B = CM.bread(m)
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

function CM.stderror(k::CM.AVarEstimator, m::RegressionModel; kwargs...)
    sqrt.(diag(CM.vcov(k, m; kwargs...)))
end

## Make df correction - only useful for HAC - for other estimator HR CR it depends on the type
dofcorrect!(V, k::CM.AVarEstimator, m) = nothing

## Add method for Other if needed
function dofcorrect!(V, k::HAC, m)
    dof = dofresiduals(m)
    n = numobs(m)
    rmul!(V, n/dof)
end

function CM.setkernelweights!(
    k::HAC{T},
    X::RegressionModel,
) where {T<:Union{CM.NeweyWest,CM.Andrews}}
    CM.setkernelweights!(k, modelmatrix(X))
    k.wlock .= true
end

end
