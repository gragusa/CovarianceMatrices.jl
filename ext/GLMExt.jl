module GLMExt

using CovarianceMatrices, GLM, LinearAlgebra
using Statistics
##=================================================
## Moment Matrix 
##=================================================
const FAM = Union{GLM.Gamma, GLM.Bernoulli, GLM.InverseGaussian}

weights(r::RegressionModel) = weights(r.model)
weights(m::GLM.LinearModel) = isempty(m.rr.wts) ? ones(eltype(m.rr.wts),1) : m.rr.wts
weights(m::GLM.GeneralizedLinearModel) = m.rr.wrkwt

_modelmatrix(r::RegressionModel) = _modelmatrix(r.model)
_modelmatrix(m::GLM.LinearModel) = modelmatrix(m).*sqrt.(weights(m))
_modelmatrix(m::GLM.GeneralizedLinearModel) = modelmatrix(m).*sqrt.(weights(m))

numobs(r::GLM.RegressionModel) = size(r.model.pp.X,1)
numobs(m::GLM.LinPredModel) = size(m.pp.X,1)

_dispersion(r::RegressionModel) = _dispersion(r.model)
_dispersion(m::GLM.LinearModel) = 1
_dispersion(m::GLM.GeneralizedLinearModel) = _dispersion(m.rr)

_dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2, T3} = 1
_dispersion(rr::GLM.LmResp) = 1
function _dispersion(rr::GLM.GlmResp{T1, T2, T3}) where {T1, T2<:FAM, T3}
    sum(abs2, rr.wrkwt.*rr.wrkresid)/sum(rr.wrkwt)
end

CovarianceMatrices.bread(r::RegressionModel) = CovarianceMatrices.bread(r.model)
CovarianceMatrices.bread(m::GLM.LinearModel) = GLM.invchol(m.pp)
CovarianceMatrices.bread(m::GLM.GeneralizedLinearModel) = GLM.invchol(m.pp).*_dispersion(m)

GLM.residuals(m::GLM.GeneralizedLinearModel) = m.rr.wrkresid

# CovarianceMatrices.resid(r::RegressionModel) = CovarianceMatrices.resid(r.model)

# function CovarianceMatrices.resid(m::GeneralizedLinearModel)
#   sqrt.(m.rr.wrkwt).*m.rr.wrkresid
# end

# function CovarianceMatrices.resid(m::LinearModel)
#   weights(m).*residuals(m)  
# end

# CovarianceMatrices.momentmatrix(r::RegressionModel) = CovarianceMatrices.momentmatrix(r.model)

# function CovarianceMatrices.momentmatrix(m::M) where M<:Union{LinearModel, GeneralizedLinearModel}
#   return CovarianceMatrices.residuals(m).*modelmatrix(m).*weight
#   s(m)
# end

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
  idx = map(x->allequal(x), eachcol(m))
  kw .= 1.0 .- idx  
end


function _aVar(k::K, m::RegressionModel; kwargs...) where K<:Union{HR, HAC, EWC}
  mm = begin    
    X = modelmatrix(m)
    setglmkernelweights!(k, X)    
    midx = mask(m)
    Xm = X[:, midx]
    u = adjustedresiduals(k, m)
    Xm.*u.*weights(m)
  end  
  return aVar(k, mm; kwargs...)
end   

function CovarianceMatrices.aVar(k::K, m::RegressionModel; kwargs...) where K<:Union{HR, HAC, EWC}
   midx = mask(m)
   Σ = _aVar(k, m; kwargs...)   
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

function CovarianceMatrices.leverage(r::RegressionModel) 
    X = modelmatrix(r).*sqrt.(weights(r))
    _, k = size(X)
    _leverage(r.model.pp, X)
end

function _leverage(pp::GLM.DensePredChol{F, C}, X) where {F, C<:LinearAlgebra.CholeskyPivoted}
    ch = pp.chol
    rnk = rank(ch)
    p = ch.p
    idx = invperm(p)[1:rnk]
    sum(x -> x^2, view(X, :, 1:rnk)/ch.U[1:rnk, idx], dims=2)
end

function _leverage(pp::GLM.DensePredChol{F, C}, X) where {F, C<:LinearAlgebra.Cholesky}
    sum(x -> x^2, X/pp.chol.U, dims=2)
end


# function mask(pp::GLM.DensePredChol{F, C}, X) where {F, C<:LinearAlgebra.CholeskyPivoted}
#   B = CovarianceMatrices.bread(m)
#   ## Find columns with all NaNs
#   mask_col = vec(all(isnan, B, dims=1))
#   ## Find rows with all NaNs
#   mask_row = vec(all(isnan, B, dims=2))
#   ## Subset B thos thos columns and rows without NaNs
#   B_masked = B[.!mask_row, .!mask_col]
#   return B, B_masked, mask_col, mask_row
# end

dofresiduals(r::RegressionModel) = numobs(r) - rank(modelmatrix(r))
residuals(r::GeneralizedLinearModel) = residuals(r.model)

adjustedresiduals(k::CovarianceMatrices.AVarEstimator, r::RegressionModel) = CovarianceMatrices.residuals(r.model)

adjustedresiduals(k::HR0, r::RegressionModel) = CovarianceMatrices.residuals(r)
adjustedresiduals(k::HR1, r::RegressionModel) = CovarianceMatrices.residuals(r).*((√numobs(r))/√dofresiduals(r))
adjustedresiduals(k::HR2, r::RegressionModel) = CovarianceMatrices.residuals(r).*( 1.0 ./ (1 .- CovarianceMatrices.leverage(r)).^0.5)
adjustedresiduals(k::HR3, r::RegressionModel) = CovarianceMatrices.residuals(r).*( 1.0 ./ (1 .- CovarianceMatrices.leverage(r)))


function adjustedresiduals(k::HR4, r::RegressionModel)
  n = length(response(r))
  h = CovarianceMatrices.leverage(r)
  p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(4.0, n*h[j]/p)
        h[j] = 1/(1-h[j])^(delta/2)
    end
    CovarianceMatrices.residuals(r).*h
end

function adjustedresiduals(k::HR4m, r::RegressionModel)
    n = length(response(r))
    h = CovarianceMatrices.leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(1, n*h[j]/p) + min(1.5, n*h[j]/p)
        h[j] = 1/(1-h[j])^(delta/2)
    end
    return CovarianceMatrices.residuals(r).* h
end

function adjustedresiduals(k::HR5, r::RegressionModel)
  n = length(response(r))
  h = CovarianceMatrices.leverage(r)
  p = round(Int, sum(h))
    mx = max(n*0.7*maximum(h)/p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n*h[j]/p, mx)
        h[j] = 1/(1-h[j])^(alpha/4)
    end
    return CovarianceMatrices.residuals(r).*h
end

##=================================================
## RegressionModel - CR
##=================================================
adjustment!(k::CR0, r) = 1/(length(level(k.f))-1)
adjustment!(k::CR1, r) = ((nobs(n)-1)/dof_residuls(m) * length(level(k.f))/(length(level(k.f))-1))

function adjustment(v::CR2, r::RegressionModel)    
    X, u = modelmatrix(r), residual(r)
    XX⁻¹ = crossmodelmatrix(r)
    indices = clustersindices(v)
    for j in eachindex(indices)
        Xg = view(X, index[j], :)
        ug = view(u, index[j], :)
        xAx = Symmetric(I - Xg*(XX⁻¹)*Xg')
        ldiv!(cholesky!(xAx; check=false).L, ug)
    end
    return u
end

Base.@propagate_inbounds function adjustment(k::CR3,  r::RegressionModel)
    X, u = modelmatrix(r), resid(c)
    n, p = size(X)
    invxx, indices = invcrossx(c), clustersindices(c)
    Threads.@threads for index in indices
        Xv = view(X, index, :)
        uv = view(u, index, :)
        xAx = Xv*invxx*Xv'
        ldiv!(cholesky!(Symmetric(I - xAx); check=false), uv)
    end
    return rmul!(u, 1/sqrt(dofadjustment(k, c)))
end



"""
    dofadjustment(k::CRHC, ::CRHCCache)

Calculate the default degrees-of-freedom adjsutment for `CRHC`

# Arguments
- `k::CRHC`: cluster robust variance type
- `c::CRHCCache`: the `CRHCCache` from which to extract the information
# Return
- `Float`: the degrees-of-fredom adjustment
# Note: the adjustment is a multyplicative factor.
"""
function dofadjustment(k::CR0, m::RegressionModel)
    g = length(clustersindices(k))::Int64
    return g/(g-1)
end

function dofadjustment(k::CR1, m::RegressionModel)
    g, (n, p) = length(clustersindices(k)), size(modelmatrix(m))
    return ((n-1)/(n-p) * g/(g-1))
end

dofadjustment(k::CR2, m::RegressionModel) = 1

function dofadjustment(k::CR3, m::RegressionModel)
     g, (n, p) = length(clustersindices(k)), size(modelmatrix(m))
    return (g/(g-1))
end

function CovarianceMatrices.vcov(k::CovarianceMatrices.AVarEstimator, m::RegressionModel; kwargs...)  
    A = _aVar(k, m; kwargs...)
    T = numobs(m)
    B = CovarianceMatrices.bread(m)
    k = size(B, 2)
    midx = mask(m)
    Bm = sum(midx) > 0 ? Bm = B[midx, midx] : B
    V = T.*Bm*A*Bm
    if sum(midx) > 0
      Vo = similar(A, (k, k))
      Vo[midx, midx] .= V
      Vo[.!midx, :] .= NaN
      Vo[:, .!midx] .= NaN
    else
      Vo = V
    end
    return Vo
end

end
