module GLMExt

using CovarianceMatrices, GLM, LinearAlgebra


## df = DataFrame(y=randn(100), x = randn(100), w=rand(100))

## lm1 = lm(@formula(y~x), df)
## lm2 = lm(@formula(y~x), df; wts=df.w)
## ols = glm(@formula(y~x), df, Normal(), IdentityLink())






##=================================================
## Moment Matrix 
##=================================================

CovarianceMatrices.bread(r::RegressionModel) = CovarianceMatrices.bread(r.model)
CovarianceMatrices.bread(m::LinearModel) = CovarianceMatrices.bread(r.model)



CovarianceMatrices.resid(r::RegressionModel) = CovarianceMatrices.resid(r.model)

function CovarianceMatrices.resid(m::GeneralizedLinearModel)
  sqrt.(m.rr.wrkwt).*m.rr.wrkresid
end

function CovarianceMatrices.resid(m::LinearModel)
  u = residuals(m)
  isempty(m.rr.wts) ? u : sqrt.(m.rr.wts).u
end

CovarianceMatrices.momentmatrix(r::RegressionModel) = CovarianceMatrices.momentmatrix(r.model)

function CovarianceMatrices.momentmatrix(m::M) where M<:Union{LinearModel, GeneralizedLinearModel}
  return CovarianceMatrices.resid(m).*modelmatrix(m) 
end

function CovarianceMatrices.aVar(k::K, m::RegressionModel; kwargs...) where K<:Union{HR, HAC, EWC}    
    mm = if K isa HR0 || K isa HR1
        momentmatrix(m)
    else
        X = modelmatrix(m)
        u = adjustedresiduals(k, m)
        X.*u
    end
    Σ = aVar(k, mm; kwargs...)    
end


CovarianceMatrices.leverage(r::RegressionModel) = leverage(r.model.pp.chol)

function CovarianceMatrices.leverage(pp::LinearAlgebra.CholeskyPivoted)
    X = modelmatrix(pp)
    _, k = size(X)
    ch = pp.chol
    rnk = rank(ch)
    p = ch.p
    idx = invperm(p)[1:rnk]
    sum(x -> x^2, view(X, :, 1:rnk)/ch.U[1:rnk, idx], dims=2)
end

function CovarianceMatrices.leverage(pp::LinearAlgebra.Cholesky)
    X = modelmatrix(pp)
    sum(x -> x^2, X/pp.chol.U, dims=2)
end

function CovarianceMatrices.leverage(pp::QRPivoted)
  X = modelmatrix(pp)
  _, k = size(X)
  ch = pp.qr
  rnk = length(ch.p)
  p = ch.p
  idx = invperm(p)[1:rnk]
  sum(x -> x^2, view(X, :, 1:rnk)/ch.R[1:rnk, idx], dims=2)
end

function CovarianceMatrices.leverage(pp::GLM.DensePredQR{C}) where {C<:GLM.QRPivoted}
  X = modelmatrix(pp)
  sum(x -> x^2, X/pp.chol.R, dims=2)
end


adjustedresiduals(k::CovarianceMatrices.AVarEstimator, r::RegressionModel) = CovarianceMatrices.resid(r.model)

adjustedresiduals(k::HR0, r::RegressionModel) = CovarianceMatrices.resid(r).*(1/√length(residuals(r)))
adjustedresiduals(k::HR1, r::RegressionModel) = CovarianceMatrices.resid(r).*(√length(residuals(r))/dof_residual(r))
adjustedresiduals(k::HR2, r::RegressionModel) = CovarianceMatrices.resid(r).*(length(residuals(r)) ./ (1 .- hatvalues(r)))
adjustedresiduals(k::HR3, r::RegressionModel) = CovarianceMatrices.resid(r).*(length(residuals(r)) ./ (1 .- hatvalues(r)).^2)

function adjustedresiduals(k::HR4, m::RegressionModel)
    n, p = nobs(m), sum(.!(coef(m).==0))
    h = GLM.hatvalues(m)
    @inbounds for j in eachindex(h)
        delta = min(4.0, n*h[j]/p)
        h[j] = 1/(1-h[j])^delta
    end
    return residuals(m).*h  ## This can be done above
end

function adjustedresiduals(k::HR4m, m::RegressionModel)
    n, p = length(response(y)), sum(.!(coef(m).==0))
    h = GLM.hatvalues(m)
    @inbounds for j in eachindex(h)
        delta = min(1, n*h[j]/p) + min(1.5, n*h[j]/p)
        h[j] = 1/(1-h[j])^delta
    end
    return residuals(m).*h
end

function adjustedresiduals(k::HR5, m::RegressionModel)
    n, p = length(response(y)), sum(.!(coef(m).==0))
    h = GLM.hatvalues(m)
    mx = max(n*0.7*maximum(h)/p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n*h[j]/p, mx)
        h[j] = 1/sqrt((1-h[j])^alpha)
    end
    return residuals(m).*h
end


##=================================================
## RegressionModel - CR
##=================================================
adjustment!(k::CR0, m) = 1/(length(level(k.f))-1)
adjustment!(k::CR1, m) = ((nobs(n)-1)/dof_residuls(m) * length(level(k.f))/(length(level(k.f))-1))

function adjustment(v::CR2, m::RegressionModel)    
    X, u = modelmatrix(m), residual(m)
    XX⁻¹ = crossmodelmatrix(m)
    indices = clustersindices(v)
    for j in eachindex(indices)
        Xg = view(X, index[j], :)
        ug = view(u, index[j], :)
        xAx = Symmetric(I - Xg*(XX⁻¹)*Xg')
        ldiv!(cholesky!(xAx; check=false).L, ug)
    end
    return u
end

Base.@propagate_inbounds function adjustment(k::CR3,  m::RegressionModel)
    
    X, u = modelmatrix(m), resid(c)
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



end
