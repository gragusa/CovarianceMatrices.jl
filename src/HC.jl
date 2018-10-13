struct HCConfig{F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector}
    q::F1
    X::F1
    x::F2
    v::V
    η::V
    u::V
end

function HCConfig(X::AbstractMatrix{T1}; returntype::Type{T1} = eltype(X)) where T1
    n, p = size(X)
    HCConfig(similar(X), similar(X), Array{T1, 2}(undef, p, p),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n), Array{T1, 1}(undef, size(X,1)))
end

function HCConfig(m::StatsModels.DataFrameRegressionModel; returntype::Type{T1} = Float64) where T1
    s = size(ModelMatrix(m.mf).m)
    HCConfig(similar(Array{T1, 2}(undef, s...)); returntype = returntype)
end

function HCConfig(m::GLM.LinearModel; returntype::Type{T1} = Float64) where T1
    s = size(m.pp.X)
    HCConfig(similar(Array{T1, 2}(undef, s...)); returntype = T1)
end

## -----
## Residual methods
## -----

_residuals(cfg, m::StatsModels.DataFrameRegressionModel, w) = _residuals(cfg, m.model, w)
_residuals(cfg, m::StatsModels.DataFrameRegressionModel) = _residuals(cfg, m.model, Val{true})

function _residuals(cfg, m::T, weighted::Type{Val{false}}) where T<:GLM.GeneralizedLinearModel
    copyto!(cfg.u, m.model.rr.wrkresid)
    return cfg.u
end

function _residuals(cfg, m::T, weighted::Type{Val{false}}) where T<:GLM.LinearModel
    copyto!(cfg.u, residuals(m.model.rr))
    return cfg.u
end

function _residuals(cfg, m::T, weighted::Type{Val{true}}) where T<:GLM.GeneralizedLinearModel
    copyto!(cfg.u, m.model.rr.wrkresid)
    isempty(m.model.rr.wts) || broadcast!(*, cfg.u, cfg.u, m.model.rr.wts)
    return cfg.u
end

function _residuals(cfg, m::T, weighted::Type{Val{true}}) where T<:GLM.LinearModel
    copyto!(cfg.u, residuals(m.model.rr))
    isempty(m.model.rr.wts) || broadcast!(*, cfg.u, cfg.u, m.model.rr.wts)
    return cfg.u
end

## -----
## Regressors methods
## -----
function _modelmatrix(cfg, m::StatsModels.DataFrameRegressionModel{T}, weighted::Type{Val{false}}) where T<:GLM.GeneralizedLinearModel
    copyto!(cfg.X, m.mm.m)  ## Check this ModelMatrix(m.mf).m
    return cfg.X
end

function _modelmatrix(cfg, m::StatsModels.DataFrameRegressionModel{T}, weighted::Type{Val{false}}) where T<:GLM.LinearModel
    copyto!(cfg.X, m.model.pp.X)
    return cfg.X
end

function _modelmatrix(cfg, m::GLM.LinearModel, weighted::Type{Val{false}})
    copyto!(cfg.X, m.pp.X)
    return cfg.X
end

function _modelmatrix(cfg, m::StatsModels.DataFrameRegressionModel{T}, weighted::Type{Val{true}}) where T<:GLM.GeneralizedLinearModel
    copyto!(cfg.X, m.mm.m)
    isempty(m.model.rr.wts) || (cfg.u .= sqrt.(m.model.rr.wts); broadcast!(*, cfg.X, cfg.X, cfg.u))
    return cfg.X
end

function _modelmatrix(cfg, m::StatsModels.DataFrameRegressionModel{T}, weighted::Type{Val{true}}) where T<:GLM.LinearModel
    copyto!(cfg.X, m.pp.X)
    isempty(m.rr.wts) || (cfg.u .= sqrt.(m.rr.wts); broadcast!(*, cfg.X, cfg.X, cfg.u))
    return cfg.X
end

function _modelmatrix(cfg, m::GLM.LinearModel, weighted::Type{Val{true}})
    copyto!(cfg.X, m.pp.X)
    return cfg.X
end

choleskyfactor(m::StatsModels.DataFrameRegressionModel) = m.model.pp.chol.UL
choleskyfactor(m::GLM.LinearModel) = m.pp.chol.UL

## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----

function esteq!(cfg, m::StatsModels.RegressionModel)
    _modelmatrix(cfg, m, Val{false})
    _residuals(cfg, m, Val{true})
    broadcast!(*, cfg.q, cfg.X, cfg.u)
end

pseudohessian(cfg, m::StatsModels.DataFrameRegressionModel) = GLM.invchol(m.model.pp)

function pseudohessian(cfg, m::GLM.LinearModel)
    X = m.pp.X
    mul!(cfg.x, X', X)
    inv(cholesky!(cfg.x))
end



function StatsBase.vcov(m::StatsModels.RegressionModel, k::HC, cfg::HCConfig)
    mf = esteq!(cfg, m)
    br = pseudohessian(cfg, m)
    adjfactor!(cfg, m, k)
    mul!(cfg.x, mf', broadcast!(*, cfg.X, mf, cfg.η))
    br*cfg.x*br'
end

## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----

function hatmatrix(cfg, m::StatsModels.DataFrameRegressionModel)
      z = _modelmatrix(cfg, m, Val{true})
      cf = choleskyfactor(m)::UpperTriangular
      rdiv!(z, cf)
      sum!(cfg.v, z.^2)
 end

  adjfactor!(cfg, m::StatsModels.RegressionModel, k::HC0) = cfg.η .= one(eltype(cfg.u))
  adjfactor!(cfg, m::StatsModels.RegressionModel, k::HC1) = cfg.η .= nobs(m)./dof_residual(m)
  adjfactor!(cfg, m::StatsModels.RegressionModel, k::HC2) = cfg.η .= one(eltype(cfg.u))./(one(eltype(cfg.u)).-hatmatrix(cfg, m))
  adjfactor!(cfg, m::StatsModels.RegressionModel, k::HC3) = cfg.η .= one(eltype(cfg.u))./(one(eltype(cfg.u)).-hatmatrix(cfg, m)).^2

  function adjfactor!(cfg, m::StatsModels.RegressionModel, k::HC4)
      n, p = size(cfg.X)
      tone = one(eltype(cfg.u))
      h = hatmatrix(cfg, m)
      @inbounds for j in eachindex(h)
          delta = min(4, n*h[j]/p)
          cfg.η[j] = tone/(tone-h[j])^delta
      end
      cfg.η
  end

  function adjfactor!(cfg, m::StatsModels.RegressionModel, k::HC4m)
      n, p = size(cfg.X)
      tone = one(eltype(cfg.u))
      h = hatmatrix(cfg, m)
      @inbounds for j in eachindex(h)
          delta = min(tone, n*h[j]/p) + min(1.5, n*h[j]/p)
          cfg.η[j] = tone/(tone-h[j])^delta
      end
      cfg.η
  end

  function adjfactor!(cfg, m::StatsModels.RegressionModel, k::HC5)
      n, p = size(cfg.X)
      tone = one(eltype(cfg.u))
      h     = hatmatrix(cfg, m)
      mx    = max(n*0.7*maximum(h)/p, 4)
      @inbounds for j in eachindex(h)
          alpha =  min(n*h[j]/p, mx)
          cfg.η[j] = tone/(tone-h[j])^alpha
      end
      cfg.η
  end

# ## -----
# ## DataFramesRegressionModel/AbstractGLM methods
# ## -----
#
# CovarianceModels = Union{RegressionModel, AbstractGLM}
#
# npars(r::CovarianceModels) = length(coef(r))
# residualdof(r::CovarianceModels) = numobs(r) - npars(r)
#
# function modelresiduals(r::CovarianceModels)
#     u = rawresiduals(r)
#     wts = modelweights(r)
#     u.*sqrt.(wts)
# end
#
# #region Adjustment factors
# adjfactor!(u, r::CovarianceModels, k::HC0) = u[:] .= one(Float64)
# adjfactor!(u, r::CovarianceModels, k::HC1) = u[:] .= numobs(r)./residualdof(r)
# adjfactor!(u, r::CovarianceModels, k::HC2) = u[:] .= 1.0./(1.0.-hatmatrix(r))
# adjfactor!(u, r::CovarianceModels, k::HC3) = u[:] .= 1.0./(1.0.-hatmatrix(r)).^2
#
# function adjfactor!(u, r::CovarianceModels, k::HC4)
#     h = hatmatrix(r)
#     n = numobs(r)
#     p = npars(r)
#     @inbounds for j in eachindex(h)
#         delta = min(4, n*h[j]/p)
#         u[j] = 1.0/(1.0-h[j])^delta
#     end
# end
#
# function adjfactor!(u, r::CovarianceModels, k::HC4m)
#     h = hatmatrix(r)
#     n = numobs(r)
#     p = npars(r)
#     @inbounds for j in eachindex(h)
#         delta = min(1.0, n*h[j]/p) + min(1.5, n*h[j]/p)
#         u[j] = 1.0/(1.0-h[j])^delta
#     end
# end
#
# function adjfactor!(u, r::CovarianceModels, k::HC5)
#     h     = hatmatrix(r)
#     n     = numobs(r)
#     p     = npars(r)
#     mx    = max(n*0.7*maximum(h)/p, 4)
#     @inbounds for j in eachindex(h)
#         alpha =  min(n*h[j]/p, mx)
#         u[j] = 1.0/(1.0-h[j])^alpha
#     end
# end
# #endregion
#
# #region Model Matrix and hatmatrix
# function weightedmodelmatrix(r::CovarianceModels)
#     w = modelweights(r)
#     if isempty(w)
#         copy(modelmatrix(r))
#     else
#         modelmatrix(r).*sqrt.(w)
#     end
# end
#
# function fullyweightedmodelmatrix(r::CovarianceModels)
#     w = modelweights(r)
#     if isempty(w)
#         copy(modelmatrix(r))
#     else
#         modelmatrix(r).*w
#     end
# end
#
# function hatmatrix(r::CovarianceModels)
#     z = weightedmodelmatrix(r)
#     cf = choleskyfactor(r)
#     rdiv!(z, cf)
#     diag(z*transpose(z))
# end
# #endregion
#
# #region bread and meat
# function meat(r::CovarianceModels, k::HC)
#     u = copy(rawresiduals(r))
#     X = fullyweightedmodelmatrix(r)
#     z = X.*u
#     adjfactor!(u, r, k)
#     rmul!(transpose(z)*(z.*u), 1/nobs(r))
# end
#
# function bread(r::CovarianceModels)
#     A = invXX(r)
#     rmul!(A, nobs(r))
# end
#
# function meat(r::CovarianceModels, k::CRHC)
#     idx   = sortperm(k.cl)
#     cls   = k.cl[idx]
#     ichol = invXX(r)
#     X     = fullyweightedmodelmatrix(r)[idx,:]
#     e     = rawresiduals(r)[idx]
#     # w     = modelweights(r)
#     bstarts = [searchsorted(cls, j[2]) for j in enumerate(unique(cls))]
#     adjresid!(k, X, e, ichol, bstarts)
#     M = zeros(size(X, 2), size(X, 2))
#     clusterize!(M, X.*e, bstarts)
#     return rmul!(M, 1/nobs(r))
# end
# #endregion
#
#
# function sandwhich(r::T, k::R) where {T<:CovarianceModels, R<:RobustVariance}
#     B = meat(r, k)
#     A = bread(r)
#     rmul!(A*B*A, 1/nobs(r))
# end
#
# function vcov(X::AbstractMatrix, v::HC)
#     N, p = size(X)
#     XX = transpose(X)*X
#     return rmul!(XX, 1/N)
# end
#
# ## -----
# ## DataFrame methods
# ## -----
# numobs(r::DataFrameRegressionModel) = size(r.model.pp.X, 1)
# modelmatrix(r::DataFrameRegressionModel) = r.mm.m
# rawresiduals(r::DataFrameRegressionModel) = r.model.rr.wrkresid
# modelweights(r::DataFrameRegressionModel) = r.model.rr.wrkwt
#
# function modelweights(r::DataFrameRegressionModel{T}) where T<:LinearModel
#     wts = r.model.rr.wts
#     isempty(wts) ? one(eltype(wts)) : wts
# end
#
# modelweights(r::LinearModel) = r.rr.wts
# modelweights(r::LmResp) = r.model.rr.wts
#
# choleskyfactor(r::DataFrameRegressionModel) = r.model.pp.chol.UL
# function XX(r::DataFrameRegressionModel)
#     cf = choleskyfactor(r)
#     cf'cf
# end
# invXX(r::DataFrameRegressionModel) = GLM.invchol(r.model.pp)
# modelresponse(r::DataFrameRegressionModel) = r.model.rr.y
#
# function rawresiduals(r::DataFrameRegressionModel{T}) where T<:LinearModel
#     y = r.model.rr.y
#     mu = r.model.rr.mu
#     if isempty(r.model.rr.wts)
#         y - mu
#     else
#         wts = r.model.rr.wts
#         resid = similar(y)
#         @simd for i = eachindex(resid,y,mu,wts)
#             @inbounds resid[i] = (y[i] - mu[i]) * sqrt(wts[i])
#         end
#         resid
#     end
# end
#
# ## -----
# ## GeneralizedLinearModel methods
# ## -----
# FlatModels = Union{GeneralizedLinearModel, LinearModel}
#
# numobs(r::FlatModels) = size(r.pp.X, 1)
# modelmatrix(r::FlatModels) = r.pp.X
# rawresiduals(r::GeneralizedLinearModel) = r.rr.wrkresid
# rawresiduals(r::LinearModel) = modelresiduals(r)
# modelweights(r::FlatModels) = r.rr.wrkwt
# modelresponse(r::FlatModels) = r.rr.y
# function modelresiduals(r::LinearModel)
#     y = r.rr.y
#     mu = r.rr.mu
#     if isempty(modelweights(r))
#         y - mu
#     else
#         wts = r.rr.wts
#         resid = similar(y)
#         @simd for i = eachindex(resid,y,mu,wts)
#             @inbounds resid[i] = (y[i] - mu[i]) * sqrt(wts[i])
#         end
#         resid
#     end
# end
#
# choleskyfactor(r::FlatModels) = r.pp.chol.UL
# XX(r::FlatModels) = choleskyfactor(r)'*choleskyfactor(r)
# invXX(r::FlatModels) = GLM.invchol(r.pp)
#
#
#
# ## -----
# ## Clusters methods
# ## -----
#
# nclus(k::CRHC) = length(unique(k.cl))
#
# #region Residual adjustments
# adjresid!(v::CRHC0, X, e, ichol, bstarts) = identity(e)
# adjresid!(v::CRHC1, X, e, ichol, bstarts) = e[:] = scalaradjustment(X, bstarts)*e
# adjresid!(v::CRHC2, X, e, ichol, bstarts) = getqii(v, e, X, ichol, bstarts)
# adjresid!(v::CRHC3, X, e, ichol, bstarts) = scalaradjustment(X, bstarts).*getqii(v, e, X, ichol, bstarts)
#
# function getqii(v::CRHC2, e, X, A, bstarts)
#     @inbounds for j in 1:length(bstarts)
#         rnge = bstarts[j]
#         se = view(e, rnge)
#         sx = view(X, rnge,:)
#         BB = Symmetric(I - sx*A*sx')
#         e[rnge] =  cholesky(BB, Val(false))\se
#     end
#     return e
# end
#
# function getqii(v::CRHC3, e, X, A, bstarts)
#     @inbounds for j in 1:length(bstarts)
#         rnge = bstarts[j]
#         se = view(e, rnge)
#         sx = view(X, rnge, :)
#         e[rnge] =  (I - sx*A*sx')\se
#     end
#     return e
# end
#
# function scalaradjustment(X, bstarts)
#     n, k = size(X)
#     g    = length(bstarts)
#     sqrt.((n-1)/(n-k) * g/(g-1))
# end
# #endregion
#
# #region Utility
# function clusterize!(M::Matrix, U::Matrix, bstarts)
#     k, k = size(M)
#     s = Array{Float64}(undef, k)
#     for m = 1:length(bstarts)
#         for i = 1:k
#             @inbounds s[i] = zero(Float64)
#         end
#         for j = 1:k, i = bstarts[m]
#             @inbounds s[j] += U[i, j]
#         end
#         for j = 1:k, i = 1:k
#             @inbounds M[i, j] += s[i]*s[j]
#         end
#     end
# end
# #endregion
#
#
#
#
# # vcov(r::T, k::HC) where {T<:CovarianceModels} = sandwhich(r, k)
# # vcov(r::T, k::Type{RobustVariance}) where {T<:CovarianceModels} = sandwhich(r, k())
#
# # vcov(r::T, k::CRHC) where {T<:CovarianceModels} = sandwhich(r, k)
#
# vcov(r::T, k::CRHC) where {T<:DataFrameRegressionModel} = sandwhich(r, k)
# vcov(r::T, k::CRHC) where {T<:AbstractGLM} = sandwhich(r, k)
# vcov(r::T, k::CRHC) where {T<:LinearModel} = sandwhich(r, k)
#
# vcov(r::T, k::HC) where {T<:DataFrameRegressionModel} = sandwhich(r, k)
# vcov(r::T, k::HC) where {T<:AbstractGLM} = sandwhich(r, k)
# vcov(r::T, k::HC) where {T<:LinearModel} = sandwhich(r, k)
#
# vcov(r::T, k::Type{R}) where {T<:DataFrameRegressionModel, R<:HC} = sandwhich(r, k())
# vcov(r::T, k::Type{R}) where {T<:AbstractGLM, R<:HC} = sandwhich(r, k())
# vcov(r::T, k::Type{R}) where {T<:LinearModel, R<:HC} = sandwhich(r, k())
#
# stderror(r::T, k::CRHC) where {T<:DataFrameRegressionModel} = sqrt.(diag(sandwhich(r, k)))
# stderror(r::T, k::CRHC) where {T<:AbstractGLM} = sqrt.(diag(sandwhich(r, k)))
# stderror(r::T, k::CRHC) where {T<:LinearModel} = sqrt.(diag(sandwhich(r, k)))
#
# stderror(r::T, k::HC) where {T<:DataFrameRegressionModel} = sqrt.(diag(sandwhich(r, k)))
# stderror(r::T, k::HC) where {T<:AbstractGLM} = sqrt.(diag(sandwhich(r, k)))
# stderror(r::T, k::HC) where {T<:LinearModel} = sqrt.(diag(sandwhich(r, k)))
#
# stderror(r::T, k::Type{R}) where {T<:DataFrameRegressionModel, R<:HC} = sqrt.(diag(sandwhich(r, k())))
# stderror(r::T, k::Type{R}) where {T<:AbstractGLM, R<:HC} = sqrt.(diag(sandwhich(r, k())))
# stderror(r::T, k::Type{R}) where {T<:LinearModel, R<:HC} = sqrt.(diag(sandwhich(r, k())))
