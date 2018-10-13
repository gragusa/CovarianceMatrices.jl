struct HCConfig{F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector}
    q::F1
    X::F1
    x::F2
    v::V
    w::V
    η::V
    u::V
end

function HCConfig(X::AbstractMatrix{T1}; returntype::Type{T1} = eltype(X)) where T1
    n, p = size(X)
    HCConfig(similar(X), similar(X), Array{T1, 2}(undef, p, p),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n),
             Array{T1, 1}(undef, n), Array{T1, 1}(undef, n))
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
## Model Regressors methods
## -----
StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::HC) = vcov(m, k, HCConfig(m))
StatsBase.vcov(m::GLM.LinearModel, k::HC) = vcov(m, k, HCConfig(m))

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::HC, cfg::CovarianceMatrices.HCConfig)
    installxuw!(cfg, m)
    mf = esteq!(cfg, m)
    br = pseudohessian(cfg, m)
    adjfactor!(cfg, m, k)
    mul!(cfg.x, mf', broadcast!(*, cfg.X, mf, cfg.η))
    br*cfg.x*br'
end

function StatsBase.vcov(m::GLM.LinearModel, k::HC, cfg::CovarianceMatrices.HCConfig)
    installxuw!(cfg, m)
    mf = esteq!(cfg, m)
    br = pseudohessian(cfg, m)
    adjfactor!(cfg, m, k)
    mul!(cfg.x, mf', broadcast!(*, cfg.X, mf, cfg.η))
    br*cfg.x*br'
end

StatsBase.stderror(m::StatsModels.DataFrameRegressionModel, k::HC, args...) = sqrt.(diag(vcov(m, k, args...)))
StatsBase.stderror(m::GLM.LinearModel, k::HC, args...) = sqrt.(diag(vcov(m, k, args...)))

## -----
## Model Regressors methods
## -----

StatsModels.modelmatrix(m::T) where T<:StatsModels.DataFrameRegressionModel = m.mm.m
StatsModels.modelmatrix(m::T) where T<:GLM.LinearModel = m.pp.X

function modelmatrix!(cfg, m::T, weighted::Type{Val{false}}) where T<:Union{StatsModels.DataFrameRegressionModel, GLM.LinearModel}
    return cfg.X
end

residuals(m::StatsModels.DataFrameRegressionModel{T}) where T<:GLM.GeneralizedLinearModel = m.model.rr.wrkresid
getweights(m::StatsModels.DataFrameRegressionModel) = m.model.rr.wts
getweights(m::GLM.LinearModel) = m.rr.wts

function installxuw!(cfg, m::T) where T<:Union{StatsModels.DataFrameRegressionModel, GLM.LinearModel}
    copyto!(cfg.X, modelmatrix(m))
    copyto!(cfg.u, residuals(m))
    if !isempty(getweights(m))
        cfg.w .= sqrt.(getweights(m))
        broadcast!(*, cfg.X, cfg.X, cfg.w)
        broadcast!(*, cfg.u, cfg.u, cfg.w)
    end
end

choleskyfactor(m::StatsModels.DataFrameRegressionModel) = m.model.pp.chol.UL
choleskyfactor(m::GLM.LinearModel) = m.pp.chol.UL

## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----

function esteq!(cfg, m::StatsModels.RegressionModel)
    broadcast!(*, cfg.q, cfg.X, cfg.u)
    return cfg.q
end

pseudohessian(cfg, m::StatsModels.DataFrameRegressionModel) = GLM.invchol(m.model.pp)
pseudohessian(m::StatsModels.DataFrameRegressionModel) = GLM.invchol(m.model.pp)

function pseudohessian(cfg, m::GLM.LinearModel)
    X = cfg.X
    mul!(cfg.x, X', X)
    inv(cholesky!(cfg.x))
end

## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----

function hatmatrix(cfg, m::T) where T<:Union{StatsModels.DataFrameRegressionModel, GLM.LinearModel}
      z = cfg.X  ## THIS ASSUME THAT X IS WEIGHTED BY SQRT(W)
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
