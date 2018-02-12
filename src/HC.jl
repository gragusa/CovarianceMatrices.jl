## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----

CovarianceModels = Union{RegressionModel, AbstractGLM}

npars(r::CovarianceModels) = length(coef(r))
residualdof(r::CovarianceModels) = numobs(r) - npars(r)

function modelresiduals(r::CovarianceModels) 
    u = rawresiduals(r)
    wts = modelweights(r)
    u.*sqrt.(wts)
end

#region Adjustment factors
adjfactor!(u, r::CovarianceModels, k::HC0) = u[:] = one(Float64)
adjfactor!(u, r::CovarianceModels, k::HC1) = u[:] = numobs(r)./residualdof(r)
adjfactor!(u, r::CovarianceModels, k::HC2) = u[:] = 1./(1.-hatmatrix(r))
adjfactor!(u, r::CovarianceModels, k::HC3) = u[:] = 1./(1.-hatmatrix(r)).^2

function adjfactor!(u, r::CovarianceModels, k::HC4)
    h = hatmatrix(r)
    n = numobs(r)
    p = npars(r)
    @inbounds for j in eachindex(h)
        delta = min(4, n*h[j]/p)
        u[j] = 1/(1-h[j])^delta
    end
end

function adjfactor!(u, r::CovarianceModels, k::HC4m)
    h = hatmatrix(r)
    n = numobs(r)
    p = npars(r)
    @inbounds for j in eachindex(h)
        delta = min(1.0, n*h[j]/p) + min(1.5, n*h[j]/p)
        u[j] = 1/(1-h[j])^delta
    end
end

function adjfactor!(u, r::CovarianceModels, k::HC5)
    h     = hatmatrix(r)
    n     = numobs(r)
    p     = npars(r)
    mx    = max(n*0.7*maximum(h)/p, 4)
    @inbounds for j in eachindex(h)
        alpha =  min(n*h[j]/p, mx)
        u[j] = 1/(1-h[j])^alpha
    end
end
#endregion

#region Model Matrix and hatmatrix
function weightedmodelmatrix(r::CovarianceModels)
    w = modelweights(r)
    if isempty(w)
        copy(modelmatrix(r))
    else
        modelmatrix(r).*sqrt.(w)
    end
end

function fullyweightedmodelmatrix(r::CovarianceModels)
    w = modelweights(r)
    if isempty(w)
        copy(modelmatrix(r))
    else
        modelmatrix(r).*w
    end
end

function hatmatrix(r::CovarianceModels)
    z = weightedmodelmatrix(r)
    cf = choleskyfactor(r)
    Base.LinAlg.A_rdiv_B!(z, cf)
    diag(Base.LinAlg.A_mul_Bt(z, z))
end
#endregion

#region bread and meat
function meat(r::CovarianceModels, k::HC)
    u = copy(rawresiduals(r))
    X = fullyweightedmodelmatrix(r)
    z = X.*u
    adjfactor!(u, r, k)
    scale!(Base.LinAlg.At_mul_B(z, z.*u), 1/nobs(r))
end

function bread(r::CovarianceModels)
    A = invXX(r)
    scale!(A, nobs(r))
end

function meat(r::CovarianceModels, k::CRHC)
    idx   = sortperm(k.cl)
    cls   = k.cl[idx]
    ichol = invXX(r)
    X     = fullyweightedmodelmatrix(r)[idx,:]
    e     = rawresiduals(r)[idx]
    # w     = modelweights(r)
    bstarts = [searchsorted(cls, j[2]) for j in enumerate(unique(cls))]
    adjresid!(k, X, e, ichol, bstarts)
    M = zeros(size(X, 2), size(X, 2))
    clusterize!(M, X.*e, bstarts)
    return scale!(M, 1/nobs(r))
end
#endregion


function sandwhich(r::T, k::R) where {T<:CovarianceModels, R<:RobustVariance}
    B = meat(r, k)
    A = bread(r)
    scale!(A*B*A, 1/nobs(r))
end

function vcov(X::AbstractMatrix, v::HC)
    N, p = size(X)
    XX = Base.LinAlg.At_mul_B(X, X)
    return scale!(XX, 1/N)
end

## -----
## DataFrame methods
## -----
numobs(r::DataFrameRegressionModel) = size(r.model.pp.X, 1)
modelmatrix(r::DataFrameRegressionModel) = r.mm.m
rawresiduals(r::DataFrameRegressionModel) = r.model.rr.wrkresid
modelweights(r::DataFrameRegressionModel) = r.model.rr.wrkwt
modelweights(r::LinearModel) = r.rr.wts
choleskyfactor(r::DataFrameRegressionModel) = cholfact(r.model.pp)[:UL]
XX(r::DataFrameRegressionModel) = choleskyfactor(r)'*choleskyfactor(r)
invXX(r::DataFrameRegressionModel) = GLM.invchol(r.model.pp)
modelresponse(r::DataFrameRegressionModel) = r.model.rr.y
## -----
## GeneralizedLinearModel methods
## -----
FlatModels = Union{GeneralizedLinearModel, LinearModel}

numobs(r::FlatModels) = size(r.pp.X, 1)
modelmatrix(r::FlatModels) = r.pp.X
rawresiduals(r::GeneralizedLinearModel) = r.rr.wrkresid
rawresiduals(r::LinearModel) = modelresiduals(r)
modelweights(r::FlatModels) = r.rr.wrkwt
modelresponse(r::LinearModel) = r.rr.y
function modelresiduals(r::LinearModel)
    y = r.rr.y
    mu = r.rr.mu
    if isempty(modelweights(r))
        y - mu
    else
        wts = r.rr.wts
        resid = similar(y)
        @simd for i = eachindex(resid,y,mu,wts)
            @inbounds resid[i] = (y[i] - mu[i]) * sqrt(wts[i])
        end
        resid
    end
end

choleskyfactor(r::FlatModels) = cholfact(r.pp)[:UL]
XX(r::FlatModels) = choleskyfactor(r)'*choleskyfactor(r)
invXX(r::FlatModels) = GLM.invchol(r.pp)



## -----
## Clusters methods
## -----

nclus(k::CRHC) = length(unique(k.cl))

#region Residual adjustments
adjresid!(v::CRHC0, X, e, ichol, bstarts) = identity(e)
adjresid!(v::CRHC1, X, e, ichol, bstarts) = e[:] = scalaradjustment(X, bstarts)*e
adjresid!(v::CRHC2, X, e, ichol, bstarts) = getqii(v, e, X, ichol, bstarts)
adjresid!(v::CRHC3, X, e, ichol, bstarts) = scale!(scalaradjustment(X, bstarts), getqii(v, e, X, ichol, bstarts))

function getqii(v::CRHC2, e, X, A, bstarts)
    @inbounds for j in 1:length(bstarts)
        rnge = bstarts[j]
        se = view(e, rnge)
        sx = view(X, rnge,:)        
        BB = Symmetric(I - sx*A*sx')
        e[rnge] =  cholfact(BB)\se
    end
    return e
end

function getqii(v::CRHC3, e, X, A, bstarts)
    @inbounds for j in 1:length(bstarts)
        rnge = bstarts[j]
        se = view(e, rnge)
        sx = view(X, rnge, :)
        e[rnge] =  (I - sx*A*sx')\se
    end
    return e
end

function scalaradjustment(X, bstarts)
    n, k = size(X)
    g    = length(bstarts)
    sqrt.((n-1)/(n-k) * g/(g-1))
end
#endregion

#region Utility
function clusterize!(M::Matrix, U::Matrix, bstarts)
    k, k = size(M)
    s = Array{Float64}(k)
    for m = 1:length(bstarts)
        for i = 1:k
            @inbounds s[i] = zero(Float64)
        end
        for j = 1:k, i = bstarts[m]
            @inbounds s[j] += U[i, j]
        end
        for j = 1:k, i = 1:k
            @inbounds M[i, j] += s[i]*s[j]
        end
    end
end
#endregion




# vcov(r::T, k::HC) where {T<:CovarianceModels} = sandwhich(r, k)
# vcov(r::T, k::Type{RobustVariance}) where {T<:CovarianceModels} = sandwhich(r, k())

# vcov(r::T, k::CRHC) where {T<:CovarianceModels} = sandwhich(r, k)

vcov(r::T, k::CRHC) where {T<:DataFrameRegressionModel} = sandwhich(r, k)
vcov(r::T, k::CRHC) where {T<:AbstractGLM} = sandwhich(r, k)
vcov(r::T, k::CRHC) where {T<:LinearModel} = sandwhich(r, k)

vcov(r::T, k::HC) where {T<:DataFrameRegressionModel} = sandwhich(r, k)
vcov(r::T, k::HC) where {T<:AbstractGLM} = sandwhich(r, k)
vcov(r::T, k::HC) where {T<:LinearModel} = sandwhich(r, k)

vcov(r::T, k::Type{R}) where {T<:DataFrameRegressionModel, R<:HC} = sandwhich(r, k())
vcov(r::T, k::Type{R}) where {T<:AbstractGLM, R<:HC} = sandwhich(r, k())
vcov(r::T, k::Type{R}) where {T<:LinearModel, R<:HC} = sandwhich(r, k())

stderr(r::T, k::CRHC) where {T<:DataFrameRegressionModel} = sqrt.(diag(sandwhich(r, k)))
stderr(r::T, k::CRHC) where {T<:AbstractGLM} = sqrt.(diag(sandwhich(r, k)))
stderr(r::T, k::CRHC) where {T<:LinearModel} = sqrt.(diag(sandwhich(r, k)))

stderr(r::T, k::HC) where {T<:DataFrameRegressionModel} = sqrt.(diag(sandwhich(r, k)))
stderr(r::T, k::HC) where {T<:AbstractGLM} = sqrt.(diag(sandwhich(r, k)))
stderr(r::T, k::HC) where {T<:LinearModel} = sqrt.(diag(sandwhich(r, k)))

stderr(r::T, k::Type{R}) where {T<:DataFrameRegressionModel, R<:HC} = sqrt.(diag(sandwhich(r, k())))
stderr(r::T, k::Type{R}) where {T<:AbstractGLM, R<:HC} = sqrt.(diag(sandwhich(r, k())))
stderr(r::T, k::Type{R}) where {T<:LinearModel, R<:HC} = sqrt.(diag(sandwhich(r, k())))




