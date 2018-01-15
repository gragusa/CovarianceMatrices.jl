function vcov(X::AbstractMatrix, v::HC)
    N, p = size(X)
    XX = Base.LinAlg.At_mul_B(X, X)
    return scale!(XX, 1/N)
end

const GenLinMod=GeneralizedLinearModel

_nobs(r::DataFrameRegressionModel) = size(r.model.pp.X, 1)
_df_residual(r::DataFrameRegressionModel) = _nobs(r) - length(coef(r))


adjfactor!(u, r::DataFrameRegressionModel, k::HC0) = u[:] = one(Float64)
adjfactor!(u, r::DataFrameRegressionModel, k::HC1) = u[:] = _nobs(r)./_df_residual(r)
adjfactor!(u, r::DataFrameRegressionModel, k::HC2) = u[:] = 1./(1.-hatmatrix(r))
adjfactor!(u, r::DataFrameRegressionModel, k::HC3) = u[:] = 1./(1.-hatmatrix(r)).^2
function adjfactor!(u, r::DataFrameRegressionModel, k::HC4)
    h = hatmatrix(r)
    n = _nobs(r)
    p = npars(r)
    @inbounds for j in eachindex(h)
        delta = min(4, n*h[j]/p)
        u[j] = 1/(1-h[j])^delta
    end
end

function adjfactor!(u, r::DataFrameRegressionModel, k::HC4m)
    h = hatmatrix(r)
    n = _nobs(r)
    p = npars(r)
    @inbounds for j in eachindex(h)
        delta = min(1.0, n*h[j]/p) + min(1.5, n*h[j]/p)
        u[j] = 1/(1-h[j])^delta
    end
end

function adjfactor!(u, r::DataFrameRegressionModel, k::HC5)
    h     = hatmatrix(r)
    n     = _nobs(r)
    p     = npars(r)
    mx    = max(n*0.7*maximum(h)/p, 4)
    @inbounds for j in eachindex(h)
        alpha =  min(n*h[j]/p, mx)
        u[j] = 1/(1-h[j])^alpha
    end
end

nclus(k::CRHC) = length(unique(k.cl))
npars(r::DataFrameRegressionModel) = length(coef(r))


# residuals(l::LinPredModel, k::HC) = residuals(l)
# residuals(l::LinPredModel, k::HAC) = residuals(l)

function residuals(r::GLM.ModResp)
    a = r.wrkwt
    u = copy(r.wrkresid)
    broadcast!(*, u, u, a)    
end

# function wrkresidwts(r::DataFrameRegressionModel)
#     a = r.model.rr.wrkwt
#     u = copy(wrkresid(r))
#     broadcast!(*, u, u, a)
# end

function weightedModelMatrix(r::DataFrameRegressionModel)
    w = r.model.rr.wrkwt
    (r.mm.m).*sqrt.(w)
end

function hatmatrix(r::DataFrameRegressionModel)
    z = weightedModelMatrix(r)
    cf = cholfact(r.model.pp)[:UL]
    Base.LinAlg.A_rdiv_B!(z, cf)
    diag(Base.LinAlg.A_mul_Bt(z, z))
end


## Entry point

vcov(r::DataFrameRegressionModel, k::HC) = variance(r, k)
vcov(r::DataFrameRegressionModel, k::Type{T}) where {T<:RobustVariance} = variance(r, k())

stderr(r::DataFrameRegressionModel, k::Type{T}) where {T<:HC} = sqrt.(diag(vcov(r, k())))
stderr(r::DataFrameRegressionModel, k::T) where {T<:CRHC} = sqrt.(diag(vcov(r, k)))

function variance(r::DataFrameRegressionModel, k::HC)
    B = meat(r, k)
    A = bread(r)
    scale!(A*B*A, 1/nobs(r))
end

## Note to myself: the function residuals(r::DataFrameRegressoinModel) returns 
## the weighted residualds

function meat(r::DataFrameRegressionModel, k::HC)
    u = residuals(r)
    X = r.mm.m
    z = X.*u
    adjfactor!(u, r, k)
    scale!(Base.LinAlg.At_mul_B(z, z.*u), 1/nobs(r))
end

function bread(r::DataFrameRegressionModel)
    A = inv(cholfact(r.model.pp))::Array{Float64, 2}
    scale!(A, nobs(r))
end


################################################################################
## Cluster
################################################################################

vcov(r::DataFrameRegressionModel, k::CRHC) = variance(r, k)

function variance(r::DataFrameRegressionModel, k::CRHC)   
    M = meat(r, k)::Array{Float64,2}
    B = bread(r)::Array{Float64, 2}    
    scale!(B*M*B, 1/nobs(r))
end

function meat(r::DataFrameRegressionModel, k::CRHC)
    idx   = sortperm(k.cl)
    cls   = k.cl[idx]
    ichol = inv(cholfact(r.model.pp))::Array{Float64, 2}
    X     = r.mm.m[idx,:]
    e     = GLM.residuals(r)[idx]
    w     = r.model.rr.wts
    
    # if !isempty(w)
    #     w = w[idx]
    #     broadcast!(*, X, X, sqrt.(w))
    #     broadcast!(*, e, e, sqrt.(w))
    # end
    bstarts = [searchsorted(cls, j[2]) for j in enumerate(unique(cls))]
    adjresid!(k, X, e, ichol, bstarts)
    M = zeros(size(X, 2), size(X, 2))
    clusterize!(M, X.*e, bstarts)
    return scale!(M, 1/nobs(r))
end



function clusterize!(M, U, bstarts)
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

function getqii(v::CRHC3, e, X, A, bstarts)
    @inbounds for j in 1:length(bstarts)
        rnge = bstarts[j]
        se = view(e, rnge)
        sx = view(X, rnge, :)
        e[rnge] =  (I - sx*A*sx')\se
    end
    return e
end


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

_adjresid!(v::CRHC, X, e, chol, bstarts) =  getqii(v, e, X, chol, bstarts)
_adjresid!(v::CRHC, X, e, ichol, bstarts, c::Float64) = scale!(c, _adjresid!(v::CRHC, X, e, ichol, bstarts))

function scalar_adjustment(X, bstarts)
    n, k = size(X)
    g    = length(bstarts)
    sqrt.((n-1)/(n-k) * g/(g-1))
end

adjresid!(v::CRHC0, X, e, ichol, bstarts) = identity(e)
adjresid!(v::CRHC1, X, e, ichol, bstarts) = e[:] = scalar_adjustment(X, bstarts)*e
adjresid!(v::CRHC2, X, e, ichol, bstarts) = _adjresid!(v, X, e, ichol, bstarts, 1.0)
adjresid!(v::CRHC3, X, e, ichol, bstarts) = _adjresid!(v, X, e, ichol, bstarts, scalar_adjustment(X, bstarts))

