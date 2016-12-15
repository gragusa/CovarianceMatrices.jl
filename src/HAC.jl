function k_tr{T}(x::T)
    if(isnan(x) || abs(x)<= one(T))
        return one(Float64)
    else
        return zero(Float64)
    end
end

function k_bt{T}(x::T)
    if isnan(x)
        one(Float64)
    end
    float(max(one(T)-abs(x), zero(T)))
end

function k_pr{T}(x::T)
    if isnan(x)
        one(Float64)
    end
    ax = abs(x)
    if(ax > one(T))
        zero(Float64)
    elseif ax <= .5
        float(1 - 6 * x^2 + 6 * ax^3)
    else
        float(2 * (1-ax)^3)
    end
end

function k_qs{T <: Number}(x::T)
    if isnan(x)
        one(Float64)
    end
    if(isequal(x, zero(eltype(x))))
        one(Float64)
    else
        return (25/(12*π²*x^2))*(sin(sixπ*x/5)/(sixπ*x/5)-cos(sixπ*x/5))
    end
end

function k_th{T <: Number}(x::T)
    if isnan(x)
        one(Float64)
    end
    ax = abs(x)
    if(ax < one(T))
        (1 + cos(π*x))/2
    else
        zero(Float64)
    end
end

##############################################################################
##
## Optimal band-width
##
##############################################################################
type TruncatedKernel <: HAC
  kernel::Function
  bw::Function
end

type BartlettKernel <: HAC
  kernel::Function
  bw::Function
end

type ParzenKernel <: HAC
  kernel::Function
  bw::Function
end

type TukeyHanningKernel <: HAC
  kernel::Function
  bw::Function
end

type QuadraticSpectralKernel <: HAC
  kernel::Function
  bw::Function
end

type VARHAC <: HAC
    imax::Int64
    ilag::Int64
    imodel::Int64
end

typealias TRK TruncatedKernel
typealias BTK BartlettKernel
typealias PRK ParzenKernel
typealias THK TukeyHanningKernel
typealias QSK QuadraticSpectralKernel

TruncatedKernel()                   = TRK(k_tr, optimalbw_ar_one)
BartlettKernel()                    = BTK(k_bt, optimalbw_ar_one)
ParzenKernel()                      = PRK(k_pr, optimalbw_ar_one)
TukeyHanningKernel()                = THK(k_th, optimalbw_ar_one)
QuadraticSpectralKernel()           = QSK(k_qs, optimalbw_ar_one)

TruncatedKernel(bw::Number)         = TRK(k_tr, (x, k) -> float(bw))
BartlettKernel(bw::Number)          = BTK(k_bt, (x, k) -> float(bw))
ParzenKernel(bw::Number)            = PRK(k_pr, (x, k) -> float(bw))
TukeyHanningKernel(bw::Number)      = THK(k_th, (x, k) -> float(bw))
QuadraticSpectralKernel(bw::Number) = QSK(k_qs, (x, k) -> float(bw))

VARHAC()                            = VARHAC(2, 2, 1)
VARHAC(imax::Int64)                 = VARHAC(imax, 2, 1)

function bandwidth(k::HAC, X::AbstractMatrix)
  return floor(k.bw(X, k))
end

function bandwidth(k::QuadraticSpectralKernel, X::AbstractMatrix)
    return k.bw(X, k)
end

kernel(k::HAC, x::Real) = k.kernel(x)

function Γ(X::AbstractMatrix, j::Int64)
    T, p = size(X)
    Q = zeros(eltype(X), p, p)
    if j>=0
        for h=1:p, s = 1:h
            for t = j+1:T
                @inbounds Q[s, h] = Q[s, h] + X[t, s]*X[t-j, h]
            end
        end
    else
        for h=1:p, s = 1:h
            for t = -j+1:T
                @inbounds Q[s,h] = Q[s ,h] + X[t+j, s]*X[t,h]
            end
        end
    end
    return Q
end

vcov(X::AbstractMatrix, k::VARHAC) = varhac(X, k.imax, k.ilag, k.imodel)

function vcov(X::AbstractMatrix, k::HAC; prewhite::Bool = true)
    n, p = size(X)
    !prewhite || ((X, D) = pre_white(X))
    bw = bandwidth(k, X)
    Q  = zeros(eltype(X), p, p)
    for j=-bw:bw
        Base.BLAS.axpy!(kernel(k, j/bw), Γ(X, Int(j)), Q)
    end
    Base.LinAlg.copytri!(Q, 'U')
    if prewhite
        Q[:] = D*Q*D'
    end
    return scale!(Q, 1/n)
end

function vcov(X::AbstractMatrix, k::QuadraticSpectralKernel; prewhite::Bool = true)
    n, p = size(X)
    !prewhite || ((X, D) = pre_white(X))
    bw = bandwidth(k, X)
    Q = zeros(eltype(X), p, p)
    for j=-n:n
        Base.BLAS.axpy!(kernel(k, j/bw), Γ(X, Int(j)), Q)
    end
    Base.LinAlg.copytri!(Q, 'U')
    if prewhite
        Q[:] = D*Q*D'
    end
    return scale!(Q, 1/n)
end

vcov(x::DataFrameRegressionModel, k::HAC; args...) = vcov(x.model, k; args...)

function vcov(ll::LinPredModel, k::HAC; args...)
    B = meat(ll, k; args...)
    A = bread(ll)
    scale!(A*B*A, 1/nobs(ll))
end

function meat(l::LinPredModel,  k::HAC; args...)
    u = wrkresidwts(l.rr)
    X = ModelMatrix(l)
    z = X.*u
    vcov(z, k; args...)
end
