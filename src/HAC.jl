kernel(k::HAC, x) = isnan(x) ? (return 1.0) : kernel(k, float(x))

kernel(k::TruncatedKernel, x::Float64)    = (abs(x) <= 1.0) ? 1.0 : 0.0
kernel(k::BartlettKernel, x::Float64)     = (abs(x) <= 1.0) ? (1.0 - abs(x)) : 0.0
kernel(k::TukeyHanningKernel, x::Float64) = (abs(x) <= 1.0) ? 0.5 * (1.0 + cospi(x)) : 0.0

function kernel(k::ParzenKernel, x::Float64)
    ax = abs(x)
    if ax > 1.0
        0.0
    elseif ax <= 0.5
        1.0 - 6.0 * ax^2 + 6.0 * ax^3
    else
        2.0 * (1.0 - ax)^3
    end
end

function kernel(k::QuadraticSpectralKernel, x::Float64)
    iszero(x) ? 1.0 : (- cosc(1.2 * x) * twohalftoπ² / x)
end

##############################################################################
##
## Optimal bandwidth
##
##############################################################################

abstract type BandwidthType{G} end
abstract type OptimalBandwidth end

struct NeweyWest <: OptimalBandwidth end
struct Andrews <: OptimalBandwidth end

struct Fixed <: BandwidthType{G where G} end
struct Optimal{G<:OptimalBandwidth} <: BandwidthType{G where G<:OptimalBandwidth} end

struct TruncatedKernel{G <: BandwidthType} <: HAC{G}
  bwtype::G
  bw::Vector{Float64}
  weights::Vector{Float64}
end

struct BartlettKernel{G <: BandwidthType} <: HAC{G}
    bwtype::G
    bw::Vector{Float64}
    weights::Vector{Float64}
end

struct ParzenKernel{G <: BandwidthType} <: HAC{G}
    bwtype::G
    bw::Vector{Float64}
    weights::Vector{Float64}
end

struct TukeyHanningKernel{G <: BandwidthType} <: HAC{G}
    bwtype::G
    bw::Vector{Float64}
    weights::Vector{Float64}
end

struct QuadraticSpectralKernel{G <: BandwidthType} <: HAC{G}
    bwtype::G
    bw::Vector{Float64}
    weights::Vector{Float64}
end

struct VARHAC
    imax::Int64
    ilag::Int64
    imodel::Int64
end

const TRK=TruncatedKernel
const BTK=BartlettKernel
const PRK=ParzenKernel
const THK=TukeyHanningKernel
const QSK=QuadraticSpectralKernel

Optimal() = Optimal{Andrews}()

TruncatedKernel() = TRK(Optimal(), Array{Float64}(1), Array{Float64}(0))
BartlettKernel() = BTK(Optimal(), Array{Float64}(1), Array{Float64}(0))
ParzenKernel() = PRK(Optimal(), Array{Float64}(1), Array{Float64}(0))
TukeyHanningKernel() = THK(Optimal(), Array{Float64}(1), Array{Float64}(0))
QuadraticSpectralKernel() = QSK(Optimal(), Array{Float64}(1), Array{Float64}(0))


BartlettKernel(x::Type{NeweyWest}) = BTK(Optimal{NeweyWest}(), Array{Float64}(1), Array{Float64}(0))
ParzenKernel(x::Type{NeweyWest}) = PRK(Optimal{NeweyWest}(), Array{Float64}(1), Array{Float64}(0))
QuadraticSpectralKernel(x::Type{NeweyWest}) = QSK(Optimal{NeweyWest}(), Array{Float64}(1), Array{Float64}(0))
TukeyHanningKernel(x::Type{NeweyWest}) = error("Newey-West optimal bandwidth does not support TukeyHanningKernel")
TruncatedKernel(x::Type{NeweyWest}) = error("Newey-West optimal bandwidth does not support TuncatedKernel")

TruncatedKernel(x::Type{Andrews}) = TRK(Optimal{Andrews}(), Array{Float64}(1), Array{Float64}(0))
BartlettKernel(x::Type{Andrews}) = BTK(Optimal{Andrews}(), Array{Float64}(1), Array{Float64}(0))
ParzenKernel(x::Type{Andrews}) = PRK(Optimal{Andrews}(), Array{Float64}(1), Array{Float64}(0))
TukeyHanningKernel(x::Type{Andrews}) = THK(Optimal{Andrews}(), Array{Float64}(1), Array{Float64}(0))
QuadraticSpectralKernel(x::Type{Andrews}) = QSK(Optimal{Andrews}(), Array{Float64}(1), Array{Float64}(0))


TruncatedKernel(bw::Number) = TRK(Fixed(), [float(bw)], Array{Float64}(0))
BartlettKernel(bw::Number) = BTK(Fixed(), [float(bw)], Array{Float64}(0))
ParzenKernel(bw::Number) = PRK(Fixed(), [float(bw)], Array{Float64}(0))
TukeyHanningKernel(bw::Number) = THK(Fixed(), [float(bw)], Array{Float64}(0))
QuadraticSpectralKernel(bw::Number) = QSK(Fixed(), [float(bw)], Array{Float64}(0))

VARHAC() = VARHAC(2, 2, 1)
VARHAC(imax::Int64) = VARHAC(imax, 2, 1)

bandwidth(k::HAC{G}, X::AbstractMatrix) where {G<:Fixed} = k.bw
bandwidth(k::HAC{Optimal{G}}, X::AbstractMatrix) where {G<:Andrews} = bwAndrews(k, )

function bandwidth(k::QuadraticSpectralKernel, X::AbstractMatrix)
    return k.bw(X, k)
end

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

function vcov(X::AbstractMatrix, k::HAC, bw, D, prewhite::Bool)
    n, p = size(X)
    Q  = zeros(p, p)
    for j in -floor(Int, bw):floor(Int, bw)
        Base.BLAS.axpy!(kernel(k, j/bw), Γ(X, j), Q)
    end
    Base.LinAlg.copytri!(Q, 'U')
    if prewhite
        Q[:] = D*Q*D'
        n += 1
    end
    return scale!(Q, 1/n)
end

function vcov(X::AbstractMatrix, k::QuadraticSpectralKernel, bw, D, prewhite::Bool)
    n, p = size(X)
    Q  = zeros(p, p)
    for j in -n:n
        Base.BLAS.axpy!(kernel(k, j/bw), Γ(X, j), Q)
    end
    Base.LinAlg.copytri!(Q, 'U')
    if prewhite
        Q[:] = D*Q*D'
        n += 1
    end
    return scale!(Q, 1/n)
end

function vcov(X::AbstractMatrix, k::HAC{Fixed}; prewhite::Bool=true)
    D = I
    !prewhite || ((X, D) = pre_white(X))
    bw = k.bw[1]
    vcov(X, k, bw, D, prewhite)
end

function vcov(X::AbstractMatrix, k::HAC{Optimal{T}}; prewhite::Bool=true) where T<:Fixed
    p = size(X, 2)
    D = I
    !prewhite || ((X, D) = pre_white(X))
    #isempty(k.weights) && (k.weights = ones(p))
    bw = optimal_bw(X, k, T(), ones(p), prewhite)
    vcov(X, k, bw, D, prewhite)
end

function vcov(X::AbstractMatrix, k::HAC{Optimal{T}}; prewhite::Bool=true) where T<:OptimalBandwidth
    p = size(X, 2)
    D = I
    !prewhite || ((X, D) = pre_white(X))
    if isempty(k.weights)
        for j in 1:p
            push!(k.weights, 1.0)
        end
    end
    bw = optimal_bw(X, k, T(), k.weights, prewhite)
    vcov(X, k, bw, D, prewhite)
end

function vcov(r::DataFrameRegressionModel, k::HAC{Optimal{T}}; args...) where T<:OptimalBandwidth
    p = size(r.model.pp.X, 2)
    for j in coefnames(r.mf)
        if j == "(Intercept)"
            push!(k.weights, 0.0)
        else
            push!(k.weights, 1.0)
        end
    end
    # w = ones(p)
    # "(Intercept)" ∈ coefnames(r.mf) && (w[find("(Intercept)" .== coefnames(r.mf))] = 0)
    # k.weights = w
    variance(r, k; args...)
end

vcov(r::DataFrameRegressionModel, k::HAC{T}; args...) where {T<:Fixed} = variance(r, k; args...)

stderr(x::DataFrameRegressionModel, k::HAC; kwargs...) = sqrt.(diag(vcov(x, k; kwargs...)))

function variance(r::DataFrameRegressionModel, k::HAC; args...) 
    B = meat(r, k; args...)
    A = bread(r, k)
    scale!(A*B*A, 1/nobs(r))
end

function meat(r::DataFrameRegressionModel, k::HAC; args...)
    u = modelresiduals(r)    
    X = modelmatrix(r)
    z = X.*u
    vcov(z, k; args...)
end

function bread(r::DataFrameRegressionModel, k::HAC; arg...)
    A = invXX(r)
    scale!(A, nobs(r))
end
