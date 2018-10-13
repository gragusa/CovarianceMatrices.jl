##############################################################################
##
## Optimal bandwidth
##
##############################################################################

abstract type BandwidthType{G} end
abstract type OptimalBandwidth end

struct NeweyWest<:OptimalBandwidth end
struct Andrews<:OptimalBandwidth end

struct Fixed<:BandwidthType{G where G} end
struct Optimal{G<:OptimalBandwidth}<:BandwidthType{G where G<:OptimalBandwidth} end

struct Prewhitened end
struct Unwhitened end

struct TruncatedKernel{G<:BandwidthType, F}<:HAC{G}
  bwtype::G
  bw::Vector{F}
  weights::Vector{F}
  prewhiten::Bool
end

struct BartlettKernel{G<:BandwidthType, F}<:HAC{G}
    bwtype::G
    bw::Vector{F}
    weights::Vector{F}
    prewhiten::Bool
end

struct ParzenKernel{G<:BandwidthType, F}<:HAC{G}
    bwtype::G
    bw::Vector{F}
    weights::Vector{F}
    prewhiten::Bool
end

struct TukeyHanningKernel{G<:BandwidthType, F}<:HAC{G}
    bwtype::G
    bw::Vector{F}
    weights::Vector{F}
    prewhiten::Bool
end

struct QuadraticSpectralKernel{G<:BandwidthType, F}<:HAC{G}
    bwtype::G
    bw::Vector{F}
    weights::Vector{F}
    prewhiten::Bool
end


const TRK=TruncatedKernel
const BTK=BartlettKernel
const PRK=ParzenKernel
const THK=TukeyHanningKernel
const QSK=QuadraticSpectralKernel

Optimal() = Optimal{Andrews}()

TruncatedKernel(;prewhiten=false)    = TRK(Optimal(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
BartlettKernel(;prewhiten=false)     = BTK(Optimal(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
ParzenKernel(;prewhiten=false)       = PRK(Optimal(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
TukeyHanningKernel(;prewhiten=false) = THK(Optimal(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
QuadraticSpectralKernel(;prewhiten=false) = QSK(Optimal(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)

BartlettKernel(x::Type{NeweyWest};prewhiten=false) = BTK(Optimal{NeweyWest}(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
ParzenKernel(x::Type{NeweyWest};prewhiten=false) = PRK(Optimal{NeweyWest}(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
QuadraticSpectralKernel(x::Type{NeweyWest};prewhiten=false) = QSK(Optimal{NeweyWest}(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
TukeyHanningKernel(x::Type{NeweyWest};prewhiten=false) = error("Newey-West optimal bandwidth does not support TukeyHanningKernel")
TruncatedKernel(x::Type{NeweyWest};prewhiten=false) = error("Newey-West optimal bandwidth does not support TuncatedKernel")

TruncatedKernel(x::Type{Andrews};prewhiten=false) = TRK(Optimal{Andrews}(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
BartlettKernel(x::Type{Andrews};prewhiten=false) = BTK(Optimal{Andrews}(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
ParzenKernel(x::Type{Andrews};prewhiten=false) = PRK(Optimal{Andrews}(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
TukeyHanningKernel(x::Type{Andrews};prewhiten=false) = THK(Optimal{Andrews}(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)
QuadraticSpectralKernel(x::Type{Andrews};prewhiten=false) = QSK(Optimal{Andrews}(), Array{Float64}(undef,1), Array{Float64}(undef,0), prewhiten)

TruncatedKernel(bw::Number;prewhiten=false) = TRK(Fixed(), [float(bw)], Array{Float64}(undef,0), prewhiten)
BartlettKernel(bw::Number;prewhiten=false) = BTK(Fixed(), [float(bw)], Array{Float64}(undef,0), prewhiten)
ParzenKernel(bw::Number;prewhiten=false) = PRK(Fixed(), [float(bw)], Array{Float64}(undef,0), prewhiten)
TukeyHanningKernel(bw::Number;prewhiten=false) = THK(Fixed(), [float(bw)], Array{Float64}(undef,0), prewhiten)
QuadraticSpectralKernel(bw::Number;prewhiten=false) = QSK(Fixed(), [float(bw)], Array{Float64}(undef,0), prewhiten)

bandwidth(k::HAC{G}, X::AbstractMatrix) where {G<:Fixed} = k.bw
bandwidth(k::HAC{Optimal{G}}, X::AbstractMatrix) where {G<:Andrews} = bwAndrews(k, )

function bandwidth(k::QuadraticSpectralKernel, X::AbstractMatrix)
    return k.bw(X, k)
end

isprewhiten(k::HAC) = k.prewhiten


struct HACConfig{TYPE, T1<:Real, F<:AbstractMatrix, V<:AbstractVector}
    prew::TYPE
    X_demean::F
    YY::F
    XX::F
    Y_lagged::F
    X_lagged::F
    μ::F
    Q::F
    V::F
    D::F
    U::V
    ρ::V
    σ⁴::V
    u::F
    chol::Cholesky{T1, F}
end

function HACConfig(X::AbstractMatrix{T}, k::HAC; returntype::Type{T1} = Float64) where {T<:Real, T1<:AbstractFloat}
    nr, p = size(X)
    ip = isprewhiten(k)
    TYPE = ip ? Prewhitened() : Unwhitened()
    n = ip ? nr-1 : nr
    if ip
    return HACConfig(TYPE, copy(convert(Array{T1}, X)),
                     Array{T1}(undef, n, p),
                     Array{T1}(undef, n, p),
                     Array{T1}(undef, n-1, p),
                     Array{T1}(undef, n-1, p),
                     Array{T1}(undef, 1, p),
                     Array{T1}(undef, p, p),
                     Array{T1}(undef, p, p),
                     Array{T1}(undef, p, p),
                     Array{T1}(undef, n-1),
                     Array{T1}(undef, p),
                     Array{T1}(undef, p),
                     Array{T1}(undef, n, p),
                     cholesky(Matrix(one(T1)I, p, p)))
    else
        return HACConfig(TYPE, copy(convert(Array{T1}, X)),
                                Array{T1}(undef, 0, 0),
                                Array{T1}(undef, n, p),
                                Array{T1}(undef, n-1, p),
                                Array{T1}(undef, n-1, p),
                                Array{T1}(undef, 1, p),
                                Array{T1}(undef, p, p),
                                Array{T1}(undef, p, p),
                                Matrix(one(T1)I, p, p),
                                Array{T1}(undef, n-1),
                                Array{T1}(undef, p),
                                Array{T1}(undef, p),
                                Array{T1}(undef, 0, 0),
                                cholesky(Matrix(one(T1)I, p, p)))
    end
end


function variance(X::AbstractMatrix, k::HAC; arg...)
    cfg = HACConfig(X, k)
    variance(X, k, cfg; arg...)
end

function variance(X::Matrix{F}, k::T, cfg::HACConfig; demean::Type{T1} = Val{true}, calculatechol::Bool = false) where {F, T, T1}
    demean!(cfg, X, demean)
    prewhiten!(cfg)
    _variance(k, cfg, calculatechol)
end

function _variance(k::HAC{Optimal{T}}, cfg, calculatechol) where T<:OptimalBandwidth
    n, p = size(cfg.XX)
    setupkernelweights!(k, p, eltype(cfg.XX))
    optimal_bw!(cfg, k, T())
    __variance(k::HAC, cfg, calculatechol)
end

function _variance(k::HAC{T}, cfg, calculatechol) where T<:Fixed
    __variance(k::HAC, cfg, calculatechol)
end

function __variance(k::HAC, cfg, calculatechol)
    n, p = size(cfg.XX)
    fill!(cfg.V, zero(eltype(cfg.XX)))
    bw = first(k.bw)
    mul!(cfg.V, cfg.XX', cfg.XX)
    triu!(cfg.V)
    idxs = getcovindeces(k, n)
    @inbounds for j in idxs
        k_j = CovarianceMatrices.kernel(k, j/bw)
        LinearAlgebra.axpy!(k_j, CovarianceMatrices.Γ!(cfg, j), cfg.V)
    end
    LinearAlgebra.copytri!(cfg.V, 'U')
    swhiten!(cfg)
    rmul!(cfg.V, 1/(n+isprewhiten(k)))
    calculatechol && makecholesky!(cfg)
    return cfg.V
end

getcovindeces(k::T, n) where T<:QuadraticSpectralKernel = Iterators.filter(x -> x!=0, -n:n)
getcovindeces(k::HAC, n) = Iterators.filter(x -> x!=0, -floor(Int, k.bw[1]):floor(Int, k.bw[1]))

function Γ!(cfg, j)
    X = cfg.XX
    T, p = size(X)
    Q = fill!(cfg.Q, zero(eltype(X)))
    if j >= 0
        for h=1:p, s = 1:h
            for t = j+1:T
                @inbounds Q[s, h] = Q[s, h] + X[t, s]*X[t-j, h]
            end
        end
    elseif j<0
        for h=1:p, s = 1:h
            for t = -j+1:T
                @inbounds Q[s,h] = Q[s ,h] + X[t+j, s]*X[t,h]
            end
        end
    end
    return cfg.Q
end

function demean!(cfg::HACConfig, X, ::Type{Val{true}})
    sum!(cfg.μ, X)
    rmul!(cfg.μ, 1/size(X,1))
    cfg.X_demean .= X .- cfg.μ
end

function demean!(cfg::HACConfig, X, ::Type{Val{false}})
    copyto!(cfg.X_demean, X)
end

prewhiten!(cfg::HACConfig{T}) where T<:Unwhitened = copyto!(cfg.XX, cfg.X_demean)
prewhiten!(cfg::HACConfig{T}) where T<:Prewhitened = fit_var!(cfg)
swhiten!(cfg::HACConfig{T}) where T<:Unwhitened = nothing

function swhiten!(cfg::HACConfig{T}) where T<:Prewhitened
    fill!(cfg.Q, zero(eltype(cfg.Q)))
    for i = 1:size(cfg.Q, 2)
        cfg.Q[i,i] = one(eltype(cfg.Q))
    end
    v = ldiv!(qr(I-cfg.D'), cfg.Q)
    cfg.V .= v*cfg.V*v'
end

function makecholesky!(cfg)
    chol = cholesky(Symmetric(cfg.V), check = false)
    copyto!(cfg.chol.UL.data, chol.UL.data)
    copyto!(cfg.chol.U.data, chol.U.data)
    copyto!(cfg.chol.L.data, chol.L.data)
end

##############################################################################
##
## Kernel methods
##
##############################################################################

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
    iszero(x) ? 1.0 : (z = 1.2*π*x; 3*(sin(z)/z-cos(z))*(1/z)^2)
end

function setupkernelweights!(k, p, xtype)
    if isempty(k.weights)
        for j in 1:p
            push!(k.weights, one(xtype))
        end
    elseif all(iszero.(k.weights))
        fill!(k.weights, one(xtype))
    end
end

##############################################################################
##
## Fit functions
##
##############################################################################

 function fit_var!(cfg::HACConfig)
     X, Y, Z, u, D = cfg.XX, cfg.YY, cfg.X_demean, cfg.u, cfg.D
     n, p = size(Z)
     @inbounds for j in 1:p, i = 1:n-1
         X[i,j] = Z[i,  j]
         Y[i,j] = Z[i+1,j]
     end
     ldiv!(D, qr(X), Y)
     @inbounds for j in 1:p, i = 1:n-1
         Y[i,j] = Z[i+1,j]
     end
     mul!(u, X, D)
     broadcast!(-, X, Y, u)
 end

 function fit_ar!(cfg)
     ## Estimate
     ##
     ## y_{t,j} = ρ y_{t-1,j} + ϵ
     σ⁴ = cfg.σ⁴
     ρ = cfg.ρ
     U = cfg.U
     n, p = size(cfg.XX)
     lag!(cfg)
     Y = cfg.Y_lagged
     X = cfg.X_lagged
     for j in 1:p
         y = view(Y, :, j)
         x = view(X, :, j)
         x .= x .- mean(x)
         y .= y .- mean(y)
         ρ[j] = sum(broadcast!(*, cfg.U, x, y))/sum(abs2, x)
         copyto!(U, y)
         x .= x.*ρ[j]
         broadcast!(-, U, U, x)
         σ⁴[j]  = (dot(U, U)/(n-1))^2
     end
 end

 function lag!(cfg)
     ## This construct two matrices
     ## Z_lagged we store X_demean[1:n-1, :]
     nl, pl = size(cfg.Y_lagged)
     n, p  = size(cfg.XX)
     for ic in 1:p
         for i = 2:n
             @inbounds cfg.Y_lagged[i-1, ic] = cfg.XX[i, ic]
             @inbounds cfg.X_lagged[i-1, ic] = cfg.XX[i-1, ic]
         end
     end
  end

##############################################################################
##
## Optimal bandwidth
##
##############################################################################

optimal_bw!(cfg, k::HAC, optype::T) where T<:NeweyWest = bwNeweyWest(cfg, k)
optimal_bw!(cfg, k::HAC, opttype::T) where T<:Andrews = bwAndrews(cfg, k)

function bwAndrews(cfg, k::HAC)
    isempty(k.weights) && (fill!(k.weights, 1.0))
    n, p  = size(cfg.XX)
    a1, a2 = getalpha!(cfg, k.weights)
    k.bw[1] = bw_andrews(k, a1, a2, n)
end

## ---> Andrews Optimal bandwidth <---
d_bw_andrews = Dict(:TruncatedKernel         => :(0.6611*(a2*n)^(0.2)),
                    :BartlettKernel          => :(1.1447*(a1*n)^(1/3)),
                    :ParzenKernel            => :(2.6614*(a2*n)^(0.2)),
                    :TukeyHanningKernel      => :(1.7462*(a2*n)^(0.2)),
                    :QuadraticSpectralKernel => :(1.3221*(a2*n)^(0.2)))

for kerneltype in [:TruncatedKernel, :BartlettKernel, :ParzenKernel, :TukeyHanningKernel, :QuadraticSpectralKernel]
    @eval $:(bw_andrews)(k::($kerneltype), a1, a2, n) = $(d_bw_andrews[kerneltype])
end

function getalpha!(cfg, w)
    fit_ar!(cfg)
    σ⁴, ρ = cfg.σ⁴, cfg.ρ
    nm = 4.0.*(ρ.^2).*σ⁴./(((1.0.-ρ).^6).*((1.0.+ρ).^2))
    dn = σ⁴./(1.0.-ρ).^4
    α₁ = sum(w.*nm)/sum(w.*dn)
    nm = 4.0.*(ρ.^2).*σ⁴./((1.0.-ρ).^8)
    α₂ = sum(w.*nm)/sum(w.*dn)
    return α₁, α₂
end

function bwNeweyWest(cfg, k::HAC)
    n, p = size(cfg.XX)
    l = getrates(cfg, k)
    w = k.weights
    xm = cfg.XX*w
    a = map(j -> dot(xm[1:n-j], xm[j+1:n])/n, 0:l)::Array{Float64, 1}
    aa = view(a, 2:l+1)
    a0 = a[1] + 2*sum(aa)
    a1 = 2*sum((1:l) .* aa)
    a2 = 2*sum((1:l).^2 .* aa)
    k.bw[1] = bwnw(k, a0, a1, a2)*(n+isprewhiten(k))^growthrate(k)
end

function getrates(cfg, k)
    n, p = size(cfg.X_demean)
    lrate = lagtruncation(k)
    adj = isprewhiten(k) ? 3 : 4
    floor(Int, adj*(n/100)^lrate)
end

@inline bwnw(k::BartlettKernel, s0, s1, s2) = 1.1447*((s1/s0)^2)^growthrate(k)
@inline bwnw(k::ParzenKernel, s0, s1, s2) = 2.6614*((s2/s0)^2)^growthrate(k)
@inline bwnw(k::QuadraticSpectralKernel, s0, s1, s2) = 1.3221*((s2/s0)^2)^growthrate(k)

## --> Newey-West Optimal bandwidth <---
@inline growthrate(k::HAC) = 1/5
@inline growthrate(k::BartlettKernel) = 1/3

@inline lagtruncation(k::BartlettKernel) = 2/9
@inline lagtruncation(k::ParzenKernel) = 4/25
@inline lagtruncation(k::QuadraticSpectralKernel) = 2/25


##############################################################################
##
## GLM
##
##############################################################################

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::HAC, cfg::CovarianceMatrices.HACConfig; demean = Val{false}, kwargs...)
    mf = esteq_hac!(cfg, m)
    br = pseudohessian(m)
    V = variance(mf, k, cfg, demean = demean)*size(cfg.X_demean,1)
    br*V*br'
end

function StatsBase.vcov(m::GLM.LinearModel, k::HAC, cfg::CovarianceMatrices.HACConfig; demean = Val{false}, kwargs...)
    mf = esteq_hac!(cfg, m)
    br = pseudohessian(m)
    V = variance(mf, k, cfg, demean = demean).*size(cfg.X_demean,1)
    br*V*br'
end

function StatsBase.vcov(m::GLM.LinearModel, k::HAC; kwargs...)
    cfg = HACConfig(modelmatrix(m), k)
    vcov(m, k, cfg, kwargs...)
end

function StatsBase.vcov(m::StatsModels.DataFrameRegressionModel, k::HAC; kwargs...)
    cfg = HACConfig(modelmatrix(m), k)
    vcov(m, k, cfg, kwargs...)
end



function esteq_hac!(cfg, m::RegressionModel)
    X = copy(modelmatrix(m))
    u = copy(residuals(m))
    if !isempty(getweights(m))
        broadcast!(*, X, X, sqrt.(getweights(m)))
        broadcast!(*, u, u, sqrt.(getweights(m)))
    end
    broadcast!(*, X, X, u)
    return X
end


## -----
## DataFramesRegressionModel/AbstractGLM methods
## -----
