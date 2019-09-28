const WFLOAT = Sys.WORD_SIZE == 64 ? Float64 : Float32

#=========
Abstraction
==========#

abstract type RobustVariance <: CovarianceEstimator end
abstract type HAC{G} <: RobustVariance end
abstract type HC <: RobustVariance end
abstract type CRHC{V} <: RobustVariance end


#=========
HAC Types
=========#

abstract type BandwidthType{G} end
abstract type OptimalBandwidth end

struct NeweyWest<:OptimalBandwidth end
struct Andrews<:OptimalBandwidth end

struct Fixed<:BandwidthType{G where G} end
struct Optimal{G<:OptimalBandwidth}<:BandwidthType{G where G<:OptimalBandwidth} end

struct Prewhiten end
struct Unwhiten end

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


#=========
HC Types
=========#
struct HC0  <: HC end
struct HC1  <: HC end
struct HC2  <: HC end
struct HC3  <: HC end
struct HC4  <: HC end
struct HC4m <: HC end
struct HC5  <: HC end

#const CLVector{T<:Integer} = DenseArray{T,1}

mutable struct CRHC0{V<:AbstractVector}  <: CRHC{V}
    cl::V
end

mutable struct CRHC1{V<:AbstractVector}  <: CRHC{V}
    cl::V
end

mutable struct CRHC2{V<:AbstractVector}  <: CRHC{V}
    cl::V
end

mutable struct CRHC3{V<:AbstractVector}  <: CRHC{V}
    cl::V
end

struct CovarianceMatrix{T2<:Factorization, T3<:CovarianceMatrices.RobustVariance, F1, T1<:AbstractMatrix{F1}} <: AbstractMatrix{F1}
    F::T2       ## Factorization
    K::T3       ## RobustVariance, e.g. HC0()
    V::T1       ## The Variance Covariance
end



#=======
Caches
=======#

abstract type AbstractCache end

struct HACCache{TYPE, F<:AbstractMatrix, T<:AbstractVector} <: AbstractCache
    prew::TYPE
    q::F         ## nxp
    YY::F        ## nxp
    XX::F        ## nxp
    YL::F        ## (n-1)xp 
    XL::F        ## (n-1)xp
    μ::F         ## 1 x p
    Q::F         ## p x p
    V::F         ## p x p
    D::F         ## p x p
    U::T         ## (n-1) x 1
    ρ::T         ## px1
    σ⁴::T        ## px1
    u::F         ## nxp
end

function HACCache(X::AbstractMatrix{T}, ::Type{Val{:prewhiten}}) where T
    N, p = size(X)
    n = N - 1
    return HACCache(Prewhiten(), 
                     collect(X),                    ## q
                     Matrix{T}(undef, n, p),        ## YY
                     Matrix{T}(undef, n, p),        ## XX
                     Matrix{T}(undef, n-1, p),      ## YL
                     Matrix{T}(undef, n-1, p),      ## XL
                     Matrix{T}(undef, 1, p),        ## μ
                     Matrix{T}(undef, p, p),        ## Q
                     Matrix{T}(undef, p, p),        ## V
                     Matrix{T}(undef, p, p),        ## D
                     Vector{T}(undef, n-1),         ## U
                     Vector{T}(undef, p),           ## ρ
                     Vector{T}(undef, p),           ## σ⁴
                     Matrix{T}(undef, n, p))        ## u
end

function HACCache(X::AbstractMatrix{T}, ::Type{Val{:unwhiten}}) where T
    n, p = size(X)
    return HACCache(Unwhiten(), 
                     collect(X),                    ## q
                     Matrix{T}(undef, 0, 0),        ## YY
                     Matrix{T}(undef, n, p),        ## XX
                     Matrix{T}(undef, n-1, p),      ## YL
                     Matrix{T}(undef, n-1, p),      ## XL
                     Matrix{T}(undef, 1, p),        ## μ
                     Matrix{T}(undef, p, p),        ## Q
                     Matrix{T}(undef, p, p),        ## V
                     Matrix{T}(undef, p, p),        ## D
                     Vector{T}(undef, n-1),         ## U
                     Vector{T}(undef, p),           ## ρ
                     Vector{T}(undef, p),           ## σ⁴
                     Matrix{T}(undef, 0, 0))        ## u
end

function HACCache(X::AbstractMatrix{T}; prewhiten::Bool = false) where T<:Int
    HACCache(convert(Matrix{WFLOAT}, X), prewhiten = prewhiten)
end

function HACCache(X::AbstractMatrix{T}; prewhiten::Bool = false) where T
    wtype = prewhiten ? :prewhiten : :unwhiten
    HACCache(X, Val{wtype})
end

function HACCache(X::AbstractMatrix, k::HAC; kwargs...)
    ip = isprewhiten(k)
    HACCache(X, prewhiten = ip; kwargs...)
end


struct CRHCCache{VN<:AbstractVector, F1<:AbstractMatrix, F2<:AbstractMatrix, V<:AbstractVector, IN<:AbstractVector} <: AbstractCache
    q::F1   # nxm
    X::F1   # nxm
    V::F2   # nxm
    v::V    # (n-1)xp
    w::V    # (n-1)xp
    μ::V    # 1xp
    u::V    
    M::F1
    clusidx::IN
    clus::VN
end


struct HCCache{F1<:AbstractMatrix, V1<:AbstractVector} <: AbstractCache
    q::F1   # NxM
    X::F1   # NxM
    v::V1   # nx1 
    η::V1   # nx1
    u::V1   # nx1
    V::F1   # mxm
    μ::F1   # 1xp
end

function HCCache(X::AbstractMatrix{T}; kwargs...) where T
    n, p = size(X)
    HCCache(collect(X),               ## q
            collect(X),               ## X
            Vector{T}(undef, n),      ## v
            Vector{T}(undef, n),      ## η
            Vector{T}(undef, n),      ## u
            Matrix{T}(undef, p, p),   ## V
            Matrix{T}(undef, 1, p))   ## μ 
end
