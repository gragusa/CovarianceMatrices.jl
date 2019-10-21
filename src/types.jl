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
end

struct BartlettKernel{G<:BandwidthType, F}<:HAC{G}
    bwtype::G
    bw::Vector{F}
    weights::Vector{F}
end

struct ParzenKernel{G<:BandwidthType, F}<:HAC{G}
    bwtype::G
    bw::Vector{F}
    weights::Vector{F}
end

struct TukeyHanningKernel{G<:BandwidthType, F}<:HAC{G}
    bwtype::G
    bw::Vector{F}
    weights::Vector{F}
end

struct QuadraticSpectralKernel{G<:BandwidthType, F}<:HAC{G}
    bwtype::G
    bw::Vector{F}
    weights::Vector{F}
end

mutable struct VARHAC <: RobustVariance
    maxlag::Int
    lagstrategy::Int
    selectionstrategy::Symbol
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

#=========
HC Types
=========#

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
    V::T1       ## Covariance matrix
end
