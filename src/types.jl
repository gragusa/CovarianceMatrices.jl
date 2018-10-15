#=========
Abstraction
==========#

abstract type RobustVariance end
abstract type HAC{G} <: RobustVariance end
abstract type HC <: RobustVariance end
abstract type CRHC <: RobustVariance end


#=========
HAC Types
=========#

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

mutable struct CRHC0{V<:AbstractVector}  <: CRHC
    cl::V
end

mutable struct CRHC1{V<:AbstractVector}  <: CRHC
    cl::V
end

mutable struct CRHC2{V<:AbstractVector}  <: CRHC
    cl::V
end

mutable struct CRHC3{V<:AbstractVector}  <: CRHC
    cl::V
end
