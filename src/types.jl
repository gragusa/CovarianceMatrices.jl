const WFLOAT = Sys.WORD_SIZE == 64 ? Float64 : Float32

#=========
Abstraction
==========#
abstract type RobustVariance <: CovarianceEstimator end
abstract type HAC{G} <: RobustVariance end
abstract type HC <: RobustVariance end
abstract type CRHC{V,D} <: RobustVariance end

#=========
HAC Types
=========#


abstract type BandwidthType end
#abstract type OptimalBandwidth end

struct NeweyWest<:BandwidthType end
struct Andrews<:BandwidthType end
struct Fixed<:BandwidthType end

#struct Optimal{G<:OptimalBandwidth}<:BandwidthType{G where G<:OptimalBandwidth} end

# struct Prewhiten end
# struct Unwhiten end
"""
    `TruncatedKernel`

# Constructors
    TruncatedKernel{Fixed}(x::Int)
    TruncatedKernel{Andrews}()
    TruncatedKernel{NeweyWest}()
# Note
- `Fixed`: fixed bandwidth
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct TruncatedKernel{G<:BandwidthType}<:HAC{G}
  bw::Vector{WFLOAT}
  weights::Vector{WFLOAT}
end

"""
    `BartlettKernel`

# Constructors
    BartlettKernel(x::Int)
    BartlettKernel(::Type{Andrews})
    BartlettKernel(::Type{NeweyWest})
# Note
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct BartlettKernel{G<:BandwidthType}<:HAC{G}
    bw::Vector{WFLOAT}
    weights::Vector{WFLOAT}
end

"""
    `ParzenKernel`

# Constructors
    ParzenKernel(x::Int)
    ParzenKernel(::Type{Andrews})
    ParzenKernel(::Type{NeweyWest})
# Note
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct ParzenKernel{G<:BandwidthType}<:HAC{G}
    bw::Vector{WFLOAT}
    weights::Vector{WFLOAT}
end

"""
    `TukeyHanningKernel`

# Constructors
    TukeyHanningKernel(x::Int)
    TukeyHanningKernel(::Type{Andrews})
    TukeyHanningKernel(::Type{NeweyWest})
# Note
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct TukeyHanningKernel{G<:BandwidthType}<:HAC{G}
    bw::Vector{WFLOAT}
    weights::Vector{WFLOAT}
end

"""
    `QuadraticSpectralKernel`

# Constructors
    QuadraticSpectralKernel(x::Int)
    QuadraticSpectralKernel(::Type{Andrews})
    QuadraticSpectralKernel(::Type{NeweyWest})
# Note
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct QuadraticSpectralKernel{G<:BandwidthType}<:HAC{G}
    bw::Vector{WFLOAT}
    weights::Vector{WFLOAT}
end

mutable struct VARHAC <: RobustVariance
    maxlag::Int
    lagstrategy::Int
    selectionstrategy::Symbol
end

## ------------------------------------------------------------------------------------------
# Define HAC constructor
# Three type of constructors
# - HAC(::Andrews) or HAC(::NeweyWest) Optimal bandwidth selection a la Andrews or a la NeweyWest
# - HAC(bw) Fixed bandwidth
# - HAC() -> Optimal bandwidth selection a la Andrews

# Constructors
const kernels = [
    :BartlettKernel,
    :ParzenKernel,
    :QuadraticSpectralKernel,
    :TruncatedKernel,
    :TukeyHanningKernel,
]

for kerneltype in kernels
    @eval ($kerneltype(x::Number)) = ($kerneltype){Fixed}(WFLOAT[x], WFLOAT[])
    @eval ($kerneltype{Fixed}(x::Number)) = ($kerneltype){Fixed}(WFLOAT[x], WFLOAT[])
end

for kerneltype in kernels
    for opt in [:Andrews, :NeweyWest]
        if !(opt == :NeweyWest && kerneltype in [:TukeyHanningKernel, :TruncatedKernel])
            @eval ($kerneltype){$opt}() = ($kerneltype){$opt}(WFLOAT[0], WFLOAT[])
        else
            msg = "$kerneltype does not support Newey-West optimal bandwidth"
            @eval ($kerneltype){$opt}() = throw(ArgumentError($msg))
        end
    end
end

function Base.show(io::IO, x::HAC{T}) where T<:Union{Andrews, NeweyWest}
    print(typeof(x).name, "{", T,"}")
end
function Base.show(io::IO, x::HAC{T}) where T<:Fixed
    print(typeof(x).name, "{", T,"}(", first(x.bw), ")")
end
## Makes the default bandwidth selection
#Optimal() = Optimal{Andrews}()

## Accessor
bandwidth(x::HAC) = x.bw
kernelweights(x::HAC) = x.weights

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

mutable struct CRHC0{V, D}  <: CRHC{V,D}
    cl::V
    df::D
end

mutable struct CRHC1{V, D}  <: CRHC{V,D}
    cl::V
    df::D
end

mutable struct CRHC2{V, D}  <: CRHC{V,D}
    cl::V
    df::D
end

mutable struct CRHC3{V, D}  <: CRHC{V,D}
    cl::V
    df::D
end

## CRHC accessor
clusterindicator(x::CRHC) = x.cl


## CRHC Constructors
for tp in [:CRHC0, :CRHC1, :CRHC2, :CRHC3]
    @eval $(tp)() = $(tp)(nothing, nothing)
end

for tp in [:CRHC0, :CRHC1, :CRHC2, :CRHC3]
    @eval $(tp)(v::AbstractVector) = $(tp)(v, nothing)
end


struct CovarianceMatrix{T2<:Factorization, T3<:CovarianceMatrices.RobustVariance, F1, T1<:AbstractMatrix{F1}} <: AbstractMatrix{F1}
    F::T2       ## Factorization
    K::T3       ## RobustVariance, e.g. HC0()
    V::T1       ## Covariance matrix
end
