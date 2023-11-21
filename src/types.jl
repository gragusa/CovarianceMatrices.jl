const WFLOAT = Sys.WORD_SIZE == 64 ? Float64 : Float32

#=========
Abstraction
==========#

abstract type AVarEstimator end

abstract type HAC{G} <: AVarEstimator end
abstract type VARHAC{G} <: AVarEstimator end

abstract type CrossSectionEstimator <: AVarEstimator end
abstract type HR <: AVarEstimator end
abstract type CR{G} <: AVarEstimator end

#=========
HAC Types
=========#
abstract type BandwidthType end

struct NeweyWest<:BandwidthType end
struct Andrews<:BandwidthType end
struct Fixed<:BandwidthType end

#struct Optimal{G<:OptimalBandwidth}<:BandwidthType{G where G<:OptimalBandwidth} end

# struct Prewhiten end
# struct Unwhiten end
"""
    `TruncatedKernel`

# Constructors
    Truncated(x::Int)
    Truncated{Andrews}()
    Truncated{NeweyWest}()
# Note
- `Fixed`: fixed bandwidth
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct TruncatedKernel{G<:BandwidthType}<:HAC{G}
  bw::Vector{WFLOAT}
  weights::Vector{WFLOAT}
  prewhiten::Base.RefValue{Bool} 
end

const Truncated = TruncatedKernel

"""
    `Bartlett`

# Constructors
    Bartlett(x::Int)
    Bartlett(::Type{Andrews})
    Bartlett(::Type{NeweyWest})
# Note
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct BartlettKernel{G<:BandwidthType}<:HAC{G}
    bw::Vector{WFLOAT}
    weights::Vector{WFLOAT}
    prewhiten::Base.RefValue{Bool}
end

const Bartlett = BartlettKernel

"""
    `Parzen`

# Constructors
    Parzen(x::Int)
    Parzen(::Type{Andrews})
    Parzen(::Type{NeweyWest})
# Note
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct ParzenKernel{G<:BandwidthType}<:HAC{G}
    bw::Vector{WFLOAT}
    weights::Vector{WFLOAT}
    prewhiten::Base.RefValue{Bool}
end

const Parzen = ParzenKernel
"""
    `TukeyHanning`

# Constructors
    TukeyHanning(x::Int)
    TukeyHanning(::Type{Andrews})
    TukeyHanning(::Type{NeweyWest})
# Note
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct TukeyHanningKernel{G<:BandwidthType}<:HAC{G}
    bw::Vector{WFLOAT}
    weights::Vector{WFLOAT}
    prewhiten::Base.RefValue{Bool}
end

const TukeyHanning = TukeyHanningKernel

"""
    `QuadraticSpectral`

# Constructors
    QuadraticSpectral(x::Int)
    QuadraticSpectral(::Type{Andrews})
    QuadraticSpectral(::Type{NeweyWest})
# Note
- `Andrews`: bandwidth selection a la Andrews
- `NeweyWest`: bandwidth selection a la Andrews
"""
struct QuadraticSpectralKernel{G<:BandwidthType}<:HAC{G}
    bw::Vector{WFLOAT}
    weights::Vector{WFLOAT}
    prewhiten::Base.RefValue{Bool}
end

const QuadraticSpectral = QuadraticSpectralKernel

# mutable struct VARHAC <: TimeSeries
#     maxlag::Int
#     lagstrategy::Int
#     selectionstrategy::Symbol
# end



## ------------------------------------------------------------------------------------------
# Define HAC constructor
# Three type of constructors
# - HAC(::Andrews) or HAC(::NeweyWest) Optimal bandwidth selection a la Andrews or a la NeweyWest
# - HAC(bw) Fixed bandwidth
# - HAC() -> Optimal bandwidth selection a la Andrews

# Constructors
const kernels = [
    :Bartlett,
    :Parzen,
    :QuadraticSpectral,
    :Truncated,
    :TukeyHanning,
]

for kerneltype in kernels
    @eval ($kerneltype(x::Number)) = ($kerneltype){Fixed}(WFLOAT[x], WFLOAT[], Base.RefValue(false))
    @eval ($kerneltype{Fixed}(x::Number)) = ($kerneltype){Fixed}(WFLOAT[x], WFLOAT[], Base.RefValue(false))
end

for kerneltype in kernels
    for opt in [:Andrews, :NeweyWest]
        if !(opt == :NeweyWest && kerneltype in [:TukeyHanning, :Truncated])
            @eval ($kerneltype){$opt}() = ($kerneltype){$opt}(WFLOAT[0], WFLOAT[], Base.RefValue(false))
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
EWC 
=========#

struct EWC <: AVarEstimator
  B::Int64
end

#=========
HC 
=========#
struct HR0  <: HR end
struct HR1  <: HR end
struct HR2  <: HR end
struct HR3  <: HR end
struct HR4  <: HR end
struct HR4m <: HR end
struct HR5  <: HR end

#=========
CR 
=========#


struct CR0{G}
    g::G
    CR0(g::G) where G <: AbstractVector = new{Tuple}((g,))
    CR0(g::G) where G <: Tuple = new{Tuple}(g)
  
end

struct CR1{G}
  g::G
  CR1(g::G) where G <: AbstractVector = new{Tuple}((g,))
  CR1(g::G) where G <: Tuple = new{Tuple}(g)

end

struct CR2{G}
  g::G
  CR2(g::G) where G <: AbstractVector = new{Tuple}((g,))
  CR2(g::G) where G <: Tuple = new{Tuple}(g)

end

struct CR3{G}
  g::G
  CR3(g::G) where G <: AbstractVector = new{Tuple}((g,))
  CR3(g::G) where G <: Tuple = new{Tuple}(g)
end

for k in [:CR0, :CR1, :CR2, :CR3]
  @eval $(k)(args...) = $(k)(args)
end



##mutable struct CR0{V, D}  <: CR{V,D}
##    cl::V
##    df::D
##end
##
##mutable struct CR1{V, D}  <: CR{V,D}
##    cl::V
##    df::D
##end
##
##mutable struct CR2{V, D}  <: CR{V,D}
##    cl::V
##    df::D
##end
##
##mutable struct CR3{V, D}  <: CR{V,D}
##    cl::V
##    df::D
##end
##
#### CRHC Constructors
##for tp in [:CR0, :CR1, :CR2, :CR3]
##    @eval $(tp)() = $(tp)(nothing, nothing)
##end
##
##for tp in [:CR0, :CR1, :CR2, :CR3]
##    @eval $(tp)(v::AbstractVector) = $(tp)((categorical(v),), nothing)
##end
##
##for tp in [:CR0, :CR1, :CR2, :CR3]
##  @eval $(tp)(v::AbstractVector, z::AbstractVector) = $(tp)((categorical(v),categorical(z)), nothing)
##end

#======== 
##DriscollKraay
=========#
"""
DriscollKraay

  t the time dimension indeces
  i the cross-section dimension indeces
  K the kernel type
  df the degrees of freedom

"""
mutable struct DriscollKraay{K, D} <: AVarEstimator
    t::CategoricalArray
    i::CategoricalArray
    K::K
    df::D
end

DriscollKraay(t::AbstractVector, i::AbstractVector, K::HAC) = DriscollKraay(categorical(t), categorical(i), K, nothing)

const HC0  = HR0 
const HC1  = HR1 
const HC2  = HR2 
const HC3  = HR3 
const HC4  = HR4 
const HC4m = HR4m
const HC5  = HR5

Base.String(::Type{T}) where T<:Truncated = "Truncated"
Base.String(::Type{T}) where T<:Parzen = "Parzen"
Base.String(::Type{T}) where T<:TukeyHanning = "Tukey-Hanning"
Base.String(::Type{T}) where T<:Bartlett = "Bartlett"
Base.String(::Type{T}) where T<:QuadraticSpectral = "Quadratic Spectral"
Base.String(::Type{T}) where T<:HR0 = "HR0"
Base.String(::Type{T}) where T<:HR1 = "HR1"
Base.String(::Type{T}) where T<:HR2 = "HR2"
Base.String(::Type{T}) where T<:HR3 = "HR3"
Base.String(::Type{T}) where T<:HR4 = "HR4"
Base.String(::Type{T}) where T<:HR4m = "HR4m"
Base.String(::Type{T}) where T<:HR5 = "HR5"
Base.String(::Type{T}) where T<:CR0 = "CR0"
Base.String(::Type{T}) where T<:CR1 = "CR1"
Base.String(::Type{T}) where T<:CR2 = "CR2"
Base.String(::Type{T}) where T<:CR3 = "CR3"
Base.String(::Type{T}) where T<:EWC = "EWC"
Base.String(::Type{T}) where T<:DriscollKraay = "Driscoll-Kraay"
Base.String(::Type{T}) where T<:VARHAC = "VARHAC"



# struct CovarianceMatrix{T2<:Factorization, T3<:CovarianceMatrices.RobustVariance, F1, T1<:AbstractMatrix{F1}} <: AbstractMatrix{F1}
#     F::T2       ## Factorization
#     K::T3       ## RobustVariance, e.g. HC0()
#     V::T1       ## Covariance matrix
# end
