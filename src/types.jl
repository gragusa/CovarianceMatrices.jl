const WFLOAT = Sys.WORD_SIZE == 64 ? Float64 : Float32

#=========
Abstraction
==========#

abstract type AVarEstimator end

abstract type HAC{G} <: AVarEstimator end
abstract type VARHAC{G} <: AVarEstimator end

abstract type CrossSectionEstimator <: AVarEstimator end
abstract type HR <: AVarEstimator end
abstract type CR <: AVarEstimator end

#=========
HAC Types
=========#
abstract type BandwidthType end

struct NeweyWest <: BandwidthType end
struct Andrews <: BandwidthType end
struct Fixed <: BandwidthType end

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
struct TruncatedKernel{G<:BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    "When `wlock` is false, the kernelweights are allowed to updated by
    aVar, if true the kernelweights are locked."
    wlock::Vector{Bool}
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
struct BartlettKernel{G<:BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    wlock::Vector{Bool}
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
struct ParzenKernel{G<:BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    wlock::Vector{Bool}
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
struct TukeyHanningKernel{G<:BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    wlock::Vector{Bool}
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
struct QuadraticSpectralKernel{G<:BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    wlock::Vector{Bool}
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
const kernels = [:Bartlett,
                 :Parzen,
                 :QuadraticSpectral,
                 :Truncated,
                 :TukeyHanning]

for kerneltype ∈ kernels
    @eval ($kerneltype(x::Number)) = ($kerneltype){Fixed}(WFLOAT[x], WFLOAT[], [false])
    @eval ($kerneltype{Fixed}(x::Number)) = ($kerneltype){Fixed}(WFLOAT[x], WFLOAT[],
                                                                 [false])
end

for kerneltype ∈ kernels
    for opt ∈ [:Andrews, :NeweyWest]
        if !(opt == :NeweyWest && kerneltype in [:TukeyHanning, :Truncated])
            @eval ($kerneltype){$opt}() = ($kerneltype){$opt}(WFLOAT[0], WFLOAT[], [false])
        else
            msg = "$kerneltype does not support Newey-West optimal bandwidth"
            @eval ($kerneltype){$opt}() = throw(ArgumentError($msg))
        end
    end
end

function Base.show(io::IO, x::HAC{T}) where {T<:Union{Andrews,NeweyWest}}
    return print(typeof(x).name, "{", T, "}")
end
function Base.show(io::IO, x::HAC{T}) where {T<:Fixed}
    return print(typeof(x).name, "{", T, "}(", first(x.bw), ")")
end
## Makes the default bandwidth selection
#Optimal() = Optimal{Andrews}()

## Accessor
bandwidth(x::HAC) = x.bw
# kernelweights(x::HAC) = x.weights

#=========
EWC 
=========#
struct EWC <: AVarEstimator
    B::Int64
end

#=========
HC 
=========#
struct HR0 <: HR end
struct HR1 <: HR end
struct HR2 <: HR end
struct HR3 <: HR end
struct HR4 <: HR end
struct HR4m <: HR end
struct HR5 <: HR end

#=========
CR 
=========#
struct CR0{G} <: CR
    g::G
    CR0(g::G) where {G<:AbstractVector} = new{Tuple}(map(x -> GroupedArray(x), (g,)))
    CR0(g::G) where {G<:Tuple} = new{Tuple}(map(x -> GroupedArray(x), g))
end

struct CR1{G} <: CR
    g::G
    CR1(g::G) where {G<:AbstractVector} = new{Tuple}(map(x -> GroupedArray(x), (g,)))
    CR1(g::G) where {G<:Tuple} = new{Tuple}(map(x -> GroupedArray(x), g))
end

struct CR2{G} <: CR
    g::G
    CR2(g::G) where {G<:AbstractVector} = new{Tuple}(map(x -> GroupedArray(x), (g,)))
    CR2(g::G) where {G<:Tuple} = new{Tuple}(map(x -> GroupedArray(x), g))
end

struct CR3{G} <: CR
    g::G
    CR3(g::G) where {G<:AbstractVector} = new{Tuple}(map(x -> GroupedArray(x), (g,)))
    CR3(g::G) where {G<:Tuple} = new{Tuple}(map(x -> GroupedArray(x), g))
end

for k ∈ [:CR0, :CR1, :CR2, :CR3]
    @eval $(k)(args...) = $(k)(args)
end

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
mutable struct DriscollKraay{K,D} <: AVarEstimator
    K::K
    tis::D
    iis::D
end

function DriscollKraay(K::HAC; tis=nothing, iis=nothing)
    return DriscollKraay(K, GroupedArray(tis), GroupedArray(iis))
end
function DriscollKraay(K::HAC, tis::AbstractArray{T},
                       iis::AbstractArray{T}) where {T<:AbstractFloat}
    return DriscollKraay(K, GroupedArray(tis), GroupedArray(iis))
end
const HC0 = HR0
const HC1 = HR1
const HC2 = HR2
const HC3 = HR3
const HC4 = HR4
const HC4m = HR4m
const HC5 = HR5

Base.String(::Type{T}) where {T<:Truncated} = "Truncated"
Base.String(::Type{T}) where {T<:Parzen} = "Parzen"
Base.String(::Type{T}) where {T<:TukeyHanning} = "Tukey-Hanning"
Base.String(::Type{T}) where {T<:Bartlett} = "Bartlett"
Base.String(::Type{T}) where {T<:QuadraticSpectral} = "Quadratic Spectral"
Base.String(::Type{T}) where {T<:HR0} = "HR0"
Base.String(::Type{T}) where {T<:HR1} = "HR1"
Base.String(::Type{T}) where {T<:HR2} = "HR2"
Base.String(::Type{T}) where {T<:HR3} = "HR3"
Base.String(::Type{T}) where {T<:HR4} = "HR4"
Base.String(::Type{T}) where {T<:HR4m} = "HR4m"
Base.String(::Type{T}) where {T<:HR5} = "HR5"
Base.String(::Type{T}) where {T<:CR0} = "CR0"
Base.String(::Type{T}) where {T<:CR1} = "CR1"
Base.String(::Type{T}) where {T<:CR2} = "CR2"
Base.String(::Type{T}) where {T<:CR3} = "CR3"
Base.String(::Type{T}) where {T<:EWC} = "EWC"
Base.String(::Type{T}) where {T<:DriscollKraay} = "Driscoll-Kraay"
Base.String(::Type{T}) where {T<:VARHAC} = "VARHAC"
