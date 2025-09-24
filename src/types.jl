const WFLOAT = Sys.WORD_SIZE == 64 ? Float64 : Float32

#=========
Abstraction
==========#

abstract type AVarEstimator end

abstract type HAC{G} <: AVarEstimator end

# struct VARHAC{T<:Int} <: AVarEstimator
#     maxlag::T
#     ilag::T
#     lagstrategy::T
# end

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
struct TruncatedKernel{G <: BandwidthType} <: HAC{G}
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
struct BartlettKernel{G <: BandwidthType} <: HAC{G}
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
struct ParzenKernel{G <: BandwidthType} <: HAC{G}
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
struct TukeyHanningKernel{G <: BandwidthType} <: HAC{G}
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
struct QuadraticSpectralKernel{G <: BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    wlock::Vector{Bool}
end

const QuadraticSpectral = QuadraticSpectralKernel

## ------------------------------------------------------------------------------------------
# Define HAC constructor
# Three type of constructors
# - HAC(::Andrews) or HAC(::NeweyWest) Optimal bandwidth selection a la Andrews or a la NeweyWest
# - HAC(bw) Fixed bandwidth
# - HAC() -> Optimal bandwidth selection a la Andrews
const kernels = [:Bartlett, :Parzen, :QuadraticSpectral, :Truncated, :TukeyHanning]

for kerneltype in kernels
    @eval ($kerneltype(x::Number)) = ($kerneltype){Fixed}(WFLOAT[x], WFLOAT[], [false])
    @eval ($kerneltype{Fixed}(x::Number)) = ($kerneltype){Fixed}(WFLOAT[x], WFLOAT[], [false])
end

for kerneltype in kernels
    for opt in [:Andrews, :NeweyWest]
        if !(opt == :NeweyWest && kerneltype in [:TukeyHanning, :Truncated])
            @eval ($kerneltype){$opt}() = ($kerneltype){$opt}(WFLOAT[0], WFLOAT[], [false])
        else
            msg = "$kerneltype does not support Newey-West optimal bandwidth"
            @eval ($kerneltype){$opt}() = throw(ArgumentError($msg))
        end
    end
end

function Base.show(io::IO, x::HAC{T}) where {T <: Union{Andrews, NeweyWest}}
    return print(typeof(x).name, "{", T, "}")
end
function Base.show(io::IO, x::HAC{T}) where {T <: Fixed}
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
    CR0(g::G) where {G <: AbstractVector} = new{Tuple}(map(x -> GroupedArray(x), (g,)))
    CR0(g::G) where {G <: Tuple} = new{Tuple}(map(x -> GroupedArray(x), g))
end

struct CR1{G} <: CR
    g::G
    CR1(g::G) where {G <: AbstractVector} = new{Tuple}(map(x -> GroupedArray(x), (g,)))
    CR1(g::G) where {G <: Tuple} = new{Tuple}(map(x -> GroupedArray(x), g))
end

struct CR2{G} <: CR
    g::G
    CR2(g::G) where {G <: AbstractVector} = new{Tuple}(map(x -> GroupedArray(x), (g,)))
    CR2(g::G) where {G <: Tuple} = new{Tuple}(map(x -> GroupedArray(x), g))
end

struct CR3{G} <: CR
    g::G
    CR3(g::G) where {G <: AbstractVector} = new{Tuple}(map(x -> GroupedArray(x), (g,)))
    CR3(g::G) where {G <: Tuple} = new{Tuple}(map(x -> GroupedArray(x), g))
end

for k in [:CR0, :CR1, :CR2, :CR3]
    @eval $(k)(args...) = $(k)(args)
end

#=========
VARHAC
=========#
abstract type LagSelector end
struct AICSelector <: LagSelector end
struct BICSelector <: LagSelector end
struct FixedSelector <: LagSelector end
abstract type LagStrategy end

struct FixedLags <: LagStrategy
    maxlag::Int
end

FixedLags(x::Real) = FixedLags(round(Int, x))
FixedLags() = FixedLags(5)

struct SameLags <: LagStrategy
    maxlag::Int
end

SameLags(x::Real) = SameLags(round(Int, x))
SameLags() = SameLags(8)  # Better default based on practical experience

# Auto-selection strategy for K_max based on sample size
struct AutoLags <: LagStrategy end

# Function to compute automatic lag selection based on T^(1/3) rule
function compute_auto_maxlag(T::Int, N::Int)
    # Rule from literature: K_max should grow no faster than T^(1/3)
    # Also ensure it doesn't exceed (T-1)/N to avoid overfitting
    max_theoretical = max(1, floor(Int, T^(1/3)))
    max_practical = max(1, floor(Int, (T - 1) / N))
    return min(max_theoretical, max_practical, 20)  # Cap at reasonable maximum
end

struct DifferentOwnLags <: LagStrategy
    maxlags::Vector{Int}
end

DifferentOwnLags(x::Tuple{Int, Int}) = DifferentOwnLags([x[1], x[2]])
function DifferentOwnLags(x::Tuple{A, A}) where {A <: Real}
    DifferentOwnLags([round(Int, x[1]), round(Int, x[2])])
end
DifferentOwnLags() = DifferentOwnLags([5, 5])

mutable struct VARHAC{S <: LagSelector, L <: LagStrategy, T <: Real} <: AVarEstimator
    AICs::Union{Array{T}, Nothing}
    BICs::Union{Array{T}, Nothing}
    order_aic::Union{Vector{Int}, Nothing}
    order_bic::Union{Vector{Int}, Nothing}
    selector::S
    strategy::L
end

function VARHAC(selector = AICSelector(), strategy = SameLags(8); T::Type{<:Real} = Float64)
    isa(strategy, FixedLags) && (selector = FixedSelector())
    return VARHAC{typeof(selector), typeof(strategy), T}(nothing, nothing, nothing, nothing, selector, strategy)
end

# Convenient constructors for common usage patterns
VARHAC(f::FixedLags; T::Type{<:Real} = Float64) = VARHAC(FixedSelector(), f; T = T)

# Quick selector construction: VARHAC(:aic) or VARHAC(:bic)
VARHAC(selector_symbol::Symbol; T::Type{<:Real} = Float64) = VARHAC(_symbol_to_selector(selector_symbol), SameLags(8); T = T)

# Quick max lags construction: VARHAC(12)
VARHAC(max_lags::Integer; T::Type{<:Real} = Float64) = VARHAC(AICSelector(), SameLags(max_lags); T = T)

# Auto-selection constructor: VARHAC(:auto)
VARHAC(::Val{:auto}; T::Type{<:Real} = Float64) = VARHAC(AICSelector(), AutoLags(); T = T)

function _symbol_to_strategy(s::Symbol)
    if s === :auto
        return AutoLags()
    else
        throw(ArgumentError("Invalid strategy symbol: $s. Use :auto for automatic lag selection"))
    end
end

# Helper function to convert symbols to selectors
function _symbol_to_selector(s::Symbol)
    if s === :aic
        return AICSelector()
    elseif s === :bic
        return BICSelector()
    elseif s === :fixed
        return FixedSelector()
    else
        throw(ArgumentError("Invalid selector symbol: $s. Use :aic, :bic, or :fixed"))
    end
end

maxlags(k::VARHAC{S, L, T}) where {S <: LagSelector, L <: SameLags, T} = k.strategy.maxlag
function maxlags(k::VARHAC{S, L, T}) where {S <: LagSelector, L <: DifferentOwnLags, T}
    k.strategy.maxlags
end
maxlags(k::VARHAC{S, L, T}) where {S <: LagSelector, L <: FixedLags, T} = k.strategy.maxlag

# For AutoLags, we need the data dimensions to compute optimal lag length
function maxlags(k::VARHAC{S, AutoLags, T}, T_data::Int, N::Int) where {S <: LagSelector, T}
    return compute_auto_maxlag(T_data, N)
end
# Fallback that throws informative error if AutoLags used without dimensions
maxlags(k::VARHAC{S, AutoLags, T}) where {S <: LagSelector, T} =
    error("AutoLags requires data dimensions. Use maxlags(estimator, T, N) where T is sample size and N is number of variables.")

AICs(k::VARHAC) = k.AICs
BICs(k::VARHAC) = k.BICs
order_aic(k::VARHAC) = k.order_aic
order_bic(k::VARHAC) = k.order_bic
order(k::VARHAC{AICSelector, S}) where {S} = order_aic(k)
order(k::VARHAC{BICSelector, S}) where {S} = order_bic(k)

## Show method for VARHAC
function Base.show(io::IO, k::VARHAC)
    println(io, "VARHAC{", k.selector, ", ", k.strategy, "}")
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
mutable struct DriscollKraay{K, D} <: AVarEstimator
    K::K
    tis::D
    iis::D
end

function DriscollKraay(K::HAC; tis = nothing, iis = nothing)
    return DriscollKraay(K, GroupedArray(tis), GroupedArray(iis))
end
function DriscollKraay(
        K::HAC,
        tis::AbstractArray{T},
        iis::AbstractArray{T}
) where {T <: AbstractFloat}
    return DriscollKraay(K, GroupedArray(tis), GroupedArray(iis))
end
const HC0 = HR0
const HC1 = HR1
const HC2 = HR2
const HC3 = HR3
const HC4 = HR4
const HC4m = HR4m
const HC5 = HR5

Base.String(::Type{T}) where {T <: Truncated} = "Truncated"
Base.String(::Type{T}) where {T <: Parzen} = "Parzen"
Base.String(::Type{T}) where {T <: TukeyHanning} = "Tukey-Hanning"
Base.String(::Type{T}) where {T <: Bartlett} = "Bartlett"
Base.String(::Type{T}) where {T <: QuadraticSpectral} = "Quadratic Spectral"
Base.String(::Type{T}) where {T <: HR0} = "HR0"
Base.String(::Type{T}) where {T <: HR1} = "HR1"
Base.String(::Type{T}) where {T <: HR2} = "HR2"
Base.String(::Type{T}) where {T <: HR3} = "HR3"
Base.String(::Type{T}) where {T <: HR4} = "HR4"
Base.String(::Type{T}) where {T <: HR4m} = "HR4m"
Base.String(::Type{T}) where {T <: HR5} = "HR5"
Base.String(::Type{T}) where {T <: CR0} = "CR0"
Base.String(::Type{T}) where {T <: CR1} = "CR1"
Base.String(::Type{T}) where {T <: CR2} = "CR2"
Base.String(::Type{T}) where {T <: CR3} = "CR3"
Base.String(::Type{T}) where {T <: EWC} = "EWC"
Base.String(::Type{T}) where {T <: DriscollKraay} = "Driscoll-Kraay"
Base.String(::Type{T}) where {T <: VARHAC} = "VARHAC"
