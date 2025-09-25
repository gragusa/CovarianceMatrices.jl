const WFLOAT = Sys.WORD_SIZE == 64 ? Float64 : Float32

#=========
Abstraction
==========#

abstract type AVarEstimator end

abstract type HAC{G} <: AVarEstimator end

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

Implements the truncated (uniform) kernel for HAC covariance estimation.

# Mathematical Formula

The truncated kernel function is:
```math
k(x) = \\begin{cases}
1 & \\text{if } |x| \\leq 1 \\\\
0 & \\text{otherwise}
\\end{cases}
```

This provides equal weighting for all lags within the bandwidth and zero weight beyond.

# Properties
- Simple rectangular window
- Sharp cutoff at bandwidth boundary
- Provides consistent but not necessarily positive semi-definite estimates
- Computationally efficient

# Constructors

Truncated(x::Int)                # Fixed bandwidth
Truncated{Andrews}()            # Andrews bandwidth selection
Truncated(Andrews)              # Andrews bandwidth selection (alternative syntax)

# Bandwidth Selection

  - `Fixed`: fixed bandwidth
  - `Andrews`: bandwidth selection a la Andrews

**Note**: NeweyWest bandwidth selection is not supported for Truncated kernel.
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

Implements the Bartlett (triangular) kernel for HAC covariance estimation.

# Mathematical Formula

The Bartlett kernel function is:
```math
k(x) = \\begin{cases}
1 - |x| & \\text{if } |x| \\leq 1 \\\\
0 & \\text{otherwise}
\\end{cases}
```

This provides linearly declining weights that reach zero at the bandwidth boundary.

# Properties
- Triangular weighting scheme
- Guarantees positive semi-definite covariance matrices
- Most commonly used kernel in practice
- Good balance between bias and variance
- Equivalent to Newey-West estimator

# Constructors

Bartlett(x::Int)               # Fixed bandwidth
Bartlett{Andrews}()            # Andrews bandwidth selection
Bartlett{NeweyWest}()          # Newey-West bandwidth selection
Bartlett(Andrews)              # Andrews bandwidth selection (alternative syntax)
Bartlett(NeweyWest)            # Newey-West bandwidth selection (alternative syntax)

# Bandwidth Selection

  - `Andrews`: bandwidth selection a la Andrews
  - `NeweyWest`: bandwidth selection a la Newey-West
"""
struct BartlettKernel{G <: BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    wlock::Vector{Bool}
end

const Bartlett = BartlettKernel

"""
`Parzen`

Implements the Parzen kernel for HAC covariance estimation.

# Mathematical Formula

The Parzen kernel function is:
```math
k(x) = \\begin{cases}
1 - 6x^2 + 6|x|^3 & \\text{if } |x| \\leq 1/2 \\\\
2(1-|x|)^3 & \\text{if } 1/2 < |x| \\leq 1 \\\\
0 & \\text{otherwise}
\\end{cases}
```

This provides a smooth, continuous kernel that is more efficient than the Bartlett kernel.

# Properties
- Smooth cubic spline kernel
- Guarantees positive semi-definite covariance matrices
- More efficient than Bartlett kernel (better rate of convergence)
- Continuous first derivative
- Good for data with strong persistence

# Constructors

Parzen(x::Int)                 # Fixed bandwidth
Parzen{Andrews}()              # Andrews bandwidth selection
Parzen{NeweyWest}()            # Newey-West bandwidth selection
Parzen(Andrews)                # Andrews bandwidth selection (alternative syntax)
Parzen(NeweyWest)              # Newey-West bandwidth selection (alternative syntax)

# Bandwidth Selection

  - `Andrews`: bandwidth selection a la Andrews
  - `NeweyWest`: bandwidth selection a la Newey-West
"""
struct ParzenKernel{G <: BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    wlock::Vector{Bool}
end

const Parzen = ParzenKernel
"""
`TukeyHanning`

Implements the Tukey-Hanning kernel for HAC covariance estimation.

# Mathematical Formula

The Tukey-Hanning kernel function is:
```math
k(x) = \\begin{cases}
\\frac{1}{2}\\left(1 + \\cos(\\pi x)\\right) & \\text{if } |x| \\leq 1 \\\\
0 & \\text{otherwise}
\\end{cases}
```

This provides a smooth, bell-shaped weighting scheme based on the cosine function.

# Properties
- Smooth cosine-based kernel
- Guarantees positive semi-definite covariance matrices
- Similar efficiency to Parzen kernel
- Symmetric bell-shaped weights
- Good spectral properties

# Constructors

TukeyHanning(x::Int)            # Fixed bandwidth
TukeyHanning{Andrews}()         # Andrews bandwidth selection
TukeyHanning(Andrews)           # Andrews bandwidth selection (alternative syntax)

# Bandwidth Selection

  - `Fixed`: fixed bandwidth
  - `Andrews`: bandwidth selection a la Andrews

**Note**: NeweyWest bandwidth selection is not supported for TukeyHanning kernel.
"""
struct TukeyHanningKernel{G <: BandwidthType} <: HAC{G}
    bw::Vector{WFLOAT}
    kw::Vector{WFLOAT}
    wlock::Vector{Bool}
end

const TukeyHanning = TukeyHanningKernel

"""
`QuadraticSpectral`

Implements the Quadratic Spectral kernel for HAC covariance estimation.

# Mathematical Formula

The Quadratic Spectral kernel function is:
```math
k(x) = \\frac{25}{12\\pi^2 x^2}\\left[\\frac{\\sin(6\\pi x/5)}{6\\pi x/5} - \\cos(6\\pi x/5)\\right]
```

For x = 0, the kernel value is k(0) = 1 by continuity.

# Properties
- Spectral kernel with infinite support
- Theoretically optimal among kernels with infinite support
- Guarantees positive semi-definite covariance matrices
- Most efficient kernel (optimal rate of convergence)
- Data-driven bandwidth selection possible
- Can handle very persistent data well

# Constructors

QuadraticSpectral(x::Int)      # Fixed bandwidth
QuadraticSpectral{Andrews}()   # Andrews bandwidth selection
QuadraticSpectral{NeweyWest}() # Newey-West bandwidth selection
QuadraticSpectral(Andrews)     # Andrews bandwidth selection (alternative syntax)
QuadraticSpectral(NeweyWest)   # Newey-West bandwidth selection (alternative syntax)

# Bandwidth Selection

  - `Andrews`: bandwidth selection a la Andrews
  - `NeweyWest`: bandwidth selection a la Newey-West
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
    @eval ($kerneltype{Fixed}(x::Number)) = ($kerneltype){Fixed}(
        WFLOAT[x], WFLOAT[], [false])
end

for kerneltype in kernels
    for opt in [:Andrews, :NeweyWest]
        if !(opt == :NeweyWest && kerneltype in [:TukeyHanning, :Truncated])
            @eval ($kerneltype){$opt}() = ($kerneltype){$opt}(WFLOAT[0], WFLOAT[], [false])
            # Add constructor that takes bandwidth type as argument: Bartlett(NeweyWest())
            @eval ($kerneltype(::Type{$opt})) = ($kerneltype){$opt}()
        else
            msg = "$kerneltype does not support Newey-West optimal bandwidth"
            @eval ($kerneltype){$opt}() = throw(ArgumentError($msg))
            @eval ($kerneltype(::Type{$opt})) = throw(ArgumentError($msg))
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
"""
    EWC <: AVarEstimator

Equal Weighted Cosine (EWC) covariance matrix estimator.

The EWC estimator provides a non-parametric approach to robust covariance estimation
by using cosine similarity weighting. This method is particularly useful for
financial time series and other applications where traditional HAC estimators
may be sensitive to the choice of kernel or bandwidth.

# Constructor
    EWC(B::Integer)

# Arguments
- `B::Integer`: Number of basis functions (must be positive)

# Mathematical Foundation
The EWC estimator computes the covariance matrix using:
```
Ω̂ = (1/T) Σ_{t=1}^T Σ_{j=1}^B w_j(t) g_t g_t'
```
where w_j(t) are cosine-based weights and g_t are the moment conditions.


# Examples
```julia
# EWC with 10 basis functions
ve = EWC(10)

# Compute covariance matrix
Ω = aVar(ve, moment_matrix)
```
"""
struct EWC <: AVarEstimator
    B::Int
    function EWC(B::Integer)
        B > 0 || throw(ArgumentError("B must be positive"))
        return new(B)
    end
end

function EWC(B::Real)
    B == round(B) ? EWC(round(Int, B)) :
    throw(ArgumentError("B must be an integer or parsable as integer"))
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
    max_theoretical = max(1, floor(Int, T^(1 / 3)))
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

"""
    VARHAC{S<:LagSelector, L<:LagStrategy, T<:Real} <: AVarEstimator

Vector Autoregression HAC (VARHAC) estimator for robust covariance estimation.

The VARHAC estimator provides a data-driven approach to HAC estimation by
fitting a Vector Autoregression (VAR) model to the moment conditions and using
the spectral density at frequency zero as the covariance estimator. This approach
automatically accounts for serial correlation and cross-correlation patterns
without requiring bandwidth selection.

# Constructor
    VARHAC(selector=AICSelector(), strategy=SameLags(8); T::Type{<:Real}=Float64)

# Arguments
- `selector::LagSelector`: Method for selecting optimal lag length
  - `AICSelector()`: Use Akaike Information Criterion (default)
  - `BICSelector()`: Use Bayesian Information Criterion
  - `FixedSelector()`: Use fixed lag length (automatically set with FixedLags)
- `strategy::LagStrategy`: Strategy for lag specification
  - `SameLags(k)`: Same lag length k for all variables (default: k=8)
  - `FixedLags(k)`: Fixed lag length k (forces FixedSelector)
  - `AutoLags()`: Automatic lag selection based on sample size
  - `DifferentOwnLags([k1,k2])`: Different own lag lengths for bivariate case

# Convenience Constructors
- `VARHAC(8)`: Same as `VARHAC(AICSelector(), SameLags(8))`
- `VARHAC(:aic)`: Same as `VARHAC(AICSelector(), SameLags(8))`
- `VARHAC(:bic)`: Same as `VARHAC(BICSelector(), SameLags(8))`
- `VARHAC(FixedLags(5))`: Fixed lag length of 5

# References
- den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust
  Covariance Matrix Estimation". Handbook of Statistics, Vol. 15.

# Examples
```julia
# Basic usage with AIC selection
ve1 = VARHAC()

# BIC selection with specific max lags
ve2 = VARHAC(:bic, SameLags(12))

# Fixed lag length
ve3 = VARHAC(FixedLags(6))

# Automatic lag selection based on sample size
ve4 = VARHAC(AICSelector(), AutoLags())
```
"""
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
    return VARHAC{typeof(selector), typeof(strategy), T}(
        nothing, nothing, nothing, nothing, selector, strategy)
end

# Convenient constructors for common usage patterns
VARHAC(f::FixedLags; T::Type{<:Real} = Float64) = VARHAC(FixedSelector(), f; T = T)

# Quick selector construction: VARHAC(:aic) or VARHAC(:bic)
function VARHAC(selector_symbol::Symbol; T::Type{<:Real} = Float64)
    VARHAC(_symbol_to_selector(selector_symbol), SameLags(8); T = T)
end

# Quick max lags construction: VARHAC(12)
function VARHAC(max_lags::Integer; T::Type{<:Real} = Float64)
    VARHAC(AICSelector(), SameLags(max_lags); T = T)
end

# Auto-selection constructor: VARHAC(:auto)
VARHAC(::Val{:auto}; T::Type{<:Real} = Float64) = VARHAC(AICSelector(), AutoLags(); T = T)

function _symbol_to_strategy(s::Symbol)
    if s === :auto
        return AutoLags()
    else
        throw(
            ArgumentError(
            "Invalid strategy symbol: $s. Use :auto for automatic lag selection",
        ),
        )
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
function maxlags(k::VARHAC{S, AutoLags, T}) where {S <: LagSelector, T}
    error("AutoLags requires data dimensions. Use maxlags(estimator, T, N) where T is sample size and N is number of variables.")
end

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
    DriscollKraay{K, D} <: AVarEstimator

Driscoll-Kraay estimator for panel data with cross-sectional and temporal dependence.

The Driscoll-Kraay estimator extends HAC estimation to panel data settings,
accounting for both serial correlation within panels and spatial correlation
across panels. This estimator is particularly useful for macro panels, regional
data, and other applications where both dimensions of dependence are relevant.

# Constructor
    DriscollKraay(K::HAC; tis=nothing, iis=nothing)
    DriscollKraay(K::HAC, tis::AbstractArray, iis::AbstractArray)

# Arguments
- `K::HAC`: HAC kernel for temporal dependence (Bartlett, Parzen, etc.)
- `tis`: Time dimension indices (panel identifier for time)
- `iis`: Cross-section dimension indices (panel identifier for units)

# Mathematical Foundation
The Driscoll-Kraay estimator computes:
```
Ω̂ = Σ_{h=-H}^H Σ_{g=-G}^G K₁(h/H₁) K₂(g/G₁) Σ̂(h,g)
```
where:
- K₁ is the temporal kernel (with bandwidth H₁)
- K₂ is the spatial kernel (with bandwidth G₁)
- Σ̂(h,g) are the space-time autocovariances

# Properties
- Handles both serial correlation (within panels) and spatial correlation (across panels)
- Consistent for large T and/or large N
- Automatic positive semi-definiteness (when using appropriate kernels)
- Particularly suited for macro panels and regional data

# Applications
- Macro panels with countries/regions and time
- Firm-level data with industry clustering
- Regional economic data
- Social networks with temporal evolution

# References
- Driscoll, J.C. and Kraay, A.C. (1998). "Consistent Covariance Matrix Estimation
  with Spatially Dependent Panel Data". Review of Economics and Statistics, 80(4), 549-560.
- Hoechle, D. (2007). "Robust Standard Errors for Panel Regressions with
  Cross-Sectional Dependence". Stata Journal, 7(3), 281-312.

# Examples
```julia
# Basic panel setup
ve = DriscollKraay(Bartlett{Andrews}(), tis=time_ids, iis=unit_ids)

# With fixed bandwidth
ve = DriscollKraay(Parzen(4), tis=years, iis=countries)
```
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
