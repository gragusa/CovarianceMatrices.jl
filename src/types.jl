const WFLOAT = Sys.WORD_SIZE == 64 ? Float64 : Float32

#=========
Abstraction
==========#

"""
`AbstractAsymptoticVarianceEstimator`

Abstract supertype for all asymptotic variance (covariance) estimators.

This is the root type of the estimator hierarchy in CovarianceMatrices.jl. All robust covariance estimators inherit from this type.

# Type Hierarchy

```
AbstractAsymptoticVarianceEstimator (abstract)
  * Uncorrelated
  * HR (abstract): HR0/HC0, HR1/HC1, HR2/HC2, HR3/HC3, HR4/HC4, HR4m/HC4m, HR5/HC5
  * Correlated (abstract)
      - HAC{G} (abstract): Bartlett, Parzen, QuadraticSpectral, TukeyHanning, Truncated
      - CR (abstract): CR0, CR1, CR2, CR3, Cluster, CachedCR
      - EWC
      - VARHAC
      - DriscollKraay
  * MomentSmoother (abstract): UniformSmoother, TriangularSmoother
```

# Interface

All subtypes implement the core interface:
- `aVar(estimator, X)` - compute asymptotic variance from data matrix
- `vcov(estimator, model)` - compute variance-covariance matrix from fitted model
- `stderror(estimator, model)` - compute standard errors from fitted model
"""
abstract type AbstractAsymptoticVarianceEstimator end

"""
`Uncorrelated`

Estimator for uncorrelated moment conditions (iid errors).

Use this estimator when the moment conditions are uncorrelated across observations,
such as in correctly specified MLE or GMM models with iid errors. This provides
the appropriate covariance matrix without unnecessary robust corrections.

# Usage
```julia
using CovarianceMatrices
# For MLE or GMM with uncorrelated moments
se_uncorr = stderror(Uncorrelated(), model)
vcov_uncorr = vcov(Uncorrelated(), model)
```

# When to use
- MLikeModel or GMMLikeModel with iid errors
- Correctly specified models without heteroskedasticity or serial correlation
- When you want the efficient (non-robust) covariance estimator
"""
struct Uncorrelated <: AbstractAsymptoticVarianceEstimator end

"""
`Correlated`

Abstract type for estimators that account for various forms of correlation.

This encompasses all estimators that handle:
- Serial correlation (HAC estimators)
- Cluster correlation (CR estimators)
- Spatial and temporal correlation (DriscollKraay)
- Other specialized correlation patterns (EWC, VARHAC)

The smoothed-moments estimators ([`MomentSmoother`](@ref)) also handle serial
correlation but are not subtypes of `Correlated`.
"""
abstract type Correlated <: AbstractAsymptoticVarianceEstimator end

abstract type HAC{G} <: Correlated end

abstract type HR <: AbstractAsymptoticVarianceEstimator end
"""
    CR

Abstract supertype for the cluster-robust estimators [`CR0`](@ref), [`CR1`](@ref),
[`CR2`](@ref), [`CR3`](@ref), [`Cluster`](@ref) and [`CachedCR`](@ref).
"""
abstract type CR <: Correlated end

"""
`Cluster`

Cluster-robust estimator without finite-sample correction. `Cluster(g)` gives the
same result as [`CR0`](@ref)`(g)` on a data or moment matrix and on a fitted model.

Given a `T × k` data or moment matrix and a vector of group identifiers,
`aVar(Cluster(groups), X)` returns the cluster long-run variance

```math
\\hat{\\Omega} = \\sum_{g=1}^{G} \\hat{u}_g \\hat{u}_g', \\qquad \\hat{u}_g = \\sum_{t \\in g} \\tilde{X}_t,
```

treating observations as correlated within groups and independent across groups.
`Cluster` is the clustering counterpart of [`Uncorrelated`](@ref), which equals `HC0`.
For degrees-of-freedom or leverage corrections on a fitted model use [`CR1`](@ref),
[`CR2`](@ref) or [`CR3`](@ref).

# Usage
```julia
using CovarianceMatrices
clusters = [1, 1, 2, 2, 3, 3, 4, 4]
Ω = aVar(Cluster(clusters), X)

# Multi-way clustering
Ω = aVar(Cluster((firm_ids, year_ids)), X)
```
"""
struct Cluster{G} <: CR
    g::G
    function Cluster(g::G) where {G <: AbstractVector}
        t = (Clustering(g),)
        new{typeof(t)}(t)
    end
    Cluster(g::G) where {G <: Tuple{Vararg{Symbol}}} = new{G}(g)
    function Cluster(g::G) where {G <: Tuple}
        t = map(Clustering, g)
        new{typeof(t)}(t)
    end
end

#=========
HAC Types
=========#
abstract type BandwidthType end

"""
`NeweyWest`

Newey-West automatic bandwidth selection for HAC estimators.

# Mathematical Formula

The Newey-West bandwidth selection rule chooses:
```math
T^{1/4} \\left(\\frac{4\\hat{\\rho}^2}{(1-\\hat{\\rho})^4}\\right)^{1/4}
```

where ``\\hat{\\rho}`` is an estimate of the first-order autocorrelation.

# Properties
- Automatic bandwidth selection based on data-dependent rule
- Designed specifically for the Bartlett kernel
- Generally produces smaller bandwidths than Andrews selection
- Widely used in econometric applications

# Usage
```julia
using CovarianceMatrices
# Bartlett kernel with Newey-West bandwidth
se_nw = stderror(Bartlett{NeweyWest}(), model)
se_nw_alt = stderror(Bartlett(NeweyWest), model)  # Alternative syntax
```

**Note**: NeweyWest bandwidth selection is primarily designed for Bartlett kernels.
"""
struct NeweyWest <: BandwidthType end

"""
`Andrews`

Andrews automatic bandwidth selection for HAC estimators.

# Mathematical Formula

The Andrews bandwidth selection rule chooses:
```math
T^{1/3} \\left(\\frac{2\\hat{\\sigma}^4}{\\hat{\\lambda}^2}\\right)^{1/3}
```

where ``\\hat{\\sigma}^4`` and ``\\hat{\\lambda}^2`` are estimated from an AR(1) approximation to the data.

# Properties
- Automatic bandwidth selection based on parametric approximation
- Works with all kernel types (Bartlett, Parzen, QuadraticSpectral, etc.)
- Generally produces larger bandwidths than Newey-West
- Asymptotically optimal rate

# Usage
```julia
using CovarianceMatrices
# Various kernels with Andrews bandwidth
se_bartlett = stderror(Bartlett{Andrews}(), model)
se_parzen = stderror(Parzen{Andrews}(), model)
se_qs = stderror(QuadraticSpectral{Andrews}(), model)

# Alternative syntax
se_bartlett_alt = stderror(Bartlett(Andrews), model)
se_parzen_alt = stderror(Parzen(Andrews), model)
```

**Note**: Andrews bandwidth selection is compatible with all HAC kernels.
"""
struct Andrews <: BandwidthType end

"""
`Fixed`

Fixed (user-supplied) bandwidth for HAC estimators.

    Fixed(bw::Real)

A HAC kernel with a fixed bandwidth `bw` has bandwidth type `Fixed`: the bandwidth
is taken as given rather than selected from the data. `Bartlett(Fixed(4))` and the
shorthand `Bartlett(4)` construct the same kernel. Use the type parameter to
dispatch on fixed-bandwidth kernels via `HAC{Fixed}`.

`bw` must be finite and nonnegative; a bandwidth of zero gives the
heteroskedasticity-robust (HC0) estimate.

# Usage
```julia
using CovarianceMatrices
k = Bartlett(Fixed(4))       # Bartlett kernel, fixed bandwidth 4
k == Bartlett(4)            # true
bandwidth(k)                # 4.0
k isa HAC{Fixed}            # true

# Dispatch on fixed-bandwidth kernels
describe(::HAC{Fixed}) = "fixed bandwidth"
describe(::HAC) = "data-driven bandwidth"
```

See also [`Andrews`](@ref) and [`NeweyWest`](@ref) for data-driven bandwidth selection.
"""
struct Fixed <: BandwidthType
    bw::WFLOAT
    function Fixed(bw)
        isfinite(bw) && bw >= 0 ||
            throw(ArgumentError("a fixed bandwidth must be finite and nonnegative, got $bw"))
        return new(bw)
    end
end

bandwidth(f::Fixed) = f.bw

# The bandwidth value a kernel stores. A fixed bandwidth must be finite and nonnegative.
# Zero means no lags (the HC0 estimate) for every kernel except `QuadraticSpectral`,
# whose weights at all lags are undefined for a zero bandwidth. A data-driven kernel
# takes no bandwidth. `bw === nothing` means no bandwidth was given.
function _checkbw(::Type{Fixed}, K, bw)
    bw === nothing &&
        throw(ArgumentError("a fixed-bandwidth $(_kernelname(K)) needs a bandwidth, as in `$(_kernelname(K))(5.0)`"))
    b = bw isa Fixed ? bandwidth(bw) : bw
    lo = K <: QuadraticSpectralKernel ? "positive" : "nonnegative"
    ok = isfinite(b) && (K <: QuadraticSpectralKernel ? b > 0 : b >= 0)
    ok ||
        throw(ArgumentError("the bandwidth of a $(nameof(K)) must be finite and $lo, got $b"))
    return Fixed(b)
end
function _checkbw(::Type{G}, K, bw) where {G}
    if G <: NeweyWest && K <: Union{TruncatedKernel, TukeyHanningKernel}
        throw(ArgumentError("the Newey-West (1994) bandwidth rule is defined only for the Bartlett, Parzen and QuadraticSpectral kernels; use `$(_kernelname(K)){Andrews}()` or a fixed bandwidth such as `$(_kernelname(K))(5.0)`"))
    end
    bw === nothing ||
        throw(ArgumentError("`$(_kernelname(K)){$(nameof(G))}` selects its bandwidth from the data and takes no bandwidth, got $bw. Use `$(_kernelname(K)){$(nameof(G))}()`, or `$(_kernelname(K))($bw)` for a fixed bandwidth."))
    return G()
end

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

# Constructors

Truncated(x::Real)              # Fixed bandwidth
Truncated(Fixed(x))             # Fixed bandwidth (same kernel)
Truncated{Andrews}()            # Andrews bandwidth selection
Truncated(Andrews)              # Andrews bandwidth selection (alternative syntax)

# Bandwidth Selection

  - `Fixed`: fixed bandwidth
  - `Andrews`: bandwidth selection a la Andrews

**Note**: Provides consistent but not necessarily positive semi-definite estimates. `NeweyWest` bandwidth selection is not supported for Truncated kernel.
"""
struct TruncatedKernel{G <: BandwidthType} <: HAC{G}
    bandwidth::G
    TruncatedKernel{G}(bw = nothing) where {G} = new{G}(_checkbw(G, TruncatedKernel, bw))
end

"""
`Truncated`

Alias for `TruncatedKernel`. Implements the truncated (uniform) kernel for HAC covariance estimation.
See [`TruncatedKernel`](@ref) for detailed documentation.
"""
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

# Constructors

Bartlett(x::Real)              # Fixed bandwidth
Bartlett(Fixed(x))             # Fixed bandwidth (same kernel)
Bartlett{Andrews}()            # Andrews bandwidth selection
Bartlett{NeweyWest}()          # Newey-West bandwidth selection
Bartlett(Andrews)              # Andrews bandwidth selection (alternative syntax)
Bartlett(NeweyWest)            # Newey-West bandwidth selection (alternative syntax)

# NotesL
- Equivalent to Newey-West estimator
"""
struct BartlettKernel{G <: BandwidthType} <: HAC{G}
    bandwidth::G
    BartlettKernel{G}(bw = nothing) where {G} = new{G}(_checkbw(G, BartlettKernel, bw))
end

"""
`Bartlett`

Alias for `BartlettKernel`. Implements the Bartlett (triangular) kernel for HAC covariance estimation.
See [`BartlettKernel`](@ref) for detailed documentation.
"""
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

# Constructors

Parzen(x::Real)                # Fixed bandwidth
Parzen(Fixed(x))               # Fixed bandwidth (same kernel)
Parzen{Andrews}()              # Andrews bandwidth selection
Parzen{NeweyWest}()            # Newey-West bandwidth selection
Parzen(Andrews)                # Andrews bandwidth selection (alternative syntax)
Parzen(NeweyWest)              # Newey-West bandwidth selection (alternative syntax)

"""
struct ParzenKernel{G <: BandwidthType} <: HAC{G}
    bandwidth::G
    ParzenKernel{G}(bw = nothing) where {G} = new{G}(_checkbw(G, ParzenKernel, bw))
end

"""
`Parzen`

Alias for `ParzenKernel`. Implements the Parzen kernel for HAC covariance estimation.
See [`ParzenKernel`](@ref) for detailed documentation.
"""
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

# Constructors

TukeyHanning(x::Real)           # Fixed bandwidth
TukeyHanning(Fixed(x))          # Fixed bandwidth (same kernel)
TukeyHanning{Andrews}()         # Andrews bandwidth selection
TukeyHanning(Andrews)           # Andrews bandwidth selection (alternative syntax)

# Notes
`NeweyWest` bandwidth selection is not supported for TukeyHanning kernel.
"""
struct TukeyHanningKernel{G <: BandwidthType} <: HAC{G}
    bandwidth::G
    function TukeyHanningKernel{G}(bw = nothing) where {G}
        new{G}(_checkbw(G, TukeyHanningKernel, bw))
    end
end

"""
`TukeyHanning`

Alias for `TukeyHanningKernel`. Implements the Tukey-Hanning kernel for HAC covariance estimation.
See [`TukeyHanningKernel`](@ref) for detailed documentation.
"""
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



# Constructors

QuadraticSpectral(x::Real)     # Fixed bandwidth
QuadraticSpectral(Fixed(x))    # Fixed bandwidth (same kernel)
QuadraticSpectral{Andrews}()   # Andrews bandwidth selection
QuadraticSpectral{NeweyWest}() # Newey-West bandwidth selection
QuadraticSpectral(Andrews)     # Andrews bandwidth selection (alternative syntax)
QuadraticSpectral(NeweyWest)   # Newey-West bandwidth selection (alternative syntax)

"""
struct QuadraticSpectralKernel{G <: BandwidthType} <: HAC{G}
    bandwidth::G
    function QuadraticSpectralKernel{G}(bw = nothing) where {G}
        new{G}(_checkbw(G, QuadraticSpectralKernel, bw))
    end
end

"""
`QuadraticSpectral`

Alias for `QuadraticSpectralKernel`. Implements the Quadratic Spectral kernel for HAC covariance estimation.
See [`QuadraticSpectralKernel`](@ref) for detailed documentation.
"""
const QuadraticSpectral = QuadraticSpectralKernel

## ------------------------------------------------------------------------------------------
# Define HAC constructor
# Three type of constructors
# - HAC(::Andrews) or HAC(::NeweyWest) Optimal bandwidth selection a la Andrews or a la NeweyWest
# - HAC(bw) Fixed bandwidth
# - HAC() -> Optimal bandwidth selection a la Andrews
const kernels = [:Bartlett, :Parzen, :QuadraticSpectral, :Truncated, :TukeyHanning]

for kerneltype in kernels
    @eval ($kerneltype(x::Number)) = ($kerneltype){Fixed}(x)
    @eval ($kerneltype(f::Fixed)) = ($kerneltype){Fixed}(f)
    # Data-driven bandwidth given as a type: Bartlett(NeweyWest)
    @eval ($kerneltype(::Type{G}) where {G <:
                                         Union{Andrews, NeweyWest}}) = ($kerneltype){G}()
end

# Kernels print under their exported alias, as constructor syntax: `Bartlett(5.0)`,
# `Bartlett{Andrews}()`.
for kerneltype in kernels
    @eval _kernelname(::Type{<:$kerneltype}) = $(string(kerneltype))
end
_kernelname(k::HAC) = _kernelname(typeof(k))
function Base.show(io::IO, x::HAC{T}) where {T <: Union{Andrews, NeweyWest}}
    return print(io, _kernelname(x), "{", T, "}()")
end
function Base.show(io::IO, x::HAC{Fixed})
    print(io, _kernelname(x), "(")
    show(io, bandwidth(x))
    return print(io, ")")
end
## Makes the default bandwidth selection
#Optimal() = Optimal{Andrews}()

## Accessor
"""
    bandwidth(k::HAC{Fixed})
    bandwidth(f::Fixed)

Return the fixed bandwidth of `k` or `f`.

A kernel with data-driven bandwidth selection (`Andrews`, `NeweyWest`) carries no
bandwidth of its own — the bandwidth is a property of an estimate, not of the
specification. Read it from the result with [`bandwidth(::CovarianceMatrix)`](@ref),
or compute it directly with [`optimalbw`](@ref).

# Usage
```julia
bandwidth(Bartlett(4))                  # 4.0

V = aVar(Bartlett{Andrews}(), X)
bandwidth(V)                            # the selected bandwidth
optimalbw(Bartlett{Andrews}(), X)       # the same value, computed on its own
```
"""
bandwidth(k::HAC{Fixed}) = bandwidth(k.bandwidth)

function bandwidth(k::HAC{T}) where {T <: Union{Andrews, NeweyWest}}
    throw(ArgumentError("`$(nameof(typeof(k))){$(nameof(T))}` selects its bandwidth from the data, so the estimator does not carry one. Read it from a result with `bandwidth(aVar(k, X))`, or compute it with `optimalbw(k, X)`."))
end

#=========
EWC
=========#
"""
    EWC <: Correlated

Equal Weighted Cosine (EWC) covariance matrix estimator.

The EWC estimator projects the moment conditions ``g_1, \\dots, g_T`` onto ``B``
cosine basis functions and averages the outer products of the projections:

```math
\\hat{\\Omega} = \\frac{1}{B} \\sum_{j=1}^{B} \\Lambda_j \\Lambda_j',
\\qquad
\\Lambda_j = \\sqrt{2} \\sum_{t=1}^{T} \\cos\\!\\left(\\frac{\\pi j (t - 1/2)}{T}\\right) g_t.
```

The number of basis functions ``B`` is the single tuning parameter; it plays the
role the bandwidth plays in a HAC estimator. Because the estimate is a sum of
outer products, it is positive semi-definite by construction.

# Constructor
    EWC(B::Integer)


# Examples
```julia
# EWC with 10 basis functions
ve = EWC(10)

# Compute covariance matrix
Ω = aVar(ve, moment_matrix)
```
"""
struct EWC <: Correlated
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

"""
`HR0` (equivalent to `HC0`)

Basic heteroskedasticity-robust covariance estimator without finite-sample corrections.

# Mathematical Formula

Letting `X_i` denote the i-th row of the design matrix `X` and ``\\hat{u}_i`` the residuals, the HC0 estimator is given by:
```math
\\hat{\\Omega}_{HC0} = n\\right(\\sum_{i=1}^n X_i'X_i\\left)^{-1} \\sum_{i=1}^n u_i^2 X'_iX_i \\right(\\sum_{i=1}^n X_i'X_i\\left)^{-1}
```

# Usage
```julia
using CovarianceMatrices
se_hc0 = stderror(HC0(), model)
vcov_hc0 = vcov(HR0(), model)
```
"""
struct HR0 <: HR end

"""
`HR1` (equivalent to `HC1`)

Heteroskedasticity-robust estimator with degrees-of-freedom correction.

# Mathematical Formula

```math
\\hat{\\Omega}_{HC1} = \\frac{n}{n-k} \\hat{\\Omega}_{HC0}
```

where ``n`` is sample size and ``k`` is number of parameters.

# Usage
```julia
using CovarianceMatrices
se_hc1 = stderror(HC1(), model)
vcov_hc1 = vcov(HR1(), model)
```
"""
struct HR1 <: HR end

"""
`HR2` (equivalent to `HC2`)

Heteroskedasticity-robust estimator with leverage-based corrections.

# Mathematical Formula

```math
\\hat{\\Omega}_{HC2} = (X'X)^{-1} X'\\text{diag}\\left(\\frac{\\hat{u}_i^2}{1-h_i}\\right) X (X'X)^{-1}
```

where ``h_i`` are the diagonal elements of the hat matrix ``H = X(X'X)^{-1}X'``.

# Usage
```julia
using CovarianceMatrices
se_hc2 = stderror(HC2(), model)
vcov_hc2 = vcov(HR2(), model)
```
"""
struct HR2 <: HR end

"""
`HR3` (equivalent to `HC3`)

Heteroskedasticity-robust estimator with squared leverage corrections.

# Mathematical Formula

```math
\\hat{\\Omega}_{HC3} = (X'X)^{-1} X'\\text{diag}\\left(\\frac{\\hat{u}_i^2}{(1-h_i)^2}\\right) X (X'X)^{-1}
```

where ``h_i`` are the diagonal elements of the hat matrix ``H = X(X'X)^{-1}X'``.


# Usage
```julia
using CovarianceMatrices
se_hc3 = stderror(HC3(), model)
vcov_hc3 = vcov(HR3(), model)
```
"""
struct HR3 <: HR end

"""
`HR4` (equivalent to `HC4`)

Heteroskedasticity-robust estimator with alternative leverage correction.

# Mathematical Formula

```math
\\hat{\\Omega}_{HC4} = (X'X)^{-1} X'\\text{diag}\\left(\\frac{\\hat{u}_i^2}{(1-h_i)^{\\delta_i}}\\right) X (X'X)^{-1}
```

where ``\\delta_i = \\min\\{4, h_i/\\bar{h}\\}`` and ``\\bar{h} = k/n``.

# Usage
```julia
using CovarianceMatrices
se_hc4 = stderror(HC4(), model)
vcov_hc4 = vcov(HR4(), model)
```
"""
struct HR4 <: HR end

"""
`HR4m` (equivalent to `HC4m`)

Modified version of HC4 with different leverage cutoff.

# Mathematical Formula

Similar to HC4 but with ``\\delta_i = \\min\\{1, h_i/\\bar{h}\\} + \\min\\{1.5, h_i/\\bar{h}\\}``.


# Usage
```julia
using CovarianceMatrices
se_hc4m = stderror(HC4m(), model)
vcov_hc4m = vcov(HR4m(), model)
```
"""
struct HR4m <: HR end

"""
`HR5` (equivalent to `HC5`)

Heteroskedasticity-robust estimator with maximum leverage correction.

# Mathematical Formula

```math
\\hat{\\Omega}_{HC5} = (X'X)^{-1} X'\\text{diag}\\left(\\frac{\\hat{u}_i^2}{\\sqrt{(1-h_i)(1-h_i^*)}}\\right) X (X'X)^{-1}
```

where ``h_i^* = \\max\\{4h_i/k, h_i\\}``.

# Usage
```julia
using CovarianceMatrices
se_hc5 = stderror(HC5(), model)
vcov_hc5 = vcov(HR5(), model)
```
"""
struct HR5 <: HR end

#=========
CR
=========#

"""
`CR0`

Basic cluster-robust covariance estimator without small-sample adjustments.

# Mathematical Formula

```math
\\hat{\\Omega}_{CR0} = \\sum_{g=1}^G \\hat{u}_g \\hat{u}_g'
```

where ``\\hat{u}_g = \\sum_{t \\in g} g_t`` is the cluster-level sum of moment conditions.


# Usage
```julia
using CovarianceMatrices
cluster_ids = [1, 1, 2, 2, 3, 3, 4, 4]
se_cr0 = stderror(CR0(cluster_ids), model)

# Multi-way clustering
se_cr0_multi = stderror(CR0((firm_ids, year_ids)), model)
```
"""
struct CR0{G} <: CR
    g::G
    function CR0(g::G) where {G <: AbstractVector}
        t = (Clustering(g),)
        new{typeof(t)}(t)
    end
    CR0(g::G) where {G <: Tuple{Vararg{Symbol}}} = new{G}(g)
    function CR0(g::G) where {G <: Tuple}
        t = map(Clustering, g)
        new{typeof(t)}(t)
    end
end

"""
`CR1`

Cluster-robust estimator with degrees-of-freedom correction.

# Mathematical Formula

```math
\\hat{\\Omega}_{CR1} = \\frac{G}{G-1} \\cdot \\frac{N-1}{N-K} \\hat{\\Omega}_{CR0}
```

where ``N`` is total sample size, ``K`` is number of parameters, and ``G`` is number of clusters.


# Usage
```julia
using CovarianceMatrices
cluster_ids = [1, 1, 2, 2, 3, 3, 4, 4]
se_cr1 = stderror(CR1(cluster_ids), model)

# Multi-way clustering
se_cr1_multi = stderror(CR1((firm_ids, year_ids)), model)
```
"""
struct CR1{G} <: CR
    g::G
    function CR1(g::G) where {G <: AbstractVector}
        t = (Clustering(g),)
        new{typeof(t)}(t)
    end
    CR1(g::G) where {G <: Tuple{Vararg{Symbol}}} = new{G}(g)
    function CR1(g::G) where {G <: Tuple}
        t = map(Clustering, g)
        new{typeof(t)}(t)
    end
end

"""
`CR2`

Cluster-robust estimator with leverage-based corrections.

# Mathematical Formula

```math
\\hat{\\Omega}_{CR2} = \\frac{G}{G-1} \\sum_{g=1}^G \\frac{1}{1-h_g} \\hat{u}_g \\hat{u}_g'
```

where ``h_g`` represents cluster-level leverage values.


# Usage
```julia
using CovarianceMatrices
cluster_ids = [1, 1, 2, 2, 3, 3, 4, 4]
se_cr2 = stderror(CR2(cluster_ids), model)

# Multi-way clustering
se_cr2_multi = stderror(CR2((firm_ids, year_ids)), model)
```
"""
struct CR2{G} <: CR
    g::G
    function CR2(g::G) where {G <: AbstractVector}
        t = (Clustering(g),)
        new{typeof(t)}(t)
    end
    CR2(g::G) where {G <: Tuple{Vararg{Symbol}}} = new{G}(g)
    function CR2(g::G) where {G <: Tuple}
        t = map(Clustering, g)
        new{typeof(t)}(t)
    end
end

"""
`CR3`

Cluster-robust estimator with squared leverage corrections.

# Mathematical Formula

```math
\\hat{\\Omega}_{CR3} = \\frac{G}{G-1} \\sum_{g=1}^G \\frac{1}{(1-h_g)^2} \\hat{u}_g \\hat{u}_g'
```

where ``h_g`` represents cluster-level leverage values.


# Usage
```julia
using CovarianceMatrices
cluster_ids = [1, 1, 2, 2, 3, 3, 4, 4]
se_cr3 = stderror(CR3(cluster_ids), model)

# Multi-way clustering
se_cr3_multi = stderror(CR3((firm_ids, year_ids)), model)
```
"""
struct CR3{G} <: CR
    g::G
    function CR3(g::G) where {G <: AbstractVector}
        t = (Clustering(g),)
        new{typeof(t)}(t)
    end
    CR3(g::G) where {G <: Tuple{Vararg{Symbol}}} = new{G}(g)
    function CR3(g::G) where {G <: Tuple}
        t = map(Clustering, g)
        new{typeof(t)}(t)
    end
end

for k in [:CR0, :CR1, :CR2, :CR3, :Cluster]
    # Single symbol: wrap in tuple for consistent representation
    @eval $(k)(g::Symbol) = $(k)((g,))
    # Varargs for 2+ args: multi-way clustering CR0(a, b, ...) -> CR0((a, b, ...))
    @eval $(k)(g1, g2, gs...) = $(k)((g1, g2, gs...))
end

# Cluster estimators that name their clusters by column. A package that owns the data
# table replaces the names with the group vectors before any variance is computed.
const _UnresolvedClusterEstimator = Union{
    CR0{<:Tuple{Vararg{Symbol}}}, CR1{<:Tuple{Vararg{Symbol}}},
    CR2{<:Tuple{Vararg{Symbol}}}, CR3{<:Tuple{Vararg{Symbol}}},
    Cluster{<:Tuple{Vararg{Symbol}}}}

_check_resolved(k) = k
function _check_resolved(k::_UnresolvedClusterEstimator)
    E = nameof(typeof(k))
    throw(ArgumentError(
        "$(sprint(show, k)) names its clusters by column ($(join(repr.(k.g), ", "))), " *
        "which CovarianceMatrices cannot look up. Pass the group identifiers instead, " *
        "e.g. `$E(df.$(first(k.g)))`, or use a modeling package that resolves column " *
        "names against its data."))
end

#=========
CRCache, CRModelCache and CachedCR
=========#

"""
    precompute_cluster_indices(g::Clustering)

Compute observation indices for each cluster in a single O(n) pass.
Returns `Vector{Vector{Int}}` where `result[cluster_id]` contains all row indices for that cluster.
"""
function precompute_cluster_indices(g::Clustering)
    indices = [Int[] for _ in 1:g.ngroups]
    for i in eachindex(g.groups)
        push!(indices[g.groups[i]], i)
    end
    return indices
end

# The clusterings, per-cluster observation indices and inclusion-exclusion signs for
# every nonempty combination of the clustering dimensions `f`.
function _cluster_combinations(f)
    grouped_arrays = Clustering[]
    cluster_indices = Vector{Vector{Int}}[]
    signs = Int[]
    # Filter out empty combinations (Combinatorics.jl < 1.1 includes empty set)
    for c in Iterators.filter(!isempty, combinations(1:length(f)))
        g = length(c) == 1 ? Clustering(f[c[1]]) : Clustering((f[i] for i in c)...)
        push!(grouped_arrays, g)
        push!(signs, (-1)^(length(c) - 1))
        push!(cluster_indices, precompute_cluster_indices(g))
    end
    return grouped_arrays, cluster_indices, signs
end

"""
    CRCache{T}

Preallocated buffers and precomputed data for fast repeated cluster-robust
variance calculations on a moment matrix. Built by `CachedCR(k, ncols)`.

# Fields
- `X2_buffers`: Tuple of preallocated matrices (ngroups × ncols) for cluster aggregation
- `S_buffer`: Preallocated output matrix (ncols × ncols)
- `grouped_arrays`: Precomputed Clusterings for each combination (for multi-way clustering)
- `cluster_indices`: Precomputed observation indices for each cluster (enables fast gather)
- `signs`: Precomputed signs for inclusion-exclusion (-1)^(length(c)-1)
- `ncols`: Number of columns the cache was built for

# Notes
Using a cache makes the variance calculation non-differentiable with AD.
For AD compatibility, use the standard non-cached CR estimators.
"""
struct CRCache{T <: Real}
    X2_buffers::Vector{Matrix{T}}            # One buffer per combination
    S_buffer::Matrix{T}                       # Output buffer (ncols × ncols)
    grouped_arrays::Vector{Clustering}      # Precomputed for each combination
    cluster_indices::Vector{Vector{Vector{Int}}}  # [combination][cluster] -> obs indices
    signs::Vector{Int}                        # (-1)^(length(c)-1) for each combination
    ncols::Int                                # Number of columns
end

"""
    CRCache(k::CR, ncols::Int, ::Type{T}=Float64) where T

Construct a cache for cluster-robust variance calculations on a moment matrix with
`ncols` columns. `CachedCR(k, ncols)` builds one and wraps it with `k`.
"""
function CRCache(k::CR, ncols::Int, ::Type{T} = Float64) where {T}
    grouped_arrays, cluster_indices, signs = _cluster_combinations(k.g)
    X2_buffers = [zeros(T, g.ngroups, ncols) for g in grouped_arrays]
    S_buffer = zeros(T, ncols, ncols)
    return CRCache{T}(X2_buffers, S_buffer, grouped_arrays, cluster_indices, signs, ncols)
end

"""
    CRModelCache{T, H}

Cache for cluster-robust variance calculations on a `RegressionModel`. Built by
`CachedCR(k, model)`. Stores the leverage adjustments, which depend only on the model
matrix and the cluster structure, not on the residuals. This enables fast repeated
variance calculations (e.g., wild bootstrap) where only residuals change between
iterations.

# Type Parameters
- `T`: Element type (e.g., Float64)
- `H`: Type of leverage adjustments (Vector of BlockDiagonal for CR2/CR3, Vector of scalars for CR0/CR1)

# Fields
- `grouped_arrays`: Precomputed Clusterings for each clustering combination
- `cluster_indices`: Precomputed observation indices [combination][cluster] -> obs indices
- `signs`: Precomputed signs for inclusion-exclusion formula
- `bread_matrix`: (X'X)^-1 of the model the cache was built from
- `leverage_adjustments`: Precomputed leverage adjustments (BlockDiagonal for CR2/CR3)
- `ncols`: Number of model coefficients (for validation)
- `nobs`: Number of observations (for validation)

# Notes
Using this cache makes the variance calculation non-differentiable with AD.
For AD compatibility, use the standard non-cached CR estimators.
"""
struct CRModelCache{T <: Real, H}
    grouped_arrays::Vector{Clustering}
    cluster_indices::Vector{Vector{Vector{Int}}}
    signs::Vector{Int}
    bread_matrix::Matrix{T}
    leverage_adjustments::H
    ncols::Int
    nobs::Int
end

"""
    CachedCR(k::CR, ncols::Integer, ::Type{T}=Float64)
    CachedCR(k::CR, model::RegressionModel)

Wrap the cluster-robust estimator `k` with precomputed cluster indices and buffers,
for repeated variance calculations with the same cluster structure, such as a wild
bootstrap or a Monte Carlo study.

- `CachedCR(k, ncols)` preallocates buffers for moment matrices with `ncols` columns
  (`aVar(cached, X)` reuses them and returns the buffer itself, overwritten on the
  next call).
- `CachedCR(k, model)` also precomputes the leverage adjustments of `CR2`/`CR3`, which
  depend only on the model matrix and the clusters, so `vcov(cached, m)` recomputes
  only the residual-dependent part. `m` must have the same model matrix as `model`.

A `CachedCR` is a `CR` estimator and works wherever `k` does, on moment matrices and
on models, with the same results. Two `CachedCR` values are equal when their
estimators are equal; the cache is ignored.

# Warning
The cached computation is not differentiable with automatic differentiation (AD).
For AD compatibility, use the standard non-cached CR estimators directly.

# Example
```julia
using CovarianceMatrices

cluster_ids = repeat(1:100, inner=10)
X = randn(1000, 5)
cached_k = CachedCR(CR0(cluster_ids), size(X, 2))

# Wild bootstrap - same cluster structure, different moment matrices
for b in 1:1000
    weights = rand([-1, 1], 1000)
    S = aVar(cached_k, X .* weights)  # Uses cached buffers
end
```

See also: [`CR0`](@ref), [`CR1`](@ref), [`CR2`](@ref), [`CR3`](@ref), [`CRCache`](@ref),
[`CRModelCache`](@ref)
"""
struct CachedCR{K <: CR, C <: Union{CRCache, CRModelCache}} <: CR
    estimator::K
    cache::C
end

function CachedCR(k::CR, ncols::Integer, ::Type{T} = Float64) where {T}
    _check_resolved(k)
    return CachedCR(k, CRCache(k, Int(ncols), T))
end

Base.@deprecate_binding CachedCRModel CachedCR

# Forward nclusters to underlying estimator
nclusters(k::CachedCR) = nclusters(k.estimator)

# Access underlying estimator's grouped arrays
function Base.getproperty(k::CachedCR, s::Symbol)
    s === :g ? getfield(k, :estimator).g : getfield(k, s)
end

#=========
VARHAC
=========#
"""
    LagSelector

Abstract supertype for the lag-order criteria of [`VARHAC`](@ref): `AICSelector` and
`BICSelector`.
"""
abstract type LagSelector end

"""
`AICSelector`

AIC-based lag selector for VARHAC estimation.

Selects the optimal number of lags by minimizing the Akaike Information Criterion (AIC).
Used in conjunction with VARHAC for automatic bandwidth selection.

# Usage
```julia
using CovarianceMatrices
varhac_aic = VARHAC(AICSelector(), SameLags(10))
```
"""
struct AICSelector <: LagSelector end

"""
`BICSelector`

BIC-based lag selector for VARHAC estimation.

Selects the optimal number of lags by minimizing the Bayesian Information Criterion (BIC).
Generally produces more parsimonious models than AIC.

# Usage
```julia
using CovarianceMatrices
varhac_bic = VARHAC(BICSelector(), SameLags(10))
```
"""
struct BICSelector <: LagSelector end

"""
`FixedSelector`

Internal selector stored by `VARHAC(FixedLags(k))`, which fits a VAR of fixed order
and selects nothing. `VARHAC` rejects it as an explicit argument.
"""
struct FixedSelector <: LagSelector end

"""
    LagStrategy

Abstract supertype for the lag specifications of [`VARHAC`](@ref): `FixedLags`,
`SameLags`, `DifferentOwnLags` and `AutoLags`.
"""
abstract type LagStrategy end

# Lag orders are integers; an integer-valued `Real` converts silently.
function _lag_integer(x::Real, name::Symbol)
    isinteger(x) ||
        throw(ArgumentError("`$name($x)`: a lag order must be an integer, got $x"))
    return Int(x)
end

"""
`FixedLags`

Lag strategy that fits a VAR of fixed order, without lag selection.

# Fields
- `maxlag::Int`: the VAR order, at least 1

# Usage
```julia
using CovarianceMatrices
varhac = VARHAC(FixedLags(8))
```
"""
struct FixedLags <: LagStrategy
    maxlag::Int
    function FixedLags(maxlag::Integer)
        maxlag >= 1 || throw(ArgumentError(
            "FixedLags: the VAR order must be at least 1, got $maxlag"))
        return new(maxlag)
    end
end

FixedLags(x::Real) = FixedLags(_lag_integer(x, :FixedLags))
FixedLags() = FixedLags(5)

"""
`SameLags`

Lag strategy that uses the same maximum number of lags for all variables.

Similar to `FixedLags` but with different default behavior in VARHAC.

# Fields
- `maxlag::Int`: Maximum number of lags to search, at least 1

# Usage
```julia
using CovarianceMatrices
same_lags = SameLags(10)
varhac = VARHAC(BICSelector(), same_lags)
```
"""
struct SameLags <: LagStrategy
    maxlag::Int
    function SameLags(maxlag::Integer)
        maxlag >= 1 || throw(ArgumentError(
            "SameLags: the maximum lag order must be at least 1, got $maxlag"))
        return new(maxlag)
    end
end

SameLags(x::Real) = SameLags(_lag_integer(x, :SameLags))
SameLags() = SameLags(8)  # Better default based on practical experience

"""
`AutoLags`

Automatic lag selection strategy based on sample size.

Uses a data-driven rule to select the maximum number of lags based on the sample size,
typically following a T^(1/3) growth rule with practical constraints.

# Usage
```julia
using CovarianceMatrices
auto_lags = AutoLags()
varhac = VARHAC(AICSelector(), auto_lags)
```
"""
struct AutoLags <: LagStrategy end

# Function to compute automatic lag selection based on T^(1/3) rule
function compute_auto_maxlag(T::Int, N::Int)
    # Rule from literature: K_max should grow no faster than T^(1/3)
    # Also ensure it doesn't exceed (T-1)/N to avoid overfitting
    max_theoretical = max(1, floor(Int, T^(1 / 3)))
    max_practical = max(1, floor(Int, (T - 1) / N))
    return min(max_theoretical, max_practical, 20)  # Cap at reasonable maximum
end

"""
`DifferentOwnLags`

Lag strategy that searches the lags of a variable's own series and the lags of the
other series separately. For each equation of the VAR, the selector picks an own
order in `0:own` and a cross order in `0:cross`, applied to every other variable.

# Fields
- `maxlags::Vector{Int}`: `[own, cross]`, the maximum own and cross lag orders

# Usage
```julia
using CovarianceMatrices
diff_lags = DifferentOwnLags([3, 5])  # own lags up to 3, cross lags up to 5
varhac = VARHAC(AICSelector(), diff_lags)

# Alternative constructors
diff_lags = DifferentOwnLags((3, 5))  # From tuple
diff_lags = DifferentOwnLags()        # Default: [5, 5]
```
"""
struct DifferentOwnLags <: LagStrategy
    maxlags::Vector{Int}
    function DifferentOwnLags(maxlags::AbstractVector{<:Integer})
        length(maxlags) == 2 || throw(ArgumentError(
            "DifferentOwnLags takes the maximum own and cross lag orders, " *
            "`DifferentOwnLags([own, cross])`; got $(length(maxlags)) entries"))
        all(>=(0), maxlags) || throw(ArgumentError(
            "DifferentOwnLags: lag orders must be non-negative, got $maxlags"))
        return new(Vector{Int}(maxlags))
    end
end

function DifferentOwnLags(x::AbstractVector{<:Real})
    return DifferentOwnLags(map(v -> _lag_integer(v, :DifferentOwnLags), x))
end
DifferentOwnLags(x::Tuple) = DifferentOwnLags(collect(x))
DifferentOwnLags() = DifferentOwnLags([5, 5])

"""
    VARHAC{S<:LagSelector, L<:LagStrategy} <: Correlated

Vector Autoregression HAC (VARHAC) estimator for robust covariance estimation.

The VARHAC estimator provides a data-driven approach to HAC estimation by
fitting a Vector Autoregression (VAR) model to the moment conditions and using
the spectral density at frequency zero as the covariance estimator. This approach
automatically accounts for serial correlation and cross-correlation patterns
without requiring bandwidth selection.

# Construction

The recommended form supplies a selector and a lag strategy explicitly:

    VARHAC(selector=AICSelector(), strategy=SameLags(8))

`VARHAC()` uses the defaults (AIC, `SameLags(8)`) and is the headline form for
routine use. Spell out both arguments — `VARHAC(BICSelector(), AutoLags())` —
when you need full control.

The following convenience aliases expand to that explicit form:

    - `VARHAC(8)`           => `VARHAC(AICSelector(), SameLags(8))`
    - `VARHAC(:aic)`        => `VARHAC(AICSelector(), SameLags(8))`
    - `VARHAC(:bic)`        => `VARHAC(BICSelector(), SameLags(8))`
    - `VARHAC(Val(:auto))`  => `VARHAC(AICSelector(), AutoLags())`

For sample-size-driven lag selection, prefer the explicit
`VARHAC(AICSelector(), AutoLags())` over the `Val(:auto)` alias — it reads more
clearly and lets you choose the selector.

## Arguments

- `selector::LagSelector`: Method for selecting optimal lag length
  - `AICSelector()`: Use Akaike Information Criterion (default)
  - `BICSelector()`: Use Bayesian Information Criterion

- `strategy::LagStrategy`: Strategy for lag specification
  - `SameLags(k)`: search orders `1:k`, the same for all variables (default: k=8)
  - `AutoLags()`: search range set from the sample size
  - `DifferentOwnLags([own, cross])`: search own and cross lag orders separately

A VAR of fixed order, with no selection, is `VARHAC(FixedLags(k))`; it takes no
selector: passing `AICSelector` or `BICSelector` together with `FixedLags` throws an
`ArgumentError`. The estimate takes the element type of the data.

The selected orders are reported by [`order`](@ref) on the result and always come from
the criterion the estimator is configured with.

# References

- den Haan, W.J. and Levin, A. (1997). "A Practitioner's Guide to Robust
  Covariance Matrix Estimation". Handbook of Statistics, Vol. 15.

# Examples
```julia
# Headline form: defaults (AIC, SameLags(8))
ve1 = VARHAC()

# BIC selection with a larger lag search
ve2 = VARHAC(BICSelector(), SameLags(12))

# Fixed lag length, no selection
ve3 = VARHAC(FixedLags(6))

# Automatic lag selection based on sample size
ve4 = VARHAC(AICSelector(), AutoLags())

# Convenience aliases
ve5 = VARHAC(:bic)   # BICSelector(), SameLags(8)
ve6 = VARHAC(12)     # AICSelector(), SameLags(12)
```
"""
struct VARHAC{S <: LagSelector, L <: LagStrategy} <: Correlated
    selector::S
    strategy::L
    # Inner constructor: suppresses the default `VARHAC(selector, strategy)`, which
    # would bypass the checks in `_varhac`.
    VARHAC{S, L}(selector, strategy) where {S, L} = new{S, L}(selector, strategy)
end

VARHAC(selector = AICSelector(), strategy = SameLags(8)) = _varhac(selector, strategy)

VARHAC(f::FixedLags) = VARHAC{FixedSelector, FixedLags}(FixedSelector(), f)

# Quick selector construction: VARHAC(:aic) or VARHAC(:bic)
VARHAC(selector_symbol::Symbol) = VARHAC(_symbol_to_selector(selector_symbol), SameLags(8))

# Quick max lags construction: VARHAC(12)
VARHAC(max_lags::Integer) = VARHAC(AICSelector(), SameLags(max_lags))

# Auto-selection constructor: VARHAC(Val(:auto))
VARHAC(::Val{:auto}) = VARHAC(AICSelector(), AutoLags())

function _varhac(selector::LagSelector, strategy::LagStrategy)
    if strategy isa FixedLags
        throw(ArgumentError("`VARHAC($(sprint(show, selector)), $(sprint(show, strategy)))`: " *
                            "`FixedLags` fits a VAR of fixed order and selects nothing, so " *
                            "it takes no selector. Use `VARHAC($(sprint(show, strategy)))`."))
    elseif selector isa FixedSelector
        throw(ArgumentError("`FixedSelector` is not a lag-order criterion. Use " *
                            "`AICSelector()` or `BICSelector()`, or " *
                            "`VARHAC(FixedLags(k))` for a fixed order."))
    end
    return VARHAC{typeof(selector), typeof(strategy)}(selector, strategy)
end

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
    else
        throw(ArgumentError("Invalid selector symbol: $s. Use :aic or :bic, or " *
                            "`VARHAC(FixedLags(k))` for a fixed order."))
    end
end

"""
    maxlags(k::VARHAC)

Return the maximum number of lags used in VARHAC estimation.

The return type depends on the lag strategy:
- `FixedLags`/`SameLags`: Returns a single integer
- `DifferentOwnLags`: Returns `[own, cross]`, the maximum own and cross lag orders
- `AutoLags`: Requires calling with data dimensions `maxlags(k, T, N)`

# Usage
```julia
# Fixed/Same lags
varhac = VARHAC(AICSelector(), SameLags(5))
max_lag = maxlags(varhac)  # Returns 5

# Different own lags
varhac = VARHAC(BICSelector(), DifferentOwnLags([3, 5]))
max_lags = maxlags(varhac)  # Returns [3, 5]
```
"""
maxlags(k::VARHAC{S, L}) where {S <: LagSelector, L <: SameLags} = k.strategy.maxlag
function maxlags(k::VARHAC{S, L}) where {S <: LagSelector, L <: DifferentOwnLags}
    k.strategy.maxlags
end
maxlags(k::VARHAC{S, L}) where {S <: LagSelector, L <: FixedLags} = k.strategy.maxlag

# For AutoLags, we need the data dimensions to compute optimal lag length
function maxlags(k::VARHAC{S, AutoLags}, T_data::Int, N::Int) where {S <: LagSelector}
    return compute_auto_maxlag(T_data, N)
end
# Fallback that throws informative error if AutoLags used without dimensions
function maxlags(k::VARHAC{S, AutoLags}) where {S <: LagSelector}
    throw(ArgumentError("AutoLags requires data dimensions. Use maxlags(estimator, T, N) where T is sample size and N is number of variables."))
end

"""
    AICs(V::CovarianceMatrix)

Return the AIC values over the lag orders searched during VARHAC estimation.

# Usage
```julia
V = aVar(VARHAC(AICSelector(), SameLags(5)), X)
AICs(V)
```
"""
AICs(V::CovarianceMatrix) = V.info.AICs

"""
    BICs(V::CovarianceMatrix)

Return the BIC values over the lag orders searched during VARHAC estimation.

# Usage
```julia
V = aVar(VARHAC(BICSelector(), SameLags(5)), X)
BICs(V)
```
"""
BICs(V::CovarianceMatrix) = V.info.BICs

"""
    order_aic(V::CovarianceMatrix)

Return the lag orders the AIC criterion selected during VARHAC estimation.
"""
order_aic(V::CovarianceMatrix) = V.info.order_aic

"""
    order_bic(V::CovarianceMatrix)

Return the lag orders the BIC criterion selected during VARHAC estimation.
"""
order_bic(V::CovarianceMatrix) = V.info.order_bic

"""
    order(V::CovarianceMatrix)

Return the lag orders selected by the criterion the estimator was configured with:
AIC for `AICSelector`, BIC for `BICSelector`.

# Usage
```julia
V = aVar(VARHAC(AICSelector(), SameLags(5)), X)
order(V)
```
"""
order(V::CovarianceMatrix) = V.info.order

function Base.show(io::IO, k::VARHAC)
    print(io, "VARHAC(")
    if !(k.strategy isa FixedLags)
        show(io, k.selector)
        print(io, ", ")
    end
    show(io, k.strategy)
    return print(io, ")")
end

#========
##DriscollKraay
=========#
"""
    DriscollKraay{K, D, I} <: Correlated

Driscoll-Kraay estimator for panel data with cross-sectional and temporal dependence.

The moments of all units observed in the same period are summed, and the HAC kernel
`K` is applied to the resulting series of period sums. Any correlation across units
within a period is absorbed by the sum, and the kernel accounts for serial correlation
across periods. The estimator is consistent as the number of periods grows,
irrespective of the number of units.

# Constructor
    DriscollKraay(K::HAC; tis, iis, correction = :HC0)
    DriscollKraay(K::HAC, tis::AbstractVector, iis::AbstractVector; correction = :HC0)

where

- `K::HAC`: HAC kernel for temporal dependence (Bartlett, Parzen, etc.)
- `tis`: time index of each observation. Periods are ordered by `sort(unique(tis))`, so
  the identifiers must sort chronologically (numbers, `Date`s, or zero-padded strings).
- `iis`: unit index of each observation. It does not enter the estimate; it is used to
  check that each (period, unit) pair occurs at most once.
- `correction`: finite-sample correction applied by `vcov(k, model)` for a
  `RegressionModel`; one of `:HC0` (none), `:HC1` or `:sss`. See
  [`vcov(::DriscollKraay, ::RegressionModel)`](@ref).

Both index arrays are required, must have the same length, and may hold identifiers
of any sortable type; the two arrays need not share an element type.

# Mathematical Foundation
With ``g_t`` the sum of the moments of the observations in period ``t = 1, …, T``
(after demeaning the moments), the estimator computes

```math
\\hat{S} = \\sum_{t=1}^{T} \\sum_{s=1}^{T} w\\!\\left(\\frac{t - s}{b}\\right) g_t g_s'
```

where ``w`` is the kernel of `K` and ``b`` its bandwidth. `aVar` divides ``\\hat{S}`` by
the number of observations `n` (with `scale = true`), the same per-observation scale as
every other estimator.

# References
- Driscoll, J.C. and Kraay, A.C. (1998). "Consistent Covariance Matrix Estimation
  with Spatially Dependent Panel Data". Review of Economics and Statistics, 80(4), 549-560.

# Examples
```julia
# Basic panel setup
ve = DriscollKraay(Bartlett{Andrews}(), tis=time_ids, iis=unit_ids)

# With fixed bandwidth and the Stata-style small-sample correction
ve = DriscollKraay(Parzen(4), tis=years, iis=countries, correction=:sss)
```
"""
struct DriscollKraay{K, D, I} <: Correlated
    K::K
    tis::D
    iis::I
    correction::Symbol

    function DriscollKraay{K, D, I}(kernel, tis, iis, correction = :HC0) where {K, D, I}
        _check_dk_correction(correction)
        _check_dk_panel(tis, iis)
        return new{K, D, I}(kernel, tis, iis, correction)
    end
end

function DriscollKraay(K::HAC; tis = nothing, iis = nothing, correction = :HC0)
    return DriscollKraay(K, tis, iis; correction)
end

function DriscollKraay(K::HAC, tis, iis; correction = :HC0)
    tis === nothing &&
        throw(ArgumentError("DriscollKraay requires time indices; pass `tis`."))
    iis === nothing &&
        throw(ArgumentError("DriscollKraay requires entity indices; pass `iis`."))
    ct, ci = _sorted_clustering(tis), Clustering(iis)
    return DriscollKraay{typeof(K), typeof(ct), typeof(ci)}(K, ct, ci, correction)
end

# Group codes follow the sort order of the identifiers, so that code `t` is the
# `t`-th period in time: the kernel weights depend on the distance between codes.
function _sorted_clustering(v::AbstractVector)
    u = sort!(unique(v))
    return Clustering(Int[searchsortedfirst(u, x) for x in v], length(u))
end
_sorted_clustering(c::Clustering) = c

const _DK_CORRECTIONS = (:HC0, :HC1, :sss)

function _check_dk_correction(correction)
    correction in _DK_CORRECTIONS || throw(ArgumentError(
        "unsupported Driscoll-Kraay correction $(repr(correction)); use :HC0, :HC1, or :sss"))
    return nothing
end

function _check_dk_panel(tis::Clustering, iis::Clustering)
    length(tis.groups) == length(iis.groups) || throw(DimensionMismatch(
        "DriscollKraay: `tis` has $(length(tis.groups)) observations but `iis` has $(length(iis.groups))"))
    Clustering(tis, iis).ngroups == length(tis.groups) || throw(ArgumentError(
        "DriscollKraay: some (period, unit) pairs in `tis` and `iis` occur more than once; " *
        "each unit can be observed at most once per period"))
    return nothing
end

"""
`HC0`

Alias for `HR0`. Basic heteroskedasticity-robust covariance estimator without finite-sample corrections.
See [`HR0`](@ref) for details.
"""
const HC0 = HR0

"""
`HC1`

Alias for `HR1`. Heteroskedasticity-robust estimator with degrees-of-freedom correction.
See [`HR1`](@ref) for details.
"""
const HC1 = HR1

"""
`HC2`

Alias for `HR2`. Heteroskedasticity-robust estimator with leverage-based corrections.
See [`HR2`](@ref) for details.
"""
const HC2 = HR2

"""
`HC3`

Alias for `HR3`. Heteroskedasticity-robust estimator with squared leverage corrections.
See [`HR3`](@ref) for details.
"""
const HC3 = HR3

"""
`HC4`

Alias for `HR4`. Heteroskedasticity-robust estimator with alternative leverage correction.
See [`HR4`](@ref) for details.
"""
const HC4 = HR4

"""
`HC4m`

Alias for `HR4m`. Modified version of HC4 with different leverage cutoff.
See [`HR4m`](@ref) for details.
"""
const HC4m = HR4m

"""
`HC5`

Alias for `HR5`. Heteroskedasticity-robust estimator with maximum leverage correction.
See [`HR5`](@ref) for details.
"""
const HC5 = HR5

# Names returned by the deprecated `String(::Type)` methods below.
_legacy_string(::Type{T}) where {T <: Truncated} = "Truncated"
_legacy_string(::Type{T}) where {T <: Parzen} = "Parzen"
_legacy_string(::Type{T}) where {T <: TukeyHanning} = "Tukey-Hanning"
_legacy_string(::Type{T}) where {T <: Bartlett} = "Bartlett"
_legacy_string(::Type{T}) where {T <: QuadraticSpectral} = "Quadratic Spectral"
_legacy_string(::Type{T}) where {T <: HR0} = "HR0"
_legacy_string(::Type{T}) where {T <: HR1} = "HR1"
_legacy_string(::Type{T}) where {T <: HR2} = "HR2"
_legacy_string(::Type{T}) where {T <: HR3} = "HR3"
_legacy_string(::Type{T}) where {T <: HR4} = "HR4"
_legacy_string(::Type{T}) where {T <: HR4m} = "HR4m"
_legacy_string(::Type{T}) where {T <: HR5} = "HR5"
_legacy_string(::Type{T}) where {T <: CR0} = "CR0"
_legacy_string(::Type{T}) where {T <: CR1} = "CR1"
_legacy_string(::Type{T}) where {T <: CR2} = "CR2"
_legacy_string(::Type{T}) where {T <: CR3} = "CR3"
_legacy_string(::Type{T}) where {T <: EWC} = "EWC"
_legacy_string(::Type{T}) where {T <: DriscollKraay} = "Driscoll-Kraay"
_legacy_string(::Type{T}) where {T <: VARHAC} = "VARHAC"

const _LegacyStringTypes = Union{
    Truncated, Parzen, TukeyHanning, Bartlett, QuadraticSpectral,
    HR0, HR1, HR2, HR3, HR4, HR4m, HR5, CR0, CR1, CR2, CR3, EWC, DriscollKraay, VARHAC}

function Base.String(T::Type{<:_LegacyStringTypes})
    Base.depwarn(
        "`String(::Type)` for estimator types is deprecated and will be removed in a " *
        "future release.", :String)
    return _legacy_string(T)
end

#=========
Display

Estimators print as constructor syntax that rebuilds an equal estimator. Cluster
assignments print as their integer group codes, which define the same partition as
the identifiers they were built from.
=========#

_show_groups(io::IO, g::Clustering) = show(io, g.groups)
_show_groups(io::IO, g) = show(io, g)

function _show_clusterspec(io::IO, g::Tuple)
    if length(g) == 1
        _show_groups(io, only(g))
    else
        print(io, "(")
        for (i, gi) in enumerate(g)
            i > 1 && print(io, ", ")
            _show_groups(io, gi)
        end
        print(io, ")")
    end
    return nothing
end

function Base.show(io::IO, k::Union{CR0, CR1, CR2, CR3, Cluster})
    print(io, nameof(typeof(k)), "(")
    _show_clusterspec(io, k.g)
    return print(io, ")")
end

function Base.show(io::IO, k::DriscollKraay)
    print(io, "DriscollKraay(")
    show(io, k.K)
    print(io, ", tis = ")
    _show_groups(io, k.tis)
    print(io, ", iis = ")
    _show_groups(io, k.iis)
    k.correction === :HC0 || print(io, ", correction = ", repr(k.correction))
    return print(io, ")")
end

function Base.show(io::IO, k::CachedCR{<:CR, <:CRCache{T}}) where {T}
    print(io, "CachedCR(")
    show(io, k.estimator)
    return print(io, ", ", k.cache.ncols, ", ", T, ")")
end

# The cache is built from a fitted model, which the estimator does not retain, so
# this form is a summary rather than constructor syntax.
function Base.show(io::IO, k::CachedCR{<:CR, <:CRModelCache})
    print(io, "CachedCR(")
    show(io, k.estimator)
    c = getfield(k, :cache)
    return print(io, ", <model cache: nobs = ", c.nobs, ", ncols = ", c.ncols, ">)")
end
