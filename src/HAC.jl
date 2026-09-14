"""
    avar(k::HAC, X::AbstractMatrix; prewhite=false)

Compute the asymptotic variance-covariance matrix using a Heteroskedasticity
and Autocorrelation Consistent (HAC) estimator.

# Arguments
- `k::HAC`: HAC kernel estimator (e.g., Bartlett, Parzen, QuadraticSpectral)
- `X::AbstractMatrix{F}`: Data matrix of size T×p where T is the sample size
  and p is the number of variables
- `prewhite::Bool=false`: If true, pre-whiten the data using VAR(1) before
  applying the HAC estimator

# Returns
- `Matrix{F}`: The asymptotic variance-covariance matrix of size p×p

# Details
The function computes V̂ = (I - D')⁻¹ V (I - D')⁻¹' where V is the kernel-weighted
long-run covariance matrix and D is the VAR(1) coefficient matrix (if prewhitening).

# Examples
```julia
X = randn(100, 3)
k = Bartlett()
V = avar(k, X; prewhite=true)
```
"""
function avar(k::K, X::AbstractMatrix{F}; prewhite = false) where {K <: HAC, F <: Real}
    V, _ = avar_with_info(k, X; prewhite = prewhite)
    return V
end

"""
    avar_with_info(k::HAC, X::AbstractMatrix; prewhite=false)

Compute the HAC estimate together with the quantities selected from the data.

Returns `(V, info)` where `info` is a `NamedTuple` carrying the `bandwidth` used and
the `kernelweights` that entered the bandwidth selection. `aVar` puts `info` into
the returned [`CovarianceMatrix`](@ref).
"""
function avar_with_info(
        k::K, X::AbstractMatrix{F}; prewhite = false,
        weights = nothing) where {K <: HAC, F <: Real}
    Z, D = finalize_prewhite(X, Val(prewhite))
    T, p = size(Z)
    kw = weights === nothing ? kernelweights(k, X) : weights
    bw = _optimalbandwidth(k, Z, kw, prewhite)
    V = zeros(F, p, p)
    Q = similar(V)
    kernelestimator!(k, V, Q, Z, bw)
    v = inv(one(F)*I - D')
    return v * V * v', (bandwidth = bw, kernelweights = kw)
end

"""
    finalize_prewhite(X, ::Val{true})
    finalize_prewhite(X, ::Val{false})

Conditionally pre-whiten data matrix X using VAR(1).

# Arguments
- `X::AbstractMatrix`: Data matrix
- `Val{true}`: Perform pre-whitening via VAR(1) fit
- `Val{false}`: Skip pre-whitening

# Returns
- When pre-whitening: Tuple of (residuals, coefficient matrix) from VAR(1)
- When not pre-whitening: Tuple of (X, ZeroMat())
"""
finalize_prewhite(X, ::Val{true}) = fit_var(X)
finalize_prewhite(X, ::Val{false}) = X, ZeroMat()

"""
    ZeroMat

Singleton type representing a zero matrix for efficient computation.
Supports operations with UniformScaling without allocation.
"""
struct ZeroMat end
Base.:-(J::UniformScaling, Z::ZeroMat) = J
Base.:+(J::UniformScaling, Z::ZeroMat) = J
LinearAlgebra.adjoint(Z::ZeroMat) = Z

"""
    kernelestimator!(k::HAC, V::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix, bw)

Compute the HAC variance estimator in-place using kernel weighting.

# Arguments
- `k::HAC`: Kernel specification
- `V::AbstractMatrix{F}`: Output matrix for variance estimate (modified in-place)
- `Q::AbstractMatrix`: Temporary workspace matrix
- `Z::AbstractMatrix`: Data matrix of size T×p

# Details
Computes: V̂ = Γ₀ + Σⱼ κ(j/bw) * (Γⱼ + Γⱼ')

where:
- Γⱼ is the sample autocovariance at lag j
- κ(·) is the kernel function
- bw is the bandwidth

"""
function kernelestimator!(
        k::K, V::AbstractMatrix{F}, Q, Z, bw_in) where {K <: HAC, F <: Real}
    T, _ = size(Z)
    bw = convert(F, bw_in)
    idx = covindices(k, T, bw)

    # Pre-allocate kernel weights
    κ = Vector{F}(undef, length(idx))
    bw_inv = inv(bw)  # Compute once
    @inbounds for j in eachindex(idx)
        κ[j] = kernel(k, idx[j] * bw_inv)
    end

    # Calculate the variance at lag 0
    mul!(V, Z', Z)
    #copy!(V, Q)

    # Calculate Γ₁, Γ₂, ..., Γⱼ with symmetric contribution
    @inbounds for j in eachindex(idx)
        Zₜ = view(Z, 1:(T - idx[j]), :)
        Zₜ₊ⱼ = view(Z, (1 + idx[j]):T, :)
        mul!(Q, Zₜ', Zₜ₊ⱼ)

        # Combine symmetric terms
        κⱼ = κ[j]
        @. V += κⱼ * (Q + Q')
    end

    return V
end

"""
    avarscaler(K::HAC, X; prewhite=false)

Return the scaling factor for asymptotic variance computation.

# Arguments
- `K::HAC`: Kernel estimator
- `X::AbstractMatrix`: Data matrix
- `prewhite::Bool=false`: Pre-whitening flag

# Returns
- Sample size T (number of rows in X)
"""
@inline avarscaler(K::HAC, X; prewhite = false) = size(X, 1)

"""
    covindices(k::HAC, n::Int, bw)

Determine the lag indices to use in covariance computation based on kernel type.

# Arguments
- `k::HAC`: Kernel specification
- `n::Int`: Sample size
- `bw`: Bandwidth

# Returns
- Range or collection of lag indices

# Details
Different kernels use different lag ranges:
- `QuadraticSpectral`: All lags 1:n (infinite support)
- `Bartlett`: Truncated at bandwidth
- `HR`: No lags (0 lags)
"""
covindices(k::T, n, bw) where {T <: QuadraticSpectral} = 1:n
covindices(k::HAC, n, bw) = 1:floor(Int, bw)
covindices(k::T, n, bw) where {T <: HR} = 1:0
# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------
"""
    kernel(k::Truncated, x::Real)

Truncated (uniform) kernel: K(x) = 𝟙(|x| ≤ 1)

Returns 1 if |x| ≤ 1, otherwise 0.
"""
@inline kernel(k::Truncated, x::Real) = (abs(x) <= 1) ? one(x) : zero(x)

"""
    kernel(k::Bartlett, x::Real)

Bartlett (triangular) kernel: K(x) = (1 - |x|) 𝟙(|x| < 1)

Also known as the Newey-West kernel.
"""
@inline kernel(k::Bartlett, x::Real) = (abs(x) < 1) ? (one(x) - abs(x)) : zero(x)

"""
    kernel(k::TukeyHanning, x::Real)

Tukey-Hanning kernel: K(x) = (1 + cos(πx))/2 𝟙(|x| ≤ 1)
"""
@inline function kernel(k::TukeyHanning, x::Real)
    return (abs(x) <= 1) ? (one(x) + cospi(x)) / 2 : zero(x)
end

"""
    kernel(k::Parzen, x::Real)

Parzen kernel:
- K(x) = 1 - 6x² + 6|x|³  for |x| ≤ 1/2
- K(x) = 2(1-|x|)³         for 1/2 < |x| ≤ 1
- K(x) = 0                 for |x| > 1
"""
@inline function kernel(k::Parzen, x::Real)
    ax = abs(x)
    return ax <= 1 / 2 ? one(x) - 6 * ax^2 + 6 * ax^3 :
           ax <= 1 ? 2 * one(x) * (1 - ax)^3 : zero(x)
end

"""
    kernel(k::QuadraticSpectral, x::Real)

Quadratic Spectral kernel:
K(x) = (25/12π²z²) * [sin(z)/z - cos(z)]

where z = 6πx/5. This kernel has infinite support.
"""
@inline function kernel(k::QuadraticSpectral, x::Real)
    z = (6 * π / 5) * x  # Reduced one multiplication
    iszero(x) && return one(x)  # Handle x=0 case
    sz = sin(z)
    return 3 * (sz / z - cos(z)) / (z * z)
end

"""
    kernelweights(k::HAC{Union{Andrews,NeweyWest}}, X::AbstractMatrix)

Per-column weights entering the data-driven bandwidth selection for `X`.

A column that is constant carries no information about serial correlation and gets
weight `0`; every other column gets weight `1`.
"""
function kernelweights(k::HAC{T}, m::RegressionModel) where {T <:
                                                             Union{Andrews, NeweyWest}}
    return kernelweights(k, modelmatrix(m))
end

function kernelweights(k::HAC{T}, X::AbstractMatrix) where {T <:
                                                            Union{Andrews, NeweyWest}}
    w = Vector{WFLOAT}(undef, size(X, 2))
    for (i, col) in zip(eachindex(w), eachcol(X))
        w[i] = allequal(col) ? 0.0 : 1.0
    end
    return w
end

"""
    kernelweights(k::HAC{Fixed}, X)
    kernelweights(k::AbstractAsymptoticVarianceEstimator, X)

Estimators that do not select a bandwidth from the data use no kernel weights.
"""
kernelweights(k::HAC{T}, X) where {T <: Fixed} = nothing
kernelweights(k::AbstractAsymptoticVarianceEstimator, X) = nothing

# -----------------------------------------------------------------------------
# Optimal bandwidth
# -----------------------------------------------------------------------------
"""
    workingoptimalbw(k::HAC{Union{Andrews,NeweyWest}}, m::AbstractMatrix; prewhite=false)

Internal function to compute optimal bandwidth with pre-whitening.

# Returns
- Tuple: (processed data matrix, VAR coefficient matrix, bandwidth)
"""
function workingoptimalbw(
        k::HAC{T},
        A::AbstractMatrix;
        prewhite::Bool = false
) where {T <: Union{Andrews, NeweyWest}}
    X, D = prewhiter(A, prewhite)
    kw = kernelweights(k, X)
    bw = _optimalbandwidth(k, X, kw, prewhite)
    return X, D, bw
end

"""
    workingoptimalbw(k::HAC{Fixed}, m::AbstractMatrix; kwargs...)

For fixed bandwidth kernels, return data and pre-set bandwidth.
"""
function workingoptimalbw(k::HAC{T}, m::AbstractMatrix; kwargs...) where {T <: Fixed}
    return (m, Matrix{eltype(m)}(undef, 0, 0), k.bw)
end

"""
    optimalbw(k::HAC{Union{Andrews,NeweyWest}}, m::AbstractMatrix;
              demean=false, dims=1, means=nothing, prewhite=false)

Calculate the optimal bandwidth for HAC estimation.

# Arguments
- `k::HAC`: Kernel estimator (Andrews or NeweyWest)
- `m::AbstractMatrix`: Data matrix
- `demean::Bool=false`: Remove means before bandwidth calculation
- `dims::Int=1`: Dimension along which to compute means (if demeaning)
- `means::Union{Nothing,AbstractArray}=nothing`: Pre-computed means (optional)
- `prewhite::Bool=false`: Apply VAR(1) pre-whitening

# Returns
- Optimal bandwidth value

# Methods
- **Andrews (1991)**: Uses AR(1) approximation with kernel-specific constants
- **Newey-West (1994)**: Uses lag-window with data-driven bandwidth

# References
Andrews, D.W.K. (1991). Heteroskedasticity and Autocorrelation Consistent
Covariance Matrix Estimation. Econometrica, 59(3), 817-858.

Newey, W.K. and West, K.D. (1994). Automatic Lag Selection in Covariance
Matrix Estimation. Review of Economic Studies, 61(4), 631-653.
"""
function optimalbw(
        k::HAC{T},
        m::AbstractMatrix;
        demean::Bool = false,
        dims::Int = 1,
        means::Union{Nothing, AbstractArray} = nothing,
        prewhite::Bool = false
) where {T <: Union{Andrews, NeweyWest}}
    X = demean ? demeaner(m; means = means, dims = dims) : m
    _, _, bw = workingoptimalbw(k, X; prewhite = prewhite)
    return bw
end

"""
    optimalbw(k::HAC{Fixed}, m::AbstractMatrix; kwargs...)

Return the bandwidth the kernel was constructed with. A fixed bandwidth does not depend
on the data, so `m` and the keyword arguments are ignored.

# Examples
```julia
optimalbw(Bartlett(4), X)   # 4.0
```
"""
function optimalbw(k::HAC{T}, m::AbstractMatrix; kwargs...) where {T <: Fixed}
    return k.bw
end

function _optimalbandwidth(k::HAC{T}, mm, w, prewhite) where {T <: NeweyWest}
    return bwNeweyWest(k, mm, w, prewhite)
end

function _optimalbandwidth(k::HAC{T}, mm, w, prewhite) where {T <: Andrews}
    return bwAndrews(k, mm, w, prewhite)
end

_optimalbandwidth(k::HAC{T}, mm, w, prewhite) where {T <: Fixed} = k.bw

"""
    bwAndrews(k::HAC, mm::AbstractMatrix, w, prewhite::Bool)

Compute Andrews (1991) optimal bandwidth using AR(1) approximation.

# Formula
bw = cₖ * (α̂₂ * n)^(1/5)

where cₖ is kernel-specific constant and α̂₂ depends on AR(1) parameters.
"""
function bwAndrews(k::HAC, mm, w, prewhite::Bool)
    n, p = size(mm)
    a1, a2 = getalpha(k, mm, w)
    return bw_andrews(k, a1, a2, n)
end

"""
    bwNeweyWest(k::HAC, mm::AbstractMatrix, w, prewhite::Bool)

Compute Newey-West (1994) automatic bandwidth selection.

# Formula
bw = cₖ * (ŝ₁/ŝ₀)^(2/3) * n^(1/3)  [for Bartlett]
bw = cₖ * (ŝ₂/ŝ₀)^(2/5) * n^(1/5)  [for Parzen, QS]

where ŝⱼ are lag-weighted autocovariances.
"""
function bwNeweyWest(k::HAC, mm, w, prewhite::Bool)
    n, _ = size(mm)
    l = getrates(k, mm, prewhite)
    xm = mm * w
    a = Vector{eltype(xm)}(undef, l + 1)
    @inbounds for j in 0:l
        a[j + 1] = dot(
            view(xm, firstindex(xm):(lastindex(xm) - j)),
            view(xm, (j + firstindex(xm)):lastindex(xm))
        ) / n
    end
    aa = view(a, 2:(l + 1))
    a0 = a[1] + 2 * sum(aa)
    a1 = 2 * sum((1:l) .* aa)
    a2 = 2 * sum((1:l) .^ 2 .* aa)
    return bwnw(k, a0, a1, a2) * (n + prewhite)^growthrate(k)
end

## ---> Andrews Optimal bandwidth <---
"""
Dictionary of kernel-specific bandwidth constants for Andrews method.
"""
d_bw_andrews = Dict(
    :Truncated => :(0.6611 * (a2 * n)^(0.2)),
    :Bartlett => :(1.1447 * (a1 * n)^(1 / 3)),
    :Parzen => :(2.6614 * (a2 * n)^(0.2)),
    :TukeyHanning => :(1.7462 * (a2 * n)^(0.2)),
    :QuadraticSpectral => :(1.3221 * (a2 * n)^(0.2))
)

for kerneltype in kernels
    @eval $:(bw_andrews)(k::($kerneltype), a1, a2, n) = $(d_bw_andrews[kerneltype])
end

"""
    getalpha(k::HAC, mm::AbstractMatrix, w)

Compute α₁ and α₂ parameters for Andrews bandwidth selection.

# Returns
- Tuple: (α₁, α₂) based on AR(1) approximation

# Formula
α₁ = Σ wⱼ * 4ρⱼ²σⱼ⁴ / [(1-ρⱼ)⁶(1+ρⱼ)²] / Σ wⱼ * σⱼ⁴/(1-ρⱼ)⁴
α₂ = Σ wⱼ * 4ρⱼ²σⱼ⁴ / (1-ρⱼ)⁸ / Σ wⱼ * σⱼ⁴/(1-ρⱼ)⁴
"""
function getalpha(k, mm, w)
    rho, σ⁴ = fit_ar(mm)

    # Pre-compute common terms
    one_minus_rho = @. 1.0 - rho
    one_plus_rho = @. 1.0 + rho
    rho_sq = @. rho^2

    # Compute α₁
    nm = @. 4.0 * rho_sq * σ⁴ / (one_minus_rho^6 * one_plus_rho^2)
    dn = @. σ⁴ / one_minus_rho^4
    α₁ = dot(w, nm) / dot(w, dn)

    # Compute α₂
    nm = @. 4.0 * rho_sq * σ⁴ / one_minus_rho^8
    α₂ = dot(w, nm) / dot(w, dn)

    return α₁, α₂
end

"""
    getrates(k::HAC, mm::AbstractMatrix, prewhite::Bool)

Determine lag truncation parameter for Newey-West bandwidth selection.

# Returns
- Integer lag length based on sample size and kernel type
"""
function getrates(k, mm, prewhite::Bool)
    n, _ = size(mm)
    lrate = lagtruncation(k)
    adj = prewhite ? 3 : 4
    return floor(Int, adj * ((n + prewhite) / 100)^lrate)
end

@inline bwnw(k::BartlettKernel, s0, s1, s2) = 1.1447 * ((s1 / s0)^2)^growthrate(k)
@inline bwnw(k::ParzenKernel, s0, s1, s2) = 2.6614 * ((s2 / s0)^2)^growthrate(k)
@inline bwnw(k::QuadraticSpectralKernel, s0, s1, s2) = 1.3221 * ((s2 / s0)^2)^growthrate(k)

## --> Newey-West Optimal bandwidth <---
@inline growthrate(k::HAC) = 1 / 5
@inline growthrate(k::BartlettKernel) = 1 / 3

@inline lagtruncation(k::BartlettKernel) = 2 / 9
@inline lagtruncation(k::ParzenKernel) = 4 / 25
@inline lagtruncation(k::QuadraticSpectralKernel) = 2 / 25

"""
    allequal(x::AbstractVector)

Check if all elements in vector x are equal.

# Returns
- `true` if all elements equal, `false` otherwise

# Performance
- Short-circuits on first inequality
- Handles length < 2 efficiently
"""
function allequal(x)
    lx = length(x)
    lx < 2 && return true
    e1 = x[1]
    @inbounds for i in 2:lx
        x[i] == e1 || return false
    end
    return true
end

# -----------------------------------------------------------------------------
# Fit function
# -----------------------------------------------------------------------------
Base.@propagate_inbounds function fit_var(A::AbstractMatrix{T}) where {T}
    fi = firstindex(A, 1)
    li = lastindex(A, 1)
    Y = view(A, (fi + 1):li, :)
    X = view(A, fi:(li - 1), :)
    B = cholesky(X'X) \ X'Y
    E = Y - X * B
    return E, B
end

"""
    fit_ar(Z::AbstractMatrix)

Fit AR(1) models separately for each column: yₜⱼ = ρⱼ yₜ₋₁ⱼ + εₜⱼ

# Arguments
- `Z::AbstractMatrix{T}`: Data matrix of size T×p

# Returns
- Tuple: (ρ, σ⁴) where
  - ρ: Vector of AR(1) coefficients (length p)
  - σ⁴: Vector of fourth powers of residual standard deviations (length p)

"""
function fit_ar(Z::AbstractMatrix{T}) where {T}
    A = parent(Z)
    n, p = size(A)
    rho = Vector{T}(undef, p)
    σ⁴ = similar(rho)
    xy = Vector{T}(undef, n - 1)

    for j in axes(A, 2)
        y = A[2:lastindex(A, 1), j]
        x = A[1:(lastindex(A, 1) - 1), j]

        # Handle constant columns
        allequal(x) && (rho[j] = 0; σ⁴[j] = 0; continue)

        # PERFORMANCE: In-place centering
        x_mean = mean(x)
        y_mean = mean(y)

        @simd for i in eachindex(xy, x, y)
            xy[i] = (x[i] - x_mean) * (y[i] - y_mean)
        end

        # Compute AR coefficient
        x_var = sum(i -> (x[i] - x_mean)^2, eachindex(x))
        rho[j] = sum(xy) / x_var

        # Compute residual variance
        resid_ss = zero(T)
        @simd for i in eachindex(xy, x, y)
            resid = (y[i] - y_mean) - rho[j] * (x[i] - x_mean)
            resid_ss += resid^2
        end
        σ⁴[j] = (resid_ss / (n - 1))^2
    end

    return rho, σ⁴
end

# -----------------------------------------------------------------------------
# Prewhiter
# -----------------------------------------------------------------------------
function prewhiter(M::AbstractMatrix{T}, prewhite::Bool) where {T <: Real}
    if prewhite
        return fit_var(M)
    else
        if eltype(M) ∈ (Float32, Float64)
            return (M::Matrix{T}, Matrix{T}(undef, 0, 0))
        else
            return (float(M), zeros(0, 0))
        end
    end
end
