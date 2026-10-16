"""
    MomentSmoother

Abstract supertype for Smith's smoothed-moments estimators, [`UniformSmoother`](@ref)
and [`TriangularSmoother`](@ref). They smooth the moments first and then take outer
products, which gives a positive semidefinite HAC estimate.

`MomentSmoother` is a direct subtype of `AbstractAsymptoticVarianceEstimator`, not of
`Correlated`.

# References
Smith, R. J. (2005). "Automatic positive semidefinite HAC covariance matrix and GMM
estimation." Econometric Theory, 21, 158-170.
"""
abstract type MomentSmoother <: AbstractAsymptoticVarianceEstimator end

"""
    UniformSmoother(m_T::Integer)

Uniform kernel smoother for Smith's smoothed moments HAC estimation.

The uniform kernel is defined as k(x) = 1 if |x| ≤ 1, and 0 otherwise.
The induced HAC kernel is Bartlett: up to end effects, `UniformSmoother(m_T)` gives the
same estimate as `Bartlett(2m_T + 1)`.

# Arguments
- `m_T::Integer`: Bandwidth parameter (must be a non-negative integer).
  The smoothing window size is 2m_T + 1.
"""
struct UniformSmoother <: MomentSmoother
    m_T::Int
    function UniformSmoother(m_T::Real)
        if !(m_T ≥ 0)
            throw(ArgumentError("m_T must be non-negative"))
        end
        if !isinteger(m_T)
            throw(ArgumentError("m_T must be an integer"))
        end
        return new(Int(m_T))
    end
end

"""
    TriangularSmoother(m_T::Integer)

Triangular (Bartlett) kernel smoother for Smith's smoothed moments HAC estimation.

The triangular kernel is defined as k(x) = (1 - |x|) if |x| ≤ 1, and 0 otherwise.
The induced HAC kernel is approximately Parzen: `TriangularSmoother(m_T)` gives an
estimate close to `Parzen(2m_T + 1)`.

# Arguments
- `m_T::Integer`: Bandwidth parameter (must be a non-negative integer).
  The smoothing window size is 2m_T + 1.
"""
struct TriangularSmoother <: MomentSmoother
    m_T::Int
    function TriangularSmoother(m_T::Real)
        if !(m_T ≥ 0)
            throw(ArgumentError("m_T must be non-negative"))
        end
        if !isinteger(m_T)
            throw(ArgumentError("m_T must be an integer"))
        end
        return new(Int(m_T))
    end
end

S_T(k::MomentSmoother) = float((2 * k.m_T + 1) / 2)

# Kernel function evaluations
# Calculate k(s/S_T) for the uniform kernel
function kernel_func(k::UniformSmoother, s::T) where {T <: Real}
    x = s / S_T(k)
    abs(x) ≤ one(T) ? one(T) : zero(T)
end

function kernel_func(k::TriangularSmoother, s::T) where {T <: Real}
    x = s / S_T(k)
    abs(x) ≤ one(T) ? one(T) - abs(x) : zero(T)
end

# ∑ₛ k(s/S_T)² over the window |s| ≤ m_T, the normalization of the smoothed outer products.
function k2hat(k::MomentSmoother)
    mT = k.m_T
    sum(abs2, (kernel_func(k, s) for s in (-mT):mT))
end

"""
    optimalbw(kernel::MomentSmoother, G::AbstractMatrix) -> Float64

Return the bandwidth the smoothing kernel's rate rule selects for the moment matrix `G`
(observations in rows).

Unlike the HAC kernel methods of `optimalbw`, a moment smoother's rate rule depends on
the data only through the number of observations `T = size(G, 1)`.

# Examples
```julia
G = randn(1000, 2)
optimalbw(UniformSmoother(0), G)      # 2 * 1000^(1/3)
optimalbw(TriangularSmoother(0), G)   # 1.5 * 1000^(1/5)
```
"""
optimalbw(k::MomentSmoother, G::AbstractMatrix) = _ratebw(k, size(G, 1))

# Optimal rate T^(1/3) for the uniform kernel
_ratebw(::UniformSmoother, T::Integer) = 2.0 * T^(1.0 / 3.0)
# Optimal rate T^(1/5) for the triangular kernel
_ratebw(::TriangularSmoother, T::Integer) = 1.5 * T^(1.0 / 5.0)

"""
    smooth_moments(G::AbstractMatrix, kernel::MomentSmoother) -> AbstractMatrix

Return the smoothed moments of the T × k moment matrix `G` (observations in rows).

Row `t` of the result is `∑ₛ k(s/S_T) G[t-s, :]`, summed over `|s| ≤ m_T` with
`1 ≤ t-s ≤ T`, where `k` is the smoother's kernel and `S_T = (2m_T + 1)/2`. The factor
`1/S_T` of Smith (2005) is not applied; `avar` normalizes the outer products by
`∑ₛ k(s/S_T)²` instead.

Each column takes O(T) operations whatever the bandwidth. The result has a floating-point
element type; `smooth_moments!` writes into a preallocated matrix.

# Examples
```julia
G = randn(100, 3)
smooth_moments(G, UniformSmoother(5))
smooth_moments(G, TriangularSmoother(5))
```
"""
function smooth_moments(G::AbstractMatrix, kernel::MomentSmoother)
    return smooth_moments!(similar(G, float(eltype(G))), G, kernel)
end

"""
    smooth_moments!(dest::AbstractMatrix, G::AbstractMatrix, kernel::MomentSmoother) -> dest

Store the smoothed moments of `G` in `dest`, which must have the size of `G`. See
`smooth_moments`.

# Examples
```julia
G = randn(100, 3)
G_smooth = similar(G)
smooth_moments!(G_smooth, G, UniformSmoother(5))
```
"""
function smooth_moments!(dest::AbstractMatrix, G::AbstractMatrix, kernel::MomentSmoother)
    Base.require_one_based_indexing(dest, G)
    size(dest) == size(G) || throw(DimensionMismatch(
        "`dest` has size $(size(dest)) but `G` has size $(size(G))"))
    n = size(G, 1)
    n == 0 && return dest
    R = promote_type(float(eltype(G)), eltype(dest))
    buffer = Vector{R}(undef, _bufferlength(kernel, n))
    for j in axes(G, 2)
        _smooth_column!(view(dest, :, j), view(G, :, j), kernel, buffer)
    end
    return dest
end

# Both kernels read the prefix sums P[i] = ∑ col[1:i-1] from a buffer padded with
# P[i] = 0 for i < 1 and P[i] = P[n+1] for i > n + 1, which treats col as zero outside
# 1:n. The window half-width is h = min(m, n - 1): a wider window covers no further
# observations. Without padding, clamping each index to 1:(n + 1) makes the loops
# several times slower.
function _prefixsums!(buffer, col, h)
    n = length(col)
    # buffer[i + h + 1] holds P[i] for i in -h:(length(buffer) - h - 1).
    o = h + 1
    fill!(view(buffer, 1:(o + 1)), zero(eltype(buffer)))
    for i in 1:n
        buffer[i + o + 1] = buffer[i + o] + col[i]
    end
    fill!(view(buffer, (n + o + 2):length(buffer)), buffer[n + o + 1])
    return buffer
end

# Uniform window: dest[t] = ∑_{|s|≤h} col[t+s] = P[t+h+1] - P[t-h].
_bufferlength(k::UniformSmoother, n) = n + 2min(k.m_T, n - 1) + 2

function _smooth_column!(dest, col, kernel::UniformSmoother, buffer)
    n = length(col)
    h = min(kernel.m_T, n - 1)
    _prefixsums!(buffer, col, h)
    o = h + 1
    for t in 1:n
        dest[t] = buffer[t + h + 1 + o] - buffer[t - h + o]
    end
    return dest
end

# Triangular window: the weights are k(s/S_T) = (w - 2|s|)/w with w = 2m + 1. The
# unscaled sum y_t = ∑_{|s|≤h} (w - 2|s|) col[t+s] satisfies
#
#     y_{t+1} - y_t = P[t+h+1] + P[t+h+2] - 4P[t+1] + P[t-h+1] + P[t-h].
#
# For h < m the window covers every observation, and the terms whose coefficient
# depends on m are zero. The recursion only accumulates rounding errors of the size of
# the window sums. A prefix sum weighted by the observation index, ∑ i⋅col[i], grows
# like n^1.5 and loses that many digits to cancellation.
_bufferlength(k::TriangularSmoother, n) = n + 2min(k.m_T, n - 1) + 3

function _smooth_column!(dest, col, kernel::TriangularSmoother, buffer)
    n = length(col)
    m = kernel.m_T
    h = min(m, n - 1)
    R = eltype(buffer)
    _prefixsums!(buffer, col, h)
    o = h + 1
    w = 2m + 1
    y = zero(R)
    for i in 1:(h + 1)
        y += (w - 2(i - 1)) * col[i]
    end
    c = inv(R(w))
    dest[1] = c * y
    for t in 1:(n - 1)
        y += (buffer[t + h + 1 + o] + buffer[t + h + 2 + o]) - 4buffer[t + 1 + o] +
             (buffer[t - h + 1 + o] + buffer[t - h + o])
        dest[t + 1] = c * y
    end
    return dest
end

"""
    avar(kernel::MomentSmoother, X::AbstractMatrix; prewhite::Bool=false) -> Matrix

Compute the unscaled asymptotic variance matrix of the moments `X` (observations in
rows) with Smith's smoothed-moments estimator. [`aVar`](@ref) calls this method after
demeaning and divides the result by the number of observations.

The estimator smooths the moments first and then takes outer products, so the estimate
is positive semidefinite for every bandwidth.

# Arguments
- `kernel::MomentSmoother`: Smoother object (UniformSmoother or TriangularSmoother) specifying
  the kernel type and bandwidth parameter m_T
- `X::AbstractMatrix`: T × k matrix of moment conditions or score contributions
- `prewhite::Bool=false`: If true, apply VAR(1) prewhitening before smoothing (can improve
  finite sample performance for highly autocorrelated data)

# Returns
- k × k matrix `G'G / k̂₂`

# Algorithm
1. Optionally prewhiten X using VAR(1) fit
2. Smooth the (possibly prewhitened) moments with `smooth_moments`
3. Compute `V = G'G / k̂₂`, where `G` is the smoothed moments and
   `k̂₂ = ∑_{|s|≤m_T} k(s/S_T)²`
4. Transform back if prewhitened: V = (I - D')^(-1) V (I - D')^(-1)'

# Examples
```julia
using CovarianceMatrices

X = randn(500, 3)
Ω_u = aVar(UniformSmoother(5), X)
Ω_t = aVar(TriangularSmoother(5), X; prewhite=true)
```

# References
Smith, R. J. (2005). "Automatic Positive Semidefinite HAC Covariance Matrix and GMM Estimation."
Econometric Theory, 21, 158-170.

Smith, R. J. (2011). "GEL Criteria for Moment Condition Models."
Econometric Theory, 27(6), 1192-1235.
"""
function avar(
        k::MomentSmoother, X::AbstractMatrix{F}; prewhite::Bool = false) where {F <:
                                                                                Real}
    Z, D = finalize_prewhite(X, Val(prewhite))
    G = smooth_moments(Z, k)
    V = G' * G
    rdiv!(V, k2hat(k))

    # Transform back if prewhitened: V_final = (I - D')^(-1) * V * (I - D')^(-1)'
    if prewhite
        v = inv(one(F) * I - D')
        V = v * V * v'
    end

    return V
end
