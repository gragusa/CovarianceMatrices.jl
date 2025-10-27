"""
Smith's Smoothed Moments Variance Estimation

This implementation follows Smith (2005, 2011) for smoothed moments variance estimation.
The method smooths moments first, then takes outer products to get automatically p.s.d. HAC estimators.

References
----------
 Smith, R. J. (2005). "AUTOMATIC POSITIVE SEMIDEFINITE HAC COVARIANCE MATRIX AND GMM ESTIMATION." Econometric Theory, 21, 158-170.

"""

# Kernel functions for smoothing (on observation scale)
abstract type MomentSmoother <: AbstractAsymptoticVarianceEstimator end

"""
    UniformSmoother(m_T::Integer)

Uniform (box) kernel smoother for Smith's smoothed moments HAC estimation.

The uniform kernel is defined as k(x) = 1 if |x| ≤ 1, and 0 otherwise.
This induces a Bartlett HAC kernel in the final variance estimator.

# Arguments
- `m_T::Integer`: Bandwidth parameter (must be a non-negative integer).
  The smoothing window size is 2m_T + 1.

# Optimal Bandwidth
The optimal bandwidth scales as S_T ∝ T^(1/3) where T is the sample size.
Use `optimal_bandwidth(UniformSmoother(0), T)` to compute the optimal value.

# Examples
```julia
using CovarianceMatrices

# Create a uniform smoother with bandwidth parameter m_T = 5
smoother = UniformSmoother(5)

# Generate some moment data (T × k matrix)
G = randn(100, 3)

# Smooth the moments
G_smooth = smooth_moments(G, smoother)

# Compute HAC variance estimate
Ω = aVar(smoother, G)
```

# References
Smith, R. J. (2005). "Automatic Positive Semidefinite HAC Covariance Matrix and GMM Estimation."
Econometric Theory, 21, 158-170.
"""
struct UniformSmoother <: MomentSmoother
    m_T::Int
    function UniformSmoother(m_T::Real)
        if !(m_T ≥ 0)
            throw(ArgumentError("m_T must be positive"))
        end
        if !isinteger(m_T)
            throw(ArgumentError("m_T must be a positive integer"))
        end
        return new(Int(m_T))
    end
end

"""
    TriangularSmoother(m_T::Integer)

Triangular (Bartlett) kernel smoother for Smith's smoothed moments HAC estimation.

The triangular kernel is defined as k(x) = (1 - |x|) if |x| ≤ 1, and 0 otherwise.
This induces a Parzen HAC kernel in the final variance estimator.

# Arguments
- `m_T::Integer`: Bandwidth parameter (must be a non-negative integer).
  The smoothing window size is 2m_T + 1.

# Optimal Bandwidth
The optimal bandwidth scales as S_T ∝ T^(1/5) where T is the sample size.
Use `optimal_bandwidth(TriangularSmoother(0), T)` to compute the optimal value.

# Examples
```julia
using CovarianceMatrices

# Create a triangular smoother with bandwidth parameter m_T = 5
smoother = TriangularSmoother(5)

# Generate some moment data (T × k matrix)
G = randn(100, 3)

# Smooth the moments
G_smooth = smooth_moments(G, smoother)

# Compute HAC variance estimate
Ω = aVar(smoother, G)
```

# References
Smith, R. J. (2005). "Automatic Positive Semidefinite HAC Covariance Matrix and GMM Estimation."
Econometric Theory, 21, 158-170.
"""
struct TriangularSmoother <: MomentSmoother
    m_T::Int
    function TriangularSmoother(m_T::Real)
        if !(m_T ≥ 0)
            throw(ArgumentError("m_T must be positive"))
        end
        if !isinteger(m_T)
            throw(ArgumentError("m_T must be a positive integer"))
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

function k1hat(k::MomentSmoother)
    mT = k.m_T
    [kernel_func(k, s) for s in (-mT - 2):(mT + 2)]
end

function k2hat(k::MomentSmoother)
    mT = k.m_T
    sum(abs2, (kernel_func(k, s) for s in (-mT):mT))
end

function k3hat(k::MomentSmoother)
    mT = k.m_T
    sum((kernel_func(k, s)^3 for s in (-mT - 2):(mT + 2)))
end

# Kernel constants k₂ = ∫ k(x)² dx (precomputed for efficiency)
kernel_k1(::UniformSmoother) = 2.0     # ∫k(a) da = 2
kernel_k2(::UniformSmoother) = 2.0     # ∫k(a)² da = 2
kernel_k3(::UniformSmoother) = 2.0     # ∫k(a)³ da = 2
kernel_k1(::TriangularSmoother) = 1.0  # ∫k(a) da = 1
kernel_k2(::TriangularSmoother) = 2 / 3  # ∫k(a)² da = 2/3
kernel_k3(::TriangularSmoother) = 1 / 2  # ∫k(a) da = 1/2

"""
    optimal_bandwidth(kernel::MomentSmoother, T::Int) -> Float64

Compute optimal bandwidth for given kernel and sample size T.
"""
function optimal_bandwidth(::UniformSmoother, T::Int)
    # Optimal rate T^(1/3) for uniform kernel
    return 2.0 * T^(1.0 / 3.0)
end

function optimal_bandwidth(::TriangularSmoother, T::Int)
    # Optimal rate T^(1/5) for triangular kernel
    return 1.5 * T^(1.0 / 5.0)
end

"""
    smooth_moments(G::AbstractMatrix, kernel::MomentSmoother) -> AbstractMatrix

Apply kernel-based smoothing to moment matrix G.

This function implements Smith's (2005, 2011) smoothed moments approach for HAC estimation.
Each row of G represents moment conditions at time t, and smoothing is applied to produce
a smoothed moment matrix that automatically yields positive semi-definite variance estimates.

# Arguments
- `G::AbstractMatrix`: T × k matrix of moment conditions, where T is sample size and k is
  the number of moment conditions
- `kernel::MomentSmoother`: Smoother object (UniformSmoother or TriangularSmoother) with
  bandwidth parameter m_T

# Returns
- Smoothed moment matrix of the same size as G

# Performance
The implementation uses prefix sums for O(T) complexity per column, making it efficient
even for large sample sizes. For very large matrices, consider using the in-place version
`smooth_moments!` to reduce allocations.

# Examples
```julia
using CovarianceMatrices

# Generate moment data
T, k = 100, 3
G = randn(T, k)

# Uniform kernel smoothing
smoother_u = UniformSmoother(5)
G_smooth_u = smooth_moments(G, smoother_u)

# Triangular kernel smoothing
smoother_t = TriangularSmoother(5)
G_smooth_t = smooth_moments(G, smoother_t)
```

# References
Smith, R. J. (2011). "GEL Criteria for Moment Condition Models."
Econometric Theory, 27(6), 1192-1235.
"""
function smooth_moments(
        G::AbstractMatrix, kernel::T; threaded::Bool = false) where {T <: UniformSmoother}
    return uniform_sum(G, kernel.m_T)
end

"""
    smooth_moments!(dest::AbstractMatrix, G::AbstractMatrix, kernel::MomentSmoother) -> AbstractMatrix

In-place version of `smooth_moments`. Stores the result in `dest`.

This function applies kernel-based smoothing to moment matrix G, storing the result in the
pre-allocated destination matrix `dest`. Use this for performance-critical code to avoid
allocations.

# Arguments
- `dest::AbstractMatrix`: Pre-allocated T × k matrix to store the smoothed moments
- `G::AbstractMatrix`: T × k matrix of moment conditions to be smoothed
- `kernel::MomentSmoother`: Smoother object with bandwidth parameter m_T

# Returns
- The destination matrix `dest` containing smoothed moments

# Examples
```julia
using CovarianceMatrices

# Generate moment data
T, k = 100, 3
G = randn(T, k)

# Pre-allocate destination
G_smooth = similar(G)

# In-place smoothing
smoother = UniformSmoother(5)
smooth_moments!(G_smooth, G, smoother)
```
"""
function smooth_moments!(dest::AbstractMatrix, G::AbstractMatrix, kernel::T;
        threaded::Bool = false) where {T <: UniformSmoother}
    return uniform_sum!(dest, G, kernel.m_T)
end

function smooth_moments(G::AbstractMatrix, kernel::T;
        threaded::Bool = false) where {T <: TriangularSmoother}
    return triangular_sum(G, kernel.m_T)
end

function smooth_moments!(dest::AbstractMatrix, G::AbstractMatrix, kernel::T;
        threaded::Bool = false) where {T <: TriangularSmoother}
    return triangular_sum!(dest, G, kernel.m_T)
end

using Base.Threads

# ================
# UNIFORM WINDOW
# ================
"""
    uniform_sum(G::AbstractMatrix, m_T::Integer) -> AbstractMatrix

Computes sum_{s=max(t-T,-m_T)}^{min(t-1,m_T)} G[t-s, j] using prefix sums for O(T) per column.
"""
function uniform_sum!(
        dest::AbstractMatrix{T}, G::AbstractMatrix{<:Int}, m_T) where {T <: Real}
    uniform_sum!(dest, float.(G), m_T)
end

function uniform_sum!(dest::AbstractMatrix{T}, G::AbstractMatrix{T}, m_T) where {T <: Real}
    n, m = size(G)
    @assert size(dest)==(n, m) "Destination matrix must have the same size as G, ($n, $m)"

    P = Vector{T}(undef, n + 1)  # Single thread: one P vector is sufficient
    for j in axes(G, 2)
        _col_uniform_sum!(view(dest, :, j), view(G, :, j), m_T, P)
    end

    return dest
end

function uniform_sum(G::AbstractMatrix{<:Int}, m_T)
    uniform_sum(float.(G), m_T)
end

function uniform_sum(G::AbstractMatrix{T}, m_T) where {T <: Real}
    dest = similar(G)
    return uniform_sum!(dest, G, m_T)
end

# One column, O(T) using prefix sums
function _col_uniform_sum!(
        dest::AbstractVector{T}, col::AbstractVector{T}, m_T, P) where {T <: Real}
    n = length(col)
    mT = Int(m_T)
    # Build prefix sum
    P[1] = zero(T)
    @inbounds for i in eachindex(col)
        P[i + 1] = P[i] + col[i]
    end
    # Compute windowed sums
    @inbounds for t in eachindex(col)
        a = max(1, t - mT)
        b = min(n, t + mT)
        dest[t] = P[b + 1] - P[a]
    end
    return dest
end

# ==========================
# TRIANGULAR (UNSCALED)
# ==========================
# Computes sum_{s=max(t-T,-m_T)}^{min(t-1,m_T)} (1 - 2s/D) * G[t-s, j]
# which equals sum_{k=a}^{b} (1 - 2(t-k)/D) * G[k, j]
# with D = 2m_T + 1. No outer factor 2/D is applied here.

"""
    triangular_sum(G::AbstractMatrix, m_T::Integer; threaded::Bool = true) -> AbstractMatrix

Computes triangular kernel smoothing using prefix sums for O(T) per column.
"""
function triangular_sum!(
        dest::AbstractMatrix{T}, G::AbstractMatrix{<:Int}, m_T) where {T <: Real}
    triangular_sum!(dest, float.(G), m_T)
end

function triangular_sum!(
        dest::AbstractMatrix{T}, G::AbstractMatrix{T}, m_T) where {T <: Real}
    n, m = size(G)
    @assert size(dest)==(n, m) "Destination matrix must have the same size as G, ($n, $m)"

    P = Vector{T}(undef, n + 1)
    W = Vector{T}(undef, n + 1)
    for j in axes(G, 2)
        _col_triangular_sum_fma!(view(dest, :, j), view(G, :, j), m_T, P, W)
    end

    return dest
end

function triangular_sum(G::AbstractMatrix{<:Int}, m_T)
    triangular_sum(float.(G), m_T)
end

function triangular_sum(G::AbstractMatrix{T}, m_T) where {T <: Real}
    dest = similar(G)
    return triangular_sum!(dest, G, m_T)
end

# One column, O(T) using two prefix sum arrays
function _col_triangular_sum!(
        dest::AbstractVector{T}, col::AbstractVector{T}, m_T, P, W) where {T <: Real}
    n = length(col)
    mT = Int(m_T)
    scale = 2.0 / (2 * mT + 1)

    # Build prefix sums: P for values, W for index-weighted values
    P[1] = zero(T)
    W[1] = zero(T)
    @inbounds for i in 1:n
        P[i + 1] = P[i] + col[i]
        W[i + 1] = W[i] + i * col[i]
    end

    # Compute windowed triangular sums
    @inbounds for t in 1:n
        a = max(1, t - mT)
        b = min(n, t + mT)
        # Left part: indices [a, t-1] with weights (1 - scale*(t-i))
        # sum_{i=a}^{t-1} (1 - scale*(t-i)) * G[i]
        # = sum G[i] - scale * sum (t-i)*G[i]
        # = sum G[i] - scale * (t*sum(G[i]) - sum(i*G[i]))
        left_sum = P[t] - P[a]
        left_weighted = t * left_sum - (W[t] - W[a])
        left_contrib = left_sum - scale * left_weighted
        # Center: index t with weight 1
        center_contrib = col[t]
        # Right part: indices [t+1, b] with weights (1 - scale*(i-t))
        # sum_{i=t+1}^{b} (1 - scale*(i-t)) * G[i]
        # = sum G[i] - scale * sum (i-t)*G[i]
        # = sum G[i] - scale * (sum(i*G[i]) - t*sum(G[i]))
        right_sum = P[b + 1] - P[t + 1]
        right_weighted = (W[b + 1] - W[t + 1]) - t * right_sum
        right_contrib = right_sum - scale * right_weighted
        dest[t] = left_contrib + center_contrib + right_contrib
    end
    return dest
end

function _col_triangular_sum_fma!(
        dest::AbstractVector{T}, col::AbstractVector{T}, m_T, P, W) where {T <: Real}
    n = length(col)
    mT = Int(m_T)
    scale = float(2) / float(2 * mT + 1)

    # Build prefix sums
    P[1] = zero(T)
    W[1] = zero(T)
    @inbounds for i in 1:n
        P[i + 1] = P[i] + col[i]
        W[i + 1] = W[i] + i * col[i]
    end

    @inbounds for t in 1:n
        a = max(1, t - mT)
        b = min(n, t + mT)

        Pa = P[a]
        Pt = P[t]
        Pb1 = P[b + 1]

        Wa = W[a]
        Wt = W[t]
        Wb1 = W[b + 1]

        # Left contribution using muladd
        left_sum = Pt - Pa
        left_weighted = Wt - Wa
        left = muladd(-scale * t, left_sum, left_sum + scale * left_weighted)

        # Right contribution using muladd
        right_sum = Pb1 - P[t + 1]
        right_weighted = Wb1 - W[t + 1]
        right = muladd(scale * t, right_sum, right_sum - scale * right_weighted)

        dest[t] = left + col[t] + right
    end

    return dest
end

"""
    avar(kernel::MomentSmoother, X::AbstractMatrix; prewhite::Bool=false) -> Matrix

Compute the asymptotic variance matrix using Smith's smoothed moments HAC estimator.

This estimator smooths moments first, then computes outer products, automatically yielding
a positive semi-definite HAC covariance matrix. This is superior to traditional HAC estimators
which can produce non-PSD matrices in finite samples.

# Arguments
- `kernel::MomentSmoother`: Smoother object (UniformSmoother or TriangularSmoother) specifying
  the kernel type and bandwidth parameter m_T
- `X::AbstractMatrix`: T × k matrix of moment conditions or score contributions
- `prewhite::Bool=false`: If true, apply VAR(1) prewhitening before smoothing (can improve
  finite sample performance for highly autocorrelated data)

# Returns
- k × k asymptotic variance matrix Ω̂

# Algorithm
1. Optionally prewhiten X using VAR(1) fit
2. Smooth the (possibly prewhitened) moments using the specified kernel
3. Compute variance as V = (1/k₂) * G'G where G is smoothed moments and k₂ is kernel constant
4. Transform back if prewhitened: V = (I - D')^(-1) V (I - D')^(-1)'

# Examples
```julia
using CovarianceMatrices

# Generate moment data
T, k = 500, 3
X = randn(T, k)

# Compute variance with uniform smoother
smoother_u = UniformSmoother(5)
Ω_u = aVar(smoother_u, X)

# Compute variance with triangular smoother and prewhitening
smoother_t = TriangularSmoother(5)
Ω_t = aVar(smoother_t, X; prewhite=true)
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
    # Apply prewhitening if requested (using same approach as HAC)
    Z, D = finalize_prewhite(X, Val(prewhite))
    T, m = size(Z)

    # Smooth the (possibly prewhitened) moments using kernel-based approach
    # Use threading automatically for large samples (T > 800) or if explicitly requested
    G_smoothed = smooth_moments(Z, k)

    ## The normaliation is k₂*S_T where
    k₂ = k2hat(k)
    #sT = S_T(k)
    # Compute variance: Ω̂ = c * (G^T)' * G^T (scaling by T handled by aVar)
    V = Matrix{float(F)}(undef, m, m)
    mul!(V, G_smoothed', G_smoothed, 1 / (k₂), 0.0)

    # Transform back if prewhitened: V_final = (I - D')^(-1) * V * (I - D')^(-1)'
    if prewhite
        v = inv(one(F) * I - D')
        V = v * V * v'
    end

    return V
end

"""
    smooth_uniform_plain(G, m_T)

Smooth a T×k matrix G using uniform kernel weights.

The smoothing formula is:
G_smooth[t,j] = Σ_{s=t-T}^{t-1} k(2s/(2m_T+1)) * G[t-s,j]

where k(x) = 1 if |x| ≤ 1, and 0 otherwise (uniform kernel).

Arguments:
- G: T×k matrix to be smoothed
- m_T: positive integer controlling the smoothing window size

Returns:
- G_smooth: smoothed matrix with uniform kernel weights
"""
function smooth_uniform_plain(G::Matrix{Float64}, m_T::Int)
    T, k = size(G)
    G_smooth = zeros(T, k)

    for t in 1:T
        for j in 1:k
            # Sum from s = t-T to s = t-1
            for s in (t - T):(t - 1)
                idx = t - s  # Index in original matrix (must be in 1:T)

                # Check if index is valid
                if 1 <= idx <= T
                    # Compute kernel argument
                    x = 2 * s / (2 * m_T + 1)

                    # Uniform kernel: k(x) = 1 if |x| <= 1, else 0
                    if abs(x) <= 1.0
                        G_smooth[t, j] += G[idx, j]
                    end
                end
            end
        end
    end

    return G_smooth
end

"""
    smooth_triangular_plain(G, m_T)

Smooth a T×k matrix G using triangular kernel weights.

The smoothing formula is:
G_smooth[t,j] = Σ_{s=t-T}^{t-1} k(2s/(2m_T+1)) * G[t-s,j]

where k(x) = 1-|x| if |x| ≤ 1, and 0 otherwise (triangular kernel).

Arguments:
- G: T×k matrix to be smoothed
- m_T: positive integer controlling the smoothing window size

Returns:
- G_smooth: smoothed matrix with triangular kernel weights
"""
function smooth_triangular_plain(G::Matrix{Float64}, m_T::Int)
    T, k = size(G)
    G_smooth = zeros(T, k)

    for t in 1:T
        for j in 1:k
            # Sum from s = t-T to s = t-1
            for s in (t - T):(t - 1)
                idx = t - s  # Index in original matrix (must be in 1:T)

                # Check if index is valid
                if 1 <= idx <= T
                    # Compute kernel argument
                    x = 2 * s / (2 * m_T + 1)

                    # Triangular kernel: k(x) = 1-|x| if |x| <= 1, else 0
                    if abs(x) <= 1.0
                        weight = 1.0 - abs(x)
                        G_smooth[t, j] += weight * G[idx, j]
                    end
                end
            end
        end
    end

    return G_smooth
end

"""
    smooth_uniform_plain2(G, m_T)

Smooth a T×k matrix G using uniform kernel weights.

Arguments:
- G: T×k matrix to be smoothed
- m_T: positive integer controlling the smoothing window size

Returns:
- G_smooth: smoothed matrix with uniform kernel weights
"""
function smooth_uniform_plain2(G::Matrix{Float64}, m_T::Int)
    T, k = size(G)
    G_smooth = zeros(T, k)

    for t in 1:T
        for j in 1:k
            # Only sum where kernel is non-zero: |s| <= m_T
            s_min = max(t - T, -m_T)
            s_max = min(t - 1, m_T)

            for s in s_min:s_max
                idx = t - s
                # Uniform kernel weight is 1 (no need to check condition)
                G_smooth[t, j] += G[idx, j]
            end
        end
    end

    return G_smooth
end

"""
    smooth_triangular_plain2(G, m_T)

Smooth a T×k matrix G using triangular kernel weights.

Arguments:
- G: T×k matrix to be smoothed
- m_T: positive integer controlling the smoothing window size

Returns:
- G_smooth: smoothed matrix with triangular kernel weights
"""
function smooth_triangular_plain2(G::Matrix{Float64}, m_T::Int)
    T, k = size(G)
    G_smooth = zeros(T, k)

    for t in 1:T
        for j in 1:k
            # Only sum where kernel is non-zero: |s| <= m_T
            s_min = max(t - T, -m_T)
            s_max = min(t - 1, m_T)

            for s in s_min:s_max
                idx = t - s
                # Triangular kernel weight
                x = 2 * s / (2 * m_T + 1)
                weight = 1.0 - abs(x)
                G_smooth[t, j] += weight * G[idx, j]
            end
        end
    end

    return G_smooth
end
