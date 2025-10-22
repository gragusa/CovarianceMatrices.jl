"""
Smith's Smoothed Moments Variance Estimation

This implementation follows Smith (2005, 2011) for smoothed moments variance estimation.
The method smooths moments first, then takes outer products to get automatically p.s.d. HAC estimators.

## Threading Performance Analysis

The implementation includes both single-threaded and multi-threaded smoothing options.
Threading performance depends heavily on sample size due to overhead costs.

Performance benchmarks (4 columns, 10 threads available):

| Sample Size (T) | Single-threaded (ms) | Multi-threaded (ms) | Speedup | Recommendation |
|-----------------|---------------------|---------------------|---------|----------------|
| 100             | 0.006               | 0.044              | 0.14x   | Use threaded=false |
| 400             | 0.032               | 0.053              | 0.61x   | Use threaded=false |
| 1000            | 0.103               | 0.068              | 1.52x   | Use threaded=true  |
| 10000           | 2.331               | 0.492              | 4.74x   | Use threaded=true  |

**Automatic Threading**:
The implementation automatically uses threading when T > 800, regardless of the `threaded` parameter.
- T ≤ 800: Single-threaded (unless `threaded=true` is explicitly set)
- T > 800: Multi-threaded automatically
- Use `threaded=true` to force threading for smaller samples
- Use `threaded=false` with manual threading control if needed

**Code used for benchmarking**:
```julia
using CovarianceMatrices, BenchmarkTools, Random
Random.seed!(123)
for T in [100, 400, 1000, 10000]
    X = randn(T, 4)
    sm_single = SmoothedMoments(threaded=false)
    sm_threaded = SmoothedMoments(threaded=true)
    t_single = @belapsed aVar(\$sm_single, \$X) samples=5 evals=3
    t_threaded = @belapsed aVar(\$sm_threaded, \$X) samples=5 evals=3
    speedup = t_single / t_threaded
    println("T=\$T: \$(round(speedup, digits=2))x speedup")
end
```
"""

# Kernel functions for smoothing (on observation scale)
abstract type SmoothingKernel end

"""
    UniformKernel()

Uniform/box kernel: k(x) = 1(|x| ≤ 1)
Induces Bartlett HAC kernel, optimal bandwidth S_T ∝ T^(1/3)
"""
struct UniformKernel <: SmoothingKernel
    m_T::Int64
    S_T::WFLOAT
    function UniformKernel(; m_T::Union{Int, Nothing}=nothing, S_T::Union{Int, Nothing}=nothing)
        if m_T === nothing && S_T === nothing
            throw(ArgumentError("Either m_T or S_T must be provided"))
        elseif m_T !== nothing && S_T !== nothing
            throw(ArgumentError("Only one of m_T or S_T should be provided"))
        elseif m_T !== nothing
            S_T_val = WFLOAT((2 * m_T + 1)/2)
            return new(m_T, S_T_val)
        else
            m_T_val = floor(Int, ((S_T - 1) / 2))
            return new(m_T_val, WFLOAT(S_T))
        end
    end
end

"""
    TriangularKernel()

Bartlett/triangular kernel: k(x) = (1 - |x|) * 1(|x| ≤ 1)
Induces Parzen HAC kernel, optimal bandwidth S_T ∝ T^(1/5)
"""
struct TriangularKernel <: SmoothingKernel
    m_T::Int64
    S_T::WFLOAT
    function TrianularKernel(; m_T::Union{Int, Nothing}=nothing, S_T::Union{Int, Nothing}=nothing)
        if m_T === nothing && S_T === nothing
            throw(ArgumentError("Either m_T or S_T must be provided"))
        elseif m_T !== nothing && S_T !== nothing
            throw(ArgumentError("Only one of m_T or S_T should be provided"))
        elseif m_T !== nothing
            S_T_val = WFLOAT((2 * m_T + 1)/2)
            return new(m_T, S_T_val)
        else
            m_T_val = floor(Int, ((S_T - 1) / 2))
            return new(m_T_val, WFLOAT(S_T))
        end
    end
end

lagtruncation(k::SmoothingKernel) = k.m_T
bandwidth(k::SmoothingKernel) = k.S_T



# Kernel function evaluations
@inline kernel_func(::UniformKernel, x::T) where {T <:
                                                  Real} = abs(x) ≤ one(T) ? one(T) : zero(T)
@inline kernel_func(::TriangularKernel, x::T) where {T <:
                                                     Real} = abs(x) ≤ one(T) ?
                                                             one(T) - abs(x) : zero(T)

# Kernel constants k₂ = ∫ k(x)² dx (precomputed for efficiency)
kernel_k1(::UniformKernel) = 2.0  # ∫k(a) da = 2
kernel_k2(::UniformKernel) = 2.0  # ∫k(a)² da = 2
kernel_k3(::UniformKernel) = 2.0  # ∫k(a)³ da = 2
kernel_k1(::TriangularKernel) = 1.0  # ∫k(a) da = 1
kernel_k2(::TriangularKernel) = 2.0/3.0  # ∫k(a)² da = 2/3
kernel_k3(::TriangularKernel) = 1.0/2.0  # ∫k(a) da = 1/2

"""
    SmoothedMoments{K<:SmoothingKernel} <: Correlated

Smith's smoothed moments variance estimator.

# Constructor
    SmoothedMoments(kernel::SmoothingKernel, bandwidth::Real; threaded::Bool=true)
    SmoothedMoments(kernel::SmoothingKernel; auto_bandwidth::Bool=true, threaded::Bool=true)

# Arguments
- `kernel`: Smoothing kernel (UniformKernel or TriangularKernel)
- `bandwidth`: Bandwidth S_T (if auto_bandwidth=false)
- `auto_bandwidth`: If true, uses optimal bandwidth scaling S_T = c * T^α where α depends on kernel
- `threaded`: If true, forces multithreaded smoothing. If false, uses single-threaded for small samples (T ≤ 800) and automatic threading for large samples (T > 800)

# References
- Smith, R.J. (2005). Automatic positive semidefinite HAC covariance matrix and GMM estimation
- Smith, R.J. (2011). GEL criteria for moment condition models
"""
# struct SmoothedMoments{K <: SmoothingKernel} <: Correlated
#     kernel::K
#     bandwidth::WFLOAT
#     auto_bandwidth::Bool
#     threaded::Bool

#     function SmoothedMoments(kernel::K, bandwidth::Real; threaded::Bool = true) where {K <:
#                                                                                        SmoothingKernel}
#         bandwidth > 0 || throw(ArgumentError("Bandwidth must be positive"))
#         new{K}(kernel, WFLOAT(bandwidth), false, threaded)
#     end

#     function SmoothedMoments(kernel::K; auto_bandwidth::Bool = true,
#             threaded::Bool = true) where {K <: SmoothingKernel}
#         new{K}(kernel, WFLOAT(0.0), auto_bandwidth, threaded)
#     end
# end

# # Convenience constructors
# function SmoothedMoments(bandwidth::Real; threaded::Bool = true)
#     SmoothedMoments(UniformKernel(), bandwidth; threaded = threaded)
# end
# function SmoothedMoments(; threaded::Bool = true)
#     SmoothedMoments(UniformKernel(); threaded = threaded)
# end

"""
    optimal_bandwidth(kernel::SmoothingKernel, T::Int) -> Float64

Compute optimal bandwidth for given kernel and sample size T.
"""
function optimal_bandwidth(::UniformKernel, T::Int)
    # Optimal rate T^(1/3) for uniform kernel
    return 2.0 * T^(1.0/3.0)
end

function optimal_bandwidth(::TriangularKernel, T::Int)
    # Optimal rate T^(1/5) for triangular kernel
    return 1.5 * T^(1.0/5.0)
end



function smooth_moments(G::AbstractMatrix{F}, kernel::SmoothingKernel) where {F <: AbstractFloat}
    T_data, m = size(G)
    result = similar(G, float(F), T_data, m)
    smooth_moments!(result, G, kernel)
    return result
end


function smooth_moments!(result::AbstractMatrix{F}, G::AbstractMatrix{F},
        kernel::UniformKernel) where {F <: AbstractFloat}
    T_data, m = size(G)
    @boundscheck size(result) == (T_data, m) ||
                 throw(DimensionMismatch("result and G must have same size"))

    S_T = bandwidth(kernel)
    m_T = lagtruncation(kernel)
    # For uniform kernel, max_lag is just the bandwidth
    max_lag = floor(Int, S_T)

    if max_lag == 0
        copyto!(result, G)
        return result
    end

    return uniform_smooth!(result, G, m_T)

end

function uniform_smooth!(out::AbstractMatrix, G::AbstractMatrix, m_T::Integer)
    T, m = size(G)
    @assert m_T ≥ 0 "m_T must be nonnegative"

    # Output element type (promote to floating)
    factor = 2 / float(2m_T + 1)
    for j in axes(G, 2)
        _col_uniform_smooth!(view(out, :, j), view(G, :, j), m_T, factor)
    end
    return out
end

# Internal: compute one column using a prefix-sum (O(T))
@inline function _col_uniform_smooth!(dest::AbstractVector, col::AbstractVector, mT::Int, factor)
    T = length(col)
    # Prefix sums P with P[k] = sum(col[1:k]) and P[0] = 0
    P = Vector{float(eltype(col))}(undef, T + 1)
    P[1] = zero(eltype(P))
    for i in 1:T
        P[i+1] = P[i] + col[i]
    end
    for t in eachindex(dest)
        a = max(1, t - mT)
        b = min(T, t + mT)
        # sum(col[a:b]) = P[b+1] - P[a]   (no 0 index!)
        dest[t] = factor * (P[b+1] - P[a])
    end
    return dest
end





# function smooth_moments!(result::AbstractMatrix{F}, G::AbstractMatrix{F},
#         kernel::TriangularKernel, bandwidth::Real, T::Int) where {F <: AbstractFloat}
#     T_data, m = size(G)
#     @boundscheck size(result) == (T_data, m) ||
#                  throw(DimensionMismatch("result and G must have same size"))

#     # For triangular kernel, max_lag is the bandwidth
#     max_lag = floor(Int, bandwidth)

#     if max_lag == 0
#         copyto!(result, G)
#         return result
#     end

#     fill!(result, zero(F))

#     # Process column by column for cache efficiency
#     @inbounds for j in 1:m
#         # Compute smoothed values
#         for t in 1:T_data
#             smooth_val = zero(F)

#             # Sum over the bandwidth window with triangular weights
#             for lag in (-max_lag):max_lag
#                 source_idx = t - lag
#                 if 1 ≤ source_idx ≤ T_data
#                     weight = (1 / bandwidth) * (1 - abs(lag) / bandwidth)  # Include 1/S_T factor
#                     smooth_val += weight * G[source_idx, j]
#                 end
#             end

#             result[t, j] = smooth_val
#         end
#     end

#     return result
# end

# # Fallback for complex kernels
# function smooth_moments!(result::AbstractMatrix{F}, G::AbstractMatrix{F},
#         kernel::SmoothingKernel, bandwidth::Real, T::Int) where {F <: AbstractFloat}
#     weights = compute_weights(kernel, bandwidth, T, F)
#     return smooth_moments!(result, G, weights, T)
# end

# # Old weight-based version for backward compatibility
# function smooth_moments!(result::AbstractMatrix{F}, G::AbstractMatrix{F},
#         weights::AbstractVector{F}, T::Int) where {F <: AbstractFloat}
#     T_data, m = size(G)
#     @boundscheck size(result) == (T_data, m) ||
#                  throw(DimensionMismatch("result and G must have same size"))

#     n_weights = length(weights)
#     offset = n_weights ÷ 2  # Center of weight vector

#     # Precompute non-zero weights and their lags to avoid checking in inner loop
#     nz_weights = F[]
#     nz_lags = Int[]
#     for i in eachindex(weights)
#         w = weights[i]
#         if w != zero(F)
#             push!(nz_weights, w)
#             push!(nz_lags, i - offset - 1)
#         end
#     end

#     # Calculate effective max lag based on actual non-zero weights
#     max_lag = if isempty(nz_lags)
#         0
#     else
#         max(abs(minimum(nz_lags)), abs(maximum(nz_lags)))
#     end

#     n_nz = length(nz_weights)

#     # Simple two-argument version - assumes result and G are different objects
#     # If they're the same, user should call the single-argument version instead
#     fill!(result, zero(F))
#     safe_start = max_lag + 1
#     safe_end = T_data - max_lag

#     # Column-major optimization: iterate by columns first for better cache locality
#     @inbounds for j in 1:m
#         if safe_start <= safe_end
#             # Head region: need bounds checking
#             for t in 1:(safe_start - 1)
#                 for k in 1:n_nz
#                     source_idx = t - nz_lags[k]
#                     if 1 ≤ source_idx ≤ T_data
#                         result[t, j] += nz_weights[k] * G[source_idx, j]
#                     end
#                 end
#             end

#             # Middle region: no bounds checking needed (fastest)
#             for t in safe_start:safe_end
#                 for k in 1:n_nz
#                     source_idx = t - nz_lags[k]
#                     result[t, j] += nz_weights[k] * G[source_idx, j]
#                 end
#             end

#             # Tail region: need bounds checking
#             for t in (safe_end + 1):T_data
#                 for k in 1:n_nz
#                     source_idx = t - nz_lags[k]
#                     if 1 ≤ source_idx ≤ T_data
#                         result[t, j] += nz_weights[k] * G[source_idx, j]
#                     end
#                 end
#             end
#         else
#             # No safe middle region - all points need bounds checking
#             for t in 1:T_data
#                 for k in 1:n_nz
#                     source_idx = t - nz_lags[k]
#                     if 1 ≤ source_idx ≤ T_data
#                         result[t, j] += nz_weights[k] * G[source_idx, j]
#                     end
#                 end
#             end
#         end
#     end
# end

# """
#     smooth_moments_threaded!(result::AbstractMatrix, G::AbstractMatrix, kernel::SmoothingKernel, bandwidth::Real, T::Int)

# Multithreaded version of smooth_moments! using kernel-based approach.
# Parallelizes computation column by column for cache efficiency.
# """
# function smooth_moments_threaded!(result::AbstractMatrix{F}, G::AbstractMatrix{F},
#         kernel::UniformKernel, bandwidth::Real, T::Int) where {F <: AbstractFloat}
#     T_data, m = size(G)
#     @boundscheck size(result) == (T_data, m) ||
#                  throw(DimensionMismatch("result and G must have same size"))

#     # For uniform kernel, max_lag is just the bandwidth
#     max_lag = floor(Int, bandwidth)

#     if max_lag == 0
#         copyto!(result, G)
#         return result
#     end

#     fill!(result, zero(F))

#     # Process columns in parallel for cache efficiency
#     Threads.@threads for j in 1:m
#         @inbounds for t in 1:T_data
#             smooth_val = zero(F)

#             # Sum over the bandwidth window with uniform weights (all = 1)
#             for lag in (-max_lag):max_lag
#                 source_idx = t - lag
#                 if 1 ≤ source_idx ≤ T_data
#                     smooth_val += G[source_idx, j]
#                 end
#             end

#             result[t, j] = smooth_val / bandwidth  # Normalize by bandwidth
#         end
#     end

#     return result
# end

# function smooth_moments_threaded!(result::AbstractMatrix{F}, G::AbstractMatrix{F},
#         kernel::TriangularKernel, bandwidth::Real, T::Int) where {F <: AbstractFloat}
#     T_data, m = size(G)
#     @boundscheck size(result) == (T_data, m) ||
#                  throw(DimensionMismatch("result and G must have same size"))

#     # For triangular kernel, max_lag is the bandwidth
#     max_lag = floor(Int, bandwidth)

#     if max_lag == 0
#         copyto!(result, G)
#         return result
#     end

#     fill!(result, zero(F))

#     # Process columns in parallel for cache efficiency
#     Threads.@threads for j in 1:m
#         @inbounds for t in 1:T_data
#             smooth_val = zero(F)

#             # Sum over the bandwidth window with triangular weights
#             for lag in (-max_lag):max_lag
#                 source_idx = t - lag
#                 if 1 ≤ source_idx ≤ T_data
#                     weight = (1 / bandwidth) * (1 - abs(lag) / bandwidth)  # Include 1/S_T factor
#                     smooth_val += weight * G[source_idx, j]
#                 end
#             end

#             result[t, j] = smooth_val
#         end
#     end

#     return result
# end

# # Fallback for complex kernels
# function smooth_moments_threaded!(result::AbstractMatrix{F}, G::AbstractMatrix{F},
#         kernel::SmoothingKernel, bandwidth::Real, T::Int) where {F <: AbstractFloat}
#     weights = compute_weights(kernel, bandwidth, T, F)
#     return smooth_moments_threaded!(result, G, weights, T)
# end

# # Old weight-based version for backward compatibility
# function smooth_moments_threaded!(result::AbstractMatrix{F}, G::AbstractMatrix{F},
#         weights::AbstractVector{F}, T::Int) where {F <: AbstractFloat}
#     T_data, m = size(G)
#     @boundscheck size(result) == (T_data, m) ||
#                  throw(DimensionMismatch("result and G must have same size"))

#     n_weights = length(weights)
#     offset = n_weights ÷ 2  # Center of weight vector

#     fill!(result, zero(F))

#     # Precompute non-zero weights and their lags to avoid checking in inner loop
#     nz_weights = F[]
#     nz_lags = Int[]
#     for i in eachindex(weights)
#         w = weights[i]
#         if w != zero(F)
#             push!(nz_weights, w)
#             push!(nz_lags, i - offset - 1)
#         end
#     end

#     # Calculate effective max lag based on actual non-zero weights
#     max_lag = if isempty(nz_lags)
#         0
#     else
#         max(abs(minimum(nz_lags)), abs(maximum(nz_lags)))
#     end

#     # Find the range where no bounds checking is needed
#     safe_start = max_lag + 1
#     safe_end = T_data - max_lag

#     @inbounds for j in axes(G, 2)
#         if safe_start <= safe_end
#             # We have three regions: head, middle (safe), tail

#             # Head region: need bounds checking (single-threaded)
#             for t in 1:(safe_start - 1)
#                 for (w, lag) in zip(nz_weights, nz_lags)
#                     source_idx = t - lag
#                     if 1 ≤ source_idx ≤ T_data
#                         result[t, j] += w * G[source_idx, j]
#                     end
#                 end
#             end

#             # Middle region: no bounds checking needed - THREADED!
#             Threads.@threads for t in safe_start:safe_end
#                 for (w, lag) in zip(nz_weights, nz_lags)
#                     source_idx = t - lag
#                     result[t, j] += w * G[source_idx, j]
#                 end
#             end

#             # Tail region: need bounds checking (single-threaded)
#             for t in (safe_end + 1):T_data
#                 for (w, lag) in zip(nz_weights, nz_lags)
#                     source_idx = t - lag
#                     if 1 ≤ source_idx ≤ T_data
#                         result[t, j] += w * G[source_idx, j]
#                     end
#                 end
#             end
#         else
#             # No safe middle region - all points need bounds checking (single-threaded)
#             for t in 1:T_data
#                 for (w, lag) in zip(nz_weights, nz_lags)
#                     source_idx = t - lag
#                     if 1 ≤ source_idx ≤ T_data
#                         result[t, j] += w * G[source_idx, j]
#                     end
#                 end
#             end
#         end
#     end
# end

# """
#     smooth_moments(G::AbstractMatrix, kernel::SmoothingKernel, bandwidth::Real, T::Int) -> Matrix

# Out-of-place smoothing of moments matrix G using kernel-based approach.
# """
# function smooth_moments(G::AbstractMatrix{F}, kernel::SmoothingKernel,
#         bandwidth::Real, T::Int) where {F <: AbstractFloat}
#     result = similar(G)
#     smooth_moments!(result, G, kernel, bandwidth, T)
#     return result
# end

# # Old weight-based version for backward compatibility
# function smooth_moments(G::AbstractMatrix{F}, weights::AbstractVector{F}, T::Int) where {F <:
#                                                                                          AbstractFloat}
#     result = similar(G)
#     smooth_moments!(result, G, weights, T)
#     return result
# end

# """
#     smooth_moments_threaded(G::AbstractMatrix, kernel::SmoothingKernel, bandwidth::Real, T::Int) -> Matrix

# Out-of-place threaded smoothing of moments matrix G using kernel-based approach.
# """
# function smooth_moments_threaded(G::AbstractMatrix{F}, kernel::SmoothingKernel,
#         bandwidth::Real, T::Int) where {F <: AbstractFloat}
#     result = similar(G)
#     smooth_moments_threaded!(result, G, kernel, bandwidth, T)
#     return result
# end

# # Old weight-based version for backward compatibility
# function smooth_moments_threaded(
#         G::AbstractMatrix{F}, weights::AbstractVector{F}, T::Int) where {F <: AbstractFloat}
#     result = similar(G)
#     smooth_moments_threaded!(result, G, weights, T)
#     return result
# end

# """
#     compute_weights(kernel::SmoothingKernel, S_T::Real, T::Int) -> Vector{Float64}

# Compute discrete smoothing weights wₛ = (1/S_T) * k(s/S_T) for s ∈ {-(T-1), ..., T-1}.
# """
# function compute_weights(kernel::SmoothingKernel, S_T::Real, T::Int,
#         ::Type{F} = WFLOAT) where {F <: AbstractFloat}
#     # Weight indices: s = -(T-1), ..., -1, 0, 1, ..., T-1
#     n_weights = 2 * T - 1
#     weights = Vector{F}(undef, n_weights)
#     S_T_F = F(S_T)

#     @inbounds for i in eachindex(weights)
#         s = i - T  # Convert to lag: -(T-1)...(T-1)
#         weights[i] = (one(F) / S_T_F) * F(kernel_func(kernel, F(s) / S_T_F))
#     end

#     return weights
# end

# """
#     compute_normalization(kernel::SmoothingKernel, weights::AbstractVector, S_T::Real; discrete::Bool=true) -> Float64

# Compute normalization constant c for variance estimation.
# - If discrete=true: c = 1 / Σₛ k(s/S_T)²
# - If discrete=false: c = S_T / k₂ where k₂ = ∫ k(x)² dx
# """
# function compute_normalization(
#         kernel::SmoothingKernel, weights::AbstractVector, S_T::Real; discrete::Bool = true)
#     if discrete
#         # Discrete normalization: c = 1 / Σ wₛ²
#         weight_sum_sq = sum(abs2, weights)
#         return weight_sum_sq > 0 ? 1.0 / weight_sum_sq : 1.0
#     else
#         # Continuous normalization: c = S_T / k₂
#         k2 = kernel_k2(kernel)
#         return S_T / k2
#     end
# end

"""
    avar(estimator::SmoothedMoments, X::AbstractMatrix{F}; prewhite::Bool=false) where {F<:Real}

Main implementation of Smith's smoothed moments variance estimator.
Supports optional prewhitening which can improve finite sample performance.
"""
function avar(kernel::K, X::AbstractMatrix{F}; prewhite::Bool = false) where {F <: Real, K <: Union{UniformKernel, TriangularKernel}}
    # Apply prewhitening if requested (using same approach as HAC)
    Z, D = finalize_prewhite(X, Val(prewhite))
    T, m = size(Z)

    # Determine bandwidth (use prewhitened data size if different)
    S_T = bandwidth(kernel)

    # Smooth the (possibly prewhitened) moments using kernel-based approach
    # Use threading automatically for large samples (T > 800) or if explicitly requested
    #use_threading = estimator.threaded || T > 800
    G_smoothed = smooth_moments(Z, kernel)
    k₂ = kernel_k2(kernel)
    # For kernel-based approach, we still need normalization constant for discrete case
    # We compute it from the kernel and bandwidth for consistency
    #weights = compute_weights(estimator.kernel, S_T, T, F)  # Only used for normalization
    #c = compute_normalization(estimator.kernel, weights, S_T; discrete = true)

    # Compute variance: Ω̂ = c * (G^T)' * G^T (scaling by T handled by aVar)
    V = Matrix{F}(undef, m, m)
    mul!(V, G_smoothed', G_smoothed, 1/(k₂*T), 0.0)  # V = (G_smoothed' * G_smoothed) / k2

    # Transform back if prewhitened: V_final = (I - D')^(-1) * V * (I - D')^(-1)'
    if prewhite
        v = inv(one(F)*I - D')
        V = v * V * v'
    end

    return V
end

# # For backward compatibility, keep old smoothers but mark as deprecated
# abstract type AbstractSmoother <: AbstractAsymptoticVarianceEstimator end

# struct IdentitySmoother <: AbstractSmoother end
# IdentitySmoother(args...) = IdentitySmoother()
# (k::IdentitySmoother)(G) = G

# """
#     TruncatedSmoother(S::Real)

# DEPRECATED: Use SmoothedMoments(UniformKernel(), S) instead.
# Legacy truncated smoother for backward compatibility.
# """
# struct TruncatedSmoother <: AbstractSmoother
#     ξ::Int
#     S::WFLOAT
#     κ::Vector{WFLOAT}

#     function TruncatedSmoother(S::Real)
#         S > 0 || throw(ArgumentError("The bandwidth must be positive"))
#         ξ = floor(Int, (S * 2 - 1) / 2)
#         return new(ξ, WFLOAT(S), WFLOAT[2.0, 2.0, 2.0])
#     end
# end

# """
#     BartlettSmoother(S::Real)

# DEPRECATED: Use SmoothedMoments(TriangularKernel(), S) instead.
# Legacy Bartlett smoother for backward compatibility.
# """
# struct BartlettSmoother <: AbstractSmoother
#     ξ::Int
#     S::WFLOAT
#     κ::Vector{WFLOAT}

#     function BartlettSmoother(S::Real)
#         S > 0 || throw(ArgumentError("The bandwidth must be positive"))
#         ξ = floor(Int, (S * 2 - 1) / 2)
#         return new(ξ, WFLOAT(S), WFLOAT[1.0, 2.0 / 3.0, 0.5])
#     end
# end

# # Legacy implementations (for backward compatibility only)
# inducedkernel(x::Type{TruncatedSmoother}) = Bartlett
# inducedkernel(x::Type{BartlettSmoother}) = Parzen

# bw(s::AbstractSmoother) = s.S
# κ₁(s::AbstractSmoother) = s.κ[1]
# κ₂(s::AbstractSmoother) = s.κ[2]
# ξ(s::AbstractSmoother) = s.ξ

# # Legacy smoother implementations (deprecated - these don't implement Smith's method correctly)
# function (s::TruncatedSmoother)(G::Matrix)
#     @warn "TruncatedSmoother is deprecated. Use SmoothedMoments(UniformKernel(), $(s.S)) instead." maxlog=1
#     N, M = size(G)
#     nG = zeros(WFLOAT, N, M)
#     b = bw(s)
#     xi = ξ(s)
#     for m in axes(G, 2)
#         for t in axes(G, 1)
#             low = max((t - N), -xi)::Int
#             high = min(t - 1, xi)::Int
#             for s_lag in low:high
#                 @inbounds nG[t, m] += G[t - s_lag, m]
#             end
#         end
#     end
#     return nG ./ b
# end

# function (s::BartlettSmoother)(G::Matrix)
#     @warn "BartlettSmoother is deprecated. Use SmoothedMoments(TriangularKernel(), $(s.S)) instead." maxlog=1
#     N, M = size(G)
#     b = bw(s)
#     xi = ξ(s)
#     nG = zeros(WFLOAT, N, M)
#     for m in axes(G, 2)
#         for t in axes(G, 1)
#             low = max((t - N), -xi)::Int
#             high = min(t - 1, xi)::Int
#             for s_lag in low:high
#                 κ = 1 - abs(s_lag / b)
#                 nG[t, m] += κ * G[t - s_lag, m]
#             end
#         end
#     end
#     return nG
# end

# # Legacy avar methods for old smoothers (these are broken - they don't follow Smith's method)
# function avar(k::Union{BartlettSmoother, TruncatedSmoother}, X; kwargs...)
#     @warn "Legacy smoother avar methods are deprecated and do not implement Smith's method correctly. Use SmoothedMoments instead." maxlog=1
#     n, p = size(X)
#     sm = k(X)
#     V = (sm'sm) ./ k.κ[2]
#     return V
# end
