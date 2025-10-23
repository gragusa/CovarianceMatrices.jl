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
UniformSmoother()

Uniform/box kernel: k(x) = 1(|x| ≤ 1)
Induces Bartlett HAC kernel, thus optimal bandwidth S_T ∝ T^(1/3)
"""
struct UniformSmoother{T <: AbstractFloat} <: MomentSmoother
    m_T::T
    S_T::T
end

function UniformSmoother(; m_T::Union{Real, Nothing} = nothing, S_T::Union{Real, Nothing} = nothing)
    if m_T === nothing && S_T === nothing
        throw(ArgumentError("Either m_T or S_T must be provided"))
    elseif m_T !== nothing && S_T !== nothing
        throw(ArgumentError("Only one of m_T or S_T should be provided"))
    elseif m_T !== nothing
        S_T_val = float((2 * m_T + 1)/2)
        return UniformSmoother(float(m_T), S_T_val)
    else
        m_T_val = float(floor(Int, (2S_T - 1) / 2))
        return UniformSmoother(m_T_val, float(S_T))
    end
end

"""
    TriangularSmoother()

Bartlett/triangular kernel: k(x) = (1 - |x|) * 1(|x| ≤ 1)
Induces Parzen HAC kernel, optimal bandwidth S_T ∝ T^(1/5)
"""
struct TriangularSmoother{T <: Real} <: MomentSmoother
    m_T::T
    S_T::T
end

function TriangularSmoother(; m_T::Union{Real, Nothing} = nothing, S_T::Union{
        Real, Nothing} = nothing)
    if m_T === nothing && S_T === nothing
        throw(ArgumentError("Either m_T or S_T must be provided"))
    elseif m_T !== nothing && S_T !== nothing
        throw(ArgumentError("Only one of m_T or S_T should be provided"))
    elseif m_T !== nothing
        S_T_val = float((2 * m_T + 1)/2)
        return TriangularSmoother(float(m_T), S_T_val)
    else
        m_T_val = float(floor(Int, (2S_T - 1) / 2))
        return TriangularSmoother(m_T_val, float(S_T))
    end
end

lagtruncation(k::MomentSmoother) = k.m_T
bandwidth(k::MomentSmoother) = k.S_T

# Kernel function evaluations
# Calculate k(s/S_T) for the uniform kernel
function kernel_func(k::UniformSmoother, s::T) where {T <: Real}
    x = s/k.S_T
    abs(x) ≤ one(T) ? one(T) : zero(T)
end

function kernel_func(k::TriangularSmoother, s::T) where {T <: Real}
    x = s/k.S_T
    abs(x) ≤ one(T) ? one(T) - abs(x) : zero(T)
end

function kernel_func(k, s::AbstractVector)
    return map(s) do si
        kernel_func(k, si)
    end
end

function kernel_func(k, s::AbstractRange)
    w = float(eltype(s))[]
    for j in s
        push!(w, kernel_func(k, j))
    end
    return w
end

# Calculate k(s/S_T) for the triangular kernel

# Kernel constants k₂ = ∫ k(x)² dx (precomputed for efficiency)
kernel_k1(::UniformSmoother{T}) where {T} = T(2)  # ∫k(a) da = 2
kernel_k2(::UniformSmoother{T}) where {T} = T(2)  # ∫k(a)² da = 2
kernel_k3(::UniformSmoother{T}) where {T} = T(2)  # ∫k(a)³ da = 2
kernel_k1(::TriangularSmoother{T}) where {T} = T(1)  # ∫k(a) da = 1
kernel_k2(::TriangularSmoother{T}) where {T} = T(2)/T(3)  # ∫k(a)² da = 2/3
kernel_k3(::TriangularSmoother{T}) where {T} = T(1)/T(2)  # ∫k(a) da = 1/2

"""
    optimal_bandwidth(kernel::MomentSmoother, T::Int) -> Float64

Compute optimal bandwidth for given kernel and sample size T.
"""
function optimal_bandwidth(::UniformSmoother, T::Int)
    # Optimal rate T^(1/3) for uniform kernel
    return 2.0 * T^(1.0/3.0)
end

function optimal_bandwidth(::TriangularSmoother, T::Int)
    # Optimal rate T^(1/5) for triangular kernel
    return 1.5 * T^(1.0/5.0)
end

function smooth_moments(G::AbstractMatrix, kernel::T; threaded::Bool = false) where {T <:
                                                                                     UniformSmoother}
    return uniform_sum(G, kernel.m_T; threaded = threaded)
end

function smooth_moments!(dest::AbstractMatrix, G::AbstractMatrix, kernel::T;
        threaded::Bool = false) where {T <: UniformSmoother}
    return uniform_sum!(dest, G, kernel.m_T; threaded = threaded)
end

function smooth_moments(G::AbstractMatrix, kernel::T; threaded::Bool = false) where {T <:
                                                                                     TriangularSmoother}
    return triangular_sum(G, kernel.m_T, kernel.S_T; threaded = threaded)
end

function smooth_moments!(dest::AbstractMatrix, G::AbstractMatrix, kernel::T;
        threaded::Bool = false) where {T <: TriangularSmoother}
    return triangular_sum!(dest, G, kernel.m_T, kernel.S_T; threaded = threaded)
end

using Base.Threads

# ================
# UNIFORM WINDOW
# ================
"""
    uniform_sum(G::AbstractMatrix, m_T::Integer; threaded::Bool = true) -> AbstractMatrix

    Computes sum_{s=max(t-T,-m_T)}^{min(t-1,m_T)} G[t-s, j].
"""
function uniform_sum!(dest::AbstractMatrix{T}, G::AbstractMatrix{<:Int},
        m_T; threaded::Bool = true) where {T <: Real}
    uniform_sum!(dest, float.(G), m_T; threaded = threaded)
end

function uniform_sum!(dest::AbstractMatrix{T}, G::AbstractMatrix{T},
        m_T; threaded::Bool = true) where {T <: Real}
    n, m = size(G)
    @assert size(dest) == (n, m) "Destination matrix must have the same size as G, ($n, $m)"
    P = Vector{T}(undef, n + 1)
    if threaded && nthreads() > 1
        @threads for j in axes(G, 2)
            _col_uniform_sum!(view(dest, :, j), view(G, :, j), m_T, P)
        end
    else
        for j in axes(G, 2)
            _col_uniform_sum!(view(dest, :, j), view(G, :, j), m_T, P)
        end
    end
    return dest
end

function uniform_sum(G::AbstractMatrix{<:Int}, m_T; threaded::Bool = true)
    uniform_sum(float.(G), m_T; threaded = threaded)
end

function uniform_sum(G::AbstractMatrix{T}, m_T; threaded::Bool = true) where {T <: Real}
    dest = similar(G)
    return uniform_sum!(dest, G, m_T; threaded = threaded)
end

# One column, O(T), raw sum over the window (no scaling)
@inline function _col_uniform_sum!(dest::AbstractVector{T}, col::AbstractVector{T}, m_T, P) where {T <:
                                                                                                   Real}
    n = length(col)
    begin
        mT = Int(m_T)
        P[1] = zero(T)
        for i in eachindex(col)
            P[i + 1] = P[i] + col[i]
        end
        for t in eachindex(col)
            a = max(1, t - mT)
            b = min(n, t + mT)
            dest[t] = P[b + 1] - P[a]   # raw sum, no factor
        end
    end
    return dest
end

# ==========================
# TRIANGULAR (UNSCALED)
# ==========================
# Computes sum_{s=max(t-T,-m_T)}^{min(t-1,m_T)} (1 - 2s/D) * G[t-s, j]
# which equals sum_{k=a}^{b} (1 - 2(t-k)/D) * G[k, j]
# with D = 2m_T + 1. No outer factor 2/D is applied here.

function triangular_sum!(dest::AbstractMatrix{T}, G::AbstractMatrix{<:Int},
        m_T, S_T; threaded::Bool = true) where {T <: Real}
    triangular_sum!(dest, float.(G), S_T; threaded = threaded)
end

function triangular_sum!(dest::AbstractMatrix{T}, G::AbstractMatrix{T},
        m_T, S_T; threaded::Bool = true) where {T <: Real}
    n, m = size(G)
    @assert size(dest) == (n, m) "Destination matrix must have the same size as G, ($n, $m)"

    P0 = Vector{T}(undef, n + 1)
    P1 = Vector{T}(undef, n + 1)
    if threaded && (nthreads() > 1)
        @threads for j in axes(G, 2)
            _col_triangular_sum!(view(dest, :, j), view(G, :, j), m_T, S_T, P0, P1)
        end
    else
        for j in axes(G, 2)
            _col_triangular_sum!(view(dest, :, j), view(G, :, j), m_T, S_T, P0, P1)
        end
    end
    return dest
end

function triangular_sum(G::AbstractMatrix{<:Int}, m_T, S_T; threaded::Bool = true)
    triangular_sum(float.(G), m_T, S_T; threaded = threaded)
end

function triangular_sum(G::AbstractMatrix{T}, m_T, S_T; threaded::Bool = true) where {T <:
                                                                                      Real}
    dest = similar(G, T)
    return triangular_sum!(dest, G, m_T, S_T; threaded = threaded)
end

# One column, O(T), using two prefix sums
@inline function _col_triangular_sum!(
        dest::AbstractVector{T}, col::AbstractVector{T}, m_T, S_T,
        P0::AbstractVector{T}, P1::AbstractVector{T}) where {T <: Real}
    n = length(col)
    begin
        mT = Int(m_T)
        invS_T = one(T) / T(S_T)
        P0[1] = zero(T)
        P1[1] = zero(T)
        for i in eachindex(col)
            xi = col[i]
            P0[i + 1] = P0[i] + xi
            P1[i + 1] = P1[i] + i*xi
        end

        for t in eachindex(col)
            a = max(1, t - mT)
            b = min(n, t + mT)
            S0 = P0[b + 1] - P0[a]   # sum G[k]
            S1 = P1[b + 1] - P1[a]   # sum k*G[k]
            dest[t] = (1 - t*invS_T)*S0 + invS_T*S1
        end
    end
    return dest
end

"""
    avar(estimator::SmoothedMoments, X::AbstractMatrix{F}; prewhite::Bool=false) where {F<:Real}

Main implementation of Smith's smoothed moments variance estimator.
Supports optional prewhitening which can improve finite sample performance.
"""
function avar(k::MomentSmoother, X::AbstractMatrix{F}; prewhite::Bool = false) where {F <:
                                                                                      Real}
    # Apply prewhitening if requested (using same approach as HAC)
    Z, D = finalize_prewhite(X, Val(prewhite))
    T, m = size(Z)

    # Smooth the (possibly prewhitened) moments using kernel-based approach
    # Use threading automatically for large samples (T > 800) or if explicitly requested
    use_threading = T > 1800 && m > 5 && nthreads() > 1
    G_smoothed = if use_threading
        smooth_moments(Z, k; threaded = true)
    else
        smooth_moments(Z, k; threaded = false)
    end

    ## The normaliation is k₂*S_T where
    k₂ = kernel_k2(k)
    S_T = bandwidth(k)
    # Compute variance: Ω̂ = c * (G^T)' * G^T (scaling by T handled by aVar)
    V = Matrix{float(F)}(undef, m, m)
    mul!(V, G_smoothed', G_smoothed, 1/(k₂ * S_T), 0.0)

    # Transform back if prewhitened: V_final = (I - D')^(-1) * V * (I - D')^(-1)'
    if prewhite
        v = inv(one(F)*I - D')
        V = v * V * v'
    end

    return V
end
