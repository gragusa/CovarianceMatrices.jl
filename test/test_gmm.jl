"""
Simple overidentified IV example demonstrating the new CovarianceMatrices.jl API for GMM.

This example implements a basic linear IV model with 2 instruments for 1 endogenous variable,
creating an overidentified system to demonstrate the GMM variance forms.
"""

using CovarianceMatrices
using LinearAlgebra
using Statistics
using StatsAPI
using StatsBase
using Random
using Test

# Simple IV model structure
struct LinearGMM{T, V, K} <: CovarianceMatrices.GMMLikeModel
    data::T
    beta_fs::V
    beta::V      # Estimated coefficients
    v::K
end

# Implement CovarianceMatrices.jl interface
StatsAPI.coef(m::LinearGMM) = m.beta
StatsAPI.nobs(m::LinearGMM) = length(m.data.y)

function CovarianceMatrices.momentmatrix(p::LinearGMM, beta)
    y, X, Z = p.data
    Z .* (y .- X*beta)
end

function CovarianceMatrices.momentmatrix(p::LinearGMM)
    y, X, Z = p.data
    Z .* (y .- X*coef(p))
end

function CovarianceMatrices.jacobian_momentfunction(p::LinearGMM)
    y, X, Z = p.data
    # Jacobian of sum of moment functions: ∂(∑ᵢ Zᵢ(yᵢ - Xᵢβ))/∂β = -Z'X
    return -(Z' * X)
end

## Constructor - We estimate the parameters
## using the TSLS initial matrix.
function LinearGMM(data; v::CovarianceMatrices.AbstractAsymptoticVarianceEstimator = HR0())
    y, X, Z = data
    ## First Step GMM
    W = pinv(Z'Z)
    #Main.@infiltrate
    beta_fs = (X'*Z)*W*(Z'X)\(X'*Z)*W*(Z'y)
    gmm = LinearGMM(data, beta_fs, similar(beta_fs), v)
    ## Second Step
    M = CovarianceMatrices.momentmatrix(gmm, beta_fs)
    Omega = aVar(v, M)
    beta_fs = (X'*Z)*Omega*(Z'X)\(X'*Z)*Omega*(Z'y)
    copy!(gmm.beta, beta_fs)
    return gmm
end

function CovarianceMatrices.hessian_objective(p::LinearGMM)
    y, X, Z = p.data
    M = CovarianceMatrices.momentmatrix(p, coef(p))
    Omega = aVar(p.v, M; scale = false)
    H = -(X'Z)*pinv(Omega)*(Z'X)
    return H
end

# ============================================================================
# Example Usage
# ============================================================================
"""
    simulate_iv(rng=Random.default_rng(); n, K=1, R2=0.1, ρ=0.1, β0=0.0)

Simulate one sample from the linear IV model:
y = x*β0 + ε ;  x = Z*γ + u
- Z ~ N(0, I_K)
- (ε, u) ~ N(0, Σ) with Σ = [1 ρ; ρ 1]

Returns (y::Vector, x::Vector, Z::Matrix).
"""
function simulate_iv(
        rng = Random.default_rng();
        n::Int,
        K::Int = 1,
        R2::Float64 = 0.1,
        ρ::Float64 = 0.1,
        β0::Float64 = 0.0
)
    @assert -0.999 ≤ ρ ≤ 0.999 "ρ must be in [-0.999, 0.999] for a valid covariance."
    γ = _gamma_vector(K, R2)
    # Draw instruments
    Z = randn(rng, n, K)
    # Draw (ε, u) with correlation ρ
    Σ = [1.0 ρ; ρ 1.0]
    U = cholesky(Symmetric(Σ)).U
    E = randn(rng, n, 2) * U
    ε = view(E, :, 1)
    u = view(E, :, 2)
    x = Z * γ .+ u
    y = x .* β0 .+ ε
    x_exo = randn(rng, n, 5)
    return (y = y, x = [x x_exo], z = [Z x_exo])
end

"""
    _gamma_vector(K, R2)

Return γ ∈ ℝ^K such that, with Z ~ N(0, I_K) and Var(u)=1, the first-stage R² is R2.
γ = sqrt(R2 / (K*(1 - R2))) * ones(K)
"""
function _gamma_vector(K::Int, R2::Float64)
    @assert 0.0 ≤ R2 < 1.0 "R2 must be in [0,1)."
    scale = sqrt(R2 / (K * (1 - R2)))
    return fill(scale, K)
end

# Fit IV model
rng = Random.Xoshiro(9898)
data = simulate_iv(rng; n = 1000, K = 5, R2 = 0.5, ρ = 0.3)
model = LinearGMM(data)
println("IV estimates: $(round.(model.beta, digits=3))")

# # ============================================================================
# # Test New API - GMM Forms
# # ============================================================================

V1 = vcov(HR0(), Information(), model)
V2 = vcov(HR0(), Misspecified(), model)
@test V1 ≈ V2

model = LinearGMM(data; v = Bartlett(5))
V3 = vcov(Bartlett(5), Information(), model)
V4 = vcov(Bartlett(5), Misspecified(), model)
@test V3 ≈ V4

# ============================================================================
# Test GMM formulas with known matrices (analytical verification)
# ============================================================================

# Small deterministic matrices to verify formulas exactly
k = 3  # parameters
m = 5  # moments

Random.seed!(42)
G_test = randn(m, k)
# Make Ω positive definite symmetric
A_tmp = randn(m, m)
Ω_test = A_tmp' * A_tmp + I(m)
# Arbitrary weight matrix (positive definite, NOT equal to inv(Ω))
B_tmp = randn(m, m)
W_test = B_tmp' * B_tmp + I(m)
# Hessian (positive definite)
H_test = G_test' * W_test * G_test + 0.1 * I(k)

# --- Test _compute_gmm_information (efficient, no W) ---
V_info = CovarianceMatrices._compute_gmm_information(G_test, Ω_test)
V_info_manual = inv(G_test' * inv(Ω_test) * G_test)
@test V_info ≈ V_info_manual

# --- Test _compute_gmm_information_weighted (suboptimal W) ---
# Standard sandwich: V = inv(G'WG) * G'WΩWG * inv(G'WG)
V_info_w = CovarianceMatrices._compute_gmm_information_weighted(G_test, Ω_test, W_test)
GWG = G_test' * W_test * G_test
GWG_inv = inv(GWG)
meat_info = G_test' * (W_test * Ω_test * W_test) * G_test
V_info_w_manual = GWG_inv * meat_info * GWG_inv
@test V_info_w ≈ V_info_w_manual

# --- Test _compute_gmm_misspecified (efficient, W=nothing) ---
V_mis = CovarianceMatrices._compute_gmm_misspecified(H_test, G_test, Ω_test, nothing)
Hinv = inv(H_test)
B_eff = G_test' * inv(Ω_test) * G_test
V_mis_manual = Hinv * B_eff * Hinv
@test V_mis ≈ V_mis_manual

# --- Test _compute_gmm_misspecified (suboptimal W) ---
# V = inv(H) * G'WΩWG * inv(H)
V_mis_w = CovarianceMatrices._compute_gmm_misspecified(H_test, G_test, Ω_test, W_test)
meat_mis = G_test' * (W_test * Ω_test * W_test) * G_test
V_mis_w_manual = Hinv * meat_mis * Hinv
@test V_mis_w ≈ V_mis_w_manual

# --- Verify: with efficient weight (W = inv(Ω)), Information = Misspecified ---
# When H = G'Ω⁻¹G and W = Ω⁻¹, both should give inv(G'Ω⁻¹G)
Ωinv_test = inv(Ω_test)
H_eff = G_test' * Ωinv_test * G_test
V_mis_eff = CovarianceMatrices._compute_gmm_misspecified(H_eff, G_test, Ω_test, nothing)
@test V_mis_eff ≈ V_info atol = 1e-10

# --- Verify: weighted Information ≠ simple inv(G'WΩ⁻¹WG) ---
# The old (buggy) formula was inv(G'WΩ⁻¹WG). Verify we do NOT match that.
V_old_buggy = inv(G_test' * W_test * inv(Ω_test) * W_test * G_test)
@test !(V_info_w ≈ V_old_buggy)
