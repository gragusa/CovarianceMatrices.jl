"""
Simple overidentified IV example demonstrating the new CovarianceMatrices.jl API for GMM.

This example implements a basic linear IV model with 2 instruments for 1 endogenous variable,
creating an overidentified system to demonstrate the GMM variance forms.
"""

using CovarianceMatrices
using LinearAlgebra
using Statistics
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
StatsBase.coef(m::LinearGMM) = m.beta
StatsBase.nobs(m::LinearGMM) = length(m.data.y)

function CovarianceMatrices.momentmatrix(p::LinearGMM, beta)
    y, X, Z = p.data
    Z .* (y .- X*beta)
end

function CovarianceMatrices.momentmatrix(p::LinearGMM)
    y, X, Z = p.data
    Z .* (y .- X*coef(p))
end

function CovarianceMatrices.score(p::LinearGMM)
    y, X, Z = p.data
    return -(Z' * X) ./ nobs(p)
end

## Constructor - We estimate the parameters
## using the TSLS initial matrix.
function LinearGMM(data; v::CovarianceMatrices.AVarEstimator = HR0())
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

function CovarianceMatrices.objective_hessian(p::LinearGMM)
    y, X, Z = data
    M = CovarianceMatrices.momentmatrix(p, coef(p))
    Omega = aVar(p.v, M)
    n = nobs(p)
    H = -(X'Z/n)*pinv(Omega)*(Z'X/n)
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
@test maximum(abs.(V1 .- V2)) < 0.004

model = LinearGMM(data; v = Bartlett(5))
V3 = vcov(Bartlett(3), Information(), model)
V4 = vcov(Bartlett(3), Misspecified(), model)
@test maximum(abs.(V3 .- V4)) < 0.004
