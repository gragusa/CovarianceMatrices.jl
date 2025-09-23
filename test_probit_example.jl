"""
Comprehensive Probit example demonstrating the new CovarianceMatrices.jl API.

This example implements a complete Probit model with all required interface methods
and demonstrates all variance estimation forms available in the new API.
"""

using CovarianceMatrices
using LinearAlgebra
using Statistics
using StatsBase
using Random
using Distributions
using StatsFuns

# ============================================================================
# Probit Model Implementation
# ============================================================================

"""
Simple Probit model implementation for binary choice problems.

The model assumes:
    P(y_i = 1 | x_i) = Φ(x_i'β)

where Φ is the standard normal CDF.
"""
mutable struct ProbitModel{T<:Real}
    # Data
    y::Vector{Int}           # Binary outcomes (0 or 1)
    X::Matrix{T}             # Design matrix (n × k)
    # Estimated parameters
    β::Vector{T}             # Coefficient estimates
    # Cached computations
    Xβ::Vector{T}            # Linear predictor X*β
    Φ::Vector{T}             # Φ(X*β) - fitted probabilities
    φ::Vector{T}             # φ(X*β) - density values
    # Convergence info
    converged::Bool
    iterations::Int
    loglik::T
end

function ProbitModel(y::Vector{Int}, X::Matrix{T}) where {T<:Real}
    n, k = size(X)
    @assert length(y) == n "Dimension mismatch: y has $(length(y)) elements, X has $n rows"
    @assert all(y .∈ Ref([0, 1])) "y must contain only 0s and 1s"

    # Initialize with OLS estimates (very crude starting values!)
    β_init = (X'X) \ (X'y)

    return ProbitModel(
        y, X, β_init,
        zeros(T, n), zeros(T, n), zeros(T, n),
        false, 0, T(-Inf)
    )
end

"""
Update cached quantities for given β.
"""
function update_predictions!(model::ProbitModel, β::AbstractVector)
    model.β .= β
    mul!(model.Xβ, model.X, β)

    # Compute Φ(Xβ) and φ(Xβ)
    for i in eachindex(model.Xβ)
        model.Φ[i] = normcdf(model.Xβ[i])
        model.φ[i] = normpdf(model.Xβ[i])
    end

    # Compute log-likelihood
    model.loglik = sum(model.y .* log.(max.(model.Φ, 1e-15)) .+
                       (1 .- model.y) .* log.(max.(1 .- model.Φ, 1e-15)))
end

"""
Fit Probit model using Newton-Raphson algorithm.
"""
function fit!(model::ProbitModel{T}; maxiter::Int=100, tol::T=T(1e-8)) where {T}

    for iter in 1:maxiter
        # Update predictions
        update_predictions!(model, model.β)

        # Compute score (gradient)
        residuals = model.y .- model.Φ
        weights = model.φ ./ (model.Φ .* (1 .- model.Φ) .+ 1e-15)
        score = model.X' * (residuals .* weights)

        # Compute Hessian (information matrix)
        W = Diagonal(model.φ .^ 2 ./ (model.Φ .* (1 .- model.Φ) .+ 1e-15))
        hessian = model.X' * W * model.X

        # Newton-Raphson update
        try
            Δβ = hessian \ score
            model.β .+= Δβ

            # Check convergence
            if norm(Δβ) < tol
                model.converged = true
                model.iterations = iter
                update_predictions!(model, model.β)
                break
            end
        catch e
            @warn "Singular Hessian at iteration $iter"
            break
        end
    end

    if !model.converged
        @warn "Probit estimation did not converge in $maxiter iterations"
    end

    return model
end

# ============================================================================
# CovarianceMatrices.jl Interface Implementation
# ============================================================================

StatsBase.nobs(model::ProbitModel) = size(model.X, 1)

"""
Return coefficient estimates.
"""
StatsBase.coef(model::ProbitModel) = model.β

"""
Return the moment matrix (score contributions).

For Probit MLE, the moment conditions are the score contributions:
g_i(β) = x_i * (y_i - Φ(x_i'β)) * φ(x_i'β) / [Φ(x_i'β) * (1 - Φ(x_i'β))]
"""
function CovarianceMatrices.momentmatrix(model::ProbitModel)
    residuals = model.y .- model.Φ
    weights = model.φ ./ (model.Φ .* (1 .- model.Φ) .+ 1e-15)

    # Each row is g_i(β)'
    return model.X .* (residuals .* weights)
end

"""
Return the Jacobian matrix (average derivative of moment conditions).

For Probit MLE, this is the negative of the average Hessian:
J = -E[∂g_i/∂β'] = -H/n
"""
function CovarianceMatrices.jacobian(model::ProbitModel)
    W = Diagonal(model.φ .^ 2 ./ (model.Φ .* (1 .- model.Φ) .+ 1e-15))
    hessian = model.X' * W * model.X
    return -hessian / length(model.y)
end

"""
Return the objective Hessian (negative Hessian of log-likelihood).

For MLE, this equals the Fisher Information matrix.
"""
function CovarianceMatrices.objective_hessian(model::ProbitModel)
    W = Diagonal(model.φ .^ 2 ./ (model.Φ .* (1 .- model.Φ) .+ 1e-15))
    return model.X' * W * model.X / length(model.y)
end

# ============================================================================
# Test Data Generation
# ============================================================================

"""
Generate synthetic Probit data for testing.
"""
function generate_probit_data(rng::AbstractRNG, n::Int, k::Int; beta_true=nothing)
    if beta_true == nothing || length(beta_true) != k
        beta_0 = rand(rng, k)
    else
        beta_0 = copy(beta_true)
    end

    # Generate design matrix
    X = [ones(n) randn(n, k - 1)]  # Include intercept

    # Generate true linear predictor
    Xβ_0 = X * beta_0

    # Generate binary outcomes
    prob_0 = normal_cdf.(Xβ_0)
    y = Int.(rand(n) .< prob_0)

    return y, X, beta_0, prob_0
end

# ============================================================================
# Comprehensive Testing
# ============================================================================

println("="^70)
println("Comprehensive Probit Example - New CovarianceMatrices.jl API")
println("="^70)

# Generate test data
rng = Random.Xoshiro(888111)
n, k = 100, 2
β_true = [0.5, 1.0]
y, X, β_true, prob_true = generate_probit_data(rng, n, k; beta_true=β_true)
## estimated par in R glm =>
thetahat_ = [0.2772741, 0.9134372]
vcov_hat = [0.022112370 0.005763997;
    0.005763997 0.029212377]
vcov_bartlett_hat = [0.018643385 0.008982916;
    0.008982916 0.026860191]
println("\nData Summary:")
println("  Observations: $n")
println("  Parameters: $k")
println("  True β: $(round.(β_true, digits=3))")
println("  Response rate: $(round(100*mean(y), digits=1))%")

# Fit Probit model
println("\nFitting Probit model...")
model = ProbitModel(y, X)
fit!(model)

if model.converged
    println("✓ Converged in $(model.iterations) iterations")
    println("  Log-likelihood: $(round(model.loglik, digits=2))")
    println("  Estimated β: $(round.(model.β, digits=3))")
    println("  Estimation error: $(round.(model.β .- β_true, digits=3))")
else
    println("✗ Failed to converge")
    exit(1)
end

# Test all variance estimators and forms
println("\n" * "="^70)
println("Testing Variance Estimation Forms")
println("="^70)

V1 = vcov_new(HC0(), Information(), model)
@test maximum(abs.(V1 .- vcov_hat)) < 1e-06
V2 = vcov_new(HC0(), Robust(), model)
@test maximum(abs.(V2 .- vcov_hat)) <= 0.01
V3 = vcov_new(Bartlett(3), Robust(), model)
@test maximum(abs.(V2 .- vcov_bartlett_hat)) <= 0.01


# ============================================================================
# Manual Matrix API Test
# ============================================================================

println("\n" * "="^70)
println("Manual Matrix API Test")
println("="^70)

# Test the matrix-based API
Z_manual = CovarianceMatrices.momentmatrix(model)
G_manual = CovarianceMatrices.jacobian(model)
H_manual = CovarianceMatrices.objective_hessian(model)

println("Testing manual matrix API...")

try
    # Test with full specification
    V_manual = vcov_new(HC1(), Robust(), Z_manual;
        jacobian=G_manual, objective_hessian=H_manual)
    println("✓ Manual API with full specification works")

    # Test with minimal specification
    V_manual_min = vcov_new(HC1(), Information(), Z_manual;
        objective_hessian=H_manual)
    println("✓ Manual API with minimal specification works")

    # Compare with model-based API
    V_model = vcov_new(HC1(), Robust(), model)
    diff = maximum(abs.(V_manual .- V_model))
    println("✓ Manual vs Model API difference: $(round(diff, digits=8))")

catch e
    println("✗ Manual API test failed: $e")
end
