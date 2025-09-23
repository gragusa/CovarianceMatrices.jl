"""
Basic test of the new API functionality.
"""

using CovarianceMatrices
using LinearAlgebra
using Random
using StatsBase

# Create a simple test model
struct TestModel
    Z::Matrix{Float64}
    G::Matrix{Float64}
    θ::Vector{Float64}
end

# Implement required interface
CovarianceMatrices.momentmatrix(m::TestModel) = m.Z
CovarianceMatrices.jacobian(m::TestModel) = m.G
StatsBase.coef(m::TestModel) = m.θ

# Generate test data (well-conditioned)
Random.seed!(123)
n, k = 100, 3
m = k  # Exactly identified for testing

Z = randn(n, m) / 10  # Scale down to avoid numerical issues
G = Matrix{Float64}(I, m, k) + 0.1 * randn(m, k)  # Start with identity + noise
θ = randn(k)

model = TestModel(Z, G, θ)

println("Testing new API...")

# Test basic functionality
try
    # Test with Information form
    V_info = vcov_new(HC1(), Information(), model)
    println("✓ Information form works: size = $(size(V_info))")

    # Test with Robust form
    V_robust = vcov_new(HC1(), Robust(), model)
    println("✓ Robust form works: size = $(size(V_robust))")

    # Test auto form selection
    V_auto = vcov_new(HC1(), model; form=:auto)
    println("✓ Auto form selection works: size = $(size(V_auto))")

    # Test standard errors
    se = stderror_new(HC1(), Information(), model)
    println("✓ Standard errors work: length = $(length(se))")

    println("\nAll basic tests passed! ✓")

catch e
    println("Error in testing: $e")
    rethrow(e)
end