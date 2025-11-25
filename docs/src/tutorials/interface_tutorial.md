# Package Interface Tutorial

This tutorial demonstrates how to extend CovarianceMatrices.jl to work with custom model types. The package provides a minimal, duck-typed interface that allows any statistical model to obtain robust covariance matrices.

## Overview

CovarianceMatrices.jl uses a **duck-typing** approach: any model that implements a few required methods automatically gains access to all robust covariance estimators. You don't need to inherit from special types (though you can for semantic clarity).

### Key Benefits

1. **Minimal interface**: Implement 1-3 methods depending on your needs
2. **Automatic compatibility**: Works with all estimators (HAC, HC, CR, VARHAC, etc.)
3. **Flexible design**: Choose between Information and Misspecified variance forms
4. **Type-safe**: Optional abstract types provide compile-time guarantees

## The Core Interface

### Required Methods

Every model must implement:

1. **`momentmatrix(model)`**: Return the $T \times m$ matrix of moment conditions
2. **`StatsAPI.coef(model)`**: Return the $k$-dimensional parameter vector
3. **`StatsAPI.nobs(model)`**: Return the number of observations

### Optional Methods (for advanced features)

4. **`cross_score(model)`**: Return $G = \sum_i g_i g_i'$ (has default implementation)
5. **`hessian_objective(model)`**: Required for Misspecified form
6. **`jacobian_momentfunction(model)`**: Required for GMM models
7. **`weight_matrix(model)`**: For inefficient GMM

## Example 1: Simple M-Estimator

Let's implement a custom robust regression estimator:

```julia
using CovarianceMatrices, StatsBase, LinearAlgebra

"""
Custom robust regression using Huber's M-estimator.
"""
struct HuberRegression
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
    c::Float64  # Tuning parameter for Huber loss

    function HuberRegression(X, y; c=1.345)
        β = huber_fit(X, y, c)
        new(X, y, β, c)
    end
end

# Helper: Huber loss derivative (ψ function)
function huber_ψ(r, c)
    return abs(r) ≤ c ? r : c * sign(r)
end

# Simple IRLS fitting (for illustration)
function huber_fit(X, y, c; maxiter=100, tol=1e-6)
    β = X \ y  # Start with OLS
    for iter in 1:maxiter
        r = y - X * β
        ψ_r = huber_ψ.(r, c)
        β_new = (X' * X) \ (X' * (y - r + ψ_r))

        if norm(β_new - β) < tol
            return β_new
        end
        β = β_new
    end
    return β
end

# ============================================================================
# Required Interface Implementation
# ============================================================================

"""
Moment matrix: ψ(residual) ⊗ X
"""
function CovarianceMatrices.momentmatrix(model::HuberRegression)
    r = model.y - model.X * model.β
    ψ_r = huber_ψ.(r, model.c)
    return ψ_r .* model.X  # Broadcasting: T × k matrix
end

"""
Parameter estimates
"""
StatsAPI.coef(model::HuberRegression) = model.β

"""
Number of observations
"""
StatsAPI.nobs(model::HuberRegression) = length(model.y)

# ============================================================================
# Usage
# ============================================================================

# Generate data with outliers
using Random
Random.seed!(123)

n = 200
k = 3
X = randn(n, k)
β_true = [2.0, -1.5, 1.0]
ε = randn(n)

# Add outliers
outlier_idx = rand(1:n, 10)
ε[outlier_idx] .+= 10 * randn(10)

y = X * β_true + ε

# Fit model
model = HuberRegression(X, y)

println("Estimated coefficients:")
println(round.(coef(model), digits=3))

# Robust standard errors with HC3
se_hc3 = stderror(HC3(), model)
println("\nHC3 standard errors:")
println(round.(se_hc3, digits=3))

# HAC standard errors (if errors are autocorrelated)
se_hac = stderror(Bartlett{Andrews}(), model)
println("\nHAC standard errors:")
println(round.(se_hac, digits=3))

# Covariance matrix
vcov_hc3 = vcov(HC3(), model)
println("\nCovariance matrix condition number: $(round(cond(vcov_hc3), digits=2))")
```

## Example 2: Maximum Likelihood Model

For MLE models, you can optionally inherit from `MLikeModel` for semantic clarity:

```julia
using CovarianceMatrices, StatsBase, Distributions, Optim

"""
Poisson regression via maximum likelihood.
"""
struct PoissonMLE <: CovarianceMatrices.MLikeModel
    X::Matrix{Float64}
    y::Vector{Int}
    β::Vector{Float64}
    H::Matrix{Float64}  # Negative Hessian at optimum

    function PoissonMLE(X, y)
        β, H = fit_poisson_mle(X, y)
        new(X, y, β, H)
    end
end

# Fit via numerical optimization
function fit_poisson_mle(X, y)
    n, k = size(X)

    # Negative log-likelihood
    function neg_loglik(β)
        λ = exp.(X * β)
        return -sum(y .* log.(λ) - λ)
    end

    # Optimize
    result = optimize(neg_loglik, zeros(k), BFGS(), autodiff=:forward)
    β_hat = Optim.minimizer(result)

    # Compute Hessian
    H = ForwardDiff.hessian(neg_loglik, β_hat)

    return β_hat, H
end

# ============================================================================
# Interface Implementation
# ============================================================================

"""
Score functions (gradient of log-likelihood for each observation)
"""
function CovarianceMatrices.momentmatrix(model::PoissonMLE)
    λ = exp.(model.X * model.β)
    residuals = model.y - λ
    return residuals .* model.X  # T × k
end

"""
Hessian of objective (negative log-likelihood)
"""
CovarianceMatrices.hessian_objective(model::PoissonMLE) = model.H

StatsAPI.coef(model::PoissonMLE) = model.β
StatsAPI.nobs(model::PoissonMLE) = length(model.y)

# ============================================================================
# Usage with Variance Forms
# ============================================================================

using Random
Random.seed!(456)

n = 300
k = 2
X = [ones(n) randn(n)]
β_true = [0.5, 0.3]
λ_true = exp.(X * β_true)
y = [rand(Poisson(λ)) for λ in λ_true]

# Fit model
poisson_model = PoissonMLE(X, y)

# Information form (assumes correct specification)
# V = inv(H) where H is Fisher Information
vcov_info = vcov(HC0(), Information(), poisson_model)
se_info = stderror(HC0(), Information(), poisson_model)

println("Information form standard errors:")
println(round.(se_info, digits=4))

# Misspecified form (robust sandwich)
# V = inv(H) * G * inv(H) where G is outer product of scores
vcov_robust = vcov(HC3(), Misspecified(), poisson_model)
se_robust = stderror(HC3(), Misspecified(), poisson_model)

println("\nMisspecified (robust) standard errors:")
println(round.(se_robust, digits=4))

# Compare with GLM.jl
using GLM, DataFrames
df = DataFrame(y=y, x1=X[:,2])
glm_model = glm(@formula(y ~ x1), df, Poisson(), LogLink())

println("\nGLM.jl standard errors (for comparison):")
println(round.(stderror(glm_model), digits=4))
```

## Example 3: GMM Estimator

For GMM models with overidentification ($m > k$):

```julia
"""
Instrumental variables GMM estimator.
"""
struct IVGMM <: CovarianceMatrices.GMMLikeModel
    y::Vector{Float64}
    X::Matrix{Float64}  # Endogenous regressors
    Z::Matrix{Float64}  # Instruments
    β::Vector{Float64}
    W::Matrix{Float64}  # Weight matrix

    function IVGMM(y, X, Z; W=nothing)
        β, W_used = fit_iv_gmm(y, X, Z, W)
        new(y, X, Z, β, W_used)
    end
end

function fit_iv_gmm(y, X, Z, W)
    if W === nothing
        # Two-stage least squares (2SLS): W = (Z'Z)^{-1}
        W = inv(Z' * Z)
    end

    # GMM estimator: β = (X'Z W Z'X)^{-1} X'Z W Z'y
    β = (X' * Z * W * Z' * X) \ (X' * Z * W * Z' * y)

    return β, W
end

# ============================================================================
# GMM Interface
# ============================================================================

"""
Moment conditions: Z ⊗ (y - Xβ)
"""
function CovarianceMatrices.momentmatrix(model::IVGMM)
    residuals = model.y - model.X * model.β
    return residuals .* model.Z  # T × m (m = number of instruments)
end

"""
Jacobian of moment function: E[∂g/∂β'] = -Z'X
"""
function CovarianceMatrices.jacobian_momentfunction(model::IVGMM)
    return -(model.Z' * model.X)  # m × k
end

"""
Weight matrix used in GMM
"""
CovarianceMatrices.weight_matrix(model::IVGMM) = model.W

StatsAPI.coef(model::IVGMM) = model.β
StatsAPI.nobs(model::IVGMM) = length(model.y)

# ============================================================================
# Usage
# ============================================================================

using Random
Random.seed!(789)

n = 500
# True structural model: y = X*β + ε where X is endogenous
β_true = [1.5, -0.8]

# Instruments (2 instruments for 2 endogenous variables)
Z = randn(n, 3)  # 3 instruments for overidentification

# Endogenous regressors (correlated with errors)
u = randn(n)  # Common shock
X = Z * [0.5, 0.3, 0.2, 0.4, 0.1, 0.3] |> x -> reshape(x, n, 2)
X .+= 0.3 * u  # Endogeneity

# Outcome
ε = u + 0.5 * randn(n)
y = X * β_true + ε

# Fit GMM
iv_model = IVGMM(y, X, Z)

println("IV-GMM estimates:")
println(round.(coef(iv_model), digits=3))

# Robust GMM standard errors (allows misspecification)
se_gmm = stderror(HC1(), iv_model)
println("\nGMM robust standard errors:")
println(round.(se_gmm, digits=4))

# With HAC (for time series applications)
se_gmm_hac = stderror(Bartlett{Andrews}(), iv_model)
println("\nGMM-HAC standard errors:")
println(round.(se_gmm_hac, digits=4))
```

## Example 4: Custom Model Without Inheritance

You don't need to inherit from `MLikeModel` or `GMMLikeModel`. The package uses duck typing:

```julia
"""
Quantile regression (not inheriting from any abstract type).
"""
struct QuantileRegression
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
    τ::Float64  # Quantile level
end

function QuantileRegression(X, y, τ=0.5)
    β = quantile_fit(X, y, τ)
    QuantileRegression(X, y, β, τ)
end

# Simple quantile regression fit (using convex optimization)
function quantile_fit(X, y, τ)
    n, k = size(X)

    # Minimize: sum(ρ_τ(y - Xβ)) where ρ_τ(u) = u(τ - I(u<0))
    function objective(β)
        residuals = y - X * β
        return sum(r -> r * (τ - (r < 0)), residuals)
    end

    # Use numerical optimization
    result = optimize(objective, zeros(k), BFGS())
    return Optim.minimizer(result)
end

# ============================================================================
# Minimal Interface
# ============================================================================

function CovarianceMatrices.momentmatrix(model::QuantileRegression)
    residuals = model.y - model.X * model.β
    # Gradient of check function
    ψ = [r < 0 ? model.τ - 1 : model.τ for r in residuals]
    return ψ .* model.X
end

StatsAPI.coef(model::QuantileRegression) = model.β
StatsAPI.nobs(model::QuantileRegression) = length(model.y)

# Works immediately with all CovarianceMatrices.jl estimators!
# (No inheritance required)
```

## Interface Quick Reference

### Methods Summary

| Method | Required? | Purpose | Return Type |
|--------|-----------|---------|-------------|
| `momentmatrix(model)` | ✅ Yes | Moment conditions or scores | `T × m` matrix |
| `StatsAPI.coef(model)` | ✅ Yes | Parameter estimates | `k`-vector |
| `StatsAPI.nobs(model)` | ✅ Yes | Sample size | Integer |
| `cross_score(model)` | Optional | $\sum_i g_i g_i'$ | `m × m` matrix |
| `hessian_objective(model)` | Conditional | Objective Hessian | `k × k` matrix |
| `jacobian_momentfunction(model)` | For GMM | $\partial g/\partial \beta'$ | `m × k` matrix |
| `weight_matrix(model)` | For GMM | GMM weight matrix | `m × m` matrix |

### When to Implement Each Method

**Always implement:**
- `momentmatrix(model)`: Core of the interface
- `StatsAPI.coef(model)`: Parameter vector
- `StatsAPI.nobs(model)`: Sample size

**Implement for Misspecified form:**
- `hessian_objective(model)`: Required for sandwich variance

**Implement for GMM:**
- `jacobian_momentfunction(model)`: Derivative of moments w.r.t. parameters
- Optionally `weight_matrix(model)`: If using suboptimal weight

**Usually skip (has default):**
- `cross_score(model)`: Defaults to `momentmatrix(model)' * momentmatrix(model)`

## Best Practices

### 1. Scaling Conventions

**All matrices should be unscaled** (sums, not averages):

```julia
# ✅ Correct: unscaled cross-product
function CovarianceMatrices.cross_score(model)
    g = momentmatrix(model)
    return g' * g  # Sum, not mean
end

# ❌ Wrong: scaled by sample size
function CovarianceMatrices.cross_score(model)
    g = momentmatrix(model)
    T = nobs(model)
    return (g' * g) / T  # Don't do this!
end
```

### 2. Type Stability

Ensure all methods return concretely-typed arrays:

```julia
# ✅ Good: concrete return type
function CovarianceMatrices.momentmatrix(model::MyModel)
    # ... computation ...
    return Matrix{Float64}(result)
end

# ❌ Bad: abstract return type
function CovarianceMatrices.momentmatrix(model::MyModel)
    # ... computation ...
    return AbstractMatrix(result)  # Type instability!
end
```

### 3. Semantic Type Hierarchy

Use abstract types when appropriate for compile-time guarantees:

```julia
# For MLE models (m = k)
struct MyMLE <: CovarianceMatrices.MLikeModel
    # ... fields ...
end

# For GMM models (m ≥ k)
struct MyGMM <: CovarianceMatrices.GMMLikeModel
    # ... fields ...
end

# For other models, no inheritance needed
struct MyCustomModel
    # ... fields ...
end
```

### 4. Documentation

Document the moment conditions clearly:

```julia
"""
    momentmatrix(model::MyModel)

Return the T × m matrix of moment conditions.

For this model, the moment conditions are:
    g_t = ψ(y_t - x_t'β) ⊗ x_t
where ψ is the Huber influence function.
"""
function CovarianceMatrices.momentmatrix(model::MyModel)
    # ...
end
```

## Advanced: Manual Variance Computation

For maximum control, you can compute variances manually:

```julia
using CovarianceMatrices

# Your moment matrix
G = randn(500, 3)  # 500 observations, 3 moment conditions

# Estimate long-run covariance of moments
Ω_hat = aVar(VARHAC(), G; scale=false)

# If you have the Jacobian and Hessian separately
∇g = randn(3, 3)  # Jacobian of moments
H = randn(3, 3)   # Hessian of objective

# Sandwich variance (Misspecified form for MLE)
using LinearAlgebra
V_sandwich = inv(H) * Ω_hat * inv(H')

# GMM variance
V_gmm = inv(∇g' * inv(Ω_hat) * ∇g)
```

## Testing Your Implementation

```julia
using Test

@testset "MyModel Interface" begin
    # Create a model instance
    model = MyModel(...)

    # Test required methods exist and return correct types
    @test hasmethod(CovarianceMatrices.momentmatrix, (typeof(model),))
    @test hasmethod(StatsAPI.coef, (typeof(model),))
    @test hasmethod(StatsAPI.nobs, (typeof(model),))

    # Test dimensions
    Z = CovarianceMatrices.momentmatrix(model)
    β = StatsAPI.coef(model)
    T = StatsAPI.nobs(model)

    @test size(Z, 1) == T  # Rows = observations
    @test size(Z, 2) >= length(β)  # Cols ≥ parameters

    # Test variance computation works
    @test_nowarn vcov(HC3(), model)
    @test_nowarn stderror(HC3(), model)

    # Test variance forms if applicable
    if model isa CovarianceMatrices.MLikeModel
        @test_nowarn vcov(HC3(), Information(), model)
    end
end
```

## Summary

1. **Minimal effort**: Implement 3 methods, get all estimators
2. **Maximum flexibility**: Duck typing means no forced inheritance
3. **Optional features**: Add methods incrementally as needed
4. **Type safety**: Use abstract types for compile-time checks
5. **Full power**: Access to all HAC, HC, CR, VARHAC, etc. estimators

The interface is designed to be **easy to implement** but **powerful to use**. Start minimal, extend as needed.

## Further Reading

- [Introduction & Mathematical Foundation](../introduction.md): Theory behind the estimators
- [GLM Tutorial](glm_tutorial.md): See how GLM.jl integration works
- [API Reference](../api.md): Complete API documentation
- Source code: `src/model_interface.jl` and `src/api.jl`
